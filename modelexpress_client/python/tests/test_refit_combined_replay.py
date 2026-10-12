# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combined trainer conversion and ordered loader capture share one generation."""

import pytest
import torch
from torch import nn

from modelexpress_rl.refit.reshard.types import IncompleteRefit
from tests.test_refit_ordered_installer import (
    _persistent_installer,
    _transfer_and_install,
)


class _OverlappingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backing = torch.zeros(4)
        self.register_buffer("a", self.backing[:3])
        self.register_buffer("b", self.backing[1:])
        self.calls = 0

    def load_weights(self, weights):
        self.calls += 1
        for name, value in weights:
            # Native loading adds another view after the conversion prefix.
            target = self.b if name == "first" else self.a
            target.copy_(value.reshape(-1))


def _convert(weights):
    native = weights["trainer.weight"][:3].t()
    # Reversed destination order matters because a and b overlap.
    return {"first": native[0], "second": native[1]}


def _fixture(monkeypatch):
    model = _OverlappingModel()
    installer = _persistent_installer(monkeypatch, model)
    installer._debug_validate_generator_layout = True
    installer._convert_native_to_hf = _convert
    source = torch.arange(6.0).reshape(3, 2)
    manifest = [("trainer.weight", source.dtype, tuple(source.shape))]
    return model, installer, source, manifest


@pytest.mark.parametrize("bounded", [False, True])
def test_combined_conversion_replay_matches_direct_loader(monkeypatch, bounded):
    model, installer, source, manifest = _fixture(monkeypatch)
    original = [(value, value.data_ptr()) for value in model.buffers()]
    captured, layout = installer.capture(manifest)
    assert [copy.src_name for copy in captured.copies] == ["trainer.weight"] * 2
    assert [copy.param_name for copy in captured.copies] == ["b", "a"]
    assert all(copy.op_chain for copy in captured.copies)
    for update in range(3):
        source.add_(update + 1)
        direct = _OverlappingModel()
        direct.load_weights(_convert({"trainer.weight": source}).items())
        _transfer_and_install(
            installer, {"trainer.weight": source}, captured, layout, bounded
        )
        torch.testing.assert_close(model.backing, direct.backing, rtol=0, atol=0)
        for value, (previous, pointer) in zip(model.buffers(), original):
            assert value is previous and value.data_ptr() == pointer


def test_combined_capture_reuse_keeps_schedule_and_snapshot(monkeypatch):
    model, installer, _, manifest = _fixture(monkeypatch)
    captured, layout = installer.capture(manifest)
    plan = installer.source_install_plan
    generation = captured.generation
    captured.copies.clear()
    layout.clear()
    reused, reused_layout = installer.capture(manifest)
    assert len(reused.copies) == 2 and set(reused_layout) == {"a", "b"}
    compatible, _ = installer.capture(manifest)
    assert compatible.generation == reused.generation == generation
    assert installer.source_install_plan is plan
    assert model.calls == 1


@pytest.mark.parametrize("change", ["converter", "schema"])
def test_combined_incompatible_generation_rebuilds_and_rejects_old_batch(
    monkeypatch, change
):
    model, installer, source, manifest = _fixture(monkeypatch)
    old, old_layout = installer.capture(manifest)
    old_plan = installer.source_install_plan
    if change == "converter":

        def changed(weights):
            native = weights["trainer.weight"][:3].t()
            return {"first": native[1], "second": native[0]}

        installer._convert_native_to_hf = changed
    else:
        manifest = [("trainer.weight", torch.float32, (4, 2))]
        # The unchanged converter selects three rows, preserving destinations.
    new, _ = installer.capture(manifest)
    assert new.generation != old.generation
    assert installer.source_install_plan is not old_plan
    assert installer.source_install_plan.schedule is not old_plan.schedule
    assert model.calls == 2
    with pytest.raises(IncompleteRefit, match="invalidated"):
        _transfer_and_install(
            installer, {"trainer.weight": source}, old, old_layout, False
        )


def test_failed_combined_recapture_invalidates_and_retry_rebuilds(monkeypatch):
    model, installer, _, manifest = _fixture(monkeypatch)
    old, _ = installer.capture(manifest)
    fail = True

    def converter(weights):
        if fail:
            raise RuntimeError("conversion failed")
        return _convert(weights)

    installer._convert_native_to_hf = converter
    with pytest.raises(RuntimeError, match="conversion failed"):
        installer.capture(manifest)
    assert installer._capture_cache is None
    assert installer.source_install_plan is None
    fail = False
    retried, _ = installer.capture(manifest)
    assert retried.generation > old.generation
    assert installer.source_install_plan is not None
    assert installer.source_install_plan.schedule.total_copies == len(retried.copies)
    assert model.calls == 2


def test_incomplete_combined_capture_returns_without_caching(monkeypatch):
    model, installer, _, manifest = _fixture(monkeypatch)
    model.extra = nn.Parameter(torch.zeros(1), requires_grad=False)
    captured, layout = installer.capture(manifest)
    assert "extra" in layout
    assert "extra" not in {copy.param_name for copy in captured.copies}
    assert installer.source_install_plan is None
    assert installer._capture_cache is None
    installer.capture(manifest)
    assert model.calls == 2
