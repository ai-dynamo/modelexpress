# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from modelexpress_rl import envs
from modelexpress_rl.inference import receiver
from tests.refit_fixtures import _checkpoint, _delta, _FakeS3, _tensors_delta_snapshot


def test_sparse_verification_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("MX_REFIT_SPARSE_CHECK", raising=False)
    assert envs.MX_REFIT_SPARSE_CHECK is False
    assert receiver._sparse_base({"a": "00000001"}, "a", depth=2) is None
    monkeypatch.setenv("MX_REFIT_SPARSE_CHECK", "invalid")
    with pytest.raises(ValueError):
        _ = envs.MX_REFIT_SPARSE_CHECK


def test_full_verification_is_staggered_per_tensor(monkeypatch):
    monkeypatch.setenv("MX_REFIT_SPARSE_CHECK", "true")
    for name in ["a", "b", "model.layers.0.weight"]:
        readings = [
            receiver._sparse_base({name: "00000001"}, name, depth=depth)
            for depth in range(8)
        ]
        assert readings.count(None) == 1
        assert readings.count("00000001") == 7


@pytest.mark.parametrize("native", [True, False])
def test_checked_chain_and_overlays_match_with_sparse_on_or_off(
    tmp_path, monkeypatch, native
):
    if native and receiver._delta_ext is None:
        pytest.skip("optional native checker unavailable")
    if not native:
        monkeypatch.setattr(receiver, "_delta_ext", None)
    monkeypatch.setenv("MX_REFIT_STAGED_LAYERS", "4")
    monkeypatch.setenv("MX_REFIT_SPARSE_CHECK", "true")
    states = [{"a": [float(i), 2.0], "b": [3.0, float(i + 4)]} for i in range(4)]
    objects = {}
    for index in range(1, 4):
        objects.update(
            _tensors_delta_snapshot(
                states[index - 1],
                states[index],
                identity=f"v{index}",
                base_identity="base" if index == 1 else f"v{index - 1}",
            )
        )
    checkpoint = _checkpoint(
        tmp_path, _FakeS3(objects), tensors=states[0], stage_deltas=True
    )
    first = checkpoint.prepare_chain((_delta("v1", base="base", identity="v1"),))
    assert first.metrics["perf/mx_receive_delta_sparse_checks"] == 0
    assert set(checkpoint.store.checksums("v1")) == {"a", "b"}
    second = checkpoint.prepare_chain((_delta("v2", base="v1", identity="v2"),))
    assert second.metrics["perf/mx_receive_delta_sparse_checks"] == (2 if native else 0)
    monkeypatch.setenv("MX_REFIT_SPARSE_CHECK", "false")
    third = checkpoint.prepare_chain((_delta("v3", base="v2", identity="v3"),))
    assert third.metrics["perf/mx_receive_delta_sparse_checks"] == 0
    from modelexpress_rl.inference.engines.vllm.installer import _with_staged_deltas
    import torch

    tensors = [(name, torch.tensor(value)) for name, value in states[0].items()]
    target = dict(_with_staged_deltas(tensors, third.overlay))
    for name, value in states[3].items():
        assert torch.equal(target[name], torch.tensor(value))


def test_sparse_flag_never_supplies_base_checksum_when_disabled(monkeypatch, tmp_path):
    if receiver._delta_ext is None:
        pytest.skip("optional native checker unavailable")
    from tests.refit_fixtures import (
        _staged_scene,
        _check_passes,
        _checksum,
        _sparse_bytes,
    )

    rng = np.random.default_rng(7)
    checkpoint, head, layers = _staged_scene(tmp_path, rng, {"a": 128}, density=0.1)
    delta = {"a": _sparse_bytes(rng, 128, 0.1)}
    calls = []
    check = receiver._delta_ext.check

    def record(*args):
        calls.append(args[5].tolist())
        return check(*args)

    monkeypatch.setattr(receiver._delta_ext, "check", record)
    monkeypatch.setenv("MX_REFIT_SPARSE_CHECK", "false")
    try:
        checkpoint._run_pass(
            _check_passes(head, layers, delta, {"a": _checksum(head["a"])}),
            index_metadata={"compression_format": "zstd", "checksum_format": "adler32"},
        )
    finally:
        for layer in layers:
            layer.close()
    assert calls == [[-1]]
