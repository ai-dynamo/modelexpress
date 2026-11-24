# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch
from modelexpress_rl import envs
from modelexpress_rl.inference.engines.vllm.installer import _VllmInstaller
from modelexpress_rl.refit.reshard.types import IncompleteRefit
from safetensors.torch import save_file


def installer(**kwargs):
    return _VllmInstaller(
        model=torch.nn.Linear(2, 2),
        vllm_config=SimpleNamespace(quant_config=kwargs.pop("quant_config", None)),
        model_config=SimpleNamespace(),
        device=torch.device("cpu"),
        **kwargs,
    )


def checkpoint(tmp_path, dtype=torch.bfloat16):
    save_file(
        {"q": torch.arange(4).reshape(2, 2).to(dtype)}, tmp_path / "a.safetensors"
    )
    save_file({"v": torch.ones(2, 2, dtype=dtype)}, tmp_path / "b.safetensors")
    return tmp_path


def test_replay_preserves_capture_order_repeated_sources_and_lifetimes(
    tmp_path, monkeypatch
):
    import safetensors

    path = checkpoint(tmp_path)
    target = installer()
    real_open = safetensors.safe_open
    opened = []

    class Reader:
        def __init__(self, *args, **kwargs):
            self.reader = real_open(*args, **kwargs)

        def __enter__(self):
            opened.append(self)
            return self.reader.__enter__()

        def __exit__(self, *args):
            opened.remove(self)
            return self.reader.__exit__(*args)

    monkeypatch.setattr(safetensors, "safe_open", Reader)
    copies = tuple(SimpleNamespace(src_name=n) for n in ("v", "q", "q"))
    replay_id = uuid4()
    manifests, applied = [], []

    def capture(manifest):
        manifests.append(manifest)
        return SimpleNamespace(
            copies=copies, unsupported=(), unattributed=(), replay_id=replay_id
        ), {}

    def apply(batch):
        assert len(opened) == 2  # Remain open through install_tensors' synchronization.
        assert batch.replay_id == replay_id
        assert [index for index, _, _ in batch.writes] == [0, 1, 2]
        assert [item for _, item, _ in batch.writes] == list(copies)
        assert batch.writes[1][2] is batch.writes[2][2]
        applied.append(batch.writes[1][2].clone())

    monkeypatch.setattr(target, "capture", capture)
    monkeypatch.setattr(target, "install_tensors", apply)
    target.install_checkpoint_replay(path)
    save_file(
        {"q": torch.full((2, 2), 9, dtype=torch.bfloat16)}, path / "a.safetensors"
    )
    target.install_checkpoint_replay(path)
    assert manifests[0] == manifests[1]
    assert torch.equal(applied[0], torch.arange(4).reshape(2, 2).bfloat16())
    assert torch.equal(applied[1], torch.full((2, 2), 9, dtype=torch.bfloat16))
    assert opened == []


@pytest.mark.parametrize("problem", ["unsupported", "unattributed", "empty", "missing"])
def test_incomplete_capture_never_installs(tmp_path, monkeypatch, problem):
    target = installer()
    capture = SimpleNamespace(
        copies=(SimpleNamespace(src_name="missing" if problem == "missing" else "q"),),
        unsupported=("q",) if problem == "unsupported" else (),
        unattributed=("weight",) if problem == "unattributed" else (),
        replay_id=uuid4(),
    )
    if problem == "empty":
        capture.copies = ()
    monkeypatch.setattr(target, "capture", lambda _: (capture, {}))
    monkeypatch.setattr(
        target, "install_tensors", lambda _: pytest.fail("must not install")
    )
    with pytest.raises(IncompleteRefit):
        target.install_checkpoint_replay(checkpoint(tmp_path))


@pytest.mark.parametrize(
    "options", [{"quant_config": object()}, {"convert_native_to_hf": lambda x: x}]
)
def test_unsupported_engine_rejected_before_opening_files(tmp_path, options):
    with pytest.raises(IncompleteRefit, match="unquantized HF"):
        installer(**options).install_checkpoint_replay(tmp_path / "nonexistent")


def test_integer_sources_rejected_before_capture(tmp_path, monkeypatch):
    target = installer()
    monkeypatch.setattr(target, "capture", lambda _: pytest.fail("must not capture"))
    with pytest.raises(IncompleteRefit, match="floating-point"):
        target.install_checkpoint_replay(checkpoint(tmp_path, torch.int64))


def test_capture_and_install_failures_propagate_without_retry(tmp_path, monkeypatch):
    target = installer()
    path = checkpoint(tmp_path)
    for stage in ("capture", "install_tensors"):
        calls = []
        result = SimpleNamespace(
            copies=(SimpleNamespace(src_name="q"),),
            unsupported=(),
            unattributed=(),
            replay_id=uuid4(),
        )
        monkeypatch.setattr(target, "capture", lambda _, result=result: (result, {}))
        monkeypatch.setattr(target, "install_tensors", lambda _: None)

        def fail(_, calls=calls, stage=stage):
            calls.append(stage)
            raise RuntimeError("injected failure")

        monkeypatch.setattr(target, stage, fail)
        with pytest.raises(RuntimeError, match="injected failure"):
            target.install_checkpoint_replay(path)
        assert calls == [stage]


def test_checkpoint_replay_flag_is_opt_in_and_routes_without_native_loader(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MX_REFIT_CHECKPOINT_SOURCE_REPLAY", raising=False)
    assert envs.MX_REFIT_CHECKPOINT_SOURCE_REPLAY is False
    monkeypatch.setenv("MX_REFIT_CHECKPOINT_SOURCE_REPLAY", "true")
    target = installer()
    calls = []
    monkeypatch.setattr(target, "install_checkpoint_replay", calls.append)
    target.install_checkpoint(tmp_path)
    assert calls == [tmp_path]
    monkeypatch.setenv("MX_REFIT_CHECKPOINT_SOURCE_REPLAY", "invalid")
    with pytest.raises(ValueError):
        target.install_checkpoint(tmp_path)


def test_failed_replay_retains_sources_and_refuses_retry(tmp_path, monkeypatch):
    target = installer()
    path = checkpoint(tmp_path)
    result = SimpleNamespace(
        copies=(SimpleNamespace(src_name="q"),),
        unsupported=(),
        unattributed=(),
        replay_id=uuid4(),
    )
    monkeypatch.setattr(target, "capture", lambda _, result=result: (result, {}))

    def fail(_):
        raise RuntimeError("device fence failed")

    monkeypatch.setattr(target, "install_tensors", fail)
    with pytest.raises(RuntimeError, match="device fence"):
        target.install_checkpoint_replay(path)
    sources, handles = target._failed_checkpoint_replay
    assert sources["q"].shape == (2, 2)
    assert len(handles._exit_callbacks) == 2
    with pytest.raises(IncompleteRefit, match="recreate the engine"):
        target.install_checkpoint_replay(path)
    monkeypatch.setenv("MX_REFIT_CHECKPOINT_SOURCE_REPLAY", "false")
    with pytest.raises(IncompleteRefit, match="recreate the engine"):
        target.install_checkpoint(path)
    handles.close()  # CPU-only test has no outstanding device work.
