# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Real CPU staging and source replay, with vLLM reload hooks stubbed."""

from types import SimpleNamespace

import pytest
import torch
from modelexpress_rl import WeightVersionRef
from modelexpress_rl.inference.engines.vllm.installer import _VllmInstaller

from tests import test_refit_stage_process as staging
from tests import test_vllm_stage_process as adapter
from tests.test_refit_vllm_installer import _install_fake_vllm, _replay_batches

harness = staging.harness


@pytest.mark.parametrize("stage_process", [False, True])
def test_staged_checkpoint_replays_only_at_apply(harness, monkeypatch, stage_process):
    monkeypatch.setenv("MX_REFIT_CHECKPOINT_SOURCE_REPLAY", "true")
    monkeypatch.setenv("MX_REFIT_DELTA_SURGICAL", "false")
    _install_fake_vllm(monkeypatch, lambda model: None)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for name, value in harness.tensors["v0"].items():
        layer, _, leaf = name.rpartition(".")
        model.get_submodule(layer).register_parameter(
            leaf,
            torch.nn.Parameter(torch.from_numpy(value.copy()), requires_grad=False),
        )
    target = _VllmInstaller(
        model=model,
        device=torch.device("cpu"),
        vllm_config=SimpleNamespace(quant_config=None),
        model_config=SimpleNamespace(),
    )
    captures = []

    def capture(manifest):
        captures.append(manifest)
        tensors = {
            name: torch.empty(shape, dtype=dtype) for name, dtype, shape in manifest
        }
        batch = next(_replay_batches(target, lambda: iter([tensors]), tensors))
        return SimpleNamespace(
            copies=tuple(item for _, item, _ in batch.writes),
            replay_id=batch.replay_id,
            unsupported=(),
            unattributed=(),
        ), {}

    monkeypatch.setattr(target, "capture", capture)
    monkeypatch.setattr(adapter, "_Installer", lambda: target)
    rig = adapter._Rig(harness, monkeypatch, stage_process=stage_process)
    pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
    previous = "v0"
    for version in ("v1", "v2", "v3", "v4"):
        count = len(captures)
        handle = rig.client.stage_weight(version=WeightVersionRef(version))
        assert len(captures) == count
        for name, value in model.named_parameters():
            assert torch.equal(value, torch.from_numpy(harness.tensors[previous][name]))
        result = rig.client.apply_weight(handle)
        assert result["installation_checkpoint_source_replay"] == 1
        for name, value in model.named_parameters():
            assert torch.equal(value, torch.from_numpy(harness.tensors[version][name]))
            assert value.data_ptr() == pointers[name]
        handle.release()
        previous = version
    assert len(captures) == 4
    if stage_process:
        assert rig.parent_reads == []
        assert harness.shard_reads(rig.client._stage_process.pid)
    else:
        assert rig.parent_reads
    staging._wait_until(lambda: not harness.service.leases, "leases to be released")
