# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from modelexpress.refit.reshard.rendezvous import structural_manifest_digest
import hashlib
import json
import time
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from modelexpress.refit.reshard.rendezvous import (
    PublishedShard,
    PublishedTensor,
    wrap_rendezvous_blob,
)
from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard.transfer_plan import (
    ConvertSource,
    SourceInfo,
    LogicalTransferPlan,
    plan_transfer,
)
from modelexpress.refit.reshard.types import (
    CaptureResult,
    IncompleteRefit,
    RecordedCopy,
)
from modelexpress_rl.inference.nixl_staged_transfer import (
    _bounded_batches,
    _NixlStagedTransfer,
    _PreparedNixlTransfer,
    _ResolvedSources,
    _SourceBinding,
    _resolve_sources,
    _WeightUpdatePlan,
)

from modelexpress_rl.inference.adapter import TrainerSourceShard
from modelexpress_rl.inference.plan import TrainerSourceSnapshot

from tests.test_refit_bounded_descriptors import harness, _check_values


def _manifest(
    *,
    agent_name: str,
    endpoint: str,
    offset: int,
    address: int,
    name: str = "weight",
    dtype: str = "torch.float32",
    elsize: int = 4,
    full_shape: tuple[int, ...] = (4,),
    shard_shape: tuple[int, ...] = (2,),
) -> bytes:
    return wrap_rendezvous_blob(
        b"nixl-metadata",
        agent_name,
        endpoint,
        [
            PublishedTensor(
                name=name,
                dtype=dtype,
                elsize=elsize,
                full_shape=full_shape,
                shards=[
                    PublishedShard(
                        agent_name=agent_name,
                        device_id=0,
                        addr=address,
                        shard_offset=(offset,),
                        shape=shard_shape,
                    )
                ],
            )
        ],
    )


@pytest.mark.parametrize("padded", [False, True])
def test_converted_copy_uses_captured_slice_with_arena_storage_offset(
    monkeypatch, padded
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    width = 6 if padded else 4
    offset = 1 if padded else 0
    arena = torch.full((8 * width + 16,), -17.0, dtype=torch.bfloat16)
    target = arena[8 : 8 + 8 * width].view(8, width)
    source = torch.arange(32, dtype=torch.float32).view(8, 4) / 1000
    capture = CaptureResult(
        copies=[
            RecordedCopy(
                "source", (), "layer.weight", offset, (8, 4), (width, 1), torch.bfloat16
            )
        ]
    )
    plan = LogicalTransferPlan(
        converts=(ConvertSource("layer.weight", (8, 4), torch.float32, ()),)
    )
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._recv_buffers = {"layer.weight": target}
    transfer._convert_buffers = {"layer.weight": source}
    transfer._device = torch.device("cpu")
    transfer._device_id = 0
    prepared = _PreparedNixlTransfer(
        plan,
        capture,
        {},
        (),
        SimpleNamespace(await_reads=lambda _: None),
        _SourceBinding(_ResolvedSources({}, {}, {}, {}), {}, ()),
    )
    for version in range(3):
        source.add_(1)
        transfer._complete_stage(prepared, [], time.perf_counter())
        assert torch.equal(target[:, offset : offset + 4], source.to(torch.bfloat16))
        assert torch.all(arena[:8] == -17) and torch.all(arena[-8:] == -17)
        if padded:
            assert torch.all(target[:, 0] == -17) and torch.all(target[:, -1] == -17)


def _bounded_cache_inputs():
    manifests = [
        _manifest(agent_name="a", endpoint="a:19000", offset=0, address=100),
        _manifest(agent_name="b", endpoint="b:19000", offset=2, address=200),
    ]
    return {
        "manifests": manifests,
        "resolved": _resolve_sources(manifests),
        "capture": CaptureResult(
            copies=[
                RecordedCopy("weight", (), "layer.weight", 0, (4,), (1,), torch.float32)
            ]
        ),
        "parameter_layout": {"layer.weight": ((4,), torch.float32)},
        "max_staging_bytes": 512,
    }


def _transfer_with_resolved_plan(monkeypatch, manifests: list[bytes]) -> tuple:
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._weight_update_plan = None
    resolved = transfer._resolve_metadata(manifests, {})
    transfer._weight_update_plan = _WeightUpdatePlan(
        trainer_source_snapshot=TrainerSourceSnapshot(
            "mesh",
            1,
            tuple(
                TrainerSourceShard(
                    str(index),
                    str(index),
                    hashlib.sha256(blob).hexdigest(),
                    "source:19000",
                    blob,
                    structural_manifest_digest(blob),
                )
                for index, blob in enumerate(manifests)
            ),
            resolved,
        ),
        layout_capture_snapshot=CaptureResult(copies=[]),
        parameter_layout=MappingProxyType({}),
        transfer_plan=LogicalTransferPlan(),
        source_binding=_SourceBinding(resolved, {}, ()),
        transport=None,
    )
    return transfer, resolved


@pytest.mark.parametrize(
    "defect", ["empty", "malformed", "duplicate", "mutable", "geometry"]
)
def test_metadata_rejects_bad_manifests_even_with_a_cached_plan(
    monkeypatch, defect
) -> None:
    manifest = _manifest(agent_name="a", endpoint="a:19000", offset=0, address=100)
    transfer, _ = _transfer_with_resolved_plan(monkeypatch, [manifest])
    if defect == "geometry":
        payload = json.loads(
            _manifest(agent_name="b", endpoint="b:19000", offset=2, address=200)
        )
        payload["tensors"][0]["full_shape"] = [8]
        bad = [manifest, json.dumps(payload).encode()]
    else:
        bad = {
            "empty": [],
            "malformed": [b"not-json"],
            "duplicate": [manifest, manifest],
            "mutable": [bytearray(manifest)],
        }[defect]
    with pytest.raises((ValueError, TypeError)):
        transfer._resolve_metadata(bad, {})




@pytest.mark.parametrize(
    "defect", ["unattributed", "unsupported", "missing", "unknown", "fallback"]
)
def test_supplied_complete_plan_keeps_global_coverage_gate(monkeypatch, defect):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    source = SourceInfo((4,), torch.float32, 4, [Shard((0,), (4,), "s", 100, 4)])
    capture = CaptureResult(
        copies=[RecordedCopy("w", (), "layer.w", 0, (4,), (1,), torch.float32)]
    )
    layout = {"layer.w": ((4,), torch.float32)}
    if defect == "unattributed":
        capture.unattributed = 1
    elif defect == "unsupported":
        capture.unsupported.append("w")
    elif defect == "missing":
        layout["other.w"] = ((4,), torch.float32)
    elif defect == "unknown":
        capture.copies.append(replace(capture.copies[0], param_name="unknown.w"))
    complete = plan_transfer(capture, {"w": source})
    if defect == "fallback":
        complete = replace(complete, fallback=("w",))
    with pytest.raises(IncompleteRefit):
        _bounded_batches(capture, layout, {"w": source}, 512, complete_plan=complete)


@pytest.mark.parametrize("diagnostic", ["plan"])
@pytest.mark.parametrize("missing_parameter", [False, True])
def test_diagnostic_plan_validation_keeps_values(harness, diagnostic, missing_parameter) -> None:
    prepare = harness.prepare
    first = prepare()
    harness.collect(first)
    if missing_parameter:
        harness.layout["missing.weight"] = ((4, 4), torch.float32)
        with pytest.raises(IncompleteRefit, match="missing"):
            prepare()
        return
    prepared = prepare()
    _, installed = harness.collect(prepared)
    _check_values(harness, installed)
    assert prepared.metrics["bounded_whole_validation_s"] > 0
