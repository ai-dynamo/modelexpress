# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace
from modelexpress_rl.version import MeshIdentity

from modelexpress_rl.inference.trainer_plan import TrainerTransferPlan

import pytest
import torch
from modelexpress_rl.refit.reshard.rendezvous import (
    PublishedShard,
    PublishedTensor,
    unwrap_rendezvous_blob,
    wrap_rendezvous_blob,
)
from modelexpress_rl.refit.reshard.slice_plan import Shard
from modelexpress_rl.refit.reshard.transfer_plan import (
    SourceInfo,
    plan_transfer,
)
from modelexpress_rl.refit.reshard.types import (
    CaptureResult,
    IncompleteRefit,
    RecordedCopy,
)
from modelexpress_rl.inference.nixl_staged_transfer import (
    _NixlStagedTransfer,
    _SourceBinding,
    _required_agent_metadata,
)

from modelexpress_rl.inference.adapter import TrainerSourceShard
from modelexpress_rl.inference.plan import TrainerSourceSnapshot

from tests.test_refit_bounded_descriptors import harness, _check_values
from tests.helpers.refit_replay import resolve_sources as _resolve_sources


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


def _bounded_cache_inputs():
    manifests = [
        _manifest(agent_name="a", endpoint="a:19000", offset=0, address=100),
        _manifest(agent_name="b", endpoint="b:19000", offset=2, address=200),
    ]
    return {
        "manifests": manifests,
        "resolved": _resolve_sources(manifests),
        "capture": CaptureResult(source_replay=True,
            copies=[
                RecordedCopy("weight", (), "layer.weight", 0, (4,), (1,), torch.float32)
            ]
        ),
        "parameter_layout": {"layer.weight": ((4,), torch.float32)},
        "max_staging_bytes": 512,
    }


def _trainer_snapshot(manifests: list[bytes]) -> TrainerSourceSnapshot:
    return TrainerSourceSnapshot(
        MeshIdentity("mesh", 1),
        tuple(
            TrainerSourceShard(
                str(index),
                str(index),
                hashlib.sha256(blob).hexdigest(),
                unwrap_rendezvous_blob(blob).metadata_endpoint,
                blob,
            )
            for index, blob in enumerate(manifests)
        ),
    )


def test_binding_keeps_original_selection_and_connects_only_needed_sources():
    manifests = [
        _manifest(agent_name="unused", endpoint="unused:1", offset=0, address=100,
                  name="unused", shard_shape=(4,)),
        _manifest(agent_name="needed", endpoint="needed:1", offset=0, address=200,
                  shard_shape=(4,)),
    ]
    trainer = _trainer_snapshot(manifests)
    binding = _SourceBinding.from_snapshot(trainer, {})
    resolved = binding.resolved
    capture = CaptureResult(source_replay=True, copies=[
        RecordedCopy("weight", (), "layer.weight", 0, (4,), (1,), torch.float32)
    ])
    plan = plan_transfer(capture, resolved.sources,
                         source_slots={"unused": "0", "needed": "1"})

    assert [shard.source_slot_id for shard in trainer.shards] == ["0", "1"]
    assert binding.agents_by_slot == {"0": "unused", "1": "needed"}
    assert binding.physical_fingerprint == trainer.physical_fingerprint
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._binding = binding
    metrics = {}
    assert transfer._resolve_metadata(trainer, metrics) is binding
    assert metrics["source_cache_hits"] == 1
    assert _required_agent_metadata(plan, resolved, binding) == {
        "needed": b"nixl-metadata"
    }
    for read in plan._all_segments():
        assert binding.resolve(plan.source_buffers[read.source_buffer]) == ("needed", 200)


@pytest.mark.parametrize(
    "defect", ["empty", "malformed", "duplicate", "duplicate_slot", "mutable", "geometry"]
)
def test_metadata_rejects_bad_manifests_even_with_a_cached_source(
    monkeypatch, defect
) -> None:
    manifest = _manifest(agent_name="a", endpoint="a:19000", offset=0, address=100)
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._binding = _SourceBinding.from_snapshot(_trainer_snapshot([manifest]), {})
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
            "duplicate_slot": [
                manifest,
                _manifest(agent_name="b", endpoint="b:19000", offset=2, address=200),
            ],
            "mutable": [bytearray(manifest)],
        }[defect]
    with pytest.raises((ValueError, TypeError)):
        snapshot = TrainerSourceSnapshot(
            MeshIdentity("invalid", 1),
            tuple(
                TrainerSourceShard(
                    "0" if defect == "duplicate_slot" else str(index),
                    str(index), "digest", "source:19000", blob,
                )
                for index, blob in enumerate(bad)
            ),
        )
        transfer._resolve_metadata(snapshot, {})


@pytest.mark.parametrize(
    "defect", ["unattributed", "unsupported", "missing", "unknown", "fallback"]
)
def test_supplied_complete_plan_keeps_global_coverage_gate(monkeypatch, defect):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    source = SourceInfo((4,), torch.float32, 4, [Shard((0,), (4,), "s", 100, 4)])
    capture = CaptureResult(source_replay=True,
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
        TrainerTransferPlan.validate_capture(capture, layout, complete)


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
    assert prepared.metrics["plan_cache_validate_s"] >= 0
