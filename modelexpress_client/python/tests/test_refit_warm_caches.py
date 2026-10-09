# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import modelexpress_rl.inference.nixl_staged_transfer as transfer_module
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
    TransferPlan,
)
from modelexpress.refit.reshard.types import (
    CaptureResult,
    IncompleteRefit,
    RecordedCopy,
)
from modelexpress_rl.inference.nixl_staged_transfer import (
    _bounded_batches,
    _BoundedPlanCache,
    _NixlStagedTransfer,
    _plan_staged_transfer,
    _PreparedNixlTransfer,
    _resolve_sources,
    _SourceResolutionCache,
)


def _manifest(*, agent_name: str, endpoint: str, offset: int, address: int) -> bytes:
    return wrap_rendezvous_blob(
        b"nixl-metadata",
        agent_name,
        endpoint,
        [
            PublishedTensor(
                name="weight",
                dtype="torch.float32",
                elsize=4,
                full_shape=(4,),
                shards=[
                    PublishedShard(
                        agent_name=agent_name,
                        device_id=0,
                        addr=address,
                        shard_offset=(offset,),
                        shape=(2,),
                    )
                ],
            )
        ],
    )


@pytest.mark.parametrize("padded", [False, True])
def test_converted_copy_uses_captured_slice_with_arena_storage_offset(
    monkeypatch, padded
):
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
    plan = TransferPlan(
        converts=[ConvertSource("layer.weight", (8, 4), torch.float32, [])]
    )
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._recv_buffers = {"layer.weight": target}
    transfer._convert_buffers = {"layer.weight": source}
    transfer._device = torch.device("cpu")
    transfer._device_id = 0
    prepared = _PreparedNixlTransfer(
        plan, capture, {}, (), SimpleNamespace(await_reads=lambda _: None)
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
        "enabled": True,
    }


def test_resolved_source_cache_requires_complete_ordered_manifest_bytes():
    cache = _SourceResolutionCache()
    manifests = [
        _manifest(agent_name="a", endpoint="a:19000", offset=0, address=100),
        _manifest(agent_name="b", endpoint="b:19000", offset=2, address=200),
    ]
    metrics = {}
    first = cache.resolve(manifests, enabled=True, metrics=metrics)
    assert metrics["source_cache_misses"] == 1
    assert metrics["source_manifest_bytes"] == sum(map(len, manifests))
    copied = [bytes(bytearray(blob)) for blob in manifests]
    assert copied[0] is not manifests[0]
    assert cache.resolve(copied, enabled=True, metrics=metrics) is first
    assert metrics["source_cache_hits"] == 1
    assert all(
        metrics[f"source_{phase}_s"] == 0 for phase in ("decode", "merge", "build")
    )

    changed = json.loads(manifests[0])
    changed["tensors"][0]["shards"][0]["addr"] = 300
    changed["tensors"][0]["shards"][0]["digest"] = "new-version-digest"
    second = cache.resolve(
        [json.dumps(changed).encode(), manifests[1]], enabled=True, metrics=metrics
    )
    assert second.sources["weight"].shards[0].addr == 300
    assert second.sources["weight"].shards[0].digest == "new-version-digest"
    assert first.sources["weight"].shards[0].addr == 100
    assert metrics["source_cache_misses"] == 1
    assert cache.resolve(manifests, enabled=True, metrics=metrics) is not first
    assert metrics["source_cache_misses"] == 1  # The previous version was evicted.
    cache.resolve(list(reversed(manifests)), enabled=True, metrics=metrics)
    assert metrics["source_cache_misses"] == 1
    cache.resolve(manifests[:1], enabled=True, metrics=metrics)
    assert metrics["source_cache_misses"] == 1
    cache.clear()
    cache.resolve(manifests[:1], enabled=True, metrics=metrics)
    assert metrics["source_cache_misses"] == 1
    cache.resolve(manifests[:1], enabled=False, metrics=metrics)
    assert metrics["source_cache_enabled"] == metrics["source_cache_hits"] == 0
    cache.resolve(manifests[:1], enabled=True, metrics=metrics)
    assert metrics["source_cache_misses"] == 1


@pytest.mark.parametrize(
    "field", ["digest", "device_id", "agent_meta_b64", "publisher_step"]
)
def test_resolved_source_cache_refreshes_changed_version_and_transport_fields(field):
    cache = _SourceResolutionCache()
    manifest = _manifest(agent_name="a", endpoint="a:19000", offset=0, address=100)
    metrics = {}
    first = cache.resolve([manifest], enabled=True, metrics=metrics)
    payload = json.loads(manifest)
    if field in ("digest", "device_id"):
        payload["tensors"][0]["shards"][0][field] = (
            "new-digest" if field == "digest" else 1
        )
    else:
        payload[field] = "bmV3LW1ldGFkYXRh" if field == "agent_meta_b64" else 2
    changed = json.dumps(payload).encode()
    resolved = cache.resolve([changed], enabled=True, metrics=metrics)
    assert resolved is not first
    expected = _resolve_sources([changed])
    assert resolved.session_to_agent == expected.session_to_agent
    assert resolved.session_to_device == expected.session_to_device
    assert resolved.agent_metadata == expected.agent_metadata
    assert tuple(resolved.sources) == tuple(expected.sources)
    for name, source in resolved.sources.items():
        original = expected.sources[name]
        assert (source.global_shape, source.dtype, source.elsize) == (
            original.global_shape,
            original.dtype,
            original.elsize,
        )
        assert [tuple(shard) for shard in source.shards] == [
            tuple(vars(shard).values()) for shard in original.shards
        ]
    assert metrics["source_cache_misses"] == 1


@pytest.mark.parametrize(
    "defect", ["empty", "malformed", "duplicate", "mutable", "geometry"]
)
def test_resolved_source_cache_does_not_reuse_previous_entry_after_bad_inputs(defect):
    cache = _SourceResolutionCache()
    manifest = _manifest(agent_name="a", endpoint="a:19000", offset=0, address=100)
    metrics = {}
    first = cache.resolve([manifest], enabled=True, metrics=metrics)
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
        cache.resolve(bad, enabled=True, metrics=metrics)
    assert cache.resolve([manifest], enabled=True, metrics=metrics) is not first
    assert metrics["source_cache_misses"] == 1


@pytest.mark.parametrize("copy_key_on_miss", [False, True])
def test_bounded_plan_cache_snapshots_callback_inputs_and_revalidates(
    monkeypatch, copy_key_on_miss
):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setenv("MX_REFIT_COPY_PLAN_KEY_ON_MISS", str(int(copy_key_on_miss)))
    cache = _BoundedPlanCache()
    args = _bounded_cache_inputs()
    pristine = copy.deepcopy(args)
    metrics = {}
    first = cache.compile(**args, metrics=metrics)
    key_capture = cache._entry[0][5][0]
    assert key_capture is not args["capture"]
    assert key_capture.copies[0] is not args["capture"].copies[0]
    assert first.module_batches[0].capture.copies[0] is not key_capture.copies[0]
    assert len(first.fingerprint) == 64
    assert metrics["plan_cache_misses"] == metrics["owner_plan_builds"] == 1
    assert cache.compile(**pristine, metrics=metrics) is first
    assert metrics["plan_cache_hits"] == 1
    assert metrics["plan_cache_key_copies"] == int(not copy_key_on_miss)
    assert metrics["owner_plan_builds"] == metrics["bounded_whole_plan_builds"] == 0
    args["capture"].copies[0].dest_offset = 1
    args["parameter_layout"]["layer.weight"] = ((5,), torch.float32)
    assert first.module_batches[0].capture.copies[0].dest_offset == 0
    assert first.module_batches[0].layouts[0]["layer.weight"][0] == (4,)
    assert cache.compile(**args, metrics=metrics) is not first
    assert metrics["plan_cache_misses"] == 1

    # A hit must still execute the global coverage gate.
    fresh = cache.compile(**pristine, metrics=metrics)
    fresh.plan.fallback.append("injected-unsupported")
    with pytest.raises(IncompleteRefit, match="cover every"):
        cache.compile(**pristine, metrics=metrics)
    assert cache._entry is None
    fresh = cache.compile(**pristine, metrics=metrics)
    fresh.module_batches[0].plan.fallback.append("injected-owner-unsupported")
    with pytest.raises(IncompleteRefit, match="cover every"):
        cache.compile(**pristine, metrics=metrics)
    assert cache._entry is None


def test_copy_plan_key_on_miss_keeps_exact_fingerprint_and_avoids_hit_deepcopy(
    monkeypatch,
):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    args = _bounded_cache_inputs()
    monkeypatch.setenv("MX_REFIT_COPY_PLAN_KEY_ON_MISS", "0")
    original = _BoundedPlanCache().compile(**args, metrics={})
    monkeypatch.setenv("MX_REFIT_COPY_PLAN_KEY_ON_MISS", "1")
    cache = _BoundedPlanCache()
    metrics = {}
    first = cache.compile(**args, metrics=metrics)
    assert first.fingerprint == original.fingerprint
    assert metrics["plan_cache_copy_key_on_miss_enabled"] == 1
    assert metrics["plan_cache_key_copies"] == 1

    def unexpected_copy(value):
        pytest.fail("a warm key lookup must not deepcopy callback inputs")

    monkeypatch.setattr(transfer_module, "deepcopy", unexpected_copy)
    assert cache.compile(**args, metrics=metrics) is first
    assert metrics["plan_cache_hits"] == 1
    assert metrics["plan_cache_key_copies"] == 0


@pytest.mark.parametrize("concurrent", [False, True])
def test_copy_plan_key_on_miss_rejects_overlapping_compile_and_recovers(
    monkeypatch, concurrent
):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setenv("MX_REFIT_COPY_PLAN_KEY_ON_MISS", "1")
    cache = _BoundedPlanCache()
    args = _bounded_cache_inputs()
    original = transfer_module._plan_staged_transfer
    entered = threading.Event()
    release = threading.Event()

    def planning(capture, sources):
        if concurrent:
            entered.set()
            assert release.wait(timeout=5)
        else:
            cache.compile(**args, metrics={})
        return original(capture, sources)

    monkeypatch.setattr(transfer_module, "_plan_staged_transfer", planning)
    if concurrent:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(cache.compile, **args, metrics={})
            try:
                assert entered.wait(timeout=5)
                with pytest.raises(RuntimeError, match="already in progress"):
                    cache.compile(**args, metrics={})
            finally:
                release.set()
            assert pending.result(timeout=5).fingerprint
    else:
        with pytest.raises(RuntimeError, match="already in progress"):
            cache.compile(**args, metrics={})
        assert cache._entry is None
    monkeypatch.setattr(transfer_module, "_plan_staged_transfer", original)
    assert cache.compile(**args, metrics={}).fingerprint


@pytest.mark.parametrize("copy_key_on_miss", [False, True])
@pytest.mark.parametrize(
    "component",
    [
        "manifest",
        "manifest_order",
        "address",
        "session",
        "offset",
        "shape",
        "global_shape",
        "source_dtype",
        "elsize",
        "source_elsize",
        "session_agent",
        "session_device",
        "agent_metadata",
        "capture",
        "layout",
        "cap",
        "copy_source",
        "copy_parameter",
        "copy_offset",
        "copy_shape",
        "copy_stride",
        "copy_dtype",
        "copy_operations",
        "unsupported",
        "pack",
        "reuse_complete",
        "digest_mode",
        "segment_budget",
        "staging_device",
        "staging_buffers",
        "total_staging_bytes",
    ],
)
def test_bounded_plan_cache_invalidates_each_planning_input(
    monkeypatch, component, copy_key_on_miss
):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setenv("MX_REFIT_COPY_PLAN_KEY_ON_MISS", str(int(copy_key_on_miss)))
    monkeypatch.setenv("MX_REFIT_PACK_MODULES", "0")
    monkeypatch.setenv("MX_REFIT_REUSE_COMPLETE_PLAN", "0")
    monkeypatch.setenv("MX_RESHARD_MAX_SEGMENTS_PER_COPY", "64")
    cache = _BoundedPlanCache()
    args = _bounded_cache_inputs()
    metrics = {}
    first = cache.compile(**args, metrics=metrics)
    source = args["resolved"].sources["weight"]
    shard = source.shards[0]
    if component == "manifest":
        args["manifests"][0] += b" "
    elif component == "manifest_order":
        args["manifests"].reverse()
    elif component == "address":
        shard.addr += 16
    elif component == "session":
        shard.session = "replacement"
    elif component == "offset":
        shard.shard_offset = (1,)
    elif component == "shape":
        shard.shape = (1,)
    elif component == "global_shape":
        source.global_shape = (8,)
    elif component == "source_dtype":
        source.dtype = torch.bfloat16
        source.elsize = 2
        for item in source.shards:
            item.elsize = 2
    elif component == "elsize":
        shard.elsize = 8
    elif component == "source_elsize":
        source.elsize = 8
    elif component == "session_agent":
        args["resolved"].session_to_agent["a"] = "replacement"
    elif component == "session_device":
        args["resolved"].session_to_device["a"] = 1
    elif component == "agent_metadata":
        args["resolved"].agent_metadata["a"] = b"replacement"
    elif component == "capture":
        args["capture"].unattributed = 1
    elif component == "unsupported":
        args["capture"].unsupported.append("weight")
    elif component.startswith("copy_"):
        attribute, value = {
            "copy_source": ("src_name", "missing-source"),
            "copy_parameter": ("param_name", "unknown.weight"),
            "copy_offset": ("dest_offset", 1),
            "copy_shape": ("dest_shape", (2,)),
            "copy_stride": ("dest_stride", (2,)),
            "copy_dtype": ("dest_dtype", torch.bfloat16),
            "copy_operations": ("op_chain", (("view", ([4],), ()),)),
        }[component]
        setattr(args["capture"].copies[0], attribute, value)
    elif component == "layout":
        args["parameter_layout"]["layer.extra"] = ((4,), torch.float32)
    elif component == "cap":
        args["max_staging_bytes"] = 768
    elif component.startswith("staging_") or component == "total_staging_bytes":
        args[component] = {
            "staging_device": "cpu",
            "staging_buffers": 2,
            "total_staging_bytes": 1024,
        }[component]
    else:
        variable, value = {
            "pack": ("MX_REFIT_PACK_MODULES", "1"),
            "reuse_complete": ("MX_REFIT_REUSE_COMPLETE_PLAN", "1"),
            "digest_mode": ("MX_RESHARD_PUBLISH_DIGEST", "1"),
            "segment_budget": ("MX_RESHARD_MAX_SEGMENTS_PER_COPY", "128"),
        }[component]
        monkeypatch.setenv(variable, value)
    try:
        assert cache.compile(**args, metrics=metrics) is not first
    except IncompleteRefit:
        assert cache._entry is None
    assert metrics["plan_cache_hits"] == 0
    assert metrics["plan_cache_misses"] == 1




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
    complete = _plan_staged_transfer(capture, {"w": source})
    if defect == "fallback":
        complete.fallback.append("w")
    with pytest.raises(IncompleteRefit):
        _bounded_batches(capture, layout, {"w": source}, 512, complete_plan=complete)
