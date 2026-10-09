# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
import weakref
from dataclasses import replace
from types import SimpleNamespace
from collections.abc import Iterator

import modelexpress_rl.inference.nixl_staged_transfer as module
from modelexpress_rl.inference.plan import StreamingSettings

import pytest
import torch
from modelexpress.accelerators import NIXL_ACCELERATOR_MEM_TYPE
from modelexpress.refit.reshard.rendezvous import (
    PublishedShard,
    PublishedTensor,
    wrap_rendezvous_blob,
)
from modelexpress.refit.reshard.types import (
    CaptureResult,
    IncompleteRefit,
    RecordedCopy,
)


@pytest.fixture
def harness(monkeypatch, request) -> Iterator[SimpleNamespace]:
    """Real planning and byte copies with synthetic capture metadata and CPU arenas."""
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setenv("MX_RESHARD_MAX_SEGMENTS_PER_COPY", "1")
    monkeypatch.setenv("MX_REFIT_CACHE_BOUNDED_PLANS", "1")
    monkeypatch.setenv("MX_REFIT_CACHE_RESOLVED_SOURCES", "1")
    monkeypatch.setenv("MX_REFIT_COPY_PLAN_KEY_ON_MISS", "1")
    monkeypatch.setenv("MX_REFIT_PACK_MODULES", "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    sources = {
        name: torch.arange(16, dtype=torch.float32).reshape(4, 4) + index / 16
        for index, name in enumerate(("exact", "full", "convert"))
    }
    captures = []
    copies = [
        RecordedCopy(src, ops, name, 2, (4, 4), (4, 1), dtype)
        for src, ops, name, dtype in (
            ("exact", (), "a.weight", torch.float32),
            ("full", (("transpose", (0, 1), ()),), "b.weight", torch.float32),
            ("convert", (), "c.weight", torch.bfloat16),
            ("exact", (), "d.weight", torch.float32),
        )
    ]
    capture = CaptureResult(copies=copies)
    layout = {copy.param_name: ((20,), copy.dest_dtype) for copy in copies}
    transports = []
    events = []

    class Manager:
        def __init__(self, **kwargs):
            self.registered = {}

        def initialize(self):
            pass

        def shutdown(self):
            self.registered.clear()

        def add_remote_agent(self, metadata):
            return metadata.decode()

        def register_tensors(self, tensors):
            self.registered.update(tensors)

        def register_dram_buffer(self, tensor):
            handle = object()
            self.registered[handle] = tensor
            return handle

        def deregister_memory(self, handle):
            del self.registered[handle]

    class Transport:
        def __init__(self, manager, agents, devices, **kwargs):
            self.posts = []
            self.posted = []
            self.awaited = []
            self.fail_post = False
            self.fail_wait = False
            # An omitted override uses the manager's accelerator memory type.
            self.mem_type = kwargs["local_mem_type"] or NIXL_ACCELERATOR_MEM_TYPE
            transports.append(self)

        def post_reads(self, descriptors):
            events.append("post")
            if self.fail_post:
                raise RuntimeError("post failed")
            self.posts.append(tuple(descriptors))
            for descriptor in descriptors:
                ctypes.memmove(
                    descriptor.dst_addr, descriptor.src_addr, descriptor.nbytes
                )
            handle = SimpleNamespace()
            self.posted.append(handle)
            # The transport owns its argument list; changing it must not alter cache.
            descriptors.clear()
            return [handle]

        def await_reads(self, posted):
            events.append("await")
            if self.fail_wait:
                raise RuntimeError("wait failed")
            self.awaited.extend(posted)

    monkeypatch.setattr(module, "NixlTransferManager", Manager)
    monkeypatch.setattr(module, "NixlReshardTransport", Transport)
    monkeypatch.setattr(
        module._NixlStagedTransfer,
        "_allocate_arena",
        lambda self, size: torch.empty(size, dtype=torch.uint8),
    )
    transfer = module._NixlStagedTransfer(
        agent_name="target",
        device_id=0,
        device=torch.device("cuda:0"),
        listen_port=None,
        streaming=getattr(request, "param", StreamingSettings(1024, "cpu")),
    )

    def manifests():
        return [
            wrap_rendezvous_blob(
                b"source",
                "source",
                "source:19000",
                [
                    PublishedTensor(
                        name=name,
                        dtype="torch.float32",
                        elsize=4,
                        full_shape=(4, 4),
                        shards=[
                            PublishedShard(
                                agent_name="source",
                                device_id=0,
                                addr=tensor.data_ptr(),
                                shard_offset=(0, 0),
                                shape=(4, 4),
                            )
                        ],
                    )
                    for name, tensor in sources.items()
                ],
            )
        ]

    def capture_layout(manifest):
        captures.append(manifest)
        return capture, layout

    def prepare() -> object:
        return transfer.prepare(
            manifests=manifests(),
            capture_layout=capture_layout,
        )

    def collect(prepared):
        metrics, installed = {}, {}
        for tensors in transfer.iter_bounded(prepared, metrics):
            installed.update({name: value.clone() for name, value in tensors.items()})
        return metrics, installed

    state = SimpleNamespace(
        transfer=transfer,
        streaming=getattr(request, "param", StreamingSettings(1024, "cpu")),
        sources=sources,
        capture=capture,
        layout=layout,
        prepare=prepare,
        collect=collect,
        captures=captures,
        transports=transports,
        events=events,
    )
    yield state
    transfer.close()


def _check_values(harness, installed):
    for source, name, dtype, transpose in (
        ("exact", "a.weight", torch.float32, False),
        ("full", "b.weight", torch.float32, True),
        ("convert", "c.weight", torch.bfloat16, False),
        ("exact", "d.weight", torch.float32, False),
    ):
        expected = torch.zeros(20, dtype=dtype)
        value = harness.sources[source]
        expected[2:18].copy_((value.T if transpose else value).reshape(-1))
        assert torch.equal(installed[name], expected)


@pytest.mark.parametrize(
    "harness",
    [
        StreamingSettings(1024, device, count)
        for device in ("cpu", "cuda")
        for count in (1, 2)
    ],
    indirect=True,
)
@pytest.mark.parametrize("pack", [False, True])
def test_warm_descriptors_still_transfer_new_values(harness, monkeypatch, pack) -> None:
    monkeypatch.setenv("MX_REFIT_PACK_MODULES", str(int(pack)))
    prepare = harness.prepare
    first = prepare()
    assert any(batch.plan.full_pulls for batch in first.batches)
    assert any(batch.plan.converts for batch in first.batches)
    assert any(batch.plan.segments for batch in first.batches)
    cold, installed = harness.collect(first)
    _check_values(harness, installed)
    assert cold["descriptor_builds"] == len(first.batches)
    assert cold["descriptor_cache_hits"] == 0
    for values in harness.sources.values():
        values.add_(3)
    for arena in harness.transfer._staging_arenas:
        arena.fill_(255)
    second = prepare()
    warm, installed = harness.collect(second)
    _check_values(harness, installed)
    assert warm["descriptor_cache_hits"] == len(second.batches)
    assert warm["descriptor_cache_misses"] == warm["descriptor_builds"] == 0
    assert second.transport.mem_type == (
        "DRAM" if harness.streaming.staging_device == "cpu" else "VRAM"
    )
    assert first.transport.posts == second.transport.posts
    assert second.transport.awaited == second.transport.posted
    descriptor = second.transport.posts[0][0]
    with pytest.raises(AttributeError):
        descriptor.dst_addr = 0


@pytest.mark.parametrize("change", ["address", "capture", "disabled"])
def test_changed_plan_does_not_reuse_descriptors(harness, monkeypatch, change) -> None:
    first = harness.prepare()
    harness.collect(first)
    if change == "address":
        harness.sources["exact"] = harness.sources["exact"].clone() + 5
    elif change == "capture":
        harness.capture.copies[0] = replace(harness.capture.copies[0], dest_offset=1)
    else:
        monkeypatch.setenv("MX_REFIT_CACHE_BOUNDED_PLANS", "0")
    second = harness.prepare()
    metrics, _ = harness.collect(second)
    assert metrics["descriptor_cache_hits"] == 0
    assert metrics["descriptor_builds"] == len(second.batches)


@pytest.mark.parametrize(
    "change", ["replace", "same_address", "reorder", "resize", "registration"]
)
@pytest.mark.parametrize("harness", [StreamingSettings(1024, "cpu", 2)], indirect=True)
def test_arena_change_after_prepare_is_checked_before_each_post(
    harness, change
) -> None:
    first = harness.prepare()
    harness.collect(first)
    prepared = harness.prepare()
    arenas = harness.transfer._staging_arenas
    if change == "replace":
        arenas[1] = arenas[1].clone()
    elif change == "same_address":
        arenas[1] = arenas[1].view_as(arenas[1])
    elif change == "reorder":
        arenas.reverse()
    elif change == "resize":
        arenas[1].resize_(arenas[1].numel() + 256)
    else:
        harness.transfer._release_staging_registrations()
        harness.transfer._staging_registrations = [
            harness.transfer._manager.register_dram_buffer(arena) for arena in arenas
        ]
    metrics, installed = harness.collect(prepared)
    _check_values(harness, installed)
    assert metrics["descriptor_cache_hits"] == 0
    assert metrics["descriptor_builds"] == len(prepared.batches)
    assert harness.transfer._descriptor_cache is None


def test_arena_change_between_batches_is_not_hidden_by_first_hit(harness):
    harness.collect(harness.prepare())
    prepared = harness.prepare()
    metrics = {}
    iterator = harness.transfer.iter_bounded(prepared, metrics)
    next(iterator)
    assert metrics["descriptor_cache_hits"] == 1
    harness.transfer._staging_arenas[0] = harness.transfer._staging_arenas[0].clone()
    for tensors in iterator:
        assert tensors
    assert metrics["descriptor_cache_hits"] == 1
    assert metrics["descriptor_builds"] == len(prepared.batches) - 1


@pytest.mark.parametrize(
    "failure", ["coverage", "transport", "prepared", "registration"]
)
def test_failed_prepare_discards_descriptors(harness, monkeypatch, failure):
    harness.collect(harness.prepare())
    assert harness.transfer._descriptor_cache is not None
    if failure == "coverage":
        harness.layout["missing.weight"] = ((4,), torch.float32)
        expected = IncompleteRefit
    else:

        def fail(*args, **kwargs):
            raise RuntimeError("preparation failed")

        if failure == "transport":
            monkeypatch.setattr(module, "NixlReshardTransport", fail)
        elif failure == "prepared":
            monkeypatch.setattr(module, "_PreparedBoundedTransfer", fail)
        else:
            harness.transfer.reset_workspace()
            monkeypatch.setattr(harness.transfer._manager, "register_dram_buffer", fail)
        expected = RuntimeError
    with pytest.raises(expected):
        harness.prepare()
    assert harness.transfer._descriptor_cache is None


@pytest.mark.parametrize("failure", ["post", "wait", "abandon", "drain"])
@pytest.mark.parametrize("harness", [StreamingSettings(1024, "cpu", 2)], indirect=True)
def test_incomplete_iteration_discards_cache_and_preserves_drain(
    harness, failure
) -> None:
    harness.collect(harness.prepare())
    prepared = harness.prepare()
    if failure in ("post", "wait"):
        setattr(prepared.transport, f"fail_{failure}", True)
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            harness.collect(prepared)
    else:
        iterator = harness.transfer.iter_bounded(prepared, {})
        next(iterator)
        assert len(prepared.transport.posted) == 2
        if failure == "drain":
            prepared.transport.fail_wait = True
            with pytest.raises(RuntimeError, match="could not be drained"):
                iterator.close()
        else:
            iterator.close()
            assert prepared.transport.awaited == prepared.transport.posted
    assert harness.transfer._descriptor_cache is None


@pytest.mark.parametrize("cleanup", ["reset", "close"])
def test_metadata_does_not_own_arenas_or_handles(harness, cleanup):
    prepared = harness.prepare()
    harness.collect(prepared)
    entry = harness.transfer._descriptor_cache
    arena_refs = [weakref.ref(arena) for arena in harness.transfer._staging_arenas]
    transport_ref = weakref.ref(prepared.transport)
    # Drop test-only observation lists and all prepared state.
    harness.transports.clear()
    del prepared
    harness.transfer._active = None
    getattr(harness.transfer, cleanup if cleanup == "close" else "reset_workspace")()
    gc.collect()
    assert all(reference() is None for reference in arena_refs)
    assert transport_ref() is None
    assert entry.batches and harness.transfer._descriptor_cache is None


@pytest.mark.parametrize("harness", [StreamingSettings(2048, "cpu")], indirect=True)
def test_descriptor_order_duplicates_and_empty_reads_survive_reuse(
    harness, monkeypatch
) -> None:
    monkeypatch.setenv("MX_REFIT_PACK_MODULES", "1")
    original = harness.transfer._descriptors
    builds = []

    def descriptors(*args, **kwargs):
        result = original(*args, **kwargs)
        result += [result[0], replace(result[0], nbytes=0)]
        builds.append(
            tuple((d.session, d.src_addr, d.dst_addr, d.nbytes) for d in result)
        )
        return result

    monkeypatch.setattr(harness.transfer, "_descriptors", descriptors)
    first = harness.prepare()
    assert len(first.batches) == 1
    harness.collect(first)
    second = harness.prepare()
    metrics, installed = harness.collect(second)
    _check_values(harness, installed)
    assert tuple(second.transport.posts[0]) == builds[0]
    assert metrics["descriptor_cache_hits"] == 1


def test_descriptor_build_time_stays_outside_wire_time(harness, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: now[0])
    original = harness.transfer._descriptors

    def delayed_descriptors(*args, **kwargs):
        now[0] += 100
        return original(*args, **kwargs)

    monkeypatch.setattr(harness.transfer, "_descriptors", delayed_descriptors)
    cold, _ = harness.collect(harness.prepare())
    warm, _ = harness.collect(harness.prepare())
    assert now[0] == 400
    assert cold["wire_s"] == warm["wire_s"] == 0
    assert warm["descriptor_builds"] == 0
