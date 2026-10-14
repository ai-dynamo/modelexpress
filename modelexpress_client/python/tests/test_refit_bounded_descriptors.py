# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from modelexpress.refit.reshard.rendezvous import structural_manifest_digest
import ctypes
import hashlib
import gc
import weakref
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace, MappingProxyType

import modelexpress_rl.inference.nixl_staged_transfer as module
from modelexpress_rl.inference.adapter import TrainerSourceShard
from modelexpress_rl.inference.plan import StreamingSettings

import pytest
import torch
from modelexpress import envs
from modelexpress.accelerators import NIXL_ACCELERATOR_MEM_TYPE
from modelexpress.refit.reshard.rendezvous import (
    PublishedShard,
    PublishedTensor,
    wrap_rendezvous_blob,
)
from modelexpress.refit.reshard.verify import tensor_digest
from modelexpress_rl.inference.plan import TrainerSourceSnapshot
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
    parameters = getattr(getattr(request.node, "callspec", None), "params", {})
    settings = getattr(request, "param", StreamingSettings(1024, "cpu"))
    pack = parameters.get("pack", False)
    conversion = parameters.get("native_conversion")
    if isinstance(settings, tuple):
        settings, pack = settings
    monkeypatch.setenv("MX_REFIT_PACK_MODULES", str(int(pack)))
    diagnostic = parameters.get("diagnostic")
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_PLAN", str(int(diagnostic == "plan")))
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
    if conversion:
        copies = [replace(
            record,
            src_name=f"native.{record.src_name}",
            op_chain=((("transpose", (0, 1), ()),) if conversion == "transpose" else ())
            + record.op_chain,
        ) for record in copies]
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
            events.append(f"connect:{metadata.decode()}")
            return metadata.decode()

        def register_tensors(self, tensors):
            events.append("register")
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

    def capture_layout(manifest):
        captures.append(manifest)
        return capture, layout

    transfer = module._NixlStagedTransfer(
        generator_fingerprint=lambda: (),
        agent_name="target",
        device_id=0,
        device=torch.device("cuda:0"),
        listen_port=None,
        streaming=settings,
        capture_layout=capture_layout,
    )

    def manifests():
        return [
            wrap_rendezvous_blob(
                b"source",
                "source",
                "source:19000",
                [
                    PublishedTensor(
                        name=f"native.{name}" if conversion else name,
                        dtype="torch.float32",
                        elsize=4,
                        full_shape=(4, 4),
                        shards=[
                            PublishedShard(
                                agent_name="source",
                                device_id=0,
                                addr=tensor.data_ptr(),
                                digest=tensor_digest(tensor) if envs.MX_RESHARD_PUBLISH_DIGEST else None,
                                shard_offset=(0, 0),
                                shape=(4, 4),
                            )
                        ],
                    )
                    for name, tensor in sources.items()
                ],
            )
        ]

    def prepare(*, generation: int = 1) -> module._PreparedBoundedTransfer:
        blobs = manifests()
        snapshot = TrainerSourceSnapshot("mesh", generation, tuple(
            TrainerSourceShard("slot", "worker", hashlib.sha256(blob).hexdigest(), "source:19000", blob, structural_manifest_digest(blob))
            for blob in blobs
        ))
        cached = transfer.cached_trainer_source()
        if (
            not parameters.get("equivalent_snapshot", False)
            and cached is not None
            and cached.physical_fingerprint == snapshot.physical_fingerprint
            and tuple(shard.manifest for shard in cached.shards) == tuple(blobs)
        ):
            snapshot = cached
        return transfer.prepare(
            trainer_snapshot=snapshot,
            manifests=blobs,
            checksums=MappingProxyType({
                (name, "source", tensor.data_ptr(), (0, 0), (4, 4)): tensor_digest(tensor)
                for name, tensor in sources.items()
            }) if envs.MX_RESHARD_PUBLISH_DIGEST else None,
        )

    def collect(prepared):
        metrics, installed = {}, {}
        for tensors in transfer.iter_bounded(prepared, metrics):
            installed.update({name: value.clone() for name, value in tensors.items()})
        return metrics, installed

    state = SimpleNamespace(
        transfer=transfer,
        streaming=settings,
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


def _check_values(harness, installed, *, source_transposed=False, check_padding=True):
    for source, name, dtype, transpose in (
        ("exact", "a.weight", torch.float32, False),
        ("full", "b.weight", torch.float32, True),
        ("convert", "c.weight", torch.bfloat16, False),
        ("exact", "d.weight", torch.float32, False),
    ):
        expected = torch.zeros(20, dtype=dtype)
        value = harness.sources[source]
        if source_transposed:
            value = value.T
        expected[2:18].copy_((value.T if transpose else value).reshape(-1))
        if check_padding:
            assert torch.equal(installed[name], expected)
        else:
            assert torch.equal(installed[name][2:18], expected[2:18])


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
@pytest.mark.parametrize("equivalent_snapshot", [False, True])
def test_warm_descriptors_still_transfer_new_values(
    harness, monkeypatch, pack, equivalent_snapshot
) -> None:
    monkeypatch.setenv("MX_REFIT_PACK_MODULES", str(int(pack)))
    prepare = harness.prepare
    first = prepare()
    assert any(batch.transfer_plan.full_pulls for batch in first.batches)
    assert any(batch.transfer_plan.converts for batch in first.batches)
    assert any(batch.transfer_plan.segments for batch in first.batches)
    cold, installed = harness.collect(first)
    _check_values(harness, installed)
    assert len(harness.captures) == 1
    assert cold["descriptor_builds"] == len(first.batches)
    assert cold["descriptor_cache_hits"] == 0
    for values in harness.sources.values():
        values.add_(3)
    for arena in harness.transfer._staging_arenas:
        arena.fill_(255)
    second = prepare()
    warm, installed = harness.collect(second)
    _check_values(harness, installed)
    assert len(harness.captures) == 1
    assert warm["descriptor_cache_hits"] == len(second.batches)
    assert warm["descriptor_cache_misses"] == warm["descriptor_builds"] == 0
    assert second.transport.mem_type == (
        "DRAM" if harness.streaming.staging_device == "cpu" else "VRAM"
    )
    assert second.transport.awaited == second.transport.posted


@pytest.mark.parametrize("harness", [None, StreamingSettings(1024, "cpu")], indirect=True)
def test_new_manifest_checksums_reuse_preparation(harness, monkeypatch):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    monkeypatch.setattr(harness.transfer, "_device", torch.device("cpu"))
    monkeypatch.setattr(module, "classic_cuda_alloc", nullcontext)
    for update in range(3):
        for tensor in harness.sources.values():
            tensor.add_(1)
        prepared = harness.prepare()
        if harness.streaming is None:
            installed = harness.transfer.stage(prepared).tensors
        else:
            _, installed = harness.collect(prepared)
        _check_values(harness, installed, check_padding=harness.streaming is not None)
        assert prepared.metrics["plan_cache_hits"] == int(update > 0)
    assert len(harness.captures) == 1


@pytest.mark.parametrize("native_conversion", ["rename", "transpose"])
@pytest.mark.parametrize("harness", [None, StreamingSettings(1024, "cpu")], indirect=True)
@pytest.mark.parametrize("pack", [False, True])
def test_native_conversion_composes_with_transfer(harness, native_conversion, pack, monkeypatch):
    monkeypatch.setattr(harness.transfer, "_device", torch.device("cpu"))
    monkeypatch.setattr(module, "classic_cuda_alloc", nullcontext)
    prepared = harness.prepare()
    if harness.streaming is None:
        installed = harness.transfer.stage(prepared).tensors
    else:
        _, installed = harness.collect(prepared)
    _check_values(harness, installed, source_transposed=native_conversion == "transpose",
                  check_padding=harness.streaming is not None)


def test_replaced_source_buffers_transfer_current_values(harness) -> None:
    for update in range(3):
        if update:
            harness.sources.update({
                name: value.clone() + update
                for name, value in harness.sources.items()
            })
        _, installed = harness.collect(harness.prepare(generation=update + 1))
        _check_values(harness, installed)


@pytest.mark.parametrize(
    "failure", ["coverage", "prepared", "registration"]
)
def test_failed_prepare_discards_descriptors(harness, monkeypatch, failure):
    harness.collect(harness.prepare())
    assert harness.transfer._descriptor_cache is not None
    if failure == "coverage":
        harness.sources["exact"] = harness.sources["exact"].clone()
        harness.layout["missing.weight"] = ((4,), torch.float32)
        expected = IncompleteRefit
    else:

        def fail(*args, **kwargs):
            raise RuntimeError("preparation failed")

        if failure == "prepared":
            monkeypatch.setattr(module, "_PreparedBoundedTransfer", fail)
        else:
            harness.transfer.reset_workspace()
            monkeypatch.setattr(harness.transfer._manager, "register_dram_buffer", fail)
        expected = RuntimeError
    with pytest.raises(expected):
        harness.prepare(generation=2 if failure == "coverage" else 1)
    assert harness.transfer._descriptor_cache is None


@pytest.mark.parametrize("failure", ["post", "wait", "abandon", "drain"])
@pytest.mark.parametrize("harness", [StreamingSettings(1024, "cpu", 2)], indirect=True)
def test_incomplete_iteration_discards_cache_and_preserves_drain(harness, failure):
    harness.collect(harness.prepare())
    prepared = harness.prepare()
    if failure in ("post", "wait"):
        setattr(prepared.transport, f"fail_{failure}", True)
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            harness.collect(prepared)
    else:
        posted_before = len(prepared.transport.posted)
        iterator = harness.transfer.iter_bounded(prepared, {})
        next(iterator)
        assert len(prepared.transport.posted) - posted_before == 2
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
    assert entry and harness.transfer._descriptor_cache is None


@pytest.mark.parametrize("harness", [(StreamingSettings(2048, "cpu"), True)], indirect=True)
def test_descriptor_order_duplicates_and_empty_reads_survive_reuse(
    harness, monkeypatch
):
    monkeypatch.setenv("MX_REFIT_PACK_MODULES", "1")
    original = harness.transfer._descriptors
    builds = []

    def descriptors(*args, **kwargs):
        result = original(*args, **kwargs)
        result += (result[0], replace(result[0], nbytes=0))
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
    assert tuple(
        (d.session, d.src_addr, d.dst_addr, d.nbytes) for d in second.transport.posts[0]
    ) == builds[0]
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
