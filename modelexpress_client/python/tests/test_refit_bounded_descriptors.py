# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import hashlib
import gc
import weakref
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace, MappingProxyType

import modelexpress_rl.inference.nixl_staged_transfer as module
from tests.helpers.refit_replay import install_replay
from modelexpress_rl.inference.adapter import TrainerSourceShard
from modelexpress_rl.inference.plan import StreamingSettings

import pytest
import torch
from modelexpress_rl import envs
from modelexpress.accelerators import NIXL_ACCELERATOR_MEM_TYPE
from modelexpress_rl.refit.reshard.rendezvous import (
    PublishedShard,
    PublishedTensor,
    wrap_rendezvous_blob,
)
from modelexpress_rl.refit.reshard.verify import tensor_digest
from modelexpress_rl.inference.plan import TrainerSourceSnapshot
from modelexpress_rl.version import MeshIdentity
from modelexpress_rl.refit.reshard.types import (
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
    conversion = parameters.get("native_conversion")
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
    capture = CaptureResult(source_replay=True, copies=copies)
    layout = {copy.param_name: ((20,), copy.dest_dtype) for copy in copies}
    transports = []
    events = []

    class Manager:
        def __init__(self, **kwargs):
            self.registered = {}

        def initialize(self):
            events.append("initialize")

        def shutdown(self):
            events.append("shutdown")
            self.registered.clear()

        def add_remote_agent(self, metadata):
            events.append(f"connect:{metadata.decode()}")
            return metadata.decode().split(":", 1)[0]

        def register_tensors(self, tensors):
            events.append("register")
            self.registered.update(tensors)

        def register_dram_buffer(self, tensor):
            events.append("register")
            handle = object()
            self.registered[handle] = tensor
            return handle

        def deregister_memory(self, handle):
            events.append("deregister")
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
        snapshot = TrainerSourceSnapshot(MeshIdentity("mesh", generation), tuple(
            TrainerSourceShard("slot", "worker", hashlib.sha256(blob).hexdigest(), "source:19000", blob)
            for blob in blobs
        ))
        state.snapshot = snapshot
        return transfer.prepare(
            trainer_snapshot=snapshot,
            checksums=MappingProxyType({
                (name, "source", tensor.data_ptr(), (0, 0), (4, 4)): tensor_digest(tensor)
                for name, tensor in sources.items()
            }) if envs.MX_RESHARD_PUBLISH_DIGEST else None,
        )

    def collect(prepared):
        metrics, installed = {}, {}
        for tensors in transfer.iter_bounded(prepared, metrics):
            install_replay(tensors, layout, installed)
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


def _check_values(
    harness, installed, *, exact_rows: int | None = None, check_padding: bool = True,
    source_transposed: bool = False,
) -> None:
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
        if source == "exact" and exact_rows is not None:
            value = value[:exact_rows]
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
def test_warm_descriptors_still_transfer_new_values(
    harness, monkeypatch
) -> None:
    prepare = harness.prepare
    first = prepare()
    assert any(batch.transfer_plan.full_pulls for batch in first.batches)
    assert all(batch.transfer_plan.source_replay for batch in first.batches)
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
    assert first.transport.posts == second.transport.posts
    assert second.transport.awaited == second.transport.posted


@pytest.mark.parametrize("harness", [None], indirect=True)
def test_full_copy_reuses_workspace_until_reset(harness, monkeypatch) -> None:
    monkeypatch.setattr(harness.transfer, "_device", torch.device("cpu"))
    monkeypatch.setattr(module, "classic_cuda_alloc", nullcontext)
    first = harness.prepare()
    harness.transfer.stage(first)
    for values in harness.sources.values():
        values.add_(3)

    with monkeypatch.context() as warm:
        def unexpected_setup(*args, **kwargs):
            pytest.fail("unchanged full-copy workspace should be reused")

        warm.setattr(harness.transfer, "_ensure_workspace", unexpected_setup)
        second = harness.prepare()
        installed = install_replay(harness.transfer.stage(second).tensors, harness.layout)
        _check_values(harness, installed, check_padding=False)
        assert second.descriptors is first.descriptors

    harness.transfer.reset_workspace()
    third = harness.prepare()
    installed = install_replay(harness.transfer.stage(third).tensors, harness.layout)
    _check_values(harness, installed, check_padding=False)
    assert third.descriptors is not second.descriptors


@pytest.mark.parametrize("native_conversion", ["rename", "transpose"])
@pytest.mark.parametrize("harness", [None, StreamingSettings(1024, "cpu")], indirect=True)
def test_native_conversion_composes_with_transfer(harness, native_conversion, monkeypatch):
    monkeypatch.setattr(harness.transfer, "_device", torch.device("cpu"))
    monkeypatch.setattr(module, "classic_cuda_alloc", nullcontext)
    prepared = harness.prepare()
    if harness.streaming is None:
        installed = install_replay(harness.transfer.stage(prepared).tensors, harness.layout)
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
def test_failed_prepare_keeps_descriptors_unless_destinations_change(harness, monkeypatch, failure):
    harness.collect(harness.prepare())
    with monkeypatch.context() as fault:
        if failure == "coverage":
            harness.sources["exact"] = harness.sources["exact"].clone()
            harness.layout["missing.weight"] = ((4,), torch.float32)
            expected = IncompleteRefit
        else:
            def fail(*args, **kwargs) -> None:
                raise RuntimeError("preparation failed")

            if failure == "prepared":
                fault.setattr(module, "_PreparedBoundedTransfer", fail)
            else:
                harness.transfer.reset_workspace()
                fault.setattr(harness.transfer._manager, "register_dram_buffer", fail)
            expected = RuntimeError
        with pytest.raises(expected):
            harness.prepare(generation=2 if failure == "coverage" else 1)
    harness.layout.pop("missing.weight", None)
    prepared = harness.prepare(generation=2 if failure == "coverage" else 1)
    metrics, installed = harness.collect(prepared)
    _check_values(harness, installed)
    assert metrics["descriptor_builds"] == (0 if failure == "prepared" else len(prepared.batches))
    assert prepared.transport.awaited == prepared.transport.posted


@pytest.mark.parametrize("failure", ["post", "wait", "abandon", "drain"])
@pytest.mark.parametrize("harness", [StreamingSettings(1024, "cpu", 2)], indirect=True)
def test_incomplete_iteration_keeps_descriptors_and_preserves_drain(harness, failure):
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
            assert len(prepared.transport.awaited) < len(prepared.transport.posted)
            registered = harness.transfer._manager.registered.values()
            assert all(
                any(resource is arena for resource in registered)
                for arena in harness.transfer._staging_arenas
            )
            return
        else:
            iterator.close()
            assert prepared.transport.awaited == prepared.transport.posted
    prepared = harness.prepare()
    metrics, installed = harness.collect(prepared)
    _check_values(harness, installed)
    assert metrics["descriptor_builds"] == 0
    assert prepared.transport.awaited == prepared.transport.posted


@pytest.mark.parametrize("cleanup", ["reset", "close"])
def test_reset_and_close_release_arenas_and_handles(harness, cleanup) -> None:
    prepared = harness.prepare()
    harness.collect(prepared)
    arena_refs = [weakref.ref(arena) for arena in harness.transfer._staging_arenas]
    transport_ref = weakref.ref(prepared.transport)
    # Drop test-only observation lists and all prepared state.
    harness.transports.clear()
    del prepared
    getattr(harness.transfer, cleanup if cleanup == "close" else "reset_workspace")()
    gc.collect()
    assert all(reference() is None for reference in arena_refs)
    assert transport_ref() is None


@pytest.mark.parametrize("harness", [StreamingSettings(2048, "cpu")], indirect=True)
def test_descriptor_order_duplicates_and_empty_reads_survive_reuse(
    harness, monkeypatch
):
    original = harness.transfer._compile_plan
    def compile_with_extra_reads(*args, **kwargs):
        schedule = original(*args, **kwargs)
        batch = schedule.batches[0]
        plan = batch.transfer_plan
        pull = plan.full_pulls[0]
        pull = replace(pull, segments=pull.segments + (pull.segments[0], pull.segments[0]._replace(nbytes=0)))
        plan = replace(plan, full_pulls=(pull, *plan.full_pulls[1:]))
        return replace(schedule, batches=(replace(batch, transfer_plan=plan),))

    monkeypatch.setattr(harness.transfer, "_compile_plan", compile_with_extra_reads)
    first = harness.prepare()
    assert len(first.batches) == 1
    harness.collect(first)
    second = harness.prepare()
    metrics, installed = harness.collect(second)
    _check_values(harness, installed)
    assert second.transport.posts[0] == first.descriptors[0]
    assert any(read.nbytes == 0 for read in first.descriptors[0])
    positions = [(read.src_addr, read.dst_addr, read.nbytes) for read in first.descriptors[0]]
    assert len(positions) > len(set(positions))
    assert metrics["descriptor_cache_hits"] == 1


def test_descriptor_build_time_stays_outside_wire_time(harness, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: now[0])
    original = module.TransferBindingCache.bind

    def delayed_bind(self, *args, **kwargs):
        now[0] += 100
        return original(self, *args, **kwargs)

    monkeypatch.setattr(module.TransferBindingCache, "bind", delayed_bind)
    cold, _ = harness.collect(harness.prepare())
    warm, _ = harness.collect(harness.prepare())
    assert now[0] == 100 * cold["descriptor_builds"]
    assert cold["wire_s"] == warm["wire_s"] == 0
    assert warm["descriptor_builds"] == 0


@pytest.mark.parametrize("mesh_id,generation,reuse", [("mesh", 1, True), ("mesh", 2, False), ("other", 1, False)])
def test_logical_plan_lifetime_is_mesh_pair(harness, mesh_id, generation, reuse) -> None:
    first = harness.prepare()
    harness.collect(first)
    harness.sources.update({name: value.clone() + 17 for name, value in harness.sources.items()})
    snapshot = replace(
        _agent_registration_snapshot(harness, generation=generation, token=b"replica-b", worker="replacement"),
        mesh_identity=MeshIdentity(mesh_id, generation),
    )
    captures = []

    def capture_layout(schema) -> tuple:
        captures.append(schema)
        return harness.capture, harness.layout

    harness.transfer._capture_layout = capture_layout
    second = harness.transfer.prepare(
        trainer_snapshot=snapshot,
    )
    assert len(captures) == int(not reuse)
    metrics, installed = harness.collect(second)
    _check_values(harness, installed)
    assert metrics["descriptor_builds"] == len(second.batches)
    assert all(read.session == "replica-b" for batch in second.transport.posts for read in batch)


def test_native_workspace_reset_preserves_logical_plan(harness) -> None:
    first = harness.prepare()
    harness.collect(first)
    harness.transfer.reset_workspace()
    harness.captures.clear()
    second = harness.prepare()
    assert harness.captures == []
    metrics, installed = harness.collect(second)
    assert metrics["descriptor_builds"] == len(second.batches)
    _check_values(harness, installed)


def _larger_source_snapshot(harness) -> tuple[TrainerSourceSnapshot, list[torch.Tensor]]:
    source = harness.sources["exact"]
    harness.sources["exact"] = source = torch.cat((source[:4], torch.zeros((16, 4), dtype=source.dtype)))
    harness.capture.copies = tuple(
        replace(copy, op_chain=(("float", (), ()), ("__getitem__", (slice(0, 4),), ())))
        if copy.src_name == "exact" else copy for copy in harness.capture.copies
    )
    rows = [source[start:start + 2].clone() for start in range(0, source.shape[0], 2)]
    blob = wrap_rendezvous_blob(
        b"source", "source", "source:19000",
        [
            PublishedTensor(
                name=name, dtype="torch.float32", elsize=4,
                full_shape=tuple(source.shape) if name == "exact" else (4, 4),
                shards=[
                    PublishedShard(
                        agent_name="source", device_id=0, addr=value.data_ptr(),
                        shard_offset=(2 * index, 0), shape=(2, 4),
                    ) for index, value in enumerate(rows)
                ] if name == "exact" else [PublishedShard(
                    agent_name="source", device_id=0, addr=tensor.data_ptr(),
                    shard_offset=(0, 0), shape=(4, 4),
                )],
            ) for name, tensor in harness.sources.items()
        ],
    )
    snapshot = TrainerSourceSnapshot(MeshIdentity("mesh", 2), (TrainerSourceShard(
        "slot", "worker", hashlib.sha256(blob).hexdigest(), "source:19000", blob,
    ),))
    return snapshot, rows


@pytest.mark.parametrize("harness", [StreamingSettings(4096, "cpu", buffers) for buffers in (1, 2)], indirect=True)
def test_larger_source_scratch_grows_then_reuses_capacity(harness) -> None:
    initial = harness.prepare()
    cold, installed = harness.collect(initial)
    _check_values(harness, installed, exact_rows=4)
    for tensor in harness.sources.values():
        tensor.add_(5)
    snapshot, _rows = _larger_source_snapshot(harness)
    blob = snapshot.shards[0].metadata
    larger = harness.transfer.prepare(
         trainer_snapshot=snapshot,
    )
    grown, installed = harness.collect(larger)
    _check_values(harness, installed, exact_rows=4)
    assert grown["staging_peak_bytes"] > cold["staging_peak_bytes"]
    assert grown["staging_peak_bytes"] <= harness.streaming.max_staging_bytes
    assert harness.events.count("connect:source") == 2
    registered = list(harness.transfer._manager.registered.values())
    assert sum(value.numel() for value in registered) == grown["staging_peak_bytes"]
    for descriptors in larger.transport.posts:
        assert all(any(
            value.data_ptr() <= descriptor.dst_addr
            and descriptor.dst_addr + descriptor.nbytes <= value.data_ptr() + value.numel()
            for value in registered
        ) for descriptor in descriptors)
    transition = harness.events[harness.events.index("shutdown"):]
    assert transition.index("shutdown") < transition.index("initialize") < transition.index("connect:source") < transition.index("register") < transition.index("post")
    for tensor in harness.sources.values():
        tensor.add_(2)
    registrations_before_smaller = harness.events.count("register")
    harness.sources["exact"] = harness.sources["exact"][:4].clone()
    smaller = harness.prepare(generation=3)
    retained, installed = harness.collect(smaller)
    _check_values(harness, installed, exact_rows=4)
    assert retained["staging_peak_bytes"] == grown["staging_peak_bytes"]
    assert harness.events.count("register") == registrations_before_smaller
    assert harness.events.count("connect:source") == 2
    assert larger.transport.awaited == larger.transport.posted
    assert smaller.transport.awaited == smaller.transport.posted


def _agent_registration_snapshot(
    harness, *, generation: int, token: bytes = b"source", worker: str = "worker",
) -> TrainerSourceSnapshot:
    agent = token.decode().split(":", 1)[0]
    blob = wrap_rendezvous_blob(
        token, agent, f"{agent}:19000",
        [PublishedTensor(
            name=name, dtype="torch.float32", elsize=4, full_shape=(4, 4),
            shards=[PublishedShard(
                agent_name=agent, device_id=0, addr=tensor.data_ptr(),
                shard_offset=(0, 0), shape=(4, 4),
            )],
        ) for name, tensor in harness.sources.items()],
    )
    return TrainerSourceSnapshot(MeshIdentity("mesh", generation), (TrainerSourceShard(
        "slot", worker, hashlib.sha256(blob).hexdigest(), f"{agent}:19000", blob,
    ),))


def _prepare_agent_registration(harness, snapshot) -> module._PreparedNixlTransfer | module._PreparedBoundedTransfer:
    return harness.transfer.prepare(
         trainer_snapshot=snapshot,
    )


def _collect_agent_registration(harness, prepared) -> dict:
    if harness.streaming is not None:
        _, installed = harness.collect(prepared)
        return installed
    return install_replay(harness.transfer.stage(prepared).tensors, harness.layout)


@pytest.mark.parametrize("harness", [None, StreamingSettings(1024, "cpu")], indirect=True)
def test_next_generation_replaces_agent_registration_before_read(harness, monkeypatch) -> None:
    monkeypatch.setattr(harness.transfer, "_device", torch.device("cpu"))
    monkeypatch.setattr(module, "classic_cuda_alloc", nullcontext)
    first = _agent_registration_snapshot(harness, generation=1)
    _check_values(harness, _collect_agent_registration(harness, _prepare_agent_registration(harness, first)), check_padding=harness.streaming is not None)
    for tensor in harness.sources.values():
        tensor.add_(5)
    harness.events.clear()
    second = _agent_registration_snapshot(harness, generation=2, token=b"source:registration-v2")
    prepared = _prepare_agent_registration(harness, second)
    _check_values(harness, _collect_agent_registration(harness, prepared), check_padding=harness.streaming is not None)
    events = harness.events
    assert events.index("shutdown") < events.index("initialize") < events.index("connect:source:registration-v2") < events.index("register") < events.index("post")
    assert prepared.transport.awaited == prepared.transport.posted


@pytest.mark.parametrize("harness", [StreamingSettings(4096, "cpu", 2)], indirect=True)
@pytest.mark.parametrize("failure", ["growth_registration", "replacement_connection"])
def test_native_replacement_failure_cleans_up_and_retries(harness, monkeypatch, failure) -> None:
    first = _agent_registration_snapshot(harness, generation=1)
    _collect_agent_registration(harness, _prepare_agent_registration(harness, first))
    for tensor in harness.sources.values():
        tensor.add_(5)
    second, _rows = _larger_source_snapshot(harness) if failure == "growth_registration" else (
        _agent_registration_snapshot(harness, generation=2, token=b"source:registration-v2"), None,
    )
    harness.events.clear()

    def fail(_resource) -> None:
        raise RuntimeError("native replacement failed")

    with monkeypatch.context() as fault:
        operation = "register_dram_buffer" if failure == "growth_registration" else "add_remote_agent"
        fault.setattr(harness.transfer._manager, operation, fail)
        with pytest.raises(RuntimeError, match="native replacement failed"):
            _prepare_agent_registration(harness, second)
    assert "post" not in harness.events
    assert harness.transfer._manager.registered == {}
    prepared = _prepare_agent_registration(harness, second)
    _check_values(harness, _collect_agent_registration(harness, prepared), exact_rows=4)
    assert prepared.transport.awaited == prepared.transport.posted


@pytest.mark.parametrize("harness", [None, StreamingSettings(1024, "cpu"), StreamingSettings(2048, "cpu", 2)], indirect=True)
def test_replica_switch_reuses_matching_group_descriptors(harness, monkeypatch):
    monkeypatch.setattr(harness.transfer, "_device", torch.device("cpu"))
    monkeypatch.setattr(module, "classic_cuda_alloc", nullcontext)
    replicas = {
        worker: {name: tensor[start:start + 2].clone() + delta for name, tensor in harness.sources.items()}
        for worker, start, delta in (("a1", 0, 0), ("a2", 0, 100), ("b1", 2, 0))
    }

    def snapshot(a_worker, generation=1):
        shards = []
        for slot, worker, start in (("a", a_worker, 0), ("b", "b1", 2)):
            blob = wrap_rendezvous_blob(worker.encode(), worker, f"{worker}:19000", [
                PublishedTensor(
                    name=name, dtype="torch.float32", elsize=4, full_shape=(4, 4),
                    shards=[PublishedShard(
                        agent_name=worker, device_id=0, addr=tensor.data_ptr(),
                        shard_offset=(start, 0), shape=(2, 4),
                    )],
                ) for name, tensor in replicas[worker].items()
            ])
            shards.append(TrainerSourceShard(slot, worker, hashlib.sha256(blob).hexdigest(), f"{worker}:19000", blob))
        return TrainerSourceSnapshot(MeshIdentity("mesh", generation), tuple(shards))

    builds = []
    original = module._SourceBinding.resolve

    def resolve(self, buffer):
        session, address = original(self, buffer)
        builds.append(session)
        return session, address

    monkeypatch.setattr(module._SourceBinding, "resolve", resolve)
    selections = []
    logical_plan = None
    for index, worker in enumerate(("a1", "a2", "a1")):
        for tensors in replicas.values():
            for tensor in tensors.values():
                tensor.add_(7)
        for name in harness.sources:
            harness.sources[name] = torch.cat((replicas[worker][name], replicas["b1"][name]))
        trainer = snapshot(worker)
        builds.clear()
        prepared = harness.transfer.prepare(
            trainer_snapshot=trainer,
        )
        plan = harness.transfer._weight_update_plan
        if index == 0:
            logical_plan = plan.transfer_plan
        assert plan.transfer_plan is logical_plan
        if harness.streaming is None:
            selected = prepared.descriptors
            installed = install_replay(harness.transfer.stage(prepared).tensors, harness.layout)
        else:
            selected = tuple(read for reads in prepared.descriptors for read in reads)
            _, installed = harness.collect(prepared)
        _check_values(harness, installed, check_padding=harness.streaming is not None)
        selections.append(selected)
        if index == 1:
            assert builds and set(builds) == {"a2"}
        elif index == 2:
            assert builds == []
            assert prepared.metrics["descriptor_builds"] == 0
        if index:
            original_b = tuple(read for read in selections[0] if read.session == "b1")
            current_b = tuple(read for read in selected if read.session == "b1")
            assert original_b and all(before is after for before, after in zip(original_b, current_b, strict=True))
    assert all(before is after for before, after in zip(selections[0], selections[2], strict=True))

    # Reset must rebuild both groups even though their mesh and addresses match.
    harness.transfer.reset_workspace()
    assert harness.transfer._weight_update_plan is plan
    builds.clear()
    trainer = snapshot("a1")
    harness.transfer._capture_layout = lambda _: pytest.fail(
        "reset must preserve the logical plan"
    )
    rebuilt = harness.transfer.prepare(
         trainer_snapshot=trainer,
    )
    assert set(builds) == {"a1", "b1"}
    if harness.streaming is None:
        installed = install_replay(harness.transfer.stage(rebuilt).tensors, harness.layout)
    else:
        _, installed = harness.collect(rebuilt)
    _check_values(harness, installed, check_padding=harness.streaming is not None)
