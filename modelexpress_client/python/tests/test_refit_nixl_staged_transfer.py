# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from modelexpress.refit.reshard.rendezvous import structural_manifest_digest
import hashlib
import ctypes
from contextlib import nullcontext
from dataclasses import replace
from types import MappingProxyType

import modelexpress_rl.inference.nixl_staged_transfer as transfer_module
from modelexpress_rl.inference.plan import StreamingSettings

import pytest
import torch
from modelexpress import envs
from modelexpress import p2p_pb2
from modelexpress.refit.reshard.rendezvous import (
    PublishedShard,
    PublishedTensor,
    unwrap_rendezvous_blob,
    wrap_rendezvous_blob,
)
from modelexpress.refit.reshard.slice_plan import LogicalRead, SourceBuffer, Shard
from modelexpress.refit.reshard.transfer_plan import SourceInfo, LogicalTransferPlan, plan_transfer
from modelexpress.refit.reshard.types import (
    CaptureResult,
    IncompleteRefit,
    RecordedCopy,
)
from modelexpress.refit.reshard.verify import tensor_digest
from modelexpress_rl.inference.adapter import TrainerSourceShard
from modelexpress_rl.inference.methods import LoadTimeTensorNixlUpdateMethod
from modelexpress_rl.inference.nixl_staged_transfer import (
    _SourceBinding,
    _bounded_batches,
    _load_agent_metadata,
    _NixlStagedTransfer,
    _pack_bounded_batches,
    _PreparedBoundedTransfer,
    _PreparedNixlTransfer,
    _required_agent_metadata,
    _resolve_sources,
    _ResolvedSources,
)
from modelexpress_rl.inference.plan import TrainerSourceSnapshot, ResolvedTrainerSource
from tests.test_refit_bounded_descriptors import harness, _check_values


def _bind_plan(plan, sources) -> tuple[LogicalTransferPlan, _SourceBinding]:
    agents = sorted(
        {shard.session for source in sources.values() for shard in source.shards}
    )
    resolved = _ResolvedSources(
        sources,
        {agent: agent for agent in agents},
        {agent: 0 for agent in agents},
        {agent: b"metadata" for agent in agents},
    )
    binding = _SourceBinding(resolved, {agent: agent for agent in agents}, ())
    return plan, binding


def _prepared_bounded(batches, sources, transport, metrics=None) -> _PreparedBoundedTransfer:
    _, binding = _bind_plan(LogicalTransferPlan(), sources)
    return _PreparedBoundedTransfer(
        batches, sources, transport, binding, {} if metrics is None else metrics
    )


def test_bounded_batches_preserve_module_groups_and_count_dtype_scratch(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    sources = _resolve_sources(
        [
            _manifest(agent_name="a", endpoint="a:19000", offset=0, address=100),
            _manifest(agent_name="b", endpoint="b:19000", offset=2, address=200),
        ]
    ).sources
    copies = [
        RecordedCopy(
            src_name="weight",
            op_chain=(),
            param_name=name,
            dest_offset=0,
            dest_shape=(4,),
            dest_stride=(1,),
            dest_dtype=dtype,
        )
        for name, dtype in [
            ("layer0.weight", torch.float32),
            ("layer1.weight", torch.bfloat16),
        ]
    ]
    layout = {c.param_name: (c.dest_shape, c.dest_dtype) for c in copies}
    batches = _bounded_batches(
        CaptureResult(copies=copies), layout, sources, 512
    )
    assert [b.nbytes for b in batches] == [256, 512]
    assert [set(b.layouts[0]) for b in batches] == [
        {"layer0.weight"},
        {"layer1.weight"},
    ]
    with pytest.raises(IncompleteRefit, match="exceeds"):
        _bounded_batches(CaptureResult(copies=copies), layout, sources, 511)
    with pytest.raises(IncompleteRefit, match="cover every"):
        _bounded_batches(CaptureResult(copies=copies[:1]), layout, sources, 512)


def test_packing_coalesces_modules_without_changing_planned_reads(monkeypatch):
    """Packing may only change how many arena residencies a refit needs. The
    copies, planned bytes, and descriptor count must match the unpacked plan."""
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    sources = _resolve_sources(
        [
            _manifest(agent_name="a", endpoint="a:19000", offset=0, address=100),
            _manifest(agent_name="b", endpoint="b:19000", offset=2, address=200),
        ]
    ).sources
    copies = [
        RecordedCopy(
            src_name="weight",
            op_chain=(),
            param_name=name,
            dest_offset=0,
            dest_shape=(4,),
            dest_stride=(1,),
            dest_dtype=dtype,
        )
        for name, dtype in [
            ("layer0.weight", torch.float32),
            ("layer1.weight", torch.bfloat16),
        ]
    ]
    layout = {c.param_name: (c.dest_shape, c.dest_dtype) for c in copies}
    batches = _bounded_batches(
        CaptureResult(copies=copies), layout, sources, 512
    )
    assert [b.nbytes for b in batches] == [256, 512]

    packed = _pack_bounded_batches(batches, 768)
    assert len(packed) == 1 and packed[0].nbytes == 768
    assert list(packed[0].layouts.recv) == ["layer0.weight", "layer1.weight"]
    assert packed[0].capture.copies == copies
    assert packed[0].transfer_plan.bytes_planned() == sum(
        b.transfer_plan.bytes_planned() for b in batches
    )
    assert packed[0].transfer_plan.descriptor_count() == sum(
        b.transfer_plan.descriptor_count() for b in batches
    )

    # A budget that only fits one module leaves the owning-module batching intact.
    assert len(_pack_bounded_batches(batches, 512)) == 2

    # Two modules pulling the same complete source cannot share one staging slot.
    conflicting = [
        replace(
            batch,
            layouts=batch.layouts._replace(full={"shared": ((4,), torch.float32)}),
            nbytes=batch.nbytes + 256,
        )
        for batch in batches
    ]
    assert len(_pack_bounded_batches(conflicting, 2048)) == 2



@pytest.mark.parametrize("pack", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_bounded_transfer_reuses_arena_and_preserves_fp32(monkeypatch, pack, padded):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    source_tensor = torch.tensor([1.001, 2.002, 3.003, 4.004])
    source = SourceInfo(
        global_shape=(4,),
        dtype=torch.float32,
        elsize=4,
        shards=[Shard((0,), (4,), "source", source_tensor.data_ptr(), 4)],
    )
    copies = [
        RecordedCopy(
            src_name="w",
            op_chain=(),
            param_name=name,
            dest_offset=2 if padded else 0,
            dest_shape=(4,),
            dest_stride=(1,),
            dest_dtype=dtype,
        )
        for name, dtype in [("a.weight", torch.float32), ("b.weight", torch.bfloat16)]
    ]
    layout = {
        c.param_name: ((8,) if padded else c.dest_shape, c.dest_dtype) for c in copies
    }
    # A 512-byte budget fits either module separately; 768 bytes lets both
    # share one residency.
    arena_bytes = 768 if pack else 512
    batches = _bounded_batches(
        CaptureResult(copies=copies), layout, {"w": source}, arena_bytes
    )
    if pack:
        batches = _pack_bounded_batches(batches, arena_bytes)

    class Transport:
        def read(self, descriptors):
            for d in descriptors:
                ctypes.memmove(d.dst_addr, d.src_addr, d.nbytes)

        def post_reads(self, descriptors):
            self.read(descriptors)
            return []

        def await_reads(self, posted):
            assert posted == []

    prepared = _prepared_bounded(batches, {"w": source}, Transport())
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._descriptor_cache = None
    transfer._closed = False
    transfer._active = prepared
    transfer._device = torch.device("cpu")
    transfer._device_id = 0
    transfer._bounded_arena = torch.empty(arena_bytes, dtype=torch.uint8)
    transfer._bounded_arena.fill_(255)
    metrics = {}
    installed = {}
    addresses = []
    for tensors in transfer.iter_bounded(prepared, metrics):
        addresses.append(next(iter(tensors.values())).data_ptr())
        installed.update({name: value.clone() for name, value in tensors.items()})
    assert set(addresses) == {transfer._bounded_arena.data_ptr()}
    for name, (_, dtype) in layout.items():
        expected = source_tensor.to(dtype)
        if padded:
            expected = torch.zeros(8, dtype=dtype)
            expected[2:6].copy_(source_tensor)
        assert torch.equal(installed[name], expected)
    assert metrics["staging_peak_bytes"] == arena_bytes
    assert metrics["bytes_received"] == 32
    assert metrics["batches"] == (1 if pack else 2)


def _manifest(
    *, agent_name: str, endpoint: str, offset: int, address: int, memory_type="VRAM"
) -> bytes:
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
                        memory_type=memory_type,
                    )
                ],
            )
        ],
    )


@pytest.mark.parametrize(
    "failure", [None, "registration", "cleanup", "metadata", "capture", "malformed", "post", "await", "replica"]
)
@pytest.mark.parametrize("harness", [None, StreamingSettings(1024)], indirect=True)
@pytest.mark.parametrize("diagnostic", [None, "plan"])
def test_fixed_mode_updates_receive_changed_values(harness, monkeypatch, failure, diagnostic):
    transfer = harness.transfer
    manager = transfer._manager
    monkeypatch.setattr(transfer, "_device", torch.device("cpu"))
    monkeypatch.setattr(transfer_module, "classic_cuda_alloc", nullcontext)
    capture_failure = False

    def capture_layout(schema):
        if capture_failure:
            raise RuntimeError("layout capture failed")
        harness.captures.append(schema)
        return harness.capture, harness.layout

    def snapshot(agent="source"):
        blob = wrap_rendezvous_blob(
            agent.encode(), agent, f"{agent}:19000",
            [PublishedTensor(
                name=name, dtype="torch.float32", elsize=4, full_shape=(4, 4),
                shards=[PublishedShard(agent_name=agent, device_id=0, addr=value.data_ptr(),
                                       shard_offset=(0, 0), shape=(4, 4))],
            ) for name, value in harness.sources.items()],
        )
        return TrainerSourceSnapshot("mesh", 1, (TrainerSourceShard(
            "slot", agent, hashlib.sha256(blob).hexdigest(), f"{agent}:19000", blob, structural_manifest_digest(blob),
        ),))

    source = snapshot()
    transfer._capture_layout = capture_layout
    method = LoadTimeTensorNixlUpdateMethod(transfer=transfer)

    def update(source):
        prepared = method.prepare(version=None, source=ResolvedTrainerSource(source))
        try:
            if harness.streaming:
                installed = {name: value.clone() for batch in prepared.batches()
                             for name, value in batch.items()}
            else:
                installed = prepared.staged.tensors
            _check_values(harness, installed, check_padding=harness.streaming is not None)
            with pytest.raises(RuntimeError, match="release"):
                method.prepare(version=None, source=ResolvedTrainerSource(source))
        finally:
            method.release(prepared)

    try:
        if failure in ("registration", "cleanup"):
            register = manager.register_tensors
            cleanup_error = RuntimeError("shutdown failed")

            def fail_register(tensors):
                register(tensors)
                raise RuntimeError("registration failed")

            def fail_cleanup():
                raise cleanup_error

            with monkeypatch.context() as fault:
                fault.setattr(manager, "register_tensors", fail_register)
                if failure == "cleanup":
                    fault.setattr(manager, "shutdown", fail_cleanup)
                error = ValueError if failure == "cleanup" else RuntimeError
                with pytest.raises(error) as raised:
                    update(source)
                if failure == "cleanup":
                    assert raised.value.__cause__ is cleanup_error
                    assert manager.registered
                    return
                assert not manager.registered
        update(source)
        setup_calls = [harness.events.count(event) for event in ("register", "connect:source")]
        if failure in ("post", "await"):
            operation = "post_reads" if failure == "post" else "await_reads"

            def fail_read(*_args):
                raise RuntimeError("read failed")

            with monkeypatch.context() as fault:
                fault.setattr(type(harness.transports[-1]), operation, fail_read)
                with pytest.raises(RuntimeError, match="read failed"):
                    update(source)
        if failure == "replica":
            harness.sources.update({name: value.clone() + 10 for name, value in harness.sources.items()})
            source = snapshot("replica")
        if failure in ("metadata", "capture", "malformed"):
            blob = source.shards[0].manifest
            generation = 2
            if failure == "malformed":
                blob = b"malformed"
            else:
                payload = unwrap_rendezvous_blob(blob)
                capture_failure = failure == "capture"
                generation = 2
                blob = wrap_rendezvous_blob(
                    b"changed" if failure == "metadata" else payload.agent_metadata,
                    payload.agent_name, payload.metadata_endpoint,
                    [replace(t, name="renamed." + t.name) for t in payload.tensors]
                    if capture_failure else payload.tensors,
                )
            bad = replace(source, mesh_generation=generation, shards=(replace(
                source.shards[0], worker_id="replacement" if failure == "metadata" else source.shards[0].worker_id,
                manifest=blob, manifest_digest=hashlib.sha256(blob).hexdigest(),
                structural_digest=structural_manifest_digest(blob) if failure != "malformed" else "invalid",
            ),))
            reads = sum(len(t.posts) for t in harness.transports)
            registrations = {key: value.data_ptr() for key, value in manager.registered.items()}
            error = ValueError if failure == "malformed" else RuntimeError
            with pytest.raises(error):
                update(bad)
            capture_failure = False
            assert sum(len(t.posts) for t in harness.transports) == reads
            if failure == "metadata":
                assert not manager.registered
            else:
                assert {
                    key: value.data_ptr() for key, value in manager.registered.items()
                } == registrations
        for _ in range(2):
            for value in harness.sources.values():
                value.add_(1)
            update(source)
        if failure is None:
            assert len(harness.captures) == len(harness.transports) == 1
            assert [harness.events.count(event) for event in ("register", "connect:source")] == setup_calls
    finally:
        method.close()
        assert not manager.registered


def test_default_transfer_timeout_matches_the_lease_budget(monkeypatch):
    monkeypatch.setenv("MX_TRANSFER_TIMEOUT", "17")

    transfer = _NixlStagedTransfer(
        generator_fingerprint=lambda: (),
        capture_layout=lambda _: None,
        device_id=0,
        device=torch.device("cpu"),
        manager=object(),
    )

    assert transfer._timeout == 17.0


@pytest.mark.parametrize("memory_type", ["VRAM", "DRAM"])
def test_exact_manifests_resolve_without_legacy_source_discovery(memory_type):
    resolved = _resolve_sources(
        [
            _manifest(
                agent_name="trainer-0",
                endpoint="trainer-0:19000",
                offset=0,
                address=100,
                memory_type=memory_type,
            ),
            _manifest(
                agent_name="trainer-1",
                endpoint="trainer-1:19001",
                offset=2,
                address=200,
            ),
        ]
    )

    assert resolved.sources["weight"].global_shape == (4,)
    assert [shard.addr for shard in resolved.sources["weight"].shards] == [100, 200]
    assert resolved.session_to_agent == {
        "trainer-0": "trainer-0",
        "trainer-1": "trainer-1",
    }
    assert resolved.agent_metadata == {
        "trainer-0": b"nixl-metadata",
        "trainer-1": b"nixl-metadata",
    }
    assert resolved.session_to_memory == {
        "trainer-0": memory_type,
        "trainer-1": "VRAM",
    }


def test_required_agent_metadata_rejects_incomplete_source_metadata():
    resolved = _ResolvedSources(
        sources={},
        session_to_agent={"session-a": "agent-a"},
        session_to_device={},
        agent_metadata={"agent-a": b"metadata"},
    )
    binding = _SourceBinding(
        resolved, {"slot-a": "agent-a"}, (("slot-a", "agent-a", "manifest-a"),)
    )
    plan = LogicalTransferPlan(
        source_buffers=(SourceBuffer("slot-a", "weight", (0,), (1,)),),
        segments=(LogicalRead(0, 0, "weight", 0, 4),)
    )
    assert _required_agent_metadata(plan, resolved, binding) == {
        "agent-a": b"metadata"
    }

    empty = _ResolvedSources({}, {}, {}, {})
    with pytest.raises(RuntimeError, match="unknown source slots"):
        _required_agent_metadata(
            LogicalTransferPlan(
                source_buffers=(SourceBuffer("missing-slot", "weight", (0,), (1,)),),
                segments=(LogicalRead(0, 0, "weight", 0, 4),)
            ),
            resolved,
            binding,
        )

    missing_metadata = _ResolvedSources({}, {}, {}, {})
    with pytest.raises(RuntimeError, match="without NIXL metadata"):
        _required_agent_metadata(
            plan,
            missing_metadata,
            _SourceBinding(missing_metadata, {"slot-a": "agent-a"}, ()),
        )


def test_load_agent_metadata_validates_embedded_agent_identity():
    calls = []

    class _Manager:
        def add_remote_agent(self, metadata):
            calls.append(metadata)
            return b"agent-a"

    _load_agent_metadata(_Manager(), {"agent-a": b"metadata"})
    assert calls == [b"metadata"]

    with pytest.raises(RuntimeError, match="does not match its manifest"):
        _load_agent_metadata(_Manager(), {"agent-b": b"metadata"})


def test_transformed_source_is_fully_reconstructed_for_verification(monkeypatch):
    # Full reconstruction + verification is the digest mode; default is minimal reads.
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    source = SourceInfo(
        global_shape=(4, 4),
        dtype=torch.float32,
        elsize=4,
        shards=[
            Shard((0, 0), (4, 2), "left", 0, 4),
            Shard((0, 2), (4, 2), "right", 32, 4),
        ],
    )
    copy = RecordedCopy(
        src_name="weight",
        op_chain=(("narrow", (1, 0, 2), ()),),
        param_name="fused_weight",
        dest_offset=0,
        dest_shape=(4, 2),
        dest_stride=(2, 1),
        dest_dtype=torch.float32,
    )

    plan = plan_transfer(CaptureResult(copies=[copy]), {"weight": source}, verify_full_sources=envs.MX_RESHARD_PUBLISH_DIGEST)

    assert not plan.segments
    assert len(plan.full_pulls) == 1
    assert sum(segment.nbytes for segment in plan.full_pulls[0].segments) == 64
    assert {plan.source_buffers[segment.source_buffer].source_slot_id for segment in plan.full_pulls[0].segments} == {
        "left",
        "right",
    }


def test_transformed_source_reads_only_required_slice_by_default(monkeypatch):
    monkeypatch.delenv("MX_RESHARD_PUBLISH_DIGEST", raising=False)
    source = SourceInfo(
        global_shape=(4, 4),
        dtype=torch.float32,
        elsize=4,
        shards=[
            Shard((0, 0), (4, 2), "left", 0, 4),
            Shard((0, 2), (4, 2), "right", 32, 4),
        ],
    )
    copy = RecordedCopy(
        src_name="weight",
        op_chain=(("narrow", (1, 0, 2), ()),),
        param_name="fused_weight",
        dest_offset=0,
        dest_shape=(4, 2),
        dest_stride=(2, 1),
        dest_dtype=torch.float32,
    )

    plan = plan_transfer(CaptureResult(copies=[copy]), {"weight": source}, verify_full_sources=envs.MX_RESHARD_PUBLISH_DIGEST)

    assert not plan.full_pulls
    assert sum(segment.nbytes for segment in plan.segments) == 32
    assert {plan.source_buffers[segment.source_buffer].source_slot_id for segment in plan.segments} == {"left"}


def _prepared(tensor: torch.Tensor, digest: str | None) -> _PreparedNixlTransfer:
    copy = RecordedCopy(
        src_name="weight",
        op_chain=(),
        param_name="weight",
        dest_offset=0,
        dest_shape=tuple(tensor.shape),
        dest_stride=tuple(tensor.stride()),
        dest_dtype=tensor.dtype,
    )
    source = SourceInfo(
        global_shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        elsize=tensor.element_size(),
        shards=[
            Shard(
                shard_offset=(0,),
                shape=tuple(tensor.shape),
                session="trainer",
                addr=0,
                elsize=tensor.element_size(),
                digest=None,
            )
        ],
    )
    plan, source_binding = _bind_plan(LogicalTransferPlan(), {"weight": source})
    return _PreparedNixlTransfer(
        transfer_plan=plan,
        capture=CaptureResult(copies=[copy]),
        sources={"weight": source},
        descriptors=(),
        transport=object(),
        source_binding=source_binding,
        checksums=MappingProxyType(
            {("weight", "trainer", 0, (0,), tuple(tensor.shape)): digest}
            if digest is not None
            else {}
        ),
    )


def test_staged_verification_rejects_missing_or_mismatched_digest():
    tensor = torch.arange(64, dtype=torch.int32)
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._descriptor_cache = None
    transfer._recv_buffers = {"weight": tensor}
    transfer._convert_buffers = {}
    transfer._full_buffers = {}

    transfer._verify(_prepared(tensor, tensor_digest(tensor)))
    with pytest.raises(RuntimeError, match="digest mismatch"):
        transfer._verify(_prepared(tensor, tensor_digest(tensor + 1)))
    with pytest.raises(RuntimeError, match="did not publish"):
        transfer._verify(_prepared(tensor, None))


def test_full_tensor_plan_fails_before_transfer_when_capture_has_holes():
    capture = CaptureResult(copies=[])
    with pytest.raises(IncompleteRefit, match="must cover every engine parameter"):
        _NixlStagedTransfer._validate_complete(
            capture,
            {"weight": ((4,), torch.float32)},
            LogicalTransferPlan(),
        )


def test_transfer_manager_is_closed_after_failed_init_and_only_once(monkeypatch):
    calls = []

    class _Manager:
        def __init__(self, **_kwargs):
            pass

        def initialize(self):
            calls.append("initialize")
            raise RuntimeError("init failed")

        def shutdown(self):
            calls.append("shutdown")

    monkeypatch.setattr(transfer_module, "NixlTransferManager", _Manager)
    with pytest.raises(RuntimeError, match="init failed"):
        _NixlStagedTransfer(
            generator_fingerprint=lambda: (),
            capture_layout=lambda _: None,
            agent_name="generator",
            device_id=0,
            device=torch.device("cpu"),
            listen_port=19000,
        )
    assert calls == ["initialize", "shutdown"]

    transfer = object.__new__(_NixlStagedTransfer)
    transfer._descriptor_cache = None
    transfer._manager = _Manager()
    transfer._owns_manager = True
    transfer._closed = False
    transfer._recv_buffers = {}
    transfer._convert_buffers = {}
    transfer._full_buffers = {}
    transfer._staging_arenas = []
    transfer._staging_registrations = []
    transfer.close()
    transfer.close()
    assert calls == ["initialize", "shutdown", "shutdown"]


def test_borrowed_manager_is_not_initialized_or_closed():
    class _Manager:
        def initialize(self):
            raise AssertionError("borrowed manager must already be initialized")

        def shutdown(self):
            raise AssertionError("borrowed manager is owned by the loader")

    transfer = _NixlStagedTransfer(
        generator_fingerprint=lambda: (),
        capture_layout=lambda _: None,
        device_id=0,
        device=torch.device("cpu"),
        manager=_Manager(),
    )
    transfer.close()


def test_borrowed_manager_survives_peer_receive_and_refuses_a_reset(monkeypatch):
    """The peer path runs on a manager the transfer does not own, so it must
    never initialize, cycle, or shut down the loader's agent."""

    class _Manager:
        def initialize(self):
            raise AssertionError("borrowed manager must already be initialized")

        def shutdown(self):
            raise AssertionError("borrowed manager is owned by the loader")

        def add_remote_agent(self, metadata):
            return "peer-agent"

        def fetch_remote_and_wait(self, **kwargs):
            pass

        def receive_from_source(self, **kwargs):
            kwargs["on_transfer_start"]()
            return 16, 1, 0.25

        def remove_remote_agent(self, agent_name):
            pass

    manifest = p2p_pb2.GetTensorManifestResponse(
        mx_source_id="source-1",
        worker_id="worker-1",
        metadata_endpoint="127.0.0.1:17000",
        agent_name="peer-agent",
        tensors=[
            p2p_pb2.TensorDescriptor(
                name="weight", addr=1234, size=16, device_id=0, dtype="torch.float32"
            )
        ],
    )

    class _Lease:
        def __init__(self):
            self.manifest = manifest

        def close(self):
            pass

    monkeypatch.setattr(
        transfer_module, "prepare_tensor_read", lambda *a, **k: (_Lease(), 0)
    )
    transfer = _NixlStagedTransfer(
        generator_fingerprint=lambda: (),
        capture_layout=lambda _: None,
        device_id=0,
        device=torch.device("cpu"),
        manager=_Manager(),
    )

    source = p2p_pb2.WorkerMetadata(worker_grpc_endpoint="127.0.0.1:18000")
    live = {"weight": torch.empty(4, dtype=torch.float32)}
    for _ in range(2):
        lease = transfer.prepare_peer_read(
            source=source,
            mx_source_id="source-1",
            worker_id="worker-1",
            destination_tensors=live,
        )
        metrics = transfer.receive_peer(
            tensor_read=lease,
            destination_tensors=live,
            on_transfer_start=lambda: None,
        )
        assert metrics["bytes_received"] == 16

    with pytest.raises(RuntimeError, match="transfer-owned NIXL agent"):
        transfer.reset_workspace()
    transfer.close()


def test_peer_receive_writes_directly_into_live_tensor_catalog(monkeypatch):
    calls = []

    class _Lease:
        def __init__(self, manifest):
            self.manifest = manifest
            self.closed = False

        def close(self):
            self.closed = True
            calls.append(("lease_release", None))

    manifest = p2p_pb2.GetTensorManifestResponse(
        mx_source_id="source-1",
        worker_id="worker-1",
        metadata_endpoint="127.0.0.1:17000",
        agent_name="live-peer-agent",
        tensors=[
            p2p_pb2.TensorDescriptor(
                name="weight",
                addr=1234,
                size=16,
                device_id=0,
                dtype="torch.float32",
            )
        ],
    )

    def prepare(*args, **kwargs):
        calls.append(("prepare", (args, kwargs)))
        return _Lease(manifest), manifest.ByteSize()

    monkeypatch.setattr(transfer_module, "prepare_tensor_read", prepare)

    class _Manager:
        def add_remote_agent(self, metadata):
            calls.append(("add", metadata))
            return "peer-agent"

        def fetch_remote_and_wait(self, **kwargs):
            calls.append(("fetch", kwargs))

        def receive_from_source(self, **kwargs):
            calls.append(("receive", kwargs))
            kwargs["on_transfer_start"]()
            return 16, 1, 0.25

        def remove_remote_agent(self, agent_name):
            calls.append(("remove", agent_name))

    transfer = object.__new__(_NixlStagedTransfer)
    transfer._descriptor_cache = None
    transfer._device = torch.device("cpu")
    transfer._device_id = 0
    transfer._timeout = 30.0
    transfer._manager = _Manager()
    transfer._closed = False
    source = p2p_pb2.WorkerMetadata(
        worker_grpc_endpoint="127.0.0.1:18000",
    )

    live = {"weight": torch.empty(4, dtype=torch.float32)}
    lease = transfer.prepare_peer_read(
        source=source,
        mx_source_id="source-1",
        worker_id="worker-1",
        destination_tensors=live,
    )
    assert lease.closed is False
    metrics = transfer.receive_peer(
        tensor_read=lease,
        destination_tensors=live,
        on_transfer_start=lambda: calls.append(("transfer_start", None)),
    )

    assert metrics["bytes_received"] == 16
    receive = next(value for name, value in calls if name == "receive")
    assert receive["remote_agent_name"] == "live-peer-agent"
    assert receive["require_exact_match"] is True
    assert receive["destination_tensors"] is live
    assert callable(receive["on_transfer_start"])
    assert calls.count(("transfer_start", None)) == 1
    assert lease.closed is False
    fetch = next(value for name, value in calls if name == "fetch")
    assert fetch == {
        "remote_agent_name": "live-peer-agent",
        "ip": "127.0.0.1",
        "port": 17000,
        "timeout_seconds": 30.0,
    }

    manifest.metadata_endpoint = ""
    with pytest.raises(RuntimeError, match="unusable metadata endpoint"):
        transfer.receive_peer(
            tensor_read=lease,
            destination_tensors=live,
            on_transfer_start=lambda: None,
        )


def test_registered_workspace_is_reused_only_for_the_same_layout(monkeypatch):
    monkeypatch.setattr(transfer_module, "classic_cuda_alloc", nullcontext)
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._descriptor_cache = None
    transfer._device = torch.device("cpu")
    buffers = {}
    layout = {"weight": ((4,), torch.float32)}

    transfer._ensure_buffers(buffers, layout, label="receive-buffer")
    pointer = buffers["weight"].data_ptr()
    transfer._ensure_buffers(buffers, layout, label="receive-buffer")
    assert buffers["weight"].data_ptr() == pointer

    with pytest.raises(RuntimeError, match="layout changed"):
        transfer._ensure_buffers(
            buffers,
            {"weight": ((8,), torch.float32)},
            label="receive-buffer",
        )


def test_double_buffered_iteration_alternates_arenas_and_prefetches(monkeypatch):
    """Batch i+1 is posted before batch i is handed to the caller."""
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    source_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    source = SourceInfo(
        global_shape=(4,),
        dtype=torch.float32,
        elsize=4,
        shards=[Shard((0,), (4,), "source", source_tensor.data_ptr(), 4)],
    )
    names = ["a.weight", "b.weight", "c.weight"]
    copies = [
        RecordedCopy(
            src_name="w",
            op_chain=(),
            param_name=name,
            dest_offset=0,
            dest_shape=(4,),
            dest_stride=(1,),
            dest_dtype=torch.float32,
        )
        for name in names
    ]
    layout = {c.param_name: (c.dest_shape, c.dest_dtype) for c in copies}
    batches = _bounded_batches(
        CaptureResult(copies=copies), layout, {"w": source}, 256
    )
    assert len(batches) == 3
    events = []

    class Transport:
        def post_reads(self, descriptors):
            events.append("post")
            for d in descriptors:
                ctypes.memmove(d.dst_addr, d.src_addr, d.nbytes)
            return ["posted"]

        def await_reads(self, posted):
            assert posted == ["posted"]
            events.append("await")

    prepared = _prepared_bounded(batches, {"w": source}, Transport())
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._descriptor_cache = None
    transfer._closed = False
    transfer._active = prepared
    transfer._device = torch.device("cpu")
    transfer._device_id = 0
    arenas = [torch.empty(256, dtype=torch.uint8) for _ in range(2)]
    transfer._staging_arenas = arenas
    transfer._staging_registrations = []
    metrics = {}
    addresses = []
    for tensors in transfer.iter_bounded(prepared, metrics):
        events.append("commit")
        (tensor,) = tensors.values()
        addresses.append(tensor.data_ptr())
        assert torch.equal(tensor, source_tensor)
    # Arena use alternates, and the next READ is posted before each commit.
    assert addresses == [
        arenas[0].data_ptr(),
        arenas[1].data_ptr(),
        arenas[0].data_ptr(),
    ]
    assert events == [
        "post",
        "await",
        "post",
        "commit",
        "await",
        "post",
        "commit",
        "await",
        "commit",
    ]
    assert metrics["staging_buffers"] == 2
    assert metrics["staging_peak_bytes"] == 512
    assert metrics["batches"] == 3
    assert transfer._active is prepared


def test_abandoned_double_buffered_iteration_drains_the_prefetched_read(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    source_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    source = SourceInfo(
        global_shape=(4,),
        dtype=torch.float32,
        elsize=4,
        shards=[Shard((0,), (4,), "source", source_tensor.data_ptr(), 4)],
    )
    copies = [
        RecordedCopy(
            src_name="w",
            op_chain=(),
            param_name=name,
            dest_offset=0,
            dest_shape=(4,),
            dest_stride=(1,),
            dest_dtype=torch.float32,
        )
        for name in ["a.weight", "b.weight"]
    ]
    layout = {c.param_name: (c.dest_shape, c.dest_dtype) for c in copies}
    batches = _bounded_batches(
        CaptureResult(copies=copies), layout, {"w": source}, 256
    )
    posted, awaited = [], []

    class Transport:
        def post_reads(self, descriptors):
            for d in descriptors:
                ctypes.memmove(d.dst_addr, d.src_addr, d.nbytes)
            handle = object()
            posted.append(handle)
            return [handle]

        def await_reads(self, handles):
            awaited.extend(handles)

    prepared = _prepared_bounded(batches, {"w": source}, Transport())
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._descriptor_cache = None
    transfer._closed = False
    transfer._active = prepared
    transfer._device = torch.device("cpu")
    transfer._device_id = 0
    transfer._staging_arenas = [torch.empty(256, dtype=torch.uint8) for _ in range(2)]
    transfer._staging_registrations = []
    iterator = transfer.iter_bounded(prepared, {})
    next(iterator)
    iterator.close()  # caller failed mid-install; the prefetched READ must not leak
    assert len(posted) == 2
    assert awaited == posted
    assert transfer._active is prepared


@pytest.mark.parametrize("budget_fits", [False, True])
@pytest.mark.parametrize("staging_buffers", [1, 2])
@pytest.mark.parametrize("source_memory_type", ["VRAM", "DRAM"])
def test_prepare_stages_in_pinned_host_memory_and_splits_the_budget(
    monkeypatch, staging_buffers, source_memory_type, budget_fits
) -> None:
    """staging_device='cpu' registers DRAM arenas and reads with a DRAM local type."""
    events = []
    source_tensor = torch.arange(4, dtype=torch.float32)
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class Manager:
        def __init__(self, **kwargs):
            self.ready = False
            self.registered = {}
            self.dram = []

        def initialize(self):
            self.ready = True

        def shutdown(self):
            events.append("shutdown")
            self.ready = False

        def register_tensors(self, tensors):
            raise AssertionError("host staging must not register VRAM arenas")

        def register_dram_buffer(self, buffer):
            assert buffer.device.type == "cpu" and buffer.dtype == torch.uint8
            events.append("register_dram")
            handle = object()
            self.dram.append(handle)
            self.registered[handle] = buffer
            return handle

        def deregister_memory(self, registered):
            events.append("deregister")
            del self.registered[registered]

        def add_remote_agent(self, metadata):
            return metadata.decode()

    transports = []

    class Transport:
        def __init__(self, manager, *args, **kwargs):
            self.manager = manager
            self.local_mem_type = kwargs.get("local_mem_type")
            assert kwargs["session_to_memory"] == {"source": source_memory_type}
            transports.append(self)

        def post_reads(self, descriptors):
            assert self.local_mem_type == "DRAM"
            for d in descriptors:
                assert any(
                    t.data_ptr() <= d.dst_addr
                    and d.dst_addr + d.nbytes <= t.data_ptr() + t.numel()
                    for t in self.manager.registered.values()
                )
                ctypes.memmove(d.dst_addr, d.src_addr, d.nbytes)
            return []

        def await_reads(self, posted):
            assert posted == []

    monkeypatch.setattr(transfer_module, "NixlTransferManager", Manager)
    monkeypatch.setattr(transfer_module, "NixlReshardTransport", Transport)
    manifest = wrap_rendezvous_blob(
        b"source",
        "source",
        "source:19000",
        [
            PublishedTensor(
                name="weight",
                dtype="torch.float32",
                elsize=4,
                full_shape=(4,),
                shards=[
                    PublishedShard(
                        agent_name="source",
                        device_id=0,
                        addr=source_tensor.data_ptr(),
                        shard_offset=(0,),
                        shape=(4,),
                        memory_type=source_memory_type,
                    )
                ],
            ),
        ],
    )
    capture = CaptureResult(
        copies=[
            RecordedCopy(
                src_name="weight",
                op_chain=(),
                param_name="layer.weight",
                dest_offset=0,
                dest_shape=(4,),
                dest_stride=(1,),
                dest_dtype=torch.float32,
            )
        ]
    )
    transfer = _NixlStagedTransfer(
        generator_fingerprint=lambda: (),
        agent_name="receiver",
        device_id=0,
        device=torch.device("cuda:0"),
        listen_port=None,
        streaming=StreamingSettings(
            256 * staging_buffers - int(not budget_fits), "cpu", staging_buffers
        ),
        capture_layout=lambda m: (capture, layout),
    )
    layout = {"layer.weight": ((4,), torch.float32)}
    snapshot = TrainerSourceSnapshot(
        "mesh",
        1,
        (TrainerSourceShard(
            "slot", "worker", hashlib.sha256(manifest).hexdigest(),
            "source:19000", manifest, structural_manifest_digest(manifest),
        ),),
    )
    try:
        if not budget_fits:
            with pytest.raises(IncompleteRefit, match="255"):
                transfer.prepare(
                    trainer_snapshot=snapshot,
                    manifests=[manifest],
                )
            assert "register_dram" not in events
            return
        prepared = transfer.prepare(
            trainer_snapshot=snapshot,
            manifests=[manifest],
        )
        assert events.count("register_dram") == staging_buffers
        assert len(transfer._staging_arenas) == staging_buffers
        assert all(a.device.type == "cpu" for a in transfer._staging_arenas)
        metrics = {}
        for tensors in transfer.iter_bounded(prepared, metrics):
            assert tensors["layer.weight"].device.type == "cpu"
            assert torch.equal(tensors["layer.weight"], source_tensor)
        assert metrics["staging_buffers"] == staging_buffers
        assert metrics["staging_peak_bytes"] == 256 * staging_buffers
    finally:
        transfer.close()
    assert not transfer._manager.registered


def test_bounded_device_validation_precedes_manager_creation(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        transfer_module, "NixlTransferManager", lambda **kwargs: events.append(kwargs)
    )
    with pytest.raises(ValueError, match="CUDA device"):
        _NixlStagedTransfer(
            generator_fingerprint=lambda: (),
            capture_layout=lambda _: None,
            agent_name="receiver",
            device_id=0,
            device=torch.device("cpu"),
            streaming=StreamingSettings(256),
        )
    assert events == []


def test_failed_prefetch_drain_is_reported_not_swallowed(monkeypatch):
    """An undrained prefetch leaves the arena writable, so it cannot pass quietly.

    With two arenas a READ for the next batch is already in flight when the
    caller stops consuming. Until that READ is drained the arena may still
    receive RDMA writes, so a drain failure is a hard condition rather than a
    cleanup nuisance. Abandoning the iterator is deliberate rather than a
    failure, so there is nothing to mask and the drain error must surface.
    """
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    source_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    source = SourceInfo(
        global_shape=(4,),
        dtype=torch.float32,
        elsize=4,
        shards=[Shard((0,), (4,), "source", source_tensor.data_ptr(), 4)],
    )
    copies = [
        RecordedCopy(
            src_name="w",
            op_chain=(),
            param_name=name,
            dest_offset=0,
            dest_shape=(4,),
            dest_stride=(1,),
            dest_dtype=torch.float32,
        )
        for name in ("a.weight", "b.weight")
    ]
    layout = {c.param_name: (c.dest_shape, c.dest_dtype) for c in copies}
    batches = _bounded_batches(
        CaptureResult(copies=copies), layout, {"w": source}, 512
    )
    assert len(batches) == 2

    class Transport:
        def post_reads(self, descriptors):
            return descriptors

        def await_reads(self, posted):
            if drained:
                raise RuntimeError("injected prefetch drain failure")
            drained.append(True)
            for d in posted:
                ctypes.memmove(d.dst_addr, d.src_addr, d.nbytes)

    drained = []
    prepared = _prepared_bounded(batches, {"w": source}, Transport())
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._descriptor_cache = None
    transfer._closed = False
    transfer._active = prepared
    transfer._device = torch.device("cpu")
    transfer._device_id = 0
    # Two arenas, so batch 1's READ is posted before batch 0 is yielded.
    transfer._staging_arenas = [torch.empty(512, dtype=torch.uint8) for _ in range(2)]

    iterator = transfer.iter_bounded(prepared, {})
    next(iterator)
    with pytest.raises(RuntimeError, match="could not be drained"):
        iterator.close()


def test_failed_prefetch_drain_does_not_mask_a_caller_error(monkeypatch):
    """A drain failure must not replace the error that caused the abandonment."""
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    source_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    source = SourceInfo(
        global_shape=(4,),
        dtype=torch.float32,
        elsize=4,
        shards=[Shard((0,), (4,), "source", source_tensor.data_ptr(), 4)],
    )
    copies = [
        RecordedCopy(
            src_name="w",
            op_chain=(),
            param_name=name,
            dest_offset=0,
            dest_shape=(4,),
            dest_stride=(1,),
            dest_dtype=torch.float32,
        )
        for name in ("a.weight", "b.weight")
    ]
    layout = {c.param_name: (c.dest_shape, c.dest_dtype) for c in copies}
    batches = _bounded_batches(
        CaptureResult(copies=copies), layout, {"w": source}, 512
    )
    drained = []

    class Transport:
        def post_reads(self, descriptors):
            return descriptors

        def await_reads(self, posted):
            if drained:
                raise RuntimeError("injected prefetch drain failure")
            drained.append(True)
            for d in posted:
                ctypes.memmove(d.dst_addr, d.src_addr, d.nbytes)

    prepared = _prepared_bounded(batches, {"w": source}, Transport())
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._descriptor_cache = None
    transfer._closed = False
    transfer._active = prepared
    transfer._device = torch.device("cpu")
    transfer._device_id = 0
    transfer._staging_arenas = [torch.empty(512, dtype=torch.uint8) for _ in range(2)]

    iterator = transfer.iter_bounded(prepared, {})
    next(iterator)
    with pytest.raises(RuntimeError, match="the install failed"):
        iterator.throw(RuntimeError("the install failed"))


def test_bounded_batch_layouts_are_named_however_they_are_built():
    """Readers use both `.full` and `[0]`, so construction style must not matter.

    The annotation alone does not enforce this: a plain 3-tuple or a
    dataclasses.replace satisfies positional access and breaks attribute
    access, which fails at only one of the two call sites.
    """
    recv = {"a.weight": ((4,), torch.float32)}
    convert: dict = {}
    full = {"w": ((4,), torch.float32)}

    built = transfer_module._StreamingBatch(
        CaptureResult(copies=[]), LogicalTransferPlan(), (recv, convert, full), 256
    )
    assert isinstance(built.layouts, transfer_module._StagingLayouts)
    assert built.layouts.recv is built.layouts[0] is recv
    assert built.layouts.full is built.layouts[2] is full

    swapped = replace(built, layouts=(recv, convert, {}))
    assert isinstance(swapped.layouts, transfer_module._StagingLayouts)
    assert swapped.layouts.full == {}


@pytest.mark.parametrize("enabled", [None, False, True])
def test_generator_layout_debug_assertion(harness, monkeypatch, enabled):
    if enabled is None:
        monkeypatch.delenv("MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT", raising=False)
    else:
        monkeypatch.setenv(
            "MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT", str(int(enabled))
        )
    fingerprint = [0]
    calls = []

    def layout_fingerprint():
        calls.append(True)
        return tuple(fingerprint)

    def capture_layout(_):
        capture = replace(
            harness.capture,
            generator_fingerprint=layout_fingerprint() if enabled else None,
        )
        return capture, harness.layout

    harness.prepare()
    source = ResolvedTrainerSource(harness.transfer.cached_trainer_source())
    transfer = _NixlStagedTransfer(
        agent_name="debug-target",
        device_id=0,
        device=torch.device("cuda:0"),
        streaming=harness.streaming,
        generator_fingerprint=layout_fingerprint,
        capture_layout=capture_layout,
    )
    method = LoadTimeTensorNixlUpdateMethod(
        transfer=transfer,
    )
    for _ in range(2):
        prepared = method.prepare(version=None, source=source)
        method.release(prepared)
    if enabled:
        assert (
            transfer._weight_update_plan.layout_capture_snapshot.generator_fingerprint
            == (0,)
        )
    fingerprint[0] = 1
    if enabled:
        with pytest.raises(AssertionError, match="generator layout changed"):
            method.prepare(version=None, source=source)
        assert transfer.cached_trainer_source() is None
    else:
        prepared = method.prepare(version=None, source=source)
        method.release(prepared)
        assert not calls
    method.close()


def test_generator_layout_debug_requires_adapter_callback(harness, monkeypatch):
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT", "1")
    harness.prepare()
    events = list(harness.events)
    with pytest.raises(ValueError, match="generator_fingerprint"):
        _NixlStagedTransfer(
            generator_fingerprint=None,
            agent_name="debug-target",
            device_id=0,
            device=torch.device("cuda:0"),
            streaming=harness.streaming,
            capture_layout=lambda _: (harness.capture, harness.layout),
        )
    assert harness.events == events


def test_changed_agent_metadata_resets_before_retry(harness, monkeypatch):
    harness.collect(harness.prepare())
    transfer = harness.transfer
    old = transfer.cached_trainer_source()
    shard = old.shards[0]
    payload = unwrap_rendezvous_blob(shard.manifest)
    blob = wrap_rendezvous_blob(
        b"replacement",
        payload.agent_name,
        payload.metadata_endpoint,
        payload.tensors,
    )
    current = replace(
        old,
        shards=(
            replace(
                shard,
                manifest=blob,
                manifest_digest=hashlib.sha256(blob).hexdigest(),
                structural_digest=structural_manifest_digest(blob),
            ),
        ),
    )
    monkeypatch.setattr(transfer._manager, "add_remote_agent", lambda _: "source")

    def prepare():
        return transfer.prepare(
            manifests=[blob],
            trainer_snapshot=current,
        )

    with pytest.raises(RuntimeError, match="metadata changed"):
        prepare()
    assert not transfer._loaded_agent_metadata
    harness.collect(prepare())
