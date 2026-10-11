# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import modelexpress_rl.inference.nixl_staged_transfer as transfer_module
import pytest
import torch
from modelexpress import p2p_pb2
from modelexpress.refit.reshard.rendezvous import (
    PublishedShard,
    PublishedTensor,
    structural_manifest_digest,
    wrap_rendezvous_blob,
)
from modelexpress.refit.reshard.slice_plan import PullSegment, Shard
from modelexpress.refit.reshard.transfer_plan import SourceInfo, TransferPlan
from modelexpress.refit.reshard.types import (
    CaptureResult,
    IncompleteRefit,
    RecordedCopy,
)
from modelexpress.refit.reshard.geometry import convert_source_weights
from modelexpress.refit.reshard.verify import tensor_digest
from modelexpress_rl import WeightPayloadFormat
from modelexpress_rl.inference.adapter import (
    GeneratorSource,
    GeneratorTransferInputs,
    NixlGeneratorSource,
)
from modelexpress_rl.inference.methods import LoadTimeTensorNixlUpdateMethod
from modelexpress_rl.inference.nixl_staged_transfer import (
    _bounded_batches,
    _load_agent_metadata,
    _NixlStagedTransfer,
    _LogicalRead,
    _pack_bounded_batches,
    _plan_staged_transfer,
    _PreparedBoundedTransfer,
    _PreparedNixlTransfer,
    _SourceBinding,
    _required_agent_metadata,
    _logical_transfer_plan,
    _resolve_sources,
    _ResolvedSources,
    _source_structure,
)
from modelexpress_rl.inference.plan import TrainerUpdateSource


def _bind_plan(plan, sources) -> tuple[TransferPlan, _SourceBinding]:
    agents = sorted(
        {shard.session for source in sources.values() for shard in source.shards}
    )
    resolved = _ResolvedSources(
        sources,
        {agent: agent for agent in agents},
        {agent: 0 for agent in agents},
        {agent: b"metadata" for agent in agents},
    )
    slots = {agent: f"slot-{agent}" for agent in agents}
    fingerprint = tuple(
        (
            agent,
            tuple(
                sorted(
                    (name, tuple(shard.shard_offset), tuple(shard.shape), shard.addr)
                    for name, source in sources.items()
                    for shard in source.shards
                    if shard.session == agent
                )
            ),
        )
        for agent in agents
    )
    binding = _SourceBinding(
        resolved, {slot: agent for agent, slot in slots.items()}, fingerprint
    )
    return _logical_transfer_plan(plan, resolved, slots), binding


def _prepared_bounded(batches, sources, transport, metrics=None) -> _PreparedBoundedTransfer:
    bound = []
    binding = None
    for batch in batches:
        plan, binding = _bind_plan(batch.plan, sources)
        bound.append(replace(batch, plan=plan))
    if binding is None:
        _, binding = _bind_plan(TransferPlan(), sources)
    return _PreparedBoundedTransfer(
        tuple(bound), sources, transport, binding, {} if metrics is None else metrics
    )


def test_prepare_plans_composed_canonical_capture(monkeypatch) -> None:
    manifest = [("packed.experts", torch.float32, (2, 2, 3))]
    convert = lambda weights: {
        f"layers.{index}.weight": weights["packed.experts"].select(0, index)
        for index in range(2)
    }
    resolved = _ResolvedSources(
        sources={
            "packed.experts": SimpleNamespace(
                dtype=torch.float32, global_shape=(2, 2, 3)
            )
        },
        session_to_agent={},
        session_to_device={},
        agent_metadata={},
    )
    transfer = _NixlStagedTransfer(
        device_id=0,
        device=torch.device("cpu"),
        manager=object(),
        convert_native_to_hf=convert,
    )
    transfer._source_cache.resolve = lambda _manifests, **_kwargs: resolved
    transfer._validate_complete = lambda *_args: None
    transfer._select_workspace_mode = lambda _mode: None
    transfer._ensure_workspace = lambda *_args: None
    transfer._descriptors = lambda _plan, **_kwargs: ()
    monkeypatch.setattr(transfer_module, "_required_agent_metadata", lambda *_args: {})
    monkeypatch.setattr(transfer_module, "_load_agent_metadata", lambda *_args: None)
    monkeypatch.setattr(transfer_module, "NixlReshardTransport", lambda *_args, **_kwargs: object())
    planned = []
    monkeypatch.setattr(
        transfer_module,
        "_plan_staged_transfer",
        lambda capture, _sources: planned.append(capture) or TransferPlan(),
    )
    captured_schema = []

    def capture_layout(schema) -> tuple:
        captured_schema.extend(schema)
        return CaptureResult(
            copies=[
                RecordedCopy(
                    src_name="layers.0.weight",
                    op_chain=(("transpose", (0, 1), ()),),
                    param_name="layers.0.weight",
                    dest_offset=0,
                    dest_shape=(2, 3),
                    dest_stride=(3, 1),
                    dest_dtype=torch.float32,
                )
            ],
            unsupported=["layers.1.weight"],
            unsupported_reasons={"layers.1.weight": "unsupported on lazy layers.1.weight"},
        ), {"layers.0.weight": ((2, 3), torch.float32)}

    transfer.prepare(
        manifests=[b"source"],
        source_fingerprints=(),
        capture_layout=capture_layout,
    )
    assert captured_schema == [
        ("layers.0.weight", torch.float32, (2, 3)),
        ("layers.1.weight", torch.float32, (2, 3)),
    ]
    assert planned[0].copies[0].src_name == "packed.experts"
    assert planned[0].copies[0].op_chain == convert_source_weights(
        convert, manifest
    )["layers.0.weight"]._ops + (("transpose", (0, 1), ()),)
    assert planned[0].unsupported == ["packed.experts"]
    assert set(planned[0].unsupported_reasons) == {"packed.experts"}


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
    batches = _bounded_batches(CaptureResult(copies=copies), layout, sources, 512)
    assert [b.nbytes for b in batches] == [256, 512]
    assert [set(b.layouts[0]) for b in batches] == [
        {"layer0.weight"},
        {"layer1.weight"},
    ]
    with pytest.raises(IncompleteRefit, match="exceeds"):
        _bounded_batches(CaptureResult(copies=copies), layout, sources, 511)
    with pytest.raises(IncompleteRefit, match="cover every"):
        _bounded_batches(CaptureResult(copies=copies[:1]), layout, sources, 512)
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            _bounded_batches(CaptureResult(copies=copies), layout, sources, invalid)


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
    batches = _bounded_batches(CaptureResult(copies=copies), layout, sources, 512)
    assert [b.nbytes for b in batches] == [256, 512]

    packed = _pack_bounded_batches(batches, 768)
    assert len(packed) == 1 and packed[0].nbytes == 768
    assert list(packed[0].layouts.recv) == ["layer0.weight", "layer1.weight"]
    assert packed[0].capture.copies == copies
    assert packed[0].plan.bytes_planned() == sum(
        b.plan.bytes_planned() for b in batches
    )
    assert packed[0].plan.descriptor_count() == sum(
        b.plan.descriptor_count() for b in batches
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

    with pytest.raises(IncompleteRefit, match="exceeds the packed staging budget"):
        _pack_bounded_batches(batches, 256)
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            _pack_bounded_batches(batches, invalid)


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
    transfer._workspace_generation = 0
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


@pytest.mark.parametrize("switch_failure", [None, "initialize", "register"])
@pytest.mark.parametrize("warm_cache", [False, True])
def test_released_updates_switch_workspaces_without_reusing_stale_plans(
    monkeypatch, switch_failure, warm_cache
):
    """Switch modes with real plans and byte copies, mocking only CUDA/NIXL."""
    events = []
    source_tensor = torch.arange(4, dtype=torch.float32)
    real_empty = torch.empty
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    for flag in (
        "MX_REFIT_CACHE_RESOLVED_SOURCES",
        "MX_REFIT_CACHE_BOUNDED_PLANS",
        "MX_REFIT_COPY_PLAN_KEY_ON_MISS",
        "MX_REFIT_REUSE_COMPLETE_PLAN",
    ):
        monkeypatch.setenv(flag, str(int(warm_cache)))
    monkeypatch.setattr(transfer_module, "classic_cuda_alloc", nullcontext)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))

    def empty(shape, **kwargs):
        if torch.device(kwargs.get("device", "cpu")).type == "cuda":
            kwargs["device"] = "cpu"
        return real_empty(shape, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)

    class Manager:
        def __init__(self, **kwargs):
            self.ready = False
            self.registered = {}
            self.fail_initialize = 0
            self.fail_register = 0

        def initialize(self):
            if self.ready:
                return
            events.append("initialize")
            if self.fail_initialize:
                self.fail_initialize -= 1
                raise RuntimeError("initialization failed")
            self.ready = True

        def shutdown(self):
            if self.registered:
                assert transfer._recv_buffers or transfer._bounded_arena is not None
            events.append("shutdown")
            self.registered.clear()
            self.ready = False

        def register_tensors(self, tensors):
            assert self.ready
            events.append("register")
            self.registered.update(tensors)
            if self.fail_register:
                self.fail_register -= 1
                raise RuntimeError("registration failed")

        def add_remote_agent(self, metadata):
            assert self.ready
            events.append("metadata")
            return metadata.decode()

    class Transport:
        def __init__(self, manager, *args, **kwargs):
            self.manager = manager

        def read(self, descriptors):
            assert self.manager.ready
            for d in descriptors:
                assert any(
                    t.data_ptr() <= d.dst_addr
                    and d.dst_addr + d.nbytes
                    <= t.data_ptr() + t.numel() * t.element_size()
                    for t in self.manager.registered.values()
                )
                ctypes.memmove(d.dst_addr, d.src_addr, d.nbytes)

        def post_reads(self, descriptors):
            self.read(descriptors)
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
                    )
                ],
            ),
        ],
    )
    source = TrainerUpdateSource(
        GeneratorTransferInputs(
            version_id="v",
            base_version_id=None,
            layout_signature="layout",
            payload_format=WeightPayloadFormat.FULL_TENSOR,
            sources=(
                GeneratorSource(
                    "rank:0",
                    "trainer",
                    "unchanged",
                    NixlGeneratorSource(
                        "source:19000", manifest, structural_manifest_digest(manifest)
                    ),
                ),
            ),
        )
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
        agent_name="receiver",
        device_id=0,
        device=torch.device("cuda:0"),
        listen_port=None,
    )
    method = LoadTimeTensorNixlUpdateMethod(
        transfer=transfer,
        capture_layout=lambda manifest: (
            capture,
            {"layer.weight": ((4,), torch.float32)},
        ),
    )
    first_plan = None
    try:
        for index, bounded in enumerate(
            (False, True, True, True, False, True, True, True)
        ):
            source_tensor.add_(1)
            if bounded:
                if index == 1 and switch_failure is not None:
                    setattr(transfer._manager, f"fail_{switch_failure}", 1)
                    with pytest.raises(RuntimeError, match="failed"):
                        method.prepare_streaming(
                            version=None, source=source, max_staging_bytes=256
                        )
                    assert method._active_plan is None
                    assert not transfer._manager.registered
                prepared = method.prepare_streaming(
                    version=None, source=source, max_staging_bytes=256
                )
                hit = warm_cache and (
                    index in (3, 7) or (index == 2 and switch_failure is not None)
                )
                assert prepared.metrics["plan_cache_hits"] == int(hit)
                assert prepared.metrics["owner_plan_builds"] == int(not hit)
                with pytest.raises(RuntimeError, match="release"):
                    method.prepare(version=None, source=source)
                for tensors in prepared.batches():
                    assert torch.equal(tensors["layer.weight"], source_tensor)
                with pytest.raises(RuntimeError, match="no longer active"):
                    transfer.stage(first_plan)
            else:
                prepared = method.prepare(version=None, source=source)
                assert torch.equal(
                    prepared.staged.tensors["layer.weight"], source_tensor
                )
                with pytest.raises(RuntimeError, match="release"):
                    method.prepare_streaming(
                        version=None, source=source, max_staging_bytes=256
                    )
                if first_plan is None:
                    first_plan = method._active_plan
                else:
                    assert method._active_plan is not first_plan
            method.release(prepared)
        # Every switch disconnects/deregisters before registering replacement storage.
        expected_registrations = 5 if switch_failure == "register" else 4
        assert events.count("metadata") == expected_registrations
        assert events.count("register") == expected_registrations
        for i, event in enumerate(events):
            if event == "shutdown" and i and events[i - 1] == "sync":
                assert events[i + 1] == "initialize"
    finally:
        method.close()
        assert transfer._source_cache._entry is None
        assert transfer._plan_cache._entry is None


def test_source_structure_uses_planner_shard_fields_and_ignores_digest():
    source = SourceInfo(
        global_shape=(4,),
        dtype=torch.float32,
        elsize=4,
        shards=[
            Shard(
                shard_offset=(0,),
                shape=(4,),
                session="trainer-0",
                addr=100,
                elsize=4,
                digest="version-a",
            )
        ],
    )

    expected = _source_structure(source)
    source.shards[0].digest = "version-b"

    assert _source_structure(source) == expected


def test_default_transfer_timeout_matches_the_lease_budget(monkeypatch):
    monkeypatch.setenv("MX_TRANSFER_TIMEOUT", "17")

    transfer = _NixlStagedTransfer(
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
        session_to_agent={"agent-a": "agent-a"},
        session_to_device={},
        agent_metadata={"agent-a": b"metadata"},
    )
    binding = _SourceBinding(
        resolved, {"slot-a": "agent-a"}, (("slot-a", "agent-a", "manifest-a"),)
    )
    plan = TransferPlan(
        segments=[_LogicalRead("slot-a", "weight", (0,), (1,), 0, "weight", 0, 4)]
    )
    assert _required_agent_metadata(plan, resolved, binding) == {
        "agent-a": b"metadata"
    }

    with pytest.raises(RuntimeError, match="unknown source slots"):
        _required_agent_metadata(
            TransferPlan(
                segments=[
                    _LogicalRead("missing-slot", "weight", (0,), (1,), 0, "weight", 0, 4)
                ]
            ),
            resolved,
            binding,
        )

    missing_metadata = _ResolvedSources({}, {}, {}, {})
    with pytest.raises(RuntimeError, match="without NIXL metadata"):
        _required_agent_metadata(
            plan,
            missing_metadata,
            _SourceBinding(
                missing_metadata,
                {"slot-a": "agent-a"},
                (("slot-a", "agent-a", "missing-metadata"),),
            ),
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

    plan = _plan_staged_transfer(CaptureResult(copies=[copy]), {"weight": source})

    assert plan.segments == []
    assert len(plan.full_pulls) == 1
    assert plan.full_pulls[0].copies == [copy]
    assert sum(segment.nbytes for segment in plan.full_pulls[0].segments) == 64
    assert {segment.session for segment in plan.full_pulls[0].segments} == {
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

    plan = _plan_staged_transfer(CaptureResult(copies=[copy]), {"weight": source})

    assert plan.full_pulls == []
    assert sum(segment.nbytes for segment in plan.segments) == 32
    assert {segment.session for segment in plan.segments} == {"left"}


def test_logical_plan_preserves_conversion_and_full_pull_reads(monkeypatch) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    source_tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    source = SourceInfo(
        global_shape=(4, 4),
        dtype=torch.float32,
        elsize=4,
        shards=[Shard((0, 0), (4, 4), "agent-a", source_tensor.data_ptr(), 4)],
    )
    resolved = _ResolvedSources(
        {"weight": source},
        {"agent-a": "agent-a"},
        {"agent-a": 0},
        {"agent-a": b"metadata"},
    )
    binding = _SourceBinding(
        resolved,
        {"slot-a": "agent-a"},
        (("slot-a", "agent-a", source_tensor.data_ptr()),),
    )
    copy = RecordedCopy(
        src_name="weight",
        op_chain=(("narrow", (1, 0, 2), ()),),
        param_name="converted", dest_offset=0,
        dest_shape=(4, 2), dest_stride=(2, 1), dest_dtype=torch.float16,
    )
    converted = _plan_staged_transfer(CaptureResult(copies=[copy]), resolved.sources)
    logical_converted = _logical_transfer_plan(converted, resolved, {"agent-a": "slot-a"})
    conversion_reads = logical_converted.converts[0].segments
    assert conversion_reads[0] == _LogicalRead(
        "slot-a", "weight", (0, 0), (4, 4), 0, "converted", 0, 8
    )
    assert sum(segment.nbytes for segment in conversion_reads) == 32
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._recv_buffers = {}
    transfer._full_buffers = {}
    transfer._convert_buffers = {}
    conversion_buffer = torch.empty((4, 2), dtype=torch.float32)
    conversion_descriptors = transfer._descriptors(
        logical_converted,
        convert={"converted": conversion_buffer},
        source_binding=binding,
    )
    for descriptor in conversion_descriptors:
        ctypes.memmove(descriptor.dst_addr, descriptor.src_addr, descriptor.nbytes)
    assert torch.equal(conversion_buffer, source_tensor[:, :2])

    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    full_pull = _plan_staged_transfer(CaptureResult(copies=[replace(copy, dest_dtype=torch.float32)]), resolved.sources)
    logical_full = _logical_transfer_plan(full_pull, resolved, {"agent-a": "slot-a"})
    assert logical_full.full_pulls[0].segments[0] == _LogicalRead(
        "slot-a", "weight", (0, 0), (4, 4), 0, "weight", 0, 64
    )
    assert binding.read(logical_full.full_pulls[0].segments[0]) == (
        "agent-a", source_tensor.data_ptr()
    )
    reconstructed = torch.empty_like(source_tensor)
    full_descriptors = transfer._descriptors(
        logical_full,
        full={"weight": reconstructed},
        source_binding=binding,
    )
    for descriptor in full_descriptors:
        ctypes.memmove(descriptor.dst_addr, descriptor.src_addr, descriptor.nbytes)
    assert torch.equal(reconstructed, source_tensor)


def test_packed_mixed_reads_keep_logical_binding_provenance(monkeypatch) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    values = torch.arange(4, dtype=torch.float32)
    source = SourceInfo(
        global_shape=(4,),
        dtype=torch.float32,
        elsize=4,
        shards=[Shard((0,), (4,), "agent-a", values.data_ptr(), 4)],
    )
    resolved = _ResolvedSources(
        {"weight": source},
        {"agent-a": "agent-a"},
        {"agent-a": 0},
        {"agent-a": b"metadata"},
    )
    copies = [
        RecordedCopy("weight", (), "layer0.weight", 0, (4,), (1,), torch.float32),
        RecordedCopy("weight", (), "layer1.weight", 0, (4,), (1,), torch.bfloat16),
    ]
    capture = CaptureResult(copies=copies)
    layout = {copy.param_name: (copy.dest_shape, copy.dest_dtype) for copy in copies}
    batches = _bounded_batches(capture, layout, resolved.sources, 512)
    packed = _pack_bounded_batches(batches, 1024)

    assert len(packed) == 1
    logical = _logical_transfer_plan(
        packed[0].plan, resolved, {"agent-a": "slot-a"}
    )
    reads = logical.segments + [
        segment for conversion in logical.converts for segment in conversion.segments
    ]
    assert len(logical.segments) == 1
    assert len(logical.converts) == 1
    assert all(isinstance(segment, _LogicalRead) for segment in reads)
    assert {segment.param_name for segment in reads} == {
        "layer0.weight", "layer1.weight"
    }
    assert sum(segment.nbytes for segment in reads) == 32


@pytest.mark.parametrize("bounded", [False, True])
def test_prepare_retains_logical_plan_and_source_binding(monkeypatch, bounded) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    values = torch.arange(4, dtype=torch.float32)
    source = SourceInfo(
        global_shape=(4,),
        dtype=torch.float32,
        elsize=4,
        shards=[Shard((0,), (4,), "agent-a", values.data_ptr(), 4)],
    )
    resolved = _ResolvedSources(
        {"weight": source},
        {"agent-a": "agent-a"},
        {"agent-a": 0},
        {"agent-a": b"metadata"},
    )
    copy = RecordedCopy("weight", (), "weight", 0, (4,), (1,), torch.float32)
    capture = CaptureResult(copies=[copy])
    layout = {"weight": ((4,), torch.float32)}
    monkeypatch.setattr(
        transfer_module, "NixlReshardTransport", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(transfer_module, "_load_agent_metadata", lambda *_args: None)
    manager = SimpleNamespace(register_tensors=lambda *_args: None)
    transfer = _NixlStagedTransfer(
        device_id=0,
        device=torch.device("cuda:0") if bounded else torch.device("cpu"),
        manager=manager,
    )
    transfer._source_cache.resolve = lambda *_args, **_kwargs: resolved
    transfer._select_workspace_mode = lambda _mode: None
    transfer._validate_complete = lambda *_args: None
    transfer._ensure_workspace = lambda *_args: None
    transfer._allocate_arena = lambda size: torch.empty(size, dtype=torch.uint8)
    transfer._recv_buffers = {"weight": torch.empty_like(values)}

    prepared = transfer.prepare(
        manifests=[b"manifest"],
        source_fingerprints=(
            ("replica-slot", "agent-a", values.data_ptr()),
        ),
        capture_layout=lambda _manifest: (capture, layout),
        max_staging_bytes=512 if bounded else None,
        source_slot_ids=("replica-slot",),
    )

    plans = [batch.plan for batch in prepared.batches] if bounded else [prepared.plan]
    reads = [segment for plan in plans for segment in plan._all_segments()]
    assert reads and all(isinstance(segment, _LogicalRead) for segment in reads)
    assert {segment.source_slot_id for segment in reads} == {"replica-slot"}
    assert prepared.source_binding.read(reads[0]) == ("agent-a", values.data_ptr())
    if not bounded:
        assert len(prepared.descriptors) == 1
        assert prepared.descriptors[0].session == "agent-a"
        assert prepared.descriptors[0].src_addr == values.data_ptr()


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
                digest=digest,
            )
        ],
    )
    plan, source_binding = _bind_plan(TransferPlan(), {"weight": source})
    return _PreparedNixlTransfer(
        plan=plan,
        capture=CaptureResult(copies=[copy]),
        sources={"weight": source},
        descriptors=(),
        transport=object(),
        source_binding=source_binding,
    )


def test_staged_verification_rejects_missing_or_mismatched_digest():
    tensor = torch.arange(64, dtype=torch.int32)
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._descriptor_cache = None
    transfer._workspace_generation = 0
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
            TransferPlan(),
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
            agent_name="generator",
            device_id=0,
            device=torch.device("cpu"),
            listen_port=19000,
        )
    assert calls == ["initialize", "shutdown"]

    transfer = object.__new__(_NixlStagedTransfer)
    transfer._descriptor_cache = None
    transfer._workspace_generation = 0
    transfer._manager = _Manager()
    transfer._owns_manager = True
    transfer._closed = False
    transfer._recv_buffers = {}
    transfer._convert_buffers = {}
    transfer._full_buffers = {}
    transfer._staging_arenas = []
    transfer._staging_registrations = []
    transfer._source_cache = transfer_module._SourceResolutionCache()
    transfer._plan_cache = transfer_module._BoundedPlanCache()
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
        device_id=0,
        device=torch.device("cpu"),
        manager=_Manager(),
    )
    assert transfer._workspace_mode is None

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
    # Direct peer receive never selects a workspace on the borrowed agent.
    assert transfer._workspace_mode is None

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
    transfer._workspace_generation = 0
    transfer._device = torch.device("cpu")
    transfer._device_id = 0
    transfer._timeout = 30.0
    transfer._manager = _Manager()
    transfer._closed = False
    transfer._workspace_mode = "full"
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
    transfer._workspace_generation = 0
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
    batches = _bounded_batches(CaptureResult(copies=copies), layout, {"w": source}, 256)
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
    transfer._workspace_generation = 0
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
    batches = _bounded_batches(CaptureResult(copies=copies), layout, {"w": source}, 256)
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
    transfer._workspace_generation = 0
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


@pytest.mark.parametrize("staging_buffers", [1, 2])
@pytest.mark.parametrize("source_memory_type", ["VRAM", "DRAM"])
def test_prepare_stages_in_pinned_host_memory_and_splits_the_budget(
    monkeypatch, staging_buffers, source_memory_type
):
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
        agent_name="receiver",
        device_id=0,
        device=torch.device("cuda:0"),
        listen_port=None,
    )
    layout = {"layer.weight": ((4,), torch.float32)}
    try:
        # 16 bytes of payload rounds to one 256-byte residency per arena; the
        # budget is split per buffer, so 256 * buffers admits it and less does not.
        expected = (
            "split across staging_buffers=2 gives 255 bytes per arena"
            if staging_buffers == 2
            else "exceeds max_staging_bytes=255; raise max_staging_bytes"
        )
        with pytest.raises(IncompleteRefit, match=expected):
            transfer.prepare(
                manifests=[manifest],
                source_fingerprints=(
                    ("source", "source", source_tensor.data_ptr()),
                ),
                capture_layout=lambda m: (capture, layout),
                max_staging_bytes=256 * staging_buffers - 1,
                staging_device="cpu",
                staging_buffers=staging_buffers,
            )
        prepared = transfer.prepare(
            manifests=[manifest],
            source_fingerprints=(
                ("source", "source", source_tensor.data_ptr()),
            ),
            capture_layout=lambda m: (capture, layout),
            max_staging_bytes=256 * staging_buffers,
            staging_device="cpu",
            staging_buffers=staging_buffers,
        )
        assert events.count("register_dram") == staging_buffers
        assert len(transfer._staging_arenas) == staging_buffers
        assert all(a.device.type == "cpu" for a in transfer._staging_arenas)
        assert transfer._workspace_mode == f"bounded:cpu:{staging_buffers}"
        metrics = {}
        for tensors in transfer.iter_bounded(prepared, metrics):
            assert tensors["layer.weight"].device.type == "cpu"
            assert torch.equal(tensors["layer.weight"], source_tensor)
        assert metrics["staging_buffers"] == staging_buffers
        assert metrics["staging_peak_bytes"] == 256 * staging_buffers
        if staging_buffers == 2:
            # Changing the buffer count is a workspace switch: the host arenas
            # are deregistered before the agent is torn down and rebuilt.
            transfer._active = None
            transfer.prepare(
                manifests=[manifest],
                source_fingerprints=(
                    ("source", "source", source_tensor.data_ptr()),
                ),
                capture_layout=lambda m: (capture, layout),
                max_staging_bytes=256,
                staging_device="cpu",
                staging_buffers=1,
            )
            assert events.count("deregister") == 2
            assert events.index("deregister") < events.index("shutdown")
            assert len(transfer._staging_arenas) == 1
    finally:
        transfer.close()
    assert not transfer._manager.registered


def test_prepare_rejects_invalid_staging_options():
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._descriptor_cache = None
    transfer._workspace_generation = 0
    transfer._closed = False
    transfer._device = torch.device("cuda:0")
    with pytest.raises(ValueError, match="staging_device"):
        transfer.prepare(
            manifests=[],
            source_fingerprints=(),
            capture_layout=None,
            max_staging_bytes=1,
            staging_device="disk",
        )
    with pytest.raises(ValueError, match="staging_buffers"):
        transfer.prepare(
            manifests=[],
            source_fingerprints=(),
            capture_layout=None,
            max_staging_bytes=1,
            staging_buffers=0,
        )


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
    batches = _bounded_batches(CaptureResult(copies=copies), layout, {"w": source}, 512)
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
    transfer._workspace_generation = 0
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
    batches = _bounded_batches(CaptureResult(copies=copies), layout, {"w": source}, 512)
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
    transfer._workspace_generation = 0
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

    built = transfer_module._BoundedBatch(
        CaptureResult(copies=[]), TransferPlan(), (recv, convert, full), 256
    )
    assert isinstance(built.layouts, transfer_module._StagingLayouts)
    assert built.layouts.recv is built.layouts[0] is recv
    assert built.layouts.full is built.layouts[2] is full

    swapped = replace(built, layouts=(recv, convert, {}))
    assert isinstance(swapped.layouts, transfer_module._StagingLayouts)
    assert swapped.layouts.full == {}


def test_logical_plan_binds_replica_addresses_without_recompiling() -> None:
    from modelexpress.refit.reshard.slice_plan import Shard
    from modelexpress.refit.reshard.transfer_plan import SourceInfo

    values = torch.arange(8, dtype=torch.float32)
    replica = values.clone()
    capture = CaptureResult(copies=[RecordedCopy(
        src_name="weight", op_chain=(), param_name="weight", dest_offset=0,
        dest_shape=(8,), dest_stride=(1,), dest_dtype=torch.float32,
    )])

    def resolve(tensor, agent) -> _ResolvedSources:
        return _ResolvedSources(
            {"weight": SourceInfo((8,), torch.float32, 4, [Shard((0,), (8,), agent, tensor.data_ptr(), 4)])},
            {agent: agent}, {agent: 0}, {agent: b"metadata"},
        )

    first = resolve(values, "replica-a")
    second = resolve(replica, "replica-b")
    physical = transfer_module._plan_staged_transfer(capture, first.sources)
    logical = transfer_module._logical_transfer_plan(physical, first, {"replica-a": "group"})
    read = logical.segments[0]
    assert read.source_slot_id == "group"
    assert read.src_byte == 0
    assert not hasattr(read, "src_addr")
    assert not hasattr(read, "session")
    receiver = object.__new__(_NixlStagedTransfer)
    receiver._recv_buffers = {"weight": torch.empty_like(values)}
    receiver._full_buffers = {}
    receiver._convert_buffers = {}
    for resolved, agent in ((first, "replica-a"), (second, "replica-b")):
        binding = transfer_module._SourceBinding(
            resolved,
            {"group": agent},
            (
                (
                    "group",
                    agent,
                    (values if agent == "replica-a" else replica).data_ptr(),
                ),
            ),
        )
        descriptors = receiver._descriptors(logical, source_binding=binding)
        assert descriptors[0].session == agent
        assert descriptors[0].src_addr == (values if agent == "replica-a" else replica).data_ptr()
        for descriptor in descriptors:
            ctypes.memmove(descriptor.dst_addr, descriptor.src_addr, descriptor.nbytes)
        assert torch.equal(receiver._recv_buffers["weight"], values)
        assert logical.segments[0] is read
