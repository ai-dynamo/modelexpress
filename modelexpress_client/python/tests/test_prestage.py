# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-stage: identity, discovery, read planning and the receive-only agent.

A helper process that is not an engine reads a source's weights into memory it
owns. These tests cover what that helper relies on without a GPU or NIXL:
the identity it builds equals the engine's, discovery picks the right rank,
follows the topology policy and skips unusable candidates like RdmaStrategy,
a read lease is released on every path, read planning maps only what the
caller stages, and cleanup reports every failure, on every call, so the caller
never hands over memory that is still registered.

Run: pytest tests/test_prestage.py
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from modelexpress import p2p_pb2, prestage
from modelexpress.engines.vllm.source_identity import (
    build_source_identity,
    build_source_identity_from_values,
)
from modelexpress.metadata import worker_server
from modelexpress.nixl_transfer import NixlTransferManager
from modelexpress.topology import TopologyPolicy
from modelexpress.types import TensorDescriptor


class FakeAgent:
    """The NIXL calls the receive-only lifecycle makes.

    Like NIXL, removing a peer that is not loaded is refused (NOT_FOUND);
    ``fail_remove`` makes the removal of a loaded peer fail as well.
    """

    def __init__(self, fail_deregister=False, ready=True, fail_remove=False):
        self.deregistered: list[object] = []
        self.removed: list[str] = []
        self.fail_deregister = fail_deregister
        self.fail_remove = fail_remove
        self.ready = ready
        self._fetched: set[str] = set()
        self._loaded: set[str] = set()

    def fetch_remote_metadata(self, name, ip, port):
        self._fetched.add(name)
        if self.ready:
            self._loaded.add(name)

    def check_remote_metadata(self, name):
        return name in self._loaded

    def remove_remote_agent(self, name):
        if name not in self._loaded:
            raise RuntimeError("NIXL_ERR_NOT_FOUND")
        if self.fail_remove:
            raise RuntimeError("NIXL_ERR_BACKEND")
        self._loaded.discard(name)
        self.removed.append(name)

    def deregister_memory(self, registered):
        if self.fail_deregister:
            raise RuntimeError("deregistration refused")
        self.deregistered.append(registered)

    def get_agent_metadata(self):
        return b"metadata"


class FakeClient:
    """ListSources / GetMetadata answers for one identity."""

    def __init__(self, instances, workers):
        self.instances = instances
        self.workers = workers
        self.metadata_calls: list[str] = []

    def list_sources(self, identity=None, status_filter=None):
        assert status_filter == p2p_pb2.SOURCE_STATUS_READY
        return SimpleNamespace(instances=self.instances)

    def get_metadata(self, mx_source_id, worker_id):
        self.metadata_calls.append(worker_id)
        worker = self.workers.get(worker_id)
        return SimpleNamespace(found=worker is not None, worker=worker)

    def worker_rpc_retry_policy(self):
        return 0, 0.0


def _instance(worker_id, rank, accelerator="cuda", zone=None):
    inst = p2p_pb2.SourceInstanceRef(
        mx_source_id="src", worker_id=worker_id, worker_rank=rank, accelerator=accelerator
    )
    if zone is not None:
        inst.topology["zone"] = zone
    return inst


def _worker(agent="agent", grpc="10.0.0.1:6000", md="10.0.0.1:5000", accelerator="cuda"):
    return SimpleNamespace(
        agent_name=agent,
        worker_grpc_endpoint=grpc,
        metadata_endpoint=md,
        accelerator=accelerator,
    )


def _manifest_response(worker_id, rank, accelerator="cuda", agent=None):
    response = p2p_pb2.GetTensorManifestResponse(
        tensors=[
            p2p_pb2.TensorDescriptor(
                name="w", addr=4096, size=64, device_id=3, dtype="torch.bfloat16"
            )
        ],
        mx_source_id="src",
        metadata_endpoint="10.0.0.1:5000",
        agent_name=f"leased-{worker_id}" if agent is None else agent,
        worker_rank=rank,
        accelerator=accelerator,
    )
    response.worker_id = worker_id
    return response


IDENTITY = p2p_pb2.SourceIdentity(model_name="m", dtype="bfloat16", quantization="fp8")


@pytest.fixture
def in_order(monkeypatch):
    """Keep candidates in listed order (the default selector shuffles)."""
    monkeypatch.setattr(
        prestage,
        "get_configured_selector",
        lambda: SimpleNamespace(order=lambda candidates, ctx: list(candidates)),
    )


class FakeLease:
    def __init__(self, manifest, fail_close=False):
        self.manifest = manifest
        self.fail_close = fail_close
        self.closed = 0

    def close(self):
        self.closed += 1
        if self.fail_close:
            raise RuntimeError("lease not found")


@pytest.fixture
def manifest(monkeypatch):
    """PrepareTensorRead / GetTensorManifest answers, as the source fills them.

    The leased manifest carries the serving process's own rank and accelerator
    (``ranks`` / ``accelerators``, set by ``_client``; a test may override one
    to model a channel that reached another process).
    """
    calls = []
    failing: set[str] = set()
    leases: list[FakeLease] = []
    no_agent: set[str] = set()
    ranks: dict[str, int] = {}
    accelerators: dict[str, str] = {}

    def prepare(endpoint, mx_source_id, *, worker_id="", timeout=5.0,
                max_retries=0, retry_backoff_seconds=0.0):
        calls.append(("lease", worker_id))
        if endpoint in failing:
            raise RuntimeError("UNAVAILABLE")
        response = _manifest_response(
            worker_id,
            ranks[worker_id],
            accelerators.get(worker_id, "cuda"),
            agent="" if worker_id in no_agent else None,
        )
        lease = FakeLease(response)
        leases.append(lease)
        return lease, response.ByteSize()

    def fetch(endpoint, mx_source_id, timeout=5.0, *, worker_id=""):
        calls.append(("manifest", worker_id))
        if endpoint in failing:
            raise RuntimeError("UNAVAILABLE")
        return list(_manifest_response(worker_id, 0).tensors), 10

    monkeypatch.setattr(worker_server, "prepare_tensor_read", prepare)
    monkeypatch.setattr(worker_server, "fetch_tensor_manifest", fetch)
    return SimpleNamespace(
        calls=calls,
        failing=failing,
        leases=leases,
        no_agent=no_agent,
        ranks=ranks,
        accelerators=accelerators,
    )


def _client(manifest, instances, workers):
    for inst in instances:
        manifest.ranks.setdefault(inst.worker_id, inst.worker_rank)
    return FakeClient(instances, workers)


@pytest.fixture
def rpc(monkeypatch):
    """The real prepare_tensor_read / TensorReadLease over an in-memory WorkerService."""
    state = SimpleNamespace(manifests={}, released=[], channels=[], release_raises=None)

    class Channel:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    class Stub:
        def __init__(self, channel):
            pass

        def PrepareTensorRead(self, request, timeout=None):
            return p2p_pb2.PrepareTensorReadResponse(
                lease_id=f"lease-{request.worker_id}",
                manifest=state.manifests[request.worker_id],
            )

        def ReleaseTensorRead(self, request, timeout=None):
            if state.release_raises is not None:
                raise state.release_raises
            state.released.append(request.lease_id)
            return p2p_pb2.ReleaseTensorReadResponse()

    def channel(endpoint):
        state.channels.append(Channel())
        return state.channels[-1]

    monkeypatch.setattr(worker_server.grpc, "insecure_channel", channel)
    monkeypatch.setattr(worker_server.p2p_pb2_grpc, "WorkerServiceStub", Stub)
    return state


class TestIdentity:
    def test_matches_the_engine_derivation(self, monkeypatch):
        monkeypatch.delenv("MX_MODEL_REVISION", raising=False)
        got = build_source_identity_from_values(
            model="m",
            dtype="torch.bfloat16",
            quantization="fp8",
            tensor_parallel_size=1,
            data_parallel_size=8,
            enable_expert_parallel=True,
        )
        parallel = SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=8,
            enable_expert_parallel=True,
            prefill_context_parallel_size=1,
        )
        want = build_source_identity(
            SimpleNamespace(parallel_config=parallel),
            SimpleNamespace(model="m", dtype="torch.bfloat16", quantization="fp8", revision=None),
        )
        assert got == want
        assert got.expert_parallel_size == 8
        assert got.dtype == "bfloat16"

    def test_unquantized_and_revision_override(self, monkeypatch):
        monkeypatch.setenv("MX_MODEL_REVISION", "rev-1")
        got = build_source_identity_from_values(
            model="m", dtype="float16", quantization=None, tensor_parallel_size=2
        )
        assert got.quantization == ""
        assert got.revision == "rev-1"
        assert got.expert_parallel_size == 1


class TestDiscovery:
    def test_picks_the_requested_rank_and_holds_a_lease(self, in_order, manifest):
        client = _client(
            manifest,
            [_instance("w0", 0), _instance("w2", 2)],
            {"w0": _worker(), "w2": _worker("agent2")},
        )
        src = prestage.discover_source(client, IDENTITY, 2, receiver_id="helper-2")
        assert (src.worker_id, src.worker_rank) == ("w2", 2)
        assert src.agent_name == "leased-w2"
        assert client.metadata_calls == ["w2"]
        assert [t.name for t in src.tensors] == ["w"]
        assert src.tensors[0].device_id == 3
        assert src.tensor_read is manifest.leases[0]
        assert src.release() == []
        assert manifest.leases[0].closed == 1
        assert src.release() == []
        assert manifest.leases[0].closed == 1

    def test_without_a_lease_uses_the_tensor_manifest(self, in_order, manifest):
        client = _client(manifest, [_instance("w0", 0)], {"w0": _worker("agent0")})
        src = prestage.discover_source(
            client, IDENTITY, 0, receiver_id="r", tensor_read_lease=False
        )
        assert src.agent_name == "agent0"
        assert src.tensor_read is None
        assert manifest.calls == [("manifest", "w0")]

    def test_skips_unusable_candidates(self, in_order, manifest):
        workers = {
            "w-centralized": _worker(grpc=""),
            "w-broken": _worker(grpc="10.0.0.9:6000"),
            "w-no-agent": _worker(),
            "w-ok": _worker(),
        }
        manifest.failing.add("10.0.0.9:6000")
        manifest.no_agent.add("w-no-agent")
        names = ("w-missing", "w-centralized", "w-broken", "w-no-agent", "w-ok")
        client = _client(manifest, [_instance(w, 1) for w in names], workers)
        src = prestage.discover_source(client, IDENTITY, 1, receiver_id="r", max_candidates=5)
        assert src.worker_id == "w-ok"
        # The lease taken on the candidate without an agent was released.
        assert [lease.closed for lease in manifest.leases] == [1, 0]

    def test_leased_manifest_from_another_process_is_refused(self, in_order, manifest):
        names = ("w-moved", "w-xpu", "w-unknown", "w-ok")
        client = _client(
            manifest, [_instance(w, 0) for w in names], {w: _worker() for w in names}
        )
        manifest.ranks["w-moved"] = 1
        manifest.accelerators["w-xpu"] = "xpu"
        # Unknown is not accepted for quantized weights on the authoritative check.
        manifest.accelerators["w-unknown"] = ""
        src = prestage.discover_source(client, IDENTITY, 0, receiver_id="r", max_candidates=4)
        assert src.worker_id == "w-ok"
        assert [lease.closed for lease in manifest.leases] == [1, 1, 1, 0]

    def test_interrupt_after_the_lease_still_releases_it(self, in_order, manifest, monkeypatch):
        client = _client(manifest, [_instance("w0", 0)], {"w0": _worker()})

        def interrupted(**kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(prestage, "TensorDescriptor", interrupted)
        with pytest.raises(KeyboardInterrupt):
            prestage.discover_source(client, IDENTITY, 0, receiver_id="r")
        assert manifest.leases[0].closed == 1

    def test_excluded_workers_are_not_tried(self, in_order, manifest):
        client = _client(
            manifest, [_instance("w0", 0), _instance("w1", 0)], {"w0": _worker(), "w1": _worker()}
        )
        src = prestage.discover_source(
            client, IDENTITY, 0, receiver_id="r", exclude_worker_ids={"w0"}
        )
        assert src.worker_id == "w1"
        with pytest.raises(prestage.PrestageError, match="2 excluded"):
            prestage.discover_source(
                client, IDENTITY, 0, receiver_id="r", exclude_worker_ids=["w0", "w1"]
            )

    def test_stops_after_max_candidates(self, in_order, manifest):
        instances = [_instance(f"w{i}", 1) for i in range(5)]
        client = _client(manifest, instances, {"w4": _worker()})
        with pytest.raises(prestage.PrestageError, match="no usable source"):
            prestage.discover_source(client, IDENTITY, 1, receiver_id="r")
        assert client.metadata_calls == ["w0", "w1", "w2"]

    def test_rechecks_the_fetched_accelerator(self, in_order, manifest):
        client = _client(
            manifest, [_instance("w0", 0, accelerator="")], {"w0": _worker(accelerator="xpu")}
        )
        with pytest.raises(prestage.PrestageError, match="accelerator"):
            prestage.discover_source(client, IDENTITY, 0, receiver_id="r")
        assert manifest.calls == []

    def test_no_source_for_the_rank(self, in_order, manifest):
        client = _client(manifest, [_instance("w0", 0)], {"w0": _worker()})
        with pytest.raises(prestage.PrestageError, match="no READY source for worker_rank 5"):
            prestage.discover_source(client, IDENTITY, 5, receiver_id="r")

    def test_required_topology_filters_and_preferred_reorders(
        self, in_order, manifest, monkeypatch
    ):
        instances = [_instance("far", 0, zone="b"), _instance("near", 0, zone="a")]
        client = _client(manifest, instances, {"far": _worker(), "near": _worker()})
        monkeypatch.setattr(
            prestage, "resolve_policy", lambda: TopologyPolicy("zone", "preferred", "a")
        )
        assert prestage.discover_source(client, IDENTITY, 0, receiver_id="r").worker_id == "near"
        monkeypatch.setattr(
            prestage, "resolve_policy", lambda: TopologyPolicy("zone", "required", "c")
        )
        with pytest.raises(prestage.PrestageError, match="no READY source"):
            prestage.discover_source(client, IDENTITY, 0, receiver_id="r")

    def test_required_topology_without_a_local_value_stops_before_listing(
        self, manifest, monkeypatch
    ):
        monkeypatch.setattr(
            prestage, "resolve_policy", lambda: TopologyPolicy("zone", "required", None)
        )

        class NoList(FakeClient):
            def list_sources(self, identity=None, status_filter=None):
                raise AssertionError("listed")

        with pytest.raises(prestage.PrestageError, match="required but unknown"):
            prestage.discover_source(NoList([], {}), IDENTITY, 0, receiver_id="r")

    @pytest.mark.parametrize("selector", ["random", "rendezvous_hash", "load_aware", "topology_aware"])
    def test_every_selector_accepts_the_receiver_context(self, selector, monkeypatch, manifest):
        monkeypatch.setenv("MX_P2P_SOURCE_SELECTOR", selector)
        client = _client(
            manifest, [_instance("w0", 0), _instance("w1", 0)], {"w0": _worker(), "w1": _worker()}
        )
        src = prestage.discover_source(client, IDENTITY, 0, receiver_id="helper-0")
        assert src.worker_id in ("w0", "w1")

    def test_a_failed_release_stays_reported(self):
        lease = FakeLease(None, fail_close=True)
        src = prestage.PrestageSource("s", "w", 0, "a", "h:1", "h:2", "cuda", [], tensor_read=lease)
        first = src.release()
        assert len(first) == 1 and "lease not found" in first[0]
        assert src.release() == first
        assert lease.closed == 1


class TestLeaseOverRpc:
    """The real PrepareTensorRead client and TensorReadLease, with an in-memory stub."""

    def test_the_lease_is_held_until_release(self, in_order, rpc):
        rpc.manifests["w0"] = _manifest_response("w0", 0)
        client = FakeClient([_instance("w0", 0)], {"w0": _worker()})
        src = prestage.discover_source(client, IDENTITY, 0, receiver_id="r")
        assert src.agent_name == "leased-w0"
        assert rpc.released == []
        assert src.release() == []
        assert rpc.released == ["lease-w0"]
        assert [c.closed for c in rpc.channels] == [1]

    def test_a_refused_manifest_releases_its_lease(self, in_order, rpc):
        rpc.manifests["w0"] = _manifest_response("w0", 1)
        rpc.manifests["w1"] = _manifest_response("w1", 0, accelerator="")
        client = FakeClient(
            [_instance("w0", 0), _instance("w1", 0)], {"w0": _worker(), "w1": _worker()}
        )
        with pytest.raises(prestage.PrestageError, match="no usable source"):
            prestage.discover_source(client, IDENTITY, 0, receiver_id="r")
        assert rpc.released == ["lease-w0", "lease-w1"]
        assert [c.closed for c in rpc.channels] == [1, 1]

    def test_an_interrupted_release_stays_reported(self, in_order, rpc):
        rpc.manifests["w0"] = _manifest_response("w0", 0)
        client = FakeClient([_instance("w0", 0)], {"w0": _worker()})
        src = prestage.discover_source(client, IDENTITY, 0, receiver_id="r")
        rpc.release_raises = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            src.release()
        errors = src.release()
        assert len(errors) == 1 and "KeyboardInterrupt" in errors[0]
        assert [c.closed for c in rpc.channels] == [1]


class TestReadPlan:
    def test_maps_only_resolved_entries(self):
        tensors = [
            TensorDescriptor("a", 1000, 16, 3, "torch.uint8"),
            TensorDescriptor("buffer", 2000, 8, 3, "torch.uint8"),
            TensorDescriptor("empty", 3000, 0, 3, "torch.uint8"),
            TensorDescriptor("b", 4000, 32, 3, "torch.uint8"),
        ]
        local = {"a": 10, "b": 50}
        plan = prestage.plan_reads(tensors, lambda t: local.get(t.name))
        assert plan.ranges == [(1000, 10, 16, 3), (4000, 50, 32, 3)]
        assert (plan.matched, plan.matched_bytes) == (2, 48)
        assert (plan.unmatched, plan.unmatched_bytes) == (["buffer"], 8)

    def test_negative_size_is_refused(self):
        with pytest.raises(prestage.PrestageError):
            prestage.plan_reads([TensorDescriptor("x", 1, -1, 0, "torch.uint8")], lambda t: 0)

    def test_resolver_errors_propagate(self):
        def resolve(t):
            raise ValueError("size mismatch")

        with pytest.raises(ValueError):
            prestage.plan_reads([TensorDescriptor("x", 1, 4, 0, "torch.uint8")], resolve)


def _receiver(agent, accelerator):
    rcv = prestage.PrestageReceiver("helper", device_id=0, accelerator_backend=accelerator)
    rcv.manager._agent = agent
    rcv.manager._metadata = b"md"
    return rcv


SOURCE = prestage.PrestageSource("s", "w", 0, "src-agent", "10.0.0.1:5000", "", "cuda", [])


class TestReceiver:
    def test_registers_views_of_the_callers_memory(self, monkeypatch, mock_accelerator_backend_cls):
        rcv = _receiver(FakeAgent(), mock_accelerator_backend_cls())

        def view(addr, size, device):
            return SimpleNamespace(
                data_ptr=lambda: addr, numel=lambda: size, device=SimpleNamespace(index=0)
            )

        registered = {}
        monkeypatch.setattr(prestage, "_raw_u8", view)
        monkeypatch.setattr(
            rcv.manager, "register_tensors", lambda t, force_per_tensor: registered.update(t)
        )
        rcv.register([(0x1000, 4096), (0x9000, 512)])
        assert sorted((v.data_ptr(), v.numel()) for v in registered.values()) == [
            (0x1000, 4096),
            (0x9000, 512),
        ]
        assert rcv.timings["regions"] == 2

    def test_refuses_a_view_that_does_not_alias(self, monkeypatch, mock_accelerator_backend_cls):
        rcv = _receiver(FakeAgent(), mock_accelerator_backend_cls())
        monkeypatch.setattr(
            prestage,
            "_raw_u8",
            lambda a, n, d: SimpleNamespace(
                data_ptr=lambda: a + 1, numel=lambda: n, device=SimpleNamespace(index=0)
            ),
        )
        with pytest.raises(prestage.PrestageError, match="alias"):
            rcv.register([(0x1000, 4096)])

    def test_connect_read_close(self, monkeypatch, mock_accelerator_backend_cls):
        agent = FakeAgent()
        rcv = _receiver(agent, mock_accelerator_backend_cls())
        rcv.manager._registered_memory = ["reg-a", "reg-b"]
        rcv.connect(SOURCE)
        with pytest.raises(prestage.PrestageError, match="already connected"):
            rcv.connect(SOURCE)
        monkeypatch.setattr(
            rcv.manager, "execute_read_batch", lambda name, ranges, timeout_seconds: (96, 2, 0.5)
        )
        plan = prestage.ReadPlan([(1, 2, 64, 0), (3, 4, 32, 0)], 2, 96, [], 0)
        assert rcv.read(plan) == (96, 2, 0.5)
        assert rcv.close() == []
        assert agent.removed == ["src-agent"]
        assert agent.deregistered == ["reg-b", "reg-a"]
        assert rcv.manager._agent is None

    def test_read_before_connect_is_refused(self, mock_accelerator_backend_cls):
        rcv = _receiver(FakeAgent(), mock_accelerator_backend_cls())
        with pytest.raises(prestage.PrestageError, match="connect"):
            rcv.read(prestage.ReadPlan([], 0, 0, [], 0))

    def test_a_handshake_that_never_loaded_the_peer_is_clean(self, mock_accelerator_backend_cls):
        # NIXL answers the removal with NOT_FOUND: nothing was loaded.
        rcv = _receiver(FakeAgent(ready=False), mock_accelerator_backend_cls())
        with pytest.raises(TimeoutError):
            rcv.connect(SOURCE, timeout_s=0.05)
        assert rcv.remote_agent is None
        assert rcv.close() == []

    def test_a_refused_peer_removal_does_not_block_the_hand_over(
        self, mock_accelerator_backend_cls
    ):
        # The source never loads this receiver's metadata, so it cannot reach
        # the caller's memory; disconnecting it is best effort, as in RdmaStrategy.
        agent = FakeAgent(fail_remove=True)
        rcv = _receiver(agent, mock_accelerator_backend_cls())
        rcv.manager._registered_memory = ["reg-a"]
        rcv.connect(SOURCE)
        assert rcv.close() == []
        assert agent.deregistered == ["reg-a"]
        assert rcv.manager._agent is None

    def test_every_memory_cleanup_failure_is_reported_on_every_close(
        self, monkeypatch, mock_accelerator_backend_cls
    ):
        rcv = _receiver(FakeAgent(fail_deregister=True), mock_accelerator_backend_cls())
        rcv.manager._registered_memory = ["reg-a"]
        rcv.connect(SOURCE)

        def shutdown():
            raise RuntimeError("agent teardown failed")

        monkeypatch.setattr(rcv.manager, "shutdown", shutdown)
        first = rcv.close()
        assert len(first) == 2
        assert "deregistration refused" in first[0]
        assert "agent teardown failed" in first[1]
        assert set(first) <= set(rcv.close())


def _manager():
    mgr = NixlTransferManager(agent_name="m", device_id=0)
    mgr._agent = MagicMock()
    mgr._agent.get_agent_metadata.return_value = b"metadata"
    mgr._agent.register_memory.side_effect = lambda *args, **kwargs: object()
    return mgr


class TestDeregisterAll:
    def test_releases_every_registration_and_clears_the_catalog(self):
        mgr = _manager()
        w = torch.zeros(8, dtype=torch.float32)
        mgr.register_tensors({"w": w})
        mgr.register_additional_tensors({"draft": torch.ones(8, dtype=torch.float32)})
        handles = list(mgr._registered_memory)
        assert len(handles) == 2

        assert mgr.deregister_all() == []
        released = [c.args[0] for c in mgr._agent.deregister_memory.call_args_list]
        assert released == handles[::-1]
        assert mgr.tensor_descriptors == []
        assert mgr._registered_memory == []

        # A tensor of the old catalog is registered again, not skipped as known.
        mgr.register_additional_tensors({"w": w})
        assert mgr._agent.register_memory.call_count == 3
        assert [d.name for d in mgr.tensor_descriptors] == ["w"]

    def test_a_failure_keeps_the_catalog_and_blocks_registration(self):
        mgr = _manager()
        w = torch.zeros(8, dtype=torch.float32)
        mgr.register_tensors({"w": w})
        mgr.register_additional_tensors({"draft": torch.ones(8, dtype=torch.float32)})
        main_handle, draft_handle = mgr._registered_memory
        mgr._agent.deregister_memory.side_effect = [
            None,
            RuntimeError("deregistration refused"),
            None,
        ]

        errors = mgr.deregister_all()
        assert len(errors) == 1 and "deregistration refused" in errors[0]
        assert mgr._registered_memory == [main_handle]
        # The catalog still holds the tensor the failed handle covers.
        assert mgr._tensors["w"] is w
        with pytest.raises(RuntimeError, match="left registrations behind"):
            mgr.register_tensors({"x": torch.zeros(2)})
        with pytest.raises(RuntimeError, match="left registrations behind"):
            mgr.register_additional_tensors({"x": torch.zeros(2)})

        assert mgr.deregister_all() == []
        assert mgr._registered_memory == []
        assert mgr.tensor_descriptors == []
        mgr.register_tensors({"x": torch.zeros(2)})

    def test_an_interrupt_keeps_the_rest_tracked_and_blocks_registration(self):
        mgr = _manager()
        w = torch.zeros(8, dtype=torch.float32)
        mgr.register_tensors({"w": w})
        mgr.register_additional_tensors({"draft": torch.ones(8, dtype=torch.float32)})
        main_handle, _ = mgr._registered_memory
        mgr._agent.deregister_memory.side_effect = [None, KeyboardInterrupt, None]

        with pytest.raises(KeyboardInterrupt):
            mgr.deregister_all()
        assert mgr._registered_memory == [main_handle]
        assert mgr._tensors["w"] is w
        with pytest.raises(RuntimeError, match="left registrations behind"):
            mgr.register_tensors({"x": torch.zeros(2)})

        assert mgr.deregister_all() == []
        assert mgr._registered_memory == []

    def test_deregister_tensors_waits_for_a_pending_deregistration(self):
        mgr = _manager()
        mgr.register_tensors({"w": torch.zeros(8, dtype=torch.float32)})
        mgr.register_additional_tensors({"draft": torch.ones(8, dtype=torch.float32)})
        mgr._agent.deregister_memory.side_effect = [None, RuntimeError("refused")]

        assert len(mgr.deregister_all()) == 1
        # The draft's handle is already released; it must not be released twice.
        with pytest.raises(RuntimeError, match="left registrations behind"):
            mgr.deregister_tensors(["draft"])
        assert mgr._agent.deregister_memory.call_count == 2

    def test_shutdown_logs_failures_with_their_traceback(self, caplog):
        mgr = _manager()
        mgr.register_tensors({"w": torch.zeros(8, dtype=torch.float32)})
        mgr._agent.deregister_memory.side_effect = RuntimeError("deregistration refused")
        assert len(mgr.deregister_all()) == 1
        with caplog.at_level(logging.WARNING, logger="modelexpress.nixl_transfer"):
            mgr.shutdown()
        assert mgr._agent is None
        assert mgr._deregistration_pending is False
        failures = [r for r in caplog.records if "Failed to deregister" in r.getMessage()]
        assert len(failures) == 1 and failures[0].exc_info is not None

    def test_without_an_agent_is_a_no_op(self):
        assert NixlTransferManager(agent_name="m", device_id=0).deregister_all() == []
