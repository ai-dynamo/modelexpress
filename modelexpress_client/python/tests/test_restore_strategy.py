# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GMS snapshot restore strategy chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from modelexpress import p2p_pb2
from modelexpress.gds_loader import (
    _GDS_ALIGNMENT,
    MxDeviceReadTarget,
    MxFileReadSource,
)
from modelexpress.restore_strategy import (
    GmsRestoreContext,
    MxGmsRestoreStrategyChain,
    RestoreStrategyFailed,
)
from modelexpress.restore_strategy import gds_strategy, rdma_strategy
import modelexpress.restore_strategy as restore_strategy


@dataclass
class _FakeAcceleratorBackend:
    rdma_p2p: bool = True
    set_device_calls: list[int] = field(default_factory=list)
    synchronize_calls: list[int | None] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "fake"

    @property
    def torch_device_type(self) -> str:
        return "fake"

    @property
    def nixl_mem_type(self) -> str:
        return "VRAM"

    def set_device(self, device_id: int) -> None:
        self.set_device_calls.append(device_id)

    def current_device(self) -> int:
        return 3

    def synchronize(self, device_id: int | None = None) -> None:
        self.synchronize_calls.append(device_id)

    def empty_cache(self) -> None:
        pass

    def torch_device(self, device_id: int):
        raise AssertionError("torch_device should not be used by these tests")

    def is_accel_tensor(self, tensor) -> bool:
        return False

    def supports_rdma_p2p(self) -> bool:
        return self.rdma_p2p

    def supports_pool_reg(self) -> bool:
        return False

    def supports_vmm(self) -> bool:
        return False

    def supports_gds(self) -> bool:
        return True


def _context(
    *,
    endpoint: str | None = "http://mx-metadata:8001",
    accelerator: _FakeAcceleratorBackend | None = None,
) -> GmsRestoreContext:
    backend_config = {}
    if endpoint is not None:
        backend_config["mx_p2p_metadata_endpoint"] = endpoint
    source = MxFileReadSource(
        allocation_id="allocation-0",
        file_path="/checkpoint/shard_0000.bin",
        file_offset=4096,
        byte_count=8192,
    )
    target = MxDeviceReadTarget(
        allocation_id="allocation-0",
        va=0x100000,
        device=3,
        byte_count=8192,
    )
    return GmsRestoreContext(
        sources=[source],
        targets={source.allocation_id: target},
        grouped_sources={source.file_path: [(source, target)]},
        device=3,
        max_workers=7,
        backend_config=backend_config,
        gds_chunk_size=4096,
        gds_max_inflight=2,
        accelerator_backend=accelerator or _FakeAcceleratorBackend(),
    )


def _context_from_env(**gds_config) -> GmsRestoreContext:
    ctx = _context()
    return GmsRestoreContext.from_env(
        sources=ctx.sources,
        targets=ctx.targets,
        grouped_sources=ctx.grouped_sources,
        device=ctx.device,
        max_workers=ctx.max_workers,
        backend_config=ctx.backend_config,
        accelerator_backend=ctx.accelerator_backend,
        **gds_config,
    )


@pytest.fixture
def gds_restore_env(monkeypatch):
    monkeypatch.delenv("MX_GDS_CHUNK_SIZE_BYTES", raising=False)
    monkeypatch.delenv("MX_GDS_MAX_INFLIGHT_BATCHES", raising=False)
    return monkeypatch


def test_gms_restore_context_from_env_uses_defaults_when_unset(gds_restore_env):
    gds_restore_env.delenv("MX_GDS_CHUNK_SIZE_BYTES", raising=False)
    gds_restore_env.delenv("MX_GDS_MAX_INFLIGHT_BATCHES", raising=False)

    ctx = _context_from_env()

    assert ctx.gds_chunk_size is None
    assert ctx.gds_max_inflight == ctx.max_workers


def test_gms_restore_context_from_env_treats_blank_values_as_unset(
    gds_restore_env,
):
    gds_restore_env.setenv("MX_GDS_CHUNK_SIZE_BYTES", "   ")
    gds_restore_env.setenv("MX_GDS_MAX_INFLIGHT_BATCHES", "   ")

    ctx = _context_from_env()

    assert ctx.gds_chunk_size is None
    assert ctx.gds_max_inflight == ctx.max_workers


def test_gms_restore_context_from_env_reads_valid_values(gds_restore_env):
    chunk_size = 2 * _GDS_ALIGNMENT
    gds_restore_env.setenv("MX_GDS_CHUNK_SIZE_BYTES", str(chunk_size))
    gds_restore_env.setenv("MX_GDS_MAX_INFLIGHT_BATCHES", "3")

    ctx = _context_from_env()

    assert ctx.gds_chunk_size == chunk_size
    assert ctx.gds_max_inflight == 3

@pytest.mark.parametrize(
    "value",
    ["not-an-int", "0", str(_GDS_ALIGNMENT + 1)],
)
def test_gms_restore_context_from_env_rejects_invalid_chunk_size(
    gds_restore_env,
    value,
):
    gds_restore_env.setenv("MX_GDS_CHUNK_SIZE_BYTES", value)

    with pytest.raises(ValueError):
        _context_from_env()


@pytest.mark.parametrize("value", ["0", "not-an-int"])
def test_gms_restore_context_from_env_rejects_invalid_max_inflight(
    gds_restore_env,
    value,
):
    gds_restore_env.setenv("MX_GDS_MAX_INFLIGHT_BATCHES", value)

    with pytest.raises(ValueError):
        _context_from_env()


def test_gms_restore_context_explicit_values_override_environment(
    gds_restore_env,
):
    gds_restore_env.setenv("MX_GDS_CHUNK_SIZE_BYTES", "not-an-int")
    gds_restore_env.setenv("MX_GDS_MAX_INFLIGHT_BATCHES", "not-an-int")

    ctx = _context_from_env(
        gds_chunk_size=str(2 * _GDS_ALIGNMENT),
        gds_max_inflight="4",
    )

    assert ctx.gds_chunk_size == 2 * _GDS_ALIGNMENT
    assert ctx.gds_max_inflight == 4


class _Strategy:
    def __init__(
        self,
        name: str,
        events: list[tuple[str, str]],
        *,
        available: bool = True,
        result: dict[str, object] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self._events = events
        self._available = available
        self._result = result or {}
        self._error = error

    def is_available(self, ctx) -> bool:
        self._events.append((self.name, "available"))
        return self._available

    def restore(self, ctx) -> dict[str, object]:
        self._events.append((self.name, "restore"))
        if self._error is not None:
            raise self._error
        return dict(self._result)

    def rollback(self, ctx) -> None:
        self._events.append((self.name, "rollback"))


def test_restore_strategy_policy_order_is_fixed():
    assert [
        strategy.name for strategy in restore_strategy._build_restore_strategies()
    ] == ["rdma", "gds", "posix"]


def test_chain_falls_through_declared_failure(monkeypatch):
    events: list[tuple[str, str]] = []
    strategies = [
        _Strategy(
            "rdma",
            events,
            error=RestoreStrategyFailed("live source miss", mutated=True),
        ),
        _Strategy("gds", events, result={"total_bytes": 8192}),
        _Strategy("posix", events, result={"total_bytes": 8192}),
    ]
    monkeypatch.setattr(
        restore_strategy,
        "_build_restore_strategies",
        lambda: strategies,
    )

    result = MxGmsRestoreStrategyChain.run(SimpleNamespace())

    assert result == {"total_bytes": 8192, "selected_strategy": "gds"}
    assert events == [
        ("rdma", "available"),
        ("gds", "available"),
        ("posix", "available"),
        ("rdma", "restore"),
        ("rdma", "rollback"),
        ("gds", "restore"),
    ]


def test_chain_stops_after_first_success(monkeypatch):
    events: list[tuple[str, str]] = []
    strategies = [
        _Strategy("rdma", events, result={"source_count": 1}),
        _Strategy("gds", events, result={"source_count": 1}),
        _Strategy("posix", events, result={"source_count": 1}),
    ]
    monkeypatch.setattr(
        restore_strategy,
        "_build_restore_strategies",
        lambda: strategies,
    )

    result = MxGmsRestoreStrategyChain.run(SimpleNamespace())

    assert result == {"source_count": 1, "selected_strategy": "rdma"}
    assert ("gds", "restore") not in events
    assert ("posix", "restore") not in events


def test_chain_aggregates_unavailable_and_declared_failures(monkeypatch):
    events: list[tuple[str, str]] = []
    strategies = [
        _Strategy("rdma", events, available=False),
        _Strategy(
            "gds",
            events,
            error=RestoreStrategyFailed("read failed", mutated=True),
        ),
        _Strategy(
            "posix",
            events,
            error=RestoreStrategyFailed("staging failed", mutated=False),
        ),
    ]
    monkeypatch.setattr(
        restore_strategy,
        "_build_restore_strategies",
        lambda: strategies,
    )

    with pytest.raises(RuntimeError, match="no MX restore strategy succeeded") as exc:
        MxGmsRestoreStrategyChain.run(SimpleNamespace())

    message = str(exc.value)
    assert "rdma: unavailable" in message
    assert "gds: read failed (mutated=True)" in message
    assert "posix: staging failed (mutated=False)" in message


def test_gds_strategy_uses_context_config_and_shuts_down(monkeypatch):
    calls = []

    class Loader:
        def __init__(self, accelerator_backend) -> None:
            calls.append(("init", accelerator_backend))

        def restore_gms_snapshot(self, **kwargs):
            calls.append(("restore", kwargs))
            return {"total_bytes": 8192, "selected_strategy": "gds"}

        def shutdown(self) -> None:
            calls.append(("shutdown", None))

    monkeypatch.setattr(gds_strategy, "is_gds_available", lambda: True)
    monkeypatch.setattr(gds_strategy, "MxGdsLoader", Loader)
    ctx = _context()
    strategy = gds_strategy.GdsRestoreStrategy()

    assert strategy.is_available(ctx)
    result = strategy.restore(ctx)

    assert result["total_bytes"] == 8192
    assert calls[0] == ("init", ctx.accelerator_backend)
    assert calls[1] == (
        "restore",
        {
            "grouped_sources": ctx.grouped_sources,
            "device": 3,
            "max_workers": 7,
            "chunk_size_bytes": 4096,
            "max_inflight_batches": 2,
        },
    )
    assert calls[2] == ("shutdown", None)


def test_gds_strategy_maps_failure_as_mutated_and_shuts_down(monkeypatch):
    shutdown_calls = []

    class Loader:
        def __init__(self, accelerator_backend) -> None:
            pass

        def restore_gms_snapshot(self, **kwargs):
            raise RuntimeError("GDS read failed")

        def shutdown(self) -> None:
            shutdown_calls.append(True)

    monkeypatch.setattr(gds_strategy, "MxGdsLoader", Loader)

    with pytest.raises(RestoreStrategyFailed, match="GDS read failed") as exc:
        gds_strategy.GdsRestoreStrategy().restore(_context())

    assert exc.value.mutated is True
    assert shutdown_calls == [True]


class _FakeMetadataClient:
    def __init__(
        self,
        instances: list[p2p_pb2.SourceInstanceRef],
        metadata: dict[tuple[str, str], p2p_pb2.GetMetadataResponse],
    ) -> None:
        self.instances = instances
        self.metadata = metadata
        self.list_calls = []
        self.get_calls = []
        self.closed = False

    def list_sources(self, *, identity, status_filter):
        self.list_calls.append((identity, status_filter))
        return p2p_pb2.ListSourcesResponse(instances=self.instances)

    def get_metadata(self, *, mx_source_id, worker_id):
        self.get_calls.append((mx_source_id, worker_id))
        return self.metadata[(mx_source_id, worker_id)]

    def close(self) -> None:
        self.closed = True


class _FakeNixlManager:
    instances: list[_FakeNixlManager] = []

    def __init__(
        self,
        *,
        agent_name,
        device_id,
        accelerator_backend,
    ) -> None:
        self.agent_name = agent_name
        self.device_id = device_id
        self.accelerator_backend = accelerator_backend
        self.initialized = False
        self.regions = None
        self.receive_call = None
        self.fetch_call = None
        self.shutdown_calls = 0
        self.__class__.instances.append(self)

    def initialize(self) -> None:
        self.initialized = True

    def register_device_regions(self, regions) -> bytes:
        self.regions = regions
        return b"target-metadata"

    def fetch_remote_and_wait(self, **kwargs) -> None:
        self.fetch_call = kwargs

    def receive_from_source(self, **kwargs):
        self.receive_call = kwargs
        total = sum(region.size for region in kwargs["source_tensors"])
        return total, len(kwargs["source_tensors"]), 0.01

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _ready_source(
    *, accelerator: str = "fake"
) -> tuple[
    p2p_pb2.SourceInstanceRef,
    p2p_pb2.GetMetadataResponse,
]:
    instance = p2p_pb2.SourceInstanceRef(
        mx_source_id="source-id",
        worker_id="worker-3",
        model_name="/checkpoint",
        worker_rank=3,
        accelerator=accelerator,
    )
    worker = p2p_pb2.WorkerMetadata(
        worker_rank=3,
        nixl_metadata=b"source-metadata",
        status=p2p_pb2.SOURCE_STATUS_READY,
        accelerator=accelerator,
        tensor_source=p2p_pb2.TensorSourceMetadata(
            tensors=[
                p2p_pb2.TensorDescriptor(
                    name="allocation-0",
                    addr=0x200000,
                    size=8192,
                    device_id=3,
                    dtype="uint8",
                )
            ]
        ),
    )
    response = p2p_pb2.GetMetadataResponse(
        found=True,
        worker=worker,
        mx_source_id=instance.mx_source_id,
        worker_id=instance.worker_id,
    )
    return instance, response


def _install_fake_rdma(
    monkeypatch,
    client: _FakeMetadataClient,
) -> list[dict[str, object]]:
    factory_calls: list[dict[str, object]] = []

    def create_client(**kwargs):
        factory_calls.append(kwargs)
        return client

    _FakeNixlManager.instances = []
    monkeypatch.setattr(rdma_strategy, "is_nixl_available", lambda: True)
    monkeypatch.setattr(rdma_strategy, "create_metadata_client", create_client)
    monkeypatch.setattr(rdma_strategy, "NixlTransferManager", _FakeNixlManager)
    monkeypatch.delenv("MODEL_EXPRESS_URL", raising=False)
    monkeypatch.delenv("MX_SERVER_ADDRESS", raising=False)
    return factory_calls


def test_rdma_discovers_rank_matched_source_and_restores(monkeypatch):
    instance, metadata = _ready_source()
    wrong_rank = p2p_pb2.SourceInstanceRef(
        mx_source_id="wrong-rank",
        worker_id="worker-1",
        worker_rank=1,
    )
    client = _FakeMetadataClient(
        [wrong_rank, instance],
        {(instance.mx_source_id, instance.worker_id): metadata},
    )
    factory_calls = _install_fake_rdma(monkeypatch, client)
    ctx = _context()

    result = MxGmsRestoreStrategyChain.run(ctx)

    assert result["selected_strategy"] == "rdma"
    assert result["total_bytes"] == 8192
    assert factory_calls == [
        {"worker_rank": 3, "server_url": "http://mx-metadata:8001"}
    ]
    identity, status_filter = client.list_calls[0]
    assert identity.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_GMS_WEIGHT_SNAPSHOT
    assert identity.model_name == "/checkpoint"
    assert identity.revision == ""
    assert len(identity.extra_parameters["gms_snapshot_extents"]) == 64
    assert status_filter == p2p_pb2.SOURCE_STATUS_READY
    assert client.get_calls == [("source-id", "worker-3")]
    assert client.closed

    manager = _FakeNixlManager.instances[0]
    assert manager.initialized
    assert manager.shutdown_calls == 1
    assert set(manager.regions) == {"allocation-0"}
    target_region = manager.regions["allocation-0"]
    assert target_region.name == "allocation-0"
    assert target_region.addr == 0x100000
    assert target_region.size == 8192
    assert target_region.device_id == 3
    assert target_region.dtype == "uint8"
    assert manager.receive_call["source_metadata"] == b"source-metadata"
    assert manager.receive_call["source_tensors"][0].name == "allocation-0"


def test_rdma_reuses_atomic_worker_manifest_for_k8s_service(
    monkeypatch,
):
    instance, metadata = _ready_source()
    metadata.worker.worker_grpc_endpoint = "mx-source.default.svc:6555"
    metadata.worker.metadata_endpoint = "10.0.0.3:5555"
    metadata.worker.agent_name = "source-agent"
    client = _FakeMetadataClient(
        [instance],
        {(instance.mx_source_id, instance.worker_id): metadata},
    )
    _install_fake_rdma(monkeypatch, client)

    from modelexpress.metadata import worker_server

    monkeypatch.setattr(
        worker_server,
        "fetch_tensor_manifest",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("metadata manifest must not be fetched twice")
        ),
    )

    result = MxGmsRestoreStrategyChain.run(_context())

    assert result["selected_strategy"] == "rdma"
    manager = _FakeNixlManager.instances[0]
    assert manager.fetch_call == {
        "remote_agent_name": "source-agent",
        "ip": "10.0.0.3",
        "port": 5555,
    }
    assert manager.receive_call["remote_agent_name"] == "source-agent"


def test_rdma_fetches_worker_manifest_for_selected_generation(monkeypatch):
    from modelexpress.metadata import worker_server

    worker = p2p_pb2.WorkerMetadata(
        worker_rank=3,
        worker_grpc_endpoint="mx-source.default.svc:6555",
    )
    descriptor = p2p_pb2.TensorDescriptor(
        name="allocation-0",
        addr=0x200000,
        size=8192,
        device_id=3,
        dtype="uint8",
    )
    calls = []

    def fetch_manifest(**kwargs):
        calls.append(kwargs)
        return [descriptor], descriptor.ByteSize()

    monkeypatch.setattr(worker_server, "fetch_tensor_manifest", fetch_manifest)

    regions = rdma_strategy._source_regions(
        worker,
        "source-id",
        "worker-3",
    )

    assert calls == [
        {
            "endpoint": "mx-source.default.svc:6555",
            "mx_source_id": "source-id",
            "worker_id": "worker-3",
        }
    ]
    assert [region.name for region in regions] == ["allocation-0"]


def test_rdma_filters_incompatible_accelerator_before_metadata_fetch():
    instance, metadata = _ready_source(accelerator="other")
    client = _FakeMetadataClient(
        [instance],
        {(instance.mx_source_id, instance.worker_id): metadata},
    )

    source = rdma_strategy.RdmaRestoreStrategy()._find_compatible_source(
        _context(),
        client,
    )

    assert source is None
    assert client.get_calls == []


def test_rdma_rechecks_authoritative_worker_accelerator(monkeypatch):
    instance, metadata = _ready_source()
    instance.accelerator = ""
    metadata.worker.accelerator = "other"
    client = _FakeMetadataClient(
        [instance],
        {(instance.mx_source_id, instance.worker_id): metadata},
    )
    monkeypatch.setattr(
        rdma_strategy,
        "_source_regions",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("incompatible worker manifest must not be fetched")
        ),
    )

    source = rdma_strategy.RdmaRestoreStrategy()._find_compatible_source(
        _context(),
        client,
    )

    assert source is None
    assert client.get_calls == [("source-id", "worker-3")]


def test_checkpoint_extent_digest_changes_with_source_extents():
    original = _context()
    changed = _context()
    changed.sources[0] = MxFileReadSource(
        allocation_id="allocation-0",
        file_path="/checkpoint/shard_0001.bin",
        file_offset=4096,
        byte_count=8192,
    )

    original_identity = rdma_strategy._build_snapshot_identity(original)
    changed_identity = rdma_strategy._build_snapshot_identity(changed)

    assert (
        original_identity.extra_parameters["gms_snapshot_extents"]
        != changed_identity.extra_parameters["gms_snapshot_extents"]
    )


def test_rdma_maps_receive_failure_as_mutated(monkeypatch):
    instance, metadata = _ready_source()
    client = _FakeMetadataClient(
        [instance],
        {(instance.mx_source_id, instance.worker_id): metadata},
    )
    _install_fake_rdma(monkeypatch, client)

    def fail_receive(self, **kwargs):
        raise RuntimeError("receive failed")

    monkeypatch.setattr(_FakeNixlManager, "receive_from_source", fail_receive)

    strategy = rdma_strategy.RdmaRestoreStrategy()
    with pytest.raises(RestoreStrategyFailed, match="receive failed") as exc_info:
        strategy.restore(_context())

    manager = _FakeNixlManager.instances[0]
    assert exc_info.value.mutated is True
    assert manager.shutdown_calls == 1


def test_no_compatible_rdma_source_falls_through_to_gds(monkeypatch):
    client = _FakeMetadataClient([], {})
    _install_fake_rdma(monkeypatch, client)
    gds_calls = []

    class Loader:
        def __init__(self, accelerator_backend) -> None:
            pass

        def restore_gms_snapshot(self, **kwargs):
            gds_calls.append(kwargs)
            return {"total_bytes": 8192}

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(gds_strategy, "is_gds_available", lambda: True)
    monkeypatch.setattr(gds_strategy, "MxGdsLoader", Loader)

    result = MxGmsRestoreStrategyChain.run(_context())

    assert result["selected_strategy"] == "gds"
    assert len(gds_calls) == 1
    assert client.closed
    assert _FakeNixlManager.instances == []


def test_rdma_endpoint_unset_is_unavailable(monkeypatch):
    monkeypatch.delenv("MODEL_EXPRESS_URL", raising=False)
    monkeypatch.delenv("MX_SERVER_ADDRESS", raising=False)
    monkeypatch.setattr(rdma_strategy, "is_nixl_available", lambda: True)

    assert not rdma_strategy.RdmaRestoreStrategy().is_available(
        _context(endpoint=None)
    )


def test_rdma_requires_nixl_and_rdma_capability(monkeypatch):
    monkeypatch.setattr(rdma_strategy, "is_nixl_available", lambda: False)
    assert not rdma_strategy.RdmaRestoreStrategy().is_available(_context())

    monkeypatch.setattr(rdma_strategy, "is_nixl_available", lambda: True)
    assert not rdma_strategy.RdmaRestoreStrategy().is_available(
        _context(accelerator=_FakeAcceleratorBackend(rdma_p2p=False))
    )
