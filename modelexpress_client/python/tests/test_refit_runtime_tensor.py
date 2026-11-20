# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import modelexpress_rl.inference.methods.runtime_tensor as runtime_module
import pytest
import torch
from modelexpress import p2p_pb2
from modelexpress.types import ManifestMismatchError

from modelexpress_rl import MeshIdentity, WeightPayloadFormat
from modelexpress_rl.inference.methods import (
    LoadTimeTensorNixlUpdateMethod,
    RuntimeTensorNixlUpdateMethod,
)
from modelexpress_rl.inference.plan import (
    GeneratorPeerUpdateSource,
    PreparedRuntimeTensors,
    TrainerSourceSnapshot,
    WeightSource,
)


_IDENTITY = p2p_pb2.SourceIdentity(
    model_name="model",
    mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
    dtype="bfloat16",
)


def _peer(identity=_IDENTITY):
    return GeneratorPeerUpdateSource(
        worker=p2p_pb2.WorkerMetadata(worker_grpc_endpoint="donor:9000"),
        mx_source_id="source-1",
        worker_id="worker-1",
        identity=identity,
    )


class _Manager:
    bounded_staging = False

    def __init__(self, accelerator="cuda"):
        self.accelerator = accelerator
        self.prepared_tensors = None
        self.receive_tensors = None
        self.prepare_calls = 0
        self.received_leases = []
        self.fail_before_start = False
        self.lease = None
        self.removed = []
        self.sources = {}

    def initialize(self):
        pytest.fail("peer refit must borrow the initialized manager")

    def shutdown(self):
        pytest.fail("peer refit must not shut down the borrowed manager")

    def reserve(self, endpoint, source_id, *, worker_id, timeout):
        self.prepare_calls += 1
        self.mx_source_id = source_id
        self.worker_id = worker_id
        self.prepared_tensors = self.expected_tensors
        self.sources = {
            name: torch.arange(tensor.numel()).reshape(tensor.shape).to(tensor.dtype)
            for name, tensor in self.expected_tensors.items()
        }
        manifest = p2p_pb2.GetTensorManifestResponse(
            agent_name="donor", metadata_endpoint="donor:19000",
            accelerator=self.accelerator,
            tensors=[p2p_pb2.TensorDescriptor(
                name=name, addr=tensor.data_ptr(), size=tensor.numel() * tensor.element_size(),
                dtype=str(tensor.dtype), device_id=0,
            ) for name, tensor in self.sources.items()],
        )
        self.lease = SimpleNamespace(manifest=manifest, closed=False)
        lease = self.lease
        lease.close = lambda: setattr(lease, "closed", True)
        return self.lease, 0

    def fetch_remote_and_wait(self, **kwargs):
        self.received_leases.append(self.lease)
        if self.fail_before_start:
            raise RuntimeError("peer disappeared before transfer")

    def receive_from_source(self, *, destination_tensors, on_transfer_start, **kwargs):
        assert kwargs["require_exact_match"] is True
        self.receive_tensors = destination_tensors
        on_transfer_start()
        for name, tensor in destination_tensors.items():
            tensor.copy_(self.sources[name])
        return 16, len(destination_tensors), 0.25

    def remove_remote_agent(self, agent):
        self.removed.append(agent)


def _runtime(monkeypatch, manager, runtime_tensors, accelerator="cuda"):
    manager.expected_tensors = runtime_tensors
    monkeypatch.setattr(runtime_module, "prepare_tensor_read", manager.reserve)
    return RuntimeTensorNixlUpdateMethod(
        manager=manager,
        device_id=0,
        accelerator=accelerator,
        runtime_tensors=runtime_tensors,
    )


def test_load_time_and_runtime_methods_have_disjoint_source_contracts(monkeypatch):
    load_time = LoadTimeTensorNixlUpdateMethod(
        transfer=_Manager(),
    )
    runtime = _runtime(monkeypatch, _Manager(), {})

    assert load_time.capabilities.sources == frozenset({WeightSource.TRAINER})
    assert runtime.capabilities.sources == frozenset({WeightSource.GENERATOR})


def test_runtime_method_defers_direct_receive_until_installation(monkeypatch):
    transfer = _Manager()
    runtime_tensors = {
        "model.weight": torch.empty((2, 3), dtype=torch.bfloat16),
        "model._mx_runtime_buffer": torch.empty(4, dtype=torch.float32),
    }
    method = _runtime(monkeypatch, transfer, runtime_tensors)
    prepared = method.prepare(version=object(), source=_peer())

    assert isinstance(prepared, PreparedRuntimeTensors)
    assert transfer.prepared_tensors is runtime_tensors
    assert transfer.receive_tensors is None
    assert transfer.lease.closed is False
    assert transfer.mx_source_id == "source-1"
    assert transfer.worker_id == "worker-1"
    assert method.capabilities.payload_formats == frozenset(
        {WeightPayloadFormat.FULL_TENSOR}
    )

    with method.installation_context(prepared):
        assert transfer.receive_tensors is runtime_tensors
        assert method.mutated_during_installation_context(prepared) is True

    assert prepared.metrics["bytes_received"] == 16
    assert transfer.lease.closed is True
    for name, tensor in runtime_tensors.items():
        assert torch.equal(tensor, transfer.sources[name])
    assert transfer.removed == ["donor"]
    method.close()


def test_runtime_method_keeps_pretransfer_failure_recoverable(monkeypatch):
    transfer = _Manager()
    transfer.fail_before_start = True
    method = _runtime(monkeypatch, transfer, {"model.weight": torch.empty(1)})
    prepared = method.prepare(version=object(), source=_peer())

    with pytest.raises(RuntimeError, match="before transfer"):
        with method.installation_context(prepared):
            pass

    assert method.mutated_during_installation_context(prepared) is False
    assert transfer.lease.closed is True


def test_runtime_method_reacquires_lease_before_pretransfer_retry(monkeypatch):
    transfer = _Manager()
    transfer.fail_before_start = True
    method = _runtime(monkeypatch, transfer, {"model.weight": torch.empty(1)})
    prepared = method.prepare(version=object(), source=_peer())
    first_lease = transfer.lease

    with pytest.raises(RuntimeError, match="before transfer"):
        with method.installation_context(prepared):
            pass

    transfer.fail_before_start = False
    with method.installation_context(prepared):
        pass

    assert transfer.prepare_calls == 2
    assert transfer.received_leases[0] is first_lease
    assert transfer.received_leases[1] is not first_lease


def test_runtime_method_rejects_retry_after_transfer_started(monkeypatch):
    transfer = _Manager()
    method = _runtime(monkeypatch, transfer, {"model.weight": torch.empty(1)})
    prepared = method.prepare(version=object(), source=_peer())

    with pytest.raises(RuntimeError, match="install failed"):
        with method.installation_context(prepared):
            raise RuntimeError("install failed")

    with pytest.raises(RuntimeError, match="cannot be retried"):
        with method.installation_context(prepared):
            pass

    assert transfer.prepare_calls == 1


def test_runtime_method_releases_reserved_peer_without_apply(monkeypatch):
    transfer = _Manager()
    method = _runtime(monkeypatch, transfer, {"model.weight": torch.empty(1)})
    prepared = method.prepare(version=object(), source=_peer())

    method.release(prepared)

    assert transfer.lease.closed is True


def test_runtime_method_rejects_a_trainer_source(monkeypatch):
    method = _runtime(monkeypatch, _Manager(), {})
    source = TrainerSourceSnapshot(mesh_identity=MeshIdentity("mesh", 1), shards=())

    try:
        method.prepare(version=object(), source=source)
    except TypeError as error:
        assert "generator" in str(error).lower()
    else:
        raise AssertionError("runtime tensor method accepted a trainer source")


def test_load_time_method_rejects_missing_transfer():
    with pytest.raises(ValueError, match="transfer is required"):
        LoadTimeTensorNixlUpdateMethod(transfer=None)


@pytest.mark.parametrize("capture_layout", [None, object(), False])
def test_staged_transfer_rejects_invalid_capture_layout(capture_layout):
    from modelexpress_rl.inference.nixl_staged_transfer import _NixlStagedTransfer

    with pytest.raises(TypeError, match="capture_layout must be callable"):
        _NixlStagedTransfer(
            generator_fingerprint=lambda: (),
            device_id=0,
            device=torch.device("cpu"),
            agent_name="receiver",
            capture_layout=capture_layout,
        )


def test_staged_transfer_requires_capture_layout():
    from modelexpress_rl.inference.nixl_staged_transfer import _NixlStagedTransfer

    with pytest.raises(TypeError, match="capture_layout"):
        _NixlStagedTransfer(
            generator_fingerprint=lambda: (),
            device_id=0,
            device=torch.device("cpu"),
            agent_name="receiver",
        )


@pytest.mark.parametrize("defect", ["empty", "name", "dtype", "size", "endpoint", "agent"])
def test_runtime_method_releases_invalid_donor_before_transfer(monkeypatch, defect):
    manager = _Manager()
    method = _runtime(monkeypatch, manager, {"weight": torch.zeros(4)})
    reserve = manager.reserve

    def invalid_manifest(*args, **kwargs):
        lease, size = reserve(*args, **kwargs)
        if defect == "empty":
            del lease.manifest.tensors[:]
        elif defect in ("name", "dtype", "size"):
            setattr(lease.manifest.tensors[0], defect, {
                "name": "other", "dtype": "torch.float16", "size": 1,
            }[defect])
        elif defect == "endpoint":
            lease.manifest.metadata_endpoint = "invalid"
        else:
            lease.manifest.agent_name = ""
        return lease, size

    monkeypatch.setattr(runtime_module, "prepare_tensor_read", invalid_manifest)
    with pytest.raises((RuntimeError, ManifestMismatchError)):
        method.prepare(version=None, source=_peer())
    assert manager.lease.closed
    assert manager.receive_tensors is None
    assert not manager.received_leases


def _identity(*, quantization="", dtype="bfloat16", mx_source_type=None):
    return p2p_pb2.SourceIdentity(
        model_name="model",
        mx_source_type=(
            p2p_pb2.MX_SOURCE_TYPE_WEIGHTS if mx_source_type is None else mx_source_type
        ),
        quantization=quantization,
        dtype=dtype,
    )


@pytest.mark.parametrize(
    ("target", "donor", "quantization", "dtype"),
    [
        ("cuda", "cuda", "fp8", "bfloat16"),
        ("xpu", "xpu", "fp8", "bfloat16"),
        ("xpu", "cuda", "", "bfloat16"),
        ("cuda", "xpu", "none", "float16"),
    ],
)
def test_runtime_method_accepts_compatible_donor_accelerator(
    monkeypatch, target, donor, quantization, dtype
):
    manager = _Manager(accelerator=donor)
    method = _runtime(monkeypatch, manager, {"weight": torch.zeros(4)}, target)

    prepared = method.prepare(
        version=None,
        source=_peer(_identity(quantization=quantization, dtype=dtype)),
    )

    assert manager.lease.closed is False
    method.release(prepared)
    assert manager.lease.closed is True


@pytest.mark.parametrize(
    ("target", "donor", "identity"),
    [
        pytest.param("xpu", "cuda", _identity(quantization="fp8"), id="cuda-to-xpu-fp8"),
        pytest.param("cuda", "xpu", _identity(quantization="fp8"), id="xpu-to-cuda-fp8"),
        pytest.param("xpu", "", _identity(quantization="fp8"), id="unknown-to-xpu-fp8"),
        pytest.param(
            "xpu", "cuda", _identity(dtype="float8_e4m3fn"), id="cuda-to-xpu-fp8-dtype"
        ),
        pytest.param(
            "xpu",
            "cuda",
            _identity(mx_source_type=p2p_pb2.MX_SOURCE_TYPE_LORA),
            id="cuda-to-xpu-lora",
        ),
    ],
)
def test_runtime_method_rejects_incompatible_donor_before_transfer(
    monkeypatch, target, donor, identity
):
    manager = _Manager(accelerator=donor)
    method = _runtime(monkeypatch, manager, {"weight": torch.zeros(4)}, target)

    with pytest.raises(ManifestMismatchError, match="accelerator is incompatible"):
        method.prepare(version=None, source=_peer(identity))

    assert manager.lease.closed is True
    assert manager.received_leases == []
    assert manager.receive_tensors is None


def test_runtime_method_prepares_next_peer_after_accelerator_rejection(monkeypatch):
    manager = _Manager(accelerator="cuda")
    runtime_tensors = {"weight": torch.zeros(4)}
    method = _runtime(monkeypatch, manager, runtime_tensors, "xpu")
    identity = _identity(quantization="fp8")

    with pytest.raises(ManifestMismatchError):
        method.prepare(version=None, source=_peer(identity))
    rejected_lease = manager.lease

    manager.accelerator = "xpu"
    prepared = method.prepare(version=None, source=_peer(identity))
    with method.installation_context(prepared):
        pass

    assert rejected_lease.closed is True
    assert manager.prepare_calls == 2
    assert manager.received_leases == [manager.lease]
    assert manager.receive_tensors is runtime_tensors
    method.close()


def test_runtime_method_rechecks_accelerator_on_reacquired_lease(monkeypatch):
    manager = _Manager(accelerator="xpu")
    manager.fail_before_start = True
    method = _runtime(monkeypatch, manager, {"weight": torch.zeros(4)}, "xpu")
    prepared = method.prepare(
        version=None, source=_peer(_identity(quantization="fp8"))
    )
    with pytest.raises(RuntimeError, match="before transfer"):
        with method.installation_context(prepared):
            pass

    manager.fail_before_start = False
    manager.accelerator = "cuda"
    with pytest.raises(ManifestMismatchError, match="accelerator is incompatible"):
        with method.installation_context(prepared):
            pass

    assert manager.prepare_calls == 2
    assert manager.lease.closed is True
    assert len(manager.received_leases) == 1
    assert manager.receive_tensors is None
    assert method.mutated_during_installation_context(prepared) is False


def test_generator_resolver_attaches_the_discovery_identity():
    from modelexpress_rl.inference.source.generator import GeneratorSourceResolver

    identity = _identity(quantization="fp8")
    client = SimpleNamespace(
        list_sources=lambda **kwargs: p2p_pb2.ListSourcesResponse(
            instances=[
                p2p_pb2.SourceInstanceRef(
                    mx_source_id="source-1", worker_id="peer", worker_rank=0
                )
            ]
        ),
        get_metadata=lambda **kwargs: p2p_pb2.GetMetadataResponse(
            found=True,
            worker=p2p_pb2.WorkerMetadata(
                worker_rank=0, worker_grpc_endpoint="peer:9000"
            ),
        ),
    )
    built = []

    def build_identity(version_id):
        built.append(version_id)
        return identity

    resolver = GeneratorSourceResolver(
        p2p_client=client,
        worker_id="self",
        worker_rank=0,
        build_identity=build_identity,
    )

    (source,) = resolver.candidates(SimpleNamespace(version_id="v1"))

    assert source.identity is identity
    assert built == ["v1"]


def test_generator_runtime_passes_the_engine_accelerator():
    from modelexpress_rl.inference.runtime import (
        FullTensorEngineCapability,
        _create_runtime_tensor_method,
    )

    capability = FullTensorEngineCapability(
        device_id=0,
        device=torch.device("cuda", 0),
        worker_rank=0,
        capture_layout=lambda: None,
        generator_fingerprint=lambda: (),
        runtime_tensors={"weight": torch.zeros(1)},
        source_worker_id="self",
        unpublish_runtime_tensors=lambda: None,
        publish_runtime_tensors=lambda _version: None,
        build_identity=lambda _version: _IDENTITY,
        nixl_manager=object(),
    )

    method = _create_runtime_tensor_method(capability=capability)

    assert method._accelerator == "cuda"
