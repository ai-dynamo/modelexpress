# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest

import modelexpress_rl.inference.source.trainer as trainer_module
from modelexpress_rl import ModelExpressGeneratorClient
from modelexpress_rl.control import TrainerMesh
from modelexpress_rl import refit_pb2
from modelexpress_rl.inference.source.trainer import TrainerSourceResolver
from modelexpress_rl.train.manifest import bound_tensor_manifest
from modelexpress_rl.version import TrainerMeshRef, TrainerTensorsMetadata


def _client(service):
    client = ModelExpressGeneratorClient()
    client.model_name = "model"
    client._channel = object()
    client._stub = service
    client._rpc_timeout_seconds = 1
    return client


def _resolver(service=lambda: None):
    return TrainerSourceResolver(service=service, rpc_timeout_seconds=1)


@pytest.mark.parametrize("difference", ["identity", "model", "generation", "missing", "empty_workers"])
def test_fetch_mesh_rejects_mismatched_or_incomplete_mesh(difference):
    mesh = refit_pb2.TrainerMesh(mesh_id="mesh", model_name="model", generation=2)
    mesh.workers["worker"].logical_shard_id = "slot"
    expected = TrainerMeshRef("mesh", 2)
    if difference == "identity":
        mesh.mesh_id = "other"
    elif difference == "model":
        mesh.model_name = "other"
    elif difference == "generation":
        mesh.generation = 3
    elif difference == "missing":
        response = refit_pb2.GetTrainerMeshResponse()
        mesh = None
    elif difference == "empty_workers":
        mesh.ClearField("workers")
    if difference != "missing":
        response = refit_pb2.GetTrainerMeshResponse(mesh=mesh)

    class Service:
        def GetTrainerMesh(self, request, timeout):
            assert request.mesh_id == "mesh"
            assert timeout == 1
            return response

    client = _client(Service())
    with pytest.raises(RuntimeError):
        client._fetch_mesh(expected)


def test_fetch_mesh_returns_exact_requested_mesh():
    mesh = refit_pb2.TrainerMesh(mesh_id="mesh", model_name="model", generation=2)
    mesh.workers["worker"].logical_shard_id = "slot"
    service = type("Service", (), {
        "GetTrainerMesh": lambda _, request, timeout: refit_pb2.GetTrainerMeshResponse(mesh=mesh)
    })()
    assert _client(service)._fetch_mesh(TrainerMeshRef("mesh", 2)) == TrainerMesh(
        mesh_id="mesh", model_name="model", generation=2,
        workers={"worker": TrainerTensorsMetadata("slot", "")},
    )


def _manifest(name="weight", shape=(4,), offset=(0,), shard_shape=(4,)):
    return bound_tensor_manifest([{
        "name": name, "dtype": "torch.bfloat16", "elsize": 2,
        "full_shape": list(shape),
        "shards": [{"shard_offset": list(offset), "shape": list(shard_shape)}],
    }])


class _Channel:
    def __init__(self, endpoint, **_kwargs):
        self.endpoint = endpoint

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _install_workers(monkeypatch, workers):
    requests = []

    class Stub:
        def __init__(self, channel):
            self.worker = workers[channel.endpoint]

        def GetTrainerShardMetadata(self, request, timeout):
            requests.append((self.worker, request, timeout))
            return self.worker(request)

    monkeypatch.setattr(trainer_module.grpc, "insecure_channel", _Channel)
    monkeypatch.setattr(trainer_module.refit_pb2_grpc, "RefitWorkerServiceStub", Stub)
    return requests


def _response(blob, digest=None):
    return refit_pb2.GetTrainerShardMetadataResponse(
        metadata=blob,
        metadata_digest=hashlib.sha256(blob).hexdigest() if digest is None else digest,
    )


def test_logical_sources_fetches_by_slot_digest_and_merges_shards(monkeypatch):
    left, right = _manifest(offset=(0,), shard_shape=(2,)), _manifest(offset=(2,), shard_shape=(2,))
    left_id, right_id = hashlib.sha256(left).hexdigest(), hashlib.sha256(right).hexdigest()
    workers = {
        "left": lambda request: _response(left),
        "right": lambda request: _response(right),
    }
    requests = _install_workers(monkeypatch, workers)
    mesh = refit_pb2.TrainerMesh()
    mesh.workers["w0"].logical_shard_id, mesh.workers["w0"].metadata_endpoint = left_id, "left"
    mesh.workers["w1"].logical_shard_id, mesh.workers["w1"].metadata_endpoint = right_id, "right"

    source = _resolver().logical_sources(mesh)["weight"]
    assert [source.global_shape, source.elsize] == [(4,), 2]
    assert sorted((shard.shard_offset, shard.shape) for shard in source.shards) == [((0,), (2,)), ((2,), (2,))]
    assert sorted(request.metadata_digest for _, request, _ in requests) == sorted([left_id, right_id])


def test_logical_sources_falls_back_to_replica(monkeypatch):
    blob = _manifest()
    slot = hashlib.sha256(blob).hexdigest()

    def unavailable(_request):
        raise RuntimeError("worker unavailable")

    requests = _install_workers(monkeypatch, {"a": unavailable, "b": lambda request: _response(blob)})
    mesh = refit_pb2.TrainerMesh()
    for worker, endpoint in (("w0", "a"), ("w1", "b")):
        mesh.workers[worker].logical_shard_id = slot
        mesh.workers[worker].metadata_endpoint = endpoint
    assert "weight" in _resolver().logical_sources(mesh)
    assert len(requests) == 2


def test_logical_sources_rejects_digest_mismatch(monkeypatch):
    blob = _manifest()
    slot = hashlib.sha256(blob).hexdigest()
    _install_workers(monkeypatch, {"worker": lambda request: _response(blob, "wrong")})
    mesh = refit_pb2.TrainerMesh()
    mesh.workers["w"].logical_shard_id = slot
    mesh.workers["w"].metadata_endpoint = "worker"
    with pytest.raises(RuntimeError, match="no logical manifest"):
        _resolver().logical_sources(mesh)


@pytest.mark.parametrize("blob", [b"{}", b'{"tensors": []}', b'{"tensors": [{"name":"x"}]}'])
def test_logical_sources_rejects_malformed_coverage_schema(monkeypatch, blob):
    slot = hashlib.sha256(blob).hexdigest()
    _install_workers(monkeypatch, {"worker": lambda request: _response(blob)})
    mesh = refit_pb2.TrainerMesh()
    mesh.workers["w"].logical_shard_id = slot
    mesh.workers["w"].metadata_endpoint = "worker"
    with pytest.raises(RuntimeError, match="no logical manifest"):
        _resolver().logical_sources(mesh)
