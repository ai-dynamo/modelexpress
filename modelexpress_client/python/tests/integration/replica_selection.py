# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only adapter for exercising the public trainer replica resolver."""
from __future__ import annotations

import hashlib
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import grpc
import torch
from modelexpress.refit.reshard.rendezvous import (
    NIXL_DRAM_MEM_TYPE,
    PublishedShard,
    PublishedTensor,
    wrap_rendezvous_blob,
    unwrap_rendezvous_blob,
)
from modelexpress_rl import refit_pb2, refit_pb2_grpc
from modelexpress_rl.control import WeightVersion, WeightVersionState
from modelexpress_rl.inference.source import TrainerSourceResolver
from modelexpress_rl.train import WeightPayloadFormat


class _Worker(refit_pb2_grpc.RefitWorkerServiceServicer):
    def __init__(self, blob: bytes):
        self.blob = blob

    def GetWeightVersionShardManifest(self, request, context):
        digest = hashlib.sha256(self.blob).hexdigest()
        return refit_pb2.GetWeightVersionShardManifestResponse(
            manifest=self.blob, manifest_digest=digest
        )

    def GetTrainerShardMetadata(self, request, context):
        return refit_pb2.GetTrainerShardMetadataResponse(
            metadata=self.blob, metadata_digest=hashlib.sha256(self.blob).hexdigest()
        )


class _Control:
    def __init__(self, mesh, shards):
        self.mesh = mesh
        self.shards = shards

    def GetTrainerMesh(self, request, timeout=None):
        return refit_pb2.GetTrainerMeshResponse(mesh=self.mesh)

    def ListWeightVersionShards(self, request, timeout=None):
        return refit_pb2.ListWeightVersionShardsResponse(shards=self.shards)


def _tensor_records(publisher: dict[str, Any]) -> list[PublishedTensor]:
    return [PublishedTensor(
        name=entry["name"], dtype="torch." + entry["dtype"],
        elsize=torch.empty((), dtype=getattr(torch, entry["dtype"])).element_size(),
        full_shape=tuple(entry["global_shape"]),
        shards=[PublishedShard(
            agent_name=publisher["session"], device_id=0, addr=entry["address"],
            shard_offset=tuple(entry["offset"]), shape=tuple(entry["shape"]),
            memory_type=NIXL_DRAM_MEM_TYPE,
        )],
    ) for entry in publisher["entries"]]


class ReplicaSelector:
    """Resolve worker candidates through TrainerSourceResolver and real gRPC."""
    def __init__(self, publishers: list[dict[str, Any]]):
        self.publishers = publishers
        self._executor = ThreadPoolExecutor(max_workers=max(4, len(publishers)))
        self._servers = []
        self._worker_by_id: dict[str, dict[str, Any]] = {}
        self._worker_endpoints: dict[str, str] = {}
        self._blobs = {}
        self._mesh = None
        self._start(publishers)
        self._control = _Control(self._mesh, [])
        self._snapshot = None
        options = dict(service=lambda: self._control, rpc_timeout_seconds=2.0)
        if "cached_source" in inspect.signature(TrainerSourceResolver).parameters:
            options["cached_source"] = lambda: self._snapshot
        self._resolver = TrainerSourceResolver(**options)

    @staticmethod
    def _worker_id(pub: dict[str, Any], index: int) -> str:
        return str(pub.get("worker_id", pub.get("session", f"worker-{index}")))

    def _start(self, publishers):
        workers = {}
        for i, pub in enumerate(publishers):
            worker_id = self._worker_id(pub, i)
            slot = str(pub.get("slot", "0"))
            agent_meta = pub.get("metadata", pub.get("nixl_metadata", ""))
            if isinstance(agent_meta, str):
                try:
                    agent_meta = bytes.fromhex(agent_meta)
                except ValueError:
                    agent_meta = agent_meta.encode()
            blob = wrap_rendezvous_blob(
                agent_metadata=agent_meta,
                agent_name=str(pub.get("agent_name", worker_id)),
                metadata_endpoint="127.0.0.1:1",
                tensors=_tensor_records(pub),
            )
            server = grpc.server(self._executor)
            refit_pb2_grpc.add_RefitWorkerServiceServicer_to_server(_Worker(blob), server)
            port = server.add_insecure_port("127.0.0.1:0")
            server.start()
            endpoint = f"127.0.0.1:{port}"
            self._servers.append(server)
            self._worker_by_id[worker_id] = pub
            self._worker_endpoints[worker_id] = endpoint
            self._blobs[worker_id] = blob
            workers[worker_id] = refit_pb2.TrainerTensorsMetadata(
                logical_shard_id=slot, metadata_endpoint=endpoint
            )
        self._mesh = refit_pb2.TrainerMesh(mesh_id="fixture-mesh", model_name="fixture", generation=1, workers=workers)

    def select(self, version: int, selected_workers: list[str]) -> list[dict[str, Any]]:
        shards = []
        for worker_id in selected_workers:
            if worker_id not in self._worker_by_id:
                continue
            slot = self._mesh.workers[worker_id].logical_shard_id
            endpoint = self._worker_endpoints[worker_id]
            blob = self._blobs[worker_id]
            fields = refit_pb2.WeightVersionShard.DESCRIPTOR.fields_by_name
            values = dict(
                version_id=str(version), logical_shard_id=slot, worker_id=worker_id,
                metadata_endpoint=endpoint,
                stable_metadata_digest=hashlib.sha256(blob).hexdigest(),
                manifest_endpoint=endpoint,
                manifest_digest=hashlib.sha256(blob).hexdigest(),
            )
            shard = refit_pb2.WeightVersionShard(**{
                name: value for name, value in values.items() if name in fields
            })
            shards.append(shard)
        self._control.shards = shards
        version_fields = WeightVersion.__dataclass_fields__
        version_values = dict(
            version_id=str(version), model_name="fixture", payload_format=WeightPayloadFormat.FULL_TENSOR,
            layout_signature="fixture", state=WeightVersionState.READY,
            created_at_unix_ms=0, trainer_mesh_id="fixture-mesh",
            trainer_mesh_generation=1,
        )
        model_version = WeightVersion(**{
            key: value for key, value in version_values.items() if key in version_fields
        })
        candidate = next(self._resolver.candidates(model_version), None)
        if candidate is None:
            return []
        selected = getattr(candidate, "snapshot", None)
        sources = selected.shards if selected is not None else candidate.inputs.sources
        self._snapshot = selected if selected is not None else self._snapshot
        result = []
        for source in sources:
            if hasattr(source, "metadata"):
                blob = source.metadata
            elif hasattr(source, "manifest"):
                blob = source.manifest
            else:
                blob = source.transport.manifest
            manifest = unwrap_rendezvous_blob(blob)
            entries = [dict(
                name=tensor.name, dtype=str(tensor.dtype).removeprefix("torch."),
                global_shape=list(tensor.full_shape), shape=list(shard.shape),
                offset=list(shard.shard_offset), address=shard.addr,
            ) for tensor in manifest.tensors for shard in tensor.shards]
            result.append(dict(session=source.worker_id, slot=source.source_slot_id,
                               metadata=manifest.agent_metadata.hex(), entries=entries))
        return result

    def close(self):
        for server in self._servers:
            server.stop(0).wait()
        self._executor.shutdown(wait=True)
        self._servers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
