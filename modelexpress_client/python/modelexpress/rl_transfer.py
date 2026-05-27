# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-agnostic RL weight-transfer helpers backed by MX/NIXL."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Iterable, Sequence
from typing import Any

import torch

from modelexpress import p2p_pb2
from modelexpress.client import MxClientBase
from modelexpress.nixl_transfer import NixlTransferManager
from modelexpress.rl_metadata import (
    RlSourceCandidate,
    RlSourceMetadata,
    RlSourceRole,
    candidates_from_response,
    select_rl_source_candidates,
    with_rl_source_metadata,
)
from modelexpress.types import TensorDescriptor

logger = logging.getLogger("modelexpress.rl_transfer")

_BACKEND_FRAMEWORKS = {
    "vllm": p2p_pb2.BACKEND_FRAMEWORK_VLLM,
    "sglang": p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
    "trtllm": p2p_pb2.BACKEND_FRAMEWORK_TRT_LLM,
    "trt_llm": p2p_pb2.BACKEND_FRAMEWORK_TRT_LLM,
    "trt-llm": p2p_pb2.BACKEND_FRAMEWORK_TRT_LLM,
}


def backend_framework_value(value: str | int) -> int:
    """Return the SourceIdentity enum value for a framework name."""
    if isinstance(value, int):
        return value
    normalized = value.strip().lower()
    if normalized not in _BACKEND_FRAMEWORKS:
        raise ValueError(
            f"unsupported backend_framework {value!r}; expected one of "
            f"{sorted(_BACKEND_FRAMEWORKS)}"
        )
    return _BACKEND_FRAMEWORKS[normalized]


def build_rl_base_identity(
    *,
    model_name: str,
    mx_version: str,
    backend_framework: str | int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
    dtype: str,
    quantization: str,
    revision: str,
) -> "p2p_pb2.SourceIdentity":
    """Build the stable, non-versioned SourceIdentity shared by RL sources."""
    if not model_name:
        raise ValueError("ModelExpress RL transfer requires model_name")
    return p2p_pb2.SourceIdentity(
        mx_version=mx_version,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=model_name,
        backend_framework=backend_framework_value(backend_framework),
        tensor_parallel_size=int(tensor_parallel_size),
        pipeline_parallel_size=int(pipeline_parallel_size),
        expert_parallel_size=int(expert_parallel_size),
        dtype=dtype,
        quantization=quantization,
        revision=revision,
    )


def identity_matches_base(
    identity: "p2p_pb2.SourceIdentity",
    base_identity: "p2p_pb2.SourceIdentity",
) -> bool:
    """Return true when an RL source identity matches the non-versioned base."""
    return (
        identity.mx_version == base_identity.mx_version
        and identity.mx_source_type == base_identity.mx_source_type
        and identity.model_name == base_identity.model_name
        and identity.backend_framework == base_identity.backend_framework
        and identity.tensor_parallel_size == base_identity.tensor_parallel_size
        and identity.pipeline_parallel_size == base_identity.pipeline_parallel_size
        and identity.expert_parallel_size == base_identity.expert_parallel_size
        and identity.dtype == base_identity.dtype
        and identity.quantization == base_identity.quantization
        and identity.revision == base_identity.revision
    )


def shape_registry_from_tensors(
    tensors: dict[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    """Build shape/dtype metadata needed to allocate receive buffers."""
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in tensors.items()
    }


def torch_dtype_from_string(dtype: str) -> torch.dtype:
    """Parse dtype strings emitted by ``str(torch.dtype)``."""
    name = dtype.removeprefix("torch.")
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported tensor dtype {dtype!r}")
    return value


def allocate_tensors_from_shape_registry(
    shape_registry: dict[str, Any],
    *,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Allocate empty tensors from shape registry metadata."""
    tensors = {}
    for name, entry in shape_registry.items():
        if not isinstance(entry, dict):
            raise ValueError(f"shape registry entry for {name!r} must be an object")
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        if not isinstance(shape, list) or dtype is None:
            raise ValueError(f"shape registry entry for {name!r} must include shape and dtype")
        tensors[name] = torch.empty(
            tuple(int(dim) for dim in shape),
            dtype=torch_dtype_from_string(str(dtype)),
            device=device,
        )
    return tensors


def source_descriptors(worker: "p2p_pb2.WorkerMetadata") -> list[TensorDescriptor]:
    """Convert protobuf descriptors into the Python transfer descriptor type."""
    return [
        TensorDescriptor(
            name=tensor.name,
            addr=tensor.addr,
            size=tensor.size,
            device_id=tensor.device_id,
            dtype=tensor.dtype,
        )
        for tensor in worker.tensors
    ]


def candidates_for_base_identity(
    response: "p2p_pb2.ListSourcesResponse",
    base_identity: "p2p_pb2.SourceIdentity",
) -> list[RlSourceCandidate]:
    """Parse RL candidates from a broad ListSources response."""
    filtered_response = p2p_pb2.ListSourcesResponse()
    for ref in response.instances:
        if not ref.HasField("identity"):
            continue
        if not identity_matches_base(ref.identity, base_identity):
            continue
        filtered_response.instances.append(ref)
    return candidates_from_response(filtered_response)


class RlNixlWeightTransfer:
    """Publish and receive versioned RL weights through MX metadata and NIXL."""

    def __init__(
        self,
        *,
        mx_client: MxClientBase,
        base_identity: "p2p_pb2.SourceIdentity",
        worker_id: str,
        retain_latest_k: int = 1,
        device_id: int | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.mx_client = mx_client
        self.base_identity = base_identity
        self.worker_id = worker_id
        self.retain_latest_k = retain_latest_k
        self.device_id = device_id
        self.timeout_seconds = timeout_seconds
        self._nixl_manager: NixlTransferManager | None = None
        self._published_tensors: dict[str, torch.Tensor] = {}
        self._mx_source_id: str | None = None
        self._worker_rank = 0

    def finalize(self) -> None:
        """Tear down transient transfer state and stop advertising this source."""
        self.mark_current_source_stale()
        self.shutdown_nixl_manager()
        self._published_tensors = {}
        self._mx_source_id = None

    def publish_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        model_version: int,
        role: RlSourceRole = RlSourceRole.TRAINER,
        worker_rank: int = 0,
        source_world_size: int = 1,
    ) -> str:
        """Publish CUDA tensors as a READY MX source for one model version."""
        if not tensors:
            raise RuntimeError("ModelExpress RL transfer got no tensors to publish")

        device_id = self.resolve_device_id(tensors.values())
        self.mark_current_source_stale()
        self.shutdown_nixl_manager()
        manager = NixlTransferManager(
            agent_name=f"mx-rl-source-{uuid.uuid4().hex[:12]}",
            device_id=device_id,
        )
        manager.initialize()
        manager.register_tensors(tensors)

        self._nixl_manager = manager
        self._published_tensors = tensors
        self._worker_rank = worker_rank

        metadata = RlSourceMetadata(
            model_version=model_version,
            role=role,
            world_size=source_world_size,
            retain_latest_k=self.retain_latest_k,
            shape_registry=shape_registry_from_tensors(tensors),
        )
        identity = with_rl_source_metadata(self.base_identity, metadata)
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=worker_rank,
            nixl_metadata=manager.nixl_metadata,
            tensors=[
                p2p_pb2.TensorDescriptor(
                    name=tensor.name,
                    addr=tensor.addr,
                    size=tensor.size,
                    device_id=tensor.device_id,
                    dtype=tensor.dtype,
                )
                for tensor in manager.tensor_descriptors
            ],
            status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
        )
        self._mx_source_id = self.mx_client.publish_metadata(
            identity,
            worker,
            self.worker_id,
        )
        self.mx_client.update_status(
            self._mx_source_id,
            self.worker_id,
            worker_rank,
            p2p_pb2.SOURCE_STATUS_READY,
        )
        logger.info(
            "Published ModelExpress RL weights: model=%s version=%d tensors=%d source_id=%s",
            self.base_identity.model_name,
            model_version,
            len(tensors),
            self._mx_source_id,
        )
        return self._mx_source_id

    async def receive_tensors(
        self,
        *,
        model_version: int,
        receiver_rank: int,
        roles: Sequence[RlSourceRole] = (RlSourceRole.TRAINER,),
        same_rank_only: bool = False,
    ) -> list[tuple[str, torch.Tensor]]:
        """Discover, allocate, and pull one model version from a source."""
        candidates = await self.wait_for_sources(
            model_version=model_version,
            receiver_rank=receiver_rank,
            roles=roles,
            same_rank_only=same_rank_only,
        )
        errors = []
        for candidate in candidates:
            try:
                return self._receive_from_candidate(candidate, model_version)
            except Exception as exc:
                errors.append(f"{candidate.mx_source_id}/{candidate.worker_id}: {exc}")
                logger.warning(
                    "ModelExpress RL source transfer failed; trying next candidate: "
                    "source_id=%s worker_id=%s",
                    candidate.mx_source_id,
                    candidate.worker_id,
                    exc_info=True,
                )
        raise RuntimeError(
            "No ModelExpress RL source transfer succeeded for "
            f"model={self.base_identity.model_name!r} version={model_version}; "
            f"errors={errors}"
        )

    def _receive_from_candidate(
        self,
        candidate: RlSourceCandidate,
        model_version: int,
    ) -> list[tuple[str, torch.Tensor]]:
        metadata_resp = self.mx_client.get_metadata(candidate.mx_source_id, candidate.worker_id)
        if not metadata_resp.found:
            raise RuntimeError(
                f"ModelExpress source metadata not found for {candidate.mx_source_id}/"
                f"{candidate.worker_id}"
            )
        if not candidate.metadata.shape_registry:
            raise RuntimeError(
                f"ModelExpress source {candidate.mx_source_id} has no RL shape registry"
            )

        device_id = self.resolve_device_id()
        target_tensors = allocate_tensors_from_shape_registry(
            dict(candidate.metadata.shape_registry),
            device=f"cuda:{device_id}",
        )
        self.shutdown_nixl_manager()
        manager = NixlTransferManager(
            agent_name=f"mx-rl-target-{uuid.uuid4().hex[:12]}",
            device_id=device_id,
        )
        manager.initialize()
        manager.register_tensors(target_tensors)
        self._nixl_manager = manager

        descriptors = source_descriptors(metadata_resp.worker)
        bytes_transferred, tensor_count, _duration = manager.receive_from_source(
            source_metadata=metadata_resp.worker.nixl_metadata,
            source_tensors=descriptors,
            timeout_seconds=self.timeout_seconds,
        )
        logger.info(
            "Received ModelExpress RL weights: model=%s version=%d tensors=%d bytes=%d",
            self.base_identity.model_name,
            model_version,
            tensor_count,
            bytes_transferred,
        )
        return [
            (descriptor.name, tensor)
            for descriptor in descriptors
            if (tensor := target_tensors.get(descriptor.name)) is not None
        ]

    async def wait_for_source(
        self,
        *,
        model_version: int,
        receiver_rank: int,
        roles: Sequence[RlSourceRole] = (RlSourceRole.TRAINER,),
        same_rank_only: bool = False,
    ) -> RlSourceCandidate:
        """Poll MX metadata until the best matching source is visible."""
        return (
            await self.wait_for_sources(
                model_version=model_version,
                receiver_rank=receiver_rank,
                roles=roles,
                same_rank_only=same_rank_only,
            )
        )[0]

    async def wait_for_sources(
        self,
        *,
        model_version: int,
        receiver_rank: int,
        roles: Sequence[RlSourceRole] = (RlSourceRole.TRAINER,),
        same_rank_only: bool = False,
    ) -> list[RlSourceCandidate]:
        """Poll MX metadata until a matching source is visible or timed out."""
        deadline = time.monotonic() + self.timeout_seconds
        last_error: RuntimeError | None = None
        while True:
            try:
                return self.select_sources(
                    model_version=model_version,
                    receiver_rank=receiver_rank,
                    roles=roles,
                    same_rank_only=same_rank_only,
                )
            except RuntimeError as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    raise last_error
                await asyncio.sleep(0.25)

    def select_source(
        self,
        *,
        model_version: int,
        receiver_rank: int,
        roles: Sequence[RlSourceRole] = (RlSourceRole.TRAINER,),
        same_rank_only: bool = False,
    ) -> RlSourceCandidate:
        """Select the best source for a receiver from MX metadata."""
        return self.select_sources(
            model_version=model_version,
            receiver_rank=receiver_rank,
            roles=roles,
            same_rank_only=same_rank_only,
        )[0]

    def select_sources(
        self,
        *,
        model_version: int,
        receiver_rank: int,
        roles: Sequence[RlSourceRole] = (RlSourceRole.TRAINER,),
        same_rank_only: bool = False,
    ) -> list[RlSourceCandidate]:
        """Select the best source for a receiver from MX metadata."""
        response = self.mx_client.list_sources(
            identity=None,
            status_filter=p2p_pb2.SOURCE_STATUS_READY,
        )
        candidates = candidates_for_base_identity(response, self.base_identity)
        candidates = [
            candidate
            for candidate in candidates
            if candidate.model_name == self.base_identity.model_name
        ]
        selected = select_rl_source_candidates(
            candidates,
            receiver_rank=receiver_rank,
            model_version=model_version,
            roles=roles,
            same_rank_only=same_rank_only,
        )
        if not selected:
            raise RuntimeError(
                f"No ModelExpress RL source found for model={self.base_identity.model_name!r} "
                f"version={model_version}"
            )
        return selected

    def resolve_device_id(
        self,
        tensors: Iterable[torch.Tensor] | None = None,
    ) -> int:
        """Resolve the CUDA device used for NIXL registration."""
        if self.device_id is not None:
            return self.device_id
        if tensors is not None:
            for tensor in tensors:
                return tensor.device.index or 0
        return torch.cuda.current_device()

    def mark_current_source_stale(self) -> None:
        """Best-effort STALE transition for a previously published source."""
        if not self._mx_source_id:
            return
        try:
            self.mx_client.update_status(
                self._mx_source_id,
                self.worker_id,
                self._worker_rank,
                p2p_pb2.SOURCE_STATUS_STALE,
            )
        except Exception:
            logger.warning(
                "Failed to mark ModelExpress RL source stale: source_id=%s",
                self._mx_source_id,
                exc_info=True,
            )

    def shutdown_nixl_manager(self) -> None:
        """Release the current NIXL manager, if any."""
        if self._nixl_manager is None:
            return
        self._nixl_manager.shutdown()
        self._nixl_manager = None
