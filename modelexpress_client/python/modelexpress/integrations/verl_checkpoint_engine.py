# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""veRL CheckpointEngine backend backed by ModelExpress.

This module is intentionally optional: veRL imports it through
``checkpoint_engine.custom_backend_module``. When veRL is present, importing
this module registers the ``modelexpress`` checkpoint-engine backend.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterable, Iterable
from typing import Any

import torch

from modelexpress import p2p_pb2
from modelexpress.client import MxClient, MxClientBase
from modelexpress.nixl_transfer import NixlTransferManager
from modelexpress.rl_metadata import (
    RlSourceMetadata,
    RlSourceRole,
    candidates_from_response,
    select_rl_source_candidates,
    with_rl_source_metadata,
)
from modelexpress.types import TensorDescriptor

logger = logging.getLogger("modelexpress.integrations.verl_checkpoint_engine")

_BACKEND_FRAMEWORKS = {
    "vllm": p2p_pb2.BACKEND_FRAMEWORK_VLLM,
    "sglang": p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
    "trtllm": p2p_pb2.BACKEND_FRAMEWORK_TRT_LLM,
    "trt_llm": p2p_pb2.BACKEND_FRAMEWORK_TRT_LLM,
    "trt-llm": p2p_pb2.BACKEND_FRAMEWORK_TRT_LLM,
}


async def _iter_weights(weights: Any) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
    if isinstance(weights, AsyncIterable):
        async for name, tensor in weights:
            yield name, tensor
        return
    for name, tensor in weights:
        yield name, tensor
        await asyncio.sleep(0)


def _shape_registry_from_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in tensors.items()
    }


def _torch_dtype_from_string(dtype: str) -> torch.dtype:
    name = dtype.removeprefix("torch.")
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported tensor dtype {dtype!r}")
    return value


def _allocate_tensors_from_shape_registry(
    shape_registry: dict[str, Any],
    *,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
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
            dtype=_torch_dtype_from_string(str(dtype)),
            device=device,
        )
    return tensors


def _source_descriptors(worker: "p2p_pb2.WorkerMetadata") -> list[TensorDescriptor]:
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


def _backend_framework_value(value: str | int) -> int:
    if isinstance(value, int):
        return value
    normalized = value.strip().lower()
    if normalized not in _BACKEND_FRAMEWORKS:
        raise ValueError(
            f"unsupported backend_framework {value!r}; expected one of "
            f"{sorted(_BACKEND_FRAMEWORKS)}"
        )
    return _BACKEND_FRAMEWORKS[normalized]


def _model_version_from_global_steps(
    global_steps: int | None,
    current_version: int,
) -> int:
    if global_steps is not None:
        return int(global_steps)
    return current_version + 1


def _build_base_identity(
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
    if not model_name:
        raise ValueError(
            "ModelExpress veRL backend requires model_name. Set "
            "actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.modelexpress.model_name "
            "or MX_RL_MODEL_NAME."
        )
    return p2p_pb2.SourceIdentity(
        mx_version=mx_version,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=model_name,
        backend_framework=_backend_framework_value(backend_framework),
        tensor_parallel_size=int(tensor_parallel_size),
        pipeline_parallel_size=int(pipeline_parallel_size),
        expert_parallel_size=int(expert_parallel_size),
        dtype=dtype,
        quantization=quantization,
        revision=revision,
    )


def _identity_matches_base(
    identity: "p2p_pb2.SourceIdentity",
    base_identity: "p2p_pb2.SourceIdentity",
) -> bool:
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


def _worker_id(prefix: str) -> str:
    return f"{prefix}-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:12]}"


class _ModelExpressCheckpointEngineMixin:
    def __init__(
        self,
        bucket_size: int,
        *,
        server_url: str | None = None,
        model_name: str | None = None,
        mx_version: str = "0.3.0",
        backend_framework: str | int = "vllm",
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        expert_parallel_size: int = 0,
        dtype: str = "bfloat16",
        quantization: str = "",
        revision: str = "",
        retain_latest_k: int = 1,
        device_id: int | None = None,
        timeout_seconds: float = 300.0,
        mx_client: MxClientBase | None = None,
        **_: Any,
    ) -> None:
        self.bucket_size = bucket_size
        self.model_name = model_name or os.environ.get("MX_RL_MODEL_NAME", "")
        self.base_identity = _build_base_identity(
            model_name=self.model_name,
            mx_version=mx_version,
            backend_framework=backend_framework,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            expert_parallel_size=expert_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
        )
        self.retain_latest_k = retain_latest_k
        self.device_id = device_id
        self.timeout_seconds = timeout_seconds
        self.mx_client = mx_client or MxClient(server_url=server_url)
        self.rank: int | None = None
        self.world_size: int | None = None
        self._local_model_version = -1
        self._nixl_manager: NixlTransferManager | None = None
        self._published_tensors: dict[str, torch.Tensor] = {}
        self._mx_source_id: str | None = None
        self._worker_id = _worker_id("verl-mx")

    def prepare(self) -> dict[str, Any]:
        return {}

    @classmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        metadata: list[dict],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        del metadata
        trainer_kwargs = {
            "rank": [0] + [-1] * (trainer_world_size - 1),
            "world_size": [rollout_world_size + 1] * trainer_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
        }
        return trainer_kwargs, rollout_kwargs

    def init_process_group(self, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size

    def finalize(self) -> None:
        self._mark_current_source_stale()
        self._shutdown_nixl_manager()
        self._published_tensors = {}
        self._mx_source_id = None

    async def send_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]] | AsyncIterable[tuple[str, torch.Tensor]],
        global_steps: int | None = None,
    ) -> None:
        if self.rank is None:
            raise RuntimeError("init_process_group must run before send_weights")
        if self.rank < 0:
            async for _name, _tensor in _iter_weights(weights):
                pass
            return

        model_version = self._resolve_model_version(global_steps)
        tensors = await self._materialize_source_tensors(weights)
        if not tensors:
            raise RuntimeError("ModelExpress veRL backend got no tensors to publish")

        device_id = self._resolve_device_id(tensors.values())
        self._mark_current_source_stale()
        self._shutdown_nixl_manager()
        manager = NixlTransferManager(
            agent_name=f"verl-mx-source-{uuid.uuid4().hex[:12]}",
            device_id=device_id,
        )
        manager.initialize()
        manager.register_tensors(tensors)

        self._nixl_manager = manager
        self._published_tensors = tensors

        metadata = RlSourceMetadata(
            model_version=model_version,
            role=RlSourceRole.TRAINER,
            world_size=1,
            retain_latest_k=self.retain_latest_k,
            shape_registry=_shape_registry_from_tensors(tensors),
        )
        identity = with_rl_source_metadata(self.base_identity, metadata)
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=0,
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
            self._worker_id,
        )
        self.mx_client.update_status(
            self._mx_source_id,
            self._worker_id,
            0,
            p2p_pb2.SOURCE_STATUS_READY,
        )
        logger.info(
            "Published ModelExpress veRL weights: model=%s version=%d tensors=%d source_id=%s",
            self.model_name,
            model_version,
            len(tensors),
            self._mx_source_id,
        )

    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        if self.rank is None:
            raise RuntimeError("init_process_group must run before receive_weights")
        if self.rank <= 0:
            raise RuntimeError("trainer rank cannot receive ModelExpress weights")

        model_version = self._resolve_model_version(global_steps)
        candidate = await self._wait_for_source(model_version)
        metadata_resp = self.mx_client.get_metadata(candidate.mx_source_id, candidate.worker_id)
        if not metadata_resp.found:
            raise RuntimeError(
                f"ModelExpress source metadata not found for {candidate.mx_source_id}/"
                f"{candidate.worker_id}"
            )

        device_id = self._resolve_device_id()
        if not candidate.metadata.shape_registry:
            raise RuntimeError(
                f"ModelExpress source {candidate.mx_source_id} has no RL shape registry"
            )
        target_tensors = _allocate_tensors_from_shape_registry(
            dict(candidate.metadata.shape_registry),
            device=f"cuda:{device_id}",
        )
        self._shutdown_nixl_manager()
        manager = NixlTransferManager(
            agent_name=f"verl-mx-target-{uuid.uuid4().hex[:12]}",
            device_id=device_id,
        )
        manager.initialize()
        manager.register_tensors(target_tensors)
        self._nixl_manager = manager

        source_descriptors = _source_descriptors(metadata_resp.worker)
        bytes_transferred, tensor_count, _duration = manager.receive_from_source(
            source_metadata=metadata_resp.worker.nixl_metadata,
            source_tensors=source_descriptors,
            timeout_seconds=self.timeout_seconds,
        )
        logger.info(
            "Received ModelExpress veRL weights: model=%s version=%d tensors=%d bytes=%d",
            self.model_name,
            model_version,
            tensor_count,
            bytes_transferred,
        )
        for source_tensor in source_descriptors:
            tensor = target_tensors.get(source_tensor.name)
            if tensor is not None:
                yield source_tensor.name, tensor

    def _resolve_model_version(self, global_steps: int | None) -> int:
        self._local_model_version = _model_version_from_global_steps(
            global_steps,
            self._local_model_version,
        )
        return self._local_model_version

    async def _materialize_source_tensors(
        self,
        weights: Iterable[tuple[str, torch.Tensor]] | AsyncIterable[tuple[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        tensors = {}
        async for name, tensor in _iter_weights(weights):
            value = tensor.detach()
            if not value.is_cuda:
                raise RuntimeError(f"ModelExpress requires CUDA tensor for {name!r}")
            if not value.is_contiguous():
                value = value.contiguous()
            tensors[name] = value
        return tensors

    def _resolve_device_id(
        self,
        tensors: Iterable[torch.Tensor] | None = None,
    ) -> int:
        if self.device_id is not None:
            return self.device_id
        if tensors is not None:
            for tensor in tensors:
                return tensor.device.index or 0
        return torch.cuda.current_device()

    async def _wait_for_source(self, model_version: int):
        deadline = time.monotonic() + self.timeout_seconds
        last_error: RuntimeError | None = None
        while True:
            try:
                return self._select_source(model_version)
            except RuntimeError as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    raise last_error
                await asyncio.sleep(0.25)

    def _select_source(self, model_version: int):
        response = self.mx_client.list_sources(
            identity=None,
            status_filter=p2p_pb2.SOURCE_STATUS_READY,
        )
        filtered_response = p2p_pb2.ListSourcesResponse()
        for ref in response.instances:
            if not ref.HasField("identity"):
                continue
            if not _identity_matches_base(ref.identity, self.base_identity):
                continue
            filtered_response.instances.append(ref)

        candidates = [
            candidate
            for candidate in candidates_from_response(filtered_response)
            if candidate.model_name == self.model_name
        ]
        selected = select_rl_source_candidates(
            candidates,
            receiver_rank=max((self.rank or 1) - 1, 0),
            model_version=model_version,
            roles=(RlSourceRole.TRAINER,),
            same_rank_only=False,
        )
        if not selected:
            raise RuntimeError(
                f"No ModelExpress RL source found for model={self.model_name!r} "
                f"version={model_version}"
            )
        return selected[0]

    def _mark_current_source_stale(self) -> None:
        if not self._mx_source_id:
            return
        try:
            self.mx_client.update_status(
                self._mx_source_id,
                self._worker_id,
                0,
                p2p_pb2.SOURCE_STATUS_STALE,
            )
        except Exception:
            logger.warning(
                "Failed to mark ModelExpress veRL source stale: source_id=%s",
                self._mx_source_id,
                exc_info=True,
            )

    def _shutdown_nixl_manager(self) -> None:
        if self._nixl_manager is None:
            return
        self._nixl_manager.shutdown()
        self._nixl_manager = None


def register_verl_checkpoint_engine():
    from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry

    class ModelExpressCheckpointEngine(_ModelExpressCheckpointEngineMixin, CheckpointEngine):
        pass

    CheckpointEngineRegistry.register("modelexpress")(ModelExpressCheckpointEngine)
    return ModelExpressCheckpointEngine


try:
    ModelExpressCheckpointEngine = register_verl_checkpoint_engine()
except ModuleNotFoundError as exc:
    if exc.name != "verl":
        raise
    ModelExpressCheckpointEngine = None
