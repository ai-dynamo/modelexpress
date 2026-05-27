# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""veRL CheckpointEngine backend backed by ModelExpress.

This module is intentionally optional: veRL imports it through
``checkpoint_engine.custom_backend_module``. When veRL is present, importing
this module registers the ``modelexpress`` checkpoint-engine backend.
"""

from __future__ import annotations

import asyncio
import os
import socket
import uuid
from collections.abc import AsyncGenerator, AsyncIterable, Iterable
from typing import Any

import torch

from modelexpress.client import MxClient, MxClientBase
from modelexpress.rl_transfer import (
    RlNixlWeightTransfer,
    build_rl_base_identity,
)


async def _iter_weights(weights: Any) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
    if isinstance(weights, AsyncIterable):
        async for name, tensor in weights:
            yield name, tensor
        return
    for name, tensor in weights:
        yield name, tensor
        await asyncio.sleep(0)


def _model_version_from_global_steps(
    global_steps: int | None,
    current_version: int,
) -> int:
    if global_steps is not None:
        return int(global_steps)
    return current_version + 1


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
        if not self.model_name:
            raise ValueError(
                "ModelExpress veRL backend requires model_name. Set "
                "actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.modelexpress.model_name "
                "or MX_RL_MODEL_NAME."
            )
        self.base_identity = build_rl_base_identity(
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
        self.mx_client = mx_client or MxClient(server_url=server_url)
        self.rank: int | None = None
        self.world_size: int | None = None
        self._local_model_version = -1
        self._worker_id = _worker_id("verl-mx")
        self._transfer = RlNixlWeightTransfer(
            mx_client=self.mx_client,
            base_identity=self.base_identity,
            worker_id=self._worker_id,
            retain_latest_k=retain_latest_k,
            device_id=device_id,
            timeout_seconds=timeout_seconds,
        )

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
        self._transfer.finalize()

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
        self._transfer.publish_tensors(tensors, model_version=model_version)

    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        if self.rank is None:
            raise RuntimeError("init_process_group must run before receive_weights")
        if self.rank <= 0:
            raise RuntimeError("trainer rank cannot receive ModelExpress weights")

        model_version = self._resolve_model_version(global_steps)
        tensors = await self._transfer.receive_tensors(
            model_version=model_version,
            receiver_rank=max(self.rank - 1, 0),
        )
        for name, tensor in tensors:
            yield name, tensor

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
