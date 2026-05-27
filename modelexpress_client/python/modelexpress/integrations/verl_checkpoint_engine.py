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
from collections.abc import AsyncGenerator, AsyncIterable, Iterable, Mapping
from typing import Any

import torch

from modelexpress.client import MxClient, MxClientBase
from modelexpress.rl_fanout import RlTreeFanoutPolicy
from modelexpress.rl_metadata import RlSourceRole
from modelexpress.rl_transfer import (
    RlNixlWeightTransfer,
    build_rl_base_identity,
)
from modelexpress.rl_transfer_lease import (
    RlTransferLeaseReportSummary,
    summarize_report_leases,
)
from modelexpress.rl_transfer_report import RlTransferReport
from modelexpress.rl_update_lifecycle import (
    RlWeightUpdateLifecycleHooks,
    iter_weight_update_lifecycle,
)

_TOPOLOGY_BROADCAST = "broadcast"
_TOPOLOGY_RANK_LOCAL = "rank_local"
_TOPOLOGY_TREE_FANOUT = "tree_fanout"
_TOPOLOGIES = {_TOPOLOGY_BROADCAST, _TOPOLOGY_RANK_LOCAL, _TOPOLOGY_TREE_FANOUT}
_SOURCE_ROLE_POLICIES = {
    "trainer_first": (RlSourceRole.TRAINER, RlSourceRole.INFERENCE_REPLICA),
    "replica_first": (RlSourceRole.INFERENCE_REPLICA, RlSourceRole.TRAINER),
    "trainer_only": (RlSourceRole.TRAINER,),
    "replica_only": (RlSourceRole.INFERENCE_REPLICA,),
}


async def _iter_weights(
    weights: Any,
) -> AsyncGenerator[tuple[str, torch.Tensor, Mapping[str, Any]], None]:
    if isinstance(weights, AsyncIterable):
        async for item in weights:
            yield _normalize_weight_item(item)
        return
    for item in weights:
        yield _normalize_weight_item(item)
        await asyncio.sleep(0)


def _normalize_weight_item(
    item: Any,
) -> tuple[str, torch.Tensor, Mapping[str, Any]]:
    try:
        values = tuple(item)
    except TypeError as exc:
        raise ValueError(
            "ModelExpress veRL weights must yield (name, tensor) or "
            "(name, tensor, metadata)"
        ) from exc
    if len(values) == 2:
        name, tensor = values
        metadata: Mapping[str, Any] = {}
    elif len(values) == 3:
        name, tensor, raw_metadata = values
        if raw_metadata is None:
            metadata = {}
        elif isinstance(raw_metadata, Mapping):
            metadata = dict(raw_metadata)
        else:
            raise ValueError(
                "ModelExpress veRL per-tensor metadata must be an object"
            )
    else:
        raise ValueError(
            "ModelExpress veRL weights must yield (name, tensor) or "
            "(name, tensor, metadata)"
        )
    if not isinstance(name, str):
        raise ValueError("ModelExpress veRL weight name must be a string")
    return name, tensor, metadata


def _model_version_from_global_steps(
    global_steps: int | None,
    current_version: int,
) -> int:
    if global_steps is not None:
        return int(global_steps)
    return current_version + 1


def _worker_id(prefix: str) -> str:
    return f"{prefix}-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:12]}"


def _topology_from_metadata(metadata: list[dict]) -> str:
    values = set()
    for item in metadata:
        if isinstance(item, dict) and item.get("modelexpress_topology"):
            values.add(str(item["modelexpress_topology"]).strip().lower())
    if not values:
        return _TOPOLOGY_BROADCAST
    if len(values) > 1:
        raise ValueError(f"conflicting ModelExpress veRL topologies: {sorted(values)}")
    topology = values.pop()
    if topology not in _TOPOLOGIES:
        raise ValueError(
            f"unsupported ModelExpress veRL topology {topology!r}; "
            f"expected one of {sorted(_TOPOLOGIES)}"
        )
    return topology


def _normalize_topology(topology: str | None) -> str:
    value = (topology or os.environ.get("MX_RL_TOPOLOGY", _TOPOLOGY_BROADCAST))
    value = value.strip().lower()
    if value not in _TOPOLOGIES:
        raise ValueError(
            f"unsupported ModelExpress veRL topology {value!r}; "
            f"expected one of {sorted(_TOPOLOGIES)}"
        )
    return value


def _normalize_bool(value: bool | str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"unsupported boolean value {value!r}")


def _normalize_positive_int(value: int | str | None, *, default: int) -> int:
    if value is None:
        return default
    result = int(value)
    if result <= 0:
        raise ValueError("value must be positive")
    return result


def _normalize_source_role_policy(value: str | None) -> tuple[RlSourceRole, ...]:
    normalized = (
        value or os.environ.get("MX_RL_SOURCE_ROLE_POLICY", "trainer_first")
    )
    normalized = normalized.strip().lower()
    try:
        return _SOURCE_ROLE_POLICIES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported ModelExpress veRL source role policy {normalized!r}; "
            f"expected one of {sorted(_SOURCE_ROLE_POLICIES)}"
        ) from exc


def _merge_tensor_metadata(
    configured_metadata: Mapping[str, Mapping[str, Any]] | None,
    inline_metadata: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]] | None:
    if not inline_metadata:
        return configured_metadata
    if not configured_metadata:
        return inline_metadata
    merged = {}
    for name, metadata in configured_metadata.items():
        if not isinstance(metadata, Mapping):
            raise ValueError(
                f"ModelExpress tensor metadata for {name!r} must be an object"
            )
        merged[name] = dict(metadata)
    for name, metadata in inline_metadata.items():
        # Inline metadata comes from the emitted tensor and can refine static config.
        entry = dict(merged.get(name, {}))
        entry.update(metadata)
        merged[name] = entry
    return merged


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
        topology: str | None = None,
        same_rank_only: bool | None = None,
        republish_received: bool | str | None = None,
        retain_sources_on_finalize: bool | str | None = None,
        source_role_policy: str | None = None,
        tree_fanout: int | str | None = None,
        tensor_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        lifecycle_hooks: RlWeightUpdateLifecycleHooks | None = None,
        pause_generation: Any | None = None,
        flush_cache: Any | None = None,
        resume_generation: Any | None = None,
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
        self.topology = _normalize_topology(topology)
        self.same_rank_only = (
            self.topology == _TOPOLOGY_RANK_LOCAL
            if same_rank_only is None
            else same_rank_only
        )
        self.tree_fanout = _normalize_positive_int(
            tree_fanout if tree_fanout is not None else os.environ.get("MX_RL_TREE_FANOUT"),
            default=2,
        )
        republish_value = (
            republish_received
            if republish_received is not None
            else os.environ.get("MX_RL_REPUBLISH_RECEIVED")
        )
        self.republish_received = _normalize_bool(
            republish_value,
            default=self.topology == _TOPOLOGY_TREE_FANOUT,
        )
        if self.topology == _TOPOLOGY_TREE_FANOUT and not self.republish_received:
            raise ValueError("ModelExpress tree_fanout topology requires republish_received")
        retain_sources_value = (
            retain_sources_on_finalize
            if retain_sources_on_finalize is not None
            else os.environ.get("MX_RL_RETAIN_SOURCES_ON_FINALIZE")
        )
        self.retain_sources_on_finalize = _normalize_bool(
            retain_sources_value,
            default=True,
        )
        self.source_roles = _normalize_source_role_policy(source_role_policy)
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
        self._is_trainer: bool | None = None
        self._receiver_rank: int | None = None
        self._source_world_size = 1
        self._replica_world_size: int | None = None
        self._tensor_metadata = tensor_metadata
        self._lifecycle_hooks = lifecycle_hooks or RlWeightUpdateLifecycleHooks(
            pause_generation=pause_generation,
            flush_cache=flush_cache,
            resume_generation=resume_generation,
        )
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
        return {"modelexpress_topology": self.topology}

    @classmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        metadata: list[dict],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        topology = _topology_from_metadata(metadata)
        if topology == _TOPOLOGY_RANK_LOCAL:
            if trainer_world_size != rollout_world_size:
                raise ValueError(
                    "ModelExpress rank_local topology requires equal trainer and rollout world sizes"
                )
            trainer_ranks = list(range(trainer_world_size))
            rollout_ranks = list(range(rollout_world_size))
            trainer_kwargs = {
                "rank": trainer_ranks,
                "world_size": [trainer_world_size] * trainer_world_size,
                "is_trainer": [True] * trainer_world_size,
                "receiver_rank": [None] * trainer_world_size,
                "source_world_size": [trainer_world_size] * trainer_world_size,
            }
            rollout_kwargs = {
                "rank": rollout_ranks,
                "world_size": [trainer_world_size] * rollout_world_size,
                "is_trainer": [False] * rollout_world_size,
                "receiver_rank": rollout_ranks,
                "source_world_size": [trainer_world_size] * rollout_world_size,
            }
            return trainer_kwargs, rollout_kwargs

        trainer_kwargs = {
            "rank": [0] + [-1] * (trainer_world_size - 1),
            "world_size": [rollout_world_size + 1] * trainer_world_size,
            "is_trainer": [True] * trainer_world_size,
            "receiver_rank": [None] * trainer_world_size,
            "source_world_size": [1] * trainer_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "is_trainer": [False] * rollout_world_size,
            "receiver_rank": list(range(rollout_world_size)),
            "source_world_size": [1] * rollout_world_size,
        }
        return trainer_kwargs, rollout_kwargs

    def init_process_group(
        self,
        rank: int,
        world_size: int,
        is_trainer: bool | None = None,
        receiver_rank: int | None = None,
        source_world_size: int = 1,
        replica_world_size: int | None = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self._is_trainer = rank <= 0 if is_trainer is None else is_trainer
        self._receiver_rank = receiver_rank
        self._source_world_size = source_world_size
        self._replica_world_size = replica_world_size

    def finalize(self) -> None:
        if self.retain_sources_on_finalize:
            self._transfer.finalize_receive_state()
        else:
            self._transfer.finalize()

    def mark_current_source_stale(self) -> None:
        self._transfer.mark_current_source_stale()

    @property
    def last_receive_report(self) -> RlTransferReport | None:
        return self._transfer.last_receive_report

    def transfer_lease_summary(
        self,
        *,
        mx_source_id: str = "",
        statuses: Iterable[int] | None = None,
        model_version: int | None = None,
        scope_to_report_model_version: bool = True,
    ) -> RlTransferLeaseReportSummary:
        """Join the last receive report to server leases for the relevant version."""
        report = self._transfer.last_receive_report
        if model_version is None and scope_to_report_model_version and report is not None:
            model_version = report.resolved_model_version
        inventory = self._transfer.list_target_transfer_leases(
            mx_source_id=mx_source_id,
            statuses=statuses,
            model_version=model_version,
        )
        return summarize_report_leases(report, inventory)

    async def send_weights(
        self,
        weights: Iterable[Any] | AsyncIterable[Any],
        global_steps: int | None = None,
    ) -> None:
        if self.rank is None:
            raise RuntimeError("init_process_group must run before send_weights")
        if self._is_trainer is False:
            raise RuntimeError("rollout rank cannot publish ModelExpress weights")
        if self.rank < 0:
            async for _name, _tensor, _metadata in _iter_weights(weights):
                pass
            return

        model_version = self._resolve_send_model_version(global_steps)
        tensors, inline_tensor_metadata = await self._materialize_source_tensors(weights)
        tensor_metadata = _merge_tensor_metadata(
            self._tensor_metadata,
            inline_tensor_metadata,
        )
        self._transfer.publish_tensors(
            tensors,
            model_version=model_version,
            worker_rank=max(self.rank, 0),
            source_world_size=self._source_world_size,
            tensor_metadata=tensor_metadata,
        )

    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        self._raise_if_receive_unready()
        async for name, tensor in iter_weight_update_lifecycle(
            self._receive_weight_items(global_steps=global_steps),
            hooks=self._lifecycle_hooks,
        ):
            yield name, tensor

    async def _receive_weight_items(
        self,
        *,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        self._raise_if_receive_unready()

        model_version = self._resolve_receive_model_version(global_steps)
        receiver_rank = self._resolved_receiver_rank()
        receive_kwargs = {
            "model_version": model_version,
            "receiver_rank": receiver_rank,
            "same_rank_only": self.same_rank_only,
            **self._receive_source_policy_kwargs(receiver_rank),
        }
        if self.republish_received:
            tensors = await self._transfer.receive_tensors_and_publish_replica(
                **receive_kwargs,
                replica_world_size=self._replica_world_size_for_publish(),
            )
        else:
            tensors = await self._transfer.receive_tensors(**receive_kwargs)
        for name, tensor in tensors:
            yield name, tensor

    def _raise_if_receive_unready(self) -> None:
        if self.rank is None:
            raise RuntimeError("init_process_group must run before receive_weights")
        if self._is_trainer:
            raise RuntimeError("trainer rank cannot receive ModelExpress weights")

    def _resolved_receiver_rank(self) -> int:
        if self.rank is None:
            raise RuntimeError("init_process_group must run before receive_weights")
        if self._receiver_rank is not None:
            return self._receiver_rank
        return max(self.rank - 1, 0)

    def _replica_world_size_for_publish(self) -> int:
        if self._replica_world_size is not None:
            return self._replica_world_size
        if self.world_size is None:
            return 1
        if self.topology == _TOPOLOGY_RANK_LOCAL:
            return self.world_size
        return max(self.world_size - self._source_world_size, 1)

    def _receive_source_policy_kwargs(self, receiver_rank: int) -> dict[str, Any]:
        if self.topology != _TOPOLOGY_TREE_FANOUT:
            return {"roles": self.source_roles}
        policy = RlTreeFanoutPolicy(
            receiver_rank=receiver_rank,
            replica_world_size=self._replica_world_size_for_publish(),
            fanout=self.tree_fanout,
        )
        return {
            "roles": policy.roles,
            "same_rank_only": False,
            "source_ranks_by_role": policy.source_ranks_by_role,
            "require_complete_version": policy.parent_replica_rank is None,
        }

    def _resolve_send_model_version(self, global_steps: int | None) -> int:
        self._local_model_version = _model_version_from_global_steps(
            global_steps,
            self._local_model_version,
        )
        return self._local_model_version

    def _resolve_receive_model_version(self, global_steps: int | None) -> int | None:
        if global_steps is None:
            return None
        self._local_model_version = int(global_steps)
        return self._local_model_version

    async def _materialize_source_tensors(
        self,
        weights: Iterable[Any] | AsyncIterable[Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Mapping[str, Any]]]:
        tensors = {}
        tensor_metadata = {}
        async for name, tensor, metadata in _iter_weights(weights):
            value = tensor.detach()
            if not value.is_cuda:
                raise RuntimeError(f"ModelExpress requires CUDA tensor for {name!r}")
            if not value.is_contiguous():
                value = value.contiguous()
            tensors[name] = value
            if metadata:
                tensor_metadata[name] = metadata
        return tensors, tensor_metadata


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
