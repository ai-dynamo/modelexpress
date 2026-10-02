# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep composition and lifecycle for trainer publication."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch

from .. import envs as rl_envs
from .. import timing
from ..s3 import S3Client
from ..utils import make_tensor_reader
from ..version import WeightVersionRef
from .adapter import (
    StagedWeightVersionShardData,
    TrainerStagingMode,
    WeightPayloadFormat,
)
from .context import TrainerEngineContext
from .engines import _create_trainer_adapter
from .methods import (
    CanonicalDeltaPublicationMethod,
    FullTensorNixlPublicationMethod,
    StagedCanonicalDelta,
    StagedFullCheckpoint,
)
from .resources import _TrainerResources

if TYPE_CHECKING:
    from .. import refit_pb2_grpc
    from .client import ObjectStorageConfig

PublicationArtifact = (
    StagedWeightVersionShardData | StagedCanonicalDelta | StagedFullCheckpoint
)

logger = logging.getLogger("modelexpress_rl.train.runtime")


def _flatten_timing(payload: dict[str, Any]) -> dict[str, int | float]:
    """Flatten one refit record into scalar metrics a framework can chart.

    Stages with no measurements are dropped rather than reported as zero, since
    a stage that did not run and a stage that took no time are different facts
    and only one of them is worth a point on a graph.
    """
    metrics: dict[str, int | float] = {
        "trainer_refit_e2e_s": float(payload["e2e_ms"]) / 1000.0,
    }
    for stage, values in payload["stages"].items():
        if values["count"]:
            metrics[f"{stage}_s"] = float(values["duration_ms"]) / 1000.0
        for name, value in values.get("metadata", {}).items():
            if isinstance(value, bool):
                metrics[name] = int(value)
            elif isinstance(value, (int, float)):
                metrics[name] = value
    return metrics


class TrainerRuntime:
    """Own one publication method and all transport resources it requires."""

    def __init__(
        self,
        *,
        method: FullTensorNixlPublicationMethod
        | CanonicalDeltaPublicationMethod
        | None,
        resources: _TrainerResources | None,
        full_tensor_factory: Callable[[], FullTensorNixlPublicationMethod]
        | None = None,
    ) -> None:
        self.method = method
        self.resources = resources
        self._full_tensor_factory = full_tensor_factory
        self.worker_endpoint = (
            resources.worker_endpoint if resources is not None else ""
        )
        self._bound_tensors: Any | None = None
        self._last_full_tensor_metrics: dict[str, int | float] = {}
        self._closed = False

    @classmethod
    def initialize(
        cls,
        *,
        engine_context: TrainerEngineContext | None,
        device_id: int | None,
        agent_name: str | None,
        model_name: str,
        staging_mode: TrainerStagingMode,
        payload_format: WeightPayloadFormat,
        worker_id: str,
        object_storage: ObjectStorageConfig | None,
        process_group: Any,
        service: Callable[[], refit_pb2_grpc.RefitServiceStub],
        rpc_timeout_seconds: float,
    ) -> TrainerRuntime:
        if object_storage is not None:
            read_seed_tensor, _ = make_tensor_reader(
                object_storage.seed_checkpoint_path
            )
            s3 = S3Client(
                endpoint_url=object_storage.endpoint_url,
                region_name=object_storage.region_name,
            )
            try:
                method = CanonicalDeltaPublicationMethod(
                    config=object_storage,
                    model_name=model_name,
                    service=service,
                    rpc_timeout_seconds=rpc_timeout_seconds,
                    process_group=process_group,
                    read_seed_tensor=read_seed_tensor,
                    s3=s3,
                    clock=lambda: perf_counter(),
                )
            except Exception:
                s3.close()
                raise
            return cls(method=method, resources=None)

        if engine_context is None:
            raise ValueError("engine_context is required for full-tensor publication")
        resolved_device_id = device_id
        if resolved_device_id is None:
            resolved_device_id = rl_envs.LOCAL_RANK
        if resolved_device_id is None:
            raise ValueError("config.device_id or LOCAL_RANK is required")
        resources = _TrainerResources.initialize(
            device_id=resolved_device_id,
            agent_name=agent_name,
        )
        try:
            manager = resources.manager
            if manager.listen_port is None:
                raise ValueError("NIXL manager must have a metadata listen port")
            from modelexpress import envs

            host = envs.MX_WORKER_HOST
            if not host.strip():
                raise ValueError("MX_WORKER_HOST is required")

            def create_full_tensor() -> FullTensorNixlPublicationMethod:
                adapter = _create_trainer_adapter(
                    engine_context,
                    manager=manager,
                    nixl_metadata_endpoint=f"{host}:{manager.listen_port}",
                )
                return FullTensorNixlPublicationMethod(
                    adapter=adapter,
                    staging_mode=staging_mode,
                    payload_format=payload_format,
                    manifest_publisher=resources.manifest_service,
                    service=service,
                    worker_id=worker_id,
                    rpc_timeout_seconds=rpc_timeout_seconds,
                )

            return cls(
                method=None,
                resources=resources,
                full_tensor_factory=create_full_tensor,
            )
        except Exception:
            resources.close()
            raise

    @property
    def requires_registration(self) -> bool:
        return self.resources is not None

    def _full_tensor(self) -> FullTensorNixlPublicationMethod:
        if isinstance(self.method, FullTensorNixlPublicationMethod):
            return self.method
        if self._full_tensor_factory is None:
            raise RuntimeError("operation requires full-tensor publication")
        try:
            self.method = self._full_tensor_factory()
        except Exception:
            self.close()
            raise
        return self.method

    def _canonical_delta(self) -> CanonicalDeltaPublicationMethod:
        if not isinstance(self.method, CanonicalDeltaPublicationMethod):
            raise RuntimeError(  # noqa: TRY004
                "operation requires canonical-delta publication"
            )
        return self.method

    @property
    def source_slot_id(self) -> str:
        return self._full_tensor().source_slot_id

    def bind_tensors(self, tensors: Any) -> str:
        if tensors is None:
            raise ValueError("tensors must not be None")
        if self._bound_tensors is not None:
            raise RuntimeError("trainer tensors are already bound")
        slot = self._full_tensor().bind_tensors(tensors)
        self._bound_tensors = tensors
        return slot

    def prepare_delta_base(
        self, *, hf_tensor_iter: Iterable[list[tuple[str, torch.Tensor]]]
    ) -> None:
        self._canonical_delta().prepare_base(hf_tensor_iter=hf_tensor_iter)

    def stage(
        self,
        *,
        version: WeightVersionRef,
        tensors: Any,
        hf_tensor_iter: Iterable[list[tuple[str, torch.Tensor]]] | None,
    ) -> PublicationArtifact:
        if self.method is None:
            self._full_tensor()
        if isinstance(self.method, FullTensorNixlPublicationMethod):
            if hf_tensor_iter is not None:
                raise ValueError("hf_tensor_iter is only supported for object storage")
            return self.method.stage(version=version, tensors=tensors)
        if tensors is not None:
            raise ValueError(
                "object storage publication accepts hf_tensor_iter, not tensors"
            )
        if hf_tensor_iter is None:
            raise ValueError(
                "hf_tensor_iter is required for object storage publication"
            )
        return self.method.stage(version=version, hf_tensor_iter=hf_tensor_iter)

    def publish(
        self, *, version: WeightVersionRef, staged: PublicationArtifact
    ) -> None:
        if self.method is None:
            raise RuntimeError("publication method is not initialized")
        self.method.publish(version=version, staged=staged)

    def publish_bound(self, *, version: WeightVersionRef) -> None:
        if self._bound_tensors is None:
            raise RuntimeError("bind_tensors() must be called before publish_version()")
        method = self._full_tensor()
        recorder = timing.start_cycle(
            version_id=version.version_id,
            rank=rl_envs.LOCAL_RANK,
            backend="rl_trainer",
        )
        try:
            with timing.active(recorder):
                staged = method.stage(
                    version=version,
                    tensors=self._bound_tensors,
                )
                method.publish(version=version, staged=staged)
        finally:
            payload = timing.emit(recorder, logger)
            if payload is not None:
                self._last_full_tensor_metrics = _flatten_timing(payload)

    def release(self, *, version: WeightVersionRef) -> None:
        if isinstance(self.method, FullTensorNixlPublicationMethod):
            self.method.release(version=version)

    def pop_metrics(self) -> dict[str, int | float]:
        if isinstance(self.method, CanonicalDeltaPublicationMethod):
            return self.method.pop_metrics()
        metrics = self._last_full_tensor_metrics
        self._last_full_tensor_metrics = {}
        return metrics

    def close(self) -> None:
        if self._closed:
            return
        if self.method is not None:
            try:
                self.method.close()
            except Exception:
                logger.warning(
                    "failed to close trainer publication method", exc_info=True
                )
        self._bound_tensors = None
        self._full_tensor_factory = None
        if self.resources is not None:
            try:
                self.resources.close()
            except Exception:
                logger.warning("failed to close trainer resources", exc_info=True)
        self._closed = True


__all__ = ["PublicationArtifact", "TrainerRuntime"]
