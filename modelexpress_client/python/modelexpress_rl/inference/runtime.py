# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep rank-local composition for one generator client."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from modelexpress import p2p_pb2
from modelexpress.client import MxClient

from .. import envs as rl_envs

from .adapter import GeneratorEngineContext
from .methods import CanonicalDeltaUpdateMethod, FullTensorNixlUpdateMethod
from .nixl_staged_transfer import _NixlStagedTransfer
from .plan import EngineInstaller, UpdateMethod, WeightSource, WeightUpdatePlanner
from .receiver import ObjectStorageGeneratorConfig
from .session import WeightUpdateSession
from .source import (
    GeneratorSourceResolver,
    ObjectStorageSourceResolver,
    TrainerSourceResolver,
)

logger = logging.getLogger("modelexpress_rl.inference.runtime")


@dataclass(frozen=True)
class FullTensorEngineCapability:
    """Engine facts required by the generic full-tensor NIXL method."""

    device_id: int
    device: Any
    worker_rank: int
    accelerator: str
    capture_layout: Callable
    parameter_layout: Callable
    build_identity: Callable[[str], p2p_pb2.SourceIdentity]


@dataclass(frozen=True)
class EngineRuntime:
    """Engine-specific installation and target geometry."""

    model_name: str
    installer: EngineInstaller
    full_tensor: FullTensorEngineCapability | None = None


class GeneratorRuntime:
    """Own source policy, update methods, transport, and update sessions."""

    def __init__(
        self,
        *,
        engine: EngineRuntime,
        methods: tuple[UpdateMethod, ...],
        session: WeightUpdateSession,
        p2p_client: MxClient | None,
        initial_version_id: str | None,
    ) -> None:
        self.engine = engine
        self.methods = methods
        self.session = session
        self.p2p_client = p2p_client
        self.initial_version_id = initial_version_id
        self._closed = False

    @classmethod
    def initialize(
        cls,
        *,
        engine_context: GeneratorEngineContext,
        worker_id: str,
        server_url: str,
        object_storage: ObjectStorageGeneratorConfig | None,
        source_order: tuple[WeightSource, ...] | None,
        max_transfer_attempts: int,
        rpc_timeout_seconds: float,
        service: Callable,
        start_lease: Callable[[str], Any],
    ) -> GeneratorRuntime:
        from .engines import _create_engine_runtime

        engine = _create_engine_runtime(engine_context)
        enable_object_storage = object_storage is not None and (
            source_order is None or WeightSource.OBJECT_STORAGE in source_order
        )
        if source_order is None:
            if enable_object_storage:
                source_order = (WeightSource.OBJECT_STORAGE,)
            else:
                defaults = []
                if engine.full_tensor is not None:
                    defaults.append(WeightSource.GENERATOR)
                defaults.append(WeightSource.TRAINER)
                source_order = tuple(defaults)

        needs_full_tensor = any(
            source in {WeightSource.GENERATOR, WeightSource.TRAINER}
            for source in source_order
        )
        if needs_full_tensor and engine.full_tensor is None:
            requested = next(
                source
                for source in source_order
                if source in {WeightSource.GENERATOR, WeightSource.TRAINER}
            )
            raise ValueError(
                f"{type(engine_context).__name__} does not support "
                f"{requested.value} refit"
            )

        supported_sources = set()
        if enable_object_storage:
            supported_sources.add(WeightSource.OBJECT_STORAGE)
        if engine.full_tensor is not None:
            supported_sources.update({WeightSource.GENERATOR, WeightSource.TRAINER})
        for source in source_order:
            if source not in supported_sources:
                raise ValueError(
                    f"{type(engine_context).__name__} does not support "
                    f"{source.value} refit"
                )

        p2p_client = None
        methods: list[UpdateMethod] = []
        if enable_object_storage:
            assert object_storage is not None
            methods.append(
                CanonicalDeltaUpdateMethod(
                    model_name=engine.model_name,
                    config=object_storage,
                )
            )
        try:
            if needs_full_tensor:
                p2p_client = MxClient(server_url=server_url)
                assert engine.full_tensor is not None
                full_tensor = engine.full_tensor
                transfer = _NixlStagedTransfer(
                    agent_name=f"mx-refit-{worker_id}",
                    device_id=full_tensor.device_id,
                    device=full_tensor.device,
                    listen_port=(
                        rl_envs.MX_REFIT_METADATA_PORT + full_tensor.device_id
                    ),
                )
                methods.append(
                    FullTensorNixlUpdateMethod(
                        transfer=transfer,
                        capture_layout=full_tensor.capture_layout,
                        parameter_layout=full_tensor.parameter_layout,
                        build_identity=full_tensor.build_identity,
                        worker_rank=full_tensor.worker_rank,
                        worker_id=worker_id,
                        accelerator=full_tensor.accelerator,
                        p2p_client=p2p_client,
                    )
                )

            resolvers = []
            for source in source_order:
                if source is WeightSource.GENERATOR:
                    assert engine.full_tensor is not None
                    assert p2p_client is not None
                    resolvers.append(
                        GeneratorSourceResolver(
                            p2p_client=p2p_client,
                            worker_id=worker_id,
                            worker_rank=engine.full_tensor.worker_rank,
                            build_identity=engine.full_tensor.build_identity,
                            rpc_timeout_seconds=rpc_timeout_seconds,
                        )
                    )
                elif source is WeightSource.TRAINER:
                    resolvers.append(
                        TrainerSourceResolver(
                            service=service,
                            rpc_timeout_seconds=rpc_timeout_seconds,
                        )
                    )
                else:
                    resolvers.append(ObjectStorageSourceResolver())

            methods_tuple = tuple(methods)
            return cls(
                engine=engine,
                methods=methods_tuple,
                session=WeightUpdateSession(
                    planner=WeightUpdatePlanner(
                        resolvers=tuple(resolvers),
                        methods=methods_tuple,
                        installer=engine.installer,
                        max_transfer_attempts=max_transfer_attempts,
                    ),
                    start_lease=start_lease,
                ),
                p2p_client=p2p_client,
                initial_version_id=(
                    object_storage.initial_base_version_id
                    if enable_object_storage and object_storage is not None
                    else None
                ),
            )
        except Exception:
            for method in methods:
                try:
                    method.close()
                except Exception:
                    logger.warning(
                        "failed to close generator method after initialization error",
                        exc_info=True,
                    )
            if p2p_client is not None:
                try:
                    p2p_client.close()
                except Exception:
                    logger.warning(
                        "failed to close P2P client after initialization error",
                        exc_info=True,
                    )
            raise

    def close(self) -> None:
        if self._closed:
            return
        for method in self.methods:
            try:
                method.close()
            except Exception:
                logger.warning("failed to close generator method", exc_info=True)
        if self.p2p_client is not None:
            try:
                self.p2p_client.close()
            except Exception:
                logger.warning("failed to close generator P2P client", exc_info=True)
        self._closed = True


__all__ = [
    "EngineRuntime",
    "FullTensorEngineCapability",
    "GeneratorRuntime",
]
