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

from ..control import WeightVersion

from .adapter import GeneratorEngineContext
from .methods import (
    CanonicalDeltaUpdateMethod,
    LoadTimeTensorNixlUpdateMethod,
    RuntimeTensorNixlUpdateMethod,
)
from .nixl_staged_transfer import _NixlStagedTransfer
from .plan import (
    EngineInstaller,
    SourceResolver,
    UpdateMethod,
    WeightSource,
    WeightUpdatePlanner,
)
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
    """Engine facts required by load-time and runtime NIXL methods."""

    # Accelerator ordinal used for device access by NIXL.
    device_id: int
    device: Any
    # Rank in the engine's weight-transfer group; peer selection matches it.
    worker_rank: int
    capture_layout: Callable
    runtime_tensors: dict[str, Any] | None
    source_worker_id: str | None
    unpublish_runtime_tensors: Callable[[], None]
    publish_runtime_tensors: Callable[[str], None]
    build_identity: Callable[[str], p2p_pb2.SourceIdentity]
    nixl_manager: Any | None = None


@dataclass(frozen=True)
class EngineRuntime:
    """Engine-specific installation and target geometry."""

    model_name: str
    installer: EngineInstaller
    full_tensor: FullTensorEngineCapability | None = None


class GeneratorRuntime:
    """Own engine resources, update methods, transport, and update sessions."""

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

    def unpublish_runtime_tensors(self) -> None:
        capability = self.engine.full_tensor
        if capability is not None and capability.runtime_tensors is not None:
            capability.unpublish_runtime_tensors()

    def publish_runtime_tensors(self, version_id: str) -> None:
        capability = self.engine.full_tensor
        if capability is not None and capability.runtime_tensors is not None:
            capability.publish_runtime_tensors(version_id)

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


def _resolve_source_order(
    *,
    engine: EngineRuntime,
    object_storage: ObjectStorageGeneratorConfig | None,
    source_order: tuple[WeightSource, ...] | None,
) -> tuple[WeightSource, ...]:
    has_generator_p2p = (
        engine.full_tensor is not None
        and bool(engine.full_tensor.runtime_tensors)
        and engine.full_tensor.nixl_manager is not None
    )
    if source_order is not None:
        if (
            WeightSource.GENERATOR in source_order
            and not has_generator_p2p
            and WeightSource.OBJECT_STORAGE in source_order
        ):
            logger.warning(
                "inference P2P state is unavailable; using object storage only"
            )
            return tuple(
                source for source in source_order if source is not WeightSource.GENERATOR
            )
        return source_order
    if object_storage is not None:
        if has_generator_p2p:
            return (WeightSource.GENERATOR, WeightSource.OBJECT_STORAGE)
        return (WeightSource.OBJECT_STORAGE,)
    defaults = []
    if has_generator_p2p:
        defaults.append(WeightSource.GENERATOR)
    defaults.append(WeightSource.TRAINER)
    return tuple(defaults)


def _validate_source_order(
    *,
    engine_context: GeneratorEngineContext,
    engine: EngineRuntime,
    object_storage: ObjectStorageGeneratorConfig | None,
    source_order: tuple[WeightSource, ...],
) -> None:
    supported_sources = set()
    if object_storage is not None:
        supported_sources.add(WeightSource.OBJECT_STORAGE)
    if engine.full_tensor is not None:
        supported_sources.add(WeightSource.TRAINER)
        if (
            engine.full_tensor.runtime_tensors
            and engine.full_tensor.nixl_manager is not None
        ):
            supported_sources.add(WeightSource.GENERATOR)
    for source in source_order:
        if source not in supported_sources:
            raise ValueError(
                f"{type(engine_context).__name__} does not support "
                f"{source.value} refit"
            )


def _create_load_time_tensor_method(
    *,
    capability: FullTensorEngineCapability,
    worker_id: str,
) -> LoadTimeTensorNixlUpdateMethod:
    transfer = _NixlStagedTransfer(
        agent_name=f"mx-refit-load-time-{worker_id}",
        device_id=capability.device_id,
        device=capability.device,
        listen_port=None,
    )
    return LoadTimeTensorNixlUpdateMethod(
        transfer=transfer,
        capture_layout=capability.capture_layout,
    )


def _create_runtime_tensor_method(
    *,
    capability: FullTensorEngineCapability,
) -> RuntimeTensorNixlUpdateMethod:
    runtime_tensors = capability.runtime_tensors
    if runtime_tensors is None:
        raise ValueError("generator P2P requires inference runtime tensors")
    if capability.nixl_manager is None:
        raise ValueError("generator P2P requires the inference NIXL manager")
    transfer = _NixlStagedTransfer(
        device_id=capability.device_id,
        device=capability.device,
        manager=capability.nixl_manager,
    )
    return RuntimeTensorNixlUpdateMethod(
        transfer=transfer,
        runtime_tensors=runtime_tensors,
    )


def _create_resolvers(
    *,
    engine: EngineRuntime,
    source_order: tuple[WeightSource, ...],
    p2p_client: MxClient | None,
    rpc_timeout_seconds: float,
    service: Callable,
) -> tuple[SourceResolver, ...]:
    resolvers = []
    for source in source_order:
        if source is WeightSource.GENERATOR:
            capability = engine.full_tensor
            if (
                capability is None
                or capability.source_worker_id is None
                or p2p_client is None
            ):
                raise ValueError("generator P2P source is not initialized")
            resolvers.append(
                GeneratorSourceResolver(
                    p2p_client=p2p_client,
                    worker_id=capability.source_worker_id,
                    worker_rank=capability.worker_rank,
                    build_identity=capability.build_identity,
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
    return tuple(resolvers)


def _close_partial_resources(
    methods: list[UpdateMethod],
    p2p_client: MxClient | None,
) -> None:
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


def initialize_generator_runtime(
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
    resolve_replay_chain: Callable[
        [str, bool], tuple[WeightVersion, ...]
    ]
    | None = None,
) -> GeneratorRuntime:
    """Resolve construction policy and build one rank-local runtime."""
    from .engines import _create_engine_runtime

    engine = _create_engine_runtime(engine_context)
    resolved_source_order = _resolve_source_order(
        engine=engine,
        object_storage=object_storage,
        source_order=source_order,
    )
    _validate_source_order(
        engine_context=engine_context,
        engine=engine,
        object_storage=object_storage,
        source_order=resolved_source_order,
    )
    methods: list[UpdateMethod] = []
    p2p_client = None
    try:
        if WeightSource.OBJECT_STORAGE in resolved_source_order:
            if object_storage is None:
                raise ValueError("object storage source requires configuration")
            methods.append(
                CanonicalDeltaUpdateMethod(
                    model_name=engine.model_name,
                    config=object_storage,
                )
            )
        if any(
            source in {WeightSource.GENERATOR, WeightSource.TRAINER}
            for source in resolved_source_order
        ):
            if engine.full_tensor is None:
                raise ValueError("full-tensor source requires engine support")
            p2p_method_start = len(methods)
            try:
                if WeightSource.TRAINER in resolved_source_order:
                    methods.append(
                        _create_load_time_tensor_method(
                            capability=engine.full_tensor,
                            worker_id=worker_id,
                        )
                    )
                if WeightSource.GENERATOR in resolved_source_order:
                    p2p_client = MxClient(server_url=server_url)
                    methods.append(
                        _create_runtime_tensor_method(
                            capability=engine.full_tensor,
                        )
                    )
            except Exception as error:
                if WeightSource.OBJECT_STORAGE not in resolved_source_order:
                    raise
                logger.warning(
                    "P2P initialization failed; using object storage only: %s",
                    error,
                )
                for method in methods[p2p_method_start:]:
                    try:
                        method.close()
                    except Exception:
                        logger.warning(
                            "failed to close unavailable P2P method",
                            exc_info=True,
                        )
                del methods[p2p_method_start:]
                if p2p_client is not None:
                    try:
                        p2p_client.close()
                    except Exception:
                        logger.warning(
                            "failed to close unavailable P2P client",
                            exc_info=True,
                        )
                p2p_client = None
                resolved_source_order = (WeightSource.OBJECT_STORAGE,)
        method_tuple = tuple(methods)
        replay_from_full_root = {
            WeightSource.GENERATOR,
            WeightSource.OBJECT_STORAGE,
        }.issubset(resolved_source_order)
        runtime = GeneratorRuntime(
            engine=engine,
            methods=method_tuple,
            session=WeightUpdateSession(
                planner=WeightUpdatePlanner(
                    resolvers=_create_resolvers(
                        engine=engine,
                        source_order=resolved_source_order,
                        p2p_client=p2p_client,
                        rpc_timeout_seconds=rpc_timeout_seconds,
                        service=service,
                    ),
                    methods=method_tuple,
                    installer=engine.installer,
                    max_transfer_attempts=max_transfer_attempts,
                ),
                start_lease=start_lease,
                resolve_replay_chain=(
                    None
                    if resolve_replay_chain is None
                    else lambda version: resolve_replay_chain(
                        version.version_id,
                        replay_from_full_root,
                    )
                ),
            ),
            p2p_client=p2p_client,
            initial_version_id=(
                object_storage.initial_base_version_id
                if WeightSource.OBJECT_STORAGE in resolved_source_order
                and object_storage is not None
                else None
            ),
        )
    except BaseException:
        _close_partial_resources(methods, p2p_client)
        raise
    return runtime


__all__ = [
    "EngineRuntime",
    "FullTensorEngineCapability",
    "GeneratorRuntime",
    "initialize_generator_runtime",
]
