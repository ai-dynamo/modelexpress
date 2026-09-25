# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-local generator lifecycle for ModelExpress RL refit."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

import grpc
from modelexpress import auth, envs
from modelexpress.client import _get_server_url
from modelexpress.refit.timing import RefitTimingRecorder, refit_span

from modelexpress_rl import envs as rl_envs
from modelexpress_rl import timing
from modelexpress_rl.version import WeightVersionRef

from .. import refit_pb2, refit_pb2_grpc
from ..control import WeightVersion, WeightVersionState, _weight_version
from ..object_storage import ObjectStorageType
from .adapter import GeneratorEngineContext
from .checkpoint_lifecycle import CheckpointCollectiveContext, run_checkpoint_lifecycle
from .checkpoint_transaction import CheckpointTransactionError
from .plan import PreparedCheckpointArtifact, WeightSource, parse_weight_source_order
from .receiver import ObjectStorageGeneratorConfig
from .runtime import GeneratorRuntime, initialize_generator_runtime
from .session import SessionUpdate
from .version_chain import resolve_replay_chain

logger = logging.getLogger("modelexpress_rl.inference.client")


class _EngineState(str, Enum):
    READY = "READY"
    UNCERTAIN = "UNCERTAIN"


def _required(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} is required")
    return value


@dataclass(frozen=True)
class ModelExpressGeneratorConfig:
    """Immutable configuration for one rank-local generator client."""

    # Live rank-local objects required by the selected inference engine adapter.
    engine_context: GeneratorEngineContext
    # Logical model identity; defaults to MX_MODEL_NAME_OVERRIDE.
    model_name: str | None = None
    # Fresh process-lifetime identity; generated when omitted.
    worker_id: str | None = None
    # Address of the central ModelExpress server; uses the standard MX default.
    server_url: str | None = None
    # Worker registration lifetime; defaults to three heartbeat intervals.
    registration_ttl_seconds: int | None = None
    # Weight-version lease lifetime; defaults to the registration lifetime.
    lease_ttl_seconds: int | None = None
    # Maximum source-discovery and transfer attempts for one staged update.
    max_transfer_attempts: int = 3
    # Maximum number of payload revisions applied by one target replay.
    max_replay_chain_length: int = 64
    # Deadline applied independently to each control-plane or manifest RPC.
    rpc_timeout_seconds: float = 30.0
    # Canonical object-storage checkpoint settings.
    object_storage: ObjectStorageGeneratorConfig | None = None
    # Version read from the engine's serving-version API after cold start.
    initial_serving_version_id: str | None = None
    # Ordered source fallback. Canonical object storage may be used alone or
    # combined with generator P2P in either order.
    source_order: tuple[WeightSource, ...] | None = None
    checkpoint_install_mode: str = "full"

    def __post_init__(self) -> None:
        """Validate explicit settings before client initialization."""
        if (
            self.initial_serving_version_id is not None
            and not self.initial_serving_version_id.strip()
        ):
            raise ValueError("initial_serving_version_id must not be empty")
        source_order_env = envs.MX_GENERATOR_SOURCE_ORDER
        if self.source_order is None and source_order_env is not None:
            object.__setattr__(
                self,
                "source_order",
                parse_weight_source_order(source_order_env),
            )
        if self.checkpoint_install_mode not in ("full", "partial_if_supported"):
            raise ValueError("checkpoint_install_mode must be full or partial_if_supported")
        if self.checkpoint_install_mode == "partial_if_supported":
            from .engines.vllm.context import VllmGeneratorContext

            if not isinstance(self.engine_context, VllmGeneratorContext) or not isinstance(
                self.engine_context.checkpoint_collective, CheckpointCollectiveContext
            ):
                raise ValueError(
                    "partial_if_supported requires VllmGeneratorContext with a "
                    "CheckpointCollectiveContext providing serving pause, fence, "
                    "and finite-timeout full-worker transport"
                )
            if self.object_storage is None or self.source_order != (
                WeightSource.OBJECT_STORAGE,
            ):
                raise ValueError(
                    "partial_if_supported requires object_storage and explicit "
                    "source_order=(WeightSource.OBJECT_STORAGE,) on every rank"
                )
        if self.registration_ttl_seconds is not None:
            rl_envs.require_positive_int(
                self.registration_ttl_seconds, "registration_ttl_seconds"
            )
        if self.lease_ttl_seconds is not None:
            rl_envs.require_positive_int(self.lease_ttl_seconds, "lease_ttl_seconds")
        rl_envs.require_positive_int(
            self.max_transfer_attempts, "max_transfer_attempts"
        )
        rl_envs.require_positive_int(
            self.max_replay_chain_length, "max_replay_chain_length"
        )
        rl_envs.require_positive_float(self.rpc_timeout_seconds, "rpc_timeout_seconds")
        if self.source_order is not None:
            if not isinstance(self.source_order, tuple) or not self.source_order:
                raise ValueError("source_order must be a non-empty tuple")
            if any(
                not isinstance(source, WeightSource) for source in self.source_order
            ):
                raise TypeError("source_order entries must be WeightSource values")
            if len(set(self.source_order)) != len(self.source_order):
                raise ValueError("source_order must not contain duplicates")
            if (
                WeightSource.OBJECT_STORAGE in self.source_order
                and self.object_storage is None
            ):
                raise ValueError(
                    "OBJECT_STORAGE source requires object_storage settings"
                )
            if (
                self.object_storage is not None
                and WeightSource.OBJECT_STORAGE not in self.source_order
            ):
                raise ValueError(
                    "object_storage settings require OBJECT_STORAGE in source_order"
                )
            if self.object_storage is not None and set(self.source_order) - {
                WeightSource.GENERATOR,
                WeightSource.OBJECT_STORAGE,
            }:
                raise ValueError(
                    "object_storage source_order may contain only "
                    "WeightSource.GENERATOR and WeightSource.OBJECT_STORAGE"
                )


class StagedWeightHandle:
    """An exact version prepared for, but not yet installed into, the engine.

    Preparation may validate and reserve a P2P peer or reconstruct an S3
    checkpoint. The live engine remains unchanged until ``apply_weight`` runs at
    its safe point.
    The handle keeps session internals private and binds idempotent release to
    the client that owns the staged resources.
    """

    def __init__(
        self,
        *,
        client: ModelExpressGeneratorClient,
        version_id: str,
        update: SessionUpdate | None,
        timing: RefitTimingRecorder | None = None,
    ) -> None:
        self._client = client
        self.version_id = version_id
        self._update = update
        self._no_op_released = False
        # A refit's stages are measured across two client calls, so the cycle's
        # recorder travels on the handle that connects them.
        self._timing = timing

    def release(self) -> None:
        """Release local staging buffers; repeated calls are idempotent."""
        self._client._release_staged(self)

    @property
    def metrics(self) -> dict[str, float]:
        """Return preparation metrics exposed by the selected adapter."""
        if self._update is None:
            return {}
        return self._update.prepared.metrics

    @property
    def applied(self) -> bool:
        """Return whether engine installation completed."""
        if self._update is None:
            return True
        return self._update.applied


class _VersionLease:
    """Keep one version protected through installation or staged release."""

    def __init__(
        self,
        *,
        client: ModelExpressGeneratorClient,
        version_id: str,
        lease_id: str,
        stop: threading.Event,
        renewal: threading.Thread,
    ) -> None:
        self._client = client
        self._version_id = version_id
        self._lease_id = lease_id
        self._stop = stop
        self._renewal = renewal
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._stop.set()
        self._renewal.join()
        self._client._delete_version_lease(
            version_id=self._version_id,
            lease_id=self._lease_id,
        )
        self._closed = True


class ModelExpressGeneratorClient:
    """Synchronous rank-local generator client for exact-version refit."""

    def __init__(self) -> None:
        """Create an uninitialized rank-local generator client."""
        self._channel: grpc.Channel | None = None
        self._stub: refit_pb2_grpc.RefitServiceStub | None = None
        self._registration_stop = threading.Event()
        self._registration_thread: threading.Thread | None = None
        self._operation_lock = threading.RLock()
        self._active_handle: StagedWeightHandle | None = None
        self._serving_version_id: str | None = None
        self._has_initial_serving_version = False
        self._engine_state = _EngineState.READY
        self._runtime: GeneratorRuntime | None = None
        self._closed = False
        self._checkpoint_collective: CheckpointCollectiveContext | None = None
        self._checkpoint_failed = False

    @classmethod
    def initialize(
        cls,
        config: ModelExpressGeneratorConfig,
    ) -> ModelExpressGeneratorClient:
        """Initialize one generator rank with immutable operating settings.

        ``config.engine_context`` contains the engine's live rank-local objects.
        Callers do not construct ModelExpress adapter or receiver implementations.
        """
        if not isinstance(config, ModelExpressGeneratorConfig):
            raise TypeError("config must be a ModelExpressGeneratorConfig")
        model_name = _required(
            config.model_name or envs.MX_MODEL_NAME_OVERRIDE or "", "model_name"
        )
        worker_id = _required(config.worker_id or uuid.uuid4().hex[:8], "worker_id")
        server_url = _get_server_url(config.server_url)
        registration_ttl_seconds = config.registration_ttl_seconds
        if registration_ttl_seconds is None:
            registration_ttl_seconds = envs.MX_HEARTBEAT_INTERVAL_SECS * 3
        lease_ttl_seconds = config.lease_ttl_seconds
        if lease_ttl_seconds is None:
            lease_ttl_seconds = registration_ttl_seconds
        registration_ttl_seconds = rl_envs.require_positive_int(
            registration_ttl_seconds, "registration_ttl_seconds"
        )
        if (
            config.object_storage is not None
            and config.object_storage.storage_type is not ObjectStorageType.S3
        ):
            raise ValueError("only S3 object storage is currently supported")

        client = cls()
        client.model_name = model_name
        client.worker_id = worker_id
        client.server_url = server_url
        client._registration_ttl_seconds = registration_ttl_seconds
        client._lease_ttl_seconds = lease_ttl_seconds
        client._rpc_timeout_seconds = config.rpc_timeout_seconds
        client._max_replay_chain_length = config.max_replay_chain_length
        if config.checkpoint_install_mode == "partial_if_supported":
            client._checkpoint_collective = config.engine_context.checkpoint_collective
        try:
            runtime = initialize_generator_runtime(
                engine_context=config.engine_context,
                worker_id=worker_id,
                server_url=server_url,
                object_storage=config.object_storage,
                source_order=config.source_order,
                max_transfer_attempts=config.max_transfer_attempts,
                rpc_timeout_seconds=config.rpc_timeout_seconds,
                service=lambda: client._service,
                start_lease=client._start_version_lease,
                resolve_replay_chain=client._resolve_replay_chain,
            )
            client._runtime = runtime
            client._has_initial_serving_version = (
                config.initial_serving_version_id is not None
            )
            client._serving_version_id = (
                config.initial_serving_version_id or runtime.initial_version_id
            )
            if client._serving_version_id is not None:
                client._validate_initial_serving_version(client._serving_version_id)
            client._register_worker()
            client._registration_thread = threading.Thread(
                target=client._renew_worker_registration,
                name=f"modelexpress-refit-renew-{worker_id}",
                daemon=True,
            )
            try:
                client._registration_thread.start()
            except Exception:
                client._registration_thread = None
                raise
        except Exception:
            client.close()
            raise
        return client

    def stage_weight(self, *, version: WeightVersionRef) -> StagedWeightHandle:
        """Synchronously transfer and verify one full-weight version."""
        if not isinstance(version, WeightVersionRef):
            raise TypeError("version must be a WeightVersionRef")
        with self._operation_lock:
            self._require_checkpoint_healthy()
            if self._active_handle is not None:
                if self._active_handle.version_id == version.version_id:
                    return self._active_handle
                raise RuntimeError("another generator update is still active")
            runtime = self._require_runtime()
            if (
                (
                    runtime.initial_version_id is not None
                    or self._has_initial_serving_version
                )
                and version.version_id == self._serving_version_id
                and self._engine_state is _EngineState.READY
            ):
                self._fetch_ready_version(
                    version.version_id,
                    target_version_id=version.version_id,
                )
                self._active_handle = StagedWeightHandle(
                    client=self,
                    version_id=version.version_id,
                    update=None,
                )
                return self._active_handle
            recorder = timing.start_cycle(
                version_id=version.version_id,
                rank=rl_envs.LOCAL_RANK,
            )
            try:
                with timing.active(recorder):
                    with refit_span("control_discovery"):
                        ready = self._get_ready_version(version.version_id)
                    update = runtime.session.stage(ready)
            except BaseException:
                # Nothing else will report this cycle: the recorder is handed on
                # through the staged handle, and staging failed before there was
                # one. A refit that died on the wire is exactly the case the
                # stage split exists to explain.
                timing.emit(recorder, logger)
                raise
            self._active_handle = StagedWeightHandle(
                client=self,
                version_id=version.version_id,
                update=update,
                timing=recorder,
            )
            return self._active_handle

    def apply_weight(self, staged: StagedWeightHandle) -> Any:
        """Install a verified local staged version at the caller's safe point."""
        if not isinstance(staged, StagedWeightHandle) or staged._client is not self:
            raise ValueError("staged handle does not belong to this client")
        with self._operation_lock:
            self._require_checkpoint_healthy()
            if self._active_handle is not staged:
                raise RuntimeError("staged weight has already been released")
            if self._checkpoint_collective is not None:
                self._agree_checkpoint_apply(staged)
            if staged._update is None:
                return None
            if staged._update.released:
                raise RuntimeError("staged weight has already been released")
            runtime = self._require_runtime()
            if self._checkpoint_collective is not None:
                return self._apply_checkpoint_collectively(staged, runtime)
            was_applied = staged._update.applied
            serving_version_id = self._serving_version_id
            if not was_applied:
                runtime.unpublish_runtime_tensors()
            try:
                with timing.active(staged._timing):
                    result = runtime.session.apply(staged._update)
            except BaseException:
                if staged._update.installation_started and not staged._update.applied:
                    self._engine_state = _EngineState.UNCERTAIN
                elif not was_applied and serving_version_id is not None:
                    try:
                        runtime.publish_runtime_tensors(serving_version_id)
                    except Exception:
                        logger.exception(
                            "failed to republish unchanged runtime tensors for %s",
                            serving_version_id,
                        )
                raise
            finally:
                # Reported even when the install raised: a refit that failed
                # after seconds on the wire is exactly the case the split has to
                # explain, and dropping the record would leave the failure with
                # no timing at all.
                timing.emit(staged._timing, logger)
            self._serving_version_id = staged.version_id
            self._engine_state = _EngineState.READY
            if not was_applied:
                try:
                    runtime.publish_runtime_tensors(staged.version_id)
                except Exception:
                    logger.exception(
                        "failed to publish installed runtime tensors for %s",
                        staged.version_id,
                    )
            return result

    def _agree_checkpoint_apply(self, staged: StagedWeightHandle) -> None:
        context = self._checkpoint_collective
        assert context is not None
        state = (
            "apply_checkpoint",
            self._serving_version_id,
            staged.version_id,
            staged._update is not None,
            staged.applied,
        )
        try:
            peers = context.gather(state)
            if len(peers) != context.world_size or any(peer != state for peer in peers):
                raise CheckpointTransactionError("checkpoint apply state differs across ranks")
        except BaseException:
            self._fence_checkpoint(self._require_runtime())
            raise

    def _fence_checkpoint(self, runtime: GeneratorRuntime) -> None:
        self._engine_state = _EngineState.UNCERTAIN
        if self._checkpoint_failed:
            return
        self._checkpoint_failed = True
        assert self._checkpoint_collective is not None
        try:
            self._checkpoint_collective.fence()
        finally:
            try:
                runtime.unpublish_runtime_tensors()
            except Exception:
                logger.exception("failed to withdraw fenced checkpoint runtime")

    def _apply_checkpoint_collectively(
        self, staged: StagedWeightHandle, runtime: GeneratorRuntime
    ) -> Any:
        from .methods.canonical_delta import CanonicalDeltaUpdateMethod

        context = self._checkpoint_collective
        update = staged._update
        assert context is not None and update is not None
        def fence() -> None:
            self._fence_checkpoint(runtime)

        def validate() -> None:
            if not self._serving_version_id:
                raise ValueError("collective checkpoint requires a known serving version")
            if not isinstance(
                update.prepared, PreparedCheckpointArtifact
            ) or not isinstance(update.plan.method, CanonicalDeltaUpdateMethod):
                raise TypeError("partial mode requires a canonical checkpoint artifact")
            if not callable(
                getattr(update.plan.installer, "install_checkpoint_collectively", None)
            ):
                raise TypeError("engine does not support collective checkpoint installation")
            if update.prepared.checkpoint.target_version != staged.version_id:
                raise ValueError("prepared checkpoint target differs from staged version")

        def commit_version() -> None:
            self._serving_version_id = staged.version_id

        def install_checkpoint() -> str:
            mode = update.plan.installer.install_checkpoint_collectively(
                update.prepared.checkpoint,
                serving_version=self._serving_version_id,
                world_size=context.world_size,
                all_gather=context.gather,
                installation_context=lambda: update.plan.method.installation_context(
                    update.prepared, activate=False
                ),
                activate=lambda: update.plan.method.activate(update.prepared),
                commit_version=commit_version,
                fence=fence,
            )
            logger.info(
                "ModelExpress checkpoint version=%s install_mode=%s",
                staged.version_id,
                mode,
            )
            return mode

        try:
            with timing.active(staged._timing):
                result = run_checkpoint_lifecycle(
                    context,
                    serving_version=self._serving_version_id,
                    target_version=staged.version_id,
                    validate=validate,
                    unpublish=runtime.unpublish_runtime_tensors,
                    install=lambda: runtime.session.apply(
                        update, checkpoint_install=install_checkpoint
                    ),
                    publish=lambda: runtime.publish_runtime_tensors(staged.version_id),
                    fence=fence,
                )
            self._engine_state = _EngineState.READY
            return result
        finally:
            timing.emit(staged._timing, logger)

    def _require_checkpoint_healthy(self) -> None:
        if self._checkpoint_failed:
            raise RuntimeError("checkpoint transaction failed; worker recovery required")

    def close(self) -> None:
        """Stop renewal and release control-plane and adapter resources."""
        if self._closed:
            return
        with self._operation_lock:
            if self._active_handle is not None:
                self._release_staged(self._active_handle)
        if self._registration_thread is not None:
            self._registration_stop.set()
            self._registration_thread.join()
            self._registration_thread = None
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self._closed = True

    def __enter__(self) -> ModelExpressGeneratorClient:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    @property
    def _service(self) -> refit_pb2_grpc.RefitServiceStub:
        if self._channel is None:
            self._channel = auth.with_auth(grpc.insecure_channel(self.server_url))
            self._stub = refit_pb2_grpc.RefitServiceStub(self._channel)
        if self._stub is None:
            raise RuntimeError("generator refit service is not initialized")
        return self._stub

    def _require_runtime(self) -> GeneratorRuntime:
        if self._runtime is None:
            raise RuntimeError("generator client is not initialized")
        return self._runtime

    def _register_worker(self) -> None:
        self._service.RegisterWorker(
            refit_pb2.RegisterWorkerRequest(
                worker=refit_pb2.WorkerRegistration(
                    worker_id=self.worker_id,
                    role=refit_pb2.WORKER_ROLE_GENERATOR,
                    model_name=self.model_name,
                ),
                ttl_seconds=self._registration_ttl_seconds,
            ),
            timeout=self._rpc_timeout_seconds,
        )

    def _renew_worker_registration(self) -> None:
        interval_seconds = max(self._registration_ttl_seconds / 3, 0.1)
        while not self._registration_stop.wait(interval_seconds):
            try:
                self._register_worker()
            except grpc.RpcError as error:
                logger.warning("worker registration renewal failed: %s", error)
                continue
            except Exception:
                logger.exception("unexpected worker registration renewal failure")
                continue

    def _get_ready_version(self, version_id: str) -> WeightVersion:
        return self._fetch_ready_version(
            version_id,
            target_version_id=version_id,
        )

    def _fetch_ready_version(
        self,
        version_id: str,
        *,
        target_version_id: str,
    ) -> WeightVersion:
        try:
            response = self._service.GetWeightVersion(
                refit_pb2.GetWeightVersionRequest(uid=version_id),
                timeout=self._rpc_timeout_seconds,
            )
        except grpc.RpcError as error:
            raise RuntimeError(
                f"target {target_version_id!r}: failed to resolve revision "
                f"{version_id!r}: {error.details()}"
            ) from error
        if not response.HasField("version"):
            raise RuntimeError(
                f"target {target_version_id!r}: MX GetWeightVersion response is "
                f"missing revision {version_id!r}"
            )
        version = _weight_version(response.version)
        if version.version_id != version_id:
            raise RuntimeError(
                f"target {target_version_id!r}: requested revision {version_id!r} "
                f"but MX returned {version.version_id!r}"
            )
        if version.state is not WeightVersionState.READY:
            raise RuntimeError(
                f"target {target_version_id!r}: revision {version_id!r} is not READY"
            )
        if version.model_name != self.model_name:
            raise RuntimeError(
                f"target {target_version_id!r}: revision {version_id!r} model_name "
                "does not match the generator"
            )
        return version

    def _resolve_replay_chain(
        self,
        target_version_id: str,
        from_full_root: bool = False,
    ) -> tuple[WeightVersion, ...]:
        """Resolve a canonical chain completely before payload preparation."""
        serving_version_id = None if from_full_root else self._serving_version_id
        if not from_full_root and serving_version_id is None:
            raise RuntimeError("canonical replay requires a known serving version")
        chain = resolve_replay_chain(
            target_version_id=target_version_id,
            fetch_ready_version=lambda version_id: self._fetch_ready_version(
                version_id,
                target_version_id=target_version_id,
            ),
            max_chain_length=self._max_replay_chain_length,
            stop_before_version_id=serving_version_id,
        )
        return chain

    def _validate_initial_serving_version(self, version_id: str) -> None:
        """Verify that the engine-observed initial version is ready and compatible."""
        response = self._service.GetWeightVersion(
            refit_pb2.GetWeightVersionRequest(uid=version_id),
            timeout=self._rpc_timeout_seconds,
        )
        if not response.HasField("version"):
            raise RuntimeError("MX GetWeightVersion response is missing version")
        version = _weight_version(response.version)
        if version.state is not WeightVersionState.READY:
            raise RuntimeError(
                f"initial serving version {version_id!r} is not READY"
            )
        if version.model_name != self.model_name:
            raise RuntimeError(
                "initial serving version model_name does not match the generator"
            )

    def _register_lease(self, version_id: str):
        response = self._service.RegisterVersionLease(
            refit_pb2.RegisterVersionLeaseRequest(
                version_id=version_id,
                worker_id=self.worker_id,
                ttl_seconds=self._lease_ttl_seconds,
            ),
            timeout=self._rpc_timeout_seconds,
        )
        if not response.HasField("lease"):
            raise RuntimeError("MX RegisterVersionLease response is missing lease")
        return response.lease

    def _start_version_lease(self, version_id: str) -> _VersionLease:
        lease = self._register_lease(version_id)
        stop = threading.Event()

        def renew() -> None:
            interval_seconds = max(self._lease_ttl_seconds / 3, 0.1)
            while not stop.wait(interval_seconds):
                try:
                    self._register_lease(version_id)
                except grpc.RpcError as error:
                    logger.warning(
                        "version %s lease renewal failed: %s",
                        version_id,
                        error,
                    )
                except Exception:
                    logger.exception(
                        "unexpected version %s lease renewal failure",
                        version_id,
                    )

        renewal = threading.Thread(
            target=renew,
            name=f"modelexpress-refit-lease-{self.worker_id}",
            daemon=True,
        )
        try:
            renewal.start()
        except Exception:
            self._delete_version_lease(
                version_id=version_id,
                lease_id=lease.lease_id,
            )
            raise
        return _VersionLease(
            client=self,
            version_id=version_id,
            lease_id=lease.lease_id,
            stop=stop,
            renewal=renewal,
        )

    def _delete_version_lease(self, *, version_id: str, lease_id: str) -> None:
        self._service.DeleteVersionLease(
            refit_pb2.DeleteVersionLeaseRequest(
                version_id=version_id,
                lease_id=lease_id,
                worker_id=self.worker_id,
            ),
            timeout=self._rpc_timeout_seconds,
        )

    def _release_staged(self, staged: StagedWeightHandle) -> None:
        if staged._client is not self:
            raise ValueError("staged handle does not belong to this client")
        with self._operation_lock:
            if self._active_handle is not staged:
                return
            try:
                if staged._update is not None and not staged._update.released:
                    try:
                        self._require_runtime().session.release(staged._update)
                    finally:
                        timing.emit(staged._timing, logger)
                elif staged._update is None:
                    staged._no_op_released = True
            finally:
                if self._active_handle is staged:
                    self._active_handle = None


__all__ = [
    "ModelExpressGeneratorClient",
    "ModelExpressGeneratorConfig",
    "StagedWeightHandle",
    "WeightSource",
]
