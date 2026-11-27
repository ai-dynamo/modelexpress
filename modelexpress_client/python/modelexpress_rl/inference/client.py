# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-local generator lifecycle for ModelExpress RL refit."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal

import grpc
from modelexpress import auth, envs
from modelexpress.client import _get_server_url
from modelexpress_rl.refit.timing import RefitTimingRecorder, refit_span

from modelexpress_rl import envs as rl_envs
from modelexpress_rl import timing
from modelexpress_rl.version import WeightVersionRef

from .. import refit_pb2, refit_pb2_grpc
from ..control import WeightVersion, WeightVersionState, _weight_version
from ..object_storage import ObjectStorageType
from .adapter import GeneratorEngineContext
from .plan import StreamingSettings, WeightSource, parse_weight_source_order
from .receiver import ObjectStorageGeneratorConfig
from .runtime import GeneratorRuntime, initialize_generator_runtime
from .session import SessionUpdate
from .stage_process import StageProcess
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
    # Capacity per bounded-streaming arena; None stages a full update.
    staging_buffer_bytes: int | None = None
    # One arena serializes reads and installs; two allow their overlap.
    staging_buffers_count: int = 1
    # Receive arenas use the engine's device (None, or its type: "cuda" or "xpu")
    # or pinned host memory ("cpu").
    staging_device: Literal["cuda", "xpu", "cpu"] | None = None
    # Canonical object-storage checkpoint settings.
    object_storage: ObjectStorageGeneratorConfig | None = None
    # Version read from the engine's serving-version API after cold start.
    initial_serving_version_id: str | None = None
    # Ordered source fallback. Canonical object storage may be used alone or
    # combined with generator P2P in either order.
    source_order: tuple[WeightSource, ...] | None = None
    # Run the staging step in a child process when object storage is the only
    # source; None follows MX_REFIT_STAGE_PROCESS.
    stage_process: bool | None = None

    def __post_init__(self) -> None:
        """Validate explicit settings before client initialization."""
        if self.stage_process is not None and not isinstance(self.stage_process, bool):
            raise TypeError("stage_process must be a bool or None")
        if (
            self.initial_serving_version_id is not None
            and not self.initial_serving_version_id.strip()
        ):
            raise ValueError("initial_serving_version_id must not be empty")
        source_order_env = rl_envs.MX_GENERATOR_SOURCE_ORDER
        if self.source_order is None and source_order_env is not None:
            object.__setattr__(
                self,
                "source_order",
                parse_weight_source_order(source_order_env),
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
        if self.staging_buffer_bytes is not None:
            if isinstance(self.staging_buffer_bytes, bool) or not isinstance(
                self.staging_buffer_bytes, int
            ):
                raise ValueError("staging_buffer_bytes must be a positive integer")
            rl_envs.require_positive_int(
                self.staging_buffer_bytes, "staging_buffer_bytes"
            )
        if isinstance(self.staging_buffers_count, bool) or not isinstance(
            self.staging_buffers_count, int
        ):
            raise ValueError("staging_buffers_count must be a positive integer")
        rl_envs.require_positive_int(
            self.staging_buffers_count, "staging_buffers_count"
        )
        if self.staging_device not in (None, "cuda", "xpu", "cpu"):
            raise ValueError("staging_device must be None, 'cuda', 'xpu', or 'cpu'")
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

    Preparation may validate and reserve a P2P peer or reconstruct a stored
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
        stage_process_metrics: dict[str, float] | None = None,
    ) -> None:
        self._client = client
        self.version_id = version_id
        self._update = update
        # What the stage process measured when it ran this update's staging
        # step. The update's own metrics are then those of adopting the result.
        self._stage_process_metrics = stage_process_metrics or {}
        # A refit's stages are measured across two client calls, so the cycle's
        # recorder travels on the handle that connects them.
        self._timing = timing

    def release(self) -> None:
        """Release this update; reusable receive buffers may remain allocated."""
        self._client._release_staged(self)

    @property
    def metrics(self) -> dict[str, float]:
        """Return metrics recorded for this update by the selected adapter."""
        if self._update is None:
            return {}
        return {**self._update.metrics, **self._stage_process_metrics}

    @property
    def applied(self) -> bool:
        """Return whether engine installation completed."""
        if self._update is None:
            return True
        return self._update.applied

    @property
    def _installation_failed(self) -> bool:
        if self._update is None:
            return False
        return self._update.installation_started and not self._update.applied

    @property
    def is_streamed(self) -> bool:
        """Whether the install still reads the weights from their source."""
        return self._update is not None and self._update.is_streamed


class _WorkerLease:
    """The version leases the stage process holds for one staged update."""

    def __init__(self, stage_process: StageProcess) -> None:
        self._stage_process = stage_process
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._stage_process.release()


@dataclass(frozen=True)
class _WorkerStaged:
    """What the stage process's staging step leaves for this client."""

    # Every version record its staging step read from the MX server, by id.
    versions: dict[str, WeightVersion]
    # What its staging step measured.
    metrics: dict[str, float]
    lease: _WorkerLease


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
        self._streaming: StreamingSettings | None = None
        self._stage_process: StageProcess | None = None
        # Set while this client adopts what the stage process staged: its
        # answers then stand in for the MX server.
        self._worker_staged: _WorkerStaged | None = None
        # The version records the current staging step read from the MX server.
        self._fetched_versions: dict[str, WeightVersion] = {}
        self._offload_staging = False
        self._closed = False

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
            and config.object_storage.storage_type
            not in {ObjectStorageType.S3, ObjectStorageType.AZURE}
        ):
            raise ValueError("only S3 and Azure object storage are currently supported")

        client = cls()
        client.model_name = model_name
        client.worker_id = worker_id
        client.server_url = server_url
        client._registration_ttl_seconds = registration_ttl_seconds
        client._lease_ttl_seconds = lease_ttl_seconds
        client._rpc_timeout_seconds = config.rpc_timeout_seconds
        client._max_replay_chain_length = config.max_replay_chain_length
        if config.staging_buffer_bytes is not None:
            client._streaming = StreamingSettings(
                max_staging_bytes=config.staging_buffer_bytes
                * config.staging_buffers_count,
                staging_device=config.staging_device,
                staging_buffers=config.staging_buffers_count,
            )
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
                streaming=client._streaming,
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
            client._start_stage_process(config)
        except Exception:
            client.close()
            raise
        return client

    def stage_weight(self, *, version: WeightVersionRef) -> StagedWeightHandle:
        """Prepare an exact version without installing it into the live engine.

        With ``staging_buffer_bytes`` configured, reserve bounded buffers and
        prepare deferred reads. ``apply_weight`` then pipelines read/install.
        Otherwise, trainer sources transfer a complete independent staged copy.
        Staging prepares an update; it does not guarantee that weight bytes have
        been transferred. Streaming transfers start during ``apply_weight``.
        With a stage process, the whole staging step runs there, including its
        MX server calls, and never in this process: a version the stage process
        cannot stage raises StageProcessError. The exception is a version whose
        install reads the weights from their source, such as a streamed full
        checkpoint: there is nothing to stage ahead, so this client stages it
        and holds its lease, as it does without a stage process.
        """
        if not isinstance(version, WeightVersionRef):
            raise TypeError("version must be a WeightVersionRef")
        with self._operation_lock:
            active = self._active_handle
            if active is not None and active._installation_failed:
                self._release_staged(active)
                active = None
            if active is not None:
                if active.version_id == version.version_id:
                    return active
                raise RuntimeError("another generator update is still active")
            if self._stage_process is None or not self._offload_staging:
                return self._stage_weight(version)
            staged = self._stage_process.stage(
                version.version_id,
                serving_version_id=self._serving_version_id,
                engine_state=self._engine_state.value,
            )
            if staged is None:
                # The install of this version reads the weights from their
                # source, in this process, so this client holds the lease.
                return self._stage_weight(version)
            worker_staged = _WorkerStaged(*staged, _WorkerLease(self._stage_process))
            try:
                return self._stage_weight(version, worker_staged)
            except BaseException:
                worker_staged.lease.close()
                raise

    def _stage_weight(
        self,
        version: WeightVersionRef,
        worker_staged: _WorkerStaged | None = None,
    ) -> StagedWeightHandle:
        """Run the staging step for ``version`` and return its handle.

        The stage process runs this in its own client. The engine's client then
        runs it with ``worker_staged``, the worker's result, to get a handle it
        can install. In that run the MX server's answers and the leases come
        from the worker, so this process asks the server nothing, and the
        session finds the version already staged.
        """
        with self._operation_lock:
            runtime = self._require_runtime()
            self._worker_staged = worker_staged
            self._fetched_versions = {}
            try:
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
                    # Failed preparation has no handle to report the timing later.
                    timing.emit(recorder, logger)
                    raise
            finally:
                self._worker_staged = None
            stage_process_metrics = (
                {} if worker_staged is None else worker_staged.metrics
            )
            self._active_handle = StagedWeightHandle(
                client=self,
                version_id=version.version_id,
                update=update,
                timing=recorder,
                stage_process_metrics=stage_process_metrics,
            )
            return self._active_handle

    def apply_weight(self, staged: StagedWeightHandle) -> Any:
        """Install a prepared update at the caller's safe point."""
        if not isinstance(staged, StagedWeightHandle) or staged._client is not self:
            raise ValueError("staged handle does not belong to this client")
        with self._operation_lock:
            if self._active_handle is not staged:
                raise RuntimeError("staged weight has already been released")
            if staged._update is None:
                return None
            if staged._update.released:
                raise RuntimeError("staged weight has already been released")
            if staged._installation_failed:
                raise RuntimeError(
                    "staged weight installation failed; stage a fresh transaction"
                )
            runtime = self._require_runtime()
            was_applied = staged._update.applied
            restore_version_id = None
            if self._engine_state is _EngineState.READY:
                restore_version_id = self._serving_version_id
            if not was_applied:
                runtime.unpublish_runtime_tensors()
            try:
                with timing.active(staged._timing):
                    result = runtime.session.apply(staged._update)
            except BaseException:
                if staged._installation_failed:
                    self._engine_state = _EngineState.UNCERTAIN
                elif not staged._update.applied and restore_version_id is not None:
                    try:
                        runtime.publish_runtime_tensors(restore_version_id)
                    except Exception:
                        logger.exception(
                            "failed to republish unchanged runtime tensors for %s",
                            restore_version_id,
                        )
                raise
            finally:
                # Installation remains committed when subsequent lease cleanup fails.
                if staged._update.applied:
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
                # Reported even when the install raised: a refit that failed
                # after seconds on the wire is exactly the case the split has to
                # explain, and dropping the record would leave the failure with
                # no timing at all.
                timing.emit(staged._timing, logger)
            if staged._stage_process_metrics and isinstance(result, dict):
                # An installer that reports the staging metrics reports this
                # process's, which found the version already staged.
                result = {**result, **staged._stage_process_metrics}
            return result

    def close(self) -> None:
        """Stop renewal and release control-plane and adapter resources."""
        if self._closed:
            return
        if self._stage_process is not None:
            # Before the lock: a stage that waits on the stage process holds it.
            self._stage_process.close()
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

    def _start_stage_process(self, config: ModelExpressGeneratorConfig) -> None:
        """Start a child process that runs this client's staging step."""
        enabled = config.stage_process
        if enabled is None:
            enabled = rl_envs.MX_REFIT_STAGE_PROCESS
        from .engines.stage_only import StageOnlyGeneratorContext

        if isinstance(config.engine_context, StageOnlyGeneratorContext):
            return
        runtime = self._require_runtime()
        source_order = runtime.session.source_order
        if (
            config.object_storage is None
            or WeightSource.OBJECT_STORAGE not in source_order
        ):
            return
        prefetch = rl_envs.MX_REFIT_PREFETCH and rl_envs.LOCAL_RANK in (None, 0)
        self._offload_staging = enabled and source_order == (
            WeightSource.OBJECT_STORAGE,
        )
        if not self._offload_staging and not prefetch:
            if enabled:
                logger.info(
                    "ModelExpress stage process not started: object storage is not "
                    "the only source (%s); staging in-process",
                    ", ".join(source.value for source in source_order),
                )
            return
        # The worker is a generator client like this one, without the engine.
        worker_config = replace(
            config,
            engine_context=StageOnlyGeneratorContext(
                model_name=runtime.engine.model_name,
                capabilities=runtime.engine.installer.capabilities,
            ),
            # Leases are keyed by version and worker, so it has its own identity.
            worker_id=f"{self.worker_id}-stage",
            source_order=(WeightSource.OBJECT_STORAGE,),
            # It joins the checkpoint cache this client has just initialized
            # and does not reset it from the launch seed.
            object_storage=replace(config.object_storage, seed_checkpoint_path=None),
            stage_process=False,
        )
        # The same supervised worker handles explicit staging and optional discovery.
        self._stage_process = StageProcess(worker_config, prefetch=prefetch)

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
            self._try_register_worker()

    def _try_register_worker(self) -> bool:
        try:
            self._register_worker()
        except grpc.RpcError as error:
            logger.warning("worker registration renewal failed: %s", error)
        except Exception:
            logger.exception("unexpected worker registration renewal failure")
        else:
            return True
        return False

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
        worker_staged = self._worker_staged
        if worker_staged is not None:
            # The stage process asked the MX server; its answers stand in here.
            if version_id not in worker_staged.versions:
                raise RuntimeError(
                    f"target {target_version_id!r}: the stage process did not "
                    f"resolve revision {version_id!r}"
                )
            return worker_staged.versions[version_id]
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
        self._fetched_versions[version_id] = version
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
            raise RuntimeError(f"initial serving version {version_id!r} is not READY")
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

    def _try_renew_lease(self, version_id: str) -> grpc.StatusCode | None:
        """Renew one lease and return the RPC failure code, never raising."""
        try:
            self._register_lease(version_id)
        except grpc.RpcError as error:
            logger.warning("version %s lease renewal failed: %s", version_id, error)
            return error.code()
        except Exception:
            logger.exception("unexpected version %s lease renewal failure", version_id)
        return None

    def _start_version_lease(self, version_id: str) -> _VersionLease | _WorkerLease:
        if self._worker_staged is not None:
            # The stage process holds the leases; closing this releases them.
            return self._worker_staged.lease
        # A restarted MX server loses registrations until the next renewal tick.
        self._register_worker()
        lease = self._register_lease(version_id)
        stop = threading.Event()

        def renew() -> None:
            interval_seconds = max(self._lease_ttl_seconds / 3, 0.1)
            while not stop.wait(interval_seconds):
                code = self._try_renew_lease(version_id)
                # FAILED_PRECONDITION includes a missing worker registration (e.g. after an
                # MX restart). RegisterWorker is an idempotent upsert, so it is harmless for
                # the other causes.
                if (
                    code is grpc.StatusCode.FAILED_PRECONDITION
                    and self._try_register_worker()
                ):
                    self._try_renew_lease(version_id)

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
        """Free the active slot once locally released, including cleanup errors."""
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
            finally:
                if self._active_handle is staged and (
                    staged._update is None or staged._update.released
                ):
                    self._active_handle = None


__all__ = [
    "ModelExpressGeneratorClient",
    "ModelExpressGeneratorConfig",
    "StagedWeightHandle",
    "WeightSource",
]
