# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-facing trainer lifecycle for ModelExpress RL refit."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import grpc
import numpy as np
import safetensors.numpy
import torch
import torch.distributed as dist
from modelexpress import auth, envs
from modelexpress.client import _get_server_url
from modelexpress_rl.s3 import S3Client, S3Object
from modelexpress_rl.utils import (
    adler32_checksum,
    compress_delta,
    compute_delta,
    make_tensor_reader,
)

from .. import envs as rl_envs
from .. import refit_pb2, refit_pb2_grpc
from ..version import CANONICAL_DELTA_SOURCE_SLOT, WeightVersionRef
from .adapter import (
    NixlMetadataProvider,
    StagedWeightVersionShardData,
    TrainerEngineAdapter,
    TrainerStagingMode,
    WeightPayloadFormat,
)
from .resources import _TrainerResources


def _required(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} is required")
    return value


def _trainer_adapter(
    *,
    manager: NixlMetadataProvider,
    nixl_metadata_endpoint: str,
) -> TrainerEngineAdapter:
    engine = rl_envs.MX_TRAINER_ENGINE
    if engine == "MEGATRON":
        from .engines.megatron import MegatronTrainerAdapter

        adapter_type = MegatronTrainerAdapter
    elif engine == "FSDP":
        from .engines.fsdp import FSDPTrainerAdapter

        adapter_type = FSDPTrainerAdapter
    else:
        raise ValueError(f"unsupported MX_TRAINER_ENGINE={engine!r}")
    return adapter_type(manager=manager, nixl_metadata_endpoint=nixl_metadata_endpoint)


def _nixl_metadata_endpoint(manager: NixlMetadataProvider) -> str:
    host = _required(envs.MX_WORKER_HOST, "MX_WORKER_HOST")
    if manager.listen_port is None:
        raise ValueError("NIXL manager must have a metadata listen port")
    return f"{host}:{manager.listen_port}"


def _staging_mode(value: TrainerStagingMode | None) -> TrainerStagingMode:
    try:
        return value or TrainerStagingMode(rl_envs.MX_TRAINER_STAGING_MODE)
    except ValueError as error:
        raise ValueError(
            f"invalid MX_TRAINER_STAGING_MODE={rl_envs.MX_TRAINER_STAGING_MODE!r}"
        ) from error


def _payload_format(value: WeightPayloadFormat | None) -> WeightPayloadFormat:
    try:
        return value or WeightPayloadFormat(rl_envs.MX_WEIGHT_PAYLOAD_FORMAT)
    except ValueError as error:
        raise ValueError(
            f"invalid MX_WEIGHT_PAYLOAD_FORMAT={rl_envs.MX_WEIGHT_PAYLOAD_FORMAT!r}"
        ) from error


@dataclass(frozen=True)
class S3Config:
    """Storage and launch-base settings for canonical XOR publication."""

    bucket: str
    initial_base_version_id: str
    launch_checkpoint: str | Path
    prefix: str = ""
    endpoint_url: str | None = None
    region_name: str | None = None

    def __post_init__(self) -> None:
        _required(self.bucket, "s3.bucket")
        _required(self.initial_base_version_id, "s3.initial_base_version_id")
        if not str(self.launch_checkpoint).strip():
            raise ValueError("s3.launch_checkpoint is required")


@dataclass
class _StagedDelta:
    base_version_id: str
    target_version_id: str
    candidate_snapshot: dict[str, np.ndarray]
    raw_deltas: dict[str, np.ndarray]
    changed_bytes: int
    total_bytes: int
    wire_bytes: int = 0


@dataclass(frozen=True)
class _S3Root:
    location: S3Object
    manifest_digest: str
    tensor_count: int
    total_bytes: int


def _s3_key(prefix: str, model_name: str, version_id: str, filename: str) -> str:
    path = (
        f"models/{quote(model_name, safe='')}/revisions/"
        f"{quote(version_id, safe='')}/canonical/{filename}"
    )
    return "/".join(part for part in (prefix.strip("/"), path) if part)


@dataclass(frozen=True)
class ModelExpressTrainerConfig:
    """Immutable configuration for one rank-local trainer client."""

    # CUDA device used by the rank-local NIXL manager; defaults to LOCAL_RANK.
    device_id: int | None = None
    # NIXL process identity; generated from the distributed rank when omitted.
    agent_name: str | None = None
    # Logical model identity; defaults to MODEL_NAME.
    model_name: str | None = None
    # How trainer tensors are staged; defaults to MX_TRAINER_STAGING_MODE.
    staging_mode: TrainerStagingMode | None = None
    # Weight representation published by this client; defaults to MX_WEIGHT_PAYLOAD_FORMAT.
    payload_format: WeightPayloadFormat | None = None
    # Fresh process-lifetime identity; generated when omitted.
    worker_id: str | None = None
    # Address of the central ModelExpress server; uses the standard MX default.
    server_url: str | None = None
    # Worker registration lifetime; defaults to three heartbeat intervals.
    registration_ttl_seconds: int | None = None
    # Deadline applied independently to each control-plane RPC.
    rpc_timeout_seconds: float = 30.0
    # Process group used by collective trainer publication.
    process_group: Any | None = None
    # Canonical S3/XOR settings. Omit to use the existing NIXL path.
    s3: S3Config | None = None

    def __post_init__(self) -> None:
        """Validate explicit settings before client initialization."""
        if self.payload_format is WeightPayloadFormat.UNSPECIFIED:
            raise ValueError("payload_format must be specified")
        if self.registration_ttl_seconds is not None:
            rl_envs.require_positive_int(
                self.registration_ttl_seconds, "registration_ttl_seconds"
            )
        rl_envs.require_positive_float(self.rpc_timeout_seconds, "rpc_timeout_seconds")


class StagedWeightVersionShard:
    """One immutable rank-local shard staged for a global weight version."""

    def __init__(
        self,
        *,
        client: ModelExpressTrainerClient,
        version: WeightVersionRef,
        staged: StagedWeightVersionShardData | _StagedDelta,
    ) -> None:
        self._client = client
        self._version = version
        self._staged = staged
        self._publish_lock = threading.Lock()
        self._published = False

    def publish(self) -> None:
        """Publish this staged shard; repeated calls are idempotent."""
        with self._publish_lock:
            if self._published:
                return
            self._client._publish_staged_shard(
                version=self._version,
                staged=self._staged,
            )
            self._published = True


class ModelExpressTrainerClient:
    """Rank-local capture, staging, and publication client for trainer actors."""

    def __init__(self) -> None:
        self._channel: grpc.Channel | None = None
        self._stub: refit_pb2_grpc.RefitServiceStub | None = None
        self._published_shards: dict[
            str, list[StagedWeightVersionShardData | _StagedDelta]
        ] = {}
        self._registration_stop = threading.Event()
        self._registration_thread: threading.Thread | None = None
        self._adapter: TrainerEngineAdapter | None = None
        self._resources: _TrainerResources | None = None
        self._s3_config: S3Config | None = None
        self._s3: S3Client | None = None
        self._process_group: Any | None = None
        self._rank = 0
        self._world_size = 1
        self._read_launch_tensor: Callable[[str], np.ndarray] | None = None
        self._current_base_version_id: str | None = None
        self._snapshot: dict[str, np.ndarray] = {}
        self._metric_delta: _StagedDelta | None = None
        self._metrics: dict[str, float] = {}
        self._bound_tensors: Any | None = None
        self._closed = False

    @property
    def source_slot_id(self) -> str:
        """Return the logical source contribution owned by this client."""
        if self._s3 is not None:
            return CANONICAL_DELTA_SOURCE_SLOT
        return self._get_adapter().source_slot_id

    def bind_tensors(self, tensors: Any) -> str:
        """Bind the stable engine tensors used by subsequent publications."""
        if self._closed:
            raise RuntimeError("trainer client is closed")
        if self._s3 is not None:
            raise RuntimeError("S3 publication uses a per-version tensor stream")
        if tensors is None:
            raise ValueError("tensors must not be None")
        if self._bound_tensors is not None:
            raise RuntimeError("trainer tensors are already bound")
        source_slot_id = self._get_adapter().bind_tensors(tensors)
        self._bound_tensors = tensors
        return source_slot_id

    @classmethod
    def initialize(
        cls,
        config: ModelExpressTrainerConfig,
    ) -> ModelExpressTrainerClient:
        """Initialize a trainer worker and connect it to the MX control plane.

        ModelExpress owns the rank-local transport, manifest service, and engine
        adapter. ``config`` contains only framework-provided settings.
        """
        if not isinstance(config, ModelExpressTrainerConfig):
            raise TypeError("config must be a ModelExpressTrainerConfig")
        model_name = _required(config.model_name or envs.MODEL_NAME or "", "model_name")
        staging_mode = _staging_mode(config.staging_mode)
        payload_format = _payload_format(config.payload_format)
        worker_id = _required(config.worker_id or uuid.uuid4().hex[:8], "worker_id")
        if staging_mode is TrainerStagingMode.UNSPECIFIED:
            raise ValueError("staging_mode must be specified")
        if payload_format is WeightPayloadFormat.UNSPECIFIED:
            raise ValueError("payload_format must be specified")
        registration_ttl_seconds = config.registration_ttl_seconds
        if registration_ttl_seconds is None:
            registration_ttl_seconds = envs.MX_HEARTBEAT_INTERVAL_SECS * 3
        registration_ttl_seconds = rl_envs.require_positive_int(
            registration_ttl_seconds, "registration_ttl_seconds"
        )
        use_s3 = config.s3 is not None
        if use_s3 and (
            staging_mode is not TrainerStagingMode.WRITE_TO_STORAGE
            or payload_format is not WeightPayloadFormat.XOR_DELTA
        ):
            raise ValueError("S3 publication requires WRITE_TO_STORAGE and XOR_DELTA")
        if not use_s3 and staging_mode is TrainerStagingMode.WRITE_TO_STORAGE:
            raise ValueError("WRITE_TO_STORAGE requires config.s3")

        resources = None
        if not use_s3:
            device_id = config.device_id
            if device_id is None:
                local_rank = os.environ.get("LOCAL_RANK")
                if local_rank is None:
                    raise ValueError("config.device_id or LOCAL_RANK is required")
                device_id = int(local_rank)
            resources = _TrainerResources.initialize(
                device_id=device_id,
                agent_name=config.agent_name,
            )

        client = cls()
        client.model_name = model_name
        client.staging_mode = staging_mode
        client.payload_format = payload_format
        client.worker_id = worker_id
        client.server_url = _get_server_url(config.server_url)
        client._adapter = None
        if resources is not None:
            client.worker_endpoint = resources.worker_endpoint
            client._manager = resources.manager
            client._nixl_metadata_endpoint = _nixl_metadata_endpoint(resources.manager)
            client._manifest_publisher = resources.manifest_service
        else:
            assert config.s3 is not None
            client.worker_endpoint = ""
            client._s3_config = config.s3
            client._process_group = config.process_group
            client._rank = dist.get_rank(config.process_group)
            client._world_size = dist.get_world_size(config.process_group)
            client._read_launch_tensor, _ = make_tensor_reader(config.s3.launch_checkpoint)
            client._s3 = S3Client(
                endpoint_url=config.s3.endpoint_url,
                region_name=config.s3.region_name,
            )
            client._current_base_version_id = config.s3.initial_base_version_id
        client._resources = resources
        client._registration_ttl_seconds = registration_ttl_seconds
        client._rpc_timeout_seconds = config.rpc_timeout_seconds
        try:
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

    @staticmethod
    def _validate_adapter(
        adapter: TrainerEngineAdapter,
        staging_mode: TrainerStagingMode,
        payload_format: WeightPayloadFormat,
    ) -> None:
        if staging_mode not in adapter.supported_staging_modes:
            raise ValueError(
                f"adapter does not support staging mode {staging_mode.value}"
            )
        if payload_format not in adapter.supported_payload_formats:
            raise ValueError(
                f"adapter does not support payload format {payload_format.value}"
            )

    def _get_adapter(self) -> TrainerEngineAdapter:
        if self._closed:
            raise RuntimeError("trainer client is closed")
        if self._s3 is not None:
            raise RuntimeError("S3 publication does not use a NIXL engine adapter")
        if self._adapter is None:
            try:
                adapter = _trainer_adapter(
                    manager=self._manager,
                    nixl_metadata_endpoint=self._nixl_metadata_endpoint,
                )
                self._validate_adapter(adapter, self.staging_mode, self.payload_format)
                self._adapter = adapter
            except Exception:
                self.close()
                raise
        return self._adapter

    @property
    def _service(self) -> refit_pb2_grpc.RefitServiceStub:
        if self._channel is None:
            self._channel = auth.with_auth(grpc.insecure_channel(self.server_url))
            self._stub = refit_pb2_grpc.RefitServiceStub(self._channel)
        assert self._stub is not None
        return self._stub

    def _register_worker(self) -> None:
        self._service.RegisterWorker(
            refit_pb2.RegisterWorkerRequest(
                worker=refit_pb2.WorkerRegistration(
                    worker_id=self.worker_id,
                    role=refit_pb2.WORKER_ROLE_TRAINER,
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
            except grpc.RpcError:
                # A later renewal retries after transient control-plane failure.
                continue

    def _stage_delta(
        self,
        *,
        target_version_id: str,
        base_version_id: str,
        hf_tensor_iter: Iterable[tuple[str, torch.Tensor]],
    ) -> _StagedDelta:
        if base_version_id != self._current_base_version_id:
            raise RuntimeError(
                f"target base {base_version_id!r} does not match retained base "
                f"{self._current_base_version_id!r}"
        )
        candidate: dict[str, np.ndarray] = {}
        raw_deltas: dict[str, np.ndarray] = {}
        changed_bytes = 0
        total_bytes = 0

        for source_name, tensor in hf_tensor_iter:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"canonical tensor {source_name!r} must be a torch.Tensor"
                )
            name = source_name.removeprefix("module.")
            current = (
                tensor.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .copy()
            )
            if name in candidate:
                if not np.array_equal(current, candidate[name]):
                    raise RuntimeError(
                        f"duplicate canonical tensor {name!r} differs"
                    )
                continue
            base = self._snapshot.get(name)
            if base is None:
                assert self._read_launch_tensor is not None
                base = np.asarray(
                    self._read_launch_tensor(name), dtype=np.uint8
                ).copy()
            if base.nbytes != current.nbytes:
                raise RuntimeError(
                    f"canonical tensor {name!r} changed byte size"
                )
            delta, changed = compute_delta(current, base)
            candidate[name] = current
            changed_bytes += changed
            total_bytes += int(current.nbytes)
            if delta is not None:
                raw_deltas[name] = delta
        return _StagedDelta(
            base_version_id=base_version_id,
            target_version_id=target_version_id,
            candidate_snapshot=candidate,
            raw_deltas=raw_deltas,
            changed_bytes=changed_bytes,
            total_bytes=total_bytes,
        )

    def _publish_delta_to_s3(self, staged: _StagedDelta) -> _S3Root | None:
        if staged.base_version_id != self._current_base_version_id:
            raise RuntimeError("staged canonical delta is stale")
        assert self._s3_config is not None
        assert self._s3 is not None

        counts: list[Any] = [None] * self._world_size
        dist.all_gather_object(
            counts,
            (self._rank, int(bool(staged.raw_deltas))),
            group=self._process_group,
        )
        counts.sort()
        offset = sum(count for rank, count in counts if rank < self._rank)
        total = sum(count for _rank, count in counts)

        local_map: dict[str, str] = {}
        shard_size = 0
        if staged.raw_deltas:
            filename = f"model-{offset:05d}-of-{total:05d}.safetensors"
            encoded = {
                name: compress_delta(staged.raw_deltas[name])
                for name in sorted(staged.raw_deltas)
            }
            checksums = {
                name: adler32_checksum(staged.candidate_snapshot[name])
                for name in encoded
            }
            shard = safetensors.numpy.save(encoded, metadata=checksums)
            self._s3.put(
                bucket=self._s3_config.bucket,
                key=_s3_key(
                    self._s3_config.prefix,
                    self.model_name,
                    staged.target_version_id,
                    filename,
                ),
                data=shard,
            )
            shard_size = len(shard)
            staged.wire_bytes = shard_size
            local_map = {name: filename for name in staged.raw_deltas}

        contributions = [None] * self._world_size if self._rank == 0 else None
        dist.gather_object(
            (self._rank, local_map, shard_size),
            contributions,
            dst=0,
            group=self._process_group,
        )
        if contributions is None:
            return None

        weight_map = {}
        for rank, rank_map, _size in contributions:
            for name, shard_name in rank_map.items():
                if name in weight_map:
                    raise RuntimeError(
                        f"duplicate canonical tensor {name!r} from rank {rank}"
                    )
                weight_map[name] = shard_name
        index = json.dumps(
            {
                "metadata": {
                    "version": staged.target_version_id,
                    "base_version": staged.base_version_id,
                    "delta_encoding": "xor",
                    "compression_format": "zstd",
                    "checksum_format": "adler32",
                },
                "weight_map": weight_map,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        location = self._s3.put(
            bucket=self._s3_config.bucket,
            key=_s3_key(
                self._s3_config.prefix,
                self.model_name,
                staged.target_version_id,
                "model.safetensors.index.json",
            ),
            data=index,
        )
        return _S3Root(
            location=location,
            manifest_digest=hashlib.sha256(index).hexdigest(),
            tensor_count=len(weight_map),
            total_bytes=sum(size for _rank, _mapping, size in contributions),
        )

    def _release_delta(self, staged: _StagedDelta) -> None:
        if self._current_base_version_id == staged.target_version_id:
            raise RuntimeError("cannot release the current canonical base version")

    def stage_shard(
        self,
        *,
        version: WeightVersionRef,
        tensors: Any = None,
        hf_tensor_iter: Iterable[tuple[str, torch.Tensor]] | None = None,
    ) -> StagedWeightVersionShard:
        """Capture one immutable rank-local shard for ``version``.

        Engine adapters receive their native ``tensors`` input. Canonical S3
        publication instead consumes ``hf_tensor_iter`` as Hugging Face
        ``(name, tensor)`` pairs; framework ranks must partition canonical names
        without overlap.
        """
        if self._closed:
            raise RuntimeError("trainer client is closed")
        if not isinstance(version, WeightVersionRef):
            raise TypeError("version must be a WeightVersionRef")
        if self._s3 is not None:
            if tensors is not None:
                raise ValueError("S3 publication accepts hf_tensor_iter, not tensors")
            if hf_tensor_iter is None:
                raise ValueError("hf_tensor_iter is required for S3 publication")
            target = self._service.GetWeightVersion(
                refit_pb2.GetWeightVersionRequest(uid=version.version_id),
                timeout=self._rpc_timeout_seconds,
            )
            if target.model_name != self.model_name:
                raise RuntimeError("target weight version belongs to a different model")
            if (
                target.payload_format != refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA
                or not target.HasField("base_version_id")
            ):
                raise RuntimeError("S3 publication requires an XOR_DELTA target")
            if tuple(target.expected_source_slots) != (
                CANONICAL_DELTA_SOURCE_SLOT,
            ):
                raise RuntimeError("S3 target must expect only canonical.delta.root")
            staged = self._stage_delta(
                target_version_id=version.version_id,
                base_version_id=target.base_version_id,
                hf_tensor_iter=hf_tensor_iter,
            )
        else:
            if hf_tensor_iter is not None:
                raise ValueError("hf_tensor_iter is only supported for S3 publication")
            if tensors is None:
                raise ValueError("tensors is required for NIXL publication")
            staged = self._get_adapter().stage_shard(
                tensors=tensors,
                staging_mode=self.staging_mode,
                payload_format=self.payload_format,
            )
        return StagedWeightVersionShard(client=self, version=version, staged=staged)

    def publish_version(self, *, version: WeightVersionRef) -> None:
        """Stage and publish the bound tensors for one weight version."""
        if self._bound_tensors is None:
            raise RuntimeError("bind_tensors() must be called before publish_version()")
        self.stage_shard(version=version, tensors=self._bound_tensors).publish()

    def _publish_staged_shard(
        self,
        *,
        version: WeightVersionRef,
        staged: StagedWeightVersionShardData | _StagedDelta,
    ) -> None:
        if isinstance(staged, _StagedDelta):
            if self._s3 is None:
                raise RuntimeError("S3 transport is not initialized")
            root = self._publish_delta_to_s3(staged)

            def advertise() -> None:
                assert root is not None
                s3_fields = {
                    "bucket": root.location.bucket,
                    "key": root.location.key,
                    "checksum": root.location.checksum,
                }
                if root.location.object_version is not None:
                    s3_fields["object_version"] = root.location.object_version
                shard = refit_pb2.WeightVersionShard(
                    version_id=version.version_id,
                    source_slot_id=CANONICAL_DELTA_SOURCE_SLOT,
                    worker_id=self.worker_id,
                    tensor_count=root.tensor_count,
                    total_bytes=root.total_bytes,
                    manifest_digest=root.manifest_digest,
                    s3=refit_pb2.S3Transport(**s3_fields),
                )
                self._service.CreateWeightVersionShard(
                    refit_pb2.CreateWeightVersionShardRequest(shard=shard),
                    timeout=self._rpc_timeout_seconds,
                )

            if self._rank == 0:
                advertise()
            self._snapshot = staged.candidate_snapshot
            self._current_base_version_id = staged.target_version_id
            self._metric_delta = staged
            staged.raw_deltas.clear()
            self._published_shards.setdefault(version.version_id, []).append(staged)
            return

        source_slot_id = self._get_adapter().source_slot_id
        staged.publish_ready.wait()
        if staged.manifest.transport.upper() != "NIXL":
            raise ValueError(
                f"unsupported shard transport {staged.manifest.transport!r}"
            )
        manifest_endpoint = self._manifest_publisher.publish_manifest(
            version_id=version.version_id,
            source_slot_id=source_slot_id,
            manifest=staged.manifest,
        )
        shard = refit_pb2.WeightVersionShard(
            version_id=version.version_id,
            source_slot_id=source_slot_id,
            worker_id=self.worker_id,
            tensor_count=staged.manifest.tensor_count,
            total_bytes=staged.manifest.total_bytes,
            manifest_digest=staged.manifest.digest,
            nixl=refit_pb2.NixlTransport(
                manifest_endpoint=_required(manifest_endpoint, "manifest_endpoint"),
            ),
        )
        self._service.CreateWeightVersionShard(
            refit_pb2.CreateWeightVersionShardRequest(shard=shard),
            timeout=self._rpc_timeout_seconds,
        )
        # Keep the adapter-owned buffers alive while the published version can
        # still be selected as a source. Eviction/release is a later lifecycle
        # operation, not the staged handle's Python object lifetime.
        self._published_shards.setdefault(version.version_id, []).append(staged)

    def release_version(self, *, version: WeightVersionRef) -> None:
        """Withdraw this worker's shard after the version is retired.

        The framework must call this only after the control plane has moved the
        version to ``RELEASING``. Once the shard is deleted, ModelExpress no
        longer advertises this worker's buffers as a transfer source and an
        in-place trainer may resume mutating them.
        """
        if self._closed:
            raise RuntimeError("trainer client is closed")
        if not isinstance(version, WeightVersionRef):
            raise TypeError("version must be a WeightVersionRef")
        staged = self._published_shards.get(version.version_id)
        if staged is None:
            return
        if self._s3 is not None:
            if len(staged) != 1 or not isinstance(staged[0], _StagedDelta):
                raise RuntimeError("invalid retained canonical S3 shard")
            canonical = staged[0]
            self._release_delta(canonical)

            def delete() -> None:
                try:
                    self._service.DeleteWeightVersionShard(
                        refit_pb2.DeleteWeightVersionShardRequest(
                            version_id=version.version_id,
                            source_slot_id=CANONICAL_DELTA_SOURCE_SLOT,
                            worker_id=self.worker_id,
                        ),
                        timeout=self._rpc_timeout_seconds,
                    )
                except grpc.RpcError as error:
                    if error.code() != grpc.StatusCode.NOT_FOUND:
                        raise

            if self._rank == 0:
                delete()
            del self._published_shards[version.version_id]
            return
        self._service.DeleteWeightVersionShard(
            refit_pb2.DeleteWeightVersionShardRequest(
                version_id=version.version_id,
                source_slot_id=self._get_adapter().source_slot_id,
                worker_id=self.worker_id,
            ),
            timeout=self._rpc_timeout_seconds,
        )
        del self._published_shards[version.version_id]

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        if self._closed:
            return
        if self._registration_thread is not None:
            self._registration_stop.set()
            self._registration_thread.join()
            self._registration_thread = None
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
        self._published_shards.clear()
        self._bound_tensors = None
        if self._resources is not None:
            self._resources.close()
            self._resources = None
        if self._s3 is not None:
            self._s3.close()
            self._s3 = None
        self._closed = True

    def collect_metrics(self) -> None:
        """Collect the latest S3 publication metrics across trainer ranks."""
        staged = self._metric_delta
        if self._s3 is None or staged is None:
            return
        counts = torch.tensor(
            [staged.changed_bytes, staged.total_bytes, staged.wire_bytes],
            dtype=torch.int64,
        )
        dist.all_reduce(counts, group=self._process_group)
        changed_bytes, total_bytes, wire_bytes = counts.tolist()
        self._metrics = {
            "perf/update_weights_density": changed_bytes / max(total_bytes, 1),
            "perf/update_weights_wire_bytes": wire_bytes,
        }
        self._metric_delta = None

    def pop_metrics(self) -> dict[str, float]:
        """Return and clear trainer publication metrics."""
        metrics, self._metrics = self._metrics, {}
        return metrics

    def __enter__(self) -> ModelExpressTrainerClient:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


__all__ = [
    "ModelExpressTrainerClient",
    "ModelExpressTrainerConfig",
    "S3Config",
    "StagedWeightVersionShard",
]
