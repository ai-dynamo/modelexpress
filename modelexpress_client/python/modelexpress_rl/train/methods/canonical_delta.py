# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer canonical checkpoint publication to object storage."""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import safetensors.numpy
import safetensors.torch
import torch
import torch.distributed as dist

from ... import envs as rl_envs
from ... import refit_pb2, refit_pb2_grpc
from ...s3 import S3Client
from ...utils import (
    checksum_factory,
    compress_delta,
    compute_delta,
    threadpool_map,
)
from ...version import WeightVersionRef

logger = logging.getLogger("modelexpress_rl.train.client")


@dataclass
class StagedCanonicalDelta:
    base_version_id: str
    target_version_id: str
    object_storage_uri: str
    candidate_snapshot: dict[str, np.ndarray]
    encoded_deltas: dict[str, np.ndarray]
    checksums: dict[str, str]
    changed_bytes: int
    total_bytes: int
    wire_bytes: int = 0
    stage_delta_time: float = 0.0
    publish_object_storage_time: float = 0.0


@dataclass
class StagedFullCheckpoint:
    target_version_id: str
    object_storage_uri: str
    changed_bytes: int
    total_bytes: int
    wire_bytes: int = 0
    stage_delta_time: float = 0.0
    publish_object_storage_time: float = 0.0


def _batch_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    max_bytes: int,
) -> Iterator[dict[str, torch.Tensor]]:
    batch: dict[str, torch.Tensor] = {}
    batch_bytes = 0
    for name, tensor in tensors.items():
        tensor_bytes = tensor.numel() * tensor.element_size()
        if batch and batch_bytes + tensor_bytes > max_bytes:
            yield batch
            batch = {}
            batch_bytes = 0
        batch[name] = tensor
        batch_bytes += tensor_bytes
    if batch:
        yield batch


class CanonicalDeltaPublicationMethod:
    """Publish XOR deltas and full HF checkpoints at a selected cadence."""

    def __init__(
        self,
        *,
        config,
        model_name: str,
        service: Callable[[], refit_pb2_grpc.RefitServiceStub],
        rpc_timeout_seconds: float,
        process_group: Any,
        read_seed_tensor: Callable[[str], np.ndarray],
        s3: S3Client,
    ) -> None:
        self._config = config
        self._model_name = model_name
        self._service = service
        self._rpc_timeout_seconds = rpc_timeout_seconds
        self._process_group = process_group
        self._rank = dist.get_rank(process_group)
        self._world_size = dist.get_world_size(process_group)
        self._read_seed_tensor = read_seed_tensor
        self._s3 = s3
        self._checksum_format = rl_envs.MX_REFIT_CHECKSUM_FORMAT
        checksum_factory(self._checksum_format)
        self.current_base_version_id = config.initial_base_version_id
        self.snapshot: dict[str, np.ndarray | torch.Tensor] = {}
        self._staged: StagedCanonicalDelta | StagedFullCheckpoint | None = None
        self._metric_delta: StagedCanonicalDelta | StagedFullCheckpoint | None = None
        self._stage_threadpool: ThreadPoolExecutor | None = None
        self._stage_futures: deque[Future] = deque()
        self._stage_error: BaseException | None = None
        self._stage_started = 0.0
        self._stage_limit = 0

    def prepare_base(
        self,
        *,
        hf_tensor_iter: Iterable[list[tuple[str, torch.Tensor]]],
    ) -> None:
        if self._staged is not None:
            raise RuntimeError(
                "publish the staged canonical checkpoint before preparing a new base"
            )
        started = perf_counter()

        def read_bucket(
            bucket: list[tuple[str, torch.Tensor]],
        ) -> dict[str, np.ndarray]:
            return {
                name: np.asarray(self._read_seed_tensor(name), dtype=np.uint8)
                for name, _ in bucket
            }

        snapshot = {}
        for tensors in threadpool_map(
            (bucket for bucket in hf_tensor_iter if bucket),
            read_bucket,
            max_workers=rl_envs.MX_REFIT_DELTA_WORKERS,
            thread_name_prefix="modelexpress-delta-base",
        ):
            snapshot.update(tensors)
        self.snapshot = snapshot
        logger.info(
            "ModelExpress prepare_delta_base: rank=%d tensors=%d duration=%.3fs",
            self._rank,
            len(snapshot),
            perf_counter() - started,
        )

    def _process_delta_bucket(
        self,
        bucket: list[tuple[str, torch.Tensor]],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str], int, int]:
        candidate = {}
        encoded = {}
        checksums = {}
        changed_bytes = 0
        total_bytes = 0
        for name, tensor in bucket:
            current = (
                tensor.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .copy()
            )
            base = self.snapshot[name]
            if base.nbytes != current.nbytes:
                raise RuntimeError(f"canonical tensor {name!r} changed byte size")
            delta, changed = compute_delta(current, base)
            candidate[name] = current
            changed_bytes += changed
            total_bytes += int(current.nbytes)
            if delta is not None:
                encoded[name] = compress_delta(delta)
                checksum = checksum_factory(self._checksum_format)
                checksum.update(current)
                checksums[name] = checksum.hexdigest()
        return candidate, encoded, checksums, changed_bytes, total_bytes

    def stage(
        self,
        *,
        version: WeightVersionRef,
        hf_tensor_iter: Iterable[list[tuple[str, torch.Tensor]]],
    ) -> StagedCanonicalDelta | StagedFullCheckpoint:
        """Consume one iterator and complete staging before returning."""
        if self._staged is not None:
            if self._staged.target_version_id == version.version_id:
                self._finish_staging()
                return self._staged
            raise RuntimeError(
                "publish the staged canonical checkpoint before staging another"
            )
        self._begin_stage(version)
        try:
            for bucket in hf_tensor_iter:
                self.stage_bucket(version=version, bucket=bucket)
            self._finish_staging()
        except BaseException:
            self._finish_staging(discard=True)
            raise
        return self._staged

    def stage_bucket(
        self,
        *,
        version: WeightVersionRef,
        bucket: list[tuple[str, torch.Tensor]],
    ) -> StagedCanonicalDelta | StagedFullCheckpoint:
        """Enqueue one bucket; an empty bucket initializes a non-contributing rank."""
        if self._staged is None:
            self._begin_stage(version)
        elif self._staged.target_version_id != version.version_id:
            raise RuntimeError(
                "publish the staged canonical checkpoint before staging another"
            )
        if bucket:
            process = (
                self._process_full_checkpoint_bucket
                if isinstance(self._staged, StagedFullCheckpoint)
                else self._process_delta_bucket
            )
            self._stage_futures.append(self._stage_threadpool.submit(process, bucket))
            if len(self._stage_futures) >= self._stage_limit:
                self._collect_staged_bucket()
        return self._staged

    def _begin_stage(self, version: WeightVersionRef) -> None:
        """Called once per version; publish() clears the staged state for the next."""
        response = self._service().GetWeightVersion(
            refit_pb2.GetWeightVersionRequest(uid=version.version_id),
            timeout=self._rpc_timeout_seconds,
        )
        if not response.HasField("version"):
            raise RuntimeError("MX GetWeightVersion response is missing version")
        target = response.version
        if target.model_name != self._model_name:
            raise RuntimeError("target weight version belongs to a different model")
        if (
            not target.HasField("object_storage")
            or target.object_storage.storage_type != refit_pb2.OBJECT_STORAGE_TYPE_S3
            or not target.object_storage.uri
        ):
            raise RuntimeError("S3 target is missing its URI")
        uri_prefix = f"{self._config.uri_prefix.rstrip('/')}/"
        if not target.object_storage.uri.startswith(uri_prefix):
            raise RuntimeError("S3 target URI does not match the configured prefix")
        if target.payload_format == refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_HF_CHECKPOINT:
            if target.HasField("base_version_id"):
                raise RuntimeError(
                    "FULL_HF_CHECKPOINT target must not have base_version_id"
                )
            self.snapshot = {}
            self._staged = StagedFullCheckpoint(
                target_version_id=version.version_id,
                object_storage_uri=target.object_storage.uri,
                changed_bytes=0,
                total_bytes=0,
            )
        else:
            if (
                target.payload_format != refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA
                or not target.HasField("base_version_id")
            ):
                raise RuntimeError(
                    "S3 publication requires XOR_DELTA or FULL_HF_CHECKPOINT"
                )
            if target.base_version_id != self.current_base_version_id:
                raise RuntimeError(
                    f"target base {target.base_version_id!r} does not match retained base "
                    f"{self.current_base_version_id!r}"
                )
            self._staged = StagedCanonicalDelta(
                base_version_id=target.base_version_id,
                target_version_id=version.version_id,
                object_storage_uri=target.object_storage.uri,
                candidate_snapshot={},
                encoded_deltas={},
                checksums={},
                changed_bytes=0,
                total_bytes=0,
            )
        workers = rl_envs.MX_REFIT_DELTA_WORKERS
        self._stage_threadpool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="modelexpress-stage"
        )
        self._stage_limit = 2 * workers
        self._stage_started = perf_counter()

    @staticmethod
    def _process_full_checkpoint_bucket(
        bucket: list[tuple[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().to(device="cpu", copy=True).contiguous()
            for name, tensor in bucket
        }

    def _collect_staged_bucket(self) -> None:
        try:
            result = self._stage_futures[0].result()
        except BaseException as error:
            self._stage_error = error
            raise
        self._stage_futures.popleft()
        staged = self._staged
        if isinstance(staged, StagedFullCheckpoint):
            self.snapshot.update(result)
            size = sum(
                tensor.numel() * tensor.element_size() for tensor in result.values()
            )
            staged.changed_bytes += size
            staged.total_bytes += size
        else:
            current, encoded, checksums, changed, total = result
            staged.candidate_snapshot.update(current)
            staged.encoded_deltas.update(encoded)
            staged.checksums.update(checksums)
            staged.changed_bytes += changed
            staged.total_bytes += total

    def _finish_staging(self, *, discard: bool = False) -> None:
        try:
            if not discard:
                if self._stage_error is not None:
                    raise self._stage_error
                while self._stage_futures:
                    self._collect_staged_bucket()
                if self._stage_threadpool is not None:
                    self._staged.stage_delta_time = perf_counter() - self._stage_started
        finally:
            if self._stage_threadpool is not None:
                self._stage_threadpool.shutdown(wait=True, cancel_futures=True)
                self._stage_threadpool = None
            self._stage_futures.clear()
            if discard:
                self._staged = None
                self._stage_error = None

    def _publish_full_checkpoint_to_s3(self, staged: StagedFullCheckpoint) -> None:
        parent_uri = staged.object_storage_uri.rsplit("/", 1)[0]
        batches = list(
            _batch_tensors(
                self.snapshot,
                max_bytes=rl_envs.MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES,
            )
        )

        counts: list[Any] = [None] * self._world_size
        dist.all_gather_object(
            counts,
            (self._rank, len(batches)),
            group=self._process_group,
        )
        counts.sort()
        offset = sum(count for rank, count in counts if rank < self._rank)
        total = sum(count for _rank, count in counts)
        if total == 0:
            raise RuntimeError("FULL_HF_CHECKPOINT contains no tensors")

        uploads = [
            (
                f"model-{offset + index:05d}-of-{total:05d}.safetensors",
                batch,
            )
            for index, batch in enumerate(batches, start=1)
        ]

        def upload(
            item: tuple[str, dict[str, torch.Tensor]],
        ) -> tuple[dict[str, str], int]:
            filename, tensors = item
            checksums = {}
            for name, tensor in tensors.items():
                checksum = checksum_factory(self._checksum_format)
                checksum.update(tensor.reshape(-1).view(torch.uint8).numpy())
                checksums[name] = checksum.hexdigest()
            data = safetensors.torch.save(tensors, metadata=checksums)
            self._s3.put(uri=f"{parent_uri}/{filename}", data=data)
            return dict.fromkeys(tensors, filename), len(data)

        local_map: dict[str, str] = {}
        staged.wire_bytes = 0
        for batch_map, wire_bytes in threadpool_map(
            uploads,
            upload,
            max_workers=rl_envs.MX_S3_UPLOAD_WORKERS,
            thread_name_prefix="modelexpress-full-hf-upload",
        ):
            local_map.update(batch_map)
            staged.wire_bytes += wire_bytes

        contributions = [None] * self._world_size if self._rank == 0 else None
        dist.gather_object(
            (self._rank, local_map, staged.total_bytes),
            contributions,
            dst=0,
            group=self._process_group,
        )
        if contributions is None:
            return

        weight_map = {}
        total_size = 0
        for rank, rank_map, rank_size in contributions:
            total_size += rank_size
            for name, shard_name in rank_map.items():
                if name in weight_map:
                    raise RuntimeError(
                        f"duplicate canonical tensor {name!r} from rank {rank}"
                    )
                weight_map[name] = shard_name
        index = json.dumps(
            {
                "metadata": {
                    "total_size": total_size,
                    "checksum_format": self._checksum_format,
                },
                "weight_map": weight_map,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self._s3.put(uri=staged.object_storage_uri, data=index)

    def publish(self, *, version: WeightVersionRef, staged: object) -> None:
        if not isinstance(staged, (StagedCanonicalDelta, StagedFullCheckpoint)):
            raise TypeError("canonical publication received an invalid artifact")
        if staged is not self._staged or version.version_id != staged.target_version_id:
            raise RuntimeError("canonical staged artifact is no longer active")

        # Wait for all bucket processing to finish before uploading.
        self._finish_staging()

        if isinstance(staged, StagedFullCheckpoint):
            started = perf_counter()
            self._publish_full_checkpoint_to_s3(staged)
            staged.publish_object_storage_time = perf_counter() - started
            self.snapshot = {
                name: tensor.reshape(-1).view(torch.uint8).numpy()
                for name, tensor in self.snapshot.items()
            }
            self.current_base_version_id = staged.target_version_id
            self._metric_delta = staged
            self._staged = None
            return
        if staged.base_version_id != self.current_base_version_id:
            raise RuntimeError("staged canonical delta is stale")
        started = perf_counter()
        parent_uri = staged.object_storage_uri.rsplit("/", 1)[0]
        counts: list[Any] = [None] * self._world_size
        dist.all_gather_object(
            counts,
            (self._rank, int(bool(staged.encoded_deltas))),
            group=self._process_group,
        )
        counts.sort()
        offset = sum(count for rank, count in counts if rank < self._rank)
        total = sum(count for _rank, count in counts)

        local_map: dict[str, str] = {}
        shard_size = 0
        if staged.encoded_deltas:
            shard = safetensors.numpy.save(
                staged.encoded_deltas, metadata=staged.checksums
            )
            filename = f"model-{offset:05d}-of-{total:05d}.safetensors"
            self._s3.put(uri=f"{parent_uri}/{filename}", data=shard)
            shard_size = len(shard)
            staged.wire_bytes = shard_size
            local_map = dict.fromkeys(staged.encoded_deltas, filename)

        contributions = [None] * self._world_size if self._rank == 0 else None
        dist.gather_object(
            (self._rank, local_map, shard_size),
            contributions,
            dst=0,
            group=self._process_group,
        )
        index_error: Exception | None = None
        index_error_message = None
        if contributions is not None:
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
                        "checksum_format": self._checksum_format,
                    },
                    "weight_map": weight_map,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            try:
                self._s3.put(uri=staged.object_storage_uri, data=index)
            except Exception as error:  # noqa: BLE001 - synchronize S3 failures.
                index_error = error
                index_error_message = f"{type(error).__name__}: {error}"

        index_errors = [None] * self._world_size
        dist.all_gather_object(
            index_errors,
            index_error_message,
            group=self._process_group,
        )
        remote_error = next((error for error in index_errors if error), None)
        if remote_error is not None:
            if index_error is not None:
                raise index_error
            raise RuntimeError(
                f"canonical delta index publication failed on rank 0: {remote_error}"
            )

        staged.publish_object_storage_time = perf_counter() - started
        self.snapshot = staged.candidate_snapshot
        staged.candidate_snapshot = {}
        self.current_base_version_id = staged.target_version_id
        self._metric_delta = staged
        staged.encoded_deltas.clear()
        staged.checksums.clear()
        self._staged = None

    def pop_metrics(self) -> dict[str, int | float]:
        staged = self._metric_delta
        if staged is None:
            return {}
        self._metric_delta = None
        return {
            "changed_bytes": staged.changed_bytes,
            "total_bytes": staged.total_bytes,
            "wire_bytes": staged.wire_bytes,
            "stage_delta_time": staged.stage_delta_time,
            "publish_object_storage_time": staged.publish_object_storage_time,
        }

    def close(self) -> None:
        self._finish_staging(discard=True)
        self.snapshot = {}
        self._metric_delta = None
        self._s3.close()


__all__ = [
    "CanonicalDeltaPublicationMethod",
    "StagedCanonicalDelta",
    "StagedFullCheckpoint",
]
