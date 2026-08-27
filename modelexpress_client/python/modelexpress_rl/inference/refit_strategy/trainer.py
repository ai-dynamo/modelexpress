# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer-publication refit strategy."""

import hashlib
from collections import defaultdict
from collections.abc import Callable

import grpc

from ... import refit_pb2, refit_pb2_grpc
from ...control import WeightVersion
from ..adapter import (
    GeneratorEngineAdapter,
    GeneratorSource,
    GeneratorTransferInputs,
    NixlGeneratorSource,
)
from .base import RefitStrategy


class _TrainerRefitStrategy(RefitStrategy):
    """Stage from the trainer publications attached to a weight version."""

    def __init__(
        self,
        *,
        adapter: GeneratorEngineAdapter,
        service: Callable[[], refit_pb2_grpc.RefitServiceStub],
        max_transfer_attempts: int,
        rpc_timeout_seconds: float,
    ) -> None:
        self._adapter = adapter
        self._service = service
        self._max_transfer_attempts = max_transfer_attempts
        self._rpc_timeout_seconds = rpc_timeout_seconds
        self._cached_plan: object | None = None
        self._cached_fingerprint: tuple | None = None

    def stage(self, version: WeightVersion) -> object:
        if version.payload_format not in self._adapter.supported_payload_formats:
            raise RuntimeError(
                "generator adapter does not support weight version payload format "
                f"{version.payload_format.value}"
            )
        last_error: grpc.RpcError | RuntimeError | None = None
        for attempt in range(self._max_transfer_attempts):
            try:
                inputs = self._discover_sources(version, candidate_offset=attempt)
                return self._adapter.stage_weight(self._transfer_plan(inputs))
            except (grpc.RpcError, RuntimeError) as error:
                last_error = error
                # A failed transfer may have invalidated transport state even
                # when the source metadata fingerprint is unchanged.
                self._cached_plan = None
                self._cached_fingerprint = None
        assert last_error is not None
        raise last_error

    def _resolve_source(self, shard: refit_pb2.WeightVersionShard) -> GeneratorSource:
        if not shard.manifest_endpoint:
            raise RuntimeError("NIXL source is missing its manifest endpoint")
        with grpc.insecure_channel(shard.manifest_endpoint) as channel:
            response = refit_pb2_grpc.RefitWorkerServiceStub(
                channel
            ).GetWeightVersionShardManifest(
                refit_pb2.GetWeightVersionShardManifestRequest(
                    version_id=shard.version_id,
                    source_slot_id=shard.source_slot_id,
                ),
                timeout=self._rpc_timeout_seconds,
            )
        digest = hashlib.sha256(response.manifest).hexdigest()
        if (
            response.manifest_digest != shard.manifest_digest
            or digest != shard.manifest_digest
        ):
            raise RuntimeError(
                f"manifest digest mismatch for source slot {shard.source_slot_id!r}"
            )
        if not shard.manifest_digest:
            raise RuntimeError("source is missing its manifest digest")
        return GeneratorSource(
            source_slot_id=shard.source_slot_id,
            worker_id=shard.worker_id,
            manifest_digest=shard.manifest_digest,
            transport=NixlGeneratorSource(
                manifest_endpoint=shard.manifest_endpoint,
                manifest=response.manifest,
            ),
        )

    def _discover_sources(
        self,
        version: WeightVersion,
        *,
        candidate_offset: int,
    ) -> GeneratorTransferInputs:
        response = self._service().ListWeightVersionShards(
            refit_pb2.ListWeightVersionShardsRequest(version_id=version.version_id),
            timeout=self._rpc_timeout_seconds,
        )
        candidates = defaultdict(list)
        for shard in response.shards:
            candidates[shard.source_slot_id].append(shard)

        selected = []
        for source_slot_id in version.expected_source_slots:
            failures = []
            ordered = sorted(
                candidates[source_slot_id], key=lambda item: item.worker_id
            )
            if ordered:
                offset = candidate_offset % len(ordered)
                ordered = ordered[offset:] + ordered[:offset]
            for shard in ordered:
                try:
                    source = self._resolve_source(shard)
                except (grpc.RpcError, RuntimeError) as error:
                    failures.append(str(error))
                    continue
                selected.append(source)
                break
            else:
                detail = f": {failures[-1]}" if failures else ""
                raise RuntimeError(
                    f"no usable source for required slot {source_slot_id!r}{detail}"
                )

        return GeneratorTransferInputs(
            version_id=version.version_id,
            base_version_id=version.base_version_id,
            layout_signature=version.layout_signature,
            payload_format=version.payload_format,
            sources=tuple(selected),
        )

    def _transfer_plan(self, inputs: GeneratorTransferInputs) -> object:
        reusable = (
            self._cached_plan is not None
            and self._cached_fingerprint == inputs.physical_fingerprint
            and self._adapter.validate_transfer_plan(self._cached_plan, inputs)
        )
        if not reusable:
            self._cached_plan = self._adapter.create_transfer_plan(inputs)
            self._cached_fingerprint = inputs.physical_fingerprint
        assert self._cached_plan is not None
        return self._cached_plan


__all__: list[str] = []
