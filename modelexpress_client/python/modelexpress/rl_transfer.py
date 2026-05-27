# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-agnostic RL weight-transfer helpers backed by MX/NIXL."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch

from modelexpress import p2p_pb2
from modelexpress.client import MxClientBase
from modelexpress.metadata.heartbeat import HeartbeatThread
from modelexpress.nixl_transfer import NixlTransferManager
from modelexpress.rl_fanin_transfer import (
    DenseFanInReceiveResult,
    candidate_fanin_groups,
    candidate_group_label,
    execute_dense_fanin_receive,
    prepare_dense_fanin_receive,
    preferred_dense_fanin_groups,
    source_descriptors_from_worker,
)
from modelexpress.rl_metadata import (
    RlSourceCandidate,
    RlSourceMetadata,
    RlSourceRole,
    select_rl_source_candidates,
    with_rl_source_metadata,
)
from modelexpress.rl_publication import (
    RlPublicationStore,
    RlPublishedSource,
    snapshot_tensors_for_retention,
)
from modelexpress.rl_reshard import (
    TensorReceiveSpec,
    plan_dense_reshard_transfers,
    plan_exact_transfers,
    receive_specs_from_shape_registry,
    receive_specs_from_tensors,
    source_specs_from_shape_registry,
    tensor_metadata_from_receive_specs,
)
from modelexpress.rl_shape_registry import (
    allocate_tensors_from_shape_registry,
    shape_registry_from_tensors,
    torch_dtype_from_string,
)
from modelexpress.rl_slice_transfer import build_slice_transfer_manifest
from modelexpress.rl_transfer_identity import (
    backend_framework_value,
    build_rl_base_identity,
    candidates_for_base_identity,
    identity_matches_base,
)
from modelexpress.rl_transfer_lease import (
    LeasedTransferError,
    RlTransferLeaseCoordinator,
    RlTransferLeaseInventory,
    transfer_lease_key,
)
from modelexpress.rl_transfer_report import (
    failed_attempt_from_candidate,
    RlTransferAttempt,
    RlTransferReport,
    successful_attempt_from_candidate,
    _ReceiveCandidateResult,
)

logger = logging.getLogger("modelexpress.rl_transfer")

_DEFAULT_RECEIVE_ROLES = (RlSourceRole.INFERENCE_REPLICA, RlSourceRole.TRAINER)


class RlNixlWeightTransfer:
    """Publish and receive versioned RL weights through MX metadata and NIXL."""

    def __init__(
        self,
        *,
        mx_client: MxClientBase,
        base_identity: "p2p_pb2.SourceIdentity",
        worker_id: str,
        retain_latest_k: int = 1,
        device_id: int | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        if retain_latest_k <= 0:
            raise ValueError("retain_latest_k must be positive")
        self.mx_client = mx_client
        self.base_identity = base_identity
        self.worker_id = worker_id
        self.retain_latest_k = retain_latest_k
        self.device_id = device_id
        self.timeout_seconds = timeout_seconds
        self._nixl_manager: NixlTransferManager | None = None
        self._target_nixl_manager: NixlTransferManager | None = None
        self._publication_store = RlPublicationStore(
            mx_client=mx_client,
            worker_id=worker_id,
            retain_latest_k=retain_latest_k,
        )
        self._published_sources = self._publication_store.sources
        self._published_tensors: dict[str, torch.Tensor] = {}
        self._mx_source_id: str | None = None
        self._worker_rank = 0
        self._heartbeat: HeartbeatThread | None = None
        self.last_receive_report: RlTransferReport | None = None
        self._lease_coordinator = RlTransferLeaseCoordinator(
            mx_client=mx_client,
            target_worker_id=worker_id,
            ttl_seconds=min(max(timeout_seconds, 5.0), 30.0),
        )

    def finalize(self) -> None:
        """Tear down transient transfer state and stop advertising this source."""
        if self._published_sources:
            self._publication_store.close_all(mark_stale=True)
            self._refresh_current_publication()
        else:
            self.mark_current_source_stale()
        self._shutdown_target_nixl_manager()

    def list_target_transfer_leases(
        self,
        *,
        mx_source_id: str = "",
        statuses: Iterable[int] | None = None,
    ) -> RlTransferLeaseInventory:
        return self._lease_coordinator.list_target_leases(
            mx_source_id=mx_source_id,
            statuses=statuses,
        )

    def publish_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        model_version: int,
        role: RlSourceRole = RlSourceRole.TRAINER,
        worker_rank: int = 0,
        source_world_size: int = 1,
        tensor_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> str:
        """Publish CUDA tensors as a READY MX source for one model version."""
        if not tensors:
            raise RuntimeError("ModelExpress RL transfer got no tensors to publish")

        self._publication_store.close_duplicate(
            role=role,
            model_version=model_version,
            worker_rank=worker_rank,
        )
        self._refresh_current_publication()
        if self.retain_latest_k == 1:
            self._publication_store.close_all(mark_stale=True)
            self._refresh_current_publication()
        publish_tensors = (
            snapshot_tensors_for_retention(tensors)
            if self.retain_latest_k > 1
            else tensors
        )

        device_id = self.resolve_device_id(publish_tensors.values())
        manager = NixlTransferManager(
            agent_name=f"mx-rl-source-{uuid.uuid4().hex[:12]}",
            device_id=device_id,
        )
        manager.initialize()
        manager.register_tensors(publish_tensors)

        metadata = RlSourceMetadata(
            model_version=model_version,
            role=role,
            world_size=source_world_size,
            retain_latest_k=self.retain_latest_k,
            shape_registry=shape_registry_from_tensors(
                publish_tensors,
                tensor_metadata=tensor_metadata,
            ),
        )
        identity = with_rl_source_metadata(self.base_identity, metadata)
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=worker_rank,
            nixl_metadata=manager.nixl_metadata,
            tensors=[
                p2p_pb2.TensorDescriptor(
                    name=tensor.name,
                    addr=tensor.addr,
                    size=tensor.size,
                    device_id=tensor.device_id,
                    dtype=tensor.dtype,
                )
                for tensor in manager.tensor_descriptors
            ],
            status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
        )
        mx_source_id = self.mx_client.publish_metadata(
            identity,
            worker,
            self.worker_id,
        )
        self.mx_client.update_status(
            mx_source_id,
            self.worker_id,
            worker_rank,
            p2p_pb2.SOURCE_STATUS_READY,
        )
        heartbeat = self._start_heartbeat(
            mx_source_id=mx_source_id,
            worker_rank=worker_rank,
            nixl_manager=manager,
        )
        published_source = RlPublishedSource(
            mx_source_id=mx_source_id,
            model_version=model_version,
            role=role,
            worker_rank=worker_rank,
            tensors=publish_tensors,
            manager=manager,
            heartbeat=heartbeat,
        )
        self._publication_store.add(published_source)
        self._refresh_current_publication()
        self._publication_store.prune()
        self._refresh_current_publication()
        logger.info(
            "Published ModelExpress RL weights: model=%s version=%d tensors=%d source_id=%s",
            self.base_identity.model_name,
            model_version,
            len(publish_tensors),
            mx_source_id,
        )
        return mx_source_id

    async def receive_tensors(
        self,
        *,
        model_version: int | None,
        receiver_rank: int,
        roles: Sequence[RlSourceRole] = _DEFAULT_RECEIVE_ROLES,
        same_rank_only: bool = False,
        source_ranks_by_role: Mapping[RlSourceRole, Sequence[int]] | None = None,
        require_complete_version: bool = True,
    ) -> list[tuple[str, torch.Tensor]]:
        """Discover, allocate, and pull a requested or latest model version."""
        _version, tensors = await self._receive_from_sources(
            model_version=model_version,
            receiver_rank=receiver_rank,
            roles=roles,
            same_rank_only=same_rank_only,
            source_ranks_by_role=source_ranks_by_role,
            require_complete_version=require_complete_version,
        )
        return tensors

    async def receive_into_tensors(
        self,
        target_tensors: dict[str, torch.Tensor],
        *,
        model_version: int | None,
        receiver_rank: int,
        roles: Sequence[RlSourceRole] = _DEFAULT_RECEIVE_ROLES,
        same_rank_only: bool = False,
        target_specs: Sequence[TensorReceiveSpec] | None = None,
        source_ranks_by_role: Mapping[RlSourceRole, Sequence[int]] | None = None,
        require_complete_version: bool = True,
    ) -> list[tuple[str, torch.Tensor]]:
        """Discover and pull a requested or latest version into caller tensors."""
        if not target_tensors:
            raise RuntimeError("ModelExpress RL transfer got no target tensors")
        _version, tensors = await self._receive_from_sources(
            model_version=model_version,
            receiver_rank=receiver_rank,
            roles=roles,
            same_rank_only=same_rank_only,
            target_tensors=target_tensors,
            target_specs=target_specs,
            source_ranks_by_role=source_ranks_by_role,
            require_complete_version=require_complete_version,
        )
        return tensors

    async def receive_and_publish_replica(
        self,
        target_tensors: dict[str, torch.Tensor],
        *,
        model_version: int | None,
        receiver_rank: int,
        roles: Sequence[RlSourceRole] = _DEFAULT_RECEIVE_ROLES,
        same_rank_only: bool = False,
        target_specs: Sequence[TensorReceiveSpec] | None = None,
        replica_world_size: int = 1,
        source_ranks_by_role: Mapping[RlSourceRole, Sequence[int]] | None = None,
        require_complete_version: bool = True,
    ) -> list[tuple[str, torch.Tensor]]:
        """Receive into caller tensors, then publish them as an inference replica."""
        if not target_tensors:
            raise RuntimeError("ModelExpress RL transfer got no target tensors")
        effective_target_specs = tuple(target_specs) if target_specs is not None else None
        received_version, tensors = await self._receive_from_sources(
            model_version=model_version,
            receiver_rank=receiver_rank,
            roles=roles,
            same_rank_only=same_rank_only,
            target_tensors=target_tensors,
            target_specs=effective_target_specs,
            source_ranks_by_role=source_ranks_by_role,
            require_complete_version=require_complete_version,
        )
        self._publish_replica_tensors(
            tensors,
            model_version=received_version,
            receiver_rank=receiver_rank,
            replica_world_size=replica_world_size,
            tensor_metadata=(
                tensor_metadata_from_receive_specs(effective_target_specs)
                if effective_target_specs is not None
                else None
            ),
        )
        return tensors

    async def receive_tensors_and_publish_replica(
        self,
        *,
        model_version: int | None,
        receiver_rank: int,
        roles: Sequence[RlSourceRole] = _DEFAULT_RECEIVE_ROLES,
        same_rank_only: bool = False,
        replica_world_size: int = 1,
        source_ranks_by_role: Mapping[RlSourceRole, Sequence[int]] | None = None,
        require_complete_version: bool = True,
    ) -> list[tuple[str, torch.Tensor]]:
        """Receive allocated tensors, then publish them as an inference replica."""
        received_version, tensors = await self._receive_from_sources(
            model_version=model_version,
            receiver_rank=receiver_rank,
            roles=roles,
            same_rank_only=same_rank_only,
            source_ranks_by_role=source_ranks_by_role,
            require_complete_version=require_complete_version,
        )
        self._publish_replica_tensors(
            tensors,
            model_version=received_version,
            receiver_rank=receiver_rank,
            replica_world_size=replica_world_size,
        )
        return tensors

    async def _receive_from_sources(
        self,
        *,
        model_version: int | None,
        receiver_rank: int,
        roles: Sequence[RlSourceRole],
        same_rank_only: bool,
        target_tensors: dict[str, torch.Tensor] | None = None,
        target_specs: Sequence[TensorReceiveSpec] | None = None,
        source_ranks_by_role: Mapping[RlSourceRole, Sequence[int]] | None = None,
        require_complete_version: bool = True,
    ) -> tuple[int, list[tuple[str, torch.Tensor]]]:
        self.last_receive_report = None
        candidates = await self.wait_for_sources(
            model_version=model_version,
            receiver_rank=receiver_rank,
            roles=roles,
            same_rank_only=same_rank_only,
            source_ranks_by_role=source_ranks_by_role,
            require_complete_version=require_complete_version,
        )
        errors = []
        attempts: list[RlTransferAttempt] = []
        fanin_groups = candidate_fanin_groups(candidates)
        preferred_fanin_groups = preferred_dense_fanin_groups(
            candidates,
            target_tensors=target_tensors,
            target_specs=target_specs,
            receiver_rank=receiver_rank,
            same_rank_only=same_rank_only,
        )
        if preferred_fanin_groups:
            received = self._receive_from_candidate_groups(
                preferred_fanin_groups,
                attempts=attempts,
                errors=errors,
                model_version=model_version,
                receiver_rank=receiver_rank,
                same_rank_only=same_rank_only,
                target_tensors=target_tensors,
                target_specs=target_specs,
            )
            if received is not None:
                return received
            fanin_groups = tuple(
                group for group in fanin_groups if group not in preferred_fanin_groups
            )
        for candidate in candidates:
            try:
                result = self._receive_from_candidate(
                    candidate,
                    candidate.metadata.model_version,
                    receiver_rank=receiver_rank,
                    same_rank_only=same_rank_only,
                    target_tensors=target_tensors,
                    target_specs=target_specs,
                )
                attempts.append(
                    successful_attempt_from_candidate(
                        candidate,
                        bytes_transferred=result.bytes_transferred,
                        tensor_count=result.tensor_count,
                        duration_seconds=result.duration_seconds,
                        lease_id=result.lease_id,
                    )
                )
                self.last_receive_report = RlTransferReport(
                    requested_model_version=model_version,
                    resolved_model_version=candidate.metadata.model_version,
                    receiver_rank=receiver_rank,
                    attempts=tuple(attempts),
                )
                return candidate.metadata.model_version, result.tensors
            except Exception as exc:
                errors.append(f"{candidate.mx_source_id}/{candidate.worker_id}: {exc}")
                attempts.append(failed_attempt_from_candidate(candidate, exc))
                logger.warning(
                    "ModelExpress RL source transfer failed; trying next candidate: "
                    "source_id=%s worker_id=%s",
                    candidate.mx_source_id,
                    candidate.worker_id,
                    exc_info=True,
                )
        received = self._receive_from_candidate_groups(
            fanin_groups,
            attempts=attempts,
            errors=errors,
            model_version=model_version,
            receiver_rank=receiver_rank,
            same_rank_only=same_rank_only,
            target_tensors=target_tensors,
            target_specs=target_specs,
        )
        if received is not None:
            return received
        self.last_receive_report = RlTransferReport(
            requested_model_version=model_version,
            resolved_model_version=None,
            receiver_rank=receiver_rank,
            attempts=tuple(attempts),
        )
        raise RuntimeError(
            "No ModelExpress RL source transfer succeeded for "
            f"model={self.base_identity.model_name!r} version={_version_label(model_version)}; "
            f"errors={errors}"
        )

    def _receive_from_candidate_groups(
        self,
        candidate_groups: Sequence[Sequence[RlSourceCandidate]],
        *,
        attempts: list[RlTransferAttempt],
        errors: list[str],
        model_version: int | None,
        receiver_rank: int,
        same_rank_only: bool,
        target_tensors: dict[str, torch.Tensor] | None,
        target_specs: Sequence[TensorReceiveSpec] | None,
    ) -> tuple[int, list[tuple[str, torch.Tensor]]] | None:
        for candidate_group in candidate_groups:
            try:
                result = self._receive_from_candidate_group(
                    candidate_group,
                    receiver_rank=receiver_rank,
                    same_rank_only=same_rank_only,
                    target_tensors=target_tensors,
                    target_specs=target_specs,
                )
                attempts.extend(
                    successful_attempt_from_candidate(
                        source_result.candidate,
                        bytes_transferred=source_result.bytes_transferred,
                        tensor_count=source_result.tensor_count,
                        duration_seconds=source_result.duration_seconds,
                        lease_id=source_result.lease_id,
                    )
                    for source_result in result.source_results
                )
                self.last_receive_report = RlTransferReport(
                    requested_model_version=model_version,
                    resolved_model_version=candidate_group[0].metadata.model_version,
                    receiver_rank=receiver_rank,
                    attempts=tuple(attempts),
                )
                return candidate_group[0].metadata.model_version, result.tensors
            except Exception as exc:
                errors.append(f"{candidate_group_label(candidate_group)}: {exc}")
                attempts.extend(
                    failed_attempt_from_candidate(candidate, exc)
                    for candidate in candidate_group
                )
                logger.warning(
                    "ModelExpress RL dense fan-in transfer failed; trying next group: "
                    "group=%s",
                    candidate_group_label(candidate_group),
                    exc_info=True,
                )
        return None

    def _publish_replica_tensors(
        self,
        tensors: Sequence[tuple[str, torch.Tensor]],
        *,
        model_version: int,
        receiver_rank: int,
        replica_world_size: int,
        tensor_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.publish_tensors(
            dict(tensors),
            model_version=model_version,
            role=RlSourceRole.INFERENCE_REPLICA,
            worker_rank=receiver_rank,
            source_world_size=replica_world_size,
            tensor_metadata=tensor_metadata,
        )

    def _receive_from_candidate(
        self,
        candidate: RlSourceCandidate,
        model_version: int,
        *,
        receiver_rank: int,
        same_rank_only: bool,
        target_tensors: dict[str, torch.Tensor] | None = None,
        target_specs: Sequence[TensorReceiveSpec] | None = None,
    ) -> _ReceiveCandidateResult:
        lease = self._lease_coordinator.lease_candidate(
            candidate,
            receiver_rank=receiver_rank,
        )
        try:
            with lease:
                result = self._receive_from_candidate_unleased(
                    candidate,
                    model_version,
                    receiver_rank=receiver_rank,
                    same_rank_only=same_rank_only,
                    target_tensors=target_tensors,
                    target_specs=target_specs,
                )
        except Exception as exc:
            if lease.lease_id:
                raise LeasedTransferError(
                    exc,
                    {transfer_lease_key(candidate): lease.lease_id},
                ) from exc
            raise
        return replace(result, lease_id=lease.lease_id)

    def _receive_from_candidate_unleased(
        self,
        candidate: RlSourceCandidate,
        model_version: int,
        *,
        receiver_rank: int,
        same_rank_only: bool,
        target_tensors: dict[str, torch.Tensor] | None = None,
        target_specs: Sequence[TensorReceiveSpec] | None = None,
    ) -> _ReceiveCandidateResult:
        metadata_resp = self.mx_client.get_metadata(candidate.mx_source_id, candidate.worker_id)
        if not metadata_resp.found:
            raise RuntimeError(
                f"ModelExpress source metadata not found for {candidate.mx_source_id}/"
                f"{candidate.worker_id}"
            )
        if not candidate.metadata.shape_registry:
            raise RuntimeError(
                f"ModelExpress source {candidate.mx_source_id} has no RL shape registry"
            )

        shape_registry = dict(candidate.metadata.shape_registry)
        if target_tensors is None:
            device_id = self.resolve_device_id()
            target_tensors = allocate_tensors_from_shape_registry(
                shape_registry,
                device=f"cuda:{device_id}",
            )
            effective_target_specs = receive_specs_from_shape_registry(
                shape_registry,
                receiver_rank=receiver_rank,
            )
        else:
            device_id = self.resolve_device_id(target_tensors.values())
            effective_target_specs = tuple(
                target_specs
                if target_specs is not None
                else receive_specs_from_tensors(target_tensors, receiver_rank=receiver_rank)
            )

        source_specs = source_specs_from_shape_registry(
            shape_registry,
            worker_rank=candidate.worker_rank,
        )
        plan = plan_exact_transfers(
            source_specs,
            effective_target_specs,
            same_rank_only=same_rank_only,
        )
        if not plan.complete:
            dense_plan = plan_dense_reshard_transfers(
                source_specs,
                effective_target_specs,
                same_rank_only=same_rank_only,
            )
            if dense_plan.complete:
                plan = dense_plan
        plan.raise_if_incomplete()

        planned_names = {entry.tensor_name for entry in plan.entries}
        target_tensors = {
            name: tensor
            for name, tensor in target_tensors.items()
            if name in planned_names
        }
        if set(target_tensors) != planned_names:
            missing_names = sorted(planned_names - set(target_tensors))
            raise RuntimeError(
                f"ModelExpress target tensors missing planned entries {missing_names}"
            )

        descriptors = [
            descriptor
            for descriptor in source_descriptors_from_worker(metadata_resp.worker)
            if descriptor.name in planned_names
        ]
        descriptor_names = {descriptor.name for descriptor in descriptors}
        if descriptor_names != planned_names:
            missing_names = sorted(planned_names - descriptor_names)
            raise RuntimeError(
                f"ModelExpress source descriptors missing planned entries {missing_names}"
            )
        transfer_manifest = build_slice_transfer_manifest(
            plan,
            source_descriptors=descriptors,
            target_tensors=target_tensors,
        )

        self._shutdown_target_nixl_manager()
        manager = NixlTransferManager(
            agent_name=f"mx-rl-target-{uuid.uuid4().hex[:12]}",
            device_id=device_id,
        )
        manager.initialize()
        manager.register_tensors(transfer_manifest.target_tensors)
        self._target_nixl_manager = manager
        self._nixl_manager = manager

        bytes_transferred, tensor_count, _duration = manager.receive_from_source(
            source_metadata=metadata_resp.worker.nixl_metadata,
            source_tensors=transfer_manifest.source_descriptors,
            timeout_seconds=self.timeout_seconds,
        )
        transfer_manifest.finalize()
        logger.info(
            "Received ModelExpress RL weights: model=%s version=%d tensors=%d bytes=%d",
            self.base_identity.model_name,
            model_version,
            tensor_count,
            bytes_transferred,
        )
        return _ReceiveCandidateResult(
            tensors=transfer_manifest.output_tensors,
            bytes_transferred=bytes_transferred,
            tensor_count=tensor_count,
            duration_seconds=_duration,
        )

    def _receive_from_candidate_group(
        self,
        candidates: Sequence[RlSourceCandidate],
        *,
        receiver_rank: int,
        same_rank_only: bool,
        target_tensors: dict[str, torch.Tensor] | None,
        target_specs: Sequence[TensorReceiveSpec] | None,
    ) -> DenseFanInReceiveResult:
        lease_ids_by_candidate: dict[tuple[str, str, int], str] = {}
        try:
            with contextlib.ExitStack() as stack:
                for candidate in candidates:
                    lease = self._lease_coordinator.lease_candidate(
                        candidate,
                        receiver_rank=receiver_rank,
                    )
                    stack.enter_context(lease)
                    if lease.lease_id:
                        lease_ids_by_candidate[transfer_lease_key(candidate)] = (
                            lease.lease_id
                        )
                result = self._receive_from_candidate_group_unleased(
                    candidates,
                    receiver_rank=receiver_rank,
                    same_rank_only=same_rank_only,
                    target_tensors=target_tensors,
                    target_specs=target_specs,
                )
        except Exception as exc:
            if lease_ids_by_candidate:
                raise LeasedTransferError(exc, lease_ids_by_candidate) from exc
            raise

        source_results = tuple(
            replace(
                source_result,
                lease_id=lease_ids_by_candidate.get(
                    transfer_lease_key(source_result.candidate),
                    "",
                ),
            )
            for source_result in result.source_results
        )
        return replace(result, source_results=source_results)

    def _receive_from_candidate_group_unleased(
        self,
        candidates: Sequence[RlSourceCandidate],
        *,
        receiver_rank: int,
        same_rank_only: bool,
        target_tensors: dict[str, torch.Tensor] | None,
        target_specs: Sequence[TensorReceiveSpec] | None,
    ) -> DenseFanInReceiveResult:
        device_id = self.resolve_device_id(target_tensors.values() if target_tensors is not None else None)
        fanin_plan = prepare_dense_fanin_receive(
            mx_client=self.mx_client,
            candidates=candidates,
            target_tensors=target_tensors,
            target_specs=target_specs,
            receiver_rank=receiver_rank,
            same_rank_only=same_rank_only,
            target_device=f"cuda:{device_id}",
        )
        self._shutdown_target_nixl_manager()
        manager = NixlTransferManager(
            agent_name=f"mx-rl-target-{uuid.uuid4().hex[:12]}",
            device_id=device_id,
        )
        manager.initialize()
        self._target_nixl_manager = manager
        self._nixl_manager = manager
        result = execute_dense_fanin_receive(
            fanin_plan,
            manager=manager,
            timeout_seconds=self.timeout_seconds,
        )
        logger.info(
            "Received ModelExpress RL dense fan-in weights: model=%s version=%d "
            "sources=%d tensors=%d bytes=%d",
            self.base_identity.model_name,
            fanin_plan.model_version,
            len(result.source_results),
            result.tensor_count,
            result.bytes_transferred,
        )
        return result

    async def wait_for_source(
        self,
        *,
        model_version: int | None,
        receiver_rank: int,
        roles: Sequence[RlSourceRole] = _DEFAULT_RECEIVE_ROLES,
        same_rank_only: bool = False,
        source_ranks_by_role: Mapping[RlSourceRole, Sequence[int]] | None = None,
        require_complete_version: bool = True,
    ) -> RlSourceCandidate:
        """Poll MX metadata until the best matching source is visible."""
        return (
            await self.wait_for_sources(
                model_version=model_version,
                receiver_rank=receiver_rank,
                roles=roles,
                same_rank_only=same_rank_only,
                source_ranks_by_role=source_ranks_by_role,
                require_complete_version=require_complete_version,
            )
        )[0]

    async def wait_for_sources(
        self,
        *,
        model_version: int | None,
        receiver_rank: int,
        roles: Sequence[RlSourceRole] = _DEFAULT_RECEIVE_ROLES,
        same_rank_only: bool = False,
        source_ranks_by_role: Mapping[RlSourceRole, Sequence[int]] | None = None,
        require_complete_version: bool = True,
    ) -> list[RlSourceCandidate]:
        """Poll MX metadata until matching requested/latest sources are visible."""
        deadline = time.monotonic() + self.timeout_seconds
        last_error: RuntimeError | None = None
        while True:
            try:
                return self.select_sources(
                    model_version=model_version,
                    receiver_rank=receiver_rank,
                    roles=roles,
                    same_rank_only=same_rank_only,
                    source_ranks_by_role=source_ranks_by_role,
                    require_complete_version=require_complete_version,
                )
            except RuntimeError as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    raise last_error
                await asyncio.sleep(0.25)

    def select_source(
        self,
        *,
        model_version: int | None,
        receiver_rank: int,
        roles: Sequence[RlSourceRole] = _DEFAULT_RECEIVE_ROLES,
        same_rank_only: bool = False,
        source_ranks_by_role: Mapping[RlSourceRole, Sequence[int]] | None = None,
        require_complete_version: bool = True,
    ) -> RlSourceCandidate:
        """Select the best source for a receiver from MX metadata."""
        return self.select_sources(
            model_version=model_version,
            receiver_rank=receiver_rank,
            roles=roles,
            same_rank_only=same_rank_only,
            source_ranks_by_role=source_ranks_by_role,
            require_complete_version=require_complete_version,
        )[0]

    def select_sources(
        self,
        *,
        model_version: int | None,
        receiver_rank: int,
        roles: Sequence[RlSourceRole] = _DEFAULT_RECEIVE_ROLES,
        same_rank_only: bool = False,
        source_ranks_by_role: Mapping[RlSourceRole, Sequence[int]] | None = None,
        require_complete_version: bool = True,
    ) -> list[RlSourceCandidate]:
        """Select source candidates for a requested or latest model version."""
        response = self.mx_client.list_sources(
            identity=None,
            status_filter=p2p_pb2.SOURCE_STATUS_READY,
        )
        candidates = candidates_for_base_identity(response, self.base_identity)
        candidates = [
            candidate
            for candidate in candidates
            if candidate.model_name == self.base_identity.model_name
            and candidate.worker_id != self.worker_id
        ]
        selected = select_rl_source_candidates(
            candidates,
            receiver_rank=receiver_rank,
            model_version=model_version,
            roles=roles,
            same_rank_only=same_rank_only,
            source_ranks_by_role=source_ranks_by_role,
            require_complete_version=require_complete_version,
        )
        if not selected:
            raise RuntimeError(
                f"No ModelExpress RL source found for model={self.base_identity.model_name!r} "
                f"version={_version_label(model_version)}"
            )
        return selected

    def resolve_device_id(
        self,
        tensors: Iterable[torch.Tensor] | None = None,
    ) -> int:
        """Resolve the CUDA device used for NIXL registration."""
        if self.device_id is not None:
            return self.device_id
        if tensors is not None:
            for tensor in tensors:
                return tensor.device.index or 0
        return torch.cuda.current_device()

    def mark_current_source_stale(self) -> None:
        """Best-effort STALE transition for a previously published source."""
        if self._published_sources:
            self._publication_store.close_current(mark_stale=True)
            self._refresh_current_publication()
            return
        if not self._mx_source_id:
            return
        self._stop_heartbeat(mark_stale=False)
        try:
            self.mx_client.update_status(
                self._mx_source_id,
                self.worker_id,
                self._worker_rank,
                p2p_pb2.SOURCE_STATUS_STALE,
            )
        except Exception:
            logger.warning(
                "Failed to mark ModelExpress RL source stale: source_id=%s",
                self._mx_source_id,
                exc_info=True,
            )
        self._published_tensors = {}
        self._mx_source_id = None
        self._worker_rank = 0

    def _refresh_current_publication(self) -> None:
        current = self._publication_store.current
        if current is None:
            self._published_tensors = {}
            self._mx_source_id = None
            self._worker_rank = 0
            self._heartbeat = None
            if self._target_nixl_manager is None:
                self._nixl_manager = None
            return

        self._published_tensors = current.tensors
        self._mx_source_id = current.mx_source_id
        self._worker_rank = current.worker_rank
        self._heartbeat = current.heartbeat
        self._nixl_manager = current.manager

    def _start_heartbeat(
        self,
        *,
        mx_source_id: str,
        worker_rank: int,
        nixl_manager: NixlTransferManager,
    ) -> HeartbeatThread:
        """Start periodic READY heartbeats for the published RL source."""
        heartbeat = HeartbeatThread(
            mx_client=self.mx_client,
            mx_source_id=mx_source_id,
            worker_id=self.worker_id,
            worker_rank=worker_rank,
            nixl_manager=nixl_manager,
            initially_ready=True,
        )
        heartbeat.start()
        return heartbeat

    def _stop_heartbeat(self, *, mark_stale: bool) -> None:
        """Stop the RL source heartbeat if one is active."""
        if self._heartbeat is None:
            return
        heartbeat = self._heartbeat
        self._heartbeat = None
        heartbeat.stop(mark_stale=mark_stale)

    def _shutdown_target_nixl_manager(self) -> None:
        if self._target_nixl_manager is None:
            return
        self._target_nixl_manager.shutdown()
        self._target_nixl_manager = None
        self._refresh_current_publication()

    def shutdown_nixl_manager(self) -> None:
        """Release active NIXL managers without changing metadata status."""
        self._shutdown_target_nixl_manager()
        self._publication_store.shutdown_all()
        self._refresh_current_publication()


def _version_label(model_version: int | None) -> str:
    return "latest" if model_version is None else str(model_version)
