# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MX metadata helpers for cross-parallelism refit planning."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable, Sequence

from . import p2p_pb2
from .resharding import (
    SegmentPlan,
    SliceOwnership,
    SliceRequest,
    classify_tensor_family,
    plan_segments,
)
from .types import TensorDescriptor

_REFIT_OWNERSHIP_SIDECAR_PREFIX = "mx-refit-ownership-v1:"


@dataclass(frozen=True)
class RefitNixlEndpoint:
    """MX-discovered source endpoint for a planned NIXL segment read."""

    mx_source_id: str
    worker_id: str
    worker_rank: int
    ownership: SliceOwnership
    tensor: TensorDescriptor
    agent_name: str
    nixl_metadata: bytes
    metadata_endpoint: str = ""
    worker_grpc_endpoint: str = ""
    status: int = p2p_pb2.SOURCE_STATUS_READY
    updated_at: int = 0

    @property
    def source_id(self) -> str:
        return self.ownership.source_id

    def to_nixl_source_info(self) -> dict[str, object]:
        """Return the shape expected by the refit NIXL read adapter."""

        return {
            "rank": self.worker_rank,
            "agent_name": self.agent_name,
            "source_id": self.source_id,
            "worker_id": self.worker_id,
            "source_range": self.ownership.source_range,
            "status": self.status,
            "updated_at": self.updated_at,
            "addr": self.tensor.addr,
            "device_id": self.tensor.device_id,
            "tensor_bytes": self.tensor.size,
            "registered_bytes": self.tensor.size,
            "metadata_bytes": len(self.nixl_metadata),
            "registration_duration_ms": 0.0,
            "metadata_duration_ms": 0.0,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "mx_source_id": self.mx_source_id,
            "worker_id": self.worker_id,
            "worker_rank": self.worker_rank,
            "source_id": self.source_id,
            "agent_name": self.agent_name,
            "metadata_bytes": len(self.nixl_metadata),
            "metadata_endpoint": self.metadata_endpoint,
            "worker_grpc_endpoint": self.worker_grpc_endpoint,
            "status": self.status,
            "updated_at": self.updated_at,
            "tensor": {
                "name": self.tensor.name,
                "addr": self.tensor.addr,
                "size": self.tensor.size,
                "device_id": self.tensor.device_id,
                "dtype": self.tensor.dtype,
            },
            "ownership": self.ownership.to_dict(),
        }


def build_refit_source_identity(
    *,
    model_name: str,
    model_version: str,
    dtype: str,
    quantization: str = "",
    trainer_framework: str = "unknown",
    trainer_layout: str = "",
    mx_version: str = "0.3.0",
) -> p2p_pb2.SourceIdentity:
    """Build a SourceIdentity for trainer-published slice ownership metadata."""

    extra_parameters = {
        "mx_refit_schema": "slice-ownership-v1",
        "source_role": "trainer",
        "trainer_framework": trainer_framework,
    }
    if trainer_layout:
        extra_parameters["trainer_layout"] = trainer_layout

    return p2p_pb2.SourceIdentity(
        mx_version=mx_version,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_UNKNOWN,
        tensor_parallel_size=0,
        pipeline_parallel_size=0,
        expert_parallel_size=0,
        dtype=dtype,
        quantization=quantization,
        extra_parameters=extra_parameters,
        revision=model_version,
    )


def slice_ownership_to_proto(
    ownership: SliceOwnership,
) -> p2p_pb2.SliceOwnershipDescriptor:
    """Convert planner ownership metadata to the MX P2P proto contract."""

    return p2p_pb2.SliceOwnershipDescriptor(
        model_name=ownership.model_name,
        model_version=ownership.model_version,
        tensor_name=ownership.tensor_name,
        global_shape=list(ownership.global_shape),
        dtype=ownership.dtype,
        source_range=_range_to_proto(ownership.source_range),
        storage_offset_bytes=ownership.storage_offset_bytes,
        strides=list(ownership.strides or []),
        contiguous=ownership.contiguous,
        worker_id=ownership.worker_id,
        worker_rank=ownership.worker_rank or 0,
        source_id=ownership.source_id,
        source_lease=ownership.source_lease,
        nixl_descriptor_id=ownership.nixl_descriptor_id,
        layout_tags=_layout_tags_to_proto(ownership.layout_tags),
        quantization_scope=ownership.quantization_scope.value,
        element_size_bytes=ownership.element_size_bytes,
        tensor_family=classify_tensor_family(
            ownership.tensor_name,
            layout_tags=ownership.layout_tags,
            quantization_scope=ownership.quantization_scope,
        ),
    )


def slice_ownership_from_proto(
    descriptor: p2p_pb2.SliceOwnershipDescriptor,
) -> SliceOwnership:
    """Convert MX P2P slice ownership metadata back to planner input."""

    return SliceOwnership(
        model_name=descriptor.model_name,
        model_version=descriptor.model_version,
        tensor_name=descriptor.tensor_name,
        global_shape=tuple(int(dim) for dim in descriptor.global_shape),
        dtype=descriptor.dtype,
        source_range=_range_from_proto(descriptor.source_range),
        worker_id=descriptor.worker_id,
        source_id=descriptor.source_id,
        worker_rank=int(descriptor.worker_rank),
        source_lease=descriptor.source_lease,
        nixl_descriptor_id=descriptor.nixl_descriptor_id,
        storage_offset_bytes=int(descriptor.storage_offset_bytes),
        strides=(
            tuple(int(s) for s in descriptor.strides) if descriptor.strides else None
        ),
        contiguous=bool(descriptor.contiguous),
        layout_tags=_layout_tags_from_proto(descriptor.layout_tags),
        quantization_scope=descriptor.quantization_scope or "absent",
        element_size_bytes=(
            int(descriptor.element_size_bytes)
            if descriptor.HasField("element_size_bytes")
            else None
        ),
    )


def slice_request_to_proto(request: SliceRequest) -> p2p_pb2.SliceRequestDescriptor:
    """Convert a receiver-side slice request to proto metadata."""

    return p2p_pb2.SliceRequestDescriptor(
        model_name=request.model_name,
        model_version=request.model_version,
        tensor_name=request.tensor_name,
        requested_range=_range_to_proto(request.requested_range),
        target_shape=list(request.target_shape),
        dtype=request.dtype,
        target_offset_bytes=request.target_offset_bytes,
        destination_strides=list(request.destination_strides or []),
        target_id=request.target_id,
        runtime_framework=request.runtime_framework,
        layout_tags=_layout_tags_to_proto(request.layout_tags),
        quantization_scope=request.quantization_scope.value,
        element_size_bytes=request.element_size_bytes,
    )


def slice_request_from_proto(
    descriptor: p2p_pb2.SliceRequestDescriptor,
) -> SliceRequest:
    """Convert request proto metadata back to planner input."""

    return SliceRequest(
        model_name=descriptor.model_name,
        model_version=descriptor.model_version,
        tensor_name=descriptor.tensor_name,
        requested_range=_range_from_proto(descriptor.requested_range),
        target_shape=tuple(int(dim) for dim in descriptor.target_shape),
        dtype=descriptor.dtype,
        target_offset_bytes=int(descriptor.target_offset_bytes),
        destination_strides=(
            tuple(int(s) for s in descriptor.destination_strides)
            if descriptor.destination_strides
            else None
        ),
        target_id=descriptor.target_id,
        runtime_framework=descriptor.runtime_framework,
        layout_tags=_layout_tags_from_proto(descriptor.layout_tags),
        quantization_scope=descriptor.quantization_scope or "absent",
        element_size_bytes=(
            int(descriptor.element_size_bytes)
            if descriptor.HasField("element_size_bytes")
            else None
        ),
    )


def segment_plan_to_proto(plan: SegmentPlan) -> p2p_pb2.SegmentPlanDescriptor:
    """Convert a computed SegmentPlan to proto metadata."""

    return p2p_pb2.SegmentPlanDescriptor(
        source_id=plan.source_id,
        worker_id=plan.worker_id,
        tensor_name=plan.tensor_name,
        source_range=_range_to_proto(plan.source_range),
        target_range=_range_to_proto(plan.target_range),
        source_byte_offset=plan.source_byte_offset,
        target_byte_offset=plan.target_byte_offset,
        bytes=plan.bytes,
        lease_version=plan.lease_version,
        retry_policy=plan.retry_policy.value,
        nixl_descriptor_id=plan.nixl_descriptor_id,
        target_id=plan.target_id,
        target_runtime=plan.target_runtime,
        worker_rank=plan.worker_rank or 0,
    )


def segment_plan_from_proto(
    descriptor: p2p_pb2.SegmentPlanDescriptor,
) -> SegmentPlan:
    """Convert segment plan proto metadata back to planner output."""

    return SegmentPlan(
        source_id=descriptor.source_id,
        worker_id=descriptor.worker_id,
        tensor_name=descriptor.tensor_name,
        source_range=_range_from_proto(descriptor.source_range),
        target_range=_range_from_proto(descriptor.target_range),
        source_byte_offset=int(descriptor.source_byte_offset),
        target_byte_offset=int(descriptor.target_byte_offset),
        bytes=int(descriptor.bytes),
        lease_version=descriptor.lease_version,
        retry_policy=descriptor.retry_policy,
        nixl_descriptor_id=descriptor.nixl_descriptor_id,
        target_id=descriptor.target_id,
        target_runtime=descriptor.target_runtime,
        worker_rank=int(descriptor.worker_rank),
    )


def tensor_descriptor_to_proto(
    descriptor: TensorDescriptor,
) -> p2p_pb2.TensorDescriptor:
    """Convert a Python tensor descriptor to the MX P2P proto contract."""

    return p2p_pb2.TensorDescriptor(
        name=descriptor.name,
        addr=int(descriptor.addr),
        size=int(descriptor.size),
        device_id=int(descriptor.device_id),
        dtype=descriptor.dtype,
    )


def tensor_descriptor_from_proto(
    descriptor: p2p_pb2.TensorDescriptor,
) -> TensorDescriptor:
    """Convert an MX P2P tensor descriptor to the Python metadata model."""

    return TensorDescriptor(
        name=descriptor.name,
        addr=int(descriptor.addr),
        size=int(descriptor.size),
        device_id=int(descriptor.device_id),
        dtype=descriptor.dtype,
    )


def publish_slice_ownerships(
    mx_client,
    *,
    identity: p2p_pb2.SourceIdentity,
    ownerships: Sequence[SliceOwnership],
    worker_id: str,
    worker_rank: int,
    status: int = p2p_pb2.SOURCE_STATUS_INITIALIZING,
) -> str:
    """Publish slice ownership metadata through the existing MX P2P API."""

    ownership_list = list(ownerships)
    metadata_endpoint = (
        _encode_refit_ownership_sidecar(
            ownerships=ownership_list,
            metadata_endpoint="",
        )
        if ownership_list
        else ""
    )
    worker = p2p_pb2.WorkerMetadata(
        worker_rank=worker_rank,
        slice_ownerships=[slice_ownership_to_proto(o) for o in ownership_list],
        status=status,
        metadata_endpoint=metadata_endpoint,
    )
    return mx_client.publish_metadata(identity, worker, worker_id)


def publish_refit_nixl_endpoint(
    mx_client,
    *,
    identity: p2p_pb2.SourceIdentity,
    ownership: SliceOwnership,
    tensor: TensorDescriptor,
    agent_name: str,
    nixl_metadata: bytes,
    worker_id: str | None = None,
    worker_rank: int | None = None,
    metadata_endpoint: str = "",
    worker_grpc_endpoint: str = "",
    status: int = p2p_pb2.SOURCE_STATUS_READY,
) -> str:
    """Publish slice ownership plus NIXL endpoint metadata through MX.

    This is the multi-pod handoff contract: source processes publish the
    logical slice they own and the NIXL handle/address the target needs to read
    that slice. Targets can then discover transfer endpoints from MX records
    instead of relying on torch distributed object gathers.
    """

    resolved_worker_id = worker_id or ownership.worker_id
    resolved_worker_rank = (
        int(worker_rank) if worker_rank is not None else int(ownership.worker_rank or 0)
    )
    encoded_metadata_endpoint = _encode_refit_ownership_sidecar(
        ownerships=[ownership],
        metadata_endpoint=metadata_endpoint,
    )
    worker = p2p_pb2.WorkerMetadata(
        worker_rank=resolved_worker_rank,
        nixl_metadata=bytes(nixl_metadata),
        tensors=[tensor_descriptor_to_proto(tensor)],
        slice_ownerships=[slice_ownership_to_proto(ownership)],
        status=status,
        agent_name=agent_name,
        metadata_endpoint=encoded_metadata_endpoint,
        worker_grpc_endpoint=worker_grpc_endpoint,
    )
    return mx_client.publish_metadata(identity, worker, resolved_worker_id)


def list_slice_ownerships(
    mx_client,
    *,
    identity: p2p_pb2.SourceIdentity,
    status_filter: int | None = p2p_pb2.SOURCE_STATUS_READY,
) -> list[SliceOwnership]:
    """Query MX for source workers and collect their slice ownership metadata."""

    response = mx_client.list_sources(identity, status_filter=status_filter)
    ownerships: list[SliceOwnership] = []
    for instance in response.instances:
        metadata = mx_client.get_metadata(instance.mx_source_id, instance.worker_id)
        has_worker = (
            metadata.HasField("worker")
            if hasattr(metadata, "HasField")
            else getattr(metadata, "worker", None) is not None
        )
        if not metadata.found or not has_worker:
            continue
        worker_ownerships = [
            slice_ownership_from_proto(desc)
            for desc in metadata.worker.slice_ownerships
        ]
        sidecar = _decode_refit_ownership_sidecar(metadata.worker.metadata_endpoint)
        if not worker_ownerships and sidecar is not None:
            worker_ownerships = sidecar[0]
        ownerships.extend(worker_ownerships)
    return ownerships


def list_refit_nixl_endpoints(
    mx_client,
    *,
    identity: p2p_pb2.SourceIdentity,
    status_filter: int | None = p2p_pb2.SOURCE_STATUS_READY,
) -> list[RefitNixlEndpoint]:
    """Query MX source workers and return NIXL refit endpoints."""

    response = mx_client.list_sources(identity, status_filter=status_filter)
    endpoints: list[RefitNixlEndpoint] = []
    for instance in response.instances:
        metadata = mx_client.get_metadata(instance.mx_source_id, instance.worker_id)
        has_worker = (
            metadata.HasField("worker")
            if hasattr(metadata, "HasField")
            else getattr(metadata, "worker", None) is not None
        )
        if not metadata.found or not has_worker:
            continue

        worker = metadata.worker
        tensors = [tensor_descriptor_from_proto(desc) for desc in worker.tensors]
        if not worker.nixl_metadata or not tensors:
            continue
        tensors_by_name = {tensor.name: tensor for tensor in tensors}

        ownerships = [
            slice_ownership_from_proto(ownership_desc)
            for ownership_desc in worker.slice_ownerships
        ]
        sidecar = _decode_refit_ownership_sidecar(worker.metadata_endpoint)
        if not ownerships and sidecar is not None:
            ownerships = sidecar[0]

        metadata_endpoint = (
            sidecar[1] if sidecar is not None else worker.metadata_endpoint
        )
        for ownership in ownerships:
            tensor = tensors_by_name.get(ownership.tensor_name)
            if tensor is None and len(tensors) == 1:
                tensor = tensors[0]
            if tensor is None:
                continue
            endpoints.append(
                RefitNixlEndpoint(
                    mx_source_id=instance.mx_source_id,
                    worker_id=instance.worker_id,
                    worker_rank=int(worker.worker_rank),
                    ownership=ownership,
                    tensor=tensor,
                    agent_name=worker.agent_name,
                    nixl_metadata=bytes(worker.nixl_metadata),
                    metadata_endpoint=metadata_endpoint,
                    worker_grpc_endpoint=worker.worker_grpc_endpoint,
                    status=int(worker.status),
                    updated_at=int(worker.updated_at),
                )
            )
    return endpoints


def plan_from_mx_refit_endpoints(
    mx_client,
    *,
    identity: p2p_pb2.SourceIdentity,
    requests: Sequence[SliceRequest],
    status_filter: int | None = p2p_pb2.SOURCE_STATUS_READY,
) -> tuple[list[SegmentPlan], dict[str, RefitNixlEndpoint]]:
    """Plan requested slices and return the MX-discovered endpoints used."""

    endpoints = list_refit_nixl_endpoints(
        mx_client,
        identity=identity,
        status_filter=status_filter,
    )
    plans = plan_segments([endpoint.ownership for endpoint in endpoints], requests)
    endpoints_by_source_id = {endpoint.source_id: endpoint for endpoint in endpoints}
    planned_source_ids = {plan.source_id for plan in plans}
    endpoint_by_source_id = {
        source_id: endpoints_by_source_id[source_id]
        for source_id in planned_source_ids
        if source_id in endpoints_by_source_id
    }
    return plans, endpoint_by_source_id


def plan_from_mx_metadata(
    mx_client,
    *,
    identity: p2p_pb2.SourceIdentity,
    requests: Sequence[SliceRequest],
    status_filter: int | None = p2p_pb2.SOURCE_STATUS_READY,
) -> list[SegmentPlan]:
    """Query MX for slice ownership metadata and plan requested target slices."""

    ownerships = list_slice_ownerships(
        mx_client,
        identity=identity,
        status_filter=status_filter,
    )
    return plan_segments(ownerships, requests)


def _range_to_proto(tensor_range: Iterable[tuple[int, int]]):
    return [
        p2p_pb2.TensorAxisRange(start=int(start), end=int(end))
        for start, end in tensor_range
    ]


def _range_from_proto(ranges) -> tuple[tuple[int, int], ...]:
    return tuple((int(axis.start), int(axis.end)) for axis in ranges)


def _encode_refit_ownership_sidecar(
    *,
    ownerships: Sequence[SliceOwnership],
    metadata_endpoint: str,
) -> str:
    """Encode ownership in a legacy field for older MX server deployments.

    The current schema stores ownership in ``WorkerMetadata.slice_ownerships``.
    Some live server images can accept the new client proto but drop that field
    while preserving older string fields. The sidecar keeps cross-node POCs
    honest without replacing the primary schema path, and preserves the original
    endpoint string for callers that still need it after decode.
    """

    payload = {
        "metadata_endpoint": metadata_endpoint,
        "ownerships": [ownership.to_dict() for ownership in ownerships],
    }
    return _REFIT_OWNERSHIP_SIDECAR_PREFIX + json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )


def _decode_refit_ownership_sidecar(
    metadata_endpoint: str,
) -> tuple[list[SliceOwnership], str] | None:
    if not metadata_endpoint.startswith(_REFIT_OWNERSHIP_SIDECAR_PREFIX):
        return None
    payload = json.loads(metadata_endpoint[len(_REFIT_OWNERSHIP_SIDECAR_PREFIX) :])
    ownership_payloads = payload.get("ownerships")
    if ownership_payloads is None:
        ownership_payloads = [payload["ownership"]]
    return (
        [SliceOwnership.from_dict(ownership) for ownership in ownership_payloads],
        str(payload.get("metadata_endpoint", "")),
    )


def _layout_tags_to_proto(layout_tags: dict[str, object]) -> dict[str, str]:
    return {
        str(key): json.dumps(value, sort_keys=True, separators=(",", ":"))
        for key, value in layout_tags.items()
    }


def _layout_tags_from_proto(layout_tags) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in dict(layout_tags).items():
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value
    return parsed
