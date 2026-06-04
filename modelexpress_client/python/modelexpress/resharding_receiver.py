# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receiver-side helpers for runtime-owned refit tensors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch

from .resharding import (
    QuantizationScope,
    SegmentPlan,
    SliceRequest,
    TensorRange,
    normalize_range,
    range_extents,
)


@dataclass(frozen=True)
class InstalledSegment:
    """A copied segment for smoke tests and receiver-side metrics."""

    tensor_name: str
    target_key: str
    target_range: TensorRange
    bytes: int
    source_id: str


@dataclass(frozen=True)
class InstalledQuantizationFallback:
    """A fallback-installed global quantization metadata tensor."""

    tensor_name: str
    target_key: str
    tensor_family: str
    quantization_scope: QuantizationScope
    shape: tuple[int, ...]
    dtype: str
    bytes: int
    runtime_framework: str


@dataclass(frozen=True)
class RuntimeTensorSnapshot:
    """Metadata for one tensor snapshot captured before a versioned refit."""

    target_key: str
    shape: tuple[int, ...]
    dtype: str
    bytes: int
    checksum: float


@dataclass
class RuntimeRefitTransaction:
    """Rollback handle for installing one target model version into runtime tensors."""

    previous_model_version: str
    target_model_version: str
    _target_tensors: Mapping[str, torch.Tensor]
    _snapshots: dict[str, torch.Tensor]
    snapshots: tuple[RuntimeTensorSnapshot, ...]
    committed: bool = False
    rolled_back: bool = False

    def commit(self) -> None:
        """Mark the target version active and release rollback snapshots."""

        if self.rolled_back:
            raise RuntimeError("cannot commit a rolled-back runtime refit transaction")
        self._snapshots.clear()
        self.committed = True

    def rollback(self) -> None:
        """Restore the previous model version into the runtime-owned tensors."""

        if self.committed:
            raise RuntimeError("cannot rollback a committed runtime refit transaction")
        for target_key, snapshot in self._snapshots.items():
            if target_key not in self._target_tensors:
                raise KeyError(f"missing runtime target tensor {target_key!r}")
            target = self._target_tensors[target_key]
            if tuple(int(dim) for dim in target.shape) != tuple(
                int(dim) for dim in snapshot.shape
            ):
                raise ValueError(
                    f"runtime target tensor {target_key!r} changed shape from "
                    f"{tuple(int(dim) for dim in snapshot.shape)} to "
                    f"{tuple(int(dim) for dim in target.shape)}"
                )
            if target.dtype != snapshot.dtype:
                raise ValueError(
                    f"runtime target tensor {target_key!r} changed dtype from "
                    f"{_torch_dtype_name(snapshot.dtype)} to "
                    f"{_torch_dtype_name(target.dtype)}"
                )
            with torch.no_grad():
                target.copy_(snapshot.to(device=target.device))
        self._snapshots.clear()
        self.rolled_back = True

    def to_dict(self) -> dict[str, object]:
        return {
            "previous_model_version": self.previous_model_version,
            "target_model_version": self.target_model_version,
            "committed": self.committed,
            "rolled_back": self.rolled_back,
            "snapshots": [
                {
                    "target_key": snapshot.target_key,
                    "shape": list(snapshot.shape),
                    "dtype": snapshot.dtype,
                    "bytes": snapshot.bytes,
                    "checksum": snapshot.checksum,
                }
                for snapshot in self.snapshots
            ],
        }


def begin_runtime_refit_transaction(
    target_tensors: Mapping[str, torch.Tensor],
    *,
    previous_model_version: str,
    target_model_version: str,
    target_keys: Iterable[str] | None = None,
) -> RuntimeRefitTransaction:
    """Snapshot runtime-owned tensors before installing a new model version.

    The snapshots stay on the tensors' current devices, so GPU callers can keep
    rollback state device-resident. CPU tests use the same path.
    """

    if not previous_model_version:
        raise ValueError("previous_model_version is required")
    if not target_model_version:
        raise ValueError("target_model_version is required")
    if previous_model_version == target_model_version:
        raise ValueError("target_model_version must differ from previous_model_version")

    selected_keys = (
        tuple(target_keys) if target_keys is not None else tuple(target_tensors)
    )
    if not selected_keys:
        raise ValueError("at least one runtime tensor must be snapshotted")

    snapshots: dict[str, torch.Tensor] = {}
    snapshot_metadata: list[RuntimeTensorSnapshot] = []
    for target_key in selected_keys:
        if target_key not in target_tensors:
            raise KeyError(f"missing runtime target tensor {target_key!r}")
        target = target_tensors[target_key]
        snapshot = target.detach().clone()
        snapshots[target_key] = snapshot
        snapshot_metadata.append(
            RuntimeTensorSnapshot(
                target_key=target_key,
                shape=tuple(int(dim) for dim in target.shape),
                dtype=_torch_dtype_name(target.dtype),
                bytes=int(target.numel() * target.element_size()),
                checksum=_tensor_checksum(target),
            )
        )

    return RuntimeRefitTransaction(
        previous_model_version=previous_model_version,
        target_model_version=target_model_version,
        _target_tensors=target_tensors,
        _snapshots=snapshots,
        snapshots=tuple(snapshot_metadata),
    )


def build_receiver_requests_from_runtime_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    model_name: str,
    model_version: str,
    runtime_framework: str,
    requested_ranges: Mapping[str, TensorRange] | None = None,
    layout_tags_by_tensor: Mapping[str, Mapping[str, str | int | bool]] | None = None,
    target_id_prefix: str = "",
) -> list[SliceRequest]:
    """Create receiver-side requests from framework-owned target tensors.

    ``requested_ranges`` maps each local runtime tensor to the global tensor
    range it represents. If omitted, the tensor is treated as a full-tensor
    request.
    """

    requests: list[SliceRequest] = []
    for tensor_name, tensor in tensors.items():
        requested_range = _requested_range_for_tensor(
            tensor_name,
            tensor,
            requested_ranges=requested_ranges,
        )
        layout_tags = _target_layout_tags(
            tensor,
            runtime_framework=runtime_framework,
        )
        if layout_tags_by_tensor and tensor_name in layout_tags_by_tensor:
            layout_tags.update(layout_tags_by_tensor[tensor_name])

        target_id = f"{target_id_prefix}:{tensor_name}" if target_id_prefix else ""
        requests.append(
            SliceRequest(
                tensor_name=tensor_name,
                requested_range=requested_range,
                target_shape=tuple(int(dim) for dim in tensor.shape),
                dtype=_torch_dtype_name(tensor.dtype),
                target_id=target_id,
                model_name=model_name,
                model_version=model_version,
                destination_strides=tuple(int(stride) for stride in tensor.stride()),
                runtime_framework=runtime_framework,
                layout_tags=layout_tags,
                element_size_bytes=int(tensor.element_size()),
            )
        )
    return requests


def install_segment_payloads_into_runtime_tensors(
    segment_payloads: Iterable[tuple[SegmentPlan, torch.Tensor]],
    target_tensors: Mapping[str, torch.Tensor],
    *,
    target_ranges: Mapping[str, TensorRange] | None = None,
    allow_dtype_cast: bool = False,
) -> list[InstalledSegment]:
    """Copy planned segment payloads into runtime-owned target tensors.

    This is the receiver install equivalent of NIXL landing bytes into a target
    buffer. Tests pass payload tensors directly; the production data plane should
    provide those bytes through one-sided reads.
    """

    installed: list[InstalledSegment] = []
    for plan, payload in segment_payloads:
        target_key, target = _resolve_target_tensor(plan, target_tensors)
        base_range = _target_base_range(
            plan,
            target,
            target_key=target_key,
            target_ranges=target_ranges,
        )
        local_slices = _local_slices(plan.target_range, base_range)
        expected_shape = range_extents(plan.target_range)
        prepared_payload = _prepare_payload(
            payload,
            target=target,
            expected_shape=expected_shape,
            allow_dtype_cast=allow_dtype_cast,
        )

        with torch.no_grad():
            target[local_slices].copy_(prepared_payload)

        installed.append(
            InstalledSegment(
                tensor_name=plan.tensor_name,
                target_key=target_key,
                target_range=plan.target_range,
                bytes=plan.bytes,
                source_id=plan.source_id,
            )
        )
    return installed


def install_global_required_quantization_payloads_into_runtime_tensors(
    fallback_payloads: Iterable[tuple[object, torch.Tensor]],
    target_tensors: Mapping[str, torch.Tensor],
    *,
    target_key_by_tensor: Mapping[str, str] | None = None,
    allow_dtype_cast: bool = False,
    runtime_framework: str = "",
) -> list[InstalledQuantizationFallback]:
    """Install global-required quantization metadata into runtime tensors.

    ``GLOBAL_REQUIRED`` tensors are intentionally rejected by the zero-copy
    segment planner because their correctness depends on global quantization
    context. This helper is the receiver-side fallback: after another path has
    materialized the complete metadata payload, copy it into the framework-owned
    runtime tensor without trainer-side inference-layout conversion.
    """

    installed: list[InstalledQuantizationFallback] = []
    for manifest_entry, payload in fallback_payloads:
        tensor_name = _manifest_attr(manifest_entry, "tensor_name")
        tensor_family = _manifest_attr(manifest_entry, "tensor_family", default="")
        quantization_scope = QuantizationScope(
            _manifest_attr(manifest_entry, "quantization_scope")
        )
        if quantization_scope != QuantizationScope.GLOBAL_REQUIRED:
            raise ValueError(
                f"{tensor_name!r} is {quantization_scope.value}; expected "
                f"{QuantizationScope.GLOBAL_REQUIRED.value} fallback metadata"
            )

        target_key, target = _resolve_manifest_target_tensor(
            tensor_name,
            target_tensors,
            target_key_by_tensor=target_key_by_tensor,
        )
        expected_shape = tuple(
            int(dim) for dim in _manifest_attr(manifest_entry, "global_shape")
        )
        if tuple(int(dim) for dim in target.shape) != expected_shape:
            raise ValueError(
                f"target tensor for {tensor_name!r} has shape "
                f"{tuple(int(dim) for dim in target.shape)}, expected {expected_shape}"
            )
        expected_dtype = _normalize_dtype_name(
            str(_manifest_attr(manifest_entry, "dtype"))
        )
        target_dtype = _normalize_dtype_name(_torch_dtype_name(target.dtype))
        if target_dtype != expected_dtype:
            raise TypeError(
                f"target tensor for {tensor_name!r} has dtype {target_dtype}, "
                f"expected manifest dtype {expected_dtype}"
            )
        prepared_payload = _prepare_payload(
            payload,
            target=target,
            expected_shape=expected_shape,
            allow_dtype_cast=allow_dtype_cast,
        )

        with torch.no_grad():
            target.copy_(prepared_payload)

        installed.append(
            InstalledQuantizationFallback(
                tensor_name=tensor_name,
                target_key=target_key,
                tensor_family=tensor_family,
                quantization_scope=quantization_scope,
                shape=expected_shape,
                dtype=_torch_dtype_name(target.dtype),
                bytes=int(target.numel() * target.element_size()),
                runtime_framework=runtime_framework,
            )
        )
    return installed


def _requested_range_for_tensor(
    tensor_name: str,
    tensor: torch.Tensor,
    *,
    requested_ranges: Mapping[str, TensorRange] | None,
) -> TensorRange:
    if requested_ranges and tensor_name in requested_ranges:
        requested_range = normalize_range(requested_ranges[tensor_name])
    else:
        requested_range = tuple((0, int(dim)) for dim in tensor.shape)
    if range_extents(requested_range) != tuple(int(dim) for dim in tensor.shape):
        raise ValueError(
            f"requested range for {tensor_name!r} does not match target tensor shape"
        )
    return requested_range


def _target_layout_tags(
    tensor: torch.Tensor,
    *,
    runtime_framework: str,
) -> dict[str, str | int | bool]:
    return {
        "runtime_framework": runtime_framework,
        "storage_layout": "row-major" if tensor.is_contiguous() else "strided",
        "target_contiguous": bool(tensor.is_contiguous()),
    }


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _tensor_checksum(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().sum().item())


def _normalize_dtype_name(dtype: str) -> str:
    normalized = dtype.removeprefix("torch.").lower()
    if normalized == "bf16":
        return "bfloat16"
    if normalized in {"f16", "half"}:
        return "float16"
    if normalized in {"f32", "float"}:
        return "float32"
    if normalized in {"f64", "double"}:
        return "float64"
    return normalized


def _resolve_target_tensor(
    plan: SegmentPlan,
    target_tensors: Mapping[str, torch.Tensor],
) -> tuple[str, torch.Tensor]:
    if plan.target_id and plan.target_id in target_tensors:
        return plan.target_id, target_tensors[plan.target_id]
    if plan.tensor_name in target_tensors:
        return plan.tensor_name, target_tensors[plan.tensor_name]
    raise KeyError(
        f"no target tensor for plan target_id={plan.target_id!r} "
        f"tensor_name={plan.tensor_name!r}"
    )


def _resolve_manifest_target_tensor(
    tensor_name: str,
    target_tensors: Mapping[str, torch.Tensor],
    *,
    target_key_by_tensor: Mapping[str, str] | None,
) -> tuple[str, torch.Tensor]:
    if target_key_by_tensor and tensor_name in target_key_by_tensor:
        target_key = target_key_by_tensor[tensor_name]
        if target_key not in target_tensors:
            raise KeyError(
                f"target_key_by_tensor maps {tensor_name!r} to missing "
                f"target key {target_key!r}"
            )
        return target_key, target_tensors[target_key]

    if tensor_name in target_tensors:
        return tensor_name, target_tensors[tensor_name]

    suffix = f":{tensor_name}"
    matching_keys = [key for key in target_tensors if key.endswith(suffix)]
    if len(matching_keys) == 1:
        target_key = matching_keys[0]
        return target_key, target_tensors[target_key]
    if matching_keys:
        raise KeyError(
            f"multiple runtime target tensors match {tensor_name!r}: {matching_keys}"
        )
    raise KeyError(
        f"no runtime target tensor for quantization metadata {tensor_name!r}"
    )


def _manifest_attr(
    manifest_entry: object, name: str, *, default: object = None
) -> object:
    if isinstance(manifest_entry, Mapping):
        if name in manifest_entry:
            return manifest_entry[name]
        if default is not None:
            return default
        raise KeyError(name)
    if hasattr(manifest_entry, name):
        return getattr(manifest_entry, name)
    if default is not None:
        return default
    raise AttributeError(name)


def _target_base_range(
    plan: SegmentPlan,
    target: torch.Tensor,
    *,
    target_key: str,
    target_ranges: Mapping[str, TensorRange] | None,
) -> TensorRange:
    if target_ranges:
        if target_key in target_ranges:
            return normalize_range(target_ranges[target_key])
        if plan.target_id and plan.target_id in target_ranges:
            return normalize_range(target_ranges[plan.target_id])
        if plan.tensor_name in target_ranges:
            return normalize_range(target_ranges[plan.tensor_name])
    return tuple((0, int(dim)) for dim in target.shape)


def _local_slices(
    target_range: TensorRange, base_range: TensorRange
) -> tuple[slice, ...]:
    target_range = normalize_range(target_range)
    base_range = normalize_range(base_range)
    if len(target_range) != len(base_range):
        raise ValueError("target range rank must match target tensor base range")

    slices: list[slice] = []
    for axis, ((start, end), (base_start, base_end)) in enumerate(
        zip(target_range, base_range)
    ):
        if start < base_start or end > base_end:
            raise ValueError(
                f"target_range axis {axis} {(start, end)} is outside target "
                f"base range {(base_start, base_end)}"
            )
        slices.append(slice(start - base_start, end - base_start))
    return tuple(slices)


def _prepare_payload(
    payload: torch.Tensor,
    *,
    target: torch.Tensor,
    expected_shape: tuple[int, ...],
    allow_dtype_cast: bool,
) -> torch.Tensor:
    if tuple(int(dim) for dim in payload.shape) != expected_shape:
        payload = payload.reshape(expected_shape)
    if payload.dtype != target.dtype:
        if not allow_dtype_cast:
            raise TypeError(
                f"payload dtype {payload.dtype} does not match target dtype "
                f"{target.dtype}"
            )
        payload = payload.to(dtype=target.dtype)
    if payload.device != target.device:
        payload = payload.to(device=target.device)
    return payload
