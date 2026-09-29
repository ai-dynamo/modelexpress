# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private helpers shared by framework adapters."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from .. import envs
from ..types import MeshSpec, PlacementKind, ReshardPlan


def _text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    return text


def _endpoint(value: object) -> str:
    endpoint = str(value).strip()
    for prefix in ("https://", "grpcs://"):
        if endpoint.startswith(prefix):
            raise ValueError(
                "secure ModelExpress endpoints are not supported by the "
                "collective integration"
            )
    for prefix in ("grpc://", "http://"):
        if endpoint.startswith(prefix):
            endpoint = endpoint.removeprefix(prefix)
            break
    if not endpoint:
        raise ValueError("ModelExpress endpoint must not be empty")
    return endpoint


def _dtype_label(value: object) -> str:
    return str(value).removeprefix("torch.")


# Element sizes for the dtypes a collective plan may carry. Anything absent is
# rejected rather than guessed: a wrong size silently mis-groups the transfer.
_DTYPE_BYTES = {
    "bfloat16": 2,
    "bool": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
}


class _FrozenPlan:
    def __init__(self, plan: ReshardPlan) -> None:
        snapshot = copy.deepcopy(plan)
        snapshot.validate()
        if snapshot.misc:
            raise ValueError(
                "the initial MILES/SGLang collective integration supports "
                "all-bulk plans only"
            )
        names = snapshot.parameter_names()
        if not names:
            raise ValueError("the collective plan must contain at least one parameter")
        if len(names) != len(set(names)):
            raise ValueError(
                "the collective plan must not contain duplicate parameters"
            )
        unsupported = [
            entry.name
            for entry in snapshot.bulk
            if _dtype_label(entry.dtype) != "bfloat16"
        ]
        if unsupported:
            raise ValueError(
                "the initial MILES/SGLang collective integration supports BF16 "
                f"base weights only; unsupported: {unsupported[:5]}"
            )
        self._plan = snapshot
        self._by_name = {entry.name: entry for entry in snapshot.bulk}

    @property
    def source_partition_count(self) -> int:
        return self._plan.source_partition_count

    def names(self) -> list[str]:
        return self._plan.parameter_names()

    def entry(self, name: str):
        return self._by_name[name]

    def capture(self) -> ReshardPlan:
        return copy.deepcopy(self._plan)

    def validate_topology(self, topology: Any) -> None:
        if self.source_partition_count != topology.source_partition_count:
            raise ValueError(
                "plan source_partition_count does not match the collective "
                f"topology: {self.source_partition_count} != "
                f"{topology.source_partition_count}"
            )
        trainers_per_lane = (
            len(topology.trainer_slots) // topology.source_partition_count
        )
        expected_dst_ranks = list(
            range(
                trainers_per_lane,
                trainers_per_lane + len(topology.generator_slots),
            )
        )
        for entry in self._plan.bulk:
            src_ranks = entry.src_mesh.ranks()
            contiguous = bool(src_ranks) and src_ranks == list(
                range(src_ranks[0], src_ranks[0] + len(src_ranks))
            )
            if (
                not contiguous
                or len(src_ranks) != len(set(src_ranks))
                or src_ranks[0] < 0
                or src_ranks[-1] >= trainers_per_lane
            ):
                raise ValueError(
                    f"{entry.name}: src_mesh ranks {src_ranks} must be a non-empty, "
                    "duplicate-free contiguous subset of the trainer membership "
                    f"for reshard lane {entry.partition_id}: "
                    f"{list(range(trainers_per_lane))}"
                )
            if entry.dst_mesh.ranks() != expected_dst_ranks:
                raise ValueError(
                    f"{entry.name}: dst_mesh ranks {entry.dst_mesh.ranks()} do "
                    "not match the generator membership of reshard lane "
                    f"{entry.partition_id}: {expected_dst_ranks}"
                )


def _local_shape(
    global_shape: tuple[int, ...],
    mesh: MeshSpec,
    placements: tuple[Any, ...],
) -> tuple[int, ...]:
    shape = list(global_shape)
    for axis, placement in enumerate(placements):
        if placement.kind is PlacementKind.SHARD:
            dim = placement.dim
            if dim is None:
                raise ValueError("a shard placement must name a tensor dimension")
            shape[dim] //= mesh.shape[axis]
    return tuple(shape)


@dataclass(frozen=True)
class _TensorSignature:
    address: int
    shape: tuple[int, ...]
    dtype: str
    device: str


def _tensor_signature(
    name: str,
    tensor: Any,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
) -> _TensorSignature:
    is_contiguous = getattr(tensor, "is_contiguous", None)
    if not callable(is_contiguous) or not is_contiguous():
        raise ValueError(f"{name}: tensor storage must be contiguous")
    data_ptr = getattr(tensor, "data_ptr", None)
    if not callable(data_ptr):
        raise TypeError(f"{name}: tensor storage must expose data_ptr()")
    shape = tuple(int(dim) for dim in tensor.shape)
    if shape != expected_shape:
        raise ValueError(
            f"{name}: tensor shape {shape} does not match expected local shape "
            f"{expected_shape}"
        )
    dtype = _dtype_label(tensor.dtype)
    if dtype != _dtype_label(expected_dtype):
        raise ValueError(
            f"{name}: tensor dtype {dtype} does not match plan dtype "
            f"{_dtype_label(expected_dtype)}"
        )
    device = str(getattr(tensor, "device", ""))
    device_kind, separator, device_index = device.partition(":")
    if device_kind != "cuda" or not separator or not device_index.isdecimal():
        raise ValueError(
            f"{name}: tensor storage must name an indexed CUDA device such as "
            f"cuda:0, got {device or '<missing>'}"
        )
    return _TensorSignature(
        address=int(data_ptr()),
        shape=shape,
        dtype=dtype,
        device=device,
    )


def _check_stable(
    name: str,
    tensor: Any,
    baseline: _TensorSignature,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
) -> None:
    try:
        current = _tensor_signature(
            name,
            tensor,
            expected_shape=expected_shape,
            expected_dtype=expected_dtype,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"{name}: stable tensor validation failed: {error}"
        ) from error
    if current.address != baseline.address:
        raise RuntimeError(
            f"{name}: tensor storage address changed from "
            f"{baseline.address:#x} to {current.address:#x}"
        )
    if current.shape != baseline.shape:
        raise RuntimeError(
            f"{name}: tensor shape changed from {baseline.shape} to {current.shape}"
        )
    if current.dtype != baseline.dtype:
        raise RuntimeError(
            f"{name}: tensor dtype changed from {baseline.dtype} to {current.dtype}"
        )
    if current.device != baseline.device:
        raise RuntimeError(
            f"{name}: tensor device changed from {baseline.device} to {current.device}"
        )


def _layer_groups(
    groups: tuple[tuple[str, ...], ...],
    expected_names: list[str],
) -> list[list[str]]:
    if not groups:
        groups = (tuple(expected_names),)
    normalized = [list(group) for group in groups]
    if any(not group for group in normalized):
        raise ValueError("layer groups must not be empty")
    flattened = [name for group in normalized for name in group]
    if flattened != expected_names:
        missing = sorted(set(expected_names) - set(flattened))
        unknown = sorted(set(flattened) - set(expected_names))
        duplicates = sorted({name for name in flattened if flattened.count(name) > 1})
        if not missing and not unknown and not duplicates:
            raise ValueError(
                "layer groups must publish every bulk parameter in plan order; "
                "the groups are a pure reordering of the right names"
            )
        raise ValueError(
            "layer groups must cover every bulk parameter exactly once in plan "
            f"order (missing={missing[:5]}, unknown={unknown[:5]}, "
            f"duplicates={duplicates[:5]})"
        )
    return normalized


def _single_device(signatures: dict[str, _TensorSignature], label: str) -> str:
    devices = sorted({signature.device for signature in signatures.values()})
    if len(devices) != 1:
        raise ValueError(f"{label} tensors must share one CUDA device; got {devices}")
    return devices[0]


def _client_device(requested: Any, storage_device: str, label: str) -> Any:
    if requested is None:
        return storage_device
    requested_label = (
        f"cuda:{requested}" if isinstance(requested, int) else str(requested)
    )
    if requested_label == "cuda":
        import torch

        requested_label = f"cuda:{torch.cuda.current_device()}"
    if requested_label != storage_device:
        raise ValueError(
            f"{label} client device {requested_label} does not match tensor storage "
            f"device {storage_device}"
        )
    return requested_label


def _collective_streams(streams: list[Any] | None, *, device: Any) -> list[Any]:
    if streams:
        return list(streams)

    import torch

    if not torch.cuda.is_available():
        return [None]
    device_context = torch.cuda.device(device) if device is not None else nullcontext()
    with device_context:
        return [
            torch.cuda.Stream(device=device)
            for _ in range(envs.MX_NCCL_REFIT_NUM_STREAMS)
        ]


def _order_current_cuda_stream_before(
    streams: list[Any],
    *,
    device: Any,
) -> None:
    if not streams:
        return

    import torch

    if not torch.cuda.is_available():
        if all(stream is None for stream in streams):
            return
        raise RuntimeError("CUDA lane streams require an available CUDA runtime")
    device_context = torch.cuda.device(device) if device is not None else nullcontext()
    with device_context:
        producer = torch.cuda.current_stream(device=device)
        ready = torch.cuda.Event()
        ready.record(producer)
        for stream in streams:
            if stream is None:
                target = torch.cuda.default_stream(device=device)
            elif isinstance(stream, torch.cuda.Stream):
                target = stream
            elif isinstance(stream, int):
                target = torch.cuda.ExternalStream(stream)
            elif hasattr(stream, "cuda_stream"):
                target = torch.cuda.ExternalStream(int(stream.cuda_stream))
            else:
                raise TypeError(
                    "collective lane streams must be torch CUDA streams or raw "
                    f"CUDA stream handles, got {type(stream).__name__}"
                )
            target.wait_event(ready)
