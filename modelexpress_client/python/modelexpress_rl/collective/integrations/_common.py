# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private helpers shared by framework adapters."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from .. import envs
from ..types import MeshSpec, Placement, PlacementKind, ReshardPlan


def _text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    return text


def _endpoint(value: object) -> str:
    # Not modelexpress.client._parse_server_address: that helper silently
    # strips https://, while this boundary must reject secure schemes.
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


#: The receiver build each destination layout needs; part of the plan digest,
#: so a trainer and a receiver that disagree never form a group.
REPLICATED_DESTINATION_ABI = "miles-sglang-bf16-replicated-v1"
SHARDED_DESTINATION_ABI = "miles-sglang-bf16-sharded-v1"


def _expected_index(
    global_shape: tuple[int, ...],
    mesh: MeshSpec,
    placements: tuple[Placement, ...],
    rank: int,
) -> tuple[slice, ...]:
    """The slice of a parameter that lane rank ``rank`` holds on one plan side.

    Ranks map onto the mesh in row-major order (the order ``MeshSpec.nested()``
    hands the reshard op); a ``Shard`` placement splits that dim evenly.
    """
    offset = rank - mesh.rank_offset
    if not 0 <= offset < mesh.size:
        raise ValueError(f"rank {rank} is not in the mesh {mesh.canonical()}")
    coord = []
    for extent in reversed(mesh.shape):
        coord.append(offset % extent)
        offset //= extent
    coord.reverse()
    index = [slice(0, int(extent)) for extent in global_shape]
    for axis, placement in enumerate(placements):
        if placement.kind is not PlacementKind.SHARD:
            continue
        piece = int(global_shape[placement.dim]) // mesh.shape[axis]
        index[placement.dim] = slice(coord[axis] * piece, (coord[axis] + 1) * piece)
    return tuple(index)


def _local_shape(index: tuple[slice, ...]) -> tuple[int, ...]:
    return tuple(axis.stop - axis.start for axis in index)


def _check_layout_placements(
    name: str,
    side: str,
    mesh: MeshSpec,
    placements: tuple[Placement, ...],
) -> None:
    """One sharded tensor dim at most, on the innermost mesh axis only."""
    outer = placements[:-1]
    if any(placement.kind is not PlacementKind.REPLICATE for placement in outer):
        raise ValueError(
            f"{name}: {side} placements {[p.canonical() for p in placements]} "
            f"over mesh {mesh.canonical()} may shard only the innermost axis"
        )


class _FrozenPlan:
    def __init__(self, plan: ReshardPlan) -> None:
        snapshot = copy.deepcopy(plan)
        snapshot.validate()
        if snapshot.misc:
            raise ValueError(
                "the MILES/SGLang collective integration supports all-bulk plans only"
            )
        if snapshot.bulk != sorted(snapshot.bulk, key=lambda entry: entry.canonical()):
            raise ValueError("MILES/SGLang bulk plans must be in canonical wire order")
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
                "the MILES/SGLang collective integration supports BF16 "
                f"base weights only; unsupported: {unsupported[:5]}"
            )
        if snapshot.source_partition_count != 1:
            raise ValueError(
                "the MILES/SGLang collective integration uses one reshard lane "
                f"(source_partition_count 1), got {snapshot.source_partition_count}"
            )
        # One lane holds every trainer rank, so every entry shares one source
        # mesh starting at lane rank 0 and one destination mesh right after it.
        src_meshes = {entry.src_mesh for entry in snapshot.bulk}
        dst_meshes = {entry.dst_mesh for entry in snapshot.bulk}
        if len(src_meshes) != 1 or len(dst_meshes) != 1:
            raise ValueError(
                "every MILES/SGLang plan entry must share one source mesh and "
                f"one destination mesh; got {sorted(m.canonical() for m in src_meshes)}"
                f" and {sorted(m.canonical() for m in dst_meshes)}"
            )
        src_mesh = next(iter(src_meshes))
        dst_mesh = next(iter(dst_meshes))
        if src_mesh.rank_offset != 0:
            raise ValueError(
                f"the source mesh must start at lane rank 0, got {src_mesh.canonical()}"
            )
        if dst_mesh.rank_offset != src_mesh.size:
            raise ValueError(
                "the destination mesh must start right after the trainer ranks: "
                f"{dst_mesh.canonical()} after {src_mesh.canonical()}"
            )
        sharded_sources = [
            entry.name
            for entry in snapshot.bulk
            if any(
                placement.kind is not PlacementKind.REPLICATE
                for placement in entry.src_placements
            )
        ]
        if sharded_sources:
            raise ValueError(
                "the MILES/SGLang collective integration takes a gathered "
                "(replicated) trainer source only; trainer-local shards are "
                f"not supported: {sharded_sources[:5]}"
            )
        for entry in snapshot.bulk:
            _check_layout_placements(
                entry.name, "src", entry.src_mesh, entry.src_placements
            )
            _check_layout_placements(
                entry.name, "dst", entry.dst_mesh, entry.dst_placements
            )
        self._plan = snapshot
        self._by_name = {entry.name: entry for entry in snapshot.bulk}
        self._src_mesh = src_mesh
        self._dst_mesh = dst_mesh
        self._sharded_destination = any(
            placement.kind is PlacementKind.SHARD
            for entry in snapshot.bulk
            for placement in entry.dst_placements
        )

    @property
    def source_partition_count(self) -> int:
        return self._plan.source_partition_count

    @property
    def src_mesh(self) -> MeshSpec:
        return self._src_mesh

    @property
    def dst_mesh(self) -> MeshSpec:
        return self._dst_mesh

    @property
    def sharded_destination(self) -> bool:
        return self._sharded_destination

    def names(self) -> list[str]:
        return self._plan.parameter_names()

    def entry(self, name: str):
        return self._by_name[name]

    def source_index(self, name: str, rank: int) -> tuple[slice, ...]:
        entry = self._by_name[name]
        return _expected_index(
            entry.global_shape, entry.src_mesh, entry.src_placements, rank
        )

    def destination_index(self, name: str, rank: int) -> tuple[slice, ...]:
        entry = self._by_name[name]
        return _expected_index(
            entry.global_shape, entry.dst_mesh, entry.dst_placements, rank
        )

    def capture(self) -> ReshardPlan:
        return copy.deepcopy(self._plan)

    def validate_topology(self, topology: Any) -> None:
        if self.source_partition_count != topology.source_partition_count:
            raise ValueError(
                "plan source_partition_count does not match the collective "
                f"topology: {self.source_partition_count} != "
                f"{topology.source_partition_count}"
            )
        # The single lane holds every trainer slot in source-mesh order, then
        # every generator slot in destination-mesh order.
        if len(topology.trainer_slots) != self._src_mesh.size:
            raise ValueError(
                "the collective topology must name one trainer slot per "
                f"source-mesh rank: {len(topology.trainer_slots)} != "
                f"{self._src_mesh.size} ({self._src_mesh.canonical()})"
            )
        if len(topology.generator_slots) != self._dst_mesh.size:
            raise ValueError(
                "the collective topology must name one generator slot per "
                f"destination-mesh rank: {len(topology.generator_slots)} != "
                f"{self._dst_mesh.size} ({self._dst_mesh.canonical()})"
            )
        if (
            self._sharded_destination
            and topology.m2n_abi_version != SHARDED_DESTINATION_ABI
        ):
            raise ValueError(
                "a plan with sharded destinations requires the "
                f"{SHARDED_DESTINATION_ABI!r} receiver ABI, got "
                f"{topology.m2n_abi_version!r}"
            )


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
    # Shape and dtype equality with the baseline is already guaranteed:
    # _tensor_signature validated both against the same expected values the
    # baseline was captured with. Only the storage address and device can
    # drift between captures.
    if current.address != baseline.address:
        raise RuntimeError(
            f"{name}: tensor storage address changed from "
            f"{baseline.address:#x} to {current.address:#x}"
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
    # Numeric order, so the error lists cuda:10 after cuda:2. Signatures
    # always carry indexed CUDA devices.
    devices = sorted(
        {signature.device for signature in signatures.values()},
        key=lambda device: int(device.rpartition(":")[2]),
    )
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
            else:
                raise TypeError(
                    "collective lane streams must be torch CUDA streams, got "
                    f"{type(stream).__name__}"
                )
            target.wait_event(ready)
