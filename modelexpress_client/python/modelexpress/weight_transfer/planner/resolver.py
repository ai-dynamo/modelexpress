# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side op-chain resolution (requires torch).

Replays op chains recorded during the lazy bake pass on meta tensors to
extract (storage_offset, shape, stride), then decomposes the strided region
into contiguous element runs.  Output is pure data suitable for server routing.
"""

from __future__ import annotations

import math

import torch

from ..engine.lazy import RecordedCopy
from ..protocol.ops import OpChain, OpSpec, SUPPORTED_OPS
from ..protocol.types import ResolvedRegion


def apply_chain(tensor: torch.Tensor, chain: OpChain) -> torch.Tensor:
    """Replay an OpChain on *tensor* and return the result."""
    for op in chain:
        fn = getattr(tensor, op.name, None) or getattr(torch, op.name, None)
        if fn is None:
            raise ValueError(f"Cannot replay op {op.name!r}: not found on tensor or torch")
        tensor = fn(*op.args, **op.kwargs)
    return tensor


def resolve_chain_region(
    chain: OpChain,
    root_shape: torch.Size,
    root_dtype: torch.dtype,
) -> tuple[int, torch.Size, tuple[int, ...]]:
    """Simulate a chain on a meta tensor and return (storage_offset, shape, stride)."""
    root = torch.empty(root_shape, dtype=root_dtype, device="meta")
    result = apply_chain(root, chain)
    return (result.storage_offset(), result.shape, tuple(result.stride()))


def region_elem_runs(
    storage_offset: int,
    shape: torch.Size,
    stride: tuple[int, ...],
) -> list[tuple[int, int]]:
    """Decompose a strided region into contiguous (elem_offset, count) runs.

    A fully-contiguous tensor produces exactly one run.
    """
    if len(shape) == 0:
        return [(storage_offset, 1)]
    if math.prod(shape) == 0:
        return []

    runs: list[tuple[int, int]] = []

    def _collect(base: int, dim: int) -> None:
        if dim == len(shape) - 1:
            if stride[dim] == 1:
                # Emit the whole row as one run.  Without this a contiguous
                # lm_head or embedding builds one tuple per element - hundreds
                # of millions of them - before the merge pass ever runs.
                runs.append((base, shape[dim]))
            else:
                for i in range(shape[dim]):
                    runs.append((base + i * stride[dim], 1))
            return
        for i in range(shape[dim]):
            _collect(base + i * stride[dim], dim + 1)

    _collect(storage_offset, 0)

    merged: list[tuple[int, int]] = [runs[0]]
    for offset, count in runs[1:]:
        po, pc = merged[-1]
        if po + pc == offset:
            merged[-1] = (po, pc + count)
        else:
            merged.append((offset, count))
    return merged


def resolve_copies(
    copies: list[RecordedCopy],
    tensor_shapes: dict[str, tuple[int, ...]],
    tensor_dtypes: dict[str, torch.dtype],
) -> list[ResolvedRegion]:
    """Resolve RecordedCopy objects into ResolvedRegion objects.

    Every recorded copy must have a source entry in tensor_shapes/tensor_dtypes:
    both maps and the lazy weights fed to the bake pass are built from the same
    TrainerTable, so a miss means the two have drifted apart.  Skipping it would
    drop the parameter from the plan and still report the transfer as a success.
    """
    regions: list[ResolvedRegion] = []

    for copy in copies:
        shape = tensor_shapes.get(copy.src_name)
        dtype = tensor_dtypes.get(copy.src_name)
        if shape is None or dtype is None:
            raise KeyError(
                f"recorded copy from {copy.src_name!r} has no "
                f"{'shape' if shape is None else 'dtype'} in the trainer table; "
                "dropping it would leave the parameter stale while the transfer "
                "still reported success"
            )

        src_offset, src_shape, src_stride = resolve_chain_region(
            copy.op_chain, torch.Size(shape), dtype
        )
        src_runs_pairs = region_elem_runs(src_offset, src_shape, src_stride)

        dst_runs_pairs = region_elem_runs(
            0,  # dst_addr is absolute; offset=0, elem address computed in router
            torch.Size(copy.dst_shape),
            copy.dst_stride,
        )

        elem_size = dtype.itemsize if hasattr(dtype, "itemsize") else (
            torch.finfo(dtype).bits // 8
            if dtype.is_floating_point
            else torch.iinfo(dtype).bits // 8
        )

        regions.append(ResolvedRegion(
            tensor_name=copy.src_name,
            src_elem_runs=[x for pair in src_runs_pairs for x in pair],
            dst_addr=copy.dst_addr,
            dst_elem_runs=[x for pair in dst_runs_pairs for x in pair],
            element_size=elem_size,
            dst_device_id=copy.dst_device_id,
        ))

    return regions
