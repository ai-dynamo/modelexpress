# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Slice resolution + intersection.

Turn a ``RecordedCopy`` (which slice of a full source tensor a param needs, and
where it lands) plus the published source shards into the exact byte segments to
RDMA-read per shard. No data moves; the intersection arithmetic (intersect /
paired_runs) is pure index/stride math. Slice RESOLUTION replays the op-chain on
a zero-storage ``meta`` tensor, so ``torch`` is imported lazily - only
``plan_pull`` at plan time needs it; the module and its arithmetic helpers
import (and unit-test) without torch.

Pipeline per copy:

  1. ``resolve_slice`` replays the recorded op-chain on a ``meta`` source to the
     strided view, then derives a per-dim ``[start, stop)`` box in the FULL
     source's index space PLUS the view-dim -> source-dim permutation (so
     transpose/permute slice, not just narrow/getitem), and reindexes the dest
     strides into source-dim order.
  2. ``intersect`` overlaps that needed box with each published shard's box.
  3. ``paired_runs`` walks each overlap into maximal runs that are contiguous in
     BOTH the shard buffer and the destination param, emitting one
     ``PullSegment`` per run.

Two properties worth noting:
  * Destination strides come from the ACTUAL captured ``dest_stride`` (read off
    the real dest view during capture), not a re-assumed row-major layout - a
    non-contiguous destination yields correct finer-grained runs instead of
    silently-wrong bytes.
  * ``__getitem__`` unit-step slices AND transpose/permute/t resolve via the
    meta-tensor replay, not just ``narrow``.

SCOPE: the needed slice must resolve to a permuted axis-aligned box of the source
(narrow/getitem/transpose/permute/t). A copy (``contiguous``/``reshape`` that
materializes), a rank change (int-index collapse, step != 1 slice), or a
dim-merging reshape - plus src/dst dtype mismatch - raise ``UnsupportedReshard``
-> the caller full-pulls.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from modelexpress.refit.reshard.types import (
    OpChain,
    RecordedCopy,
    UnsupportedReshard,
)


@dataclass
class Shard:
    """A published source shard covering full[shard_offset : shard_offset+shape].

    ``addr`` is the transfer-engine base address of element 0 of the shard's
    (row-major, contiguous) local buffer; ``session`` identifies the remote
    endpoint to READ from; ``elsize`` is bytes per element."""

    shard_offset: tuple  # per-dim start in the full tensor
    shape: tuple  # per-dim local shard shape
    session: str  # transfer-engine session id
    addr: int  # shard base address (element 0)
    elsize: int  # bytes per element


@dataclass
class PullSegment:
    """One contiguous run: read ``nbytes`` at ``src_addr`` (absolute, already
    shard.addr + offset) -> write at ``param.data_ptr() + dst_byte`` for the
    param resolved by ``param_name`` at refit time."""

    session: str
    src_addr: int
    param_name: str
    dst_byte: int
    nbytes: int


def _row_major_strides(shape) -> list:
    strides = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    return strides


def op_chain_to_box(op_chain: OpChain, global_shape) -> list:
    """Resolve a view/slice op-chain to a per-dim ``[start, stop)`` box in the
    full tensor's index space. Supports ``narrow``, unit-step ``__getitem__``
    slices, and layout-only ``contiguous``. Anything else raises
    ``UnsupportedReshard`` (caller full-pulls)."""
    box = [[0, int(g)] for g in global_shape]
    for op_name, args, _kw in op_chain:
        if op_name == "narrow":
            dim, start, size = args
            box[dim][0] += int(start)
            box[dim][1] = box[dim][0] + int(size)
        elif op_name == "contiguous":
            pass  # layout-only, no index change
        elif op_name == "__getitem__":
            key = args[0] if args else slice(None)
            keys = key if isinstance(key, tuple) else (key,)
            dim = 0
            for k in keys:
                if k is Ellipsis:
                    raise UnsupportedReshard(
                        "ellipsis indexing is not box-derivable -> full-pull"
                    )
                if not isinstance(k, slice):
                    # int index collapses a dim; changes rank -> not a plain box scatter.
                    raise UnsupportedReshard(
                        f"integer index {k!r} collapses a dim; not box-derivable -> full-pull"
                    )
                if k.step not in (None, 1):
                    raise UnsupportedReshard(
                        f"strided slice step={k.step} is not box-derivable -> full-pull"
                    )
                lo, hi = box[dim]
                extent = hi - lo
                start = 0 if k.start is None else int(k.start)
                stop = extent if k.stop is None else int(k.stop)
                box[dim][0] = lo + start
                box[dim][1] = lo + stop
                dim += 1
        else:
            raise UnsupportedReshard(
                f"op {op_name!r} is not box-derivable (transpose/reshape/permute/chunk) -> full-pull"
            )
    return [tuple(b) for b in box]


def _replay_view(op_chain: OpChain, global_shape) -> tuple:
    """Replay a recorded view/slice op-chain against a zero-storage ``meta``
    source tensor and read off the needed region's ``(storage_offset, shape,
    stride)`` in the source's row-major storage. Handles box ops (narrow/getitem)
    AND non-box ops (transpose/permute/t/...) uniformly - torch tracks the strided
    view for us. Raises ``UnsupportedReshard`` if any op is NOT a pure view (a
    copy - e.g. ``contiguous``/``reshape`` on a non-contiguous tensor - detaches
    storage, so the region can't be sliced). torch is imported lazily so the
    module stays importable, and its pure-arithmetic helpers testable, without
    torch (only plan-time slice resolution needs it)."""
    import torch  # noqa: PLC0415 - lazy: keep module import + arithmetic helpers torch-free

    src = torch.empty(tuple(int(g) for g in global_shape), device="meta")
    t: Any = src
    for op_name, args, kw in op_chain:
        t = getattr(t, op_name)(*args, **dict(kw))
    if not isinstance(t, torch.Tensor):
        raise UnsupportedReshard(
            f"op-chain resolved to non-tensor {type(t).__name__} -> full-pull"
        )
    # Must remain a pure VIEW of the source (shared storage). A copy resets the
    # base chain; PyTorch collapses view-of-view to the root, so a pure view has
    # ``t._base is src`` (or ``t is src`` for an empty chain).
    if t is not src and t._base is not src:
        raise UnsupportedReshard(
            "op-chain includes a copy (not a pure view) -> full-pull"
        )
    return (
        int(t.storage_offset()),
        tuple(int(s) for s in t.shape),
        tuple(int(s) for s in t.stride()),
    )


def _view_to_box_perm(
    v_offset: int, v_shape: tuple, v_stride: tuple, global_shape
) -> tuple:
    """From a replayed strided view ``(offset, shape, stride)``, derive the source
    index box + the view-dim -> source-dim permutation, for a RANK-PRESERVING
    permuted box (transpose/permute/t of a narrowed/sliced region). Raises
    ``UnsupportedReshard`` for a rank change or a non-permutation stride (a
    dim-merging reshape) -> caller full-pulls.

    A permuted box has each ``v_stride[d]`` equal to some source dim's row-major
    stride (the dim that view-dim ``d`` iterates); ``v_offset`` unravels to the
    box's per-source-dim start."""
    ndim = len(global_shape)
    if len(v_shape) != ndim:
        raise UnsupportedReshard(
            f"rank change {len(v_shape)}!={ndim} (reshape/squeeze) not box-derivable -> full-pull"
        )
    gstrides = _row_major_strides(global_shape)
    stride_to_dim: dict = {}
    for s in range(ndim):
        stride_to_dim.setdefault(gstrides[s], s)
    perm: list = [None] * ndim
    for d in range(ndim):
        if v_shape[d] == 1:
            continue  # size-1 dim: stride is ambiguous; assign a leftover dim below
        sdim = stride_to_dim.get(v_stride[d])
        if sdim is None:
            raise UnsupportedReshard(
                f"non-permutation stride {v_stride[d]} (dim-merging reshape) -> full-pull"
            )
        perm[d] = sdim
    used = {s for s in perm if s is not None}
    remaining = [s for s in range(ndim) if s not in used]
    ri = 0
    for d in range(ndim):
        if perm[d] is None:
            perm[d] = remaining[ri]
            ri += 1
    if sorted(perm) != list(range(ndim)):
        raise UnsupportedReshard(
            "view strides are not a permutation of source dims -> full-pull"
        )
    box_lo = [(v_offset // gstrides[s]) % global_shape[s] for s in range(ndim)]
    box = [[box_lo[s], box_lo[s]] for s in range(ndim)]
    for d in range(ndim):
        box[perm[d]][1] = box[perm[d]][0] + v_shape[d]

    # Self-consistency guard: the derived box must faithfully reconstruct the
    # replayed view - a clean permuted box wholly inside the source. Catches a
    # partial-dim / non-decomposing offset, an out-of-bounds extent (overflow
    # into the next dim), and any strided view that isn't actually box-shaped
    # (a dim-merge or L-shaped/overlapping region that slipped the stride check).
    if any(box[s][0] < 0 or box[s][1] > global_shape[s] for s in range(ndim)):
        raise UnsupportedReshard("resolved box exceeds source bounds -> full-pull")
    if sum(box[s][0] * gstrides[s] for s in range(ndim)) != v_offset:
        raise UnsupportedReshard(
            "view offset does not decompose to a box start -> full-pull"
        )
    return [tuple(b) for b in box], perm


def resolve_slice(copy: RecordedCopy, global_shape) -> tuple:
    """Resolve a copy's op-chain to ``(needed_box, dest_stride_in_source_order)`` -
    the one entry point ``plan_pull`` uses; it hides all op-chain detail.

    Replays the op-chain on a meta source to the strided view, then derives the
    source index box + the view-dim -> source-dim permutation. Box op-chains
    (narrow/getitem) come out as an identity permutation; transpose/permute come
    out permuted. The dest strides are reindexed from view-dim order into
    source-dim order so ``paired_runs`` (which walks the overlap in source-dim
    order) maps each source element to the right dest slot. Anything not a pure
    permuted view (a copy, rank change, or dim-merging reshape) raises
    ``UnsupportedReshard`` -> caller full-pulls."""
    v_off, v_shape, v_stride = _replay_view(copy.op_chain, global_shape)
    box, perm = _view_to_box_perm(v_off, v_shape, v_stride, global_shape)
    if tuple(copy.dest_shape) != tuple(v_shape):
        raise UnsupportedReshard(
            f"{copy.param_name}: dest {tuple(copy.dest_shape)} != resolved view {tuple(v_shape)} -> full-pull"
        )
    dest_stride_src = [0] * len(global_shape)
    for d, s in enumerate(perm):
        dest_stride_src[s] = copy.dest_stride[d]
    return box, dest_stride_src


def intersect(box_a: list, box_b: list):
    """Per-dim overlap of two equal-rank boxes; None if disjoint on any dim."""
    out = []
    for (a0, a1), (b0, b1) in zip(box_a, box_b, strict=True):
        lo, hi = max(a0, b0), min(a1, b1)
        if lo >= hi:
            return None
        out.append((lo, hi))
    return out


def paired_runs(overlap: list, src_origin, src_strides, dst_origin, dst_strides):
    """Decompose ``overlap`` (full-tensor coords) into runs contiguous in BOTH
    the source shard buffer (``src_origin``/``src_strides``) and the destination
    param region (``dst_origin``/``dst_strides``), yielding
    ``(src_elem_off, dst_elem_off, run_len_elems)``.

    Coalesces a trailing block of dims only while it is unit-packed in both
    layouts, so a strided source (e.g. a column-slice) yields many runs and a
    fully-contiguous overlap yields one. Works for arbitrary (possibly
    non-row-major) ``dst_strides`` - the captured real destination strides."""
    ndim = len(overlap)
    sizes = [hi - lo for lo, hi in overlap]
    src_start = [lo - o for (lo, _), o in zip(overlap, src_origin, strict=True)]
    dst_start = [lo - o for (lo, _), o in zip(overlap, dst_origin, strict=True)]

    # Grow the coalesced trailing block while dim d is exactly unit-packed
    # (stride == running run length) in both src and dst.
    run_len = 1
    p = ndim
    while (
        p - 1 >= 0 and src_strides[p - 1] == run_len and dst_strides[p - 1] == run_len
    ):
        run_len *= sizes[p - 1]
        p -= 1

    src_base = sum(src_start[d] * src_strides[d] for d in range(ndim))
    dst_base = sum(dst_start[d] * dst_strides[d] for d in range(ndim))

    if p == 0:
        yield src_base, dst_base, run_len
        return

    for idx in itertools.product(*[range(sizes[d]) for d in range(p)]):
        s_off = src_base + sum(idx[d] * src_strides[d] for d in range(p))
        d_off = dst_base + sum(idx[d] * dst_strides[d] for d in range(p))
        yield s_off, d_off, run_len


def plan_pull(
    copy: RecordedCopy,
    global_shape,
    src_dtype: Any,
    elsize: int,
    shards: list,
) -> list:
    """For one captured ``copy`` and the published ``shards`` of its source
    tensor, return the ``PullSegment``s that read exactly the needed slice.

    Raises ``UnsupportedReshard`` (caller full-pulls) on dtype mismatch, or an
    op-chain that isn't a pure permuted view (copy / rank change / dim-merge)."""
    if src_dtype != copy.dest_dtype:
        raise UnsupportedReshard(
            f"{copy.param_name}: source dtype {src_dtype} != dest dtype {copy.dest_dtype} "
            "-> convert/full-pull, not a raw byte copy"
        )

    # Resolve the op-chain (box or transpose/permute) to the source index box +
    # the dest strides reindexed into source-dim order.
    needed, dest_strides = resolve_slice(copy, global_shape)
    need_origin = [lo for lo, _ in needed]

    segments: list = []
    for sh in shards:
        shard_box = [(o, o + s) for o, s in zip(sh.shard_offset, sh.shape, strict=True)]
        overlap = intersect(needed, shard_box)
        if overlap is None:
            continue
        src_strides = _row_major_strides(sh.shape)
        for s_off, d_off, n in paired_runs(
            overlap, sh.shard_offset, src_strides, need_origin, dest_strides
        ):
            segments.append(
                PullSegment(
                    session=sh.session,
                    src_addr=sh.addr + s_off * sh.elsize,
                    param_name=copy.param_name,
                    dst_byte=(copy.dest_offset + d_off) * elsize,
                    nbytes=n * elsize,
                )
            )
    return segments
