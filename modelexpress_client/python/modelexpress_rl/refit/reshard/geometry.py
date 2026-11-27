# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Geometry capture.

Capture, per destination parameter, WHICH slice of each full source tensor the
engine's own weight loader reads. We feed zero-storage placeholder tensors
(``LazyWeight``) through the engine's ``model.load_weights`` and record:

  * the view/slice op-chain the loader applied to the source (which slice), and
  * the ``copy_`` destination's ``(offset, shape, stride)`` (where it lands).

No data ever moves here (the ``copy_`` sink is record-only). Downstream,
``slice_plan`` resolves the op-chain to a box of the full tensor and intersects
it with the published shards, and ``plan`` turns overlaps into byte segments to
pull.

Design notes:
  * Framework-neutral: the caller supplies the model (ideally a disposable
    ``meta`` twin) and the framework's default weight-loader; no engine import.
  * Bulk capture amortizes model-wide loader setup, then retries one source at a
    time when an unsupported operation interrupts it, so unsupported geometry is
    still attributed to the specific source before the receiver fails the update.
    It never produces an incorrect partial plan: the interrupted attempt rolls
    back every recorder accumulator (``_BakeRecorder.attempt``), so the retry
    starts from exactly the state the bulk attempt began with.
  * Allowlisted views, dtype conversions and loader zeroing retain a replay
    chain. Other arithmetic and bool-mask indexing raise through dispatch,
    ``UnsupportedReshard`` for that source, not wrong bytes.
"""

from __future__ import annotations

import functools
import logging
import math
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves, tree_map

from modelexpress_rl.refit.reshard.slice_plan import (
    _row_major_strides,
    _view_to_box_perm,
)
from modelexpress_rl.refit.reshard.types import (
    CaptureResult,
    EpilogueRef,
    _connected_groups,
    OpChain,
    OpSpec,
    RecordedCopy,
    UnsupportedReshard,
    summarize_unsupported,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CaptureResult",
    "LazyWeight",
    "OpChain",
    "OpSpec",
    "RecordedCopy",
    "UnsupportedReshard",
    "build_lazy_weights",
    "capture_geometry",
    "capture_weights",
    "convert_source_weights",
]

# Allowlist: pure view/slice/shape ops a weight loader may call. Maps
# ``torch.Tensor.fn`` -> the method name used on replay. Anything else reaches
# ``__torch_dispatch__`` and raises UnsupportedReshard.
_SUPPORTED_OPS: dict[Callable, str] = {
    torch.Tensor.narrow: "narrow",
    # select/split unstack a stacked source (e.g. a trainer that publishes its
    # experts stacked -> per-expert HF weights). select is rank-reducing; split is
    # a multi-return view like chunk. slice_plan resolves both to a source box.
    torch.Tensor.select: "select",
    torch.Tensor.view: "view",
    torch.Tensor.reshape: "reshape",
    torch.Tensor.__getitem__: "__getitem__",
    torch.Tensor.unsqueeze: "unsqueeze",
    torch.Tensor.squeeze: "squeeze",
    torch.Tensor.transpose: "transpose",
    torch.Tensor.t: "t",
    torch.Tensor.permute: "permute",
    torch.Tensor.flatten: "flatten",
    torch.Tensor.contiguous: "contiguous",
    torch.Tensor.chunk: "chunk",
    torch.Tensor.split: "split",
    # vLLM's fused-MoE expert loader unifies its fused and per-expert paths by
    # doing `experts_shard.unbind()`, reached via `unsqueeze(0)` for a per-expert
    # source. Without this entry every expert weight in a MoE model is classified
    # unsupported and the refit fails closed at ~5% coverage. Like `chunk` it is a
    # pure multi-return view, so the existing tuple handling in `_intercept`
    # applies unchanged.
    torch.Tensor.unbind: "unbind",
    torch.Tensor.to: "to",
    torch.Tensor.float: "float",
    torch.Tensor.bfloat16: "bfloat16",
    torch.Tensor.half: "half",
    torch.Tensor.double: "double",
    torch.Tensor.zero_: "zero_",
}

# Shape-of-another-tensor views. Recorded as the explicit-shape op so a chain
# never holds a tensor argument (mamba-style loaders `view_as` a destination
# slice).
_SHAPE_AS_OPS: dict[Callable, Callable] = {
    torch.Tensor.view_as: torch.Tensor.view,
    torch.Tensor.reshape_as: torch.Tensor.reshape,
}

_CONCAT_FUNCS = frozenset({torch.cat, torch.concat, torch.concatenate})
_STACK_FUNCS = frozenset({torch.stack})


def _snapshot_index(index):
    """Own constant indices without changing list-versus-tuple semantics."""
    if isinstance(index, torch.Tensor):
        if (
            isinstance(index, LazyWeight)
            or index.is_meta
            or index.dtype not in (torch.int32, torch.int64)
        ):
            raise UnsupportedReshard("index tensors must contain concrete integers")
        return index.detach().to(device="cpu", copy=True)
    if isinstance(index, (list, tuple)):
        return type(index)(_snapshot_index(item) for item in index)
    if isinstance(index, bool) or not isinstance(
        index, (int, slice, type(Ellipsis), type(None))
    ):
        raise UnsupportedReshard("only integer and slice indexing is supported")
    return index


def _basic_index(index):
    """Admit only indexing that yields a view: ints, slices, ``...`` and ``None``."""
    keys = index if isinstance(index, tuple) else (index,)
    for key in keys:
        if isinstance(key, bool) or not isinstance(
            key, (int, slice, type(Ellipsis), type(None))
        ):
            raise UnsupportedReshard(
                "assignment through advanced indexing is not a view of the destination"
            )
    return index


def _unmerge(shape, stride, root_shape):
    """Split each view dim that spans several row-major root dims.

    ``flatten`` over a stacked dim interleaves sources inside one view dim
    (``stack((gate, up), -1).flatten(-2)``). Returns, per view dim, the sizes
    that make every dim walk a single root dim, or None if it cannot.
    """
    root_strides = _row_major_strides(root_shape)
    plan = []
    for size, step in zip(shape, stride):
        if size == 1:
            plan.append((1,))
            continue
        dims = [
            d for d in range(len(root_shape))
            if root_strides[d] == step and root_shape[d] > 1
        ]
        if not dims:
            return None
        inner = dims[-1]
        if size <= root_shape[inner]:
            plan.append((size,))
            continue
        sizes, covered, dim = [root_shape[inner]], root_shape[inner], inner - 1
        while covered < size:
            if dim < 0 or size % covered:
                return None
            if root_shape[dim] == 1:
                dim -= 1
                continue
            outer = min(size // covered, root_shape[dim])
            sizes.insert(0, outer)
            covered *= outer
            dim -= 1
        if covered != size:
            return None
        plan.append(tuple(sizes))
    return plan


@dataclass(frozen=True)
class _Concat:
    """Sources a conversion concatenated, kept apart until the loader picks one.

    A trainer can publish pieces of one checkpoint tensor separately (gate and up
    projections that the checkpoint fuses, say), so the conversion ``torch.cat``s
    them. Every view the loader then takes is tracked against ``root``, a meta
    tensor of the concatenated shape. Once a view falls inside one part it is
    re-expressed as a view of that part, so each recorded copy still reads one
    published source.
    """

    root: torch.Tensor
    dim: int
    parts: tuple  # ((LazyWeight, start, stop), ...) along ``dim``

    def resolve(self, view: torch.Tensor):
        """Return ``(part, ops, meta)`` for a view inside one part, else None."""
        root = self.root
        if view.dtype != root.dtype or (view is not root and view._base is not root):
            return None
        try:
            box, perm = _view_to_box_perm(
                view.storage_offset(), tuple(view.shape), tuple(view.stride()),
                tuple(root.shape),
            )
        except UnsupportedReshard:
            return None
        low, high = box[self.dim]
        for part, start, stop in self.parts:
            if start <= low and high <= stop:
                break
        else:
            return None
        box[self.dim] = (low - start, high - start)
        ops = [
            ("narrow", (dim, lo, hi - lo), ())
            for dim, (lo, hi) in enumerate(box)
            if (lo, hi) != (0, part.shape[dim])
        ]
        kept = sorted(perm)
        ops += [
            ("select", (dim, 0), ())
            for dim in reversed(range(len(box)))
            if dim not in perm
        ]
        order = tuple(kept.index(dim) for dim in perm)
        if order != tuple(range(len(order))):
            ops.append(("permute", (order,), ()))
        meta = part._meta()
        with torch._C.DisableTorchFunctionSubclass():
            for name, args, kwargs in ops:
                meta = getattr(meta, name)(*args, **dict(kwargs))
        if tuple(meta.shape) != tuple(view.shape):
            return None
        return part, tuple(ops), meta

    def split(self, view: torch.Tensor, dest: torch.Tensor):
        """Pair each part's share of ``view`` with the matching ``dest`` slice."""
        root = self.root
        if view.dtype != root.dtype or (view is not root and view._base is not root):
            return None
        if tuple(dest.shape) != tuple(view.shape):
            return None
        plan = _unmerge(tuple(view.shape), tuple(view.stride()), tuple(root.shape))
        if plan is None:
            return None
        with torch._C.DisableTorchFunctionSubclass():
            for axis in reversed(range(len(plan))):
                if len(plan[axis]) > 1:
                    view = view.unflatten(axis, plan[axis])
                    dest = dest.unflatten(axis, plan[axis])
        try:
            box, perm = _view_to_box_perm(
                view.storage_offset(), tuple(view.shape), tuple(view.stride()),
                tuple(root.shape),
            )
        except UnsupportedReshard:
            return None
        if self.dim not in perm:
            return None
        axis = perm.index(self.dim)
        low, high = box[self.dim]
        pieces = []
        with torch._C.DisableTorchFunctionSubclass():
            for _, start, stop in self.parts:
                lo, hi = max(low, start), min(high, stop)
                if lo < hi:
                    pieces.append((
                        view.narrow(axis, lo - low, hi - lo),
                        dest.narrow(axis, lo - low, hi - lo),
                    ))
        return pieces


def _freeze_kwargs(kwargs: dict[str, Any]) -> tuple:
    return tuple((name, tuple(value) if isinstance(value, list) else value)
                 for name, value in sorted(kwargs.items()))


def _validate_physical_destination(shape, stride, offset, owner_shape, owner_stride):
    """Require a physical view to stay within its owner's addressed storage."""
    if (
        len(shape) != len(stride)
        or len(owner_shape) != len(owner_stride)
        or any(
            value < 0
            for values in (shape, stride, owner_shape, owner_stride)
            for value in values
        )
    ):
        raise UnsupportedReshard("invalid physical destination geometry")
    dimensions = sorted(
        (step, size) for step, size in zip(owner_stride, owner_shape) if size > 1
    )
    span = block = 1
    gaps = []
    for step, size in dimensions:
        if step < span:
            raise UnsupportedReshard("overlapping destination layout is unsupported")
        span += (size - 1) * step
        if step == block:
            block *= size
        else:
            gaps.append((step, size))
    if not math.prod(owner_shape):
        span = 0
    count = math.prod(shape)
    end = offset + (
        1 + sum((size - 1) * step for size, step in zip(shape, stride)) if count else 0
    )
    if offset < 0 or offset > span or end > span:
        raise UnsupportedReshard("destination view exceeds registered owner bounds")
    if (
        not count
        or not gaps
        or (
            offset == 0
            and tuple(shape) == tuple(owner_shape)
            and tuple(stride) == tuple(owner_stride)
        )
    ):
        return
    unit = math.gcd(*(step for step, _ in dimensions))
    if unit > 1:
        if offset % unit or any(
            size > 1 and step % unit for size, step in zip(shape, stride)
        ):
            raise UnsupportedReshard(
                "destination view addresses holes in owner storage"
            )
        return _validate_physical_destination(
            shape,
            tuple(step // unit if size > 1 else 0 for size, step in zip(shape, stride)),
            offset // unit,
            owner_shape,
            tuple(
                step // unit if size > 1 else 0
                for size, step in zip(owner_shape, owner_stride)
            ),
        )
    for start, stop in _strided_spans(shape, stride, offset):
        remainder = start
        for step, size in reversed(gaps):
            index, remainder = divmod(remainder, step)
            if index >= size:
                raise UnsupportedReshard(
                    "destination view addresses holes in owner storage"
                )
        if remainder + stop - start > block:
            raise UnsupportedReshard(
                "destination view addresses holes in owner storage"
            )


def _strided_spans(shape, strides, start, element_size=1):
    """Yield physical runs, retaining holes in a strided view."""
    block = element_size
    dimensions = []
    for stride, size in sorted(
        (stride * element_size, size)
        for size, stride in zip(shape, strides)
        if size > 1
    ):
        if not dimensions and stride <= block:
            block += stride * (size - 1)
        else:
            dimensions.append((stride, size))
    dimensions.reverse()

    def runs(axis, offset):
        if axis == len(dimensions):
            yield offset, offset + block
            return
        stride, size = dimensions[axis]
        for index in range(size):
            yield from runs(axis + 1, offset + index * stride)

    yield from runs(0, start)


def _runs_intersect(left_runs, right_runs):
    """Whether two byte-run iterators, each ordered by start, share a byte."""
    a, b = next(left_runs, None), next(right_runs, None)
    while a is not None and b is not None:
        if a[0] < b[1] and b[0] < a[1]:
            return True
        if a[1] <= b[0]:
            a = next(left_runs, None)
        else:
            b = next(right_runs, None)
    return False


def _owners_overlap(left, right):
    return _runs_intersect(
        *(
            _strided_spans(tensor.shape, tensor.stride(), start, tensor.element_size())
            for start, _, tensor, _ in (left, right)
        )
    )


def _destination_overlap_pairs(destinations):
    groups = {}
    for name, tensor in destinations:
        if not tensor.numel():
            continue
        storage = tensor.untyped_storage()
        key = (tensor.device, storage._cdata if tensor.is_meta else None)
        start = tensor.storage_offset() * tensor.element_size()
        if not tensor.is_meta:
            start += storage.data_ptr()
        end = start + tensor.element_size() * (
            1
            + sum(
                (size - 1) * stride
                for size, stride in zip(tensor.shape, tensor.stride())
            )
        )
        groups.setdefault(key, []).append((start, end, tensor, name))
    for group in groups.values():
        active = []
        for current in sorted(group, key=lambda entry: entry[0]):
            active = [entry for entry in active if entry[1] > current[0]]
            for previous in active:
                if _owners_overlap(previous, current):
                    yield previous, current
            active.append(current)


def _overlapping_owners(destinations):
    return {
        entry[3]
        for left, right in _destination_overlap_pairs(destinations)
        if left[2] is not right[2]
        for entry in (left, right)
    }


def overlapping_destination_groups(destinations):
    """Connected physical overlap components, excluding disjoint shared slabs."""
    destinations = tuple(
        (name, tensor) for name, tensor in destinations if not tensor.is_meta
    )
    links = (
        (left[3], right[3]) for left, right in _destination_overlap_pairs(destinations)
    )
    return tuple(
        group for group in _connected_groups((name for name, _ in destinations), links)
        if len(group) > 1
    )


@dataclass
class _BakeRecorder:
    copies: list = field(default_factory=list)
    unattributed: int = 0
    # Set by the loader stamp to the destination param's full name so copy_ can
    # attribute the write.
    current: Any = None
    destinations: list = field(default_factory=list)
    overlapping_owners: set = field(default_factory=set)
    weights: list = field(default_factory=list)
    source_replay: bool = False

    def destination(self, name, tensor, owner):
        if name in self.overlapping_owners:
            raise UnsupportedReshard(
                f"overlapping registered destination {name!r} requires ordered source replay"
            )
        physical_offset = tensor.storage_offset() - owner.storage_offset()
        physical_stride = tuple(tensor.stride())
        if not self.source_replay and not owner.is_contiguous():
            raise UnsupportedReshard(
                "noncontiguous destinations require ordered source replay"
            )
        _validate_physical_destination(
            tuple(tensor.shape), physical_stride, physical_offset,
            tuple(owner.shape), tuple(owner.stride()),
        )
        return name, physical_offset, physical_stride

    def resolve(self, tensor):
        for name, destination in self.destinations:
            if tensor is destination:
                return self.destination(name, tensor, destination)
        storage = tensor.untyped_storage()._cdata
        candidates = []
        for name, destination in self.destinations:
            if (
                destination.device == tensor.device
                and destination.dtype == tensor.dtype
                and destination.untyped_storage()._cdata == storage
            ):
                if name == self.current:
                    return self.destination(name, tensor, destination)
                candidates.append((name, destination))
        matches = []
        for name, destination in candidates:
            try:
                matches.append(self.destination(name, tensor, destination))
            except UnsupportedReshard:
                continue
        if len(matches) == 1:
            return matches[0]
        if candidates:
            raise UnsupportedReshard(
                "ambiguous or out-of-bounds registered destination view"
            )
        return None

    @contextmanager
    def attempt(self) -> Iterator[None]:
        """Run a capture attempt whose records are discarded unless it completes.

        Every mutable accumulator on this recorder must be restored here. A field
        added to this class without being restored would let an abandoned attempt
        contribute to the next one.
        """
        copies = len(self.copies)
        unattributed, current = self.unattributed, self.current
        weight_count = len(self.weights)
        weights = [
            (tensor, tensor._ops)
            for ref in self.weights
            if (tensor := ref()) is not None
        ]
        aliases = {
            id(tensor._aliases): (tensor._aliases, list(tensor._aliases))
            for tensor, _ in weights
        }
        try:
            yield
        except BaseException:
            del self.copies[copies:]
            self.unattributed = unattributed
            self.current = current
            for tensor, ops in weights:
                tensor._ops = ops
            for group, saved in aliases.values():
                group[:] = saved
            del self.weights[weight_count:]
            raise


class LazyWeight(torch.Tensor):
    """Zero-storage placeholder that records how a loader slices a source tensor.

    Built via ``_make_wrapper_subclass`` so ``.shape``/``.dtype``/``.device``/
    ``.size()``/``.dim()`` work with no storage. Each allowlisted op returns a new
    ``LazyWeight`` with the op appended to its chain; ``copy_`` is the sink that
    records a ``RecordedCopy`` (record-only - no data moves). Any op outside the
    allowlist raises ``UnsupportedReshard`` in ``__torch_dispatch__``.
    """

    _name: str
    _ops: tuple
    _recorder: Any
    _concat: _Concat | None = None

    @staticmethod
    def __new__(
        cls, name, shape, dtype, device, ops=(), recorder=None, aliases=None, meta=None
    ):
        if meta is None:
            meta = torch.empty(shape, dtype=dtype, device="meta")
        t = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            strides=meta.stride(),
            storage_offset=meta.storage_offset(),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        t._meta_tensor = meta
        t._name = name
        t._ops = tuple(ops)
        t._recorder = recorder
        t._aliases = [] if aliases is None else aliases
        t._aliases.append(weakref.ref(t))
        if recorder is not None:
            recorder.weights.append(weakref.ref(t))
        return t

    def _meta(self) -> torch.Tensor:
        return self._meta_tensor

    def _child(self, meta, *new_ops, device=None, aliases=True) -> LazyWeight:
        device = self.device if device is None else device
        concat = self._concat
        if concat is not None:
            resolved = concat.resolve(meta)
            if resolved is not None:
                part, ops, part_meta = resolved
                return part._child(part_meta, *ops, device=device)
        child = LazyWeight(
            self._name,
            meta.shape,
            meta.dtype,
            device,
            self._ops + new_ops,
            self._recorder,
            self._aliases if aliases else None,
            meta,
        )
        child._concat = concat
        return child

    @classmethod
    def _concatenate(cls, tensors, dim=0, *, axis=None, out=None) -> LazyWeight:
        if axis is not None:
            dim = axis
        parts = tuple(tensors) if isinstance(tensors, (list, tuple)) else ()
        if out is not None or not parts:
            raise UnsupportedReshard("concatenation needs a sequence of sources")
        first = parts[0]
        for part in parts:
            if not isinstance(part, cls) or part._concat is not None:
                raise UnsupportedReshard(
                    "concatenation mixes a published source with another tensor"
                )
            if (
                part._recorder is not first._recorder
                or part.dtype != first.dtype
                or part.device != first.device
            ):
                raise UnsupportedReshard(
                    f"concatenation of differently typed sources on lazy {part._name!r}"
                )
        with torch._C.DisableTorchFunctionSubclass():
            shape = torch.cat([part._meta() for part in parts], dim).shape
        dim = dim % len(shape)
        root = torch.empty(shape, dtype=first.dtype, device="meta")
        spans, start = [], 0
        for part in parts:
            if part.dim() != len(shape):
                # torch.cat skips a 1-D empty operand along any dim.
                continue
            stop = start + part.shape[dim]
            spans.append((part, start, stop))
            start = stop
        composite = cls(
            "cat(" + ", ".join(part._name for part in parts) + ")",
            shape, first.dtype, first.device, recorder=first._recorder, meta=root,
        )
        composite._concat = _Concat(root, dim, tuple(spans))
        return composite

    @classmethod
    def _stack(cls, tensors, dim=0, *, out=None) -> LazyWeight:
        parts = tuple(tensors) if isinstance(tensors, (list, tuple)) else ()
        if out is not None or not parts or not all(isinstance(p, cls) for p in parts):
            raise UnsupportedReshard(
                "stacking mixes a published source with another tensor"
            )
        dim = dim % (parts[0].dim() + 1)
        return cls._concatenate([part.unsqueeze(dim) for part in parts], dim)

    @classmethod
    def _record_copy(cls, dest, src):
        """The recording sink: note where the slice lands; do NOT copy."""
        pieces = ((src, dest),)
        if src._concat is not None:
            pieces = src._concat.split(src._meta(), dest)
            if pieces is None:
                raise UnsupportedReshard(
                    f"copy of concatenated sources on lazy {src._name!r} "
                    "is not a view of each source"
                )
            pieces = tuple((src._child(view), target) for view, target in pieces)
            if any(lazy._concat is not None for lazy, _ in pieces):
                raise UnsupportedReshard(
                    f"copy of concatenated sources on lazy {src._name!r} "
                    "is not a view of each source"
                )
        rec = src._recorder
        for lazy, target in pieces:
            destination = rec.resolve(target) if rec is not None else None
            if destination is not None:
                rec.copies.append(
                    RecordedCopy(
                        src_name=lazy._name,
                        op_chain=lazy._ops,
                        param_name=destination[0],
                        dest_offset=destination[1],
                        dest_shape=tuple(target.shape),
                        dest_stride=destination[2],
                        dest_dtype=target.dtype,
                    )
                )
            elif rec is not None:
                rec.unattributed += 1
        # Record-only: the loader proceeds as if the copy happened, with no
        # device/meta coupling (dest may be meta/cpu/cuda).
        return dest

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        if func is torch.Tensor.copy_:
            dest = args[0]
            src = args[1] if len(args) > 1 else kwargs.get("src")
            if isinstance(src, cls) and not isinstance(dest, cls):
                return cls._record_copy(dest, src)

        # `dest[index] = src` is `dest[index].copy_(src)` for view indexing.
        if func is torch.Tensor.__setitem__ and len(args) == 3:
            dest, index, src = args
            if isinstance(src, cls) and not isinstance(dest, cls):
                with torch._C.DisableTorchFunctionSubclass():
                    target = dest[_basic_index(index)]
                if tuple(target.shape) != tuple(src.shape):
                    raise UnsupportedReshard(
                        f"broadcasting assignment {tuple(src.shape)} -> "
                        f"{tuple(target.shape)} on lazy {src._name!r}"
                    )
                cls._record_copy(target, src)
                return None

        # detach is the identity on values; a new placeholder mirrors the new
        # tensor object (loaders set attributes on the detached weight).
        if func is torch.Tensor.detach and args and isinstance(args[0], cls):
            return args[0]._child(args[0]._meta())

        if func in _CONCAT_FUNCS:
            return cls._concatenate(*args, **kwargs)
        if func in _STACK_FUNCS:
            return cls._stack(*args, **kwargs)

        if func in _SHAPE_AS_OPS and args and isinstance(args[0], cls):
            other = args[1] if len(args) > 1 else kwargs["other"]
            target = _SHAPE_AS_OPS[func]
            return cls._intercept(
                args[0], target, _SUPPORTED_OPS[target], (tuple(other.shape),), {}
            )

        op_name = _SUPPORTED_OPS.get(func)
        if op_name is not None and args and isinstance(args[0], cls):
            return cls._intercept(args[0], func, op_name, tuple(args[1:]), kwargs)

        # Metadata reads (.shape/.size()/.dim()/.dtype/.device) resolve here from
        # the wrapper subclass. Anything needing real data falls through to
        # __torch_dispatch__ and raises.
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def _intercept(cls, self_, func, op_name, args, kwargs):
        if self_._concat is not None:
            if op_name == "contiguous":
                # Layout only: the loader reads the same values, which are
                # still tracked as a view of the concatenation.
                return self_
            try:
                return cls._intercept_view(self_, func, op_name, args, kwargs)
            except RuntimeError as error:
                if isinstance(error, UnsupportedReshard):
                    raise
                raise UnsupportedReshard(
                    f"{op_name!r} on lazy {self_._name!r} is not a view: {error}"
                ) from error
        return cls._intercept_view(self_, func, op_name, args, kwargs)

    @classmethod
    def _intercept_view(cls, self_, func, op_name, args, kwargs):
        if op_name == "__getitem__":
            args = (_snapshot_index(args[0]), *args[1:])
        meta = self_._meta()
        device = self_.device
        meta_args, meta_kwargs = args, kwargs
        if op_name == "to":
            args, kwargs = tuple(args), dict(kwargs)
            if args and isinstance(args[0], torch.Tensor):
                other, *rest = args
                args = (other.device, other.dtype, *rest)
            if args and isinstance(args[0], (str, torch.device)):
                device = torch.device(args[0])
                if device.type == "meta":
                    device = self_.device
                    args = (device, *args[1:])
                meta_args = (torch.device("meta"), *args[1:])
            else:
                meta_args = args
            meta_kwargs = dict(kwargs)
            if "device" in kwargs and kwargs["device"] is not None:
                device = torch.device(kwargs["device"])
                if device.type == "meta":
                    device = self_.device
                    kwargs["device"] = device
                meta_kwargs["device"] = torch.device("meta")
        with torch._C.DisableTorchFunctionSubclass():
            meta_result = func(meta, *meta_args, **meta_kwargs)
            if (
                op_name == "to"
                and device != self_.device
                and (
                    device.type != self_.device.type
                    or device.type == "cpu"
                    or (device.index is not None and self_.device.index is not None)
                )
                and meta_result.untyped_storage()._cdata
                == meta.untyped_storage()._cdata
            ):
                meta_result = meta_result.clone(memory_format=torch.preserve_format)
        recorded_args = (
            args if op_name == "__getitem__"
            else tuple(tuple(arg) if isinstance(arg, list) else arg for arg in args)
        )
        base_op = (op_name, recorded_args, _freeze_kwargs(kwargs))
        if op_name == "zero_":
            if self_._concat is not None:
                raise UnsupportedReshard(
                    f"zero_ of concatenated sources on lazy {self_._name!r}"
                )
            if any(ref() is not None and ref() is not self_ for ref in self_._aliases):
                raise UnsupportedReshard(
                    "source zero_ with live aliases cannot be replayed faithfully"
                )
            self_._ops += (base_op,)
            return self_
        if isinstance(meta_result, torch.Tensor):
            return self_._child(
                meta_result,
                base_op,
                device=device,
                aliases=meta_result.untyped_storage()._cdata
                == meta.untyped_storage()._cdata,
            )
        if isinstance(meta_result, (tuple, list)):
            # Multi-return (chunk/split): one child per output + a trailing
            # __getitem__ so replay can index back into the result.
            return tuple(
                self_._child(m, base_op, ("__getitem__", (i,), ()))
                for i, m in enumerate(meta_result)
            )
        raise UnsupportedReshard(
            f"{op_name!r} on lazy {self_._name!r} returned a non-tensor "
            f"({type(meta_result).__name__}); cannot defer."
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # Dispatch reaches a subclass only with one of its instances among the
        # arguments, possibly nested in a list (aten.cat). Re-running ``func``
        # here would dispatch back to this method forever.
        lazy = next(
            (a for a in tree_leaves((args, kwargs or {})) if isinstance(a, cls)), None
        )
        name, chain = (lazy._name, lazy._ops) if lazy is not None else ("?", ())
        raise UnsupportedReshard(
            f"unsupported op {func} on lazy {name!r} (chain={chain}); "
            f"supported: {sorted(_SUPPORTED_OPS.values())} + copy_."
        )


def _install_stamps(
    model: torch.nn.Module,
    recorder: _BakeRecorder,
    default_weight_loader: Callable | None,
) -> list:
    """Wrap each param's ``weight_loader`` so it sets ``recorder.current = name``
    around the loader, letting ``copy_`` attribute the write to that param.
    Returns saved ``(param, original_loader)`` pairs (a throwaway twin can skip
    restoration)."""
    saved = []
    for name, param in (*model.named_parameters(), *model.named_buffers()):
        original = getattr(param, "weight_loader", None)
        inner = original if original is not None else default_weight_loader
        if inner is None:
            continue

        def make_stamp(inner, name):
            @functools.wraps(inner)
            def stamp(*a, **kw):
                # Signature-transparent on purpose. vLLM's fused-MoE loader is
                # invoked entirely by keyword (`param=`, `loaded_weight=`,
                # `shard_id=`, `expert_id=`), so a stamp that named its first
                # parameter positionally raised TypeError for every expert.
                previous = recorder.current
                recorder.current = name
                try:
                    return inner(*a, **kw)
                finally:
                    recorder.current = previous

            return stamp

        param.weight_loader = make_stamp(inner, name)
        saved.append((param, original))
    return saved


def _restore_stamps(saved: list) -> None:
    for param, original in saved:
        if original is None:
            try:
                del param.weight_loader
            except AttributeError:
                pass
        else:
            param.weight_loader = original


def build_lazy_weights(
    manifest: list[tuple[str, Any, tuple]],
    *,
    device: torch.device | str = "meta",
) -> dict[str, LazyWeight]:
    """One ``LazyWeight`` placeholder per published source, all sharing a recorder.

    ``manifest`` is ``(name, dtype, shape)`` for every full source tensor. The
    result can be passed straight to :func:`capture_weights`, or first through a
    source conversion (a trainer-native -> HF ``dict -> dict`` mapping) whose view
    ops each lazy records - so a converted lazy's ``_name`` stays its ORIGINAL
    published source and its ``_ops`` carry the conversion prefix. The shared
    recorder is what :func:`capture_weights` reads its recorded copies from.
    """
    recorder = _BakeRecorder()
    return {
        name: LazyWeight(name, torch.Size(shape), dtype, device, recorder=recorder)
        for name, dtype, shape in manifest
    }


def _shared_recorder(weights: dict) -> _BakeRecorder:
    # Every value must be a recording LazyWeight: an ordinary tensor would be sent
    # to load_weights and its copies dropped (the copy_ sink only fires for a
    # LazyWeight), silently under-capturing the model.
    recorders: dict = {}
    for name, w in weights.items():
        if not isinstance(w, LazyWeight) or w._recorder is None:
            raise ValueError(
                f"capture_weights requires LazyWeights from build_lazy_weights; "
                f"{name!r} is a {type(w).__name__}"
            )
        recorders[id(w._recorder)] = w._recorder
    if len(recorders) != 1:
        raise ValueError(
            "capture_weights expects LazyWeights that share one recorder "
            "(build them with build_lazy_weights)"
        )
    return next(iter(recorders.values()))


def _byte_span(offset, shape, stride, element_size):
    """``[start, stop)`` bytes a strided view addresses in its storage."""
    if not math.prod(shape):
        return offset * element_size, offset * element_size
    extent = 1 + sum((size - 1) * step for size, step in zip(shape, stride))
    return offset * element_size, (offset + extent) * element_size


def _views_overlap(left, right):
    """Whether two ``(offset, shape, stride, element_size)`` views share a byte.

    The bounding spans filter first; then the runs each view addresses are
    compared, so interleaved views such as ``p[:, 0]`` and ``p[:, 1]`` stay
    apart. A view whose own elements overlap is compared by its span.
    """
    spans = [_byte_span(*view) for view in (left, right)]
    if any(start == stop for start, stop in spans) or not (
        spans[0][0] < spans[1][1] and spans[1][0] < spans[0][1]
    ):
        return False

    def runs(view, span):
        offset, shape, stride, element_size = view
        extent = 1
        for step, size in sorted(
            (step, size) for size, step in zip(shape, stride) if size > 1
        ):
            if step < extent:
                return iter((span,))
            extent += (size - 1) * step
        return _strided_spans(shape, stride, offset * element_size, element_size)

    return _runs_intersect(*(runs(view, span) for view, span in zip((left, right), spans)))


def _view_of(tensor):
    return (
        tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()),
        tensor.element_size(),
    )


def _same_view(tensor, owner):
    return (
        tensor.untyped_storage()._cdata == owner.untyped_storage()._cdata
        and tensor.storage_offset() == owner.storage_offset()
        and tuple(tensor.shape) == tuple(owner.shape)
        and tuple(tensor.stride()) == tuple(owner.stride())
        and tensor.dtype == owner.dtype
    )


def _out_of_place(func):
    """``aten.mul.Tensor`` for ``aten.mul_.Tensor``; None if there is none."""
    namespace, _, op = func._schema.name.partition("::")
    if namespace != "aten" or not op.endswith("_") or op == "copy_":
        return None
    packet = getattr(torch.ops.aten, op[:-1], None)
    overload = getattr(packet, func._overloadname, None) if packet is not None else None
    # An in-place view op (``t_``, ``squeeze_``) changes only the metadata of
    # the tensor it is called on; its out-of-place form returns a view.
    if (
        overload is None
        or _writes(overload)
        or any(value.alias_info is not None for value in overload._schema.returns)
    ):
        return None
    return overload


def _zero_write(func, args) -> bool:
    """Whether ``func`` writes only bitwise zeros: ``zero_`` or ``fill_(0)``."""
    if func is torch.ops.aten.zero_.default:
        return True
    if func is not torch.ops.aten.fill_.Scalar or len(args) < 2:
        return False
    value = args[1]
    if isinstance(value, int):
        return value == 0
    return isinstance(value, float) and value == 0 and math.copysign(1.0, value) > 0


def _writes(func) -> bool:
    return any(
        argument.alias_info is not None and argument.alias_info.is_write
        for argument in func._schema.arguments
    )


class _CaptureMutationGuard(TorchDispatchMode):
    """Reject loader writes that a recorded copy cannot reproduce.

    Any write to live destination storage would change served weights. A write
    over bytes a source was already copied into computes the value the loader
    leaves there, which replaying the recorded copy alone would miss. Writes
    elsewhere in a destination, such as zeroing the padding past a copied
    vocabulary, are left alone: replay starts every destination from zeros.

    With ``trace_epilogues`` (ordered source replay), a loader that recomputes
    a whole destination from itself after copying into it (vLLM's
    ``composed_weight_loader``: Mamba2 stores ``A = -exp(A_log)``; or an in-place
    ``param.data.copy_(w).mul_(scale)``) is recorded instead: the aten ops between
    the copy and the rewrite become the copy's ``epilogue``, which replay runs on
    the destination after the copy lands. A traced value stops being traced
    once a write lands in its storage. The rewrite is rejected when a later copy
    under any name shares its bytes, or when it reads bytes that an unrecorded
    non-zero write set, since replay would not reproduce what it read.
    """

    def __init__(self, destinations, recorder=None, *, trace_epilogues=False):
        super().__init__()
        self._storages = {
            tensor.untyped_storage()._cdata
            for _, tensor in destinations
            if not tensor.is_meta
        }
        self._meta = {
            name: tensor for name, tensor in destinations if tensor.is_meta
        }
        self._meta_storages = {
            tensor.untyped_storage()._cdata for tensor in self._meta.values()
        }
        self._owners: dict[int, list] = {}
        for name, tensor in self._meta.items():
            self._owners.setdefault(tensor.untyped_storage()._cdata, []).append(
                (name, tensor)
            )
        self._recorder = recorder
        self._trace = trace_epilogues and recorder is not None
        self._by_name: dict[str, list] = {}
        self._indexed = 0
        self._last = None
        self.reset()

    def reset(self):
        """Forget loader state from an abandoned capture attempt."""
        # id(tensor) -> (param, copies into param when the trace began, steps,
        # index of this tensor's step); ``_alive`` keeps the ids stable.
        self._derived: dict[int, tuple] = {}
        self._alive: list = []
        # storage -> ids of the traced values that live in it.
        self._traced: dict[int, list] = {}
        # Destination storage -> views that unrecorded non-zero writes set.
        self._unrecorded: dict[int, list] = {}

    def _copies_by_name(self):
        # Indexed incrementally: capture can record a copy per expert slice, and
        # this runs for every op that reads or writes a destination.
        copies = self._recorder.copies
        indexed = self._indexed
        if indexed > len(copies) or (indexed and copies[indexed - 1] is not self._last):
            self._by_name.clear()
            indexed = 0
        for copy in copies[indexed:]:
            self._by_name.setdefault(copy.param_name, []).append(copy)
        self._indexed = len(copies)
        self._last = copies[-1] if copies else None
        return self._by_name

    def _copies_into(self, name):
        return len(self._copies_by_name().get(name, ()))

    def _copy_view(self, copy):
        """The recorded copy's destination as a storage view, if it is meta."""
        owner = self._meta.get(copy.param_name)
        if owner is None:
            return None, None
        view = (
            owner.storage_offset() + copy.dest_offset, copy.dest_shape,
            copy.dest_stride, owner.element_size(),
        )
        return owner.untyped_storage()._cdata, view

    def _overwritten_copy(self, tensor):
        storage = tensor.untyped_storage()._cdata
        if self._recorder is None or storage not in self._meta_storages:
            return None
        written = _view_of(tensor)
        copies = self._copies_by_name()
        for name, _ in self._owners[storage]:
            for copy in copies.get(name, ()):
                if _views_overlap(written, self._copy_view(copy)[1]):
                    return name
        return None

    def _source_of(self, tensor):
        """``(param, copies, steps, ref)`` if ``tensor`` is a traced value."""
        derived = self._derived.get(id(tensor))
        if derived is not None:
            name, copies, steps, index = derived
            return name, copies, steps, EpilogueRef(index)
        if not tensor.is_meta:
            return None
        for name, owner in self._owners.get(tensor.untyped_storage()._cdata, ()):
            if _same_view(tensor, owner):
                copies = self._copies_into(name)
                return (name, copies, (), EpilogueRef(-1)) if copies else None
        return None

    def _step(self, func, args, kwargs):
        """Template ``func`` over traced values of one destination, or None.

        Returns ``(param, copies, steps)`` with the step appended. Inputs must be
        that destination, values computed from it, or zero-dim constants.
        """
        leaves = tree_leaves((args, kwargs))
        if any(
            isinstance(leaf, torch.device) and leaf.type == "meta" for leaf in leaves
        ):
            return None
        tensors = [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]
        sources = [self._source_of(tensor) for tensor in tensors]
        traced = [source for source in sources if source is not None]
        if not traced:
            return None
        name, copies, steps, _ = max(traced, key=lambda source: len(source[2]))
        if any(
            (other[0], other[1]) != (name, copies)
            or any(a is not b for a, b in zip(other[2], steps))
            for other in traced
        ):
            return None
        refs = {}
        for tensor, source in zip(tensors, sources):
            if source is not None:
                refs[id(tensor)] = source[3]
            elif not tensor.is_meta and tensor.dim() == 0:
                refs[id(tensor)] = tensor.detach().clone()
            else:
                return None

        def template(value):
            return refs[id(value)] if isinstance(value, torch.Tensor) else value

        step = (func, tree_map(template, tuple(args)), tree_map(template, dict(kwargs)))
        return name, copies, (*steps, step)

    def _trace_step(self, func, args, kwargs, out):
        # Views of a destination are not followed; an epilogue computes a new
        # value from the destination.
        if (
            not isinstance(out, torch.Tensor)
            or out.untyped_storage()._cdata in self._meta_storages
        ):
            return
        traced = self._step(func, args, kwargs)
        if traced is not None:
            name, copies, steps = traced
            self._derived[id(out)] = (name, copies, steps, len(steps) - 1)
            self._alive.append(out)
            self._traced.setdefault(out.untyped_storage()._cdata, []).append(id(out))

    def _forget_traces(self, tensor):
        """Drop traced values a write changed, through any view of their storage.

        A trace replays the ops that computed a value, so it no longer
        describes the value once a write lands in its storage.
        """
        for key in self._traced.pop(tensor.untyped_storage()._cdata, ()):
            self._derived.pop(key, None)

    def _record_epilogue(self, func, args, kwargs):
        """Record a whole-destination rewrite as the last copy's epilogue.

        Either ``copy_`` of a value computed from the destination, or an
        in-place op on it (``param.data.copy_(w).mul_(scale)``), replayed as the
        op's out-of-place form. Returns False when the write is not such a
        rewrite, and raises when it is one that replay cannot reproduce.
        """
        if not self._trace or not isinstance(args[0], torch.Tensor):
            return False
        storage = args[0].untyped_storage()._cdata
        names = [
            name for name, owner in self._owners[storage] if _same_view(args[0], owner)
        ]
        if len(names) != 1:
            return False
        (name,) = names
        destination = self._meta[name]
        if func is torch.ops.aten.copy_.default:
            derived = self._derived.get(id(args[1]))
            if derived is None:
                return False
            traced_name, copies, steps, index = derived
            steps = steps[: index + 1]
            value = args[1]
        else:
            out_of_place = _out_of_place(func)
            traced = (
                self._step(out_of_place, args, kwargs)
                if out_of_place is not None
                else None
            )
            if traced is None:
                return False
            traced_name, copies, steps = traced
            value = out_of_place(*args, **kwargs)
        if traced_name != name or copies != self._copies_into(name):
            return False
        recorded = self._recorder.copies
        last = max(i for i, copy in enumerate(recorded) if copy.param_name == name)
        if recorded[last].epilogue:
            return False
        # ``copy_`` broadcasts its source; an in-place op keeps its shape.
        shape = tuple(destination.shape)
        try:
            fits = isinstance(value, torch.Tensor) and (
                torch.broadcast_shapes(tuple(value.shape), shape) == shape
                if func is torch.ops.aten.copy_.default
                else tuple(value.shape) == shape
            )
        except RuntimeError:
            fits = False
        if not fits:
            raise UnsupportedReshard(
                f"loader rewrites destination {name!r} with a value that does not "
                f"fill it: {func}"
            )
        # Replay runs the epilogue right after copy ``last`` on a destination
        # that holds only recorded copies, so no later copy and no unrecorded
        # write may touch the bytes the loader's rewrite read and wrote.
        whole = _view_of(destination)
        for copy in recorded[last + 1 :]:
            other, view = self._copy_view(copy)
            if other == storage and _views_overlap(whole, view):
                raise UnsupportedReshard(
                    f"loader rewrites destination {name!r} after copying "
                    f"{copy.src_name!r} into {copy.param_name!r}, which shares its "
                    "bytes; replay would run the rewrite first"
                )
        if any(_views_overlap(whole, view) for view in self._unrecorded.get(storage, ())):
            raise UnsupportedReshard(
                f"loader rewrites destination {name!r} from bytes that an "
                "unrecorded non-zero write set"
            )
        recorded[last] = replace(
            recorded[last],
            epilogue=tuple(
                (step, step_args, tuple(sorted(step_kwargs.items())))
                for step, step_args, step_kwargs in steps
            ),
        )
        return True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for index, argument in enumerate(func._schema.arguments):
            if argument.alias_info is None or not argument.alias_info.is_write:
                continue
            value = args[index] if index < len(args) else kwargs.get(argument.name)
            for tensor in tree_leaves(value):
                if not isinstance(tensor, torch.Tensor) or isinstance(
                    tensor, LazyWeight
                ):
                    continue
                storage = tensor.untyped_storage()._cdata
                if storage in self._storages:
                    raise UnsupportedReshard(
                        f"live destination mutation during capture: {func}"
                    )
                self._forget_traces(tensor)
                overwritten = self._overwritten_copy(tensor)
                if overwritten is None:
                    # Replay starts every destination from zeros, so only a
                    # non-zero write can leave bytes replay does not reproduce.
                    if storage in self._meta_storages and not _zero_write(func, args):
                        self._unrecorded.setdefault(storage, []).append(
                            _view_of(tensor)
                        )
                elif not (index == 0 and self._record_epilogue(func, args, kwargs)):
                    raise UnsupportedReshard(
                        f"loader rewrites destination {overwritten!r} after "
                        f"copying a source into it: {func}"
                    )
        out = func(*args, **kwargs)
        if self._trace and not _writes(func):
            self._trace_step(func, args, kwargs, out)
        return out


def capture_weights(
    model: torch.nn.Module,
    weights: dict[str, Any],
    default_weight_loader: Callable | None = None,
    *,
    source_replay: bool = False,
) -> CaptureResult:
    """Capture per-param slice geometry by dry-running ``load_weights`` with the
    pre-built ``weights`` against ``model`` (ideally a disposable meta twin).

    Args:
        model: the engine model exposing ``load_weights([(name, tensor), ...])`` and
            per-param ``weight_loader`` hooks. Should be on ``meta`` so no real
            storage is touched.
        weights: ``{name: LazyWeight}`` from :func:`build_lazy_weights` (optionally
            passed through a source conversion first). ``name`` is what the loader
            routes on (HF-canonical); the lazy's ``_name`` is the original source.
        default_weight_loader: the framework's default loader, used to attribute
            copies for params that have no explicit ``weight_loader`` (e.g. vLLM's
            ``default_weight_loader``). Optional.

    Returns a ``CaptureResult`` (recorded copies + unsupported/unattributed).
    """
    recorder = _shared_recorder(weights)
    recorder.destinations = [*model.named_parameters(), *model.named_buffers()]
    previous_source_replay = recorder.source_replay
    recorder.source_replay = source_replay
    destination_layouts = (
        {
            name: (tuple(tensor.shape), tuple(tensor.stride()))
            for name, tensor in recorder.destinations
        }
        if source_replay
        else {}
    )
    if not source_replay:
        recorder.overlapping_owners = _overlapping_owners(recorder.destinations)
    saved = _install_stamps(model, recorder, default_weight_loader)
    unsupported: list[str] = []
    unsupported_reasons: dict[str, str] = {}
    try:
        # Large MoE loaders rebuild model-wide parameter/expert maps on entry.
        # LazyWeights carry their own source identity across one bulk invocation.
        with _CaptureMutationGuard(
            recorder.destinations, recorder, trace_epilogues=source_replay
        ) as guard:
            try:
                with recorder.attempt():
                    model.load_weights(sorted(weights.items()))
            except UnsupportedReshard as bulk_error:
                guard.reset()
                # The per-source retry can succeed for every source even when bulk
                # order trips a loader, which would otherwise leave `unsupported`
                # empty and hide a bulk-only incompatibility entirely.
                logger.warning(
                    "reshard capture: bulk load_weights raised, retrying per source; "
                    "bulk cause: %s",
                    bulk_error,
                )
                for name, tensor in sorted(weights.items()):
                    source = getattr(tensor, "_name", name)
                    try:
                        model.load_weights([(name, tensor)])
                    except UnsupportedReshard as exc:
                        unsupported.append(source)
                        unsupported_reasons[source] = str(exc)
    finally:
        _restore_stamps(saved)
        recorder.destinations.clear()
        recorder.overlapping_owners.clear()
        recorder.source_replay = previous_source_replay

    logger.info(
        "reshard capture: %d copies, %d unsupported source(s), %d unattributed copy_",
        len(recorder.copies),
        len(unsupported),
        recorder.unattributed,
    )
    # A whole model class can fail for one reason (every fused expert source, say),
    # so report the distinct causes with counts rather than a list of names. The
    # names alone say which tensors are missing but never why.
    for reason, count in summarize_unsupported(unsupported_reasons):
        logger.warning(
            "reshard capture: %d source(s) unsupported, cause: %s", count, reason
        )
    return CaptureResult(
        copies=recorder.copies,
        unsupported=unsupported,
        unattributed=recorder.unattributed,
        unsupported_reasons=unsupported_reasons,
        source_replay=source_replay,
        destination_layouts=destination_layouts,
    )


def capture_geometry(
    model: torch.nn.Module,
    manifest: list[tuple[str, Any, tuple]],
    default_weight_loader: Callable | None = None,
) -> CaptureResult:
    """Convenience wrapper: build lazies from ``manifest`` (source names already
    HF-canonical) and capture. See :func:`build_lazy_weights` + :func:`capture_weights`
    for the pre-built-weights entry point used when a source conversion runs first."""
    return capture_weights(
        model,
        build_lazy_weights(manifest),
        default_weight_loader=default_weight_loader,
    )


def convert_source_weights(
    convert_native_to_hf: Callable[[dict], dict] | None,
    manifest: list,
    *,
    device: torch.device | str = "meta",
) -> dict:
    """Build the source placeholders and map them into the HF-canonical layout the
    inference loader consumes.

    A trainer may publish its weights under its own native (non-HF) names/shapes.
    ``convert_native_to_hf`` is that trainer's native -> HF mapping (a
    ``dict[str, Tensor] -> dict[str, Tensor]``); we run it on ``LazyWeight``
    placeholders so it traces STRUCTURE ONLY (renames + view ops like expert
    unstack) - no weight data - and each output lazy keeps ``_name`` = its original
    published source and ``_ops`` = the native -> HF prefix. Returns
    ``{hf_name: LazyWeight}`` ready for :func:`capture_weights`.

    ``None`` means the trainer already publishes HF-canonical names (identity). A
    conversion output that is not a traced view of a source - a synthesized tensor (a plain tensor, not a ``LazyWeight``) - raises ``UnsupportedReshard`` rather than
    mis-mapping.
    """
    weights = build_lazy_weights(manifest, device=device)
    if convert_native_to_hf is None:
        return weights
    hf = convert_native_to_hf(weights)
    for name, value in hf.items():
        if not isinstance(value, LazyWeight):
            raise UnsupportedReshard(
                f"{name}: conversion produced a non-traced tensor "
                f"({type(value).__name__}); synthesized tensors are "
                "not yet supported"
            )
    return hf
