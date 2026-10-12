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
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from modelexpress.refit.reshard.types import (
    CaptureResult,
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


def _freeze_kwargs(kwargs: dict[str, Any]) -> tuple:
    return tuple((name, tuple(value) if isinstance(value, list) else value)
                 for name, value in sorted(kwargs.items()))


def _canonical_destination(tensor, owner):
    """Express a physical destination view in contiguous owner coordinates."""
    from modelexpress.refit.reshard.slice_plan import _row_major_strides

    offset = tensor.storage_offset() - owner.storage_offset()
    span = 1
    dimensions = sorted(
        (stride, size, axis)
        for axis, (size, stride) in enumerate(zip(owner.shape, owner.stride()))
        if size > 1
    )
    for stride, size, _ in dimensions:
        if stride < span:
            raise UnsupportedReshard("overlapping destination layout is unsupported")
        span += (size - 1) * stride
    extent = sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    if offset < 0 or (tensor.numel() and offset + extent >= span):
        raise UnsupportedReshard("destination view exceeds registered owner bounds")
    if owner.is_contiguous():
        return offset, tuple(tensor.stride())

    coordinates = [0] * owner.ndim
    remainder = offset
    for stride, size, axis in reversed(dimensions):
        coordinates[axis], remainder = divmod(remainder, stride)
        if coordinates[axis] >= size:
            raise UnsupportedReshard("destination view exceeds registered owner bounds")
    if remainder:
        raise UnsupportedReshard("destination view addresses holes in owner storage")
    canonical = _row_major_strides(owner.shape)
    normalized = []
    limits = list(coordinates)
    for size, stride in zip(tensor.shape, tensor.stride()):
        if size <= 1:
            normalized.append(1)
            continue
        candidates = [
            (axis, stride // step)
            for step, owner_size, axis in reversed(dimensions)
            if stride > 0
            and stride % step == 0
            and limits[axis] + (size - 1) * (stride // step) < owner_size
        ]
        if len(candidates) != 1:
            raise UnsupportedReshard(
                "destination view cannot map to canonical owner coordinates"
            )
        axis, step = candidates[0]
        limits[axis] += (size - 1) * step
        normalized.append(canonical[axis] * step)
    return sum(a * b for a, b in zip(coordinates, canonical)), tuple(normalized)


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


def _destination_spans(tensor, start):
    """Yield ordered physical byte runs, retaining holes in strided owners."""
    yield from _strided_spans(
        tensor.shape, tensor.stride(), start, tensor.element_size()
    )


def _owners_overlap(left, right):
    left_runs = _destination_spans(left[2], left[0])
    right_runs = _destination_spans(right[2], right[0])
    a, b = next(left_runs, None), next(right_runs, None)
    while a is not None and b is not None:
        if a[0] < b[1] and b[0] < a[1]:
            return True
        if a[1] <= b[0]:
            a = next(left_runs, None)
        else:
            b = next(right_runs, None)
    return False


def _overlapping_owners(destinations):
    groups = {}
    for name, tensor in destinations:
        if not tensor.numel():
            continue
        storage = tensor.untyped_storage()
        key = (tensor.device, storage._cdata if tensor.device.type == "meta" else None)
        start = tensor.storage_offset() * tensor.element_size()
        if tensor.device.type != "meta":
            start += storage.data_ptr()
        end = start + tensor.element_size() * (
            1
            + sum(
                (size - 1) * stride
                for size, stride in zip(tensor.shape, tensor.stride())
            )
        )
        groups.setdefault(key, []).append((start, end, tensor, name))
    overlaps = {}
    for group in groups.values():
        active = []
        for current in sorted(group, key=lambda entry: entry[0]):
            active = [entry for entry in active if entry[1] > current[0]]
            for previous in active:
                if previous[2] is not current[2] and _owners_overlap(previous, current):
                    overlaps[current[3]] = previous[3]
                    overlaps[previous[3]] = current[3]
            active.append(current)
    return overlaps


def overlapping_destination_groups(destinations):
    """Connected physical overlap components, excluding disjoint shared slabs."""
    by_device = {}
    parent = {}

    def root(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for name, tensor in destinations:
        if tensor.is_meta or not tensor.numel():
            continue
        parent[name] = name
        start = tensor.data_ptr()
        end = start + tensor.element_size() * (
            1
            + sum(
                (size - 1) * stride
                for size, stride in zip(tensor.shape, tensor.stride())
            )
        )
        by_device.setdefault(tensor.device, []).append((start, end, tensor, name))
    for entries in by_device.values():
        active = []
        for current in sorted(entries, key=lambda entry: entry[0]):
            active = [entry for entry in active if entry[1] > current[0]]
            for previous in active:
                if _owners_overlap(previous, current):
                    parent[root(current[3])] = root(previous[3])
            active.append(current)
    groups = {}
    for name in parent:
        groups.setdefault(root(name), []).append(name)
    return tuple(tuple(names) for names in groups.values() if len(names) > 1)


@dataclass
class _BakeRecorder:
    copies: list = field(default_factory=list)
    unattributed: int = 0
    # Set by the loader stamp to the destination param's full name so copy_ can
    # attribute the write.
    current: Any = None
    destinations: list = field(default_factory=list)
    overlapping_owners: dict = field(default_factory=dict)
    weights: list = field(default_factory=list)
    source_replay: bool = False

    def destination(self, name, tensor, owner):
        if name in self.overlapping_owners:
            raise UnsupportedReshard(
                f"overlapping registered destination {name!r} requires ordered source replay"
            )
        physical_offset = tensor.storage_offset() - owner.storage_offset()
        physical_stride = tuple(tensor.stride())
        if self.source_replay:
            _validate_physical_destination(
                tuple(tensor.shape),
                physical_stride,
                physical_offset,
                tuple(owner.shape),
                tuple(owner.stride()),
            )
            offset, stride = physical_offset, physical_stride
        else:
            offset, stride = _canonical_destination(tensor, owner)
        return (
            name,
            offset,
            stride,
            physical_offset,
            physical_stride,
        )

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
            name = matches[0][0]
            if name in self.overlapping_owners:
                return self.destination(name, tensor, dict(candidates)[name])
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
        return LazyWeight(
            self._name,
            meta.shape,
            meta.dtype,
            self.device if device is None else device,
            self._ops + new_ops,
            self._recorder,
            self._aliases if aliases else None,
            meta,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # copy_: the recording sink. Note where the slice lands; do NOT copy.
        if func is torch.Tensor.copy_:
            dest = args[0]
            src = args[1] if len(args) > 1 else kwargs.get("src")
            if isinstance(src, cls):
                rec = src._recorder
                destination = rec.resolve(dest) if rec is not None else None
                if destination is not None:
                    rec.copies.append(
                        RecordedCopy(
                            src_name=src._name,
                            op_chain=src._ops,
                            param_name=destination[0],
                            dest_offset=destination[1],
                            dest_shape=tuple(dest.shape),
                            dest_stride=destination[2],
                            dest_dtype=dest.dtype,
                            owner_offset=destination[3],
                            owner_stride=destination[4],
                        )
                    )
                elif rec is not None:
                    rec.unattributed += 1
                # Record-only: return dest so the loader proceeds as if the copy
                # happened, with no device/meta coupling (dest may be meta/cpu/cuda).
                return dest

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
        if op_name == "__getitem__":
            index = args[0]
            indices = index if isinstance(index, tuple) else (index,)
            if any(not isinstance(item, (int, slice, type(Ellipsis), type(None)))
                   for item in indices):
                raise UnsupportedReshard("advanced indexing is unsupported")
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
        base_op = (op_name, tuple(tuple(arg) if isinstance(arg, list) else arg for arg in args), _freeze_kwargs(kwargs))
        if op_name == "zero_":
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
        kwargs = kwargs or {}
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, cls):
                raise UnsupportedReshard(
                    f"unsupported op {func} on lazy {a._name!r} (chain={a._ops}); "
                    f"supported: {sorted(_SUPPORTED_OPS.values())} + copy_."
                )
        return func(*args, **kwargs)


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


class _CaptureMutationGuard(TorchDispatchMode):
    """Reject writes to live destinations that bypass recorded source copies."""

    def __init__(self, destinations):
        super().__init__()
        self.storages = {
            tensor.untyped_storage()._cdata
            for _, tensor in destinations
            if not tensor.is_meta
        }

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func._schema.is_mutable:
            for index, argument in enumerate(func._schema.arguments):
                if argument.alias_info is None or not argument.alias_info.is_write:
                    continue
                value = args[index] if index < len(args) else kwargs.get(argument.name)
                tensors = value if isinstance(value, (tuple, list)) else (value,)
                if any(
                    isinstance(tensor, torch.Tensor)
                    and not isinstance(tensor, LazyWeight)
                    and tensor.untyped_storage()._cdata in self.storages
                    for tensor in tensors
                ):
                    raise UnsupportedReshard(
                        f"live destination mutation {func} cannot be captured safely"
                    )
        return func(*args, **kwargs)


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
        try:
            with _CaptureMutationGuard(recorder.destinations), recorder.attempt():
                model.load_weights(sorted(weights.items()))
        except UnsupportedReshard as bulk_error:
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
                    with _CaptureMutationGuard(recorder.destinations):
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
