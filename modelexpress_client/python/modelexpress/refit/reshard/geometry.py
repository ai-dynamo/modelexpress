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
  * Per-source isolation: we bake one source tensor at a time, so unsupported
    geometry is attributed to the specific source before the receiver fails the
    update. It never produces an incorrect partial plan.
  * Allowlist of pure view/slice ops; anything else (arithmetic, .to/.float,
    bool-mask indexing) lands in ``__torch_dispatch__`` and raises
    ``UnsupportedReshard`` for that source, not wrong bytes.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from modelexpress.refit.reshard.types import (
    CaptureResult,
    OpChain,
    OpSpec,
    RecordedCopy,
    UnsupportedReshard,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CaptureResult",
    "LazyWeight",
    "OpChain",
    "OpSpec",
    "RecordedCopy",
    "UnsupportedReshard",
    "capture_geometry",
]

# Allowlist: pure view/slice/shape ops a weight loader may call. Maps
# ``torch.Tensor.fn`` -> the method name used on replay. Anything else reaches
# ``__torch_dispatch__`` and raises UnsupportedReshard.
_SUPPORTED_OPS: dict[Callable, str] = {
    torch.Tensor.narrow: "narrow",
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
}


def _freeze_kwargs(kwargs: dict[str, Any]) -> tuple:
    return tuple(sorted(kwargs.items()))


@dataclass
class _BakeRecorder:
    copies: list = field(default_factory=list)
    unattributed: int = 0
    # Set by the loader stamp to the destination param's full name so copy_ can
    # attribute the write.
    current: Any = None


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
    def __new__(cls, name, shape, dtype, device, ops=(), recorder=None):
        t = torch.Tensor._make_wrapper_subclass(
            cls, shape, dtype=dtype, device=device, requires_grad=False
        )
        t._name = name
        t._ops = tuple(ops)
        t._recorder = recorder
        return t

    def _meta(self) -> torch.Tensor:
        # Zero-storage meta tensor for post-op shape/dtype inference only.
        return torch.empty(self.shape, dtype=self.dtype, device="meta")

    def _child(self, new_shape, new_dtype, *new_ops) -> "LazyWeight":
        return LazyWeight(
            self._name,
            new_shape,
            new_dtype,
            self.device,
            self._ops + new_ops,
            self._recorder,
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
                if rec is not None and rec.current is not None:
                    rec.copies.append(
                        RecordedCopy(
                            src_name=src._name,
                            op_chain=src._ops,
                            param_name=rec.current,
                            dest_offset=dest.storage_offset(),
                            dest_shape=tuple(dest.shape),
                            dest_stride=tuple(dest.stride()),
                            dest_dtype=dest.dtype,
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
        meta = self_._meta()
        with torch._C.DisableTorchFunctionSubclass():
            meta_result = func(meta, *args, **kwargs)
        base_op = (op_name, tuple(args), _freeze_kwargs(kwargs))
        if isinstance(meta_result, torch.Tensor):
            return self_._child(meta_result.shape, meta_result.dtype, base_op)
        if isinstance(meta_result, (tuple, list)):
            # Multi-return (chunk/split): one child per output + a trailing
            # __getitem__ so replay can index back into the result.
            return tuple(
                self_._child(m.shape, m.dtype, base_op, ("__getitem__", (i,), ()))
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
    for name, param in model.named_parameters():
        original = getattr(param, "weight_loader", None)
        inner = original if original is not None else default_weight_loader
        if inner is None:
            continue

        def make_stamp(inner, name):
            @functools.wraps(inner)
            def stamp(p, *a, **kw):
                recorder.current = name
                try:
                    return inner(p, *a, **kw)
                finally:
                    recorder.current = None

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


def capture_geometry(
    model: torch.nn.Module,
    manifest: list[tuple[str, Any, tuple]],
    default_weight_loader: Callable | None = None,
) -> CaptureResult:
    """Capture per-param slice geometry by dry-running ``load_weights`` with
    ``LazyWeight`` placeholders against ``model`` (ideally a disposable meta twin).

    Args:
        model: the engine model exposing ``load_weights([(name, tensor)])`` and
            per-param ``weight_loader`` hooks. Should be on ``meta`` so no real
            storage is touched.
        manifest: ``(name, dtype, shape)`` for every full source tensor.
        default_weight_loader: the framework's default loader, used to attribute
            copies for params that have no explicit ``weight_loader`` (e.g. vLLM's
            ``default_weight_loader``). Optional.

    Returns a ``CaptureResult`` (recorded copies + unsupported/unattributed).
    """
    recorder = _BakeRecorder()
    saved = _install_stamps(model, recorder, default_weight_loader)
    unsupported: list[str] = []
    try:
        # One source at a time: a single unsupported loader is attributed only
        # to that tensor, never the whole bake. Fused params
        # (qkv/gate_up) still resolve - each source writes its own sub-region of
        # the persistent (meta) dest param across separate calls.
        for name, dtype, shape in manifest:
            lazy = LazyWeight(name, torch.Size(shape), dtype, "meta", recorder=recorder)
            try:
                model.load_weights([(name, lazy)])
            except UnsupportedReshard:
                unsupported.append(name)
    finally:
        _restore_stamps(saved)

    logger.info(
        "reshard capture: %d copies, %d unsupported source(s), %d unattributed copy_",
        len(recorder.copies),
        len(unsupported),
        recorder.unattributed,
    )
    return CaptureResult(
        copies=recorder.copies,
        unsupported=unsupported,
        unattributed=recorder.unattributed,
    )
