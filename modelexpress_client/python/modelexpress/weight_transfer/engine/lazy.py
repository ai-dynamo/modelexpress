# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LazyWeight tensor and BakeRecorder for op-chain capture.

This module is the engine-side half of the bake pass.  It does NOT contain
routing or plan logic -- those live in planner/.

Usage::

    recorder = BakeRecorder()
    with recorder:
        model.load_weights(adapter.iter_lazy_weights(table))
    regions = resolver.resolve_copies(recorder.copies, tensor_shapes, dtypes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from ..protocol.ops import OpChain, OpSpec, SUPPORTED_OPS
from ..protocol.types import TrainerTable

logger = logging.getLogger("modelexpress.weight_transfer.engine.lazy")

_BAKE_STACK: list[BakeRecorder] = []


@dataclass
class RecordedCopy:
    """One copy_() call captured during the bake pass."""

    src_name: str
    op_chain: OpChain
    dst_addr: int
    dst_shape: tuple[int, ...]
    dst_stride: tuple[int, ...]
    dst_dtype: torch.dtype
    dst_device_id: int


class BakeRecorder:
    """Context manager that collects RecordedCopy objects during a dry-run."""

    def __init__(self) -> None:
        self.copies: list[RecordedCopy] = []

    def __enter__(self) -> BakeRecorder:
        _BAKE_STACK.append(self)
        return self

    def __exit__(self, *_: Any) -> None:
        _BAKE_STACK.pop()

    def record(self, copy: RecordedCopy) -> None:
        self.copies.append(copy)


class LazyWeight(torch.Tensor):
    """Zero-storage tensor placeholder that records weight loader ops.

    Each op in SUPPORTED_OPS is intercepted and appended to the chain,
    producing a new LazyWeight with the updated shape.  copy_() is the
    recording sink: it captures the op chain to the active BakeRecorder
    and performs no data movement.
    """

    @staticmethod
    def __new__(
        cls,
        name: str,
        shape: torch.Size,
        dtype: torch.dtype,
        op_chain: OpChain = (),
    ) -> LazyWeight:
        tensor = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, shape, dtype=dtype, device="meta", requires_grad=False,
        )
        tensor._lazy_name = name
        tensor._lazy_chain = op_chain
        return tensor

    def __init__(
        self, name: str, shape: torch.Size, dtype: torch.dtype, op_chain: OpChain = ()
    ) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"LazyWeight({self._lazy_name!r}, shape={tuple(self.shape)}, "
            f"dtype={self.dtype}, chain_len={len(self._lazy_chain)})"
        )

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Any,
        types: Any,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        """Required by PyTorch ≥2.9 for _make_wrapper_subclass classes.

        Our primary interception lives in __torch_function__ (Python-level
        calls like narrow/view/copy_).  __torch_dispatch__ is the ATen-level
        fallback; we handle copy_ here too so both paths work, and for all
        other ops we proxy through meta tensors to get the right output shape.
        """
        kwargs = kwargs or {}
        lazy: LazyWeight | None = next(
            (a for a in args if isinstance(a, LazyWeight)), None
        )
        if lazy is None:
            return func(*args, **kwargs)

        func_name = _func_name(func)

        # copy_ sink at ATen level (aten.copy_.default)
        if "copy_" in func_name:
            dst = args[0] if args else None
            if (
                _BAKE_STACK
                and dst is not None
                and isinstance(dst, torch.Tensor)
                and not isinstance(dst, LazyWeight)
                and dst.device.type != "meta"
            ):
                _BAKE_STACK[-1].record(RecordedCopy(
                    src_name=lazy._lazy_name,
                    op_chain=lazy._lazy_chain,
                    dst_addr=dst.data_ptr(),
                    dst_shape=tuple(dst.shape),
                    dst_stride=tuple(dst.stride()),
                    dst_dtype=dst.dtype,
                    dst_device_id=dst.device.index or 0,
                ))
            return dst

        # For shape-transforming ops: proxy through plain meta tensors to get
        # the correct output shape, then wrap back as LazyWeight.
        try:
            meta_args = tuple(
                torch.empty(a.shape, dtype=a.dtype, device="meta")
                if isinstance(a, LazyWeight) else a
                for a in args
            )
            out = func(*meta_args, **kwargs)
            if isinstance(out, torch.Tensor):
                new_op = OpSpec(name=func_name, args=args[1:], kwargs=kwargs)
                return LazyWeight(
                    lazy._lazy_name, out.shape, out.dtype,
                    lazy._lazy_chain + (new_op,),
                )
            return out
        except Exception:
            return torch.empty(lazy.shape, dtype=lazy.dtype, device="meta")

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: Any,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        lazy: LazyWeight | None = next(
            (a for a in args if isinstance(a, LazyWeight)), None
        )
        if lazy is None:
            return super().__torch_function__(func, types, args, kwargs)

        func_name = _func_name(func)

        # Recording sink
        if func_name in ("copy_", "__copy__"):
            dst = args[0] if args else None
            if (
                _BAKE_STACK
                and dst is not None
                and isinstance(dst, torch.Tensor)
                and not isinstance(dst, LazyWeight)
                and dst.device.type != "meta"
            ):
                _BAKE_STACK[-1].record(RecordedCopy(
                    src_name=lazy._lazy_name,
                    op_chain=lazy._lazy_chain,
                    dst_addr=dst.data_ptr(),
                    dst_shape=tuple(dst.shape),
                    dst_stride=tuple(dst.stride()),
                    dst_dtype=dst.dtype,
                    dst_device_id=dst.device.index or 0,
                ))
            return dst

        # Op interception
        if func_name in SUPPORTED_OPS:
            new_op = OpSpec(name=func_name, args=args[1:], kwargs=kwargs)
            new_chain = lazy._lazy_chain + (new_op,)
            meta = torch.empty(lazy.shape, dtype=lazy.dtype, device="meta")
            fn = getattr(torch.Tensor, func_name, None) or getattr(torch, func_name, None)
            if fn is None:
                return super().__torch_function__(func, types, args, kwargs)
            try:
                out = fn(meta, *args[1:], **kwargs)
            except Exception:
                return super().__torch_function__(func, types, args, kwargs)

            if isinstance(out, torch.Tensor):
                return LazyWeight(lazy._lazy_name, out.shape, out.dtype, new_chain)
            if isinstance(out, (list, tuple)):
                pieces = [
                    LazyWeight(lazy._lazy_name, p.shape, p.dtype,
                               new_chain + (OpSpec("__getitem__", (i,), {}),))
                    if isinstance(p, torch.Tensor) else p
                    for i, p in enumerate(out)
                ]
                return type(out)(pieces)

        return super().__torch_function__(func, types, args, kwargs)


def _func_name(func: Any) -> str:
    name = getattr(func, "__name__", None)
    if name:
        return name
    overload = getattr(func, "_overloadpacket", None)
    if overload is not None:
        return getattr(overload, "__name__", str(overload))
    return str(func)


def bake_model(
    model: Any,
    weight_iter: Iterator[tuple[str, LazyWeight]],
) -> list[RecordedCopy]:
    """Run model.load_weights() with LazyWeights and return recorded copies."""
    recorder = BakeRecorder()
    with recorder:
        if hasattr(model, "load_weights"):
            model.load_weights(weight_iter)
        else:
            logger.warning(
                "Model %s has no load_weights(); bake pass may be incomplete",
                type(model).__name__,
            )
    logger.info("Bake pass recorded %d copies", len(recorder.copies))
    return recorder.copies
