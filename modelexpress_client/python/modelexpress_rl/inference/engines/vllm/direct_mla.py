# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private MLA refresh primitive; not whole-model or engine admission.

An engine adapter must separately qualify the module implementation, consumer
state and quiescence before using this plan. The experimental GLM adapter uses
this primitive when MX_REFIT_GLM_DIRECT is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from modelexpress.refit.reshard.types import IncompleteRefit

_SCALES = ("_q_scale", "_k_scale", "_v_scale", "_prob_scale")
_MIRRORS = ("_k_scale_cpu", "_v_scale_cpu")
_FLOATS = ("_q_scale_float", "_k_scale_float", "_v_scale_float", "_prob_scale_float")
_AITER_MODES = (
    "is_aiter_triton_fp4_bmm_enabled",
    "is_aiter_triton_fp8_bmm_enabled",
)


def _require(condition, reason):
    if not condition:
        raise IncompleteRefit("unsupported direct MLA refresh: " + reason)


def _geometry(value):
    _require(isinstance(value, torch.Tensor), "missing tensor")
    _require(
        value.layout is torch.strided and value.device.type in ("cpu", "cuda"),
        "tensor layout/device",
    )
    return (
        id(value),
        value.data_ptr(),
        tuple(value.shape),
        tuple(value.stride()),
        value.storage_offset(),
        value.dtype,
        value.device,
        value.untyped_storage().data_ptr(),
        value.untyped_storage().nbytes(),
    )


def _same_view(left, right):
    return _geometry(left)[1:] == _geometry(right)[1:]


def _nonoverlapping(tensor):
    extent = 1
    for stride, size in sorted(zip(tensor.stride(), tensor.shape, strict=True)):
        if size > 1:
            if stride < extent:
                return False
            extent += (size - 1) * stride
    return True


def _inspect(module):
    _require(module.quant_config is None, "quantized attention")
    _require(module.dcp_q_replicate is False, "alternate MLA representation: DCP")
    # The pinned ROCm capability helpers return None on NVIDIA. Both None and
    # False select vLLM's ordinary unquantized PWAL branch; other values reject.
    for name in _AITER_MODES:
        value = getattr(module, name)
        _require(
            value is False or value is None,
            f"alternate MLA representation: {name}={value!r}",
        )
    dimensions = tuple(
        getattr(module, name)
        for name in ("num_heads", "qk_nope_head_dim", "v_head_dim", "kv_lora_rank")
    )
    _require(
        all(type(value) is int and value > 0 for value in dimensions), "dimensions"
    )
    heads, key, value, latent = dimensions
    source = module.kv_b_proj.weight
    _require(
        source.ndim == 2
        and tuple(source.shape) == (heads * (key + value), latent)
        and source.is_contiguous()
        and source.dtype in (torch.bfloat16, torch.float16),
        "source geometry",
    )
    projected = source.T.view(latent, heads, key + value)
    w_key, w_value = projected.split((key, value), dim=-1)
    views = (w_key.permute(1, 2, 0), w_value.transpose(0, 1))
    targets = (module.W_UK_T, module.W_UV)
    source_storage = source.untyped_storage().data_ptr()
    separate_storages = set()
    copies = []
    for target, derived in zip(targets, views, strict=True):
        _require(
            target.shape == derived.shape
            and target.dtype == source.dtype
            and target.device == source.device,
            "derived geometry",
        )
        exact_view = _same_view(target, derived)
        storage = target.untyped_storage().data_ptr()
        if not exact_view:
            _require(
                storage != source_storage and storage not in separate_storages,
                "unclassified derived storage overlap",
            )
            _require(_nonoverlapping(target), "derived strides")
            separate_storages.add(storage)
        copies.append(not exact_view)
    scales = tuple(getattr(module, name) for name in (*_SCALES, *_MIRRORS))
    occupied = {source_storage, *separate_storages}
    for index, scale in enumerate(scales):
        _require(
            scale.shape == ()
            and scale.dtype is torch.float32
            and scale.device.type in ("cpu", "cuda"),
            "scale geometry",
        )
        if index < len(_SCALES):
            _require(scale.device == source.device, "device scale placement")
        storage = (scale.device, scale.untyped_storage().data_ptr())
        _require(
            scale.untyped_storage().data_ptr() not in occupied
            and storage not in occupied,
            "scale storage overlap",
        )
        occupied.add(storage)
    _require(
        all(type(getattr(module, name)) is float for name in _FLOATS), "scalar mirrors"
    )
    tensors = (source, *targets, *scales)
    signature = (
        id(module.kv_b_proj),
        dimensions,
        tuple(_geometry(t) for t in tensors),
        tuple(copies),
    )
    return views, targets, scales, signature


@dataclass(frozen=True)
class _MlaRefreshPlan:
    module: object
    signature: tuple

    @torch.no_grad()
    def refresh(self) -> dict[str, int]:
        """Preserve every binding; caller drains device work before serving/reuse."""
        views, targets, scales, signature = _inspect(self.module)
        _require(signature == self.signature, "bindings/configuration changed")
        copies = signature[-1]
        for target, derived, needed in zip(targets, views, copies, strict=True):
            if needed:
                target.copy_(derived)
        for scale in scales:
            scale.fill_(1.0)
        for name in _FLOATS:
            setattr(self.module, name, 1.0)
        return {
            "mla_derived_copies": sum(copies),
            "mla_derived_shared_views": 2 - sum(copies),
            "mla_scale_resets": len(scales),
        }


def _prepare_mla_refresh(module) -> _MlaRefreshPlan:
    """Bind the known unquantized PWAL transition without installing any weights."""
    _, _, _, signature = _inspect(module)
    return _MlaRefreshPlan(module, signature)
