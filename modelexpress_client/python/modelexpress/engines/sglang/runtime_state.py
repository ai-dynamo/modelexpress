# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile SGLang cold-load runtime state with the donor's tensor catalog."""

from __future__ import annotations

import math

import torch

from ...types import ManifestMismatchError, TensorDescriptor


def _block_mxfp8_method(layer):
    method = getattr(layer, "quant_method", None)
    return method if getattr(method, "block_fp8_as_mxfp8", False) else None


@torch.no_grad()
def prepare_runtime_tensors(model: torch.nn.Module, source_tensors: list[TensorDescriptor]) -> None:
    """Prepare supported donor-dependent buffers before transport registration.

    For SGLang's block-FP8 CuTeDSL/CUTLASS path, the swizzled tensor's presence
    is the donor's MXFP8 decision. Its layout is defined by SGLang, not by an
    untrusted size from the manifest. Other value-dependent layouts fail the
    exact catalog check or are rejected here rather than guessed. Older manifests
    without a derived scale dtype are rejected so the loader falls back natively.
    """
    source = {tensor.name: tensor for tensor in source_tensors}
    if len(source) != len(source_tensors):
        raise ManifestMismatchError("Duplicate tensor names in SGLang source manifest")
    for name, layer in model.named_modules():
        method = _block_mxfp8_method(layer)
        if method is None:
            continue
        prefix = f"{name}." if name else ""
        if prefix + "weight" not in source:
            # A speculative draft may share layers excluded from its catalog.
            continue
        backend = method.mxfp8_dense_backend
        if not (backend.is_flashinfer_cutedsl() or backend.is_flashinfer_cutlass()):
            raise ManifestMismatchError(f"Unsupported SGLang block-FP8 runtime layout at {name!r}")
        derived = source.get(prefix + "weight_scale_inv_swizzled")
        if derived is not None:
            n, k = layer.weight.shape
            if getattr(layer, "keep_plain_weight_layout", False) or k % 32:
                raise ManifestMismatchError(f"Incompatible SGLang MXFP8 layer {name!r}")
            # Only the scale-layout operation: no floating scale priming and no
            # second pass through weight reshuffling or requantization.
            method._process_mxfp8_linear_weight_scale(
                layer,
                scale_u8=torch.zeros((n, k // 32), dtype=torch.uint8, device=layer.weight.device),
            )
            target = layer.weight_scale_inv_swizzled
            target_dtype = str(target.dtype).removeprefix("torch.")
            if (
                derived.dtype.removeprefix("torch.") != target_dtype
                or derived.size != target.numel() * target.element_size()
            ):
                raise ManifestMismatchError(
                    f"Incompatible SGLang MXFP8 scale descriptor at {name!r}"
                )
        elif getattr(layer, "weight_scale_inv_swizzled", None) is not None:
            raise ManifestMismatchError(f"Unexpected SGLang MXFP8 receiver buffer at {name!r}")
        layer.block_fp8_mxfp8_ready = False
        layer._mx_block_fp8_manifest_prepared = True


@torch.no_grad()
def finalize_runtime_state(model: torch.nn.Module) -> None:
    """Validate dispatch and refresh host mirrors without reprocessing weights."""
    for name, layer in model.named_modules():
        if getattr(layer, "_mx_block_fp8_manifest_prepared", False):
            from sglang.srt.layers.quantization.fp8_utils import (
                block_fp8_scale_to_mxfp8_e8m0,
            )

            method = _block_mxfp8_method(layer)
            ready = False
            if not getattr(layer, "keep_plain_weight_layout", False):
                try:
                    block_fp8_scale_to_mxfp8_e8m0(
                        layer.weight_scale_inv,
                        tuple(layer.weight.shape),
                        method.weight_block_size,
                    )
                    ready = True
                except ValueError:
                    pass
            received_derived = getattr(layer, "weight_scale_inv_swizzled", None) is not None
            if ready != received_derived:
                raise ManifestMismatchError(
                    f"SGLang MXFP8 scales disagree with the donor catalog at {name!r}"
                )
            layer.block_fp8_mxfp8_ready = ready
            del layer._mx_block_fp8_manifest_prepared

        # SGLang already normalized these device scales during native loading.
        # Re-running BaseKVCacheMethod would normalize FNUZ a second time.
        for kind in ("k", "v"):
            host_name = f"{kind}_scale_float"
            if not hasattr(layer, host_name):
                continue
            scale = getattr(layer, f"{kind}_scale", None)
            if scale is None:
                continue
            if not isinstance(scale, torch.Tensor) or scale.numel() != 1:
                raise ManifestMismatchError(f"Invalid SGLang KV scale at {name}.{kind}")
            value = float(scale.item())
            if not math.isfinite(value) or value <= 0:
                raise ManifestMismatchError(f"Invalid SGLang KV scale at {name}.{kind}")
            setattr(layer, host_name, value)
