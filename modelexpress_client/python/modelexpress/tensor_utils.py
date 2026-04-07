# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor collection, inspection, and registration utilities for MxModelLoader."""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager

import torch
import torch.nn as nn

logger = logging.getLogger("modelexpress.tensor_utils")


def safe_checksum(tensor: torch.Tensor) -> str:
    """Compute MD5 checksum of tensor, handling bfloat16 which numpy doesn't support."""
    try:
        t = tensor.cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return hashlib.md5(t.numpy().tobytes()).hexdigest()[:8]
    except Exception as e:
        return f"err:{e}"


@contextmanager
def capture_tensor_attrs():
    """Intercept bare CUDA tensor assignments during process_weights_after_loading.

    vLLM's post-processing (quant methods, attention backends) may create
    tensor attributes via plain setattr (e.g. self.W_UV = tensor) instead
    of register_buffer. These are invisible to named_parameters/named_buffers
    and would be missing from the RDMA manifest.

    This context manager patches nn.Module.__setattr__ to auto-promote such
    tensors to non-persistent buffers, making them discoverable by
    named_buffers() and thus included in the manifest.
    """
    original_setattr = nn.Module.__setattr__

    def capturing_setattr(self, name, value):
        if (isinstance(value, torch.Tensor)
                and not isinstance(value, nn.Parameter)
                and value.is_cuda
                and name not in self._parameters
                and name not in self._buffers
                and name not in self._modules):
            if hasattr(self, name):
                try:
                    delattr(self, name)
                except AttributeError:
                    pass
            self.register_buffer(name, value, persistent=False)
        else:
            original_setattr(self, name, value)

    nn.Module.__setattr__ = capturing_setattr
    try:
        yield
    finally:
        nn.Module.__setattr__ = original_setattr


def iter_module_tensors(
    module: nn.Module,
) -> list[tuple[str, torch.Tensor, str]]:
    """Iterate over all CUDA tensors in a module tree.

    Uses named_parameters() and named_buffers() to discover tensors.
    When used with capture_tensor_attrs() wrapping process_weights_after_loading,
    bare tensor attributes (e.g. W_UV, W_UK_T) are auto-promoted to
    non-persistent buffers and thus included in named_buffers().

    Returns:
        List of (qualified_name, tensor, tensor_type) tuples for each CUDA tensor.
    """
    results: list[tuple[str, torch.Tensor, str]] = []

    for name, param in module.named_parameters():
        if param.is_cuda:
            results.append((name, param, "parameter"))

    for name, buf in module.named_buffers():
        if buf.is_cuda:
            results.append((name, buf, "buffer"))

    return results


def storage_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return a flat contiguous uint8 view of a tensor's underlying storage.

    For RDMA we transfer raw storage bytes. Both source and target run
    the same post-processing on the same model architecture, so they
    produce identical storage layouts (same sizes, strides, offsets).
    Transferring the full storage block ensures all views into it
    (including partial views like MLA's W_UV and W_UK_T which share
    storage from a dequantized intermediate) get correct data.

    Multiple tensors sharing the same storage are deduplicated by
    data_ptr() in the caller, so only one transfer per storage block.
    """
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        tensor.untyped_storage()
    )


def collect_module_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect all CUDA tensors from a module tree into a flat dict.

    Uses iter_module_tensors (named_parameters + named_buffers) to find
    tensors, then returns them as a name -> tensor mapping suitable for
    NIXL registration. Bare tensor attributes created during
    process_weights_after_loading are captured as non-persistent buffers
    by the capture_tensor_attrs context manager.

    Contiguous tensors are registered directly. Non-contiguous tensors
    (DeepGemm TMA-aligned FP8 scales, MLA dequantized projections)
    are registered as a flat byte view of their full underlying storage,
    named as ``name.__storage``. This transfers the raw bytes correctly
    because both source and target have identical storage layouts.
    Multiple views into the same storage (e.g. W_UV and W_UK_T sharing
    a dequantized intermediate) are deduplicated by data_ptr so the
    storage is transferred only once.
    """
    tensors: dict[str, torch.Tensor] = {}
    seen_ptrs: set[int] = set()
    storage_view_count = 0
    skipped_duplicate = 0
    for name, tensor, _tensor_type in iter_module_tensors(model):
        t = tensor.data if hasattr(tensor, "data") else tensor

        if t.is_contiguous():
            ptr = t.data_ptr()
            if ptr in seen_ptrs:
                logger.debug(f"Skipping duplicate tensor '{name}' (same data_ptr)")
                skipped_duplicate += 1
                continue
            seen_ptrs.add(ptr)
            tensors[name] = t
        else:
            sv = storage_view(t)
            ptr = sv.data_ptr()
            if ptr in seen_ptrs:
                skipped_duplicate += 1
                continue
            seen_ptrs.add(ptr)
            tensors[f"{name}.__storage"] = sv
            storage_view_count += 1

    if storage_view_count:
        logger.info(
            f"Registered {storage_view_count} non-contiguous tensors "
            f"via storage-level byte transfer"
        )
    if skipped_duplicate:
        logger.info(f"Skipped {skipped_duplicate} duplicate tensors (tied weights)")
    return tensors


def log_tensor_summary(
    tensors: dict[str, torch.Tensor], global_rank: int, label: str
) -> None:
    """Log a summary of tensor count, size, scale_inv count, and sample checksums."""
    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    logger.info(
        f"[Worker {global_rank}] {label}: {len(tensors)} tensors ({total_size / 1e9:.2f} GB)"
    )

    if logger.isEnabledFor(logging.DEBUG):
        tensor_names = list(tensors.keys())
        logger.debug(f"[Worker {global_rank}] First 5 tensor names: {tensor_names[:5]}")

        for name in tensor_names[:3]:
            t = tensors[name]
            checksum = safe_checksum(t)
            logger.debug(
                f"[Worker {global_rank}] Sample tensor '{name}': "
                f"shape={t.shape}, dtype={t.dtype}, checksum={checksum}"
            )
