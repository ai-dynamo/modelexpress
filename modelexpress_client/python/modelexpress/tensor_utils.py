# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor collection, inspection, and registration utilities for MxModelLoader."""

from __future__ import annotations

import logging
from contextlib import contextmanager

import torch
import torch.nn as nn

logger = logging.getLogger("modelexpress.tensor_utils")


def safe_checksum(tensor: torch.Tensor) -> str:
    """Compute a fast fingerprint of tensor contents, staying on GPU when possible.

    Uses position-weighted mixing with Knuth's multiplicative constant so that
    permutations of the same bytes and compensating ±1 byte pairs produce
    different fingerprints — a plain byte sum collides on both.
    """
    try:
        t = tensor.detach().contiguous()
        if t.dim() == 0:
            t = t.unsqueeze(0)
        flat = t.view(torch.uint8)
        idx = torch.arange(flat.numel(), device=flat.device, dtype=torch.int64)
        weights = (idx * 2654435761 + 1) & 0xFFFFFFFF
        mixed = (flat.to(torch.int64) * weights) & 0xFFFFFFFF
        return format(mixed.sum().item() & 0xFFFFFFFF, "08x")
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
            logger.debug(
                "Captured bare CUDA tensor: %s.%s (shape=%s, dtype=%s)",
                type(self).__name__, name, list(value.shape), value.dtype,
            )
        else:
            original_setattr(self, name, value)

    nn.Module.__setattr__ = capturing_setattr
    try:
        yield
    finally:
        nn.Module.__setattr__ = original_setattr


def _find_hidden_cuda_tensors(
    obj: object, visited: set[int], depth: int = 0,
) -> list[tuple[str, torch.Tensor]]:
    """Recursively find CUDA tensors in a non-Module Python object graph.

    Known limitation: objects using ``__slots__`` are skipped because they
    lack ``__dict__``. No current vLLM quant class uses slots, but any
    upstream adoption would silently cause hidden tensors to be missed —
    which is exactly the bug class this function exists to fix.
    """
    if depth > 20 or id(obj) in visited:
        return []
    visited.add(id(obj))

    results: list[tuple[str, torch.Tensor]] = []

    if isinstance(obj, torch.Tensor) and obj.is_cuda and obj.numel() > 0:
        results.append(("t", obj))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            for path, tensor in _find_hidden_cuda_tensors(item, visited, depth + 1):
                results.append((f"{i}_{path}", tensor))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            for path, tensor in _find_hidden_cuda_tensors(v, visited, depth + 1):
                results.append((f"{k}_{path}", tensor))
    elif hasattr(obj, "__dict__") and not isinstance(obj, (type, nn.Module)):
        for attr_name, attr_val in vars(obj).items():
            if attr_name.startswith("__"):
                continue
            for path, tensor in _find_hidden_cuda_tensors(attr_val, visited, depth + 1):
                results.append((f"{attr_name}_{path}", tensor))

    return results


def adopt_hidden_tensors(model: nn.Module) -> int:
    """Register hidden CUDA tensors as module buffers for RDMA transfer.

    process_weights_after_loading may create CUDA tensors stored on plain
    Python objects attached to modules (e.g. quant configs, kernel objects,
    dataclasses) rather than as nn.Module parameters or buffers. These are
    invisible to named_parameters()/named_buffers() and thus missing from
    the RDMA manifest, causing incorrect inference on the target.

    This function scans each module's non-Module attributes recursively for
    any CUDA tensors not already registered, and adopts them as non-persistent
    buffers so they appear in the manifest and get transferred.
    """
    import time
    start = time.perf_counter()

    existing_ptrs: set[int] = set()
    for _, p in model.named_parameters():
        existing_ptrs.add(p.data_ptr())
    for _, b in model.named_buffers():
        existing_ptrs.add(b.data_ptr())

    adopted = 0
    for _module_name, module in model.named_modules():
        for attr_name in list(vars(module)):
            attr_val = getattr(module, attr_name, None)
            if attr_val is None:
                continue
            if isinstance(attr_val, (torch.Tensor, nn.Parameter, nn.Module)):
                continue

            tensors = _find_hidden_cuda_tensors(attr_val, visited=set())
            for tensor_path, tensor in tensors:
                if tensor.data_ptr() in existing_ptrs:
                    continue
                safe_path = (
                    tensor_path.replace(".", "__dot__")
                    .replace("[", "")
                    .replace("]", "")
                )
                buf_name = f"_mx_{attr_name}_{safe_path}"
                if hasattr(module, buf_name):
                    suffix = 0
                    while hasattr(module, f"{buf_name}_{suffix}"):
                        suffix += 1
                    buf_name = f"{buf_name}_{suffix}"
                module.register_buffer(buf_name, tensor, persistent=False)
                existing_ptrs.add(tensor.data_ptr())
                adopted += 1
                logger.debug(
                    "Adopted hidden tensor: %s.%s "
                    "(shape=%s, dtype=%s, from %s.%s)",
                    _module_name, buf_name,
                    list(tensor.shape), tensor.dtype,
                    type(attr_val).__name__, tensor_path,
                )

    elapsed = time.perf_counter() - start
    if adopted:
        logger.info(
            f"Adopted {adopted} hidden CUDA tensors as module buffers "
            f"in {elapsed:.3f}s"
        )
    else:
        logger.debug(f"No hidden CUDA tensors found ({elapsed:.3f}s)")
    return adopted


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
    """Log a summary of tensor count, total size, and optionally per-tensor checksums.

    At DEBUG level, logs a checksum for every tensor. Expensive (GPU reduction
    per tensor) — enable via MODEL_EXPRESS_LOG_LEVEL=DEBUG.
    """
    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    logger.info(
        f"[Worker {global_rank}] {label}: {len(tensors)} tensors ({total_size / 1e9:.2f} GB)"
    )

    if logger.isEnabledFor(logging.DEBUG):
        for name, t in tensors.items():
            checksum = safe_checksum(t)
            logger.debug(
                f"[Worker {global_rank}] [CHECKSUM] {label} | {name} | "
                f"shape={list(t.shape)} dtype={t.dtype} | {checksum}"
            )
