# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor collection, inspection, and registration utilities for MxModelLoader."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

import torch
import torch.nn as nn

from .quantization_providers import (
    SOURCE_MANIFEST_TENSOR_NAMES_ATTR,
    get_quantization_provider,
)
from .quantization_providers.humming import (
    HUMMING_RUNTIME_TENSORS_ATTR,
    capture_runtime_tensors as _provider_capture_humming_runtime_tensors,
    capture_runtime_tensors_from_model as _provider_capture_humming_runtime_tensors_from_model,
)

logger = logging.getLogger("modelexpress.tensor_utils")


def _debug_tensor_patterns() -> list[str]:
    raw = os.environ.get("MX_DEBUG_TENSOR_NAME", "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def debug_tensor_enabled(name: str) -> bool:
    """Return whether verbose manifest logging is enabled for this tensor."""
    patterns = _debug_tensor_patterns()
    return any(pattern in name for pattern in patterns)


def tensor_debug_summary(name: str, tensor: torch.Tensor) -> str:
    """Build a compact layout summary for manifest mismatch diagnostics."""
    try:
        storage_nbytes = tensor.untyped_storage().nbytes()
    except Exception as e:
        storage_nbytes = f"err:{e}"
    try:
        stride = list(tensor.stride())
    except Exception as e:
        stride = f"err:{e}"
    try:
        storage_offset = tensor.storage_offset()
    except Exception as e:
        storage_offset = f"err:{e}"
    return (
        f"name={name!r} shape={list(tensor.shape)} dtype={tensor.dtype} "
        f"numel={tensor.numel()} element_size={tensor.element_size()} "
        f"nbytes={tensor.numel() * tensor.element_size()} "
        f"is_contiguous={tensor.is_contiguous()} stride={stride} "
        f"storage_offset={storage_offset} storage_nbytes={storage_nbytes} "
        f"data_ptr=0x{tensor.data_ptr():x}"
    )


def tensor_descriptor_layout(tensor: torch.Tensor, layout_kind: str = "") -> dict[str, object]:
    """Return optional manifest fields that describe tensor layout."""
    try:
        storage_nbytes = int(tensor.untyped_storage().nbytes())
    except Exception:
        storage_nbytes = int(tensor.numel() * tensor.element_size())
    layout = {
        "shape": [int(dim) for dim in tensor.shape],
        "stride": [int(dim) for dim in tensor.stride()],
        "storage_offset": int(tensor.storage_offset()),
        "storage_nbytes": storage_nbytes,
        "layout_kind": layout_kind or ("contiguous" if tensor.is_contiguous() else "view"),
    }
    layout.update(tensor_original_layout(tensor))
    layout.update(tensor_runtime_metadata(tensor))
    return layout


def tensor_original_layout(tensor: torch.Tensor) -> dict[str, object]:
    """Best-effort original pre-quantization layout metadata.

    Quantization implementations may attach framework-specific attributes to
    tensors. ModelExpress treats these as optional diagnostics/compatibility
    hints; empty/zero means unknown and callers must still verify the final
    registered layout before RDMA.
    """
    original_shape = _first_attr(tensor, (
        "mx_original_shape",
        "original_shape",
        "_original_shape",
        "input_shape",
        "logical_shape",
    ))
    original_dtype = _first_attr(tensor, (
        "mx_original_dtype",
        "original_dtype",
        "_original_dtype",
        "input_dtype",
        "logical_dtype",
    ))
    original_nbytes = _first_attr(tensor, (
        "mx_original_nbytes",
        "original_nbytes",
        "_original_nbytes",
        "input_nbytes",
        "logical_nbytes",
    ))

    shape_list: list[int] = []
    if original_shape is not None:
        try:
            shape_list = [int(dim) for dim in original_shape]
        except TypeError:
            shape_list = []

    dtype_str = "" if original_dtype is None else str(original_dtype)
    try:
        nbytes = int(original_nbytes or 0)
    except (TypeError, ValueError):
        nbytes = 0

    return {
        "original_shape": shape_list,
        "original_dtype": dtype_str,
        "original_nbytes": nbytes,
    }


def tensor_runtime_metadata(tensor: torch.Tensor) -> dict[str, object]:
    """Return optional runtime/quantization metadata attached during discovery."""
    return {
        "tensor_kind": str(getattr(tensor, "mx_tensor_kind", "") or ""),
        "owner_module": str(getattr(tensor, "mx_owner_module", "") or ""),
        "owner_class": str(getattr(tensor, "mx_owner_class", "") or ""),
        "quant_method": str(getattr(tensor, "mx_quant_method", "") or ""),
        "runtime_role": str(getattr(tensor, "mx_runtime_role", "") or ""),
        "replace_policy": str(getattr(tensor, "mx_replace_policy", "") or ""),
    }


def _first_attr(obj: object, names: tuple[str, ...]) -> object | None:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _attach_runtime_metadata(
    tensor: torch.Tensor,
    *,
    tensor_kind: str,
    owner_module: str,
    owner_class: str,
    quant_method: str,
    runtime_role: str,
    replace_policy: str,
) -> None:
    tensor.mx_tensor_kind = tensor_kind
    tensor.mx_owner_module = owner_module
    tensor.mx_owner_class = owner_class
    tensor.mx_quant_method = quant_method
    tensor.mx_runtime_role = runtime_role
    tensor.mx_replace_policy = replace_policy


def _quant_method_name(module: nn.Module) -> str:
    quant_method = getattr(module, "quant_method", None)
    if quant_method is None:
        return ""
    cls = type(quant_method)
    return f"{cls.__module__}.{cls.__qualname__}"


def capture_humming_runtime_tensors_from_model(model: nn.Module) -> int:
    """Best-effort local module pass after vLLM Humming post-load completes."""
    return _provider_capture_humming_runtime_tensors_from_model(model)


def _manifest_tensor_for_module_leaf(
    module: nn.Module,
    leaf: str,
    tensor: torch.Tensor,
    quantization: str = "",
) -> tuple[torch.Tensor, str, str]:
    runtime_role = leaf
    replace_policy = "structural_replace"
    provider = get_quantization_provider(quantization)
    decision = provider.resolve_manifest_tensor(
        module,
        leaf,
        tensor,
        quantization=quantization,
    )
    if decision is None:
        return tensor, runtime_role, replace_policy
    return decision.tensor, decision.runtime_role, decision.replace_policy


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


@contextmanager
def capture_humming_runtime_tensors(enabled: bool = True):
    """Record Humming runtime tensors as they are attached to vLLM layers.

    Humming's post-load path prepares kernel-ready packed tensors and writes
    them with ``HummingLayerMethod.may_set_param``. ModelExpress needs those
    final runtime tensors in the RDMA manifest, not the dense checkpoint
    parameter that may still be visible as ``*.weight``.
    """
    with _provider_capture_humming_runtime_tensors(enabled=enabled):
        yield


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
            if attr_name == HUMMING_RUNTIME_TENSORS_ATTR:
                continue
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


def collect_module_tensors(
    model: nn.Module,
    *,
    quantization: str = "",
) -> dict[str, torch.Tensor]:
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
    modules = dict(model.named_modules())
    allowed_names = getattr(model, SOURCE_MANIFEST_TENSOR_NAMES_ATTR, None)
    if allowed_names is not None:
        allowed_names = set(allowed_names)
    provider = get_quantization_provider(quantization)
    storage_view_count = 0
    skipped_duplicate = 0
    for name, tensor, _tensor_type in iter_module_tensors(model):
        if allowed_names is not None and (
            name not in allowed_names and f"{name}.__storage" not in allowed_names
        ):
            logger.debug(
                "Skipping tensor '%s' because it is absent from source RDMA manifest",
                name,
            )
            continue
        t = tensor.data if hasattr(tensor, "data") else tensor
        module_name, _, leaf = name.rpartition(".")
        if provider.skip_manifest_tensor(name, leaf or name, _tensor_type):
            logger.debug("Skipping runtime-only tensor '%s' from RDMA manifest", name)
            continue
        owner_module = module_name
        owner = modules.get(module_name) if module_name else model
        owner_class = type(owner).__name__ if owner is not None else ""
        quant_method = _quant_method_name(owner) if owner is not None else ""
        runtime_role = leaf or name
        replace_policy = "structural_replace"
        if owner is not None and leaf:
            t, runtime_role, replace_policy = _manifest_tensor_for_module_leaf(
                owner, leaf, t, quantization,
            )
        _attach_runtime_metadata(
            t,
            tensor_kind=_tensor_type,
            owner_module=owner_module,
            owner_class=owner_class,
            quant_method=quant_method,
            runtime_role=runtime_role,
            replace_policy=replace_policy,
        )
        if not t.is_cuda or t.numel() == 0 or t.data_ptr() == 0:
            logger.debug(
                "Skipping non-RDMA tensor '%s' "
                "(is_cuda=%s, numel=%s, data_ptr=0x%x)",
                name, t.is_cuda, t.numel(), t.data_ptr(),
            )
            continue
        if t.is_contiguous():
            ptr = t.data_ptr()
            if ptr in seen_ptrs:
                logger.debug(f"Skipping duplicate tensor '{name}' (same data_ptr)")
                skipped_duplicate += 1
                continue
            seen_ptrs.add(ptr)
            tensors[name] = t
            if debug_tensor_enabled(name):
                logger.warning("Collected manifest tensor: %s", tensor_debug_summary(name, t))
        else:
            sv = storage_view(t)
            ptr = sv.data_ptr()
            if ptr in seen_ptrs:
                skipped_duplicate += 1
                continue
            storage_name = f"{name}.__storage"
            if allowed_names is not None and storage_name not in allowed_names:
                logger.debug(
                    "Skipping storage tensor '%s' because it is absent from "
                    "source RDMA manifest",
                    storage_name,
                )
                continue
            seen_ptrs.add(ptr)
            tensors[storage_name] = sv
            if debug_tensor_enabled(name) or debug_tensor_enabled(storage_name):
                logger.warning(
                    "Collected non-contiguous manifest tensor: original=%s storage=%s",
                    tensor_debug_summary(name, t),
                    tensor_debug_summary(storage_name, sv),
                )
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
