# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refresh non-transferable vLLM attention state after runtime tensor transfer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from modelexpress.accelerators import AcceleratorBackend

logger = logging.getLogger(__name__)

_VLLM_ATTENTION_SCALE_NAMES: tuple[str, ...] = ("q", "k", "v")


@torch.no_grad()
def refresh_host_quantization_state(
    model: torch.nn.Module,
    vllm_config: object,
    accelerator_backend: AcceleratorBackend,
    *,
    allow_warm: bool = False,
) -> None:
    """Rebuild host mirrors without reprocessing received runtime tensors.

    Per-head scales use their maximum, matching vLLM's host conversion.
    Cold loading requires uninitialized attention caches. Warm eager refits
    invalidate them for vLLM to recompute on the next forward; captured graph
    scalars cannot be updated this way.
    """
    model_config = getattr(vllm_config, "model_config", None)
    if allow_warm and not getattr(model_config, "enforce_eager", False):
        raise RuntimeError("Warm vLLM host-scale refresh requires enforce_eager")

    float_names = tuple(
        f"_{scale_name}_scale_float"
        for scale_name in _VLLM_ATTENTION_SCALE_NAMES
    )
    cpu_names = ("_k_scale_cpu", "_v_scale_cpu")
    device_names = tuple(
        f"_{scale_name}_scale" for scale_name in _VLLM_ATTENTION_SCALE_NAMES
    )
    required_names = device_names + float_names + ("_o_scale_float",)
    flashinfer_cache_names = ("bmm1_scale", "bmm2_scale", "o_sf_scale")

    cache_config = getattr(vllm_config, "cache_config", None)
    configured_cache_dtype = getattr(cache_config, "cache_dtype", None)
    fp8_expected = isinstance(configured_cache_dtype, str) and (
        configured_cache_dtype.startswith("fp8")
    )

    # This is a shim over vLLM private attributes, so an upstream scale
    # contract change must fail the RDMA strategy instead of silently
    # serving with stale values. The caller must discard or fence the model
    # if validation fails after transferred tensors have mutated it.
    refreshed_values = 0
    stale_values = 0
    refreshed_modules = 0
    for module_name, module in model.named_modules():
        module_label = module_name or type(module).__name__
        kv_cache_dtype = getattr(module, "kv_cache_dtype", None)
        if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8"):
            fp8_expected = True

        host_names = float_names + cpu_names
        if not any(hasattr(module, name) for name in host_names):
            continue

        missing_names = [
            name for name in required_names if not hasattr(module, name)
        ]
        if missing_names:
            raise RuntimeError(
                "Incomplete vLLM attention scale contract on "
                f"{module_label}: missing {', '.join(missing_names)}"
            )

        for float_name in float_names:
            if not isinstance(getattr(module, float_name), float):
                raise RuntimeError(
                    "Invalid vLLM attention host scalar on "
                    f"{module_label}: {float_name} must be a float"
                )

        for cpu_name in cpu_names:
            if not hasattr(module, cpu_name):
                continue
            cpu_mirror = getattr(module, cpu_name)
            if (
                not isinstance(cpu_mirror, torch.Tensor)
                or cpu_mirror.numel() != 1
                or not torch.is_floating_point(cpu_mirror)
            ):
                raise RuntimeError(
                    "Invalid vLLM attention CPU scale on "
                    f"{module_label}: {cpu_name} must be a floating-point "
                    "singleton tensor"
                )

        impl = getattr(module, "impl", None)
        initialized_cache_names = []
        if module._o_scale_float is not None:
            initialized_cache_names.append("_o_scale_float")
        initialized_cache_names.extend(
            cache_name
            for cache_name in flashinfer_cache_names
            if hasattr(impl, cache_name) and getattr(impl, cache_name) is not None
        )
        if initialized_cache_names and not allow_warm:
            raise RuntimeError(
                "vLLM attention host-scale refresh requires a cold-load "
                f"state on {module_label}; already initialized: "
                f"{', '.join(initialized_cache_names)}"
            )

        for scale_name, tensor_name in zip(
            _VLLM_ATTENTION_SCALE_NAMES, device_names, strict=True
        ):
            scale = getattr(module, tensor_name)
            if (
                not isinstance(scale, torch.Tensor)
                or not accelerator_backend.is_accel_tensor(scale)
                or scale.numel() == 0
                or not torch.is_floating_point(scale)
            ):
                raise RuntimeError(
                    "Invalid vLLM accelerator attention scale on "
                    f"{module_label}: {tensor_name} must be a nonempty "
                    "floating-point accelerator tensor"
                )

            scale_float = scale.detach().float()
            if not bool(torch.isfinite(scale_float).all().item()) or not bool(
                (scale_float > 0).all().item()
            ):
                raise RuntimeError(
                    "Invalid vLLM accelerator attention scale on "
                    f"{module_label}: {tensor_name} must contain only finite, "
                    "positive values"
                )
            value = float(scale_float.max().item())
            float_name = f"_{scale_name}_scale_float"
            previous = getattr(module, float_name)
            if previous != value:
                stale_values += 1
                logger.debug(
                    "Refreshing vLLM host scale %s.%s: %r -> %r",
                    module_label,
                    float_name,
                    previous,
                    value,
                )
            setattr(module, float_name, value)

            cpu_mirror = getattr(module, f"_{scale_name}_scale_cpu", None)
            if isinstance(cpu_mirror, torch.Tensor):
                cpu_mirror.fill_(value)

            refreshed_values += 1
        if allow_warm:
            # vLLM lazily rebuilds these from q/k/v and the next output_scale.
            # Reset them together so output quantization is applied exactly once.
            module._o_scale_float = None
            for cache_name in flashinfer_cache_names:
                if hasattr(impl, cache_name):
                    setattr(impl, cache_name, None)
        refreshed_modules += 1

    if fp8_expected and not refreshed_modules:
        raise RuntimeError(
            "FP8 KV cache requires recognizable vLLM q/k/v host scale state, "
            "but no attention module was refreshed"
        )

    if refreshed_values:
        logger.info(
            "Refreshed %d vLLM host attention scales across %d modules "
            "after RDMA receive (%d stale values replaced)",
            refreshed_values,
            refreshed_modules,
            stale_values,
        )
