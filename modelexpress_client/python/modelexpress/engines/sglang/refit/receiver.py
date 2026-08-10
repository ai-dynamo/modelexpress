# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang hooks for the engine-agnostic NIXL reshard receiver."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from torch.nn import Module

from modelexpress.refit.reshard.geometry import capture_geometry
from modelexpress.refit.reshard.receiver import ReshardReceiver
from modelexpress.refit.reshard.types import CaptureResult, UnsupportedReshard


def sglang_layout_signature(model: Module) -> str:
    """Return a stable signature of the live destination parameter layout."""
    layout = [
        (name, list(param.shape), str(param.dtype), list(param.stride()))
        for name, param in model.named_parameters()
    ]
    encoded = json.dumps(layout, separators=(",", ":"), sort_keys=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sglang_default_weight_loader():
    try:
        from sglang.srt.model_loader.weight_utils import default_weight_loader
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "SGLang live refit requires "
            "sglang.srt.model_loader.weight_utils.default_weight_loader"
        ) from exc
    return default_weight_loader


class SglangReshardReceiver(ReshardReceiver):
    """Full-BF16, whole-model live-refit receiver for SGLang.

    Geometry is captured by dry-running the live model's real ``load_weights``
    dispatch with zero-storage placeholders. Installation copies the resulting
    destination-layout buffers into existing parameter storage, preserving
    addresses held by CUDA graphs.
    """

    def __init__(
        self,
        *,
        model: Module,
        tensor_registry: dict[str, torch.Tensor],
        model_config: Any,
        **base_kwargs: Any,
    ) -> None:
        self._model = model
        self._tensor_registry = dict(tensor_registry)
        self._model_config = model_config
        self._poisoned = False
        self._install_started = False
        self._validate_runtime()
        super().__init__(**base_kwargs)

    @property
    def layout_signature(self) -> str:
        return sglang_layout_signature(self._model)

    @property
    def poisoned(self) -> bool:
        """Whether an in-place install failed after writes began."""
        return self._poisoned

    def update_weights(self, step: int, *, timeout: float | None = None) -> dict:
        """Poison the receiver if install or its following device sync fails."""
        self._install_started = False
        try:
            metrics = super().update_weights(step, timeout=timeout)
        except BaseException:
            if self._install_started:
                self._poisoned = True
            raise
        self._install_started = False
        return metrics

    def _validate_runtime(self) -> None:
        dtype = getattr(self._model_config, "dtype", None)
        quantization = getattr(self._model_config, "quantization", None)
        model_identity = " ".join(
            [
                str(getattr(self._model_config, "model_path", "") or ""),
                str(getattr(self._model_config, "model", "") or ""),
                " ".join(getattr(self._model_config, "architectures", ()) or ()),
            ]
        )
        if "qwen3" not in model_identity.lower():
            raise UnsupportedReshard(
                "SGLang live refit is currently validated only for Qwen3 BF16"
            )
        if dtype != torch.bfloat16:
            raise UnsupportedReshard(
                f"SGLang live refit supports BF16 only, got model dtype {dtype}"
            )
        if quantization:
            raise UnsupportedReshard(
                "SGLang live refit does not support quantized/FP8 models "
                f"(quantization={quantization!r})"
            )
        if any(
            bool(getattr(self._model_config, attr, False))
            for attr in ("enable_lora", "lora_enabled", "lora_paths")
        ):
            raise UnsupportedReshard("SGLang live refit does not support LoRA")

        params = dict(self._model.named_parameters())
        if not params:
            raise RuntimeError("SGLang live refit found no destination parameters")
        adapter_names = [
            name
            for name in set(params) | set(self._tensor_registry)
            if "lora" in name.lower() or "adapter" in name.lower()
        ]
        if adapter_names:
            raise UnsupportedReshard(
                "SGLang live refit does not support LoRA/adapter tensors: "
                f"{sorted(adapter_names)[:10]}"
            )
        bad_dtype = [
            name for name, param in params.items() if param.dtype != torch.bfloat16
        ]
        if bad_dtype:
            raise UnsupportedReshard(
                "SGLang live refit requires every destination parameter to be "
                f"BF16; unsupported: {bad_dtype[:10]}"
            )
        non_contiguous = [
            name for name, param in params.items() if not param.is_contiguous()
        ]
        if non_contiguous:
            raise UnsupportedReshard(
                "SGLang live refit requires contiguous destination parameters; "
                f"unsupported: {non_contiguous[:10]}"
            )

        registered = set(self._tensor_registry)
        parameter_names = set(params)
        buffers = dict(self._model.named_buffers())
        supported_names = parameter_names | set(buffers)
        missing = sorted(parameter_names - registered)
        hidden = sorted(registered - supported_names)
        if missing or hidden:
            raise UnsupportedReshard(
                "SGLang startup tensor registry must cover every live BF16 "
                f"parameter and contain no unregistered hidden tensors "
                f"(missing={missing[:10]}, unsupported hidden tensors={hidden[:10]})"
            )
        live_tensors = {**params, **buffers}
        aliased = [
            name
            for name in registered & supported_names
            if self._tensor_registry[name].data_ptr() != live_tensors[name].data_ptr()
        ]
        if aliased:
            raise RuntimeError(
                "SGLang startup tensor registry no longer points at live parameter "
                f"storage: {aliased[:10]}"
            )

    def _capture(self, manifest: list) -> tuple[CaptureResult, dict]:
        bad_sources = [
            name
            for name, dtype, _shape in manifest
            if dtype != torch.bfloat16
            or "lora" in name.lower()
            or "adapter" in name.lower()
        ]
        if bad_sources:
            raise UnsupportedReshard(
                "SGLang live refit accepts only full BF16 base-model tensors; "
                f"unsupported sources: {bad_sources[:10]}"
            )

        capture = capture_geometry(
            self._model,
            manifest,
            default_weight_loader=_sglang_default_weight_loader(),
        )
        if capture.unsupported or capture.unattributed:
            raise UnsupportedReshard(
                "SGLang loader geometry was not completely attributable "
                f"(unsupported={capture.unsupported[:10]}, "
                f"unattributed={capture.unattributed})"
            )

        param_layout = {
            name: (tuple(param.shape), param.dtype)
            for name, param in self._model.named_parameters()
        }
        captured = {copy.param_name for copy in capture.copies}
        missing = sorted(set(param_layout) - captured)
        unexpected = sorted(captured - set(param_layout))
        if missing or unexpected:
            raise UnsupportedReshard(
                "SGLang whole-model refit requires exact destination parameter "
                f"coverage (missing={missing[:10]}, unexpected={unexpected[:10]})"
            )
        return capture, param_layout

    @torch.no_grad()
    def _install(self, recv_buffers: dict) -> None:
        if self._poisoned:
            raise RuntimeError(
                "SGLang receiver is poisoned by a prior in-place install failure; "
                "restart the worker before serving or refitting"
            )

        params = dict(self._model.named_parameters())
        if set(recv_buffers) != set(params):
            raise RuntimeError(
                "SGLang install refused non-whole-model receive buffers "
                f"(received={len(recv_buffers)}, expected={len(params)})"
            )
        for name, param in params.items():
            incoming = recv_buffers[name]
            if (
                incoming.shape != param.shape
                or incoming.dtype != param.dtype
                or incoming.device != param.device
            ):
                raise RuntimeError(
                    f"SGLang install layout mismatch for {name}: "
                    f"incoming={tuple(incoming.shape)}/{incoming.dtype}/{incoming.device}, "
                    f"live={tuple(param.shape)}/{param.dtype}/{param.device}"
                )

        self._install_started = True
        try:
            for name, param in params.items():
                param.copy_(recv_buffers[name])
        except BaseException:
            # The runtime has no transaction or shadow-model swap. A device error
            # after the first copy can leave a mixed version, so this receiver
            # permanently fails closed and the worker must be restarted.
            self._poisoned = True
            raise


__all__ = ["SglangReshardReceiver", "sglang_layout_signature"]
