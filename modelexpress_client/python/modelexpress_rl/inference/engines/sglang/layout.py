# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-mutating loader capture for Qwen3 dense/MoE BF16 refit."""

from __future__ import annotations

import torch
from modelexpress.refit.reshard.geometry import capture_geometry
from modelexpress.refit.reshard.types import CaptureResult, UnsupportedReshard


def _sglang_default_weight_loader():
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    return default_weight_loader


class SglangFullTensorLayout:
    def __init__(self, runner):
        self._model = runner.model
        self._model_config = runner.model_config
        if any(
            bool(getattr(runner.server_args, key, False))
            for key in ("enable_lora", "lora_paths", "speculative_algorithm")
        ):
            raise UnsupportedReshard(
                "SGLang live refit does not support LoRA or speculative models"
            )
        self._tensor_registry = dict(self._model.named_parameters())
        self._tensor_registry.update(self._model.named_buffers())
        for module in self._model.modules():
            if any(isinstance(value, torch.Tensor) for value in vars(module).values()):
                raise UnsupportedReshard(
                    "SGLang live refit does not support hidden tensor attributes"
                )
        self._validate_runtime()
        self._signature = self._storage_signature()

    def _storage_signature(self):
        return tuple(
            (
                name,
                value.data_ptr(),
                tuple(value.shape),
                value.dtype,
                value.device,
                tuple(value.stride()),
            )
            for name, value in self._model.named_parameters()
        )

    def validate_storage(self):
        if self._storage_signature() != self._signature:
            raise UnsupportedReshard(
                "SGLang live parameter storage changed; restart the worker"
            )

    def parameter_layout(self):
        self.validate_storage()
        return {
            name: (tuple(value.shape), value.dtype)
            for name, value in self._model.named_parameters()
        }

    def _validate_runtime(self) -> None:
        dtype = getattr(self._model_config, "dtype", None)
        quantization = getattr(self._model_config, "quantization", None)
        hf_config = getattr(self._model_config, "hf_config", self._model_config)
        architectures = getattr(hf_config, "architectures", ())
        if not architectures or not set(architectures).issubset(
            {"Qwen3ForCausalLM", "Qwen3MoeForCausalLM"}
        ):
            raise UnsupportedReshard(
                "SGLang live refit is validated only for Qwen3 dense/MoE BF16"
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

        params = dict(self._model.named_parameters(remove_duplicate=False))
        storages = [param.untyped_storage().data_ptr() for param in params.values()]
        if len(storages) != len(set(storages)):
            raise UnsupportedReshard(
                "SGLang live refit does not support aliased parameter storage"
            )
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

    def capture(self, manifest: list) -> tuple[CaptureResult, dict]:
        self.validate_storage()
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
