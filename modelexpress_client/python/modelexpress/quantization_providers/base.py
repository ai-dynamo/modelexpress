# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base interface for quantization-specific RDMA manifest behavior."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import torch
import torch.nn as nn

MANIFEST_TENSOR_OVERRIDES_ATTR = "_mx_manifest_tensor_overrides"
SOURCE_MANIFEST_TENSOR_NAMES_ATTR = "_mx_source_manifest_tensor_names"
NO_STRUCTURAL_REPLACE_POLICY = "no_structural_replace"
REJECT_IF_MISMATCH_POLICY = "reject_if_mismatch"


@dataclass(frozen=True)
class ManifestTensorDecision:
    tensor: torch.Tensor
    runtime_role: str
    replace_policy: str


def tensor_data(value: object) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    tensor = value.data if hasattr(value, "data") else value
    return tensor if isinstance(tensor, torch.Tensor) else None


class QuantizationManifestProvider:
    """Quantization-specific hooks used by generic RDMA manifest code.

    A new quantization integration should live in its own module and implement
    this interface, then register itself in ``registry.py``. The generic tensor
    walker, vLLM adapter, and RDMA strategy should not need quantization-specific
    conditionals.
    """

    name = "default"

    def enabled(self, quantization: str) -> bool:
        return False

    def capture_during_load(self, *, enabled: bool = True) -> ContextManager[None]:
        del enabled
        return nullcontext()

    def capture_from_model(self, model: nn.Module) -> int:
        del model
        return 0

    def resolve_manifest_tensor(
        self,
        module: nn.Module,
        leaf: str,
        tensor: torch.Tensor,
        *,
        quantization: str = "",
    ) -> ManifestTensorDecision | None:
        del module, leaf, tensor, quantization
        return None

    def skip_manifest_tensor(self, name: str, leaf: str, tensor_type: str) -> bool:
        del name, leaf, tensor_type
        return False

    def align_target_module_from_source(
        self,
        module: nn.Module,
        leaf: str,
        desc,
    ) -> None:
        del module, leaf, desc

    def after_target_tensor_rebuilt(
        self,
        module: nn.Module,
        leaf: str,
        desc,
    ) -> None:
        del module, leaf, desc


class DefaultManifestProvider(QuantizationManifestProvider):
    name = "default"
