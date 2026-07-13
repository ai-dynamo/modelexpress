# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantization-specific RDMA manifest providers."""

from .base import (
    MANIFEST_TENSOR_OVERRIDES_ATTR,
    NO_STRUCTURAL_REPLACE_POLICY,
    REJECT_IF_MISMATCH_POLICY,
    SOURCE_MANIFEST_TENSOR_NAMES_ATTR,
    ManifestTensorDecision,
    QuantizationManifestProvider,
)
from .registry import get_quantization_provider

__all__ = [
    "MANIFEST_TENSOR_OVERRIDES_ATTR",
    "NO_STRUCTURAL_REPLACE_POLICY",
    "REJECT_IF_MISMATCH_POLICY",
    "SOURCE_MANIFEST_TENSOR_NAMES_ATTR",
    "ManifestTensorDecision",
    "QuantizationManifestProvider",
    "get_quantization_provider",
]
