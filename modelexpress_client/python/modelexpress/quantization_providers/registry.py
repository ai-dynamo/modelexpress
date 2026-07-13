# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry for quantization RDMA manifest providers."""

from __future__ import annotations

from .base import DefaultManifestProvider, QuantizationManifestProvider
from .humming import HummingManifestProvider

_DEFAULT_PROVIDER = DefaultManifestProvider()
_PROVIDERS: tuple[QuantizationManifestProvider, ...] = (
    HummingManifestProvider(),
)


def get_quantization_provider(quantization: str = "") -> QuantizationManifestProvider:
    for provider in _PROVIDERS:
        if provider.enabled(quantization):
            return provider
    return _DEFAULT_PROVIDER
