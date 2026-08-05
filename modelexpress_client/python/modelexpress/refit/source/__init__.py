# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded framework-native canonical capture and exact-base encoding."""

from .canonical import CanonicalTensorSpec
from .megatron_bridge import (
    MegatronBridgeHfBucketConfig,
    for_each_megatron_hf_bucket,
)

__all__ = [
    "CanonicalTensorSpec",
    "MegatronBridgeHfBucketConfig",
    "for_each_megatron_hf_bucket",
]
