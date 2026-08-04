# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, framework-specific sources for canonical HF tensor buckets."""

from .base import (
    CanonicalBucket,
    CanonicalBucketConsumer,
    CanonicalSourceError,
    CanonicalTensorSpec,
)
from .canonical import CanonicalCapture, CanonicalFormatIdentity
from .fsdp import FsdpHfBucketConfig, for_each_fsdp_hf_bucket
from .megatron_bridge import (
    MegatronBridgeHfBucketConfig,
    for_each_megatron_hf_bucket,
)

__all__ = [
    "CanonicalBucket",
    "CanonicalBucketConsumer",
    "CanonicalCapture",
    "CanonicalFormatIdentity",
    "CanonicalSourceError",
    "CanonicalTensorSpec",
    "FsdpHfBucketConfig",
    "MegatronBridgeHfBucketConfig",
    "for_each_fsdp_hf_bucket",
    "for_each_megatron_hf_bucket",
]
