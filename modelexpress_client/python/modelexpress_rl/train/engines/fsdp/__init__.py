# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FSDP/DTensor trainer-engine adapter for RL refit (async-capable)."""

from .adapter import FSDPTrainerAdapter
from .publisher import (
    LocalTensorShard,
    build_fsdp_reshard_manifest,
    capture_local_shards,
)

__all__ = [
    "FSDPTrainerAdapter",
    "LocalTensorShard",
    "build_fsdp_reshard_manifest",
    "capture_local_shards",
]
