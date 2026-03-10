# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS configuration models.

Defines all configuration for the MX GMS CLI, including engine type,
mode (source/target), parallelism settings, and weight source selection.
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of StrEnum for Python 3.10."""
from typing import Optional

from pydantic import BaseModel, Field


class EngineType(StrEnum):
    """Supported inference engine types."""

    VLLM = "vllm"
    SGLANG = "sglang"
    TRTLLM = "trtllm"


class GmsMode(StrEnum):
    """GMS operating mode."""

    SOURCE = "source"
    TARGET = "target"


class WeightSourceType(StrEnum):
    """Weight loading source type."""

    DISK = "disk"
    GDS = "gds"
    S3 = "s3"
    MODEL_STREAMER = "model-streamer"


class MxConfig(BaseModel):
    """MX-specific config passed to shared hooks.

    This is the subset of GmsConfig needed by mx_hooks functions.
    It must be fully serializable (pickle-safe) for torch.mp.spawn.
    """

    mx_server: str = "localhost:8001"
    model_name: str = ""
    contiguous_reg: bool = False
    sync_start: bool = True
    expected_workers: int = 1


class GmsConfig(BaseModel):
    """Unified GMS configuration parsed from CLI arguments."""

    model: str = Field(..., description="Model name or path")
    engine: EngineType = Field(default=EngineType.VLLM)
    mode: GmsMode = Field(default=GmsMode.SOURCE)
    tp_size: int = Field(default=1, ge=1)
    ep_size: int = Field(default=1, ge=1)
    device: int = Field(default=0, ge=0, description="Base device (single-GPU only)")
    mx_server: str = Field(default="localhost:8001")
    model_name: Optional[str] = Field(
        default=None, description="Override model name for MX Server"
    )
    dtype: str = Field(default="auto")
    trust_remote_code: bool = Field(default=False)
    revision: Optional[str] = Field(default=None)
    max_model_len: Optional[int] = Field(default=None, ge=1)
    enable_expert_parallel: bool = Field(default=False)

    # Weight source configuration
    weight_source: WeightSourceType = Field(default=WeightSourceType.DISK)
    s3_bucket: Optional[str] = Field(default=None)
    s3_prefix: Optional[str] = Field(default=None)
    cache_endpoint: Optional[str] = Field(default=None)

    @property
    def total_workers(self) -> int:
        """Total number of worker processes (tp_size * ep_size)."""
        return self.tp_size * self.ep_size

    def to_mx_config(self) -> MxConfig:
        """Extract the MX-specific config for shared hooks."""
        return MxConfig(
            mx_server=self.mx_server,
            model_name=self.model_name or self.model,
            contiguous_reg=False,
            sync_start=True,
            expected_workers=self.total_workers,
        )
