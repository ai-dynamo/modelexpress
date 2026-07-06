# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress - High-performance GPU-to-GPU model weight transfers.

This package provides:
- NIXL-based RDMA transfers for GPU tensors
- GPUDirect Storage (GDS) for direct file-to-GPU loading
- vLLM worker extension for serving model weights
- Custom model loaders for FP8 model support (DeepSeek-V3, etc.)

Quick Start (vLLM):
    from modelexpress import register_modelexpress_loaders
    register_modelexpress_loaders()

    # vllm serve model --load-format modelexpress
    # Auto-detects: RDMA -> GDS -> disk
"""

import logging

from . import envs

_logger = logging.getLogger(__name__)
_loaders_registered = False


def configure_vllm_logging():
    """Ensure modelexpress loggers are visible in vLLM worker subprocesses.

    vLLM only attaches log handlers to the "vllm" namespace. Without this,
    all "modelexpress.*" output is silently dropped in EngineCore worker
    processes. Copies vLLM's handlers onto the "modelexpress" parent logger
    so every child inherits them via propagation. Idempotent.
    """
    mx_root = logging.getLogger("modelexpress")
    if mx_root.handlers:
        return
    vllm_logger = logging.getLogger("vllm")
    for handler in vllm_logger.handlers:
        mx_root.addHandler(handler)
    mx_level = envs.MODEL_EXPRESS_LOG_LEVEL
    if mx_level and hasattr(logging, mx_level):
        mx_root.setLevel(getattr(logging, mx_level))
    elif vllm_logger.level != logging.NOTSET:
        mx_root.setLevel(vllm_logger.level)


def register_modelexpress_loaders():
    """
    Register ModelExpress loaders with vLLM.

    This function ensures loaders are registered exactly once. It can be called
    multiple times safely (idempotent).

    Enables:
        --load-format modelexpress  (auto-detect: RDMA -> GDS -> disk)
        --load-format mx            (backward-compatible alias)
    """
    global _loaders_registered
    if _loaders_registered:
        return

    from .engines.vllm import register_modelexpress_loaders as register_vllm_loaders

    register_vllm_loaders()

    _loaders_registered = True
    _logger.debug("ModelExpress loaders registered")


from .client import MxClient  # noqa: F401
from .gds_loader import MxGdsLoader  # noqa: F401
from .gds_transfer import GdsTransferManager  # noqa: F401
from .metadata.publisher import PublisherThread  # noqa: F401
from .nemo_rl_v2 import (  # noqa: F401
    MxV2RefitReceiver,
    MxV2TrainingPublisher,
    TrainerWorldLayout,
)
from .training_publisher import MxTrainingPublisher  # noqa: F401
from .refit_receiver import MxRefitReceiver  # noqa: F401

# Rank-to-rank RL substrate (from the #349 lineage). These modules are
# pure-Python dataclasses + planners (torch is lazy-imported inside method
# bodies), safe to re-export eagerly.
from .shape_descriptors import NonExpertShardSpec  # noqa: F401
from .rl_slice_descriptors import (  # noqa: F401
    CoveragePlan,
    SegmentPlan,
    SliceOwnership,
    SliceRequest,
)
from .rl_reshard_planner import plan_coverage  # noqa: F401
from .rl_expert_layout import (  # noqa: F401
    ExpertPlacement,
    compute_local_expert_ids,
    validate_placement_partition,
)
from .rank_local_publisher import (  # noqa: F401
    PlacementDescriptor,
    RankLocalPublisher,
)

__all__ = [
    "CoveragePlan",
    "ExpertPlacement",
    "GdsTransferManager",
    "MxClient",
    "MxGdsLoader",
    "MxRefitReceiver",
    "MxTrainingPublisher",
    "MxV2RefitReceiver",
    "MxV2TrainingPublisher",
    "NonExpertShardSpec",
    "PlacementDescriptor",
    "PublisherThread",
    "RankLocalPublisher",
    "SegmentPlan",
    "SliceOwnership",
    "SliceRequest",
    "TrainerWorldLayout",
    "compute_local_expert_ids",
    "configure_vllm_logging",
    "plan_coverage",
    "register_modelexpress_loaders",
    "validate_placement_partition",
]
