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
import os

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
    mx_level = os.environ.get("MODEL_EXPRESS_LOG_LEVEL", "").upper()
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
from .metadata.heartbeat import HeartbeatThread  # noqa: F401

# shape_descriptors is torch-free; safe to import eagerly.
from .shape_descriptors import (  # noqa: F401
    COMPILE_TARGET_CUTLASS_FP8,
    COMPILE_TARGET_DEEPGEMM_FP8,
    COMPILE_TARGET_HF_RAW,
    COMPILE_TARGET_TRTLLM,
    COMPILE_TARGET_VLLM_FUSED,
    NonExpertShardSpec,
    TensorDescriptorV2,
    compile_target_matches,
)

# Rank-to-rank reshard contract (used by verl + NemoRL v2 + PrimeRL mx_v2).
# These three modules are pure-Python (dataclasses + collections only);
# torch is lazy-imported inside method bodies in rank_local_publisher, so
# they're safe to re-export eagerly even on torch-free CI runners.
from .rl_slice_descriptors import (  # noqa: F401
    CoveragePlan,
    PlanIncompleteError,
    QuantizationMetadataError,
    SegmentPlan,
    SliceOwnership,
    SliceRequest,
)
from .rl_reshard_planner import (  # noqa: F401
    collect_byte_savings_vs_allgather,
    plan_coverage,
    summarize_plan,
)
from .rank_local_publisher import (  # noqa: F401
    PlacementDescriptor,
    RankLocalPublisher,
)

# The v2 RL helpers (training_publisher, refit_receiver, nemo_rl_v2) all
# import torch. Keeping them as eager top-level re-exports makes
# ``import modelexpress`` fail in environments that don't ship torch
# (e.g. the ``[dev]`` Python Tests CI runner, vLLM plugin discovery on
# CPU-only images). Expose them lazily via PEP 562 ``__getattr__`` so
# they only load on demand:
#
#     from modelexpress import MxV2TrainingPublisher  # imports torch here
#
# is preserved, while ``import modelexpress`` stays light.
_LAZY_ATTRS = {
    "MxRefitReceiver": ".refit_receiver",
    "TransferStats": ".refit_receiver",
    "MxTrainingPublisher": ".training_publisher",
    "MxV2RefitReceiver": ".nemo_rl_v2",
    "MxV2TrainingPublisher": ".nemo_rl_v2",
    "SliceCoveragePlan": ".nemo_rl_v2",
    "SliceSource": ".nemo_rl_v2",
    "TargetTPLayout": ".nemo_rl_v2",
    "TrainerWorldLayout": ".nemo_rl_v2",
    "V2SourceCandidate": ".nemo_rl_v2",
}


def __getattr__(name):
    """Lazy-import attributes that pull torch (PEP 562)."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'modelexpress' has no attribute {name!r}")
    from importlib import import_module

    mod = import_module(module_name, __name__)
    attr = getattr(mod, name)
    globals()[name] = attr
    return attr


def __dir__():
    return sorted(list(globals()) + list(_LAZY_ATTRS))


__all__ = [
    "COMPILE_TARGET_CUTLASS_FP8",
    "COMPILE_TARGET_DEEPGEMM_FP8",
    "COMPILE_TARGET_HF_RAW",
    "COMPILE_TARGET_TRTLLM",
    "COMPILE_TARGET_VLLM_FUSED",
    "CoveragePlan",
    "GdsTransferManager",
    "HeartbeatThread",
    "MxClient",
    "MxGdsLoader",
    "MxRefitReceiver",
    "MxTrainingPublisher",
    "MxV2RefitReceiver",
    "MxV2TrainingPublisher",
    "PlacementDescriptor",
    "PlanIncompleteError",
    "QuantizationMetadataError",
    "RankLocalPublisher",
    "SegmentPlan",
    "SliceCoveragePlan",
    "SliceOwnership",
    "SliceRequest",
    "SliceSource",
    "TargetTPLayout",
    "NonExpertShardSpec",
    "TensorDescriptorV2",
    "TrainerWorldLayout",
    "TransferStats",
    "V2SourceCandidate",
    "collect_byte_savings_vs_allgather",
    "compile_target_matches",
    "configure_vllm_logging",
    "plan_coverage",
    "register_modelexpress_loaders",
    "summarize_plan",
]
