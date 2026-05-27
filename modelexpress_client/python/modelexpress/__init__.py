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
from .rl_fanout import RlTreeFanoutPolicy  # noqa: F401
from .rl_metadata import RlSourceMetadata, RlSourceRole  # noqa: F401
from .rl_reshard import (  # noqa: F401
    MissingTensor,
    TensorReceiveSpec,
    TensorShardSpec,
    TensorSlice,
    TransferPlan,
    TransferPlanEntry,
    plan_dense_reshard_transfers,
    plan_exact_transfers,
    receive_specs_from_shape_registry,
    receive_specs_from_tensors,
    source_specs_from_shape_registry,
    tensor_metadata_from_receive_specs,
)
from .rl_transfer_lease import (  # noqa: F401
    RlTransferLeaseInventory,
    RlTransferLeaseReportSummary,
    summarize_report_leases,
)
from .rl_update_lifecycle import RlWeightUpdateLifecycleHooks  # noqa: F401

__all__ = [
    "GdsTransferManager",
    "HeartbeatThread",
    "MxClient",
    "MxGdsLoader",
    "MissingTensor",
    "RlSourceMetadata",
    "RlSourceRole",
    "RlTreeFanoutPolicy",
    "RlTransferLeaseInventory",
    "RlTransferLeaseReportSummary",
    "RlWeightUpdateLifecycleHooks",
    "TensorReceiveSpec",
    "TensorShardSpec",
    "TensorSlice",
    "TransferPlan",
    "TransferPlanEntry",
    "configure_vllm_logging",
    "plan_dense_reshard_transfers",
    "plan_exact_transfers",
    "receive_specs_from_shape_registry",
    "receive_specs_from_tensors",
    "register_modelexpress_loaders",
    "source_specs_from_shape_registry",
    "summarize_report_leases",
    "tensor_metadata_from_receive_specs",
]
