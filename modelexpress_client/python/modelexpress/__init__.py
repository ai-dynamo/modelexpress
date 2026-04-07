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

    # vllm serve model --load-format mx
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
        --load-format mx  (auto-detect: RDMA -> GDS -> disk)
    """
    global _loaders_registered
    if _loaders_registered:
        return

    # Import triggers @register_model_loader decorators on the classes
    from . import vllm_loader  # noqa: F401

    _loaders_registered = True
    _logger.debug("ModelExpress loader registered: mx")


from .client import MxClient  # noqa: F401
from .gds_loader import MxGdsLoader  # noqa: F401
from .gds_transfer import GdsTransferManager  # noqa: F401
from .heartbeat import HeartbeatThread  # noqa: F401
from .training_publisher import MxTrainingPublisher  # noqa: F401
from .refit_receiver import MxRefitReceiver  # noqa: F401

__all__ = [
    "GdsTransferManager",
    "HeartbeatThread",
    "MxClient",
    "MxGdsLoader",
    "MxRefitReceiver",
    "MxTrainingPublisher",
    "configure_vllm_logging",
    "register_modelexpress_loaders",
]
