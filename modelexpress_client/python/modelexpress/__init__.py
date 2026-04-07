# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress - High-performance GPU-to-GPU model weight transfers.

This package provides:
- NIXL-based RDMA transfers for GPU tensors
- GPUDirect Storage (GDS) for direct file-to-GPU loading
- vLLM worker extension for serving model weights
- Custom model loaders for FP8 model support (DeepSeek-V3, etc.)
- TensorRT-LLM checkpoint transfer and engine building

Quick Start (vLLM):
    from modelexpress import register_modelexpress_loaders
    register_modelexpress_loaders()

    # vllm serve model --load-format mx
    # Auto-detects: RDMA -> GDS -> disk

Quick Start (TRT-LLM):
    from modelexpress import MxTrtllmSourcePublisher, MxTrtllmTargetLoader

    # Source: Publish checkpoint for P2P transfer
    publisher = MxTrtllmSourcePublisher(checkpoint_dir, model_name, mx_server)
    publisher.initialize()

    # Target: Receive checkpoint and build engine
    loader = MxTrtllmTargetLoader(model_name, mx_server, output_dir)
    engine_dir = loader.load()
"""

import logging

_logger = logging.getLogger(__name__)
_loaders_registered = False

from .trtllm_loader import (
    MxTrtllmSourcePublisher,
    MxTrtllmTargetLoader,
    create_trtllm_from_mx,
)


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

__all__ = [
    "GdsTransferManager",
    "HeartbeatThread",
    "MxClient",
    "MxGdsLoader",
    "register_modelexpress_loaders",
    # TRT-LLM integration
    "MxTrtllmSourcePublisher",
    "MxTrtllmTargetLoader",
    "create_trtllm_from_mx",
]
