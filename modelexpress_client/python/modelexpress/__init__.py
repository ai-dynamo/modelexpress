# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress - High-performance GPU-to-GPU model weight transfers.

This package provides:
- NIXL-based RDMA transfers for GPU tensors
- vLLM worker extension for serving model weights
- Custom model loaders for FP8 model support (DeepSeek-V3, etc.)
- GDS (GPUDirect Storage) loader for direct storage-to-GPU weight loading

Quick Start:
    # For FP8 models (DeepSeek-V3), use custom loaders:
    from modelexpress import register_modelexpress_loaders
    register_modelexpress_loaders()

    # Source: vllm serve model --load-format mx-source
    # Target: vllm serve model --load-format mx-target
    # GDS:    vllm serve model --load-format mx-gds

    # Framework-agnostic GDS loading (for sglang, etc.):
    from modelexpress import MxGdsLoader
    loader = MxGdsLoader()
    tensors = loader.load("/path/to/model")
"""

import logging

_logger = logging.getLogger(__name__)
_loaders_registered = False


def register_modelexpress_loaders():
    """
    Register ModelExpress loaders with vLLM.

    This function ensures loaders are registered exactly once. It can be called
    multiple times safely (idempotent).

    Enables:
        --load-format mx-source  (for source - loads from disk, registers raw tensors)
        --load-format mx-target  (for target - receives raw tensors via RDMA)
        --load-format mx-gds     (for GDS - loads from storage directly to GPU)
    """
    global _loaders_registered
    if _loaders_registered:
        return

    # Import triggers @register_model_loader decorators on the classes
    from . import vllm_loader  # noqa: F401
    from . import vllm_gds_loader  # noqa: F401

    _loaders_registered = True
    _logger.debug("ModelExpress loaders registered: mx-source, mx-target, mx-gds")


from .client import MxClient  # noqa: F401
from .gds_loader import MxGdsLoader  # noqa: F401

__all__ = [
    "MxClient",
    "MxGdsLoader",
    "register_modelexpress_loaders",
]
