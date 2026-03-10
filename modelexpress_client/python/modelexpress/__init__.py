# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress - High-performance GPU-to-GPU model weight transfers.

This package provides:
- NIXL-based RDMA transfers for GPU tensors
- GMS-based local weight sharing via GPU Memory Service
- vLLM worker extension for serving model weights
- Custom model loaders for FP8 model support (DeepSeek-V3, etc.)

Quick Start:
    # For FP8 models (DeepSeek-V3), use custom loaders:
    from modelexpress import register_modelexpress_loaders
    register_modelexpress_loaders()

    # Source: vllm serve model --load-format mx-source
    # Target: vllm serve model --load-format mx-target

    # For GMS-based local sharing:
    from modelexpress import register_gms_loader
    register_gms_loader()

    # GMS: python -m modelexpress.gms --model <model> --device 0
    # vLLM: vllm serve <model> --load-format gms
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
    """
    global _loaders_registered
    if _loaders_registered:
        return

    # Import triggers @register_model_loader decorators on the classes
    from . import vllm_loader  # noqa: F401

    _loaders_registered = True
    _logger.debug("ModelExpress loaders registered: mx-source, mx-target")


from .client import MxClient  # noqa: F401

def register_gms_loader():
    """
    Register GMS loader with vLLM.

    Call this function before creating an LLM instance to enable:
        --load-format mx-gms-source  (loads from disk, writes to GMS)

    Note: vLLM engines use --load-format gms (from gpu_memory_service) to read.
    """
    from .gms_loader import MxGmsSourceLoader
    from vllm.model_executor.model_loader import register_model_loader

    register_model_loader("mx-gms-source")(MxGmsSourceLoader)

    import logging
    logging.getLogger("modelexpress").info("ModelExpress GMS loader registered: mx-gms-source")


def run_gms_loader() -> int:
    """Run the multi-GPU GMS weight loader CLI.

    Entry point for multi-GPU model loading with GMS + NIXL.

    Returns:
        Exit code (0 on success).
    """
    from .gms.main import main

    return main()


__all__ = [
    "MxClient",
    "register_modelexpress_loaders",
    "register_gms_loader",
    "run_gms_loader",
]
