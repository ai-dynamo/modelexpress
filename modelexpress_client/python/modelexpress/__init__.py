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
    # Auto-detects: RDMA -> InstantTensor -> ModelStreamer -> GDS -> disk
"""

import logging

from . import envs

_logger = logging.getLogger(__name__)
_loaders_registered = False


def _configure_engine_logging(
    engine_logger_name: str,
    namespaces: tuple[str, ...] = ("modelexpress",),
) -> None:
    """Attach an engine's Python log handlers to ModelExpress namespaces."""
    engine_logger = logging.getLogger(engine_logger_name)
    mx_level = envs.MODEL_EXPRESS_LOG_LEVEL
    for namespace in namespaces:
        namespace_logger = logging.getLogger(namespace)
        if namespace_logger.handlers:
            continue
        for handler in engine_logger.handlers:
            namespace_logger.addHandler(handler)
        if mx_level and hasattr(logging, mx_level):
            namespace_logger.setLevel(getattr(logging, mx_level))
        elif engine_logger.level != logging.NOTSET:
            namespace_logger.setLevel(engine_logger.level)


def configure_vllm_logging() -> None:
    """Ensure modelexpress loggers are visible in vLLM worker subprocesses.

    vLLM only attaches log handlers to the "vllm" namespace. Without this,
    all ModelExpress output is silently dropped in EngineCore worker processes.
    Copies vLLM's handlers onto both the "modelexpress" and "modelexpress_rl"
    parent loggers so every child inherits them via propagation. Idempotent.
    """
    _configure_engine_logging("vllm", ("modelexpress", "modelexpress_rl"))


def configure_trtllm_logging() -> None:
    """Ensure modelexpress loggers are visible in TRT-LLM worker processes."""
    _configure_engine_logging("TRT-LLM")


def register_modelexpress():
    """
    Register ModelExpress integrations with vLLM.

    This function ensures loaders are registered exactly once. It can be called
    multiple times safely (idempotent).

    Enables:
        --load-format modelexpress  (auto-detect: RDMA -> InstantTensor -> ModelStreamer -> GDS -> disk)
        --load-format mx            (backward-compatible alias)
        --weight-transfer-config '{"backend":"modelexpress"}'
    """
    global _loaders_registered
    if _loaders_registered:
        return

    from .engines.vllm import register_modelexpress_loaders as register_vllm_loaders

    register_vllm_loaders()

    _loaders_registered = True
    _logger.debug("ModelExpress vLLM integrations registered")


def register_modelexpress_loaders():
    """Backward-compatible alias for :func:`register_modelexpress`."""
    register_modelexpress()


from .client import MxClient  # noqa: F401
from .gds_loader import MxGdsLoader  # noqa: F401
from .gds_transfer import GdsTransferManager  # noqa: F401
from .metadata.publisher import PublisherThread  # noqa: F401
from .model_client import ModelCacheClient  # noqa: F401

__all__ = [
    "GdsTransferManager",
    "ModelCacheClient",
    "MxClient",
    "MxGdsLoader",
    "PublisherThread",
    "configure_trtllm_logging",
    "configure_vllm_logging",
    "register_modelexpress",
    "register_modelexpress_loaders",
]
