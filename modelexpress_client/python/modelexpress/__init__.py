# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress - High-performance GPU-to-GPU model weight transfers.

This package provides:
- NIXL-based RDMA transfers for GPU tensors
- vLLM custom model loaders for FP8 model support (DeepSeek-V3, etc.)
- TensorRT-LLM checkpoint transfer and engine building

Quick Start (vLLM):
    from modelexpress import register_modelexpress_loaders
    register_modelexpress_loaders()
    
    # Source: vllm serve model --load-format mx-source
    # Target: vllm serve model --load-format mx-target

Quick Start (TRT-LLM):
    from modelexpress import MxTrtllmSourcePublisher, MxTrtllmTargetLoader
    
    # Source: Publish checkpoint for P2P transfer
    publisher = MxTrtllmSourcePublisher(checkpoint_dir, model_name, mx_server)
    publisher.initialize()
    
    # Target: Receive checkpoint and build engine
    loader = MxTrtllmTargetLoader(model_name, mx_server, output_dir)
    engine_dir = loader.load()
"""

from .trtllm_loader import (
    MxTrtllmSourcePublisher,
    MxTrtllmTargetLoader,
    create_trtllm_from_mx,
)


def register_modelexpress_loaders():
    """
    Register ModelExpress loaders with vLLM.
    
    Call this function before creating an LLM instance to enable:
        --load-format mx-source  (for source - loads from disk, registers raw tensors)
        --load-format mx-target  (for target - receives raw tensors via RDMA)
    """
    from .vllm_loader import (
        MxSourceModelLoader,
        MxTargetModelLoader,
    )
    from vllm.model_executor.model_loader import register_model_loader
    
    register_model_loader("mx-source")(MxSourceModelLoader)
    register_model_loader("mx-target")(MxTargetModelLoader)
    
    import logging
    logging.getLogger("modelexpress").info("ModelExpress loaders registered: mx-source, mx-target")


__all__ = [
    # vLLM integration
    "register_modelexpress_loaders",
    # TRT-LLM integration
    "MxTrtllmSourcePublisher",
    "MxTrtllmTargetLoader",
    "create_trtllm_from_mx",
]
