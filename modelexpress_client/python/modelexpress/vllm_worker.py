# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelExpress worker wrapper for vLLM.

This worker ensures ModelExpress loaders are registered in spawned worker processes.
When vLLM uses multiprocess executor with spawn method, worker processes start fresh
and don't inherit registered loaders from the main process.

Usage:
    Set --worker-cls=modelexpress.vllm_worker.ModelExpressWorker
"""

from vllm.v1.worker.gpu_worker import Worker


class ModelExpressWorker(Worker):
    """Worker that registers ModelExpress loaders before model loading."""

    def __init__(self, *args, **kwargs):
        # Register loaders before parent initialization.
        # This imports modelexpress.vllm_loader which has @register_model_loader
        # decorators on the MxModelLoader class.
        from modelexpress import register_modelexpress_loaders

        register_modelexpress_loaders()
        super().__init__(*args, **kwargs)
