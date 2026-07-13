# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelExpress worker wrapper for vLLM.

This worker ensures ModelExpress loaders are registered in spawned worker processes.
When vLLM uses multiprocess executor with spawn method, worker processes start fresh
and don't inherit registered loaders from the main process.

Compatibility usage:
    Set --worker-cls=modelexpress.vllm_worker.ModelExpressWorker
"""

from vllm.v1.worker.gpu_worker import Worker


class ModelExpressWorker(Worker):
    """Worker that registers ModelExpress loaders before model loading."""

    def __init__(self, *args, **kwargs):
        # Register loaders before parent initialization.
        # This imports the vLLM engine package, which registers MxModelLoader.
        from modelexpress import register_modelexpress_loaders
        from modelexpress.engines.vllm.weight_transfer import (
            register as register_mx_weight_transfer,
        )

        register_modelexpress_loaders()
        # Loader registration may happen while vLLM's transfer factory is
        # still importing and is intentionally best-effort there. Worker
        # construction is the deterministic last point before load_model()
        # asks the factory for backend="mx".
        register_mx_weight_transfer()
        super().__init__(*args, **kwargs)
