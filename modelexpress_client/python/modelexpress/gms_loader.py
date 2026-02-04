# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS source loader for Modelexpress sidecar.

Loads model weights from disk and writes them to GMS for cross-process sharing.
vLLM engines can then connect to GMS in RO mode and import the weights.

This loader is an alternative to the NIXL-based MxSourceModelLoader when using
GPU Memory Service for local weight sharing instead of RDMA transfers.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
    from torch.cuda.memory import MemPool

logger = logging.getLogger(__name__)


class MxGmsSourceLoader(DefaultModelLoader):
    """Loads model from disk and writes tensors to GMS for cross-process sharing.

    This loader:
    1. Connects to GMS as RW (write mode)
    2. Allocates model tensors via GMS memory pool
    3. Loads weights from disk into GMS memory
    4. Registers tensor metadata in GMS
    5. Commits for cross-process visibility

    After commit, vLLM engines can connect to GMS in RO mode and import weights
    using the standard GMSModelLoader with --load-format gms.

    Usage:
        # Start GMS server
        python -m gpu_memory_service --device 0

        # Run sidecar to load model into GMS
        python -m modelexpress.gms_sidecar --model <model> --device 0

        # vLLM connects and imports weights (no disk loading)
        vllm serve <model> --load-format gms
    """

    def __init__(self, load_config: LoadConfig) -> None:
        """Initialize the loader with modified config.

        Args:
            load_config: vLLM load configuration. The load_format will be
                         changed to "auto" for actual weight loading.
        """
        logger.debug("MxGmsSourceLoader.__init__ called")
        # Copy and modify load_format to "auto" for DefaultModelLoader
        modified_config = copy.copy(load_config)
        try:
            modified_config.load_format = "auto"
        except AttributeError:
            object.__setattr__(modified_config, "load_format", "auto")
        super().__init__(modified_config)
        self._gms_client: GMSClientMemoryManager | None = None
        self._pool: MemPool | None = None
        logger.debug("MxGmsSourceLoader initialized successfully")

    @property
    def gms_client(self) -> "GMSClientMemoryManager | None":
        """Get the GMS client memory manager."""
        return self._gms_client

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
    ) -> nn.Module:
        """Load model from disk and publish weights to GMS.

        Args:
            vllm_config: vLLM configuration.
            model_config: Model configuration.

        Returns:
            The loaded model in eval mode.

        Raises:
            RuntimeError: If GMS commit fails.
        """
        # Import GMS modules here to allow graceful failure if not installed
        from gpu_memory_service import get_or_create_gms_client_memory_manager
        from gpu_memory_service.client.torch.module import register_module_tensors
        from gpu_memory_service.common.types import RequestedLockType
        from gpu_memory_service.common.utils import get_socket_path
        from torch.cuda.memory import use_mem_pool

        logger.info("[MxGMS] load_model() starting")

        # Get device index from config, defaulting to current device
        device_index = torch.cuda.current_device()
        load_config = vllm_config.load_config
        if load_config.device is not None:
            dev = torch.device(load_config.device)
            if dev.index is not None:
                device_index = dev.index

        # Create target device with explicit index (required by use_mem_pool)
        target_device = torch.device("cuda", device_index)
        socket_path = get_socket_path(device_index)

        logger.info("[MxGMS] Target device: %s (index=%d)", target_device, device_index)

        # Connect to GMS as RW
        gms_client, pool = get_or_create_gms_client_memory_manager(
            socket_path,
            device_index,
            mode=RequestedLockType.RW,
            tag="weights",
        )
        self._gms_client = gms_client
        self._pool = pool

        logger.info("[MxGMS] Connected to GMS as RW (device=%d)", device_index)

        # Clear any existing allocations
        gms_client.clear_all()

        # Initialize parallel state for standalone execution
        self._init_parallel_state(vllm_config)

        # Allocate model tensors using GMS memory pool
        with set_default_torch_dtype(model_config.dtype):
            with use_mem_pool(pool, device=target_device):
                with target_device:
                    logger.info("[MxGMS] Initializing model structure...")
                    model = initialize_model(
                        vllm_config=vllm_config, model_config=model_config
                    )
                    logger.info("[MxGMS] Model structure initialized")

                logger.info("[MxGMS] Loading weights from disk...")
                self.load_weights(model, model_config)
                logger.info("[MxGMS] Weights loaded from disk")

                logger.info("[MxGMS] Processing weights...")
                process_weights_after_loading(model, model_config, target_device)
                torch.cuda.empty_cache()
                logger.info("[MxGMS] Weight processing complete")

        # Register tensor metadata in GMS
        register_module_tensors(gms_client, model)

        total_bytes = gms_client.total_bytes
        logger.info(
            "[MxGMS] Registered %.2f GiB of tensors",
            total_bytes / (1 << 30),
        )

        # Ensure all GPU writes are finished before commit
        torch.cuda.synchronize()

        if not gms_client.commit():
            raise RuntimeError("GMS commit failed")

        gms_client.switch_to_read()

        logger.info(
            "[MxGMS] Published %.2f GiB (%d mappings)",
            total_bytes / (1 << 30),
            len(gms_client.mappings),
        )

        return model.eval()

    def close(self) -> None:
        """Close the GMS connection."""
        if self._gms_client is not None:
            self._gms_client.close()
            self._gms_client = None
            self._pool = None
            logger.info("[MxGMS] GMS connection closed")

    def _init_parallel_state(self, vllm_config: VllmConfig) -> None:
        """Initialize vLLM parallel state for standalone execution.

        This is required because initialize_model() expects the distributed
        environment to be set up, which normally happens in vLLM workers.
        """
        from vllm.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        parallel_config = vllm_config.parallel_config

        # Check if already initialized
        try:
            from vllm.distributed import get_tensor_model_parallel_world_size
            get_tensor_model_parallel_world_size()
            logger.debug("[MxGMS] Parallel state already initialized")
            return
        except AssertionError:
            pass  # Not initialized, continue

        logger.info("[MxGMS] Initializing parallel state (TP=%d, PP=%d)",
                    parallel_config.tensor_parallel_size,
                    parallel_config.pipeline_parallel_size)

        # Initialize distributed environment for single process
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:0",
        )

        # Initialize model parallel groups
        initialize_model_parallel(
            tensor_model_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_model_parallel_size=parallel_config.pipeline_parallel_size,
        )
