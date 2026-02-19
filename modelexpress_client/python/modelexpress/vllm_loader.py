# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress Custom Model Loaders for vLLM.

These loaders hook into vLLM's weight loading pipeline to perform RDMA transfers
BEFORE process_weights_after_loading() runs. This is critical for FP8 models
like DeepSeek-V3 where weight scales are transformed after loading.

Usage:
    Source: --load-format mx-source  (loads from disk, registers raw tensors)
    Target: --load-format mx-target  (receives raw tensors via RDMA)
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
import uuid
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .client import MxClient  # All gRPC communication goes through MxClient

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype
from .nixl_transfer import NixlTransferManager, is_nixl_available
from .types import TensorDescriptor

if TYPE_CHECKING:
    from .nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.vllm_loader")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)

def _safe_checksum(tensor: torch.Tensor) -> str:
    """Compute MD5 checksum of tensor, handling bfloat16 which numpy doesn't support."""
    try:
        t = tensor.cpu()
        # Convert bfloat16 to float32 for numpy compatibility
        if t.dtype == torch.bfloat16:
            t = t.float()
        return hashlib.md5(t.numpy().tobytes()).hexdigest()[:8]
    except Exception as e:
        return f"err:{e}"


def _get_expected_workers() -> int:
    """Get expected worker count from env var or auto-detect from TP world size."""
    env_workers = os.environ.get("MX_EXPECTED_WORKERS")
    if env_workers:
        return int(env_workers)

    # Auto-detect from vLLM tensor parallel world size
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
        world_size = get_tensor_model_parallel_world_size()
        logger.debug(f"Auto-detected expected workers from TP world size: {world_size}")
        return world_size
    except (ImportError, RuntimeError) as e:
        logger.debug(f"Could not get TP world size: {e}, defaulting to 1")
        return 1


def _get_worker_rank(device: torch.device) -> int:
    """Get the TP rank of this worker."""
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
        logger.debug(f"Got TP rank from vllm.distributed: {rank}")
        return rank
    except (ImportError, RuntimeError) as e:
        logger.debug(f"Could not get TP rank from vllm.distributed: {e}")

    # Fallback to device index
    if hasattr(device, "index") and device.index is not None:
        logger.debug(f"Using device.index as rank: {device.index}")
        return device.index

    return 0


@register_model_loader("mx-source")
class MxSourceModelLoader(DefaultModelLoader):
    """
    Model loader for ModelExpress SOURCE instances.

    Loads weights from disk normally, but registers raw tensors with NIXL
    BEFORE process_weights_after_loading() transforms FP8 scales.

    Flow:
        1. initialize_model() - Create model structure
        2. load_weights() - Load raw weights from disk
        3. [HOOK] Register raw tensors with NIXL for RDMA
        4. process_weights_after_loading() - Transform FP8 scales
        5. Model ready for inference AND serving weights
    """

    def __init__(self, load_config: LoadConfig):
        logger.debug("MxSourceModelLoader.__init__ called!")
        # Map mx-source to auto format internally for weight loading
        # We keep our custom loader behavior but use standard weight loading
        import copy
        modified_config = copy.copy(load_config)
        # Use object.__setattr__ if LoadConfig is frozen/immutable
        try:
            modified_config.load_format = "auto"
        except AttributeError:
            object.__setattr__(modified_config, "load_format", "auto")
        super().__init__(modified_config)
        self._nixl_manager: NixlTransferManager | None = None
        self._raw_tensors: dict[str, torch.Tensor] = {}
        self._mx_client = MxClient()
        logger.debug("MxSourceModelLoader initialized successfully")

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        """Load model with NIXL registration before weight processing."""
        logger.info("MxSourceModelLoader.load_model() STARTING")

        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        logger.debug(f"Target device: {target_device}")

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                logger.info("Initializing model structure...")
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )
                logger.info("Model structure initialized")

            logger.info("Loading weights from disk...")
            self.load_weights(model, model_config)
            logger.info("Weights loaded from disk!")

            # HOOK: Register RAW tensors before processing
            logger.info("=== HOOK: Registering raw tensors BEFORE FP8 processing ===")
            self._register_raw_tensors(model, target_device)
            logger.info("=== Raw tensor registration complete ===")

            # Now run FP8 processing (transforms weight_scale_inv -> weight_scale)
            logger.info("Processing weights (FP8 transformation)...")
            process_weights_after_loading(model, model_config, target_device)
            logger.info("Weight processing complete!")

        logger.info("MxSourceModelLoader.load_model() COMPLETE")
        return model.eval()

    def _register_raw_tensors(
        self, model: nn.Module, device: torch.device
    ) -> None:
        """
        Register raw tensors with NIXL before FP8 processing.

        This captures weight_scale_inv tensors BEFORE they are deleted
        and replaced with weight_scale.
        """
        logger.debug("_register_raw_tensors() called")

        if not is_nixl_available():
            logger.warning("NIXL not available, skipping raw tensor registration")
            return

        # Collect raw tensors (including weight_scale_inv)
        raw_tensors: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.is_cuda:
                raw_tensors[name] = param.data

        self._raw_tensors = raw_tensors
        device_id = _get_worker_rank(device)

        total_size = sum(t.numel() * t.element_size() for t in raw_tensors.values())
        scale_count = sum(1 for n in raw_tensors if "scale_inv" in n.lower())

        logger.info(
            f"[Worker {device_id}] Registering {len(raw_tensors)} raw tensors "
            f"({total_size / 1e9:.2f} GB), including {scale_count} scale_inv tensors"
        )

        # Debug: print first 5 tensor names and sample checksums
        tensor_names = list(raw_tensors.keys())
        logger.debug(f"[Worker {device_id}] First 5 tensor names: {tensor_names[:5]}")

        # Sample checksums for verification
        for name in tensor_names[:3]:
            t = raw_tensors[name]
            checksum = _safe_checksum(t)
            logger.debug(f"[Worker {device_id}] Sample tensor '{name}': shape={t.shape}, dtype={t.dtype}, checksum={checksum}")

        # Initialize NIXL manager if not already done
        if self._nixl_manager is None:
            agent_name = f"mx-source-worker{device_id}-{uuid.uuid4().hex[:8]}"
            logger.debug(f"[Worker {device_id}] Initializing NIXL manager with agent_name={agent_name}")
            self._nixl_manager = NixlTransferManager(
                agent_name=agent_name,
                device_id=device_id,
            )
            self._nixl_manager.initialize()
            logger.debug(f"[Worker {device_id}] NIXL manager initialized")

        # Register with NIXL for RDMA
        logger.debug(f"[Worker {device_id}] Registering tensors with NIXL...")
        self._nixl_manager.register_tensors(raw_tensors)
        logger.debug(f"[Worker {device_id}] Tensors registered with NIXL")

        # Store in global registry for client access
        _raw_tensor_registry[device_id] = raw_tensors
        _nixl_managers[device_id] = self._nixl_manager

        logger.info(f"[Worker {device_id}] Raw tensors stored in global registry")

        # CRITICAL: Publish metadata to ModelExpress server for targets to discover
        # This was previously unreachable due to early return bug!

        # OPTIMIZATION: Source barrier - optionally wait for all workers before publishing
        # This ensures all source workers publish at approximately the same time,
        # reducing the staggered window and allowing target sync start to be more effective
        sync_publish = os.environ.get("MX_SYNC_PUBLISH", "0") == "1"
        expected_workers = _get_expected_workers()

        if sync_publish:
            logger.info(f"[Worker {device_id}] Synchronized publish enabled, signaling ready...")
            self._wait_for_all_workers_ready(device_id, expected_workers)

        logger.info(f"[Worker {device_id}] Publishing metadata to ModelExpress server...")
        self._publish_metadata_to_server(raw_tensors, device_id)

    def _wait_for_all_workers_ready(self, device_id: int, expected_workers: int) -> None:
        """
        Wait for all source workers to be ready before publishing.

        Uses vLLM's distributed barrier if available, otherwise polls
        the ModelExpress server via MxClient.
        """
        try:
            # Try to use vLLM's distributed barrier (most efficient)
            from vllm.distributed import get_tensor_model_parallel_world_size
            import torch.distributed as dist

            world_size = get_tensor_model_parallel_world_size()
            if world_size == expected_workers and dist.is_initialized():
                logger.debug(f"[Worker {device_id}] Using torch.distributed barrier for synchronization")
                dist.barrier()
                logger.info(f"[Worker {device_id}] Barrier passed - all workers ready")
                return
        except (ImportError, RuntimeError) as e:
            logger.debug(f"[Worker {device_id}] torch.distributed barrier not available: {e}")

        # Fallback: Poll the MX server via gRPC until all workers ready
        logger.debug(f"[Worker {device_id}] Using MX server barrier (fallback)")
        model_name = os.environ.get("MODEL_NAME", "unknown")

        try:
            max_wait = 300  # 5 minute max wait for other workers
            waited = 0
            poll_interval = 2

            while waited < max_wait:
                response = self._mx_client.get_metadata(model_name)

                if response.found:
                    ready_count = len([w for w in response.workers if len(w.tensors) > 0])
                    if ready_count >= expected_workers - 1:
                        logger.info(f"[Worker {device_id}] All other workers ready ({ready_count}/{expected_workers-1}), proceeding")
                        break
                    logger.debug(f"[Worker {device_id}] Waiting for other workers: {ready_count}/{expected_workers-1} ready")

                time.sleep(poll_interval)
                waited += poll_interval

        except Exception as e:
            logger.warning(f"[Worker {device_id}] Server barrier failed: {e}, proceeding anyway")

    def _publish_metadata_to_server(
        self, tensors: dict[str, torch.Tensor], device_id: int
    ) -> None:
        """Publish tensor metadata to ModelExpress server for targets to discover."""
        from . import p2p_pb2

        model_name = os.environ.get("MODEL_NAME", "unknown")

        logger.info(f"[Worker {device_id}] Publishing {len(tensors)} tensors for model '{model_name}'")

        try:
            # Check if contiguous region registration is enabled
            use_contiguous = os.environ.get("MX_CONTIGUOUS_REG", "0") == "1"

            if use_contiguous and self._nixl_manager is not None:
                region_descriptors = self._nixl_manager.get_registered_descriptors()
                tensor_protos = []
                for desc in region_descriptors:
                    tensor_protos.append(p2p_pb2.TensorDescriptor(
                        name=desc.name,
                        addr=desc.addr,
                        size=desc.size,
                        device_id=desc.device_id,
                        dtype=desc.dtype,
                    ))
                logger.info(f"[Worker {device_id}] Built {len(tensor_protos)} REGION descriptors (MX_CONTIGUOUS_REG=1)")
            else:
                tensor_protos = []
                for name, t in tensors.items():
                    tensor_protos.append(p2p_pb2.TensorDescriptor(
                        name=name,
                        addr=t.data_ptr(),
                        size=t.numel() * t.element_size(),
                        device_id=device_id,
                        dtype=str(t.dtype),
                    ))

            nixl_metadata = self._nixl_manager.nixl_metadata if self._nixl_manager else b""

            worker = p2p_pb2.WorkerMetadata(
                worker_rank=device_id,
                nixl_metadata=nixl_metadata,
                tensors=tensor_protos,
            )

            success = self._mx_client.publish_metadata(model_name, [worker])

            if success:
                logger.info(f"[Worker {device_id}] Published metadata to MX server")

                # Publish ready flag
                metadata_hash = hashlib.md5(
                    ",".join(sorted(tensors.keys())).encode()
                ).hexdigest()

                self._mx_client.publish_ready(
                    model_name=model_name,
                    worker_id=device_id,
                    session_id=self._mx_client.session_id,
                    metadata_hash=metadata_hash,
                    nixl_ready=True,
                    stability_verified=True,
                )
            else:
                logger.error(f"[Worker {device_id}] FAILED to publish metadata")

        except Exception as e:
            import traceback
            logger.error(f"[Worker {device_id}] EXCEPTION publishing metadata: {e}")
            logger.error(f"[Worker {device_id}] Traceback: {traceback.format_exc()}")
            raise

    @property
    def nixl_manager(self) -> NixlTransferManager | None:
        """Access the NIXL manager for external use."""
        return self._nixl_manager

    @property
    def raw_tensors(self) -> dict[str, torch.Tensor]:
        """Access the raw tensor registry."""
        return self._raw_tensors


@register_model_loader("mx-target")
class MxTargetModelLoader(DummyModelLoader):
    """
    Model loader for ModelExpress TARGET instances.

    Initializes dummy weights, receives raw tensors via RDMA,
    THEN runs process_weights_after_loading() to transform FP8 scales.

    Flow:
        1. initialize_model() - Create model structure
        2. load_weights() - Create dummy weights (with weight_scale_inv)
        3. [HOOK] Receive raw tensors via RDMA from source
        4. process_weights_after_loading() - Transform FP8 scales (same as source)
        5. Model ready for inference with transferred weights
    """

    def __init__(self, load_config: LoadConfig):
        logger.debug("MxTargetModelLoader.__init__ called!")
        # Map mx-target to dummy format internally for weight initialization
        import copy
        modified_config = copy.copy(load_config)
        try:
            modified_config.load_format = "dummy"
        except AttributeError:
            object.__setattr__(modified_config, "load_format", "dummy")
        super().__init__(modified_config)
        self._nixl_manager: NixlTransferManager | None = None
        self._transfer_timeout: float = 300.0  # 5 minute timeout
        self._mx_client = MxClient()
        logger.debug("MxTargetModelLoader initialized successfully")

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        """Load model with RDMA transfer before weight processing."""
        load_start = time.perf_counter()
        logger.info("=" * 60)
        logger.info("MxTargetModelLoader.load_model() STARTING")
        logger.info("=" * 60)

        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        logger.debug(f"Target device: {target_device}")

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                t0 = time.perf_counter()
                logger.info("[TIMING] Initializing model structure...")
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )
                logger.info(f"[TIMING] Model structure initialized in {time.perf_counter() - t0:.2f}s")

            t0 = time.perf_counter()
            logger.info("[TIMING] Creating dummy weights...")
            self.load_weights(model, model_config)
            logger.info(f"[TIMING] Dummy weights created in {time.perf_counter() - t0:.2f}s")

            # HOOK: Receive RAW tensors via RDMA before processing
            t0 = time.perf_counter()
            logger.info("=" * 60)
            logger.info("[TIMING] === HOOK: Receiving raw tensors via RDMA ===")
            logger.info("=" * 60)
            self._receive_raw_tensors(model, target_device)
            rdma_time = time.perf_counter() - t0
            logger.info(f"[TIMING] === RDMA reception complete in {rdma_time:.2f}s ===")

            # Now run FP8 processing (transforms weight_scale_inv -> weight_scale)
            # This will produce IDENTICAL results to source since we have same raw data
            t0 = time.perf_counter()
            logger.info("[TIMING] Processing weights (FP8 transformation)...")
            process_weights_after_loading(model, model_config, target_device)
            logger.info(f"[TIMING] Weight processing complete in {time.perf_counter() - t0:.2f}s")

        total_time = time.perf_counter() - load_start
        logger.info("=" * 60)
        logger.info("[TIMING] MxTargetModelLoader.load_model() COMPLETE")
        logger.info(f"[TIMING] Total load time: {total_time:.2f}s")
        logger.info("=" * 60)
        return model.eval()

    def _receive_raw_tensors(
        self, model: nn.Module, device: torch.device
    ) -> None:
        """
        Receive raw tensors via RDMA from source.

        The source has registered raw tensors (including weight_scale_inv).
        We receive them into our dummy tensors, then let vLLM process them.
        """
        receive_start = time.perf_counter()
        logger.debug("_receive_raw_tensors() called")

        if not is_nixl_available():
            logger.warning("NIXL not available, skipping RDMA transfer")
            return

        # Get source info from environment
        model_name = os.environ.get("MODEL_NAME", "")

        logger.debug(f"Model: {model_name}")

        if not model_name:
            logger.warning("MODEL_NAME not set, skipping transfer")
            return

        # Collect target tensors (these have dummy data)
        t0 = time.perf_counter()
        target_tensors: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.is_cuda:
                target_tensors[name] = param.data

        device_id = _get_worker_rank(device)
        scale_count = sum(1 for n in target_tensors if "scale_inv" in n.lower())
        total_size = sum(t.numel() * t.element_size() for t in target_tensors.values())
        logger.debug(f"[Worker {device_id}] [TIMING] Collected {len(target_tensors)} tensors in {time.perf_counter() - t0:.3f}s")

        logger.info(
            f"[Worker {device_id}] Target has {len(target_tensors)} tensors "
            f"({total_size / 1e9:.2f} GB), including {scale_count} scale_inv tensors"
        )

        # Debug: print first 5 tensor names and checksums BEFORE transfer
        tensor_names = list(target_tensors.keys())
        logger.debug(f"[Worker {device_id}] First 5 tensor names: {tensor_names[:5]}")

        logger.debug(f"[Worker {device_id}] Checksums BEFORE transfer:")
        pre_checksums = {}
        for name in tensor_names[:3]:
            t = target_tensors[name]
            checksum = _safe_checksum(t)
            pre_checksums[name] = checksum
            logger.debug(f"[Worker {device_id}]   '{name}': shape={t.shape}, dtype={t.dtype}, checksum={checksum}")

        # Initialize NIXL manager
        t0 = time.perf_counter()
        agent_name = f"mx-target-worker{device_id}-{uuid.uuid4().hex[:8]}"
        logger.debug(f"[Worker {device_id}] [TIMING] Initializing NIXL manager with agent_name={agent_name}")
        self._nixl_manager = NixlTransferManager(
            agent_name=agent_name,
            device_id=device_id,
        )
        self._nixl_manager.initialize()
        nixl_init_time = time.perf_counter() - t0
        logger.info(f"[Worker {device_id}] [TIMING] NIXL manager initialized in {nixl_init_time:.3f}s")

        t0 = time.perf_counter()
        logger.debug(f"[Worker {device_id}] [TIMING] Registering target tensors with NIXL...")
        self._nixl_manager.register_tensors(target_tensors)
        reg_time = time.perf_counter() - t0
        logger.info(f"[Worker {device_id}] [TIMING] Target tensors registered in {reg_time:.3f}s")

        # COORDINATION: Wait for NIXL ready flag before initiating transfer
        # The nixl_ready flag is only set AFTER:
        # 1. vLLM health endpoint returns 200
        # 2. 30s grace period for system stabilization
        # 3. Successful test inference proves stability
        logger.info(f"[Worker {device_id}] Waiting for source NIXL ready (includes stability verification)...")
        source_ready, cached_session_id, _cached_metadata_hash = self._mx_client.wait_for_ready(
            model_name=model_name,
            worker_id=device_id,
            timeout_seconds=7200,  # 2 hour timeout (matches source warmup)
            poll_interval=10,
        )

        if not source_ready:
            raise RuntimeError(
                f"[Worker {device_id}] Source NIXL never became ready after 2h timeout. "
                "Cannot proceed — model would serve with dummy/random weights."
            )

        logger.info(f"[Worker {device_id}] Source NIXL ready (stability verified), proceeding with transfer")

        # Connect to ModelExpress server and find source via MxClient
        # Retry with backoff - source takes 20-30 min to load DeepSeek-V3
        max_wait_time = 3600  # 1 hour max wait
        retry_interval = 30  # 30 seconds between retries
        total_waited = 0
        wait_start = time.perf_counter()

        # OPTIMIZATION: Synchronized start - wait for ALL source workers before transferring
        # This ensures all expected target workers start their transfers simultaneously,
        # maximizing RDMA parallelism and achieving closer to theoretical bandwidth
        sync_start = os.environ.get("MX_SYNC_START", "1") == "1"
        expected_workers = _get_expected_workers()

        if sync_start:
            logger.info(f"[Worker {device_id}] [TIMING] Synchronized start enabled, waiting for all {expected_workers} source workers")

        try:
            response = None
            source_worker = None
            all_workers_ready = False

            while total_waited < max_wait_time:
                logger.debug(f"[Worker {device_id}] Querying for source model: {model_name}...")
                response = self._mx_client.get_metadata(model_name)

                if response.found and len(response.workers) > 0:
                    available_ranks = sorted([w.worker_rank for w in response.workers])
                    workers_with_tensors = sum(1 for w in response.workers if len(w.tensors) > 0)
                    logger.debug(f"[Worker {device_id}] Response: found={response.found}, workers={len(response.workers)}/{expected_workers}, ranks={available_ranks}")

                    # OPTIMIZATION: Check if ALL expected workers are ready (synchronized start)
                    if sync_start and workers_with_tensors < expected_workers:
                        logger.info(f"[Worker {device_id}] Waiting for all workers: {workers_with_tensors}/{expected_workers} ready")
                        time.sleep(retry_interval)
                        total_waited += retry_interval
                        continue

                    if sync_start and workers_with_tensors >= expected_workers and not all_workers_ready:
                        logger.info(f"[Worker {device_id}] ALL {expected_workers} source workers are ready! Starting synchronized transfer.")
                        all_workers_ready = True

                    # Find source worker for OUR specific rank
                    for w in response.workers:
                        if w.worker_rank == device_id and len(w.tensors) > 0:
                            source_worker = w
                            break

                    if source_worker:
                        logger.info(f"[Worker {device_id}] Found matching source worker for rank {device_id} with {len(source_worker.tensors)} tensors!")
                        break
                    else:
                        logger.info(f"[Worker {device_id}] Source has {len(response.workers)} workers but rank {device_id} not ready yet. Available: {available_ranks}")
                else:
                    logger.info(f"[Worker {device_id}] No source found yet (found={response.found if response else 'N/A'}), waiting...")

                time.sleep(retry_interval)
                total_waited += retry_interval
                if total_waited % 60 == 0:
                    logger.info(f"[Worker {device_id}] Waited {total_waited}s for source rank {device_id} (max {max_wait_time}s)...")

            if not source_worker:
                available_ranks = [w.worker_rank for w in response.workers] if response and response.workers else []
                logger.error(f"[Worker {device_id}] ERROR: No source worker found for rank {device_id} after {total_waited}s. Available: {available_ranks}")
                return

            wait_time = time.perf_counter() - wait_start
            logger.info(f"[Worker {device_id}] [TIMING] Found source worker after waiting {wait_time:.2f}s")
            logger.info(f"[Worker {device_id}] Found source worker for rank {device_id} with {len(source_worker.tensors)} tensors")

            # Build source tensor descriptors
            source_tensors = [
                TensorDescriptor(
                    name=t.name,
                    addr=t.addr,
                    size=t.size,
                    device_id=t.device_id,
                    dtype=t.dtype,
                )
                for t in source_worker.tensors
            ]

            logger.info(f"[Worker {device_id}] Receiving {len(source_tensors)} tensors from source worker {device_id}")
            logger.debug(f"[Worker {device_id}] First 3 source tensor names: {[t.name for t in source_tensors[:3]]}")

            # Perform RDMA transfer with retry for transient failures
            transfer_retries = 120  # 120 * 30s = 60 min max wait
            transfer_retry_delay = 30
            bytes_transferred = 0
            tensor_count = 0

            for attempt in range(transfer_retries):
                try:
                    logger.info(f"[Worker {device_id}] [TIMING] Starting RDMA transfer attempt {attempt + 1}...")
                    transfer_start = time.perf_counter()
                    # MX_CONTIGUOUS_REG=1 enables contiguous region registration on BOTH
                    # source and target. This allows transfer-time coalescing to work.
                    coalesce = os.environ.get("MX_CONTIGUOUS_REG", "0") == "1"
                    logger.debug(f"[Worker {device_id}] Coalesce transfers: {coalesce} (MX_CONTIGUOUS_REG={os.environ.get('MX_CONTIGUOUS_REG', 'not set')})")
                    bytes_transferred, tensor_count, _ = self._nixl_manager.receive_from_source(
                        source_metadata=source_worker.nixl_metadata,
                        source_tensors=source_tensors,
                        timeout_seconds=self._transfer_timeout,
                        coalesce_transfers=coalesce,
                    )
                    transfer_time = time.perf_counter() - transfer_start

                    bandwidth_gbps = (bytes_transferred * 8) / (transfer_time * 1e9) if transfer_time > 0 else 0
                    logger.info("=" * 60)
                    logger.info(
                        f"[Worker {device_id}] [TIMING] RDMA TRANSFER COMPLETE:"
                    )
                    logger.info(f"[Worker {device_id}] [TIMING]   Tensors: {tensor_count}")
                    logger.info(f"[Worker {device_id}] [TIMING]   Data: {bytes_transferred / 1e9:.2f} GB")
                    logger.info(f"[Worker {device_id}] [TIMING]   Time: {transfer_time:.3f}s")
                    logger.info(f"[Worker {device_id}] [TIMING]   Bandwidth: {bandwidth_gbps:.1f} Gbps")
                    logger.info("=" * 60)

                    # Sync CUDA to ensure transfer is visible
                    torch.cuda.synchronize()
                    logger.debug(f"[Worker {device_id}] CUDA synchronized")

                    # Verify checksums AFTER transfer
                    logger.debug(f"[Worker {device_id}] Checksums AFTER transfer:")
                    for name in tensor_names[:3]:
                        t = target_tensors[name]
                        checksum = _safe_checksum(t)
                        changed = "CHANGED" if checksum != pre_checksums.get(name) else "UNCHANGED"
                        logger.debug(f"[Worker {device_id}]   '{name}': checksum={checksum} ({changed})")

                    break  # Success
                except Exception as transfer_err:
                    if attempt < transfer_retries - 1:
                        logger.warning(
                            f"[Worker {device_id}] Transfer attempt {attempt + 1} failed: {transfer_err}, "
                            f"retrying in {transfer_retry_delay}s..."
                        )

                        # Check if source restarted (stale metadata issue)
                        session_changed, new_session_id = self._mx_client.check_session_changed(
                            model_name=model_name,
                            worker_id=device_id,
                            cached_session_id=cached_session_id,
                        )

                        if session_changed:
                            logger.warning(
                                f"[Worker {device_id}] Source restarted detected! Re-fetching metadata..."
                            )
                            cached_session_id = new_session_id

                            # Re-fetch metadata from MX server
                            response = self._mx_client.get_metadata(model_name)

                            # Find updated source worker
                            for w in response.workers:
                                if w.worker_rank == device_id and len(w.tensors) > 0:
                                    source_worker = w
                                    break

                            if source_worker:
                                # Rebuild source tensor descriptors with fresh metadata
                                source_tensors = [
                                    TensorDescriptor(
                                        name=t.name,
                                        addr=t.addr,
                                        size=t.size,
                                        device_id=t.device_id,
                                        dtype=t.dtype,
                                    )
                                    for t in source_worker.tensors
                                ]
                                logger.info(
                                    f"[Worker {device_id}] Refreshed metadata: {len(source_tensors)} tensors from new session"
                                )
                            else:
                                logger.warning(
                                    f"[Worker {device_id}] Could not find source worker after restart, will retry"
                                )

                        time.sleep(transfer_retry_delay)
                    else:
                        logger.error(f"[Worker {device_id}] Transfer failed after {transfer_retries} attempts: {transfer_err}")
                        raise RuntimeError(
                            f"Transfer failed after {transfer_retries} attempts: {transfer_err}"
                        ) from transfer_err

            # Final timing summary
            total_receive_time = time.perf_counter() - receive_start
            logger.info("=" * 60)
            logger.info(f"[Worker {device_id}] [TIMING] _receive_raw_tensors SUMMARY:")
            logger.info(f"[Worker {device_id}] [TIMING]   Total time: {total_receive_time:.2f}s")
            logger.info(f"[Worker {device_id}] [TIMING]   NIXL init: {nixl_init_time:.3f}s")
            logger.info(f"[Worker {device_id}] [TIMING]   Tensor reg: {reg_time:.3f}s")
            logger.info(f"[Worker {device_id}] [TIMING]   Wait for source: {wait_time:.2f}s")
            logger.info(f"[Worker {device_id}] [TIMING]   RDMA transfer: {transfer_time:.3f}s")
            logger.info("=" * 60)

        except Exception as e:
            import traceback
            logger.error(f"[Worker {device_id}] EXCEPTION receiving weights: {e}")
            logger.error(f"[Worker {device_id}] Traceback: {traceback.format_exc()}")
            raise

    @property
    def nixl_manager(self) -> NixlTransferManager | None:
        """Access the NIXL manager for external use."""
        return self._nixl_manager


# Global storage for raw tensor metadata, keyed by device_id.
# Required because vLLM's loader API doesn't expose loader instances after
# load_model() returns. Source loaders store state here so the MxClient
# (running in the same worker process) can access NIXL managers and tensors.
# Each device_id maps to exactly one loader, so there are no concurrent writers.
_raw_tensor_registry: dict[int, dict[str, torch.Tensor]] = {}
_nixl_managers: dict[int, "NixlTransferManager"] = {}
