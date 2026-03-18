# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress Custom Model Loader for vLLM.

This loader hooks into vLLM's weight loading pipeline to perform RDMA transfers
BEFORE process_weights_after_loading() runs. This is critical for FP8 models
like DeepSeek-V3 where weight scales are transformed after loading.

Supports a three-tier loading strategy:
    1. RDMA (P2P GPU transfer via NIXL) - if a source is already serving
    2. GDS (GPUDirect Storage) - direct file-to-GPU, bypassing CPU
    3. Disk (vLLM DefaultModelLoader) - standard CPU-staged loading

Usage:
    --load-format mx  (auto-detect: RDMA -> GDS -> disk)
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .client import MxClient  # All gRPC communication goes through MxClient
from . import p2p_pb2

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
from .gds_loader import MxGdsLoader
from .gds_transfer import is_gds_available
from .nixl_transfer import NixlTransferManager, is_nixl_available
from .types import TensorDescriptor

if TYPE_CHECKING:
    from .nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.vllm_loader")

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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRANSFER_MAX_RETRIES = 120
_TRANSFER_RETRY_DELAY_SECONDS = 30

# ---------------------------------------------------------------------------
# Shared helpers used by multiple loaders
# ---------------------------------------------------------------------------


def _collect_cuda_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect all CUDA parameter tensors from a model."""
    return {
        name: param.data
        for name, param in model.named_parameters()
        if param.is_cuda
    }


def _init_nixl_manager(device_id: int, role: str) -> NixlTransferManager:
    """Create and initialize a NIXL transfer manager."""
    agent_name = f"mx-{role}-worker{device_id}-{uuid.uuid4().hex[:8]}"
    logger.debug(f"[Worker {device_id}] Initializing NIXL manager with agent_name={agent_name}")
    manager = NixlTransferManager(
        agent_name=agent_name,
        device_id=device_id,
    )
    manager.initialize()
    logger.debug(f"[Worker {device_id}] NIXL manager initialized")
    return manager


def _log_tensor_summary(
    tensors: dict[str, torch.Tensor], device_id: int, label: str
) -> None:
    """Log a summary of tensor count, size, scale_inv count, and sample checksums."""
    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    scale_count = sum(1 for n in tensors if "scale_inv" in n.lower())

    logger.info(
        f"[Worker {device_id}] {label}: {len(tensors)} tensors "
        f"({total_size / 1e9:.2f} GB), including {scale_count} scale_inv tensors"
    )

    tensor_names = list(tensors.keys())
    logger.debug(f"[Worker {device_id}] First 5 tensor names: {tensor_names[:5]}")

    for name in tensor_names[:3]:
        t = tensors[name]
        checksum = _safe_checksum(t)
        logger.debug(
            f"[Worker {device_id}] Sample tensor '{name}': "
            f"shape={t.shape}, dtype={t.dtype}, checksum={checksum}"
        )


def _publish_metadata_and_ready(
    mx_client: MxClient,
    nixl_manager: NixlTransferManager,
    tensors: dict[str, torch.Tensor],
    device_id: int,
    model_name: str,
) -> None:
    """Publish tensor metadata and ready flag to the ModelExpress server."""
    logger.info(
        f"[Worker {device_id}] Publishing {len(tensors)} tensors for model '{model_name}'"
    )

    try:
        use_contiguous = os.environ.get("MX_CONTIGUOUS_REG", "0") == "1"

        if use_contiguous:
            region_descriptors = nixl_manager.get_registered_descriptors()
            tensor_protos = [
                p2p_pb2.TensorDescriptor(
                    name=desc.name,
                    addr=desc.addr,
                    size=desc.size,
                    device_id=desc.device_id,
                    dtype=desc.dtype,
                )
                for desc in region_descriptors
            ]
            logger.info(
                f"[Worker {device_id}] Built {len(tensor_protos)} REGION descriptors "
                f"(MX_CONTIGUOUS_REG=1)"
            )
        else:
            tensor_protos = [
                p2p_pb2.TensorDescriptor(
                    name=name,
                    addr=t.data_ptr(),
                    size=t.numel() * t.element_size(),
                    device_id=device_id,
                    dtype=str(t.dtype),
                )
                for name, t in tensors.items()
            ]

        nixl_metadata = nixl_manager.nixl_metadata

        worker = p2p_pb2.WorkerMetadata(
            worker_rank=device_id,
            nixl_metadata=nixl_metadata,
            tensors=tensor_protos,
        )

        success = mx_client.publish_metadata(model_name, [worker])

        if success:
            logger.info(f"[Worker {device_id}] Published metadata to MX server")
            mx_client.update_status(
                model_name=model_name,
                worker_id=device_id,
                status=p2p_pb2.SOURCE_STATUS_READY,
            )
        else:
            raise RuntimeError(
                f"[Worker {device_id}] Failed to publish metadata to MX server"
            )

    except Exception:
        raise


@register_model_loader("mx")
class MxModelLoader(BaseModelLoader):
    """
    Auto-detecting model loader for ModelExpress P2P transfers.

    On load_model(), queries the MX server to see if an existing source
    is ready for this model. If yes, receives weights via RDMA. If no,
    attempts GDS (GPUDirect Storage) for direct file-to-GPU loading,
    falling back to standard disk loading if GDS is unavailable.
    Either way, registers tensors and publishes metadata so future nodes
    can discover this one as a source.

    Flow:
        1. initialize_model()
        2. _detect_source() - one-shot check against MX server
        3a. Source found -> load dummy weights, RDMA receive, register + publish
        3b. No source   -> try GDS, else disk load, register + publish
        4. process_weights_after_loading()
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self._nixl_manager: NixlTransferManager | None = None
        self._raw_tensors: dict[str, torch.Tensor] = {}
        self._mx_client = MxClient()
        logger.debug("MxModelLoader initialized successfully")

        # Build internal loaders for the two weight-loading strategies.
        # We only use their load_weights() methods - everything else is ours.
        import copy
        disk_config = copy.copy(load_config)
        try:
            disk_config.load_format = "auto"
        except AttributeError:
            object.__setattr__(disk_config, "load_format", "auto")
        self._disk_loader = DefaultModelLoader(disk_config)

        dummy_config = copy.copy(load_config)
        try:
            dummy_config.load_format = "dummy"
        except AttributeError:
            object.__setattr__(dummy_config, "load_format", "dummy")
        self._dummy_loader = DummyModelLoader(dummy_config)

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        """Load model, auto-detecting whether to use disk or RDMA."""
        load_start = time.perf_counter()

        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        device_id = _get_worker_rank(target_device)
        model_name = model_config.model

        logger.info(f"[Worker {device_id}] MxModelLoader starting (model={model_name})")

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                logger.info(f"[Worker {device_id}] Initializing model structure...")
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )
                logger.info(f"[Worker {device_id}] Model structure initialized")

            source_worker = self._detect_source(model_name, device_id)

            if source_worker is not None:
                logger.info(
                    f"[Worker {device_id}] Source found with {len(source_worker.tensors)} "
                    f"tensors - receiving via RDMA"
                )
                self._load_as_target(
                    model, model_config, target_device,
                    device_id, model_name, source_worker,
                )
            else:
                logger.info(
                    f"[Worker {device_id}] No RDMA source found - trying GDS then disk"
                )
                self._load_as_source(
                    model, model_config, target_device, device_id, model_name,
                )

        total_time = time.perf_counter() - load_start
        logger.info(
            f"[Worker {device_id}] MxModelLoader.load_model() COMPLETE "
            f"in {total_time:.2f}s"
        )
        return model.eval()

    # ------------------------------------------------------------------
    # Source detection
    # ------------------------------------------------------------------

    def _detect_source(self, model_name: str, device_id: int):
        """
        One-shot check for an existing ready source.

        Returns the matching WorkerMetadata proto if a ready source with
        our rank exists, or None otherwise. No retry loop - if the source
        is still warming up, this node becomes a source itself.
        """
        if not model_name:
            logger.debug(f"[Worker {device_id}] No model name available, defaulting to source")
            return None

        if not is_nixl_available():
            logger.debug(f"[Worker {device_id}] NIXL not available, defaulting to source")
            return None

        try:
            metadata_resp = self._mx_client.get_metadata(model_name)
            if not metadata_resp.found:
                logger.debug(f"[Worker {device_id}] No metadata found for model")
                return None

            ready = p2p_pb2.SOURCE_STATUS_READY
            for w in metadata_resp.workers:
                if w.worker_rank == device_id and w.status == ready and len(w.tensors) > 0:
                    logger.info(
                        f"[Worker {device_id}] Detected ready source: "
                        f"rank={w.worker_rank}, tensors={len(w.tensors)}"
                    )
                    return w

            available_ranks = [w.worker_rank for w in metadata_resp.workers]
            logger.debug(
                f"[Worker {device_id}] Source has metadata but no matching rank. "
                f"Available: {available_ranks}"
            )
            return None

        except Exception:
            logger.warning(
                f"[Worker {device_id}] MX server unavailable, skipping RDMA source detection"
            )
            return None

    # ------------------------------------------------------------------
    # Target path: receive via RDMA then register + publish
    # ------------------------------------------------------------------

    def _load_as_target(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        target_device: torch.device,
        device_id: int,
        model_name: str,
        source_worker,
    ) -> None:
        """Receive weights via RDMA from an existing source, then publish."""
        # Create dummy weights as receive buffers
        logger.info(f"[Worker {device_id}] Creating dummy weights as RDMA receive buffers...")
        self._dummy_loader.load_weights(model, model_config)

        # RDMA receive
        self._receive_from_peer(model, device_id, model_name, source_worker)

        # Register with NIXL + publish so future nodes can discover us
        self._register_and_publish(model, target_device, device_id, model_name)

        # FP8 processing
        logger.info(f"[Worker {device_id}] Processing weights (FP8 transformation)...")
        process_weights_after_loading(model, model_config, target_device)
        logger.info(f"[Worker {device_id}] Weight processing complete")

    def _receive_from_peer(
        self,
        model: nn.Module,
        device_id: int,
        model_name: str,
        source_worker,
    ) -> None:
        """Receive raw tensors via RDMA from the detected source."""
        receive_start = time.perf_counter()

        target_tensors = _collect_cuda_tensors(model)
        _log_tensor_summary(target_tensors, device_id, "Target tensors (RDMA buffers)")

        # Initialize NIXL manager and register target tensors for RDMA
        init_start = time.perf_counter()
        self._nixl_manager = _init_nixl_manager(device_id, "auto")
        nixl_init_time = time.perf_counter() - init_start
        logger.info(f"[Worker {device_id}] [TIMING] NIXL manager initialized in {nixl_init_time:.3f}s")

        reg_start = time.perf_counter()
        self._nixl_manager.register_tensors(target_tensors)
        reg_time = time.perf_counter() - reg_start
        logger.info(f"[Worker {device_id}] [TIMING] Target tensors registered in {reg_time:.3f}s")

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

        logger.info(
            f"[Worker {device_id}] Receiving {len(source_tensors)} tensors from source"
        )

        coalesce = os.environ.get("MX_CONTIGUOUS_REG", "0") == "1"

        for attempt in range(_TRANSFER_MAX_RETRIES):
            try:
                transfer_start = time.perf_counter()
                bytes_transferred, tensor_count, _ = self._nixl_manager.receive_from_source(
                    source_metadata=source_worker.nixl_metadata,
                    source_tensors=source_tensors,
                    timeout_seconds=300.0,
                    coalesce_transfers=coalesce,
                )
                transfer_time = time.perf_counter() - transfer_start

                bandwidth_gbps = (bytes_transferred * 8) / (transfer_time * 1e9) if transfer_time > 0 else 0
                logger.info(
                    f"[Worker {device_id}] [TIMING] RDMA transfer complete: "
                    f"{tensor_count} tensors, {bytes_transferred / 1e9:.2f} GB, "
                    f"{transfer_time:.3f}s, {bandwidth_gbps:.1f} Gbps"
                )

                torch.cuda.synchronize()
                break
            except Exception as transfer_err:
                if attempt < _TRANSFER_MAX_RETRIES - 1:
                    logger.warning(
                        f"[Worker {device_id}] Transfer attempt {attempt + 1} failed: "
                        f"{transfer_err}, retrying in {_TRANSFER_RETRY_DELAY_SECONDS}s..."
                    )

                    # Re-fetch metadata in case source was restarted
                    response = self._mx_client.get_metadata(model_name)
                    for w in response.workers:
                        if w.worker_rank == device_id and len(w.tensors) > 0:
                            source_worker = w
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
                                f"[Worker {device_id}] Refreshed metadata: "
                                f"{len(source_tensors)} tensors"
                            )
                            break

                    time.sleep(_TRANSFER_RETRY_DELAY_SECONDS)
                else:
                    raise RuntimeError(
                        f"Transfer failed after {_TRANSFER_MAX_RETRIES} attempts: {transfer_err}"
                    ) from transfer_err

        total_time = time.perf_counter() - receive_start
        logger.info(f"[Worker {device_id}] [TIMING] Total receive time: {total_time:.2f}s")

    # ------------------------------------------------------------------
    # Source path: GDS -> disk, then register + publish
    # ------------------------------------------------------------------

    def _load_as_source(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        target_device: torch.device,
        device_id: int,
        model_name: str,
    ) -> None:
        """Load weights via GDS or disk, then register + publish."""
        loaded_via_gds = False

        if is_gds_available():
            loaded_via_gds = self._try_gds_load(
                model, model_config, device_id
            )

        if not loaded_via_gds:
            logger.info(f"[Worker {device_id}] Loading weights from disk...")
            self._disk_loader.load_weights(model, model_config)
            logger.info(f"[Worker {device_id}] Weights loaded from disk")

        # Register with NIXL + publish
        self._register_and_publish(model, target_device, device_id, model_name)

        # FP8 processing
        logger.info(f"[Worker {device_id}] Processing weights (FP8 transformation)...")
        process_weights_after_loading(model, model_config, target_device)
        logger.info(f"[Worker {device_id}] Weight processing complete")

    def _try_gds_load(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        device_id: int,
    ) -> bool:
        """
        Attempt to load weights via GDS.

        Uses MxGdsLoader.load_iter() to yield (name, tensor) pairs, then
        feeds them to model.load_weights() so vLLM handles tensor name
        mapping (e.g. merging q/k/v into qkv_proj) correctly.

        Returns True if GDS loading succeeded, False to fall back to disk.
        """
        logger.info(f"[Worker {device_id}] GDS available, attempting GDS loading...")
        gds_loader = MxGdsLoader()
        try:
            weights_iter = gds_loader.load_iter(model_config.model)
            model.load_weights(weights_iter)
            logger.info(f"[Worker {device_id}] GDS weight loading complete")
            return True
        except Exception as e:
            logger.warning(
                f"[Worker {device_id}] GDS loading failed, falling back to disk: {e}"
            )
            return False
        finally:
            gds_loader.shutdown()

    # ------------------------------------------------------------------
    # Shared: register with NIXL and publish metadata
    # ------------------------------------------------------------------

    def _register_and_publish(
        self,
        model: nn.Module,
        device: torch.device,
        device_id: int,
        model_name: str,
    ) -> None:
        """Register tensors with NIXL and publish metadata to the MX server."""
        if not is_nixl_available():
            logger.warning(f"[Worker {device_id}] NIXL not available, skipping registration")
            return

        raw_tensors = _collect_cuda_tensors(model)
        self._raw_tensors = raw_tensors
        _log_tensor_summary(raw_tensors, device_id, "Registering tensors")

        if self._nixl_manager is None:
            self._nixl_manager = _init_nixl_manager(device_id, "auto")

        # Only register if not already registered (target path already did this)
        if not self._nixl_manager.tensor_descriptors:
            logger.debug(f"[Worker {device_id}] Registering tensors with NIXL...")
            self._nixl_manager.register_tensors(raw_tensors)
            logger.debug(f"[Worker {device_id}] Tensors registered with NIXL")

        _raw_tensor_registry[device_id] = raw_tensors
        _nixl_managers[device_id] = self._nixl_manager

        # Optional synchronized publish
        sync_publish = os.environ.get("MX_SYNC_PUBLISH", "0") == "1"
        if sync_publish:
            expected_workers = _get_expected_workers()
            logger.info(f"[Worker {device_id}] Synchronized publish, waiting for {expected_workers} workers...")
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.barrier()
                    logger.info(f"[Worker {device_id}] Barrier passed")
            except (ImportError, RuntimeError) as e:
                logger.debug(f"[Worker {device_id}] Barrier not available: {e}")

        if model_name:
            try:
                _publish_metadata_and_ready(
                    self._mx_client, self._nixl_manager, raw_tensors, device_id, model_name
                )
            except Exception:
                logger.warning(
                    f"[Worker {device_id}] MX server unavailable, "
                    f"skipping P2P registration"
                )
        else:
            logger.warning(f"[Worker {device_id}] No model name available, skipping publish")

    def download_model(self, model_config: ModelConfig) -> None:
        """Download the model so it can be loaded immediately."""
        self._disk_loader.download_model(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into an already-initialized model (standalone API)."""
        self._disk_loader.load_weights(model, model_config)

    @property
    def nixl_manager(self) -> NixlTransferManager | None:
        """Access the NIXL manager for external use."""
        return self._nixl_manager

    @property
    def raw_tensors(self) -> dict[str, torch.Tensor]:
        """Access the raw tensor registry."""
        return self._raw_tensors



# Global storage for raw tensor metadata, keyed by device_id.
# Required because vLLM's loader API doesn't expose loader instances after
# load_model() returns. Source loaders store state here so the MxClient
# (running in the same worker process) can access NIXL managers and tensors.
# Each device_id maps to exactly one loader, so there are no concurrent writers.
_raw_tensor_registry: dict[int, dict[str, torch.Tensor]] = {}
_nixl_managers: dict[int, "NixlTransferManager"] = {}
