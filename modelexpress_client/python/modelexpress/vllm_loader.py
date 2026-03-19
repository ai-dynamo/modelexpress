# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress Custom Model Loader for vLLM.

This loader hooks into vLLM's weight loading pipeline to perform RDMA transfers
BEFORE process_weights_after_loading() runs. This is critical for FP8 models
like DeepSeek-V3 where weight scales are transformed after loading.

Usage:
    --load-format mx  (auto-detect: RDMA if source exists, else disk)
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import sys
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


def _get_global_rank(device: torch.device) -> int:
    """Get the global rank of this worker across all TP and PP groups."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            logger.debug(f"Got global rank from torch.distributed: {rank}")
            return rank
    except (ImportError, RuntimeError) as e:
        logger.debug(f"Could not get global rank from torch.distributed: {e}")

    # Fallback to device index
    if hasattr(device, "index") and device.index is not None:
        logger.debug(f"Using device.index as rank: {device.index}")
        return device.index

    return 0


def _get_worker_rank(device: torch.device) -> int:
    """Get the local CUDA device ordinal (TP rank) of this worker."""
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


class SourceTransferError(Exception):
    """Raised when a failure is demonstrably from the remote source side.

    Only this exception triggers marking the source STALE. Target-local errors
    (OOM, process_weights_after_loading, warmup) are left as plain exceptions
    so they propagate without poisoning a healthy source.
    """


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


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


def _init_nixl_manager(global_rank: int, device_id: int, role: str) -> NixlTransferManager:
    """Create and initialize a NIXL transfer manager."""
    agent_name = f"mx-{role}-worker{global_rank}-{uuid.uuid4().hex[:8]}"
    logger.debug(f"[Worker {global_rank}] Initializing NIXL manager with agent_name={agent_name}")
    manager = NixlTransferManager(
        agent_name=agent_name,
        device_id=device_id,
    )
    manager.initialize()
    logger.debug(f"[Worker {global_rank}] NIXL manager initialized")
    return manager


def _log_tensor_summary(
    tensors: dict[str, torch.Tensor], global_rank: int, label: str
) -> None:
    """Log a summary of tensor count, size, scale_inv count, and sample checksums."""
    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    scale_count = sum(1 for n in tensors if "scale_inv" in n.lower())

    logger.info(
        f"[Worker {global_rank}] {label}: {len(tensors)} tensors "
        f"({total_size / 1e9:.2f} GB), including {scale_count} scale_inv tensors"
    )

    tensor_names = list(tensors.keys())
    logger.debug(f"[Worker {global_rank}] First 5 tensor names: {tensor_names[:5]}")

    for name in tensor_names[:3]:
        t = tensors[name]
        checksum = _safe_checksum(t)
        logger.debug(
            f"[Worker {global_rank}] Sample tensor '{name}': "
            f"shape={t.shape}, dtype={t.dtype}, checksum={checksum}"
        )


def _build_source_identity(
    vllm_config: VllmConfig, model_config: ModelConfig
) -> p2p_pb2.SourceIdentity:
    """Build a SourceIdentity from vLLM config objects."""
    from importlib.metadata import version as pkg_version

    try:
        mx_version = pkg_version("modelexpress")
    except Exception:
        mx_version = "0.0.0"

    parallel = vllm_config.parallel_config
    tp_size = getattr(parallel, "tensor_parallel_size", 1)
    pp_size = getattr(parallel, "pipeline_parallel_size", 1)
    ep_size = getattr(parallel, "expert_parallel_size", 0)

    # torch.dtype.__str__ returns e.g. "torch.bfloat16"; strip the prefix
    dtype = str(model_config.dtype).replace("torch.", "")
    quantization = model_config.quantization or ""

    return p2p_pb2.SourceIdentity(
        mx_version=mx_version,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=model_config.model,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        expert_parallel_size=ep_size,
        dtype=dtype,
        quantization=quantization,
    )


def _publish_metadata_and_ready(
    mx_client: MxClient,
    nixl_manager: NixlTransferManager,
    tensors: dict[str, torch.Tensor],
    global_rank: int,
    device_id: int,
    identity: "p2p_pb2.SourceIdentity",
    worker_id: str,
) -> None:
    """Publish tensor metadata and ready flag to the ModelExpress server."""
    logger.info(
        f"[Worker {global_rank}] Publishing {len(tensors)} tensors for model '{identity.model_name}'"
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
                f"[Worker {global_rank}] Built {len(tensor_protos)} REGION descriptors "
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
            worker_rank=global_rank,
            nixl_metadata=nixl_metadata,
            tensors=tensor_protos,
        )

        mx_source_id = mx_client.publish_metadata(identity, worker, worker_id)
        logger.info(
            f"[Worker {global_rank}] Published metadata to MX server "
            f"(mx_source_id={mx_source_id}, worker_id={worker_id})"
        )
        success = mx_client.update_status(
            mx_source_id=mx_source_id,
            worker_id=worker_id,
            worker_rank=global_rank,
            status=p2p_pb2.SOURCE_STATUS_READY,
        )
        if not success:
            logger.error(
                f"[Worker {global_rank}] UpdateStatus to READY failed for "
                f"model '{identity.model_name}' (mx_source_id={mx_source_id})"
            )

    except Exception as e:
        import traceback
        logger.error(f"[Worker {global_rank}] EXCEPTION publishing metadata: {e}")
        logger.error(f"[Worker {global_rank}] Traceback: {traceback.format_exc()}")
        raise


@register_model_loader("mx")
class MxModelLoader(BaseModelLoader):
    """
    Auto-detecting model loader for ModelExpress P2P transfers.

    On load_model(), queries the MX server to see if an existing source
    is ready for this model. If yes, receives weights via RDMA (like
    a target). If no, loads weights from disk (as a source). Either
    way, registers tensors and publishes metadata so future nodes can
    discover this one as a source.

    Flow:
        1. initialize_model()
        2. _detect_source() - one-shot check against MX server
        3a. Source found -> load dummy weights, RDMA receive, register + publish
        3b. No source   -> load from disk, register + publish
        4. process_weights_after_loading()
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self._nixl_manager: NixlTransferManager | None = None
        self._raw_tensors: dict[str, torch.Tensor] = {}
        self._mx_client = MxClient()
        self._worker_id = uuid.uuid4().hex[:8]
        logger.debug("MxModelLoader initialized (worker_id=%s)", self._worker_id)

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
        global_rank = _get_global_rank(target_device)
        device_id = _get_worker_rank(target_device)
        identity = _build_source_identity(vllm_config, model_config)

        logger.info(f"[Worker {global_rank}] MxModelLoader starting (model={identity.model_name})")

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                logger.info(f"[Worker {global_rank}] Initializing model structure...")
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )
                logger.info(f"[Worker {global_rank}] Model structure initialized")

            loaded_as_target = self._try_load_from_candidates(
                model, model_config, target_device, global_rank, device_id, identity
            )
            if not loaded_as_target:
                self._load_as_source(model, model_config, target_device, global_rank, device_id, identity)

        total_time = time.perf_counter() - load_start
        logger.info(
            f"[Worker {global_rank}] MxModelLoader.load_model() COMPLETE "
            f"in {total_time:.2f}s"
        )
        return model.eval()

    # ------------------------------------------------------------------
    # Source detection
    # ------------------------------------------------------------------

    def _fetch_worker_metadata(
        self,
        mx_source_id: str,
        worker_id: str,
        global_rank: int,
    ) -> p2p_pb2.WorkerMetadata | None:
        """Fetch tensor metadata for one worker.

        Returns None if the worker is not found or has no tensors.
        """
        metadata_resp = self._mx_client.get_metadata(
            mx_source_id=mx_source_id,
            worker_id=worker_id,
        )
        if not metadata_resp.found:
            logger.debug(
                f"[Worker {global_rank}] Metadata not found for worker {worker_id}, skipping"
            )
            return None
        worker = metadata_resp.worker
        if not worker.tensors:
            logger.debug(
                f"[Worker {global_rank}] Worker {worker_id} has no tensors, skipping"
            )
            return None
        return worker

    def _try_load_from_candidates(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        target_device: torch.device,
        global_rank: int,
        device_id: int,
        identity: p2p_pb2.SourceIdentity,
    ) -> bool:
        """Try RDMA load from each candidate source instance in order.

        Returns True if a load succeeded. Only marks an instance STALE when a
        SourceTransferError is raised, which indicates a proven source-side failure
        (RDMA receive, missing remote tensors). Target-local errors (OOM, weight
        processing) propagate normally without poisoning a healthy source.
        Falls back to disk and returns False if all candidates are exhausted.
        """
        candidates = self._find_source_instances(identity, global_rank)
        for instance in candidates:
            mx_source_id = instance.mx_source_id
            worker_id = instance.worker_id
            try:
                source_worker = self._fetch_worker_metadata(mx_source_id, worker_id, global_rank)
                if source_worker is None:
                    continue
                logger.info(
                    f"[Worker {global_rank}] Trying source worker {worker_id} "
                    f"({len(source_worker.tensors)} tensors)"
                )
                self._load_as_target(
                    model, model_config, target_device,
                    global_rank, device_id, identity, source_worker, mx_source_id, worker_id,
                )
                return True
            except SourceTransferError as e:
                logger.warning(
                    f"[Worker {global_rank}] Source-side failure for worker {worker_id}: {e}. "
                    f"Marking STALE and trying next. "
                    f"TODO: heartbeat mechanism will handle STALE marking automatically."
                )
                # TODO: heartbeat will mark workers STALE automatically on pod deletion
                self._mx_client.update_status(
                    mx_source_id=mx_source_id,
                    worker_id=worker_id,
                    worker_rank=global_rank,
                    status=p2p_pb2.SOURCE_STATUS_STALE,
                )
        if candidates:
            logger.warning(
                f"[Worker {global_rank}] All {len(candidates)} source workers failed, "
                f"loading from disk"
            )
        else:
            logger.info(f"[Worker {global_rank}] No source worker found - loading from disk")
        return False

    def _find_source_instances(
        self, identity: p2p_pb2.SourceIdentity, global_rank: int
    ) -> list[p2p_pb2.SourceInstanceRef]:
        """
        Return all READY source instances (shuffled for load balancing).

        Only calls ListSources — metadata is fetched on demand by the caller
        when each instance is actually tried.

        Returns an empty list if NIXL is unavailable or no sources are registered.
        """
        if not is_nixl_available():
            logger.debug(f"[Worker {global_rank}] NIXL not available, defaulting to source")
            return []

        try:
            list_resp = self._mx_client.list_sources(
                identity=identity,
                status_filter=p2p_pb2.SOURCE_STATUS_READY,
            )
            if not list_resp.instances:
                logger.debug(f"[Worker {global_rank}] No ready source instances found")
                return []

            candidates = [
                inst for inst in list_resp.instances
                if inst.worker_rank == global_rank
            ]
            random.shuffle(candidates)
            logger.info(
                f"[Worker {global_rank}] Found {len(candidates)} ready source worker(s)"
            )
            return candidates

        except Exception as e:
            logger.warning(
                f"[Worker {global_rank}] Error listing sources, falling back to disk: {e}"
            )
            return []

    # ------------------------------------------------------------------
    # Target path: receive via RDMA then register + publish
    # ------------------------------------------------------------------

    def _load_as_target(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        target_device: torch.device,
        global_rank: int,
        device_id: int,
        identity: "p2p_pb2.SourceIdentity",
        source_worker,
        mx_source_id: str,
        source_worker_id: str,
    ) -> None:
        """Receive weights via RDMA from an existing source, then publish."""
        # Create dummy weights as receive buffers
        logger.info(f"[Worker {global_rank}] Creating dummy weights as RDMA receive buffers...")
        self._dummy_loader.load_weights(model, model_config)

        # RDMA receive — raises SourceTransferError on source-side failures
        self._receive_from_peer(model, global_rank, device_id, source_worker)

        # Register with NIXL + publish so future nodes can discover us
        self._register_and_publish(model, target_device, global_rank, device_id, identity, self._worker_id)

        # FP8 processing — failures here are target-local, not source-side
        logger.info(f"[Worker {global_rank}] Processing weights (FP8 transformation)...")
        process_weights_after_loading(model, model_config, target_device)
        logger.info(f"[Worker {global_rank}] Weight processing complete")

    def _receive_from_peer(
        self,
        model: nn.Module,
        global_rank: int,
        device_id: int,
        source_worker,
    ) -> None:
        """Receive raw tensors via RDMA from the detected source.

        Raises SourceTransferError if the RDMA receive fails, indicating a
        proven source-side problem. Other exceptions propagate normally.
        """
        receive_start = time.perf_counter()

        target_tensors = _collect_cuda_tensors(model)
        _log_tensor_summary(target_tensors, global_rank, "Target tensors (RDMA buffers)")

        # Initialize NIXL manager and register target tensors for RDMA
        init_start = time.perf_counter()
        self._nixl_manager = _init_nixl_manager(global_rank, device_id, "auto")
        nixl_init_time = time.perf_counter() - init_start
        logger.info(f"[Worker {global_rank}] [TIMING] NIXL manager initialized in {nixl_init_time:.3f}s")

        reg_start = time.perf_counter()
        self._nixl_manager.register_tensors(target_tensors)
        reg_time = time.perf_counter() - reg_start
        logger.info(f"[Worker {global_rank}] [TIMING] Target tensors registered in {reg_time:.3f}s")

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
            f"[Worker {global_rank}] Receiving {len(source_tensors)} tensors from source"
        )

        coalesce = os.environ.get("MX_CONTIGUOUS_REG", "0") == "1"

        transfer_start = time.perf_counter()
        try:
            bytes_transferred, tensor_count, _ = self._nixl_manager.receive_from_source(
                source_metadata=source_worker.nixl_metadata,
                source_tensors=source_tensors,
                timeout_seconds=300.0,
                coalesce_transfers=coalesce,
            )
        except Exception as e:
            raise SourceTransferError(f"RDMA receive failed: {e}") from e
        transfer_time = time.perf_counter() - transfer_start

        bandwidth_gbps = (bytes_transferred * 8) / (transfer_time * 1e9) if transfer_time > 0 else 0
        logger.info(
            f"[Worker {global_rank}] [TIMING] RDMA transfer complete: "
            f"{tensor_count} tensors, {bytes_transferred / 1e9:.2f} GB, "
            f"{transfer_time:.3f}s, {bandwidth_gbps:.1f} Gbps"
        )

        torch.cuda.synchronize()

        total_time = time.perf_counter() - receive_start
        logger.info(f"[Worker {global_rank}] [TIMING] Total receive time: {total_time:.2f}s")

    # ------------------------------------------------------------------
    # Source path: load from disk then register + publish
    # ------------------------------------------------------------------

    def _load_as_source(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        target_device: torch.device,
        global_rank: int,
        device_id: int,
        identity: "p2p_pb2.SourceIdentity",
    ) -> None:
        """Load weights from disk, then register + publish."""
        logger.info(f"[Worker {global_rank}] Loading weights from disk...")
        self._disk_loader.load_weights(model, model_config)
        logger.info(f"[Worker {global_rank}] Weights loaded from disk")

        # Register with NIXL + publish
        self._register_and_publish(model, target_device, global_rank, device_id, identity, self._worker_id)

        # FP8 processing
        logger.info(f"[Worker {global_rank}] Processing weights (FP8 transformation)...")
        process_weights_after_loading(model, model_config, target_device)
        logger.info(f"[Worker {global_rank}] Weight processing complete")

    # ------------------------------------------------------------------
    # Shared: register with NIXL and publish metadata
    # ------------------------------------------------------------------

    def _register_and_publish(
        self,
        model: nn.Module,
        device: torch.device,
        global_rank: int,
        device_id: int,
        identity: "p2p_pb2.SourceIdentity",
        worker_id: str,
    ) -> None:
        """Register tensors with NIXL and publish metadata to the MX server."""
        if not is_nixl_available():
            logger.warning(f"[Worker {global_rank}] NIXL not available, skipping registration")
            return

        raw_tensors = _collect_cuda_tensors(model)
        self._raw_tensors = raw_tensors
        _log_tensor_summary(raw_tensors, global_rank, "Registering tensors")

        if self._nixl_manager is None:
            self._nixl_manager = _init_nixl_manager(global_rank, device_id, "auto")

        # Only register if not already registered (target path already did this)
        if not self._nixl_manager.tensor_descriptors:
            logger.debug(f"[Worker {global_rank}] Registering tensors with NIXL...")
            self._nixl_manager.register_tensors(raw_tensors)
            logger.debug(f"[Worker {global_rank}] Tensors registered with NIXL")

        _raw_tensor_registry[device_id] = raw_tensors
        _nixl_managers[device_id] = self._nixl_manager

        _publish_metadata_and_ready(
            self._mx_client, self._nixl_manager, raw_tensors, global_rank, device_id, identity, worker_id
        )

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



# Global storage for raw tensor metadata, keyed by device_id (local CUDA ordinal).
# Required because vLLM's loader API doesn't expose loader instances after
# load_model() returns. Source loaders store state here so the MxClient
# (running in the same worker process) can access NIXL managers and tensors.
# Each device_id maps to exactly one loader, so there are no concurrent writers.
_raw_tensor_registry: dict[int, dict[str, torch.Tensor]] = {}
_nixl_managers: dict[int, "NixlTransferManager"] = {}
