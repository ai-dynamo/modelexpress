# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress Custom Model Loader for vLLM.

This loader hooks into vLLM's weight loading pipeline to perform RDMA transfers
of fully-processed model tensors. Registration happens AFTER
process_weights_after_loading() so that all final tensors (parameters, buffers,
and bare tensor attributes like FP8 scales) are captured and transferred.

Usage:
    --load-format mx  (auto-detect: RDMA if source exists, else disk)
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
logger.setLevel(logging.INFO)
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRANSFER_MAX_RETRIES = 120
_TRANSFER_RETRY_DELAY_SECONDS = 30

# ---------------------------------------------------------------------------
# Shared helpers used by multiple loaders
# ---------------------------------------------------------------------------



def _iter_module_tensors(
    module: nn.Module,
    prefix: str = "",
) -> list[tuple[str, torch.Tensor, str]]:
    """Iterate over all CUDA tensors in a module tree.

    Discovers three categories of tensors:
    - Parameters (registered via nn.Module parameter system)
    - Buffers (registered via register_buffer)
    - Tensor attributes (bare tensors attached directly, e.g. FP8 scales)

    This is more thorough than named_parameters() which only finds parameters.
    Post-processing steps like process_weights_after_loading() often create
    new tensors as buffers or bare attributes that named_parameters() misses.

    Args:
        module: The nn.Module to iterate.
        prefix: Prefix for qualified names (used in recursion).

    Returns:
        List of (qualified_name, tensor, tensor_type) tuples for each CUDA tensor.
    """
    results: list[tuple[str, torch.Tensor, str]] = []

    for name, param in module._parameters.items():
        if param is not None and param.is_cuda:
            qualified = f"{prefix}{name}" if prefix else name
            results.append((qualified, param, "parameter"))

    for name, buf in module._buffers.items():
        if buf is not None and buf.is_cuda:
            qualified = f"{prefix}{name}" if prefix else name
            results.append((qualified, buf, "buffer"))

    skip = (
        set(module._parameters.keys())
        | set(module._buffers.keys())
        | set(module._modules.keys())
    )
    for attr_name in dir(module):
        if attr_name in skip or attr_name.startswith("__"):
            continue
        try:
            attr_val = getattr(module, attr_name, None)
        except Exception:
            continue

        if torch.is_tensor(attr_val) and attr_val.is_cuda:
            qualified = f"{prefix}{attr_name}" if prefix else attr_name
            results.append((qualified, attr_val, "tensor_attr"))
        elif isinstance(attr_val, (list, tuple)) and attr_val:
            if all(torch.is_tensor(x) and x.is_cuda for x in attr_val):
                for i, x in enumerate(attr_val):
                    qualified = (
                        f"{prefix}{attr_name}.{i}" if prefix else f"{attr_name}.{i}"
                    )
                    results.append((qualified, x, "tensor_attr"))

    for name, submodule in module._modules.items():
        if submodule is not None:
            subprefix = f"{prefix}{name}." if prefix else f"{name}."
            results.extend(_iter_module_tensors(submodule, subprefix))

    return results


def _collect_module_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect all contiguous CUDA tensors from a module tree into a flat dict.

    Uses _iter_module_tensors to find parameters, buffers, and tensor
    attributes, then returns them as a name -> tensor mapping suitable
    for NIXL registration.

    Non-contiguous tensors (e.g. transposed views like W_UK_T) are skipped
    because they are views over contiguous tensors that are already in the
    module tree. Transferring the underlying contiguous tensor automatically
    updates the view.
    """
    tensors: dict[str, torch.Tensor] = {}
    skipped = 0
    for name, tensor, _tensor_type in _iter_module_tensors(model):
        t = tensor.data if hasattr(tensor, "data") else tensor

        if not t.is_contiguous():
            logger.debug(f"Skipping non-contiguous tensor '{name}' (view of another tensor)")
            skipped += 1
            continue

        tensors[name] = t

    if skipped:
        logger.info(
            f"Skipped {skipped} non-contiguous tensors (views of contiguous tensors already registered)"
        )
    return tensors


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

    except Exception as e:
        import traceback
        logger.error(f"[Worker {device_id}] EXCEPTION publishing metadata: {e}")
        logger.error(f"[Worker {device_id}] Traceback: {traceback.format_exc()}")
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
        3a. Source found -> load dummy weights, RDMA receive
        3b. No source   -> load from disk
        4. process_weights_after_loading()
        5. Register ALL final tensors with NIXL + publish
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self._nixl_manager: NixlTransferManager | None = None
        self._tensors: dict[str, torch.Tensor] = {}
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
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )

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
                    f"[Worker {device_id}] No source found - loading from disk"
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

        except Exception as e:
            logger.warning(
                f"[Worker {device_id}] Error detecting source, falling back to disk: {e}"
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
        """Receive fully-processed weights via RDMA from an existing source."""
        # Create dummy weights as receive buffers
        self._dummy_loader.load_weights(model, model_config)

        # Process dummy weights to establish final tensor layout
        process_weights_after_loading(model, model_config, target_device)

        # RDMA receive (fully-processed tensors, no post-processing needed)
        self._receive_from_peer(model, device_id, model_name, source_worker)

        # Register with NIXL + publish so future nodes can discover us
        self._register_and_publish(model, target_device, device_id, model_name)

    def _receive_from_peer(
        self,
        model: nn.Module,
        device_id: int,
        model_name: str,
        source_worker,
    ) -> None:
        """Receive fully-processed tensors via RDMA from the detected source."""
        receive_start = time.perf_counter()

        target_tensors = _collect_module_tensors(model)
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
    # Source path: load from disk then register + publish
    # ------------------------------------------------------------------

    def _load_as_source(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        target_device: torch.device,
        device_id: int,
        model_name: str,
    ) -> None:
        """Load weights from disk, process, then register + publish."""
        self._disk_loader.load_weights(model, model_config)

        # Process weights FIRST, then register final tensors
        process_weights_after_loading(model, model_config, target_device)

        # Register with NIXL + publish
        self._register_and_publish(model, target_device, device_id, model_name)

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

        tensors = _collect_module_tensors(model)
        self._tensors = tensors
        _log_tensor_summary(tensors, device_id, "Registering tensors")

        if self._nixl_manager is None:
            self._nixl_manager = _init_nixl_manager(device_id, "auto")

        # Only register if not already registered (target path already did this)
        if not self._nixl_manager.tensor_descriptors:
            logger.debug(f"[Worker {device_id}] Registering tensors with NIXL...")
            self._nixl_manager.register_tensors(tensors)
            logger.debug(f"[Worker {device_id}] Tensors registered with NIXL")

        _tensor_registry[device_id] = tensors
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
            _publish_metadata_and_ready(
                self._mx_client, self._nixl_manager, tensors, device_id, model_name
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
    def tensors(self) -> dict[str, torch.Tensor]:
        """Access the registered tensor dict."""
        return self._tensors


# Global storage for tensor metadata, keyed by device_id.
# Required because vLLM's loader API doesn't expose loader instances after
# load_model() returns. Source loaders store state here so the MxClient
# (running in the same worker process) can access NIXL managers and tensors.
# Each device_id maps to exactly one loader, so there are no concurrent writers.
_tensor_registry: dict[int, dict[str, torch.Tensor]] = {}
_nixl_managers: dict[int, "NixlTransferManager"] = {}
