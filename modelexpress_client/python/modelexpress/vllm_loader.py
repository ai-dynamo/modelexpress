# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress Custom Model Loader for vLLM.

This loader hooks into vLLM's weight loading pipeline to perform RDMA transfers
of fully-processed model tensors. Registration happens AFTER
process_weights_after_loading() so that all final tensors are captured.
Tensor discovery uses named_parameters() and named_buffers(); bare tensor
attributes created during post-processing (e.g. FP8 scales, MLA projections)
are auto-promoted to non-persistent buffers via _capture_tensor_attrs().

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
import random
import time
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING

import grpc
import torch
import torch.nn as nn

from .client import MxClient  # All gRPC communication goes through MxClient
from .heartbeat import HeartbeatThread
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
from .transfer_safety import (
    TransferFingerprint,
    check_transfer_allowed,
    detect_model_features,
)
from .types import TensorDescriptor

if TYPE_CHECKING:
    from .nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.vllm_loader")

MAX_SOURCE_RETRIES = 3
PUBLISH_METADATA_MAX_ATTEMPTS = 3
PUBLISH_METADATA_INITIAL_BACKOFF_SECONDS = 1.0
PUBLISH_METADATA_RETRYABLE_STATUS_CODES = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
}

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


from .types import ManifestMismatchError  # noqa: E402 (after SourceTransferError for grouping)


# ---------------------------------------------------------------------------
# Shared helpers used by multiple loaders
# ---------------------------------------------------------------------------



@contextmanager
def _capture_tensor_attrs():
    """Intercept bare CUDA tensor assignments during process_weights_after_loading.

    vLLM's post-processing (quant methods, attention backends) may create
    tensor attributes via plain setattr (e.g. self.W_UV = tensor) instead
    of register_buffer. These are invisible to named_parameters/named_buffers
    and would be missing from the RDMA manifest.

    This context manager patches nn.Module.__setattr__ to auto-promote such
    tensors to non-persistent buffers, making them discoverable by
    named_buffers() and thus included in the manifest.
    """
    original_setattr = nn.Module.__setattr__

    def capturing_setattr(self, name, value):
        if (isinstance(value, torch.Tensor)
                and not isinstance(value, nn.Parameter)
                and value.is_cuda
                and name not in self._parameters
                and name not in self._buffers
                and name not in self._modules):
            if hasattr(self, name):
                try:
                    delattr(self, name)
                except AttributeError:
                    pass
            self.register_buffer(name, value, persistent=False)
        else:
            original_setattr(self, name, value)

    nn.Module.__setattr__ = capturing_setattr
    try:
        yield
    finally:
        nn.Module.__setattr__ = original_setattr


def _iter_module_tensors(
    module: nn.Module,
) -> list[tuple[str, torch.Tensor, str]]:
    """Iterate over all CUDA tensors in a module tree.

    Uses named_parameters() and named_buffers() to discover tensors.
    When used with _capture_tensor_attrs() wrapping process_weights_after_loading,
    bare tensor attributes (e.g. W_UV, W_UK_T) are auto-promoted to
    non-persistent buffers and thus included in named_buffers().

    Returns:
        List of (qualified_name, tensor, tensor_type) tuples for each CUDA tensor.
    """
    results: list[tuple[str, torch.Tensor, str]] = []

    for name, param in module.named_parameters():
        if param.is_cuda:
            results.append((name, param, "parameter"))

    for name, buf in module.named_buffers():
        if buf.is_cuda:
            results.append((name, buf, "buffer"))

    return results


def _storage_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return a flat contiguous uint8 view of a tensor's underlying storage.

    For RDMA we transfer raw storage bytes. Both source and target run
    the same post-processing on the same model architecture, so they
    produce identical storage layouts (same sizes, strides, offsets).
    Transferring the full storage block ensures all views into it
    (including partial views like MLA's W_UV and W_UK_T which share
    storage from a dequantized intermediate) get correct data.

    Multiple tensors sharing the same storage are deduplicated by
    data_ptr() in the caller, so only one transfer per storage block.
    """
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        tensor.untyped_storage()
    )


def _collect_module_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect all CUDA tensors from a module tree into a flat dict.

    Uses _iter_module_tensors (named_parameters + named_buffers) to find
    tensors, then returns them as a name -> tensor mapping suitable for
    NIXL registration. Bare tensor attributes created during
    process_weights_after_loading are captured as non-persistent buffers
    by the _capture_tensor_attrs context manager.

    Contiguous tensors are registered directly. Non-contiguous tensors
    (DeepGemm TMA-aligned FP8 scales, MLA dequantized projections)
    are registered as a flat byte view of their full underlying storage,
    named as ``name.__storage``. This transfers the raw bytes correctly
    because both source and target have identical storage layouts.
    Multiple views into the same storage (e.g. W_UV and W_UK_T sharing
    a dequantized intermediate) are deduplicated by data_ptr so the
    storage is transferred only once.
    """
    tensors: dict[str, torch.Tensor] = {}
    seen_ptrs: set[int] = set()
    storage_view_count = 0
    skipped_duplicate = 0
    for name, tensor, _tensor_type in _iter_module_tensors(model):
        t = tensor.data if hasattr(tensor, "data") else tensor

        if t.is_contiguous():
            ptr = t.data_ptr()
            if ptr in seen_ptrs:
                logger.debug(f"Skipping duplicate tensor '{name}' (same data_ptr)")
                skipped_duplicate += 1
                continue
            seen_ptrs.add(ptr)
            tensors[name] = t
        else:
            sv = _storage_view(t)
            ptr = sv.data_ptr()
            if ptr in seen_ptrs:
                skipped_duplicate += 1
                continue
            seen_ptrs.add(ptr)
            tensors[f"{name}.__storage"] = sv
            storage_view_count += 1

    if storage_view_count:
        logger.info(
            f"Registered {storage_view_count} non-contiguous tensors "
            f"via storage-level byte transfer"
        )
    if skipped_duplicate:
        logger.info(f"Skipped {skipped_duplicate} duplicate tensors (tied weights)")
    return tensors


def _is_p2p_metadata_enabled() -> bool:
    """Check if P2P metadata exchange is enabled via env var."""
    return os.environ.get("MX_P2P_METADATA", "0") == "1"


def _get_worker_host() -> str:
    """Get the routable hostname/IP for this worker.

    Priority: MX_WORKER_HOST env var, then pod IP via socket.
    Falls back to FQDN. Rejects localhost variants.
    """
    import socket
    explicit = os.environ.get("MX_WORKER_HOST", "")
    if explicit:
        return explicit
    # Try to get routable IP (works in K8s pods)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    fqdn = socket.getfqdn()
    if fqdn in ("localhost", "localhost.localdomain"):
        raise RuntimeError(
            "Cannot determine routable address for P2P metadata exchange. "
            "Set MX_WORKER_HOST or configure DNS."
        )
    return fqdn


def _init_nixl_manager(
    global_rank: int, device_id: int, role: str, listen_port: int = 0,
) -> NixlTransferManager:
    """Create and initialize a NIXL transfer manager."""
    agent_name = f"mx-{role}-worker{global_rank}-{uuid.uuid4().hex[:8]}"
    logger.debug(f"[Worker {global_rank}] Initializing NIXL manager with agent_name={agent_name}")
    manager = NixlTransferManager(
        agent_name=agent_name,
        device_id=device_id,
        listen_port=listen_port,
    )
    manager.initialize()
    logger.debug(f"[Worker {global_rank}] NIXL manager initialized")
    return manager


def _log_tensor_summary(
    tensors: dict[str, torch.Tensor], global_rank: int, label: str
) -> None:
    """Log a summary of tensor count, size, scale_inv count, and sample checksums."""
    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    logger.info(
        f"[Worker {global_rank}] {label}: {len(tensors)} tensors ({total_size / 1e9:.2f} GB)"
    )

    if logger.isEnabledFor(logging.DEBUG):
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
        extra_parameters={
            f"feature.{k}": v for k, v in detect_model_features(model_config).items()
        },
    )


def _build_tensor_protos(
    nixl_manager: NixlTransferManager,
    tensors: dict[str, torch.Tensor],
    device_id: int,
    global_rank: int,
) -> list["p2p_pb2.TensorDescriptor"]:
    """Build tensor descriptor protos from registered tensors."""
    use_contiguous = os.environ.get("MX_CONTIGUOUS_REG", "0") == "1"

    if use_contiguous:
        region_descriptors = nixl_manager.get_registered_descriptors()
        protos = [
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
            f"[Worker {global_rank}] Built {len(protos)} REGION descriptors "
            f"(MX_CONTIGUOUS_REG=1)"
        )
    else:
        protos = [
            p2p_pb2.TensorDescriptor(
                name=name,
                addr=t.data_ptr(),
                size=t.numel() * t.element_size(),
                device_id=device_id,
                dtype=str(t.dtype),
            )
            for name, t in tensors.items()
        ]
    return protos


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

    tensor_protos = _build_tensor_protos(
        nixl_manager, tensors, device_id, global_rank,
    )

    if _is_p2p_metadata_enabled():
        from .worker_server import WorkerGrpcServer

        host = _get_worker_host()

        # Publish lightweight metadata (no nixl_metadata blob, no tensors)
        # and start a worker gRPC server for tensor manifest serving.
        # Each worker gets base_port + device_id to avoid collisions.
        grpc_base = int(os.environ.get("MX_WORKER_GRPC_PORT", "6555"))
        worker_grpc_port = grpc_base + device_id

        # Publish first to get mx_source_id, then start worker server
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=global_rank,
            metadata_endpoint=f"{host}:{nixl_manager._listen_port}",
            agent_name=nixl_manager.agent_name,
            worker_grpc_endpoint="",  # will update after server starts
        )
        mx_source_id = _publish_metadata_to_server(
            mx_client=mx_client,
            identity=identity,
            worker=worker,
            worker_id=worker_id,
            global_rank=global_rank,
        )

        # Start worker gRPC server
        grpc_server = WorkerGrpcServer(
            tensor_protos=tensor_protos,
            mx_source_id=mx_source_id,
            port=worker_grpc_port,
        )
        actual_port = grpc_server.start()
        _worker_servers[device_id] = grpc_server

        # Re-publish with the actual worker_grpc_endpoint
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=global_rank,
            metadata_endpoint=f"{host}:{nixl_manager._listen_port}",
            agent_name=nixl_manager.agent_name,
            worker_grpc_endpoint=f"{host}:{actual_port}",
        )
        mx_source_id = _publish_metadata_to_server(
            mx_client=mx_client,
            identity=identity,
            worker=worker,
            worker_id=worker_id,
            global_rank=global_rank,
        )
        logger.info(
            f"[Worker {global_rank}] Published P2P metadata to MX server "
            f"(mx_source_id={mx_source_id}, worker_grpc={host}:{actual_port})"
        )
    else:
        # Centralized mode: publish full metadata (blobs + tensors)
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=global_rank,
            nixl_metadata=nixl_manager.nixl_metadata,
            tensors=tensor_protos,
        )
        mx_source_id = _publish_metadata_to_server(
            mx_client=mx_client,
            identity=identity,
            worker=worker,
            worker_id=worker_id,
            global_rank=global_rank,
        )
        logger.info(
            f"[Worker {global_rank}] Published metadata to MX server "
            f"(mx_source_id={mx_source_id}, worker_id={worker_id})"
        )

    heartbeat = HeartbeatThread(
        mx_client=mx_client,
        mx_source_id=mx_source_id,
        worker_id=worker_id,
        worker_rank=global_rank,
        nixl_manager=nixl_manager,
    )
    heartbeat.start()
    _heartbeat_threads[global_rank] = heartbeat


def _publish_metadata_to_server(
    mx_client: MxClient,
    identity: "p2p_pb2.SourceIdentity",
    worker: "p2p_pb2.WorkerMetadata",
    worker_id: str,
    global_rank: int,
) -> str:
    """Publish metadata with bounded retries and exponential backoff."""
    last_error: grpc.RpcError | None = None

    for attempt in range(1, PUBLISH_METADATA_MAX_ATTEMPTS + 1):
        try:
            return mx_client.publish_metadata(identity, worker, worker_id)
        except grpc.RpcError as exc:
            if exc.code() not in PUBLISH_METADATA_RETRYABLE_STATUS_CODES:
                raise

            last_error = exc
            if attempt == PUBLISH_METADATA_MAX_ATTEMPTS:
                break

            backoff_seconds = PUBLISH_METADATA_INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                f"[Worker {global_rank}] Publish metadata attempt {attempt}/"
                f"{PUBLISH_METADATA_MAX_ATTEMPTS} failed with retryable gRPC status "
                f"{exc.code().name}: {exc}. Retrying in {backoff_seconds:.1f}s"
            )
            time.sleep(backoff_seconds)

    message = (
        f"[Worker {global_rank}] Failed to publish metadata after "
        f"{PUBLISH_METADATA_MAX_ATTEMPTS} attempts"
    )
    logger.error("%s: %s", message, last_error)
    raise RuntimeError(f"{message}: {last_error}") from last_error


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
        3a. Source found -> load dummy weights, RDMA receive
        3b. No source   -> try GDS, else disk load
        4. process_weights_after_loading()
        5. Register final parameters/buffers with NIXL + publish
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self._nixl_manager: NixlTransferManager | None = None
        self._tensors: dict[str, torch.Tensor] = {}
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
        self._default_loader = DefaultModelLoader(disk_config)

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

        # Check if model features are safe for RDMA transfer
        transfer_allowed, allow_reason = check_transfer_allowed(model_config)

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )

            loaded_as_target = False
            if transfer_allowed:
                loaded_as_target = self._try_load_from_candidates(
                    model, model_config, target_device, global_rank, device_id, identity
                )
            else:
                logger.info(
                    f"[Worker {global_rank}] RDMA transfer disabled: {allow_reason}"
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
        fetch_start = time.perf_counter()
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
        if not worker.tensors and not worker.worker_grpc_endpoint:
            logger.debug(
                f"[Worker {global_rank}] Worker {worker_id} has no tensors "
                f"and no P2P endpoint, skipping"
            )
            return None
        fetch_time = time.perf_counter() - fetch_start
        mode = "P2P (lightweight)" if worker.worker_grpc_endpoint else "centralized"
        tensor_count = len(worker.tensors)
        logger.info(
            f"[Worker {global_rank}] [TIMING] GetMetadata ({mode}): "
            f"{fetch_time:.3f}s, {tensor_count} tensors"
        )
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
        for instance in candidates[:MAX_SOURCE_RETRIES]:
            mx_source_id = instance.mx_source_id
            worker_id = instance.worker_id

            try:
                source_worker = self._fetch_worker_metadata(mx_source_id, worker_id, global_rank)
            except Exception as e:
                logger.warning(
                    f"[Worker {global_rank}] Failed to fetch metadata for worker {worker_id}: {e}. "
                    f"Trying next candidate."
                )
                continue

            if source_worker is None:
                continue

            logger.info(
                f"[Worker {global_rank}] Trying source worker {worker_id} "
                f"({len(source_worker.tensors)} tensors)"
            )

            try:
                self._load_as_target(
                    model, model_config, target_device,
                    global_rank, device_id, identity, source_worker, mx_source_id, worker_id,
                )
                return True
            except SourceTransferError as e:
                logger.warning(
                    f"[Worker {global_rank}] Source transfer failed for worker {worker_id}: {e}. "
                    f"Trying next candidate."
                )
            except ManifestMismatchError as e:
                logger.warning(
                    f"[Worker {global_rank}] Manifest mismatch with worker {worker_id}: {e}. "
                    f"Trying next candidate."
                )
        if candidates:
            tried = min(len(candidates), MAX_SOURCE_RETRIES)
            logger.warning(
                f"[Worker {global_rank}] Tried {tried} of {len(candidates)} source workers "
                f"(max retries={MAX_SOURCE_RETRIES}), loading from disk"
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
        """Receive fully-processed weights via RDMA from an existing source."""
        # Create dummy weights as receive buffers
        self._dummy_loader.load_weights(model, model_config)

        # Process dummy weights to establish final tensor layout.
        # _capture_tensor_attrs promotes bare CUDA tensor assignments
        # (e.g. W_UV, W_UK_T) to non-persistent buffers so they appear
        # in named_buffers() and get included in the RDMA manifest.
        with _capture_tensor_attrs():
            process_weights_after_loading(model, model_config, target_device)

        # Log transfer fingerprint for debugging environment mismatches.
        target_fp = TransferFingerprint.from_environment(self._tensors)
        logger.info(
            f"[Worker {global_rank}] Transfer fingerprint: "
            f"vllm={target_fp.vllm_version}, cuda={target_fp.cuda_version}, "
            f"attn={target_fp.attention_backend}, "
            f"deepgemm={target_fp.deep_gemm_version}, "
            f"manifest={target_fp.manifest_hash[:12]}... ({target_fp.tensor_count} tensors)"
        )

        # RDMA receive (fully-processed tensors, no post-processing needed)
        # Raises SourceTransferError on source-side failures
        self._receive_from_peer(model, global_rank, device_id, source_worker, mx_source_id)

        # Publish metadata so future nodes can discover us
        self._publish_metadata(global_rank, device_id, identity)

    def _receive_from_peer(
        self,
        model: nn.Module,
        global_rank: int,
        device_id: int,
        source_worker,
        mx_source_id: str = "",
    ) -> None:
        """Receive fully-processed tensors via RDMA from the detected source.

        Auto-detects P2P vs centralized mode based on whether the source
        published P2P endpoint fields. In P2P mode, fetches tensor manifest
        from the source worker's gRPC server and NIXL metadata via the
        listen thread. In centralized mode, uses metadata from the server.

        Raises SourceTransferError if the RDMA receive fails, indicating a
        proven source-side problem. Other exceptions propagate normally.
        """
        receive_start = time.perf_counter()
        self._register_tensors(model, global_rank, device_id)

        is_p2p = bool(source_worker.worker_grpc_endpoint)
        remote_agent_name_override = None

        if is_p2p:
            # P2P mode: fetch tensors from source worker directly
            from .worker_server import fetch_tensor_manifest

            manifest_start = time.perf_counter()
            logger.info(
                f"[Worker {global_rank}] P2P mode: fetching tensor manifest from "
                f"{source_worker.worker_grpc_endpoint}"
            )
            tensor_protos = fetch_tensor_manifest(
                endpoint=source_worker.worker_grpc_endpoint,
                mx_source_id=mx_source_id,
            )
            manifest_time = time.perf_counter() - manifest_start
            source_tensors = [
                TensorDescriptor(
                    name=t.name, addr=t.addr, size=t.size,
                    device_id=t.device_id, dtype=t.dtype,
                )
                for t in tensor_protos
            ]
            logger.info(
                f"[Worker {global_rank}] [TIMING] P2P tensor manifest: "
                f"{manifest_time:.3f}s ({len(source_tensors)} tensors)"
            )

            # Fetch NIXL metadata via P2P listen thread
            nixl_fetch_start = time.perf_counter()
            ep = source_worker.metadata_endpoint
            host, port_str = ep.rsplit(":", 1)
            self._nixl_manager.fetch_remote_and_wait(
                remote_agent_name=source_worker.agent_name,
                ip=host,
                port=int(port_str),
            )
            nixl_fetch_time = time.perf_counter() - nixl_fetch_start
            logger.info(
                f"[Worker {global_rank}] [TIMING] P2P NIXL metadata fetch: "
                f"{nixl_fetch_time:.3f}s"
            )
            remote_agent_name_override = source_worker.agent_name
        else:
            # Centralized mode: tensors and NIXL blob from server
            source_tensors = [
                TensorDescriptor(
                    name=t.name, addr=t.addr, size=t.size,
                    device_id=t.device_id, dtype=t.dtype,
                )
                for t in source_worker.tensors
            ]

        logger.info(
            f"[Worker {global_rank}] Receiving {len(source_tensors)} tensors from source"
            f"{' (P2P)' if is_p2p else ''}"
        )

        coalesce = os.environ.get("MX_CONTIGUOUS_REG", "0") == "1"

        transfer_start = time.perf_counter()
        try:
            bytes_transferred, tensor_count, _ = self._nixl_manager.receive_from_source(
                source_metadata=source_worker.nixl_metadata,
                source_tensors=source_tensors,
                timeout_seconds=300.0,
                coalesce_transfers=coalesce,
                remote_agent_name=remote_agent_name_override,
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
    # Source path: GDS -> disk, then register + publish
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
        """Load weights via GDS or disk, process, then register + publish."""
        loaded_via_gds = False

        if is_gds_available():
            loaded_via_gds = self._try_gds_load(
                model, model_config, device_id
            )

        if not loaded_via_gds:
            logger.info(f"[Worker {device_id}] Loading weights from disk...")
            self._default_loader.load_weights(model, model_config)
            logger.info(f"[Worker {device_id}] Weights loaded from disk")

        # Process weights FIRST, then register final tensors.
        # _capture_tensor_attrs promotes bare CUDA tensor assignments
        # to non-persistent buffers for manifest discovery.
        with _capture_tensor_attrs():
            process_weights_after_loading(model, model_config, target_device)

        # Register tensors + publish metadata
        self._register_tensors(model, global_rank, device_id)

        # Log transfer fingerprint so targets can compare
        source_fp = TransferFingerprint.from_environment(self._tensors)
        logger.info(
            f"[Worker {global_rank}] Source fingerprint: "
            f"vllm={source_fp.vllm_version}, cuda={source_fp.cuda_version}, "
            f"attn={source_fp.attention_backend}, "
            f"deepgemm={source_fp.deep_gemm_version}, "
            f"manifest={source_fp.manifest_hash[:12]}... ({source_fp.tensor_count} tensors)"
        )

        self._publish_metadata(global_rank, device_id, identity)

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
            use_tqdm = getattr(self.load_config, "use_tqdm_on_load", True)
            revision = getattr(model_config, "revision", None)
            weights_iter = gds_loader.load_iter(
                model_config.model, use_tqdm=use_tqdm, revision=revision
            )
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

    def _register_tensors(self, model: nn.Module, global_rank: int, device_id: int) -> None:
        """Collect model tensors and register them with NIXL."""
        if not is_nixl_available():
            logger.warning(f"[Worker {global_rank}] NIXL not available, skipping registration")
            return

        self._tensors = _collect_module_tensors(model)
        _log_tensor_summary(self._tensors, global_rank, "Registering tensors")

        if self._nixl_manager is None:
            # Always enable the NIXL listen thread: targets need it to call
            # fetch_remote_metadata on P2P sources, and every node becomes a
            # source after receiving weights. Each worker needs a unique port
            # (base + device_id) to avoid collisions in multi-GPU setups.
            base_port = int(os.environ.get("MX_METADATA_PORT", "5555"))
            listen_port = base_port + device_id
            self._nixl_manager = _init_nixl_manager(global_rank, device_id, "auto", listen_port)

        if not self._nixl_manager.tensor_descriptors:
            logger.debug(f"[Worker {global_rank}] Registering tensors with NIXL...")
            self._nixl_manager.register_tensors(self._tensors)
            logger.debug(f"[Worker {global_rank}] Tensors registered with NIXL")

        _tensor_registry[device_id] = self._tensors
        _nixl_managers[device_id] = self._nixl_manager

    def _publish_metadata(self, global_rank: int, device_id: int, identity: "p2p_pb2.SourceIdentity") -> None:
        """Publish metadata to the MX server. Failures are logged but do not raise."""
        try:
            _publish_metadata_and_ready(
                self._mx_client, self._nixl_manager, self._tensors,
                global_rank, device_id, identity, self._worker_id
            )
        except Exception as e:
            logger.warning(
                f"[Worker {global_rank}] Failed to publish metadata, "
                f"worker will continue without P2P serving: {e}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        """Download the model so it can be loaded immediately."""
        self._default_loader.download_model(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into an already-initialized model (standalone API)."""
        self._default_loader.load_weights(model, model_config)

    @property
    def nixl_manager(self) -> NixlTransferManager | None:
        """Access the NIXL manager for external use."""
        return self._nixl_manager

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        """Access the registered tensor dict."""
        return self._tensors


# Global storage for tensor metadata, keyed by device_id (local CUDA ordinal).
# Required because vLLM's loader API doesn't expose loader instances after
# load_model() returns. Source loaders store state here so the MxClient
# (running in the same worker process) can access NIXL managers and tensors.
# Each device_id maps to exactly one loader, so there are no concurrent writers.
_tensor_registry: dict[int, dict[str, torch.Tensor]] = {}
_nixl_managers: dict[int, "NixlTransferManager"] = {}
_heartbeat_threads: dict[int, HeartbeatThread] = {}
_worker_servers: dict[int, "WorkerGrpcServer"] = {}  # P2P mode only
