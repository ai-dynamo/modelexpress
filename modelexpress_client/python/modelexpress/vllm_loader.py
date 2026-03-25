# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress Custom Model Loader for vLLM.

This loader hooks into vLLM's weight loading pipeline to perform RDMA transfers
of fully-processed model tensors. Registration happens AFTER
process_weights_after_loading() so that all final tensors (parameters, buffers,
and bare tensor attributes like FP8 scales) are captured and transferred.

Supports a three-tier loading strategy:
    1. RDMA (P2P GPU transfer via NIXL) - if a source is already serving
    2. GDS (GPUDirect Storage) - direct file-to-GPU, bypassing CPU
    3. Disk (vLLM DefaultModelLoader) - standard CPU-staged loading

NIXL agent blobs are exchanged peer-to-peer via NIXL's native listen thread
(fetch_remote_metadata / check_remote_metadata). The centralized MX server
only stores lightweight directory entries (endpoints + agent names + tensor
layouts).

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
from .worker_server import WorkerGrpcServer, fetch_tensor_manifest

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

_MX_METADATA_PORT_DEFAULT = 5555
_MX_WORKER_GRPC_PORT_DEFAULT = 6555

# ---------------------------------------------------------------------------
# Shared helpers used by multiple loaders
# ---------------------------------------------------------------------------


def _get_listen_port(device_id: int) -> int:
    """Compute the NIXL listen port for a given device/rank."""
    base_port = int(os.environ.get("MX_METADATA_PORT", str(_MX_METADATA_PORT_DEFAULT)))
    return base_port + device_id


def _get_worker_grpc_port(device_id: int) -> int:
    """Compute the worker gRPC port for a given device/rank."""
    base_port = int(os.environ.get("MX_WORKER_GRPC_PORT", str(_MX_WORKER_GRPC_PORT_DEFAULT)))
    return base_port + device_id


def _tensors_from_proto(tensor_protos) -> list[TensorDescriptor]:
    """Convert a list of TensorDescriptor protos to TensorDescriptor dataclasses."""
    return [
        TensorDescriptor(
            name=t.name,
            addr=t.addr,
            size=t.size,
            device_id=t.device_id,
            dtype=t.dtype,
        )
        for t in tensor_protos
    ]


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

    Deduplicates by data_ptr() to avoid registering the same GPU memory
    twice (e.g. tied weights where embed_tokens.weight and lm_head.weight
    share the same tensor).
    """
    tensors: dict[str, torch.Tensor] = {}
    seen_ptrs: set[int] = set()
    skipped_noncontiguous = 0
    skipped_duplicate = 0
    for name, tensor, _tensor_type in _iter_module_tensors(model):
        t = tensor.data if hasattr(tensor, "data") else tensor

        if not t.is_contiguous():
            logger.debug(f"Skipping non-contiguous tensor '{name}' (view of another tensor)")
            skipped_noncontiguous += 1
            continue

        ptr = t.data_ptr()
        if ptr in seen_ptrs:
            logger.debug(f"Skipping duplicate tensor '{name}' (same data_ptr as already registered tensor)")
            skipped_duplicate += 1
            continue
        seen_ptrs.add(ptr)

        tensors[name] = t

    if skipped_noncontiguous:
        logger.info(
            f"Skipped {skipped_noncontiguous} non-contiguous tensors (views of contiguous tensors already registered)"
        )
    if skipped_duplicate:
        logger.info(f"Skipped {skipped_duplicate} duplicate tensors (tied weights sharing the same memory)")
    return tensors


def _init_nixl_manager(
    global_rank: int, device_id: int, role: str, listen_port: int | None = None,
) -> NixlTransferManager:
    """Create and initialize a NIXL transfer manager.

    Args:
        global_rank: Global rank for agent naming.
        device_id: GPU device ID for this worker.
        role: Role string for agent naming (e.g., "auto").
        listen_port: If specified, enables the NIXL listen thread on this port
            for native peer-to-peer metadata exchange.
    """
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


def _get_worker_host() -> str:
    """Get the host address for this worker's NIXL listen thread."""
    import socket
    host = os.environ.get(
        "MX_WORKER_ADDRESS",
        os.environ.get("POD_IP"),
    )
    if host:
        return host
    host = socket.getfqdn()
    if host and host not in ("localhost", "localhost.localdomain"):
        return host
    raise RuntimeError(
        "Unable to determine a routable worker address. "
        "Set MX_WORKER_ADDRESS or POD_IP."
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
    metadata_endpoint: str = "",
    worker_grpc_endpoint: str = "",
) -> None:
    """Start a per-worker gRPC server for the tensor manifest, then publish
    lightweight metadata and ready flag to the ModelExpress server.

    Tensor descriptors are served directly by the worker's gRPC server.
    The centralized MX server only stores endpoints and agent names.
    """
    logger.info(
        f"[Worker {global_rank}] Publishing metadata for model '{identity.model_name}' "
        f"({len(tensors)} tensors served at {worker_grpc_endpoint})"
    )

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

    # Start per-worker gRPC server to serve tensor manifest
    grpc_port = _get_worker_grpc_port(device_id)
    server = WorkerGrpcServer(tensor_protos, port=grpc_port, alloc_ends=nixl_manager.alloc_ends)
    server.start()
    _worker_servers[device_id] = server

    worker = p2p_pb2.WorkerMetadata(
        worker_rank=global_rank,
        metadata_endpoint=metadata_endpoint,
        agent_name=nixl_manager.agent_name,
        worker_grpc_endpoint=worker_grpc_endpoint,
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
        5. Register ALL final tensors with NIXL + publish
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

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )

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
        """Fetch lightweight metadata for one worker from the MX server.

        Returns None if the worker is not found or has no worker_grpc_endpoint.
        Tensor manifests are fetched separately from the worker's gRPC server.
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
        if not worker.worker_grpc_endpoint:
            logger.debug(
                f"[Worker {global_rank}] Worker {worker_id} has no worker_grpc_endpoint, skipping"
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
                    f"(grpc={source_worker.worker_grpc_endpoint})"
                )
                self._load_as_target(
                    model, model_config, target_device,
                    global_rank, device_id, identity, source_worker, mx_source_id, worker_id,
                )
                return True
            except SourceTransferError as e:
                logger.warning(
                    f"[Worker {global_rank}] Source-side failure for worker {worker_id}: {e}. "
                        f"Marking STALE and trying next."
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
        """Receive fully-processed weights via RDMA from an existing source."""
        # Create dummy weights as receive buffers
        self._dummy_loader.load_weights(model, model_config)

        # Process dummy weights to establish final tensor layout
        process_weights_after_loading(model, model_config, target_device)

        # RDMA receive (fully-processed tensors, no post-processing needed)
        # Raises SourceTransferError on source-side failures
        self._receive_from_peer(model, global_rank, device_id, source_worker, identity)

        # Publish metadata so future nodes can discover us
        self._publish_metadata(global_rank, device_id, identity)

    def _receive_from_peer(
        self,
        model: nn.Module,
        global_rank: int,
        device_id: int,
        source_worker,
        identity: "p2p_pb2.SourceIdentity",
    ) -> None:
        """Receive fully-processed tensors via RDMA from the detected source.

        Raises SourceTransferError if the RDMA receive fails, indicating a
        proven source-side problem. Other exceptions propagate normally.
        """
        receive_start = time.perf_counter()
        self._register_tensors(model, global_rank, device_id)

        # Fetch tensor manifest directly from source worker's gRPC server
        worker_grpc_ep = source_worker.worker_grpc_endpoint
        logger.info(f"[Worker {global_rank}] Fetching tensor manifest from {worker_grpc_ep}")
        manifest_start = time.perf_counter()
        tensor_protos, source_alloc_ends = fetch_tensor_manifest(worker_grpc_ep)
        manifest_time = time.perf_counter() - manifest_start
        source_tensors = _tensors_from_proto(tensor_protos)
        logger.info(
            f"[Worker {global_rank}] [TIMING] Tensor manifest fetched in {manifest_time:.3f}s: "
            f"{len(source_tensors)} tensors, {len(source_alloc_ends)} alloc_ends"
        )

        # Parse source endpoint and agent name for NIXL native P2P exchange
        source_endpoint = source_worker.metadata_endpoint
        source_agent_name = source_worker.agent_name
        host, port_str = source_endpoint.rsplit(":", 1)
        source_port = int(port_str)

        # Fetch remote metadata via NIXL native P2P
        fetch_start = time.perf_counter()
        remote_agent = self._nixl_manager.fetch_remote_and_wait(
            agent_name=source_agent_name,
            ip=host,
            port=source_port,
        )
        fetch_time = time.perf_counter() - fetch_start
        logger.info(f"[Worker {global_rank}] [TIMING] NIXL metadata fetched in {fetch_time:.3f}s")

        try:
            transfer_start = time.perf_counter()
            bytes_transferred, tensor_count, _ = self._nixl_manager.receive_from_source(
                source_metadata=None,
                source_tensors=source_tensors,
                timeout_seconds=300.0,
                coalesce_transfers=True,
                remote_agent_name=remote_agent,
                source_alloc_ends=source_alloc_ends,
            )
            transfer_time = time.perf_counter() - transfer_start

            bandwidth_gbps = (bytes_transferred * 8) / (transfer_time * 1e9) if transfer_time > 0 else 0
            logger.info(
                f"[Worker {global_rank}] [TIMING] RDMA transfer complete: "
                f"{tensor_count} tensors, {bytes_transferred / 1e9:.2f} GB, "
                f"{transfer_time:.3f}s, {bandwidth_gbps:.1f} Gbps"
            )
        except Exception as transfer_err:
            raise SourceTransferError(
                f"RDMA transfer failed: {transfer_err}"
            ) from transfer_err

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

        # Process weights FIRST, then register final tensors
        process_weights_after_loading(model, model_config, target_device)

        # Register tensors + publish metadata
        self._register_tensors(model, global_rank, device_id)
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

        listen_port = _get_listen_port(device_id)

        if self._nixl_manager is None:
            self._nixl_manager = _init_nixl_manager(
                global_rank, device_id, "auto", listen_port=listen_port,
            )

        if not self._nixl_manager.tensor_descriptors:
            logger.debug(f"[Worker {global_rank}] Registering tensors with NIXL...")
            self._nixl_manager.register_tensors(self._tensors)
            logger.debug(f"[Worker {global_rank}] Tensors registered with NIXL")

        _tensor_registry[device_id] = self._tensors
        _nixl_managers[device_id] = self._nixl_manager

    def _publish_metadata(self, global_rank: int, device_id: int, identity: "p2p_pb2.SourceIdentity") -> None:
        """Publish metadata to the MX server."""
        listen_port = _get_listen_port(device_id)
        grpc_port = _get_worker_grpc_port(device_id)
        host = _get_worker_host()
        metadata_endpoint = f"{host}:{listen_port}"
        worker_grpc_endpoint = f"{host}:{grpc_port}"
        agent_name = self._nixl_manager.agent_name
        logger.info(
            f"[Worker {global_rank}] NIXL listen thread at {metadata_endpoint} "
            f"(agent={agent_name}), worker gRPC at {worker_grpc_endpoint}"
        )
        _publish_metadata_and_ready(
            self._mx_client, self._nixl_manager, self._tensors,
            global_rank, device_id, identity, self._worker_id,
            metadata_endpoint=metadata_endpoint,
            worker_grpc_endpoint=worker_grpc_endpoint,
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
_worker_servers: dict[int, "WorkerGrpcServer"] = {}
