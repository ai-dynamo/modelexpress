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
import json
import logging
import os
import sys
import time
import uuid
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

# Redis client for source-target coordination
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

if TYPE_CHECKING:
    from .nixl_transfer import NixlTransferManager

# Configure logging to stdout for visibility in k8s logs
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger("modelexpress.vllm_loader")
logger.setLevel(logging.DEBUG)

def _log(msg: str, level: str = "INFO") -> None:
    """Force log to stdout for k8s visibility."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {level} vllm_loader: {msg}", flush=True)


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


class SourceReadyCoordinator:
    """
    Redis-based coordination to prevent stale metadata issues.
    
    Source publishes a "ready" flag with session_id after:
    - Warmup complete
    - NIXL registered
    - Metadata published
    
    Target waits for this flag and detects restarts via session_id change.
    """
    
    # Class-level session ID (unique per process lifetime)
    _session_id: str | None = None
    
    @classmethod
    def get_session_id(cls) -> str:
        """Get or create session ID for this process."""
        if cls._session_id is None:
            cls._session_id = str(uuid.uuid4())
        return cls._session_id
    
    @staticmethod
    def _get_redis_client() -> "redis.Redis | None":
        """Get Redis client from environment."""
        if not REDIS_AVAILABLE:
            _log("Redis client not available, skipping coordination", "WARNING")
            return None
        
        redis_host = os.environ.get("MX_REDIS_HOST", "modelexpress-server")
        redis_port = int(os.environ.get("MX_REDIS_PORT", "6379"))
        
        try:
            client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            client.ping()
            return client
        except Exception as e:
            _log(f"Failed to connect to Redis at {redis_host}:{redis_port}: {e}", "WARNING")
            return None
    
    @classmethod
    def publish_source_ready(
        cls,
        model_name: str,
        worker_id: int,
        metadata_hash: str,
    ) -> bool:
        """
        Publish source ready flag to Redis.
        Called after warmup and metadata publishing is complete.
        """
        client = cls._get_redis_client()
        if client is None:
            return False
        
        session_id = cls.get_session_id()
        key = f"mx:ready:{model_name}:worker:{worker_id}"
        
        ready_info = {
            "session_id": session_id,
            "timestamp": time.time(),
            "metadata_hash": metadata_hash,
            "ready": True,
        }
        
        try:
            # Set with 2 hour TTL (matches warmup timeout)
            client.setex(key, 7200, json.dumps(ready_info))
            _log(f"[Worker {worker_id}] Published ready flag to Redis: session={session_id[:8]}...", "INFO")
            return True
        except Exception as e:
            _log(f"[Worker {worker_id}] Failed to publish ready flag: {e}", "WARNING")
            return False
    
    @classmethod
    def wait_for_source_ready(
        cls,
        model_name: str,
        worker_id: int,
        timeout_seconds: int = 7200,
        poll_interval: int = 10,
    ) -> tuple[bool, str | None, str | None]:
        """
        Wait for source ready flag.
        
        Returns:
            (success, session_id, metadata_hash)
        """
        client = cls._get_redis_client()
        if client is None:
            _log(f"[Worker {worker_id}] Redis not available, proceeding without coordination", "WARNING")
            return True, None, None
        
        key = f"mx:nixl_ready:{model_name}:worker:{worker_id}"
        start_time = time.time()
        
        _log(f"[Worker {worker_id}] Waiting for NIXL ready flag at {key}...", "INFO")
        
        while time.time() - start_time < timeout_seconds:
            try:
                data = client.get(key)
                if data:
                    ready_info = json.loads(data)
                    if ready_info.get("nixl_ready") and ready_info.get("stability_verified"):
                        session_id = ready_info.get("session_id")
                        metadata_hash = ready_info.get("metadata_hash")
                        _log(
                            f"[Worker {worker_id}] Source ready! session={session_id[:8] if session_id else 'N/A'}..., "
                            f"hash={metadata_hash[:8] if metadata_hash else 'N/A'}...",
                            "INFO"
                        )
                        return True, session_id, metadata_hash
            except Exception as e:
                _log(f"[Worker {worker_id}] Error checking ready flag: {e}", "WARNING")
            
            time.sleep(poll_interval)
            elapsed = int(time.time() - start_time)
            if elapsed % 60 == 0:
                _log(f"[Worker {worker_id}] Still waiting for source ready ({elapsed}s/{timeout_seconds}s)...", "INFO")
        
        _log(f"[Worker {worker_id}] Timeout waiting for source ready after {timeout_seconds}s", "ERROR")
        return False, None, None
    
    @classmethod
    def check_session_changed(
        cls,
        model_name: str,
        worker_id: int,
        cached_session_id: str | None,
    ) -> tuple[bool, str | None]:
        """
        Check if source session changed (indicates restart).
        
        Returns:
            (changed, new_session_id)
        """
        if cached_session_id is None:
            return False, None
        
        client = cls._get_redis_client()
        if client is None:
            return False, None
        
        key = f"mx:nixl_ready:{model_name}:worker:{worker_id}"
        
        try:
            data = client.get(key)
            if data:
                ready_info = json.loads(data)
                current_session = ready_info.get("session_id")
                if current_session and current_session != cached_session_id:
                    _log(
                        f"[Worker {worker_id}] Source restarted! "
                        f"cached={cached_session_id[:8]}... != current={current_session[:8]}...",
                        "WARNING"
                    )
                    return True, current_session
        except Exception as e:
            _log(f"[Worker {worker_id}] Error checking session: {e}", "WARNING")
        
        return False, cached_session_id


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
        _log("MxSourceModelLoader.__init__ called!", "DEBUG")
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
        _log("MxSourceModelLoader initialized successfully", "DEBUG")

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        """Load model with NIXL registration before weight processing."""
        _log("MxSourceModelLoader.load_model() STARTING", "INFO")
        
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        _log(f"Target device: {target_device}", "DEBUG")

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                _log("Initializing model structure...", "INFO")
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )
                _log("Model structure initialized", "INFO")

            _log("Loading weights from disk (this may take 20-30 min for DeepSeek-V3)...", "INFO")
            self.load_weights(model, model_config)
            _log("Weights loaded from disk!", "INFO")

            # HOOK: Register RAW tensors before processing
            _log("=== HOOK: Registering raw tensors BEFORE FP8 processing ===", "INFO")
            self._register_raw_tensors(model, target_device)
            _log("=== Raw tensor registration complete ===", "INFO")

            # Now run FP8 processing (transforms weight_scale_inv -> weight_scale)
            _log("Processing weights (FP8 transformation)...", "INFO")
            process_weights_after_loading(model, model_config, target_device)
            _log("Weight processing complete!", "INFO")

        _log("MxSourceModelLoader.load_model() COMPLETE", "INFO")
        return model.eval()

    def _register_raw_tensors(
        self, model: nn.Module, device: torch.device
    ) -> None:
        """
        Register raw tensors with NIXL before FP8 processing.
        
        This captures weight_scale_inv tensors BEFORE they are deleted
        and replaced with weight_scale.
        """
        from .nixl_transfer import NixlTransferManager, is_nixl_available

        _log("_register_raw_tensors() called", "DEBUG")
        
        if not is_nixl_available():
            _log("NIXL not available, skipping raw tensor registration", "WARNING")
            return

        # Collect raw tensors (including weight_scale_inv)
        raw_tensors: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.is_cuda:
                raw_tensors[name] = param.data

        self._raw_tensors = raw_tensors
        device_id = self._get_worker_rank(device)
        
        total_size = sum(t.numel() * t.element_size() for t in raw_tensors.values())
        scale_count = sum(1 for n in raw_tensors if "scale_inv" in n.lower())
        
        _log(
            f"[Worker {device_id}] Registering {len(raw_tensors)} raw tensors "
            f"({total_size / 1e9:.2f} GB), including {scale_count} scale_inv tensors",
            "INFO"
        )
        
        # Debug: print first 5 tensor names and sample checksums
        tensor_names = list(raw_tensors.keys())
        _log(f"[Worker {device_id}] First 5 tensor names: {tensor_names[:5]}", "DEBUG")
        
        # Sample checksums for verification
        for name in tensor_names[:3]:
            t = raw_tensors[name]
            checksum = _safe_checksum(t)
            _log(f"[Worker {device_id}] Sample tensor '{name}': shape={t.shape}, dtype={t.dtype}, checksum={checksum}", "DEBUG")

        # Initialize NIXL manager if not already done
        if self._nixl_manager is None:
            import uuid
            agent_name = f"mx-source-worker{device_id}-{uuid.uuid4().hex[:8]}"
            _log(f"[Worker {device_id}] Initializing NIXL manager with agent_name={agent_name}", "DEBUG")
            self._nixl_manager = NixlTransferManager(
                agent_name=agent_name,
                device_id=device_id,
            )
            self._nixl_manager.initialize()
            _log(f"[Worker {device_id}] NIXL manager initialized", "DEBUG")

        # Register with NIXL for RDMA
        _log(f"[Worker {device_id}] Registering tensors with NIXL...", "DEBUG")
        self._nixl_manager.register_tensors(raw_tensors)
        _log(f"[Worker {device_id}] Tensors registered with NIXL", "DEBUG")

        # Store in global registry for client access
        _raw_tensor_registry[device_id] = raw_tensors
        _nixl_managers[device_id] = self._nixl_manager
        
        _log(f"[Worker {device_id}] Raw tensors stored in global registry", "INFO")
        
        # CRITICAL: Publish metadata to ModelExpress server for targets to discover
        # This was previously unreachable due to early return bug!
        
        # OPTIMIZATION: Source barrier - optionally wait for all workers before publishing
        # This ensures all source workers publish at approximately the same time,
        # reducing the staggered window and allowing target sync start to be more effective
        sync_publish = os.environ.get("MX_SYNC_PUBLISH", "0") == "1"
        expected_workers = int(os.environ.get("MX_EXPECTED_WORKERS", "8"))
        
        if sync_publish:
            _log(f"[Worker {device_id}] Synchronized publish enabled, signaling ready...", "INFO")
            self._wait_for_all_workers_ready(device_id, expected_workers)
        
        _log(f"[Worker {device_id}] Publishing metadata to ModelExpress server...", "INFO")
        self._publish_metadata_to_server(raw_tensors, device_id)

    def _get_worker_rank(self, device: torch.device) -> int:
        """Get the TP rank of this worker."""
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
            _log(f"Got TP rank from vllm.distributed: {rank}", "DEBUG")
            return rank
        except (ImportError, RuntimeError) as e:
            _log(f"Could not get TP rank from vllm.distributed: {e}", "DEBUG")
        
        # Fallback to device index
        if hasattr(device, "index") and device.index is not None:
            _log(f"Using device.index as rank: {device.index}", "DEBUG")
            return device.index
        
        _log("Defaulting to rank 0", "DEBUG")
        return 0

    def _wait_for_all_workers_ready(self, device_id: int, expected_workers: int) -> None:
        """
        Wait for all source workers to be ready before publishing.
        
        Uses vLLM's distributed barrier if available, otherwise uses
        a simple polling mechanism via the ModelExpress server.
        """
        try:
            # Try to use vLLM's distributed barrier (most efficient)
            from vllm.distributed import get_tensor_model_parallel_world_size
            import torch.distributed as dist
            
            world_size = get_tensor_model_parallel_world_size()
            if world_size == expected_workers and dist.is_initialized():
                _log(f"[Worker {device_id}] Using torch.distributed barrier for synchronization", "DEBUG")
                dist.barrier()
                _log(f"[Worker {device_id}] Barrier passed - all workers ready", "INFO")
                return
        except (ImportError, RuntimeError) as e:
            _log(f"[Worker {device_id}] torch.distributed barrier not available: {e}", "DEBUG")
        
        # Fallback: Signal readiness to server and poll until all workers ready
        # This uses the server's Redis to coordinate
        _log(f"[Worker {device_id}] Using server-based barrier (fallback)", "DEBUG")
        
        server_address = os.environ.get("MX_SERVER_ADDRESS", "modelexpress-server:8001")
        if server_address.startswith("http://"):
            server_address = server_address[7:]
        elif server_address.startswith("https://"):
            server_address = server_address[8:]
        model_name = os.environ.get("MODEL_NAME", "unknown")
        
        try:
            import grpc
            from . import p2p_pb2, p2p_pb2_grpc
            
            options = [
                ('grpc.max_send_message_length', 10 * 1024 * 1024),
                ('grpc.max_receive_message_length', 10 * 1024 * 1024),
            ]
            channel = grpc.insecure_channel(server_address, options=options)
            stub = p2p_pb2_grpc.P2pServiceStub(channel)
            
            # Poll until all workers have published (including us in a moment)
            # We check how many workers are already registered
            max_wait = 300  # 5 minute max wait for other workers
            waited = 0
            poll_interval = 2
            
            while waited < max_wait:
                request = p2p_pb2.GetMetadataRequest(model_name=model_name)
                response = stub.GetMetadata(request)
                
                if response.found:
                    ready_count = len([w for w in response.workers if len(w.tensors) > 0])
                    # We haven't published yet, so we expect (expected_workers - 1) others
                    if ready_count >= expected_workers - 1:
                        _log(f"[Worker {device_id}] All other workers ready ({ready_count}/{expected_workers-1}), proceeding", "INFO")
                        break
                    _log(f"[Worker {device_id}] Waiting for other workers: {ready_count}/{expected_workers-1} ready", "DEBUG")
                
                time.sleep(poll_interval)
                waited += poll_interval
            
            channel.close()
            
        except Exception as e:
            _log(f"[Worker {device_id}] Server barrier failed: {e}, proceeding anyway", "WARNING")

    def _publish_metadata_to_server(
        self, tensors: dict[str, torch.Tensor], device_id: int
    ) -> None:
        """Publish tensor metadata to ModelExpress server for targets to discover."""
        server_address = os.environ.get("MX_SERVER_ADDRESS", "modelexpress-server:8001")
        # Strip http:// prefix if present (gRPC doesn't use it)
        if server_address.startswith("http://"):
            server_address = server_address[7:]
        elif server_address.startswith("https://"):
            server_address = server_address[8:]
            
        model_name = os.environ.get("MODEL_NAME", "unknown")
        
        _log(f"[Worker {device_id}] _publish_metadata_to_server() called", "DEBUG")
        _log(f"[Worker {device_id}] Server address: {server_address}, Model: {model_name}", "DEBUG")
        _log(f"[Worker {device_id}] Publishing {len(tensors)} tensors", "DEBUG")
        
        try:
            import grpc
            from . import p2p_pb2, p2p_pb2_grpc
            
            # Check if contiguous region registration is enabled
            use_contiguous = os.environ.get("MX_CONTIGUOUS_REG", "0") == "1"
            
            if use_contiguous and self._nixl_manager is not None:
                # Publish REGION descriptors (what we actually registered)
                # Target will receive these and match regions by index
                region_descriptors = self._nixl_manager.get_registered_descriptors()
                tensor_protos = []
                for desc in region_descriptors:
                    tensor_protos.append(p2p_pb2.TensorDescriptor(
                        name=desc.name,  # __region_0__, __region_1__, etc.
                        addr=desc.addr,
                        size=desc.size,
                        device_id=desc.device_id,
                        dtype=desc.dtype,
                    ))
                _log(f"[Worker {device_id}] Built {len(tensor_protos)} REGION descriptors (MX_CONTIGUOUS_REG=1)", "INFO")
            else:
                # Build tensor descriptors from individual tensors (baseline)
                tensor_protos = []
                for name, t in tensors.items():
                    tensor_protos.append(p2p_pb2.TensorDescriptor(
                        name=name,
                        addr=t.data_ptr(),
                        size=t.numel() * t.element_size(),
                        device_id=device_id,
                        dtype=str(t.dtype),
                    ))
                _log(f"[Worker {device_id}] Built {len(tensor_protos)} tensor descriptors", "DEBUG")
            
            # Get NIXL metadata
            nixl_metadata = self._nixl_manager.nixl_metadata if self._nixl_manager else b""
            _log(f"[Worker {device_id}] NIXL metadata size: {len(nixl_metadata)} bytes", "DEBUG")
            
            # Build worker metadata
            worker = p2p_pb2.WorkerMetadata(
                worker_rank=device_id,
                nixl_metadata=nixl_metadata,
                tensors=tensor_protos,
            )
            
            # Publish to server
            options = [
                ('grpc.max_send_message_length', 100 * 1024 * 1024),
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
            ]
            _log(f"[Worker {device_id}] Connecting to gRPC server at {server_address}...", "DEBUG")
            channel = grpc.insecure_channel(server_address, options=options)
            stub = p2p_pb2_grpc.P2pServiceStub(channel)
            
            request = p2p_pb2.PublishMetadataRequest(
                model_name=model_name,
                workers=[worker],  # repeated field
            )
            _log(f"[Worker {device_id}] Sending PublishMetadataRequest...", "DEBUG")
            response = stub.PublishMetadata(request)
            
            if response.success:
                _log(f"[Worker {device_id}] SUCCESS: Published metadata to server {server_address}", "INFO")
                # NOTE: NIXL ready flag is published by source entrypoint AFTER:
                # 1. vLLM health check passes
                # 2. 30s grace period
                # 3. Successful test inference proves stability
            else:
                _log(f"[Worker {device_id}] FAILED: Server returned error: {response.message}", "ERROR")
            
            channel.close()
            _log(f"[Worker {device_id}] gRPC channel closed", "DEBUG")
            
        except Exception as e:
            import traceback
            _log(f"[Worker {device_id}] EXCEPTION publishing metadata: {e}", "ERROR")
            _log(f"[Worker {device_id}] Traceback: {traceback.format_exc()}", "ERROR")
    
    @property
    def nixl_manager(self) -> NixlTransferManager | None:
        """Access the NIXL manager for external use."""
        return self._nixl_manager

    @property
    def raw_tensors(self) -> dict[str, torch.Tensor]:
        """Access the raw tensor registry."""
        return self._raw_tensors


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
        _log("MxTargetModelLoader.__init__ called!", "DEBUG")
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
        _log("MxTargetModelLoader initialized successfully", "DEBUG")

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        """Load model with RDMA transfer before weight processing."""
        import time as _time
        load_start = _time.perf_counter()
        _log("=" * 60, "INFO")
        _log("MxTargetModelLoader.load_model() STARTING", "INFO")
        _log("=" * 60, "INFO")
        
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        _log(f"Target device: {target_device}", "DEBUG")

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                t0 = _time.perf_counter()
                _log("[TIMING] Initializing model structure...", "INFO")
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )
                _log(f"[TIMING] Model structure initialized in {_time.perf_counter() - t0:.2f}s", "INFO")

            t0 = _time.perf_counter()
            _log("[TIMING] Creating dummy weights...", "INFO")
            self.load_weights(model, model_config)
            _log(f"[TIMING] Dummy weights created in {_time.perf_counter() - t0:.2f}s", "INFO")

            # HOOK: Receive RAW tensors via RDMA before processing
            t0 = _time.perf_counter()
            _log("=" * 60, "INFO")
            _log("[TIMING] === HOOK: Receiving raw tensors via RDMA ===", "INFO")
            _log("=" * 60, "INFO")
            self._receive_raw_tensors(model, target_device)
            rdma_time = _time.perf_counter() - t0
            _log(f"[TIMING] === RDMA reception complete in {rdma_time:.2f}s ===", "INFO")

            # Now run FP8 processing (transforms weight_scale_inv -> weight_scale)
            # This will produce IDENTICAL results to source since we have same raw data
            t0 = _time.perf_counter()
            _log("[TIMING] Processing weights (FP8 transformation)...", "INFO")
            process_weights_after_loading(model, model_config, target_device)
            _log(f"[TIMING] Weight processing complete in {_time.perf_counter() - t0:.2f}s", "INFO")

        total_time = _time.perf_counter() - load_start
        _log("=" * 60, "INFO")
        _log(f"[TIMING] MxTargetModelLoader.load_model() COMPLETE", "INFO")
        _log(f"[TIMING] Total load time: {total_time:.2f}s", "INFO")
        _log("=" * 60, "INFO")
        return model.eval()

    def _receive_raw_tensors(
        self, model: nn.Module, device: torch.device
    ) -> None:
        """
        Receive raw tensors via RDMA from source.
        
        The source has registered raw tensors (including weight_scale_inv).
        We receive them into our dummy tensors, then let vLLM process them.
        """
        from .nixl_transfer import NixlTransferManager, is_nixl_available
        from .types import TensorDescriptor

        import time as _time
        receive_start = _time.perf_counter()
        _log("_receive_raw_tensors() called", "DEBUG")

        if not is_nixl_available():
            _log("NIXL not available, skipping RDMA transfer", "WARNING")
            return

        # Get source info from environment
        mx_server_url = os.environ.get("MX_SERVER_ADDRESS", "modelexpress-server:8001")
        # Strip http:// prefix if present (gRPC doesn't use it)
        if mx_server_url.startswith("http://"):
            mx_server_url = mx_server_url[7:]
        elif mx_server_url.startswith("https://"):
            mx_server_url = mx_server_url[8:]
        model_name = os.environ.get("MODEL_NAME", "")

        _log(f"Server URL: {mx_server_url}, Model: {model_name}", "DEBUG")

        if not model_name:
            _log("MODEL_NAME not set, skipping transfer", "WARNING")
            return

        # Collect target tensors (these have dummy data)
        t0 = _time.perf_counter()
        target_tensors: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.is_cuda:
                target_tensors[name] = param.data

        device_id = self._get_worker_rank(device)
        scale_count = sum(1 for n in target_tensors if "scale_inv" in n.lower())
        total_size = sum(t.numel() * t.element_size() for t in target_tensors.values())
        _log(f"[Worker {device_id}] [TIMING] Collected {len(target_tensors)} tensors in {_time.perf_counter() - t0:.3f}s", "DEBUG")
        
        _log(
            f"[Worker {device_id}] Target has {len(target_tensors)} tensors "
            f"({total_size / 1e9:.2f} GB), including {scale_count} scale_inv tensors",
            "INFO"
        )
        
        # Debug: print first 5 tensor names and checksums BEFORE transfer
        tensor_names = list(target_tensors.keys())
        _log(f"[Worker {device_id}] First 5 tensor names: {tensor_names[:5]}", "DEBUG")
        
        _log(f"[Worker {device_id}] Checksums BEFORE transfer:", "DEBUG")
        pre_checksums = {}
        for name in tensor_names[:3]:
            t = target_tensors[name]
            checksum = _safe_checksum(t)
            pre_checksums[name] = checksum
            _log(f"[Worker {device_id}]   '{name}': shape={t.shape}, dtype={t.dtype}, checksum={checksum}", "DEBUG")

        # Initialize NIXL manager
        import uuid
        t0 = _time.perf_counter()
        agent_name = f"mx-target-worker{device_id}-{uuid.uuid4().hex[:8]}"
        _log(f"[Worker {device_id}] [TIMING] Initializing NIXL manager with agent_name={agent_name}", "DEBUG")
        self._nixl_manager = NixlTransferManager(
            agent_name=agent_name,
            device_id=device_id,
        )
        self._nixl_manager.initialize()
        nixl_init_time = _time.perf_counter() - t0
        _log(f"[Worker {device_id}] [TIMING] NIXL manager initialized in {nixl_init_time:.3f}s", "INFO")
        
        t0 = _time.perf_counter()
        _log(f"[Worker {device_id}] [TIMING] Registering target tensors with NIXL...", "DEBUG")
        self._nixl_manager.register_tensors(target_tensors)
        reg_time = _time.perf_counter() - t0
        _log(f"[Worker {device_id}] [TIMING] Target tensors registered in {reg_time:.3f}s", "INFO")

        # COORDINATION: Wait for NIXL ready flag before initiating transfer
        # The nixl_ready flag is only set AFTER:
        # 1. vLLM health endpoint returns 200
        # 2. 30s grace period for system stabilization
        # 3. Successful test inference proves stability
        _log(f"[Worker {device_id}] Waiting for source NIXL ready (includes stability verification)...", "INFO")
        source_ready, cached_session_id, cached_metadata_hash = SourceReadyCoordinator.wait_for_source_ready(
            model_name=model_name,
            worker_id=device_id,
            timeout_seconds=7200,  # 2 hour timeout (matches source warmup)
            poll_interval=10,
        )
        
        if not source_ready:
            _log(f"[Worker {device_id}] ERROR: Source NIXL never became ready, cannot proceed", "ERROR")
            return
        
        _log(f"[Worker {device_id}] Source NIXL ready (stability verified), proceeding with transfer", "INFO")
        
        # Connect to ModelExpress server and find source via gRPC
        # Retry with backoff - source takes 20-30 min to load DeepSeek-V3
        max_wait_time = 3600  # 1 hour max wait
        retry_interval = 30  # 30 seconds between retries
        total_waited = 0
        wait_start = _time.perf_counter()
        
        # OPTIMIZATION: Synchronized start - wait for ALL source workers before transferring
        # This ensures all 8 target workers start their transfers simultaneously,
        # maximizing RDMA parallelism and achieving closer to theoretical bandwidth
        sync_start = os.environ.get("MX_SYNC_START", "1") == "1"
        expected_workers = int(os.environ.get("MX_EXPECTED_WORKERS", "8"))
        
        if sync_start:
            _log(f"[Worker {device_id}] [TIMING] Synchronized start enabled, waiting for all {expected_workers} source workers", "INFO")
        
        try:
            import grpc
            from . import p2p_pb2, p2p_pb2_grpc

            options = [
                ('grpc.max_send_message_length', 100 * 1024 * 1024),
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
            ]
            _log(f"[Worker {device_id}] Connecting to gRPC server at {mx_server_url}...", "DEBUG")
            channel = grpc.insecure_channel(mx_server_url, options=options)
            stub = p2p_pb2_grpc.P2pServiceStub(channel)
            _log(f"[Worker {device_id}] gRPC channel created", "DEBUG")

            response = None
            source_worker = None
            all_workers_ready = False
            
            while total_waited < max_wait_time:
                # Query for source
                _log(f"[Worker {device_id}] Querying for source model: {model_name}...", "DEBUG")
                request = p2p_pb2.GetMetadataRequest(model_name=model_name)
                response = stub.GetMetadata(request)

                if response.found and len(response.workers) > 0:
                    available_ranks = sorted([w.worker_rank for w in response.workers])
                    workers_with_tensors = sum(1 for w in response.workers if len(w.tensors) > 0)
                    _log(f"[Worker {device_id}] Response: found={response.found}, workers={len(response.workers)}/{expected_workers}, ranks={available_ranks}", "DEBUG")
                    
                    # OPTIMIZATION: Check if ALL expected workers are ready (synchronized start)
                    if sync_start and workers_with_tensors < expected_workers:
                        _log(f"[Worker {device_id}] Waiting for all workers: {workers_with_tensors}/{expected_workers} ready", "INFO")
                        time.sleep(retry_interval)
                        total_waited += retry_interval
                        continue
                    
                    if sync_start and workers_with_tensors >= expected_workers and not all_workers_ready:
                        _log(f"[Worker {device_id}] ALL {expected_workers} source workers are ready! Starting synchronized transfer.", "INFO")
                        all_workers_ready = True
                    
                    # Find source worker for OUR specific rank
                    for w in response.workers:
                        if w.worker_rank == device_id and len(w.tensors) > 0:
                            source_worker = w
                            break
                    
                    if source_worker:
                        _log(f"[Worker {device_id}] Found matching source worker for rank {device_id} with {len(source_worker.tensors)} tensors!", "INFO")
                        break
                    else:
                        _log(f"[Worker {device_id}] Source has {len(response.workers)} workers but rank {device_id} not ready yet. Available: {available_ranks}", "INFO")
                else:
                    _log(f"[Worker {device_id}] No source found yet (found={response.found if response else 'N/A'}), waiting...", "INFO")
                
                time.sleep(retry_interval)
                total_waited += retry_interval
                if total_waited % 60 == 0:  # Log every 60s instead of every retry
                    _log(f"[Worker {device_id}] Waited {total_waited}s for source rank {device_id} (max {max_wait_time}s)...", "INFO")

            if not source_worker:
                available_ranks = [w.worker_rank for w in response.workers] if response and response.workers else []
                _log(f"[Worker {device_id}] ERROR: No source worker found for rank {device_id} after {total_waited}s. Available: {available_ranks}", "ERROR")
                channel.close()
                return

            wait_time = _time.perf_counter() - wait_start
            _log(f"[Worker {device_id}] [TIMING] Found source worker after waiting {wait_time:.2f}s", "INFO")
            _log(f"[Worker {device_id}] Found source worker for rank {device_id} with {len(source_worker.tensors)} tensors", "INFO")

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

            _log(f"[Worker {device_id}] Receiving {len(source_tensors)} tensors from source worker {device_id}", "INFO")
            _log(f"[Worker {device_id}] First 3 source tensor names: {[t.name for t in source_tensors[:3]]}", "DEBUG")

            # Perform RDMA transfer with retry for transient failures
            transfer_retries = 120  # 120 * 30s = 60 min max wait
            transfer_retry_delay = 30
            bytes_transferred = 0
            tensor_count = 0
            
            for attempt in range(transfer_retries):
                try:
                    _log(f"[Worker {device_id}] [TIMING] Starting RDMA transfer attempt {attempt + 1}...", "INFO")
                    transfer_start = _time.perf_counter()
                    # MX_CONTIGUOUS_REG=1 enables contiguous region registration on BOTH
                    # source and target. This allows transfer-time coalescing to work.
                    coalesce = os.environ.get("MX_CONTIGUOUS_REG", "0") == "1"
                    _log(f"[Worker {device_id}] Coalesce transfers: {coalesce} (MX_CONTIGUOUS_REG={os.environ.get('MX_CONTIGUOUS_REG', 'not set')})", "DEBUG")
                    bytes_transferred, tensor_count, _ = self._nixl_manager.receive_from_source(
                        source_metadata=source_worker.nixl_metadata,
                        source_tensors=source_tensors,
                        timeout_seconds=self._transfer_timeout,
                        coalesce_transfers=coalesce,
                    )
                    transfer_time = _time.perf_counter() - transfer_start

                    bandwidth_gbps = (bytes_transferred * 8) / (transfer_time * 1e9) if transfer_time > 0 else 0
                    _log("=" * 60, "INFO")
                    _log(
                        f"[Worker {device_id}] [TIMING] RDMA TRANSFER COMPLETE:",
                        "INFO"
                    )
                    _log(f"[Worker {device_id}] [TIMING]   Tensors: {tensor_count}", "INFO")
                    _log(f"[Worker {device_id}] [TIMING]   Data: {bytes_transferred / 1e9:.2f} GB", "INFO")
                    _log(f"[Worker {device_id}] [TIMING]   Time: {transfer_time:.3f}s", "INFO")
                    _log(f"[Worker {device_id}] [TIMING]   Bandwidth: {bandwidth_gbps:.1f} Gbps", "INFO")
                    _log("=" * 60, "INFO")
                    
                    # Sync CUDA to ensure transfer is visible
                    torch.cuda.synchronize()
                    _log(f"[Worker {device_id}] CUDA synchronized", "DEBUG")
                    
                    # Verify checksums AFTER transfer
                    _log(f"[Worker {device_id}] Checksums AFTER transfer:", "DEBUG")
                    for name in tensor_names[:3]:
                        t = target_tensors[name]
                        checksum = _safe_checksum(t)
                        changed = "CHANGED" if checksum != pre_checksums.get(name) else "UNCHANGED"
                        _log(f"[Worker {device_id}]   '{name}': checksum={checksum} ({changed})", "DEBUG")
                    
                    break  # Success
                except Exception as transfer_err:
                    if attempt < transfer_retries - 1:
                        _log(
                            f"[Worker {device_id}] Transfer attempt {attempt + 1} failed: {transfer_err}, "
                            f"retrying in {transfer_retry_delay}s...",
                            "WARNING"
                        )
                        
                        # Check if source restarted (stale metadata issue)
                        session_changed, new_session_id = SourceReadyCoordinator.check_session_changed(
                            model_name=model_name,
                            worker_id=device_id,
                            cached_session_id=cached_session_id,
                        )
                        
                        if session_changed:
                            _log(
                                f"[Worker {device_id}] Source restarted detected! Re-fetching metadata...",
                                "WARNING"
                            )
                            cached_session_id = new_session_id
                            
                            # Re-fetch metadata from gRPC server
                            request = p2p_pb2.GetMetadataRequest(model_name=model_name)
                            response = stub.GetMetadata(request)
                            
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
                                _log(
                                    f"[Worker {device_id}] Refreshed metadata: {len(source_tensors)} tensors from new session",
                                    "INFO"
                                )
                            else:
                                _log(
                                    f"[Worker {device_id}] Could not find source worker after restart, will retry",
                                    "WARNING"
                                )
                        
                        time.sleep(transfer_retry_delay)
                    else:
                        _log(f"[Worker {device_id}] Transfer failed after {transfer_retries} attempts: {transfer_err}", "ERROR")
                        raise RuntimeError(
                            f"Transfer failed after {transfer_retries} attempts: {transfer_err}"
                        )

            channel.close()
            _log(f"[Worker {device_id}] gRPC channel closed", "DEBUG")
            
            # Final timing summary
            total_receive_time = _time.perf_counter() - receive_start
            _log("=" * 60, "INFO")
            _log(f"[Worker {device_id}] [TIMING] _receive_raw_tensors SUMMARY:", "INFO")
            _log(f"[Worker {device_id}] [TIMING]   Total time: {total_receive_time:.2f}s", "INFO")
            _log(f"[Worker {device_id}] [TIMING]   NIXL init: {nixl_init_time:.3f}s", "INFO")
            _log(f"[Worker {device_id}] [TIMING]   Tensor reg: {reg_time:.3f}s", "INFO")
            _log(f"[Worker {device_id}] [TIMING]   Wait for source: {wait_time:.2f}s", "INFO")
            _log(f"[Worker {device_id}] [TIMING]   RDMA transfer: {transfer_time:.3f}s", "INFO")
            _log("=" * 60, "INFO")

        except Exception as e:
            import traceback
            _log(f"[Worker {device_id}] EXCEPTION receiving weights: {e}", "ERROR")
            _log(f"[Worker {device_id}] Traceback: {traceback.format_exc()}", "ERROR")
            raise

    def _get_worker_rank(self, device: torch.device) -> int:
        """Get the TP rank of this worker."""
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
            _log(f"Got TP rank from vllm.distributed: {rank}", "DEBUG")
            return rank
        except (ImportError, RuntimeError) as e:
            _log(f"Could not get TP rank from vllm.distributed: {e}", "DEBUG")
        
        # Fallback to device index
        if hasattr(device, "index") and device.index is not None:
            _log(f"Using device.index as rank: {device.index}", "DEBUG")
            return device.index
        
        _log("Defaulting to rank 0", "DEBUG")
        return 0

    @property
    def nixl_manager(self) -> NixlTransferManager | None:
        """Access the NIXL manager for external use."""
        return self._nixl_manager


# Global storage for raw tensor metadata (used by source loader)
# This allows the existing client to access pre-processing metadata
_raw_tensor_registry: dict[int, dict[str, torch.Tensor]] = {}
_nixl_managers: dict[int, "NixlTransferManager"] = {}


def get_raw_tensor_registry(device_id: int) -> dict[str, torch.Tensor] | None:
    """Get raw tensor registry for a device (used by client to publish pre-processing metadata)."""
    return _raw_tensor_registry.get(device_id)


def get_nixl_manager(device_id: int) -> "NixlTransferManager | None":
    """Get NIXL manager for a device."""
    return _nixl_managers.get(device_id)
