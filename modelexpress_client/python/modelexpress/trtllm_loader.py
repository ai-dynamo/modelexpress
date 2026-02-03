# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress TensorRT-LLM Integration.

Enables P2P weight transfer for TRT-LLM checkpoints via NIXL RDMA.

Architecture:
    TRT-LLM uses a three-phase workflow: Convert → Build → Run
    - Weights are stored in checkpoint format (config.json + rank*.safetensors)
    - Checkpoints are compiled into TensorRT engines
    - Engines run via the Executor/LLM API

This module provides:
    - MxTrtllmSourcePublisher: Loads checkpoint to GPU, registers with NIXL
    - MxTrtllmTargetLoader: Receives checkpoint via RDMA, optionally builds engine

Usage:
    # Source side - publish checkpoint for P2P transfer
    publisher = MxTrtllmSourcePublisher(
        checkpoint_dir="/path/to/checkpoint",
        model_name="llama-70b",
        mx_server="modelexpress-server:8001"
    )
    publisher.initialize()
    
    # Target side - receive checkpoint and build engine
    loader = MxTrtllmTargetLoader(
        model_name="llama-70b",
        mx_server="modelexpress-server:8001",
        output_dir="/path/to/output"
    )
    engine_dir = loader.load()
    
    # Use with TRT-LLM
    from tensorrt_llm import LLM
    llm = LLM(model=engine_dir)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.trtllm_loader")


def _parse_server_address(address: str) -> str:
    """Strip http:// or https:// prefix from server address."""
    if address.startswith("http://"):
        return address[7:]
    elif address.startswith("https://"):
        return address[8:]
    return address


class MxTrtllmSourcePublisher:
    """
    Publishes TRT-LLM engine or checkpoint for P2P transfer.
    
    This class supports two modes:
    
    1. **Engine mode** (recommended): Loads a pre-built TensorRT engine's weights
       to GPU and publishes them. Target receives weights and can run immediately
       without building. Requires source and target to have same GPU architecture.
       
    2. **Checkpoint mode**: Loads checkpoint files (safetensors) to GPU. Target
       receives checkpoint and must build engine locally (slower).
    
    The weights remain in GPU memory for the lifetime of this object,
    allowing multiple targets to receive them.
    
    Args:
        checkpoint_dir: Path to TRT-LLM checkpoint directory
        model_name: Model identifier for coordination
        mx_server: ModelExpress server address (host:port)
        engine_dir: Optional path to pre-built engine directory. If provided,
                    publishes the engine + weights for direct transfer.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str,
        mx_server: str = "modelexpress-server:8001",
        engine_dir: str | None = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.mx_server = _parse_server_address(mx_server)
        self.engine_dir = Path(engine_dir) if engine_dir else None
        self.config: dict = {}
        self.nixl_managers: dict[int, NixlTransferManager] = {}
        self.weights: dict[int, dict[str, torch.Tensor]] = {}
        self._initialized = False
        self._engine_bytes: bytes | None = None  # Serialized engine for transfer
        
    def initialize(self, build_if_needed: bool = True) -> None:
        """
        Load checkpoint/engine and register with NIXL.
        
        This method:
        1. If engine_dir provided: loads engine and prepares for direct transfer
        2. If no engine but build_if_needed: builds engine first, then loads
        3. Loads weights to GPU and registers with NIXL
        4. Publishes metadata to ModelExpress server
        
        Args:
            build_if_needed: If True and no engine exists, build it first
        """
        from .nixl_transfer import NixlTransferManager, is_nixl_available
        
        if not is_nixl_available():
            raise RuntimeError("NIXL is not available. Install with: pip install nixl[cu12]")
        
        if self._initialized:
            logger.warning("Already initialized, skipping")
            return
        
        logger.info(f"Initializing TRT-LLM source publisher for {self.model_name}")
        
        # 1. Parse config.json
        config_path = self.checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path) as f:
            self.config = json.load(f)
        
        world_size = self.config.get("mapping", {}).get("world_size", 1)
        tp_size = self.config.get("mapping", {}).get("tp_size", 1)
        pp_size = self.config.get("mapping", {}).get("pp_size", 1)
        
        logger.info(f"Config: world_size={world_size}, tp_size={tp_size}, pp_size={pp_size}")
        
        # 2. Check for engine or build if needed
        if self.engine_dir and self.engine_dir.exists():
            logger.info(f"Using pre-built engine from {self.engine_dir}")
            self._load_engine_for_transfer()
        elif build_if_needed:
            logger.info("No engine found, building from checkpoint...")
            self._build_engine()
            self._load_engine_for_transfer()
        else:
            logger.info("Checkpoint-only mode (no engine transfer)")
        
        # 3. Load and register each rank's weights
        total_bytes = 0
        for rank in range(world_size):
            weights_path = self.checkpoint_dir / f"rank{rank}.safetensors"
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights not found: {weights_path}")
            
            logger.info(f"Loading rank {rank} weights from {weights_path}")
            weights = self._load_safetensors_to_gpu(weights_path, rank)
            self.weights[rank] = weights
            
            rank_bytes = sum(t.numel() * t.element_size() for t in weights.values())
            total_bytes += rank_bytes
            logger.info(f"Rank {rank}: {len(weights)} tensors, {rank_bytes / 1e9:.2f} GB")
            
            # Initialize NIXL manager
            import uuid
            agent_name = f"trtllm-source-rank{rank}-{uuid.uuid4().hex[:8]}"
            nixl_manager = NixlTransferManager(
                agent_name=agent_name,
                device_id=rank
            )
            nixl_manager.initialize()
            nixl_manager.register_tensors(weights)
            self.nixl_managers[rank] = nixl_manager
            logger.info(f"Rank {rank}: Registered {len(weights)} tensors with NIXL")
        
        logger.info(f"Total: {total_bytes / 1e9:.2f} GB across {world_size} ranks")
        
        # 4. Publish to MX server
        self._publish_to_mx_server()
        
        self._initialized = True
        logger.info("TRT-LLM source publisher initialized and ready")
    
    def _build_engine(self) -> None:
        """Build TRT-LLM engine from checkpoint."""
        import subprocess
        
        if self.engine_dir is None:
            self.engine_dir = self.checkpoint_dir.parent / "engine"
        
        self.engine_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Building engine from {self.checkpoint_dir} to {self.engine_dir}")
        
        cmd = [
            "trtllm-build",
            "--checkpoint_dir", str(self.checkpoint_dir),
            "--output_dir", str(self.engine_dir),
            "--gemm_plugin", "auto",
            "--max_batch_size", "8",
            "--max_input_len", "2048", 
            "--max_seq_len", "4096",
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        build_time = time.time() - t0
        
        if result.returncode != 0:
            logger.error(f"Engine build failed:\n{result.stderr}")
            raise RuntimeError(f"trtllm-build failed: {result.stderr}")
        
        logger.info(f"Engine built in {build_time:.1f}s")
    
    def _load_engine_for_transfer(self) -> None:
        """Load engine file for transfer to targets."""
        if self.engine_dir is None:
            return
        
        # Find engine file (usually rank0.engine or similar)
        engine_files = list(self.engine_dir.glob("*.engine"))
        if not engine_files:
            logger.warning("No .engine files found in engine directory")
            return
        
        # Read engine bytes for transfer
        engine_file = engine_files[0]
        logger.info(f"Loading engine file: {engine_file}")
        
        with open(engine_file, "rb") as f:
            self._engine_bytes = f.read()
        
        logger.info(f"Engine loaded: {len(self._engine_bytes) / 1e6:.1f} MB")
    
    def _load_safetensors_to_gpu(
        self, path: Path, device_id: int
    ) -> dict[str, torch.Tensor]:
        """Load safetensors file directly to GPU memory."""
        try:
            import safetensors.torch
        except ImportError:
            raise ImportError("safetensors not installed. Install with: pip install safetensors")
        
        with torch.cuda.device(device_id):
            weights = safetensors.torch.load_file(
                str(path), device=f"cuda:{device_id}"
            )
        return weights
    
    def _publish_to_mx_server(self) -> None:
        """Publish tensor metadata to ModelExpress server."""
        import grpc
        from . import p2p_pb2, p2p_pb2_grpc
        
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
        
        logger.info(f"Connecting to ModelExpress server at {self.mx_server}")
        channel = grpc.insecure_channel(self.mx_server, options=options)
        stub = p2p_pb2_grpc.P2pServiceStub(channel)
        
        workers = []
        for rank, nixl_manager in self.nixl_managers.items():
            weights = self.weights[rank]
            
            tensor_protos = []
            for name, tensor in weights.items():
                tensor_protos.append(p2p_pb2.TensorDescriptor(
                    name=name,
                    addr=tensor.data_ptr(),
                    size=tensor.numel() * tensor.element_size(),
                    device_id=rank,
                    dtype=str(tensor.dtype),
                ))
            
            workers.append(p2p_pb2.WorkerMetadata(
                worker_rank=rank,
                nixl_metadata=nixl_manager.nixl_metadata,
                tensors=tensor_protos,
            ))
        
        # Note: model_config is not in the proto - target reads config from checkpoint
        request = p2p_pb2.PublishMetadataRequest(
            model_name=self.model_name,
            workers=workers,
        )
        
        response = stub.PublishMetadata(request)
        
        if not response.success:
            channel.close()
            raise RuntimeError(f"Failed to publish metadata: {response.message}")
        
        channel.close()
        logger.info(f"Published metadata for {len(workers)} workers to ModelExpress server")
    
    def get_engine_bytes(self) -> bytes | None:
        """Get the serialized engine bytes for transfer."""
        return self._engine_bytes
    
    def shutdown(self) -> None:
        """Clean up NIXL resources and GPU memory."""
        logger.info("Shutting down TRT-LLM source publisher")
        
        for rank, nixl_manager in self.nixl_managers.items():
            nixl_manager.shutdown()
            logger.debug(f"Rank {rank}: NIXL manager shutdown")
        
        self.nixl_managers.clear()
        self.weights.clear()
        self._initialized = False
        
        # Force garbage collection to free GPU memory
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("TRT-LLM source publisher shutdown complete")
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


class MxTrtllmTargetLoader:
    """
    Receives TRT-LLM checkpoint via P2P and optionally builds engine.
    
    This class queries the ModelExpress server for source metadata,
    receives checkpoint weights via NIXL RDMA, saves them locally,
    and optionally builds a TRT-LLM engine.
    
    Args:
        model_name: Model identifier in ModelExpress
        mx_server: ModelExpress server address (host:port)
        output_dir: Local directory for checkpoint and engine
        build_config: Optional trtllm-build configuration
    """
    
    def __init__(
        self,
        model_name: str,
        mx_server: str = "modelexpress-server:8001",
        output_dir: str = "/tmp/mx_trtllm",
        build_config: dict | None = None,
    ):
        self.model_name = model_name
        self.mx_server = _parse_server_address(mx_server)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoint"
        self.engine_dir = self.output_dir / "engine"
        
        # Default build configuration
        self.build_config = build_config or {
            "gemm_plugin": "auto",
            "max_batch_size": "8",
            "max_input_len": "2048",
            "max_seq_len": "4096",
        }
        
        self.config: dict = {}
        self.transfer_stats: dict = {}
    
    def load(self, skip_build: bool = False) -> str:
        """
        Receive weights via P2P and optionally build engine.
        
        If the source has a pre-built engine, we skip the build step entirely
        and just load the transferred engine (much faster).
        
        Args:
            skip_build: If True, only receive checkpoint without building engine
            
        Returns:
            Path to checkpoint directory (if skip_build) or engine directory
        """
        logger.info(f"Loading {self.model_name} via ModelExpress P2P transfer")
        
        # 1. Query MX server for source
        source_meta = self._query_source()
        
        # 2. Save config (remove internal _mx_ fields)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Receive weights via NIXL
        self._receive_checkpoint(source_meta)
        
        # 4. Read config from transferred checkpoint
        config_path = self.checkpoint_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
            logger.info(f"Loaded config from checkpoint: {list(self.config.keys())}")
        else:
            logger.warning("No config.json in checkpoint, using empty config")
        
        if skip_build:
            logger.info(f"Checkpoint saved to {self.checkpoint_dir}")
            return str(self.checkpoint_dir)
        
        # 5. Build engine locally
        engine_path = self._build_engine()
        logger.info(f"Engine built at {engine_path}")
        return engine_path
    
    def _query_source(self, timeout: int = 3600):
        """Query ModelExpress server for source metadata."""
        import grpc
        from . import p2p_pb2, p2p_pb2_grpc
        
        options = [
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
        
        logger.info(f"Connecting to ModelExpress server at {self.mx_server}")
        channel = grpc.insecure_channel(self.mx_server, options=options)
        stub = p2p_pb2_grpc.P2pServiceStub(channel)
        
        logger.info(f"Querying for source model: {self.model_name}")
        
        start = time.time()
        poll_interval = 30
        
        while time.time() - start < timeout:
            try:
                request = p2p_pb2.GetMetadataRequest(model_name=self.model_name)
                response = stub.GetMetadata(request)
                
                if response.found and len(response.workers) > 0:
                    logger.info(f"Found source with {len(response.workers)} workers")
                    # Config will be read from the transferred checkpoint
                    self.config = {}
                    channel.close()
                    return response
                
                elapsed = int(time.time() - start)
                logger.info(f"Source not ready, waiting... ({elapsed}s / {timeout}s)")
                
            except grpc.RpcError as e:
                logger.warning(f"gRPC error: {e}, retrying...")
            
            time.sleep(poll_interval)
        
        channel.close()
        raise TimeoutError(f"Source not found for {self.model_name} after {timeout}s")
    
    def _receive_checkpoint(self, source_meta) -> None:
        """Receive checkpoint weights via NIXL."""
        from .nixl_transfer import NixlTransferManager
        from .types import TensorDescriptor
        
        total_bytes = 0
        total_time = 0
        
        for worker in source_meta.workers:
            rank = worker.worker_rank
            tensor_count = len(worker.tensors)
            
            logger.info(f"Receiving rank {rank}: {tensor_count} tensors")
            
            # Initialize NIXL manager for this rank
            import uuid
            agent_name = f"trtllm-target-rank{rank}-{uuid.uuid4().hex[:8]}"
            nixl_manager = NixlTransferManager(
                agent_name=agent_name,
                device_id=rank
            )
            nixl_manager.initialize()
            
            # Allocate GPU tensors to receive into
            weights = self._allocate_tensors(worker.tensors, rank)
            nixl_manager.register_tensors(weights)
            
            # Build source descriptors
            source_tensors = [
                TensorDescriptor(
                    name=t.name,
                    addr=t.addr,
                    size=t.size,
                    device_id=t.device_id,
                    dtype=t.dtype,
                )
                for t in worker.tensors
            ]
            
            # Receive via RDMA
            t0 = time.perf_counter()
            bytes_received, _, duration = nixl_manager.receive_from_source(
                source_metadata=worker.nixl_metadata,
                source_tensors=source_tensors,
                timeout_seconds=600,  # 10 minute timeout per rank
                coalesce_transfers=True,
            )
            transfer_time = time.perf_counter() - t0
            
            total_bytes += bytes_received
            total_time += transfer_time
            
            bandwidth = (bytes_received * 8) / (transfer_time * 1e9) if transfer_time > 0 else 0
            logger.info(
                f"Rank {rank}: Received {bytes_received / 1e9:.2f} GB "
                f"in {transfer_time:.2f}s ({bandwidth:.1f} Gbps)"
            )
            
            # Save weights to safetensors
            self._save_weights(weights, rank)
            
            # Cleanup NIXL
            nixl_manager.shutdown()
        
        # Store transfer stats
        self.transfer_stats = {
            "total_bytes": total_bytes,
            "total_time": total_time,
            "bandwidth_gbps": (total_bytes * 8) / (total_time * 1e9) if total_time > 0 else 0,
        }
        
        logger.info(
            f"Transfer complete: {total_bytes / 1e9:.2f} GB total "
            f"in {total_time:.2f}s ({self.transfer_stats['bandwidth_gbps']:.1f} Gbps)"
        )
    
    def _allocate_tensors(
        self, tensor_protos, device_id: int
    ) -> dict[str, torch.Tensor]:
        """Allocate GPU tensors matching source layout."""
        weights = {}
        
        with torch.cuda.device(device_id):
            for t in tensor_protos:
                dtype = self._parse_dtype(t.dtype)
                elem_size = self._dtype_size(dtype)
                numel = t.size // elem_size
                
                # Create flat tensor (TRT-LLM handles reshaping)
                weights[t.name] = torch.empty(
                    numel, dtype=dtype, device=f"cuda:{device_id}"
                )
        
        return weights
    
    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        """Parse dtype string to torch.dtype."""
        dtype_map = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.bfloat16": torch.bfloat16,
            "torch.int8": torch.int8,
            "torch.int32": torch.int32,
            "torch.int64": torch.int64,
        }
        
        # Handle float8 types if available
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["torch.float8_e4m3fn"] = torch.float8_e4m3fn
        if hasattr(torch, "float8_e5m2"):
            dtype_map["torch.float8_e5m2"] = torch.float8_e5m2
        
        return dtype_map.get(dtype_str, torch.float16)
    
    def _dtype_size(self, dtype: torch.dtype) -> int:
        """Get size in bytes for dtype."""
        return torch.tensor([], dtype=dtype).element_size()
    
    def _save_weights(self, weights: dict[str, torch.Tensor], rank: int) -> None:
        """Save weights to safetensors file."""
        try:
            import safetensors.torch
        except ImportError:
            raise ImportError("safetensors not installed. Install with: pip install safetensors")
        
        # Move to CPU for saving
        cpu_weights = {k: v.cpu() for k, v in weights.items()}
        
        weights_path = self.checkpoint_dir / f"rank{rank}.safetensors"
        safetensors.torch.save_file(cpu_weights, str(weights_path))
        logger.info(f"Saved rank {rank} weights to {weights_path}")
        
        # Free GPU memory
        del weights
        torch.cuda.empty_cache()
    
    def _build_engine(self) -> str:
        """Build TRT-LLM engine from checkpoint."""
        logger.info("Building TRT-LLM engine from checkpoint...")
        
        self.engine_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            "trtllm-build",
            "--checkpoint_dir", str(self.checkpoint_dir),
            "--output_dir", str(self.engine_dir),
        ]
        
        # Add build config options
        for key, value in self.build_config.items():
            cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        t0 = time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        build_time = time.perf_counter() - t0
        
        if result.returncode != 0:
            logger.error(f"trtllm-build failed:\n{result.stderr}")
            raise RuntimeError(f"trtllm-build failed with code {result.returncode}")
        
        logger.info(f"Engine built in {build_time:.1f}s at {self.engine_dir}")
        
        return str(self.engine_dir)
    
    def get_transfer_stats(self) -> dict:
        """Get transfer statistics."""
        return self.transfer_stats


def create_trtllm_from_mx(
    model_name: str,
    mx_server: str = "modelexpress-server:8001",
    output_dir: str = "/tmp/mx_trtllm",
    build_config: dict | None = None,
    skip_build: bool = False,
):
    """
    Create TRT-LLM LLM instance using ModelExpress P2P transfer.
    
    This is a convenience function that:
    1. Receives checkpoint weights via RDMA
    2. Builds TRT-LLM engine (unless skip_build=True)
    3. Returns the engine/checkpoint path
    
    Args:
        model_name: Model identifier in ModelExpress
        mx_server: ModelExpress server address
        output_dir: Local directory for checkpoint and engine
        build_config: Optional trtllm-build configuration
        skip_build: If True, only download checkpoint
        
    Returns:
        Path to engine directory (or checkpoint if skip_build)
        
    Example:
        # Receive and build
        engine_dir = create_trtllm_from_mx("llama-70b")
        
        # Use with TRT-LLM
        from tensorrt_llm import LLM
        llm = LLM(model=engine_dir)
    """
    loader = MxTrtllmTargetLoader(
        model_name=model_name,
        mx_server=mx_server,
        output_dir=output_dir,
        build_config=build_config,
    )
    
    return loader.load(skip_build=skip_build)
