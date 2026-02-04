# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress TensorRT-LLM Integration.

Enables P2P weight transfer for TRT-LLM via NIXL RDMA.

Supports TWO modes:

1. **HuggingFace Mode** (recommended for DeepSeek-V3/R1, Llama, Qwen):
   - Source loads HuggingFace model directly to GPU
   - Transfers raw weights via RDMA (similar to vLLM P2P)
   - Target uses TRT-LLM's PyTorch backend for direct inference
   - No checkpoint conversion or engine build required!

2. **Checkpoint Mode** (traditional TRT-LLM workflow):
   - Requires pre-converted TRT-LLM checkpoint (config.json + rank*.safetensors)
   - Transfers checkpoint weights via RDMA
   - Target optionally builds TensorRT engine

This module provides:
    - MxTrtllmSourcePublisher: Loads HuggingFace model or checkpoint, registers with NIXL
    - MxTrtllmTargetLoader: Receives weights via RDMA, uses with TRT-LLM

Usage (HuggingFace mode - recommended):
    # Source side - load HuggingFace model and publish for P2P
    publisher = MxTrtllmSourcePublisher(
        hf_model_path="/path/to/deepseek-v3",  # HuggingFace format
        model_name="deepseek-v3",
        mx_server="modelexpress-server:8001",
        tp_size=8,
    )
    publisher.initialize()
    
    # Target side - receive weights and use directly with TRT-LLM
    loader = MxTrtllmTargetLoader(
        model_name="deepseek-v3",
        mx_server="modelexpress-server:8001",
        output_dir="/tmp/mx_trtllm",
    )
    model_path = loader.load(use_pytorch_backend=True)
    
    # Use with TRT-LLM PyTorch backend
    from tensorrt_llm import LLM
    llm = LLM(model=model_path, backend="pytorch")

Usage (Checkpoint mode - traditional):
    # Source with pre-converted checkpoint
    publisher = MxTrtllmSourcePublisher(
        checkpoint_dir="/path/to/trtllm-checkpoint",
        model_name="llama-70b",
        mx_server="modelexpress-server:8001"
    )
    
    # Target receives and builds engine
    loader = MxTrtllmTargetLoader(...)
    engine_dir = loader.load(skip_build=False)
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
    Publishes TRT-LLM weights for P2P transfer.
    
    This class supports three modes:
    
    1. **HuggingFace mode** (recommended for DeepSeek, Llama, Qwen):
       Loads HuggingFace model directly to GPU using TRT-LLM's PyTorch backend.
       Transfers raw weights via RDMA - fastest path, no conversion needed.
       
    2. **Checkpoint mode**: Loads TRT-LLM checkpoint files (safetensors) to GPU.
       Target receives checkpoint weights.
       
    3. **Engine mode**: Loads pre-built TensorRT engine's weights to GPU.
       Requires source and target to have same GPU architecture.
    
    The weights remain in GPU memory for the lifetime of this object,
    allowing multiple targets to receive them.
    
    Args:
        checkpoint_dir: Path to TRT-LLM checkpoint directory (for checkpoint mode)
        model_name: Model identifier for coordination
        mx_server: ModelExpress server address (host:port)
        engine_dir: Optional path to pre-built engine directory
        hf_model_path: Path to HuggingFace model (for HuggingFace mode)
        tp_size: Tensor parallelism size (for HuggingFace mode)
        dtype: Model dtype (default: "bfloat16")
    """
    
    def __init__(
        self,
        model_name: str,
        mx_server: str = "modelexpress-server:8001",
        checkpoint_dir: str | None = None,
        engine_dir: str | None = None,
        hf_model_path: str | None = None,
        tp_size: int = 1,
        dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.mx_server = _parse_server_address(mx_server)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.engine_dir = Path(engine_dir) if engine_dir else None
        self.hf_model_path = Path(hf_model_path) if hf_model_path else None
        self.tp_size = tp_size
        self.dtype = dtype
        self.config: dict = {}
        self.nixl_managers: dict[int, NixlTransferManager] = {}
        self.weights: dict[int, dict[str, torch.Tensor]] = {}
        self._initialized = False
        self._engine_bytes: bytes | None = None  # Serialized engine for transfer
        self._mode: str = "unknown"  # Will be set during initialize()
        
    def initialize(self, build_if_needed: bool = True) -> None:
        """
        Load model/checkpoint/engine and register with NIXL.
        
        This method auto-detects the mode based on provided paths:
        1. If hf_model_path provided: HuggingFace mode (recommended)
        2. If engine_dir provided: Engine mode
        3. If checkpoint_dir provided: Checkpoint mode
        
        Args:
            build_if_needed: If True and checkpoint mode with no engine, build engine
        """
        from .nixl_transfer import NixlTransferManager, is_nixl_available
        
        if not is_nixl_available():
            raise RuntimeError("NIXL is not available. Install with: pip install nixl[cu12]")
        
        if self._initialized:
            logger.warning("Already initialized, skipping")
            return
        
        logger.info(f"Initializing TRT-LLM source publisher for {self.model_name}")
        
        # Auto-detect mode
        if self.hf_model_path and self.hf_model_path.exists():
            self._mode = "huggingface"
            self._initialize_huggingface_mode()
        elif self.checkpoint_dir and self.checkpoint_dir.exists():
            self._mode = "checkpoint"
            self._initialize_checkpoint_mode(build_if_needed)
        else:
            raise ValueError(
                "Must provide either hf_model_path (HuggingFace model) or "
                "checkpoint_dir (TRT-LLM checkpoint)"
            )
        
        # Publish to MX server
        self._publish_to_mx_server()
        
        self._initialized = True
        logger.info(f"TRT-LLM source publisher initialized ({self._mode} mode)")
    
    def _initialize_huggingface_mode(self) -> None:
        """Load HuggingFace model directly to GPU (like vLLM P2P)."""
        from .nixl_transfer import NixlTransferManager
        
        logger.info(f"HuggingFace mode: Loading {self.hf_model_path} with TP={self.tp_size}")
        
        # Store HF config for metadata
        hf_config_path = self.hf_model_path / "config.json"
        if hf_config_path.exists():
            with open(hf_config_path) as f:
                hf_config = json.load(f)
            self.config = {
                "architecture": hf_config.get("model_type", "unknown"),
                "hidden_size": hf_config.get("hidden_size"),
                "num_layers": hf_config.get("num_hidden_layers"),
                "dtype": self.dtype,
                "mapping": {
                    "world_size": self.tp_size,
                    "tp_size": self.tp_size,
                    "pp_size": 1,
                },
                "_hf_model_path": str(self.hf_model_path),
                "_mode": "huggingface",
            }
            logger.info(f"Model: {self.config['architecture']}, hidden_size={self.config['hidden_size']}")
        
        # Load model weights for each rank
        total_bytes = 0
        
        for rank in range(self.tp_size):
            logger.info(f"Loading rank {rank}/{self.tp_size} weights to GPU...")
            
            # Load HuggingFace weights and shard for this rank
            weights = self._load_hf_weights_for_rank(rank)
            self.weights[rank] = weights
            
            rank_bytes = sum(t.numel() * t.element_size() for t in weights.values())
            total_bytes += rank_bytes
            logger.info(f"Rank {rank}: {len(weights)} tensors, {rank_bytes / 1e9:.2f} GB")
            
            # Initialize NIXL manager
            import uuid
            agent_name = f"trtllm-hf-source-rank{rank}-{uuid.uuid4().hex[:8]}"
            nixl_manager = NixlTransferManager(
                agent_name=agent_name,
                device_id=rank
            )
            nixl_manager.initialize()
            nixl_manager.register_tensors(weights)
            self.nixl_managers[rank] = nixl_manager
            logger.info(f"Rank {rank}: Registered with NIXL")
        
        logger.info(f"HuggingFace mode: {total_bytes / 1e9:.2f} GB across {self.tp_size} ranks")
    
    def _load_hf_weights_for_rank(self, rank: int) -> dict[str, torch.Tensor]:
        """
        Load HuggingFace model weights for a specific TP rank.
        
        For TP > 1, this shards the weights appropriately.
        For TP = 1, loads all weights to the single GPU.
        """
        try:
            import safetensors.torch
        except ImportError:
            raise ImportError("safetensors required. Install with: pip install safetensors")
        
        weights: dict[str, torch.Tensor] = {}
        
        # Find all safetensors files in the model directory
        safetensor_files = list(self.hf_model_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {self.hf_model_path}")
        
        logger.info(f"Found {len(safetensor_files)} safetensors files")
        
        # For TP=1, load all weights to GPU
        # For TP>1, we need to shard (this is model-specific)
        device = f"cuda:{rank}"
        
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        target_dtype = dtype_map.get(self.dtype, torch.bfloat16)
        
        for sf_path in safetensor_files:
            file_weights = safetensors.torch.load_file(str(sf_path), device=device)
            
            for name, tensor in file_weights.items():
                # Optionally convert dtype
                if tensor.dtype != target_dtype and tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    tensor = tensor.to(target_dtype)
                
                # For TP > 1, shard appropriate dimensions
                if self.tp_size > 1:
                    tensor = self._shard_tensor_for_rank(name, tensor, rank)
                
                if tensor is not None:
                    weights[name] = tensor
        
        return weights
    
    def _shard_tensor_for_rank(
        self, name: str, tensor: torch.Tensor, rank: int
    ) -> torch.Tensor | None:
        """
        Shard a tensor for tensor parallelism.
        
        This implements basic TP sharding based on tensor name patterns.
        For more complex models, override this method.
        """
        if self.tp_size == 1:
            return tensor
        
        # Common patterns for column-parallel (shard dim=-1) and row-parallel (shard dim=0)
        col_parallel_patterns = [
            "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj",
            "qkv_proj", "gate_up_proj", "query_key_value",
        ]
        row_parallel_patterns = [
            "o_proj", "down_proj", "out_proj", "dense",
        ]
        
        # Check if tensor should be sharded
        shard_dim = None
        for pattern in col_parallel_patterns:
            if pattern in name.lower():
                shard_dim = -1  # Shard last dimension
                break
        
        if shard_dim is None:
            for pattern in row_parallel_patterns:
                if pattern in name.lower():
                    shard_dim = 0  # Shard first dimension
                    break
        
        if shard_dim is None:
            # Not a tensor that needs sharding, return as-is
            return tensor
        
        # Perform sharding
        dim_size = tensor.shape[shard_dim]
        if dim_size % self.tp_size != 0:
            # Can't evenly shard, return full tensor
            return tensor
        
        shard_size = dim_size // self.tp_size
        start_idx = rank * shard_size
        end_idx = start_idx + shard_size
        
        if shard_dim == -1:
            return tensor[..., start_idx:end_idx].contiguous()
        else:
            return tensor[start_idx:end_idx, ...].contiguous()
    
    def _initialize_checkpoint_mode(self, build_if_needed: bool) -> None:
        """Load TRT-LLM checkpoint (traditional mode)."""
        logger.info(f"Checkpoint mode: Loading from {self.checkpoint_dir}")
        
        # Parse config.json
        config_path = self.checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.config["_mode"] = "checkpoint"
        
        world_size = self.config.get("mapping", {}).get("world_size", 1)
        tp_size = self.config.get("mapping", {}).get("tp_size", 1)
        pp_size = self.config.get("mapping", {}).get("pp_size", 1)
        
        logger.info(f"Config: world_size={world_size}, tp_size={tp_size}, pp_size={pp_size}")
        
        # Check for engine or build if needed
        if self.engine_dir and self.engine_dir.exists():
            logger.info(f"Using pre-built engine from {self.engine_dir}")
            self._load_engine_for_transfer()
        elif build_if_needed:
            logger.info("No engine found, building from checkpoint...")
            self._build_engine()
            self._load_engine_for_transfer()
        else:
            logger.info("Checkpoint-only mode (no engine transfer)")
        
        # Load and register each rank's weights
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
        
        logger.info(f"Checkpoint mode: {total_bytes / 1e9:.2f} GB across {world_size} ranks")
    
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
                    shape=list(tensor.shape),  # Preserve tensor shape for reconstruction
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
    Receives TRT-LLM weights via P2P and prepares for inference.
    
    Supports two modes based on source:
    
    1. **HuggingFace mode** (when source used hf_model_path):
       - Receives raw HuggingFace weights via RDMA
       - Saves in HuggingFace format for TRT-LLM PyTorch backend
       - No engine build required - use directly with LLM(backend="pytorch")
       
    2. **Checkpoint mode** (when source used checkpoint_dir):
       - Receives TRT-LLM checkpoint weights via RDMA
       - Optionally builds TensorRT engine
    
    Args:
        model_name: Model identifier in ModelExpress
        mx_server: ModelExpress server address (host:port)
        output_dir: Local directory for received weights
        build_config: Optional trtllm-build configuration (checkpoint mode only)
        tp_size: Expected tensor parallelism size
    """
    
    def __init__(
        self,
        model_name: str,
        mx_server: str = "modelexpress-server:8001",
        output_dir: str = "/tmp/mx_trtllm",
        build_config: dict | None = None,
        tp_size: int = 1,
    ):
        self.model_name = model_name
        self.mx_server = _parse_server_address(mx_server)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoint"
        self.engine_dir = self.output_dir / "engine"
        self.hf_output_dir = self.output_dir / "hf_model"
        self.tp_size = tp_size
        
        # Default build configuration
        self.build_config = build_config or {
            "gemm_plugin": "auto",
            "max_batch_size": "8",
            "max_input_len": "2048",
            "max_seq_len": "4096",
        }
        
        self.config: dict = {}
        self.transfer_stats: dict = {}
        self._source_mode: str = "unknown"  # Will be detected from source metadata
    
    def load(
        self,
        skip_build: bool = False,
        use_pytorch_backend: bool = False,
    ) -> str:
        """
        Receive weights via P2P and prepare for inference.
        
        Behavior depends on source mode:
        - HuggingFace mode: Returns path to HF model for PyTorch backend
        - Checkpoint mode: Returns path to checkpoint or engine
        
        Args:
            skip_build: If True, skip engine build (checkpoint mode only)
            use_pytorch_backend: If True, prepare for TRT-LLM PyTorch backend
            
        Returns:
            Path to model directory suitable for TRT-LLM LLM()
            
        Example:
            # For HuggingFace/PyTorch backend mode
            model_path = loader.load(use_pytorch_backend=True)
            from tensorrt_llm import LLM
            llm = LLM(model=model_path, backend="pytorch")
            
            # For traditional engine mode
            engine_path = loader.load(skip_build=False)
            llm = LLM(model=engine_path)
        """
        logger.info(f"Loading {self.model_name} via ModelExpress P2P transfer")
        
        # 1. Query MX server for source
        source_meta = self._query_source()
        
        # 2. Detect source mode from config
        self._source_mode = self.config.get("_mode", "checkpoint")
        logger.info(f"Source mode: {self._source_mode}")
        
        # 3. Receive weights via NIXL
        if self._source_mode == "huggingface":
            self.hf_output_dir.mkdir(parents=True, exist_ok=True)
            self._receive_hf_weights(source_meta)
            
            # Save config for TRT-LLM
            self._save_hf_config()
            
            logger.info(f"HuggingFace model saved to {self.hf_output_dir}")
            return str(self.hf_output_dir)
        else:
            # Checkpoint mode
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._receive_checkpoint(source_meta)
            
            # Save config from source metadata for TRT-LLM
            config_path = self.checkpoint_dir / "config.json"
            if self.config and not config_path.exists():
                # Create config.json from source metadata
                self._save_checkpoint_config()
            elif config_path.exists():
                with open(config_path) as f:
                    self.config = json.load(f)
                logger.info(f"Loaded config from checkpoint")
            
            if skip_build or use_pytorch_backend:
                logger.info(f"Checkpoint saved to {self.checkpoint_dir}")
                return str(self.checkpoint_dir)
            
            # Build engine locally
            engine_path = self._build_engine()
            logger.info(f"Engine built at {engine_path}")
            return engine_path
    
    def _receive_hf_weights(self, source_meta) -> None:
        """Receive HuggingFace format weights via NIXL."""
        from .nixl_transfer import NixlTransferManager
        from .types import TensorDescriptor
        
        total_bytes = 0
        total_time = 0
        all_weights: dict[str, torch.Tensor] = {}
        
        for worker in source_meta.workers:
            rank = worker.worker_rank
            tensor_count = len(worker.tensors)
            
            logger.info(f"Receiving rank {rank}: {tensor_count} tensors")
            
            # Initialize NIXL manager
            import uuid
            agent_name = f"trtllm-hf-target-rank{rank}-{uuid.uuid4().hex[:8]}"
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
                timeout_seconds=600,
                coalesce_transfers=False,
            )
            transfer_time = time.perf_counter() - t0
            
            total_bytes += bytes_received
            total_time += transfer_time
            
            bandwidth = (bytes_received * 8) / (transfer_time * 1e9) if transfer_time > 0 else 0
            logger.info(
                f"Rank {rank}: Received {bytes_received / 1e9:.2f} GB "
                f"in {transfer_time:.2f}s ({bandwidth:.1f} Gbps)"
            )
            
            # Collect all weights
            all_weights.update(weights)
            
            # Cleanup NIXL
            nixl_manager.shutdown()
        
        # Save weights in HuggingFace format (single safetensors file)
        self._save_hf_weights(all_weights)
        
        # Store transfer stats
        self.transfer_stats = {
            "total_bytes": total_bytes,
            "total_time": total_time,
            "bandwidth_gbps": (total_bytes * 8) / (total_time * 1e9) if total_time > 0 else 0,
        }
        
        logger.info(
            f"HF Transfer complete: {total_bytes / 1e9:.2f} GB total "
            f"in {total_time:.2f}s ({self.transfer_stats['bandwidth_gbps']:.1f} Gbps)"
        )
    
    def _save_hf_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Save weights in HuggingFace safetensors format."""
        try:
            import safetensors.torch
        except ImportError:
            raise ImportError("safetensors required. Install with: pip install safetensors")
        
        # Move to CPU for saving
        cpu_weights = {k: v.cpu() for k, v in weights.items()}
        
        weights_path = self.hf_output_dir / "model.safetensors"
        safetensors.torch.save_file(cpu_weights, str(weights_path))
        
        total_size = sum(t.numel() * t.element_size() for t in cpu_weights.values())
        logger.info(f"Saved {len(cpu_weights)} tensors ({total_size / 1e9:.2f} GB) to {weights_path}")
        
        # Free GPU memory
        del weights
        torch.cuda.empty_cache()
    
    def _save_hf_config(self) -> None:
        """Save HuggingFace config for TRT-LLM."""
        # Extract HF config from source metadata
        hf_config = {
            "architectures": [self.config.get("architecture", "AutoModelForCausalLM")],
            "model_type": self.config.get("architecture", "llama"),
            "hidden_size": self.config.get("hidden_size", 4096),
            "num_hidden_layers": self.config.get("num_layers", 32),
            "torch_dtype": self.config.get("dtype", "bfloat16"),
        }
        
        config_path = self.hf_output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(hf_config, f, indent=2)
        
        logger.info(f"Saved HF config to {config_path}")
    
    def _save_checkpoint_config(self) -> None:
        """Save config.json for TRT-LLM checkpoint format."""
        # For HuggingFace source mode, create a HuggingFace-compatible config
        # that TRT-LLM's PyTorch backend can understand
        config = {
            "architectures": [self.config.get("architecture", "LlamaForCausalLM")],
            "model_type": self.config.get("architecture", "llama"),
            "hidden_size": self.config.get("hidden_size", 4096),
            "num_hidden_layers": self.config.get("num_layers", 32),
            "torch_dtype": self.config.get("dtype", "bfloat16"),
            # Include mapping info for TRT-LLM
            "tensor_parallel_size": self.config.get("mapping", {}).get("tp_size", self.tp_size),
        }
        
        # If source was HuggingFace, get the original path for reference
        if self.config.get("_hf_model_path"):
            config["_source_model"] = self.config.get("_hf_model_path")
        
        config_path = self.checkpoint_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved checkpoint config to {config_path}")
    
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
                coalesce_transfers=False,  # Disabled for cross-node RDMA compatibility
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
            for i, t in enumerate(tensor_protos):
                dtype = self._parse_dtype(t.dtype)
                
                # Debug: Log first few tensors to verify shape is being received
                if i < 3:
                    logger.info(f"  Tensor {t.name}: shape={list(t.shape) if t.shape else 'NONE'}, size={t.size}")
                
                # Use shape if available, otherwise fall back to flat tensor
                if t.shape and len(t.shape) > 0:
                    shape = tuple(t.shape)
                else:
                    # Legacy fallback for protos without shape
                    elem_size = self._dtype_size(dtype)
                    numel = t.size // elem_size
                    shape = (numel,)
                    logger.warning(f"  Tensor {t.name} has no shape, using flat: {shape}")
                
                weights[t.name] = torch.empty(
                    shape, dtype=dtype, device=f"cuda:{device_id}"
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
    use_pytorch_backend: bool = False,
):
    """
    Create TRT-LLM model using ModelExpress P2P transfer.
    
    This is a convenience function that:
    1. Receives weights via RDMA from a running source
    2. For HuggingFace sources: Returns path for PyTorch backend
    3. For checkpoint sources: Optionally builds TensorRT engine
    
    Args:
        model_name: Model identifier in ModelExpress
        mx_server: ModelExpress server address
        output_dir: Local directory for received weights
        build_config: Optional trtllm-build configuration
        skip_build: If True, skip engine build (checkpoint mode)
        use_pytorch_backend: If True, prepare for TRT-LLM PyTorch backend
        
    Returns:
        Path to model directory for TRT-LLM LLM()
        
    Example (HuggingFace mode - DeepSeek-V3):
        # Receive HuggingFace weights
        model_path = create_trtllm_from_mx(
            "deepseek-v3",
            use_pytorch_backend=True
        )
        
        # Use with TRT-LLM PyTorch backend
        from tensorrt_llm import LLM
        llm = LLM(model=model_path, backend="pytorch")
        
    Example (Checkpoint mode - traditional):
        # Receive and build engine
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
    
    return loader.load(skip_build=skip_build, use_pytorch_backend=use_pytorch_backend)


# Example usage for DeepSeek-V3
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TRT-LLM P2P Transfer Example")
    parser.add_argument("--mode", choices=["source", "target"], required=True)
    parser.add_argument("--model-path", required=True, help="HuggingFace model or checkpoint path")
    parser.add_argument("--model-name", default="deepseek-v3")
    parser.add_argument("--mx-server", default="modelexpress-server:8001")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--output-dir", default="/tmp/mx_trtllm")
    
    args = parser.parse_args()
    
    if args.mode == "source":
        # Source: Load HuggingFace model and publish for P2P
        publisher = MxTrtllmSourcePublisher(
            model_name=args.model_name,
            mx_server=args.mx_server,
            hf_model_path=args.model_path,
            tp_size=args.tp_size,
        )
        publisher.initialize()
        
        print(f"Source ready! Model {args.model_name} published for P2P transfer")
        print("Press Ctrl+C to shutdown...")
        
        try:
            import signal
            signal.pause()
        except KeyboardInterrupt:
            publisher.shutdown()
    
    else:
        # Target: Receive weights and prepare for inference
        model_path = create_trtllm_from_mx(
            model_name=args.model_name,
            mx_server=args.mx_server,
            output_dir=args.output_dir,
            use_pytorch_backend=True,
        )
        
        print(f"Model received at: {model_path}")
        print("Use with TRT-LLM:")
        print(f"  from tensorrt_llm import LLM")
        print(f"  llm = LLM(model='{model_path}', backend='pytorch')")
