# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress P2P Checkpoint Loader for TensorRT-LLM.

Integrates into TRT-LLM's checkpoint loader plugin system to load model weights
via NIXL RDMA from a ModelExpress source, and model config from the MX server's
metadata store — eliminating the need for PVC access on the target.

Usage:
    import modelexpress.trtllm_checkpoint_loader  # Side-effect: registers "mx-p2p"

    from tensorrt_llm import LLM
    llm = LLM(
        model="meta-llama/Llama-3.1-70B-Instruct",  # Only used if MX server has no config
        checkpoint_format="mx-p2p",
        tensor_parallel_size=8,
    )
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger("modelexpress.trtllm_checkpoint_loader")

# ---------------------------------------------------------------------------
# Lazy imports for TRT-LLM types (only needed when loader is actually used)
# ---------------------------------------------------------------------------

def _import_trtllm():
    """Import TRT-LLM types. Raises ImportError if TRT-LLM is not installed.

    NOTE: This function requires GPU access (libcuda.so) because TRT-LLM's
    import chain loads native bindings. Only call from GPU application code,
    not at module import time.
    """
    from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import (
        BaseCheckpointLoader,
    )
    from tensorrt_llm._torch.models.checkpoints.base_config_loader import (
        BaseConfigLoader,
    )
    from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
        BaseWeightLoader,
    )
    # ConsumableWeightsDict was added in TRT-LLM 1.3+; use plain dict for older versions
    try:
        from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
            ConsumableWeightsDict,
        )
    except ImportError:
        ConsumableWeightsDict = None
    from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import (
        BaseWeightMapper,
    )
    from tensorrt_llm._torch.models.checkpoints.hf.config_loader import (
        HfConfigLoader,
    )
    from tensorrt_llm._torch.models.modeling_utils import (
        register_checkpoint_loader,
        register_checkpoint_weight_loader,
        register_config_loader,
    )
    from tensorrt_llm.mapping import Mapping

    return {
        "BaseCheckpointLoader": BaseCheckpointLoader,
        "BaseConfigLoader": BaseConfigLoader,
        "BaseWeightLoader": BaseWeightLoader,
        "BaseWeightMapper": BaseWeightMapper,
        "ConsumableWeightsDict": ConsumableWeightsDict,
        "HfConfigLoader": HfConfigLoader,
        "Mapping": Mapping,
        "register_checkpoint_loader": register_checkpoint_loader,
        "register_checkpoint_weight_loader": register_checkpoint_weight_loader,
        "register_config_loader": register_config_loader,
    }


# ---------------------------------------------------------------------------
# Helper: Parse dtype strings
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.bfloat16": torch.bfloat16,
    "torch.int8": torch.int8,
    "torch.int32": torch.int32,
    "torch.uint8": torch.uint8,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "float8_e4m3fn": torch.float8_e4m3fn,
}

def _parse_dtype(dtype_str: str) -> torch.dtype:
    return _DTYPE_MAP.get(dtype_str, torch.float16)

def _dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()

# TP sharding patterns for weight reconstruction
_COL_PARALLEL_PATTERNS = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
_ROW_PARALLEL_PATTERNS = ["o_proj", "down_proj"]

# MoE expert weight pattern: model.layers.{L}.mlp.experts.{E}.{suffix}.weight
_EXPERT_WEIGHT_PATTERN = re.compile(r"\.experts\.(\d+)\.")
# Weights that are always replicated (not sharded or partitioned)
_REPLICATED_PATTERNS = ["embed_tokens", "layer_norm", "norm", "lm_head", "mlp.gate"]


# ===========================================================================
# MxWeightLoader — loads weights via NIXL RDMA
# ===========================================================================

class MxWeightLoader:
    """
    Loads model weights via NIXL RDMA from a ModelExpress source.

    Instead of reading safetensors from disk, this loader:
    1. Queries the MX server for source metadata (NIXL agent info + tensor descs)
    2. For each TP rank: initializes a NIXL agent, allocates GPU tensors, receives via RDMA
    3. Reconstructs full HF-format weights from TP shards
    4. Returns a ConsumableWeightsDict with HF-format names
    """

    def __init__(self):
        self._source_meta = None

    def load_weights(
        self,
        checkpoint_dir: str,
        mapping: Any = None,
        model: Any = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Load weights via P2P RDMA instead of from disk.

        Args:
            checkpoint_dir: Used to derive model_name for MX server query.
            mapping: TRT-LLM Mapping object (has tp_rank, tp_size, etc.)
            model: TRT-LLM model object (when available, enables GPU-resident weights)

        Returns:
            Dict with HF-format weight names → GPU tensors (when model available)
            or CPU tensors (fallback).
        """
        from .nixl_transfer import NixlTransferManager
        from .client import MxClient
        from .types import TensorDescriptor

        mx_server = os.environ.get("MODEL_EXPRESS_URL", "localhost:8001")
        model_name = os.environ.get(
            "MODEL_NAME", os.path.basename(checkpoint_dir)
        )

        has_model = model is not None
        logger.info(
            "MxWeightLoader: Loading weights for '%s' (model_ref=%s)",
            model_name, has_model,
        )

        client = MxClient(mx_server)

        # 1. Query source metadata (with retry)
        source_meta = self._query_source(client, model_name, timeout=600)

        # 2. Determine which source ranks to receive from
        # In TP mode, each worker receives ONLY from its matching source rank.
        local_device = torch.cuda.current_device()
        num_source_ranks = len(source_meta.workers)

        if num_source_ranks > 1:
            my_rank = local_device
            workers_to_receive = [
                w for w in source_meta.workers if w.worker_rank == my_rank
            ]
            if not workers_to_receive:
                workers_to_receive = [source_meta.workers[0]]
            logger.info(
                "TP mode: device %d receiving from source rank %d only",
                local_device, workers_to_receive[0].worker_rank,
            )
        else:
            workers_to_receive = list(source_meta.workers)
            logger.info("Single rank mode on device %d", local_device)

        # 3. Receive weights via NIXL RDMA
        all_rank_weights: dict[int, dict[str, torch.Tensor]] = {}

        for worker in workers_to_receive:
            rank = worker.worker_rank
            logger.info(
                "Receiving rank %d (%d tensors) → device %d",
                rank, len(worker.tensors), local_device,
            )

            nixl_mgr = NixlTransferManager(
                agent_name=f"trtllm-target-r{rank}-{os.getpid()}",
                device_id=local_device,
            )
            nixl_mgr.initialize()

            try:
                # Allocate GPU tensors on LOCAL device
                weights = self._allocate_tensors(worker.tensors, local_device)
                nixl_mgr.register_tensors(weights)

                source_descs = [
                    TensorDescriptor(
                        name=t.name, addr=t.addr, size=t.size,
                        device_id=t.device_id, dtype=t.dtype,
                    )
                    for t in worker.tensors
                ]

                coalesce = os.environ.get("MX_COALESCE_TRANSFERS", "0") == "1"
                t0 = time.perf_counter()
                bytes_transferred, _, _ = nixl_mgr.receive_from_source(
                    source_metadata=worker.nixl_metadata,
                    source_tensors=source_descs,
                    timeout_seconds=300,
                    coalesce_transfers=coalesce,
                )
                elapsed = time.perf_counter() - t0
                bw = (bytes_transferred * 8) / (elapsed * 1e9) if elapsed > 0 else 0
                logger.info(
                    "Rank %d: %.2f GB in %.2fs (%.1f Gbps)",
                    rank, bytes_transferred / 1e9, elapsed, bw,
                )

                # Copy to CPU before NIXL shutdown (GPU tensors invalidated on shutdown).
                # TODO: With model reference, we could NIXL directly into param buffers
                # to avoid this copy. Requires knowing param layout for fused modules.
                cpu_weights = {k: v.cpu() for k, v in weights.items()}
                all_rank_weights[rank] = cpu_weights

            finally:
                nixl_mgr.shutdown()

        # 4. Reconstruct full HF weights from TP shards (only if multi-rank)
        full_weights = self._reconstruct_full_weights(all_rank_weights)

        total_bytes = sum(t.numel() * t.element_size() for t in full_weights.values())
        device_str = "GPU" if has_model else "CPU"
        logger.info(
            "Loaded %d tensors (%.2f GB) on %s",
            len(full_weights), total_bytes / 1e9, device_str,
        )

        return full_weights

    def cleanup(self):
        pass

    # --- Internal methods ---

    def _query_source(self, client, model_name: str, timeout: int = 600):
        """Query MX server for source metadata with retry."""
        import grpc
        from . import p2p_pb2, p2p_pb2_grpc

        server_addr = os.environ.get("MODEL_EXPRESS_URL", "localhost:8001")
        options = [
            ("grpc.max_receive_message_length", 200 * 1024 * 1024),
        ]
        channel = grpc.insecure_channel(server_addr, options=options)
        stub = p2p_pb2_grpc.P2pServiceStub(channel)

        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = stub.GetMetadata(
                    p2p_pb2.GetMetadataRequest(model_name=model_name)
                )
                if resp.found and len(resp.workers) > 0:
                    logger.info(
                        "Found source: %d workers", len(resp.workers)
                    )
                    # Log a few tensor shapes to verify metadata freshness
                    w0 = resp.workers[0]
                    for t in w0.tensors[:5]:
                        logger.info(
                            "  Metadata tensor: %s size=%d shape=%s",
                            t.name, t.size, list(t.shape),
                        )
                    # Store for config loader to use
                    self._source_meta = resp
                    channel.close()
                    return resp
            except Exception as e:
                logger.warning("Query failed: %s, retrying...", e)

            time.sleep(5)

        channel.close()
        raise TimeoutError(
            f"Source for '{model_name}' not found after {timeout}s"
        )

    def _allocate_tensors(
        self, tensor_protos, device_id: int
    ) -> dict[str, torch.Tensor]:
        """Allocate GPU tensors matching source layout."""
        weights = {}
        logged = 0
        with torch.cuda.device(device_id):
            for t in tensor_protos:
                dtype = _parse_dtype(t.dtype)
                if t.shape and len(t.shape) > 0:
                    shape = tuple(t.shape)
                else:
                    numel = t.size // _dtype_size(dtype)
                    shape = (numel,)
                if logged < 10 and ("k_proj" in t.name or "q_proj" in t.name or "lm_head" in t.name):
                    logger.info("Alloc %s: proto_shape=%s proto_size=%d → shape=%s", t.name, list(t.shape), t.size, shape)
                    logged += 1
                weights[t.name] = torch.empty(
                    shape, dtype=dtype, device=f"cuda:{device_id}"
                )
        return weights

    def _reconstruct_full_weights(
        self, all_rank_weights: dict[int, dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Reconstruct full HF weights from TP-sharded or EP-partitioned transfers.

        For TP-sharded weights: concatenates along the appropriate dimension.
        For MoE expert weights: each rank has different expert indices, so all
        are included without concatenation.
        For replicated weights: takes rank 0's copy.
        """
        if not all_rank_weights:
            return {}

        ranks = sorted(all_rank_weights.keys())
        if len(ranks) == 1:
            return all_rank_weights[ranks[0]]

        full_weights: dict[str, torch.Tensor] = {}

        # Collect all unique tensor names across all ranks
        all_names: set[str] = set()
        for rank_weights in all_rank_weights.values():
            all_names.update(rank_weights.keys())

        for name in sorted(all_names):
            name_lower = name.lower()

            # MoE expert weights: each rank has different experts — include all
            if _EXPERT_WEIGHT_PATTERN.search(name):
                for r in ranks:
                    if name in all_rank_weights[r]:
                        full_weights[name] = all_rank_weights[r][name]
                        break
                continue

            # Collect shards from ranks that have this tensor
            shards = [all_rank_weights[r][name] for r in ranks if name in all_rank_weights[r]]
            if not shards:
                continue

            if len(shards) == 1:
                full_weights[name] = shards[0]
                continue

            # TP-sharded: determine concat dimension
            concat_dim = None
            for pattern in _COL_PARALLEL_PATTERNS:
                if pattern in name_lower:
                    concat_dim = -1
                    break
            if concat_dim is None:
                for pattern in _ROW_PARALLEL_PATTERNS:
                    if pattern in name_lower:
                        concat_dim = 0
                        break

            if concat_dim is not None:
                full_weights[name] = torch.cat(shards, dim=concat_dim)
            else:
                full_weights[name] = shards[0]

        return full_weights


# ===========================================================================
# MxConfigLoader — loads config from MX server metadata
# ===========================================================================

class MxConfigLoader:
    """
    Loads model config from MX server's stored model_files.

    When a source publishes metadata, it includes config.json, tokenizer.json, etc.
    This loader retrieves those files from the MX server and writes them to a temp
    directory, then delegates to HfConfigLoader for actual parsing.

    Falls back to local HfConfigLoader if MX server doesn't have model_files.
    """

    def __init__(self):
        self._temp_dir: Optional[str] = None
        self._weight_loader: Optional[MxWeightLoader] = None

    def set_weight_loader(self, loader: MxWeightLoader):
        """Share reference to weight loader for accessing cached source metadata."""
        self._weight_loader = loader

    def load(self, checkpoint_dir: str, **kwargs):
        """
        Load model config from MX server or local fallback.

        1. Try to get model_files from MX server (via GetMetadata)
        2. Write config files to temp dir
        3. Delegate to HfConfigLoader to parse
        4. Fall back to local checkpoint_dir if no model_files
        """
        trtllm = _import_trtllm()
        HfConfigLoader = trtllm["HfConfigLoader"]

        # Try to get model_files from MX server
        model_files = self._get_model_files_from_mx()

        if model_files and "config.json" in model_files:
            # Write to temp dir and load from there
            self._temp_dir = tempfile.mkdtemp(prefix="mx_config_")
            config_dir = Path(self._temp_dir)

            for fname, content in model_files.items():
                fpath = config_dir / fname
                fpath.write_bytes(content)
                logger.info("Wrote %s (%d bytes) to temp dir", fname, len(content))

            logger.info("Loading config from MX server model_files (%s)", config_dir)
            return HfConfigLoader().load(str(config_dir), **kwargs)

        # Fall back to local path
        logger.info(
            "No model_files from MX server, falling back to local: %s",
            checkpoint_dir,
        )
        return HfConfigLoader().load(checkpoint_dir, **kwargs)

    def cleanup(self):
        """Clean up temp directory."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def _get_model_files_from_mx(self) -> Optional[dict[str, bytes]]:
        """Query MX server for model_files."""
        import grpc
        from . import p2p_pb2, p2p_pb2_grpc

        # If weight loader already queried, use cached response
        if self._weight_loader and self._weight_loader._source_meta:
            resp = self._weight_loader._source_meta
            if resp.model_files:
                logger.info(
                    "Using cached model_files from weight loader (%d files)",
                    len(resp.model_files),
                )
                return dict(resp.model_files)

        # Otherwise query directly
        mx_server = os.environ.get("MODEL_EXPRESS_URL", "localhost:8001")
        model_name = os.environ.get("MODEL_NAME", "")

        if not model_name:
            logger.warning("MODEL_NAME not set, cannot query MX server for config")
            return None

        try:
            options = [
                ("grpc.max_receive_message_length", 200 * 1024 * 1024),
            ]
            channel = grpc.insecure_channel(mx_server, options=options)
            stub = p2p_pb2_grpc.P2pServiceStub(channel)

            resp = stub.GetMetadata(
                p2p_pb2.GetMetadataRequest(model_name=model_name)
            )
            channel.close()

            if resp.found and resp.model_files:
                logger.info(
                    "Got %d model_files from MX server", len(resp.model_files)
                )
                return dict(resp.model_files)

        except Exception as e:
            logger.warning("Failed to get model_files from MX server: %s", e)

        return None


# ===========================================================================
# MxCheckpointLoader — main entry point, registered as "mx-p2p"
#
# Defined at MODULE LEVEL so it can be pickled by Python multiprocessing
# (TRT-LLM spawns worker processes that need to serialize the loader).
# The base class is resolved dynamically at __init_subclass__ time.
# ===========================================================================

class MxCheckpointLoader:
    """
    ModelExpress P2P checkpoint loader for TRT-LLM.

    Loads model config from MX server metadata (no PVC needed) and
    model weights via NIXL RDMA (no disk I/O needed).

    Reuses HfWeightMapper for HF→TRT-LLM name conversion.

    NOTE: This class is defined at module level (not inside a function)
    so that Python's pickle can serialize it for TRT-LLM's multi-process
    executor workers.
    """

    def __init__(
        self,
        *,
        weight_loader=None,
        weight_mapper=None,
        config_loader=None,
    ):
        self._weight_loader = weight_loader or MxWeightLoader()
        self._config_loader = config_loader or MxConfigLoader()
        self._weight_mapper = weight_mapper
        self._checkpoint_format = "mx-p2p"

        # Share weight loader ref with config loader for cached metadata
        if isinstance(self._config_loader, MxConfigLoader):
            self._config_loader.set_weight_loader(self._weight_loader)

    def get_default_weight_loader(self):
        return MxWeightLoader()

    def get_default_config_loader(self):
        return MxConfigLoader()

    def cleanup(self):
        if self._weight_mapper is not None:
            if hasattr(self._weight_mapper, 'cleanup'):
                self._weight_mapper.cleanup()
            self._weight_mapper = None
        if self._weight_loader is not None:
            self._weight_loader.cleanup()
            self._weight_loader = None
        if self._config_loader is not None:
            self._config_loader.cleanup()
            self._config_loader = None

    @property
    def weight_loader(self):
        return self._weight_loader

    @property
    def weight_mapper(self):
        return self._weight_mapper

    @weight_mapper.setter
    def weight_mapper(self, value):
        self._weight_mapper = value

    @property
    def config_loader(self):
        return self._config_loader

    @property
    def checkpoint_format(self):
        return self._checkpoint_format

    def load_config(self, checkpoint_dir: str, **kwargs):
        """Load config from MX server model_files, fallback to local."""
        logger.info("MxCheckpointLoader.load_config(%s)", checkpoint_dir)
        return self._config_loader.load(checkpoint_dir, **kwargs)

    def load_weights(self, checkpoint_dir: str, mapping=None, model=None, **kwargs):
        """Load weights via NIXL RDMA from ModelExpress source."""
        logger.info("MxCheckpointLoader.load_weights(%s, model=%s)", checkpoint_dir, model is not None)
        return self._weight_loader.load_weights(
            checkpoint_dir, mapping=mapping, model=model, **kwargs
        )

    def get_initialized_weight_mapper(self, model, config):
        """
        Return weight mapper. Reuse HfWeightMapper since our weights
        are in HF format.
        """
        if self._weight_mapper is not None:
            self._weight_mapper.init_model_and_config(model, config)
            return self._weight_mapper

        # Auto-resolve to HfWeightMapper via checkpoint format
        from tensorrt_llm._torch.models.checkpoints.auto_mapper import (
            AutoCheckpointMapper,
        )

        if (
            config.pretrained_config
            and config.pretrained_config.architectures
        ):
            model_arch = config.pretrained_config.architectures[0]
        else:
            raise ValueError(
                "Cannot determine model architecture from config"
            )

        # Use "HF" format mapper since our weights are HF-format
        weight_mapper = AutoCheckpointMapper.get("HF", model_arch)
        weight_mapper.init_model_and_config(model, config)

        # For presharded weights: override _tp_size on the mapper to 1.
        # This prevents _duplicate_kv_weights from duplicating KV heads.
        # With pre-sharded weights, each rank's k_proj already has num_kv_heads/tp
        # heads. The mapper sees shape[0]/head_dim = 1 KV head and would
        # duplicate it tp_size times if _tp_size > 1.
        if any(getattr(m, '_weights_presharded', False)
               for m in model.modules()):
            logger.info("Presharded mode: setting mapper _tp_size=1 to skip KV duplication")
            weight_mapper._tp_size = 1

        self._weight_mapper = weight_mapper
        return weight_mapper


def _register_loader():
    """Register the mx-p2p checkpoint loader with TRT-LLM.

    TRT-LLM's _construct_checkpoint_loader() does three separate registry lookups
    for a given checkpoint_format:
      1. get_checkpoint_weight_loader(format) → MODEL_CLASS_CHECKPOINT_WEIGHT_LOADER_DEFAULT_MAPPING
      2. get_config_loader(format) → MODEL_CLASS_CONFIG_LOADER_DEFAULT_MAPPING
      3. BaseCheckpointLoader.get(format) → CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING
    All three must be registered for checkpoint_format="mx-p2p" to work.

    NOTE: Must be called from a GPU environment (requires libcuda.so).
    """
    try:
        trtllm = _import_trtllm()
    except Exception as e:
        logger.error(
            "Failed to import TRT-LLM types for mx-p2p loader: %s", e,
            exc_info=True,
        )
        return None

    register_checkpoint_loader = trtllm["register_checkpoint_loader"]
    register_checkpoint_weight_loader = trtllm["register_checkpoint_weight_loader"]
    register_config_loader = trtllm["register_config_loader"]

    # Register all three registries
    register_checkpoint_weight_loader("mx-p2p")(MxWeightLoader)
    register_config_loader("mx-p2p")(MxConfigLoader)
    register_checkpoint_loader("mx-p2p")(MxCheckpointLoader)

    logger.info("Registered 'mx-p2p' checkpoint loader with TRT-LLM")
    return MxCheckpointLoader
