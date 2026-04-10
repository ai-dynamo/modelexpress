# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress Live Model P2P Transfer for TensorRT-LLM.

Transfers model weights directly between running TRT-LLM instances via NIXL RDMA.
Source registers its model parameter GPU buffers; target receives into its own
model parameter buffers. No format conversion, no disk I/O, no CPU round-trip.

Target usage (via checkpoint_loader):
    from modelexpress.trtllm_live_transfer import MxLiveCheckpointLoader
    loader = MxLiveCheckpointLoader()
    llm = LLM(model="Llama-70B", checkpoint_loader=loader,
              load_format=LoadFormat.PRESHARDED, tp=8)
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Optional

import torch

from .client import MxClient
from . import p2p_pb2

logger = logging.getLogger("modelexpress.trtllm_live_transfer")


def _build_trtllm_identity(
    model_name: str,
    tp_size: int = 1,
    ep_size: int = 1,
    dtype: str = "bfloat16",
) -> p2p_pb2.SourceIdentity:
    from importlib.metadata import version as pkg_version

    try:
        mx_version = pkg_version("modelexpress")
    except Exception:
        mx_version = "0.0.0"

    return p2p_pb2.SourceIdentity(
        mx_version=mx_version,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_TRT_LLM,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=1,
        expert_parallel_size=ep_size,
        dtype=dtype,
    )



def publish_model_params(torch_model: Any) -> None:
    """Publish this rank's model params to ModelExpress directly from a torch model.

    Called from ModelLoader.load() BEFORE post_load_weights() so that targets
    receive pre-processed weights and can run their own post_load_weights().

    Each rank publishes independently via MxClient (per-worker API).
    """
    from .nixl_transfer import NixlTransferManager

    if not hasattr(torch_model, "named_parameters"):
        logger.warning("publish_model_params: model has no named_parameters")
        return

    device_id = torch.cuda.current_device()
    try:
        from mpi4py import MPI
        mpi_rank = MPI.COMM_WORLD.Get_rank()
    except Exception:
        mpi_rank = device_id

    model_name = os.environ.get("MODEL_NAME", "unknown")
    mx_server = os.environ.get("MODEL_EXPRESS_URL", "modelexpress-server:8001")

    param_tensors = {}
    seen_data_ptrs = set()
    total_bytes = 0
    for name, param in torch_model.named_parameters():
        if param.device.type == "cuda" and param.device.index == device_id:
            ptr = param.data.data_ptr()
            if ptr in seen_data_ptrs:
                logger.debug("Skipping aliased param: %s (ptr=%x)", name, ptr)
                continue
            seen_data_ptrs.add(ptr)
            param_tensors[name] = param.data
            total_bytes += param.numel() * param.element_size()

    if not param_tensors:
        logger.warning("publish_model_params: no params on device %d (rank %d)", device_id, mpi_rank)
        return

    logger.info(
        "ModelExpress publish_model_params: '%s' rank %d (GPU %d), %d params, %.2f GB (PRE post_load_weights)",
        model_name, mpi_rank, device_id, len(param_tensors), total_bytes / 1e9,
    )

    nixl_mgr = NixlTransferManager(
        agent_name=f"trtllm-live-source-rank{mpi_rank}-{os.getpid()}",
        device_id=device_id,
    )
    nixl_mgr.initialize()
    nixl_mgr.register_tensors(param_tensors)

    if not hasattr(torch_model, '_mx_nixl_managers'):
        torch_model._mx_nixl_managers = []
    torch_model._mx_nixl_managers.append(nixl_mgr)

    tensor_protos = [
        p2p_pb2.TensorDescriptor(
            name=name,
            addr=tensor.data_ptr(),
            size=tensor.numel() * tensor.element_size(),
            device_id=device_id,
            dtype=str(tensor.dtype),
        )
        for name, tensor in param_tensors.items()
    ]

    worker = p2p_pb2.WorkerMetadata(
        worker_rank=mpi_rank,
        nixl_metadata=nixl_mgr.nixl_metadata,
        tensors=tensor_protos,
    )

    identity = _build_trtllm_identity(model_name=model_name)
    worker_id = uuid.uuid4().hex[:8]
    mx_client = MxClient(server_url=mx_server)
    try:
        mx_source_id = mx_client.publish_metadata(
            identity=identity, worker=worker, worker_id=worker_id,
        )

        logger.info(
            "ModelExpress worker rank %d (GPU %d) published %.2f GB (mx_source_id=%s)",
            mpi_rank, device_id, total_bytes / 1e9, mx_source_id,
        )
    finally:
        mx_client.close()


def publish_from_worker(worker: Any) -> None:
    """Publish this rank's model params to ModelExpress from inside a TRT-LLM executor worker.

    Call this from TensorRT-LLM's BaseWorker.setup_engine() after the engine is created,
    when MODEL_EXPRESS_SOURCE=1. The worker process has the real model (worker.engine.model_engine.model).
    Each rank publishes its own NIXL metadata and tensor descriptors to the MX server.

    Requires patching TRT-LLM's base_worker.setup_engine to call this at the end, e.g.:

        if os.environ.get("MODEL_EXPRESS_SOURCE"):
            try:
                from modelexpress.trtllm_live_transfer import publish_from_worker
                publish_from_worker(self)
            except Exception as e:
                logger.warning("ModelExpress publish_from_worker failed: %s", e)
    """
    from .nixl_transfer import NixlTransferManager

    engine = getattr(worker, "engine", None)
    if engine is None:
        logger.warning("publish_from_worker: worker has no engine")
        return
    model_engine = getattr(engine, "model_engine", None)
    if model_engine is None:
        logger.warning("publish_from_worker: engine has no model_engine (not PyExecutor?)")
        return
    torch_model = getattr(model_engine, "model", None)
    if torch_model is None or not hasattr(torch_model, "named_parameters"):
        logger.warning("publish_from_worker: model_engine has no torch model")
        return

    device_id = torch.cuda.current_device()
    try:
        from mpi4py import MPI
        mpi_rank = MPI.COMM_WORLD.Get_rank()
    except Exception:
        mpi_rank = getattr(worker, "rank", device_id)

    model_name = os.environ.get("MODEL_NAME", "unknown")
    mx_server = os.environ.get("MODEL_EXPRESS_URL", "modelexpress-server:8001")

    param_tensors = {}
    seen_data_ptrs = set()
    total_bytes = 0
    for name, param in torch_model.named_parameters():
        if param.device.type == "cuda" and param.device.index == device_id:
            ptr = param.data.data_ptr()
            if ptr in seen_data_ptrs:
                logger.debug("Skipping aliased param: %s (ptr=%x)", name, ptr)
                continue
            seen_data_ptrs.add(ptr)
            param_tensors[name] = param.data
            total_bytes += param.numel() * param.element_size()

    if not param_tensors:
        logger.warning("publish_from_worker: no params on device %d (rank %d)", device_id, mpi_rank)
        return

    logger.info(
        "ModelExpress worker publish: '%s' rank %d (GPU %d), %d params, %.2f GB",
        model_name, mpi_rank, device_id, len(param_tensors), total_bytes / 1e9,
    )

    if logger.isEnabledFor(logging.DEBUG):
        for name, tensor in list(param_tensors.items())[:5]:
            val = tensor.to(torch.float32)
            cksum = val.sum().item()
            nonzero = (tensor != 0).sum().item()
            logger.debug(
                "SOURCE CHECKSUM rank %d: %s shape=%s dtype=%s sum=%.4f nonzero=%d/%d",
                mpi_rank, name, list(tensor.shape), tensor.dtype,
                cksum, nonzero, tensor.numel(),
            )

    nixl_mgr = NixlTransferManager(
        agent_name=f"trtllm-live-source-rank{mpi_rank}-{os.getpid()}",
        device_id=device_id,
    )
    nixl_mgr.initialize()
    nixl_mgr.register_tensors(param_tensors)

    worker._mx_nixl_manager = nixl_mgr

    tensor_protos = [
        p2p_pb2.TensorDescriptor(
            name=name,
            addr=tensor.data_ptr(),
            size=tensor.numel() * tensor.element_size(),
            device_id=device_id,
            dtype=str(tensor.dtype),
        )
        for name, tensor in param_tensors.items()
    ]

    my_worker = p2p_pb2.WorkerMetadata(
        worker_rank=mpi_rank,
        nixl_metadata=nixl_mgr.nixl_metadata,
        tensors=tensor_protos,
    )

    identity = _build_trtllm_identity(model_name=model_name)
    worker_id = uuid.uuid4().hex[:8]
    mx_client = MxClient(server_url=mx_server)
    mx_source_id = mx_client.publish_metadata(
        identity=identity, worker=my_worker, worker_id=worker_id,
    )
    mx_client.close()

    logger.info(
        "ModelExpress worker rank %d (GPU %d) published %.2f GB (mx_source_id=%s)",
        mpi_rank, device_id, total_bytes / 1e9, mx_source_id,
    )


class MxLiveWeightLoader:
    """
    Loads weights via NIXL RDMA directly into model parameter buffers.

    When source publishes TRT-LLM-format param names (from a live model),
    this loader matches target params by name and does direct GPU→GPU RDMA.
    No format conversion, no fusing, no CPU round-trip.
    """

    def __init__(self, mx_server: Optional[str] = None):
        self._source_meta = None
        self._mx_server = mx_server

    def load_weights(
        self,
        checkpoint_dir: str,
        mapping: Any = None,
        model: Any = None,
        **kwargs,
    ) -> dict[str, Any]:
        from .nixl_transfer import NixlTransferManager
        from .types import TensorDescriptor

        # Use provided URL, then env var, then default
        mx_server = self._mx_server or os.environ.get("MODEL_EXPRESS_URL") or os.environ.get("MX_SERVER_ADDRESS", "localhost:8001")
        model_name = os.environ.get("MODEL_NAME", os.path.basename(checkpoint_dir))

        if model is None:
            raise RuntimeError(
                "MxLiveWeightLoader requires model reference. "
                "Use load_format=LoadFormat.PRESHARDED to pass model."
            )

        device_id = torch.cuda.current_device()

        # MPI rank may differ from local GPU index in multinode (e.g., rank 4
        # on node B sees local GPU 0). Use MPI rank for source worker matching,
        # local GPU index for NIXL agent and tensor registration.
        try:
            from mpi4py import MPI
            mpi_rank = MPI.COMM_WORLD.Get_rank()
        except Exception:
            mpi_rank = device_id

        # MPI workers' stdout is swallowed by TRT-LLM — write to per-rank file
        log_dir = os.environ.get("MX_TRANSFER_LOG_DIR", "/tmp/mx_logs")
        os.makedirs(log_dir, exist_ok=True)
        rank_log = os.path.join(log_dir, f"rank{mpi_rank}.log")
        fh = logging.FileHandler(rank_log, mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logging.getLogger("modelexpress").addHandler(fh)

        logger.info(
            "Live transfer: loading '%s' rank %d (GPU %d)", model_name, mpi_rank, device_id
        )

        # 1. Query source metadata
        query_timeout = int(os.environ.get("MX_SOURCE_QUERY_TIMEOUT", "3600"))
        source_meta = self._query_source(mx_server, model_name, timeout=query_timeout)

        # Find my rank's source worker — use MPI rank, not local GPU index
        my_workers = [w for w in source_meta.workers if w.worker_rank == mpi_rank]
        if not my_workers:
            raise RuntimeError(
                f"No source worker for rank {mpi_rank} (device_id={device_id}). "
                f"Source has workers: {[w.worker_rank for w in source_meta.workers]}"
            )
        source_worker = my_workers[0]

        # 2. Build name→param map from target model
        target_params = {}
        for name, param in model.named_parameters():
            if param.device.index == device_id:
                target_params[name] = param.data

        logger.info(
            "Target has %d params on GPU %d", len(target_params), device_id
        )

        # 3. Build source name→descriptor map
        source_descs = {t.name: t for t in source_worker.tensors}

        # 4. Match source and target by name
        matched = []
        dtype_cast_needed = []
        unmatched_source = []
        for src_name, src_desc in source_descs.items():
            if src_name in target_params:
                dst_param = target_params[src_name]
                src_size = src_desc.size
                dst_size = dst_param.numel() * dst_param.element_size()
                if src_size == dst_size:
                    matched.append((src_name, src_desc, dst_param))
                else:
                    # Check if element count matches but dtype differs
                    src_dtype_str = src_desc.dtype
                    src_elem_size = 2 if "bfloat16" in src_dtype_str or "float16" in src_dtype_str else 4 if "float32" in src_dtype_str else 1
                    src_numel = src_size // src_elem_size if src_elem_size > 0 else 0
                    if src_numel == dst_param.numel():
                        logger.info(
                            "Dtype mismatch for %s: source=%s(%d bytes) target=%s(%d bytes) — will cast after transfer",
                            src_name, src_dtype_str, src_size, dst_param.dtype, dst_size,
                        )
                        dtype_cast_needed.append((src_name, src_desc, dst_param, src_dtype_str))
                    else:
                        logger.warning(
                            "Size mismatch for %s: source=%d target=%d (numel src=%d dst=%d)",
                            src_name, src_size, dst_size, src_numel, dst_param.numel(),
                        )
            else:
                unmatched_source.append(src_name)

        if unmatched_source:
            logger.warning(
                "%d source tensors not found in target: %s...",
                len(unmatched_source), unmatched_source[:3],
            )

        # For dtype-mismatched tensors, allocate temp buffers at source dtype
        dtype_map = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16,
                     "torch.float32": torch.float32, "torch.uint8": torch.uint8,
                     "torch.float8_e4m3fn": torch.float8_e4m3fn}
        cast_buffers = {}
        for src_name, src_desc, dst_param, src_dtype_str in dtype_cast_needed:
            src_torch_dtype = dtype_map.get(src_dtype_str, torch.bfloat16)
            buf = torch.empty(dst_param.numel(), dtype=src_torch_dtype, device=f"cuda:{device_id}")
            cast_buffers[src_name] = (buf, dst_param)
            matched.append((src_name, src_desc, buf))

        logger.info(
            "Matched %d/%d params for direct RDMA transfer (%d need dtype cast)",
            len(matched), len(source_descs), len(dtype_cast_needed),
        )

        # 5. Initialize NIXL and register TARGET param buffers
        nixl_mgr = NixlTransferManager(
            agent_name=f"trtllm-live-target-rank{mpi_rank}-{os.getpid()}",
            device_id=device_id,
        )
        nixl_mgr.initialize()

        # Register target params with NIXL (includes temp cast buffers)
        dst_tensors = {name: param for name, _, param in matched}
        nixl_mgr.register_tensors(dst_tensors)

        # 6. Build source descriptors for NIXL transfer
        src_descs_for_transfer = [
            TensorDescriptor(
                name=name,
                addr=src_desc.addr,
                size=src_desc.size,
                device_id=src_desc.device_id,
                dtype=src_desc.dtype,
            )
            for name, src_desc, _ in matched
        ]

        # 7. RDMA transfer: source params → target params
        coalesce = os.environ.get("MX_COALESCE_TRANSFERS", "0") == "1"
        xfer_timeout = int(os.environ.get("MX_TRANSFER_TIMEOUT", "900"))
        t0 = time.perf_counter()
        bytes_transferred, n_tensors, _ = nixl_mgr.receive_from_source(
            source_metadata=source_worker.nixl_metadata,
            source_tensors=src_descs_for_transfer,
            timeout_seconds=xfer_timeout,
            coalesce_transfers=coalesce,
        )
        elapsed = time.perf_counter() - t0
        bw = (bytes_transferred * 8) / (elapsed * 1e9) if elapsed > 0 else 0

        logger.info(
            "Rank %d: transferred %d params (%.2f GB) in %.2fs (%.1f Gbps) — DIRECT into model params",
            mpi_rank, n_tensors, bytes_transferred / 1e9, elapsed, bw,
        )

        # Diagnostic: checksum first few params to verify RDMA data
        torch.cuda.synchronize(device_id)
        for name, _, dst_param in matched[:5]:
            val = dst_param.to(torch.float32)
            cksum = val.sum().item()
            nonzero = (dst_param != 0).sum().item()
            logger.info(
                "CHECKSUM rank %d: %s shape=%s dtype=%s sum=%.4f nonzero=%d/%d",
                mpi_rank, name, list(dst_param.shape), dst_param.dtype,
                cksum, nonzero, dst_param.numel(),
            )

        # 7.5. Apply dtype casts for mismatched tensors
        for src_name, (buf, dst_param) in cast_buffers.items():
            dst_param.data.copy_(buf.to(dst_param.dtype))
            logger.info("Cast %s: %s → %s", src_name, buf.dtype, dst_param.dtype)
        if cast_buffers:
            logger.info("Applied %d dtype casts", len(cast_buffers))

        nixl_mgr.shutdown()

        # 8. Load any size-mismatched tensors from PVC checkpoint as fallback
        fallback_weights = {}
        size_mismatched = {
            src_name for src_name, src_desc in source_descs.items()
            if src_name in target_params
            and src_desc.size != target_params[src_name].numel() * target_params[src_name].element_size()
        }
        if size_mismatched:
            logger.info(
                "Loading %d size-mismatched tensors from PVC fallback: %s...",
                len(size_mismatched), list(size_mismatched)[:3],
            )
            try:
                from safetensors import safe_open
                import glob as _glob
                safetensor_files = sorted(_glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
                for sf_path in safetensor_files:
                    with safe_open(sf_path, framework="pt", device=f"cuda:{device_id}") as f:
                        for key in f.keys():
                            if key in size_mismatched:
                                fallback_weights[key] = f.get_tensor(key)
                                size_mismatched.discard(key)
                    if not size_mismatched:
                        break
                if size_mismatched:
                    logger.warning("Still missing after PVC fallback: %s", size_mismatched)
            except Exception as e:
                logger.warning("PVC fallback failed: %s", e)

        # Return fallback weights for TRT-LLM to apply; P2P weights are already in model params
        return fallback_weights

    def cleanup(self):
        pass

    def _query_source(self, mx_server, model_name, timeout=600):
        import grpc

        identity = _build_trtllm_identity(model_name=model_name)
        mx_client = MxClient(server_url=mx_server)

        start = time.time()
        while time.time() - start < timeout:
            try:
                list_resp = mx_client.list_sources(
                    identity=identity,
                )
                if list_resp.instances:
                    workers = []
                    for inst in list_resp.instances:
                        meta_resp = mx_client.get_metadata(
                            mx_source_id=inst.mx_source_id,
                            worker_id=inst.worker_id,
                        )
                        if meta_resp.found and meta_resp.worker.tensors:
                            workers.append(meta_resp.worker)

                    if workers:
                        logger.info("Found source: %d workers", len(workers))

                        class _SourceMeta:
                            pass

                        result = _SourceMeta()
                        result.workers = workers
                        self._source_meta = result
                        mx_client.close()
                        return result
            except grpc.RpcError as e:
                logger.warning("Query failed: %s", e)
            time.sleep(5)

        mx_client.close()
        raise TimeoutError(f"Source for '{model_name}' not found after {timeout}s")


def _import_trtllm_for_config():
    from tensorrt_llm._torch.models.checkpoints.hf.config_loader import (
        HfConfigLoader,
    )

    return {"HfConfigLoader": HfConfigLoader}


class MxConfigLoader:
    def load(self, checkpoint_dir: str, **kwargs):
        trtllm = _import_trtllm_for_config()
        HfConfigLoader = trtllm["HfConfigLoader"]

        logger.info("Loading config from local path: %s", checkpoint_dir)
        return HfConfigLoader().load(checkpoint_dir, **kwargs)

    def cleanup(self):
        pass


class MxLiveCheckpointLoader:
    """
    Checkpoint loader that uses MxLiveWeightLoader for direct param-to-param transfer.

    Combines MxConfigLoader (config from MX server) with MxLiveWeightLoader
    (direct RDMA into model params).
    """

    def __init__(self, mx_server: Optional[str] = None):
        # Pass mx_server to weight loader so it's available even if env var isn't set
        # when load_weights() is called in a different process context
        self._weight_loader = MxLiveWeightLoader(mx_server=mx_server)
        self._config_loader = None  # Lazy init
        self._weight_mapper = None
        self._checkpoint_format = "mx-p2p"

    def get_default_weight_loader(self):
        return MxLiveWeightLoader()

    def get_default_config_loader(self):
        return MxConfigLoader()

    def cleanup(self):
        if self._weight_mapper is not None and hasattr(self._weight_mapper, 'cleanup'):
            self._weight_mapper.cleanup()
        if self._weight_loader is not None:
            self._weight_loader.cleanup()

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
        if self._config_loader is None:
            self._config_loader = self.get_default_config_loader()
        return self._config_loader

    @property
    def checkpoint_format(self):
        return self._checkpoint_format

    def load_config(self, checkpoint_dir: str, **kwargs):
        logger.info("MxLiveCheckpointLoader.load_config(%s)", checkpoint_dir)
        return self.config_loader.load(checkpoint_dir, **kwargs)

    def load_weights(self, checkpoint_dir: str, mapping=None, model=None, **kwargs):
        logger.info("MxLiveCheckpointLoader.load_weights(model=%s)", model is not None)
        return self._weight_loader.load_weights(
            checkpoint_dir, mapping=mapping, model=model, **kwargs
        )

    def get_initialized_weight_mapper(self, model, config):
        from tensorrt_llm._torch.models.checkpoints.auto_mapper import AutoCheckpointMapper

        if config.pretrained_config and config.pretrained_config.architectures:
            model_arch = config.pretrained_config.architectures[0]
        else:
            raise ValueError("Cannot determine model architecture from config")

        weight_mapper = AutoCheckpointMapper.get("HF", model_arch)
        weight_mapper.init_model_and_config(model, config)
        self._weight_mapper = weight_mapper
        return weight_mapper
