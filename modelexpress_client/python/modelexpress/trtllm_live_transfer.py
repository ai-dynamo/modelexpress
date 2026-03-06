# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress Live Model P2P Transfer for TensorRT-LLM.

Transfers model weights directly between running TRT-LLM instances via NIXL RDMA.
Source registers its model parameter GPU buffers; target receives into its own
model parameter buffers. No format conversion, no disk I/O, no CPU round-trip.

Source usage (after model is loaded):
    from modelexpress.trtllm_live_transfer import MxLiveSource
    llm = LLM(model="Llama-70B", tp=8)
    source = MxLiveSource(llm, "llama-70b", "modelexpress-server:8001")
    source.publish()  # Registers GPU params with NIXL, publishes metadata
    # Continue serving inference — weights stay in GPU memory

Target usage (via checkpoint_loader):
    from modelexpress.trtllm_live_transfer import MxLiveWeightLoader
    loader = MxLiveCheckpointLoader()
    llm = LLM(model="Llama-70B", checkpoint_loader=loader,
              load_format=LoadFormat.PRESHARDED, tp=8)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import torch

logger = logging.getLogger("modelexpress.trtllm_live_transfer")


class MxLiveSource:
    """
    Publishes weights from a running TRT-LLM model's GPU memory.

    The model's parameters are already fused (qkv_proj), TP-sharded, and on GPU.
    We register them directly with NIXL — no loading from disk, no sharding,
    no format conversion.
    """

    def __init__(
        self,
        model: Any,
        model_name: str,
        mx_server: str = "modelexpress-server:8001",
        model_path: Optional[str] = None,
    ):
        self._model = model
        self._model_name = model_name
        self._mx_server = mx_server
        self._model_path = model_path
        self._nixl_managers = {}
        self._published = False

    def publish(self):
        """Register model params with NIXL and publish metadata to MX server."""
        from .nixl_transfer import NixlTransferManager
        from . import p2p_pb2, p2p_pb2_grpc
        import grpc

        # Get the underlying torch model from LLM wrapper
        # TRT-LLM's LLM class wraps the model differently depending on version
        torch_model = self._get_torch_model()
        if torch_model is None:
            raise RuntimeError("Cannot access underlying torch model from LLM instance")

        device_id = torch.cuda.current_device()
        logger.info(
            "Publishing live model '%s' from GPU %d", self._model_name, device_id
        )

        # Collect model parameters on this device
        param_tensors = {}
        total_bytes = 0
        for name, param in torch_model.named_parameters():
            if param.device.index == device_id:
                param_tensors[name] = param.data
                total_bytes += param.numel() * param.element_size()

        logger.info(
            "Found %d params on GPU %d (%.2f GB)",
            len(param_tensors), device_id, total_bytes / 1e9,
        )

        # Initialize NIXL and register param buffers
        nixl_mgr = NixlTransferManager(
            agent_name=f"trtllm-live-source-rank{device_id}-{os.getpid()}",
            device_id=device_id,
        )
        nixl_mgr.initialize()
        nixl_mgr.register_tensors(param_tensors)
        self._nixl_managers[device_id] = nixl_mgr

        # Collect config files from HF cache if available
        model_files = self._collect_model_files()

        # Publish to MX server
        options = [
            ('grpc.max_send_message_length', 200 * 1024 * 1024),
            ('grpc.max_receive_message_length', 200 * 1024 * 1024),
        ]
        channel = grpc.insecure_channel(self._mx_server, options=options)
        stub = p2p_pb2_grpc.P2pServiceStub(channel)

        tensor_protos = []
        for name, tensor in param_tensors.items():
            tensor_protos.append(p2p_pb2.TensorDescriptor(
                name=name,
                addr=tensor.data_ptr(),
                size=tensor.numel() * tensor.element_size(),
                device_id=device_id,
                dtype=str(tensor.dtype),
            ))

        workers = [p2p_pb2.WorkerMetadata(
            worker_rank=device_id,
            nixl_metadata=nixl_mgr.nixl_metadata,
            tensors=tensor_protos,
        )]

        request_kwargs = {
            "model_name": self._model_name,
            "workers": workers,
        }
        if model_files and hasattr(p2p_pb2.PublishMetadataRequest, "DESCRIPTOR"):
            desc = p2p_pb2.PublishMetadataRequest.DESCRIPTOR
            if any(f.name == "model_files" for f in desc.fields):
                request_kwargs["model_files"] = model_files
            else:
                logger.warning("Proto has no model_files field, skipping config publish")
        request = p2p_pb2.PublishMetadataRequest(**request_kwargs)
        response = stub.PublishMetadata(request)
        channel.close()

        if not response.success:
            raise RuntimeError(f"Failed to publish: {response.message}")

        self._published = True
        logger.info(
            "Published %d params (%.2f GB) for '%s' rank %d",
            len(param_tensors), total_bytes / 1e9, self._model_name, device_id,
        )

    def _get_torch_model(self):
        """Extract the underlying torch model.
        
        TRT-LLM 1.3.0rc5 structure:
        - LLM._executor (PyExecutor)
        - PyExecutor.model_engine (PyTorchModelEngine)
        - PyTorchModelEngine.model (torch.nn.Module)
        """
        model = self._model
        logger.info(f"_get_torch_model: model type = {type(model)}")
        
        # If model itself has named_parameters (direct torch model), use it
        if hasattr(model, 'named_parameters'):
            try:
                # Quick check: try to iterate one param to verify it's a real model
                next(iter(model.named_parameters()), None)
                logger.info("_get_torch_model: found direct torch model")
                return model
            except (AttributeError, TypeError) as e:
                logger.debug(f"_get_torch_model: direct model check failed: {e}")
        
        # Try TRT-LLM 1.3.0rc5 path: _executor.model_engine.model
        if hasattr(model, '_executor'):
            try:
                executor = getattr(model, '_executor')
                logger.info(f"_get_torch_model: _executor = {executor}, type = {type(executor)}")
                if executor is not None:
                    if hasattr(executor, 'model_engine'):
                        try:
                            model_engine = getattr(executor, 'model_engine')
                            logger.info(f"_get_torch_model: model_engine = {model_engine}, type = {type(model_engine)}")
                            if model_engine is not None:
                                if hasattr(model_engine, 'model'):
                                    try:
                                        torch_model = getattr(model_engine, 'model')
                                        logger.info(f"_get_torch_model: torch_model = {torch_model}, type = {type(torch_model)}")
                                        if hasattr(torch_model, 'named_parameters'):
                                            try:
                                                # Verify it's a real model
                                                next(iter(torch_model.named_parameters()), None)
                                                logger.info("_get_torch_model: found model via _executor.model_engine.model")
                                                return torch_model
                                            except (AttributeError, TypeError, StopIteration) as e:
                                                logger.warning(f"_get_torch_model: torch_model has named_parameters but iteration failed: {e}")
                                    except Exception as e:
                                        logger.warning(f"_get_torch_model: failed to get model from model_engine: {e}")
                                else:
                                    logger.info("_get_torch_model: model_engine has no 'model' attribute")
                                    logger.info(f"_get_torch_model: model_engine dir = {[a for a in dir(model_engine) if not a.startswith('__')][:20]}")
                            else:
                                logger.info("_get_torch_model: model_engine is None")
                        except Exception as e:
                            logger.warning(f"_get_torch_model: failed to get model_engine from executor: {e}")
                    else:
                        logger.info("_get_torch_model: executor has no 'model_engine' attribute")
                        logger.info(f"_get_torch_model: executor dir = {[a for a in dir(executor) if not a.startswith('__')][:20]}")
                else:
                    logger.info("_get_torch_model: _executor is None")
            except Exception as e:
                logger.warning(f"_get_torch_model: failed to access _executor: {e}")
        
        # Try legacy/common attribute paths for LLM wrapper
        for attr in ['_model', 'model', '_engine']:
            if hasattr(model, attr):
                candidate = getattr(model, attr)
                logger.debug(f"_get_torch_model: trying attr '{attr}' = {candidate}, type = {type(candidate)}")
                if candidate is not None and hasattr(candidate, 'named_parameters'):
                    try:
                        # Verify it's a real model
                        next(iter(candidate.named_parameters()), None)
                        logger.debug(f"_get_torch_model: found model via '{attr}'")
                        return candidate
                    except (AttributeError, TypeError, StopIteration) as e:
                        logger.debug(f"_get_torch_model: '{attr}' candidate failed: {e}")
                        continue
        
        logger.error(f"_get_torch_model: failed to find torch model. Model type: {type(model)}, dir(model): {[a for a in dir(model) if not a.startswith('__')][:20]}")
        return None

    def _collect_model_files(self) -> dict[str, bytes]:
        """Collect config files for target config loading.

        If model_path is set, reads directly from that directory (reliable).
        Otherwise falls back to searching the HF cache (fragile when multiple
        models are cached — may pick up the wrong config.json).
        """
        config_names = [
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "generation_config.json", "special_tokens_map.json",
        ]
        model_files: dict[str, bytes] = {}

        if self._model_path and os.path.isdir(self._model_path):
            for fname in config_names:
                fpath = os.path.join(self._model_path, fname)
                if os.path.exists(fpath):
                    with open(fpath, "rb") as f:
                        model_files[fname] = f.read()
            if model_files:
                logger.info("Collected %d config files from %s", len(model_files), self._model_path)
            return model_files

        # Fallback: search HF cache (unreliable with multiple models)
        import glob
        hf_cache = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
        for fname in config_names:
            matches = glob.glob(f"{hf_cache}/**/snapshots/**/{fname}", recursive=True)
            if matches:
                try:
                    with open(matches[0], "rb") as f:
                        model_files[fname] = f.read()
                except Exception:
                    pass
        if model_files:
            logger.info("Collected %d config files for target (cache fallback)", len(model_files))
        return model_files

    def shutdown(self):
        """Clean up NIXL resources."""
        for nixl_mgr in self._nixl_managers.values():
            nixl_mgr.shutdown()
        self._nixl_managers.clear()


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
        import grpc
        from . import p2p_pb2, p2p_pb2_grpc

        # Use provided URL, then env var, then default
        mx_server = self._mx_server or os.environ.get("MODEL_EXPRESS_URL") or os.environ.get("MX_SERVER_ADDRESS", "localhost:8001")
        model_name = os.environ.get("MODEL_NAME", os.path.basename(checkpoint_dir))

        if model is None:
            raise RuntimeError(
                "MxLiveWeightLoader requires model reference. "
                "Use load_format=LoadFormat.PRESHARDED to pass model."
            )

        device_id = torch.cuda.current_device()

        # MPI workers' stdout is swallowed by TRT-LLM — write to per-rank file
        log_dir = os.environ.get("MX_TRANSFER_LOG_DIR", "/tmp/mx_logs")
        os.makedirs(log_dir, exist_ok=True)
        rank_log = os.path.join(log_dir, f"rank{device_id}.log")
        fh = logging.FileHandler(rank_log, mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logging.getLogger("modelexpress").addHandler(fh)

        logger.info(
            "Live transfer: loading '%s' into model on GPU %d", model_name, device_id
        )

        # 1. Query source metadata
        query_timeout = int(os.environ.get("MX_SOURCE_QUERY_TIMEOUT", "3600"))
        source_meta = self._query_source(mx_server, model_name, timeout=query_timeout)

        # Find my rank's source worker
        my_workers = [w for w in source_meta.workers if w.worker_rank == device_id]
        if not my_workers:
            raise RuntimeError(f"No source worker for rank {device_id}")
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
        unmatched_source = []
        for src_name, src_desc in source_descs.items():
            if src_name in target_params:
                dst_param = target_params[src_name]
                src_size = src_desc.size
                dst_size = dst_param.numel() * dst_param.element_size()
                if src_size == dst_size:
                    matched.append((src_name, src_desc, dst_param))
                else:
                    logger.warning(
                        "Size mismatch for %s: source=%d target=%d",
                        src_name, src_size, dst_size,
                    )
            else:
                unmatched_source.append(src_name)

        if unmatched_source:
            logger.warning(
                "%d source tensors not found in target: %s...",
                len(unmatched_source), unmatched_source[:3],
            )

        logger.info(
            "Matched %d/%d params for direct RDMA transfer",
            len(matched), len(source_descs),
        )

        # 5. Initialize NIXL and register TARGET param buffers
        nixl_mgr = NixlTransferManager(
            agent_name=f"trtllm-live-target-rank{device_id}-{os.getpid()}",
            device_id=device_id,
        )
        nixl_mgr.initialize()

        # Register target params with NIXL
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
        t0 = time.perf_counter()
        bytes_transferred, n_tensors, _ = nixl_mgr.receive_from_source(
            source_metadata=source_worker.nixl_metadata,
            source_tensors=src_descs_for_transfer,
            timeout_seconds=300,
            coalesce_transfers=coalesce,
        )
        elapsed = time.perf_counter() - t0
        bw = (bytes_transferred * 8) / (elapsed * 1e9) if elapsed > 0 else 0

        logger.info(
            "Rank %d: transferred %d params (%.2f GB) in %.2fs (%.1f Gbps) — DIRECT into model params",
            device_id, n_tensors, bytes_transferred / 1e9, elapsed, bw,
        )

        nixl_mgr.shutdown()

        # 8. Return empty dict — weights are already in model params
        return {}

    def cleanup(self):
        pass

    def _query_source(self, mx_server, model_name, timeout=600):
        import grpc
        from . import p2p_pb2, p2p_pb2_grpc

        options = [("grpc.max_receive_message_length", 200 * 1024 * 1024)]
        channel = grpc.insecure_channel(mx_server, options=options)
        stub = p2p_pb2_grpc.P2pServiceStub(channel)

        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = stub.GetMetadata(
                    p2p_pb2.GetMetadataRequest(model_name=model_name)
                )
                if resp.found and len(resp.workers) > 0:
                    logger.info("Found source: %d workers", len(resp.workers))
                    self._source_meta = resp
                    channel.close()
                    return resp
            except Exception as e:
                logger.warning("Query failed: %s", e)
            time.sleep(5)

        channel.close()
        raise TimeoutError(f"Source for '{model_name}' not found after {timeout}s")


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
        from .trtllm_checkpoint_loader import MxConfigLoader
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
