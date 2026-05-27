# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM implementation of the ModelExpress engine adapter contract."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import Any

import torch

from ... import p2p_pb2
from ...adapter import EngineAdapter
from ...load_strategy.context import LoadContext, LoadResult
from ...metadata.client_factory import (
    create_metadata_client,
    resolve_metadata_port,
    resolve_metadata_server_url,
)

logger = logging.getLogger("modelexpress.engines.trtllm.adapter")


class TrtllmAdapter(EngineAdapter):
    """Adapter that maps ModelExpress hooks onto TRT-LLM's live model object."""

    def __init__(
        self,
        *,
        model_name: str,
        checkpoint_dir: str | None = None,
        model: Any = None,
        mapping: Any = None,
        native_loader: Callable[[], dict[str, Any]] | None = None,
    ):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.mapping = _resolve_trtllm_mapping(mapping, model)
        self.native_loader = native_loader
        self.dtype = _resolve_trtllm_dtype(model)
        self.quantization = _resolve_trtllm_quantization(model)
        self.target_device = torch.device(f"cuda:{self.get_device_id()}")

    def build_identity(self):
        return _build_trtllm_identity(
            model_name=self.model_name,
            tp_size=int(getattr(self.mapping, "tp_size", 1) or 1),
            pp_size=int(getattr(self.mapping, "pp_size", 1) or 1),
            ep_size=int(getattr(self.mapping, "moe_ep_size", 1) or 1),
            dtype=_dtype_to_identity_string(self.dtype),
            quantization=self.quantization,
        )

    def get_worker_rank(self) -> int:
        return _get_trtllm_worker_rank(self.mapping, self.get_global_rank())

    def get_global_rank(self) -> int:
        if self.mapping is None:
            return self.get_device_id()
        return int(self.mapping.rank)

    def get_device_id(self) -> int:
        try:
            return int(torch.cuda.current_device())
        except Exception:
            return 0

    def get_target_device(self) -> torch.device:
        return self.target_device

    def is_cuda_alike(self) -> bool:
        return torch.cuda.is_available()

    def discover_tensors(self, result: LoadResult) -> dict[str, torch.Tensor]:
        model = result.model or self.model
        if model is None:
            raise RuntimeError("TRT-LLM tensor discovery requires result.model")
        tensors, _ = _collect_cuda_param_tensors(model, self.get_device_id())
        return tensors

    def load_via_native(self, result: LoadResult) -> LoadResult:
        if self.native_loader is None:
            raise RuntimeError("TRT-LLM native fallback loader is not configured")
        return LoadResult(
            value=self.native_loader(),
            model=None,
            publishable=False,
            metadata=result.metadata,
        )

    def after_rdma_receive(self, result: LoadResult) -> LoadResult:
        # TRT-LLM receivers already got weights over P2P and should not be
        # advertised as fresh sources before TRT-LLM's post-load lifecycle.
        result.publishable = False
        return result


def build_trtllm_load_context(
    *,
    model_name: str,
    checkpoint_dir: str | None,
    model: Any,
    mapping: Any = None,
    server_url: str | None = None,
    native_loader: Callable[[], dict[str, Any]] | None = None,
    source_query_timeout_s: int | None = None,
) -> LoadContext:
    """Build a LoadContext for TRT-LLM source publication and metadata lookup."""
    server_url = resolve_metadata_server_url(server_url)
    adapter = TrtllmAdapter(
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        model=model,
        mapping=mapping,
        native_loader=native_loader,
    )
    worker_rank = adapter.get_worker_rank()
    return LoadContext(
        model_config=TrtllmModelConfig(
            model_name,
            dtype=adapter.dtype,
            quantization=adapter.quantization,
            hf_text_config=_resolve_trtllm_hf_text_config(model),
        ),
        load_config=TrtllmLoadConfig(
            source_query_timeout_s=source_query_timeout_s,
        ),
        target_device=adapter.get_target_device(),
        global_rank=adapter.get_global_rank(),
        worker_rank=worker_rank,
        device_id=adapter.get_device_id(),
        identity=adapter.build_identity(),
        mx_client=create_metadata_client(
            worker_rank=worker_rank,
            server_url=server_url,
        ),
        worker_id=uuid.uuid4().hex[:8],
        metadata_server_url=server_url,
        metadata_port=resolve_metadata_port(),
        adapter=adapter,
    )


class TrtllmModelConfig:
    def __init__(
        self,
        model: str,
        *,
        dtype: Any,
        quantization: str,
        hf_text_config: Any,
    ):
        self.model = model
        self.model_weights = None
        self.hf_text_config = hf_text_config
        self.dtype = dtype
        self.quantization = quantization


class _TrtllmHfTextConfig:
    model_type = "unknown"


class TrtllmLoadConfig:
    def __init__(self, *, source_query_timeout_s: int | None = None):
        self.source_query_timeout_s = source_query_timeout_s


def _build_trtllm_identity(
    model_name: str,
    tp_size: int = 1,
    pp_size: int = 1,
    ep_size: int = 1,
    dtype: str = "unknown",
    quantization: str = "",
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
        pipeline_parallel_size=pp_size,
        expert_parallel_size=ep_size,
        dtype=dtype,
        quantization=quantization,
    )


def _resolve_trtllm_dtype(model: Any) -> Any:
    """Resolve dtype from TRT-LLM runtime state without inventing a default."""
    for param in _iter_model_parameters(model):
        dtype = getattr(param, "dtype", None)
        if dtype is not None:
            return dtype

    model_config = getattr(model, "model_config", None)
    pretrained_config = getattr(model_config, "pretrained_config", None)
    runtime_config = getattr(model, "config", None)
    for owner, attr in (
        (runtime_config, "torch_dtype"),
        (runtime_config, "dtype"),
        (pretrained_config, "torch_dtype"),
        (pretrained_config, "dtype"),
    ):
        value = getattr(owner, attr, None)
        if value is not None:
            return value
    return "unknown"


def _iter_model_parameters(model: Any):
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return ()
    try:
        return parameters()
    except Exception:
        return ()


def _dtype_to_identity_string(dtype: Any) -> str:
    if dtype is None:
        return "unknown"
    return str(dtype).replace("torch.", "")


def _resolve_trtllm_quantization(model: Any) -> str:
    model_config = getattr(model, "model_config", None)
    quantization = getattr(model_config, "quantization", None)
    if quantization:
        return str(quantization)

    quant_config = getattr(model_config, "quant_config", None)
    quant_algo = getattr(quant_config, "quant_algo", None)
    if quant_algo:
        return str(quant_algo)

    pretrained_config = getattr(model_config, "pretrained_config", None)
    hf_quant_config = getattr(pretrained_config, "quantization_config", None)
    if isinstance(hf_quant_config, dict):
        quant_method = (
            hf_quant_config.get("quant_method")
            or hf_quant_config.get("quant_algo")
            or hf_quant_config.get("type")
        )
        return str(quant_method or "")
    if hf_quant_config:
        return str(hf_quant_config)
    return ""


def _resolve_trtllm_hf_text_config(model: Any) -> Any:
    model_config = getattr(model, "model_config", None)
    pretrained_config = getattr(model_config, "pretrained_config", None)
    if pretrained_config is not None:
        return pretrained_config
    runtime_config = getattr(model, "config", None)
    if runtime_config is not None:
        return runtime_config
    return _TrtllmHfTextConfig()


def _resolve_trtllm_mapping(mapping: Any, model: Any) -> Any:
    if mapping is not None:
        return mapping
    model_mapping = getattr(model, "mapping", None)
    if model_mapping is not None:
        return model_mapping
    model_config = getattr(model, "model_config", None)
    return getattr(model_config, "mapping", None)


def _get_trtllm_worker_rank(mapping: Any, default: int) -> int:
    """Return the model-weight shard key for source/target matching."""
    if mapping is None:
        return int(default)

    tp_size = int(getattr(mapping, "tp_size", 1) or 1)
    rank = int(mapping.rank)
    cp_size = int(getattr(mapping, "cp_size", 1) or 1)
    tp_cp_size = max(1, tp_size * cp_size)
    # TRT-LLM defines pp_rank = rank // (tp_size * cp_size) and
    # tp_rank = rank % (tp_size * cp_size) // cp_size. Context-parallel ranks
    # split sequence work, not model weights, so source keys only on PP/TP shards.
    return (rank // tp_cp_size) * tp_size + (rank % tp_cp_size) // max(1, cp_size)


def _collect_cuda_param_tensors(
    torch_model: Any,
    device_id: int,
) -> tuple[dict[str, Any], int]:
    param_tensors = {}
    seen_data_ptrs = set()
    total_bytes = 0
    for name, param in torch_model.named_parameters():
        if param.device.type != "cuda" or param.device.index != device_id:
            continue
        tensor = param.data
        ptr = tensor.data_ptr()
        if ptr in seen_data_ptrs:
            logger.debug("Skipping aliased param: %s (ptr=%x)", name, ptr)
            continue
        seen_data_ptrs.add(ptr)
        param_tensors[name] = tensor
        total_bytes += tensor.numel() * tensor.element_size()
    return param_tensors, total_bytes


__all__ = [
    "TrtllmAdapter",
    "TrtllmLoadConfig",
    "TrtllmModelConfig",
    "build_trtllm_load_context",
]
