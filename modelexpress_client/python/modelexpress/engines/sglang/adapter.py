# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang implementation of the ModelExpress engine adapter contract."""

from __future__ import annotations

import copy
import logging
import os
import uuid
from importlib.metadata import version as pkg_version
from typing import TYPE_CHECKING, Iterator

import torch

from ... import p2p_pb2
from ...adapter import EngineAdapter
from ...accelerators import accelerator_backend_for
from ...load_strategy.context import LoadContext, LoadResult
from ...metadata.client_factory import create_metadata_client

logger = logging.getLogger("modelexpress.engines.sglang.adapter")

if TYPE_CHECKING:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig


class SglangAdapter(EngineAdapter):
    """Adapter that maps strategy hooks onto SGLang's native loader APIs."""

    def __init__(
        self,
        load_config: LoadConfig,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ):
        self.load_config = load_config
        self.model_config = model_config
        self.device_config = device_config
        self.target_device = torch.device(device_config.device)
        self.accelerator_backend = accelerator_backend_for(self.target_device)

    def build_identity(self) -> p2p_pb2.SourceIdentity:
        return build_sglang_source_identity(
            model_config=self.model_config,
        )

    def get_worker_rank(self) -> int:
        return _get_sglang_worker_rank(self.load_config)

    def get_global_rank(self) -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
        return self.get_worker_rank()

    def get_device_id(self) -> int:
        gpu_id = getattr(self.device_config, "gpu_id", None)
        if gpu_id is not None:
            return int(gpu_id)
        if self.target_device.index is not None:
            return int(self.target_device.index)
        return 0

    def get_target_device(self) -> torch.device:
        return self.target_device

    def is_cuda_alike(self) -> bool:
        from sglang.srt.utils import is_cuda_alike

        return bool(is_cuda_alike())

    def discover_tensors(self, result: LoadResult) -> dict[str, torch.Tensor]:
        if result.model is None:
            raise RuntimeError("SGLang tensor discovery requires result.model")
        return self._collect_tensors(result.model)

    def _collect_tensors(self, model) -> dict[str, torch.Tensor]:
        """Collect SGLang model parameters for NIXL registration.

        SGLang's current NIXL path registers contiguous parameters directly and
        registers a byte view of the underlying storage for non-contiguous
        parameters. Keep that naming behavior so source and target descriptors
        match the upstream integration.
        """
        backend = self.accelerator_backend
        tensors: dict[str, torch.Tensor] = {}
        seen_ptrs: set[int] = set()

        for name, param in model.named_parameters():
            t = param.data
            if not backend.is_accel_tensor(t):
                continue
            if t.is_contiguous():
                tensor_name = name
                registered = t
            else:
                tensor_name = f"{name}.__storage"
                registered = torch.empty(0, dtype=torch.uint8, device=t.device).set_(
                    t.untyped_storage()
                )

            ptr = registered.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            tensors[tensor_name] = registered

        return tensors

    def before_rdma_receive(self, result: LoadResult) -> LoadResult:
        return self._process_weights_after_loading(result)

    def after_rdma_receive(self, result: LoadResult) -> LoadResult:
        return self._post_load_weights(result)

    def apply_weight_iter(
        self,
        result: LoadResult,
        weights_iter: Iterator[tuple[str, torch.Tensor]],
    ) -> LoadResult:
        if result.model is None:
            raise RuntimeError("SGLang weight iterator loading requires result.model")
        result.model.load_weights(weights_iter)
        return result

    def build_model_streamer_weight_iter(
        self,
        model_uri: str,
        model: torch.nn.Module | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        if model is None:
            raise RuntimeError("SGLang ModelStreamer loading requires result.model")

        from sglang.srt.configs.load_config import LoadFormat
        from sglang.srt.model_loader.loader import RunaiModelStreamerLoader

        stream_config = copy.copy(self.load_config)
        _set_load_config_attr(stream_config, "load_format", LoadFormat.RUNAI_STREAMER)
        extra_config = dict(
            getattr(stream_config, "model_loader_extra_config", None) or {}
        )
        if self._model_streamer_distributed_enabled():
            extra_config["distributed"] = True
        _set_load_config_attr(stream_config, "model_loader_extra_config", extra_config)

        stream_model_config = copy.copy(self.model_config)
        _set_load_config_attr(stream_model_config, "model_weights", model_uri)

        loader = RunaiModelStreamerLoader(stream_config)
        loader.target_device_str = str(self.target_device)
        return loader._get_all_weights(stream_model_config, model)

    def after_weight_iter_load(self, result: LoadResult) -> LoadResult:
        return self._process_weights_after_loading(result)

    def load_via_native(self, result: LoadResult) -> LoadResult:
        if result.model is None:
            raise RuntimeError("SGLang native loading requires result.model")

        from sglang.srt.configs.load_config import LoadFormat
        from sglang.srt.model_loader.loader import DefaultModelLoader

        disk_config = copy.copy(self.load_config)
        disk_config.load_format = LoadFormat.AUTO
        disk_loader = DefaultModelLoader(disk_config)
        weights_iter = disk_loader._get_all_weights(self.model_config, result.model)
        DefaultModelLoader.load_weights_and_postprocess(
            result.model, weights_iter, self.target_device,
        )
        return result

    def reinit_for_retry(self, result: LoadResult) -> LoadResult:
        from sglang.srt.model_loader.loader import (
            _get_quantization_config,
            _initialize_model,
        )

        old_value = result.value
        result.value = None
        result.model = None
        del old_value
        self.accelerator_backend.empty_cache()

        logger.info(
            "[Worker %s] Re-initializing SGLang model after failed strategy",
            self.get_global_rank(),
        )
        quant_config = _get_quantization_config(self.model_config, self.load_config)
        with self.target_device:
            model = _initialize_model(
                self.model_config,
                self.load_config,
                quant_config,
            )
        return LoadResult(value=model, model=model, publishable=result.publishable)

    def _process_weights_after_loading(self, result: LoadResult) -> LoadResult:
        if result.model is None:
            raise RuntimeError("SGLang post-load processing requires result.model")

        from sglang.srt.model_loader.loader import device_loading_context

        for _, module in result.model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                with device_loading_context(module, self.target_device):
                    quant_method.process_weights_after_loading(module)
        return result

    def _post_load_weights(self, result: LoadResult) -> LoadResult:
        if result.model is None:
            raise RuntimeError("SGLang post-load fixup requires result.model")
        _call_sglang_post_load_weights(result.model)
        return result

    def _model_streamer_distributed_enabled(self) -> bool:
        tp_size = int(getattr(self.build_identity(), "tensor_parallel_size", 1) or 1)
        return (
            tp_size > 1
            and self.is_cuda_alike()
            and os.environ.get("MX_MS_DISTRIBUTED", "0").lower() in ("1", "true")
        )


def _set_load_config_attr(obj, name: str, value) -> None:
    try:
        setattr(obj, name, value)
    except AttributeError:
        object.__setattr__(obj, name, value)


def _call_sglang_post_load_weights(model: torch.nn.Module) -> None:
    post_load_weights = getattr(model, "post_load_weights", None)
    if callable(post_load_weights):
        post_load_weights()
        return

    for child in model.children():
        post_load_weights = getattr(child, "post_load_weights", None)
        if callable(post_load_weights):
            post_load_weights()


def build_sglang_source_identity(model_config: ModelConfig) -> p2p_pb2.SourceIdentity:
    """Build a ModelExpress SourceIdentity from SGLang model state."""
    try:
        mx_version = pkg_version("modelexpress")
    except Exception:
        mx_version = "0.0.0"

    return p2p_pb2.SourceIdentity(
        mx_version=mx_version,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=_get_model_name(model_config),
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        tensor_parallel_size=_get_parallel_size(
            "get_tensor_model_parallel_world_size"
        ),
        pipeline_parallel_size=_get_parallel_size(
            "get_pipeline_model_parallel_world_size"
        ),
        expert_parallel_size=_get_parallel_size(
            "get_moe_expert_parallel_world_size"
        ),
        dtype=_get_dtype(model_config),
        quantization=_get_quantization(model_config),
        revision=_get_revision(model_config),
    )


def _get_model_name(model_config: ModelConfig) -> str:
    return str(
        getattr(
            model_config,
            "model_path",
            getattr(model_config, "model", ""),
        )
    )


def _get_dtype(model_config: ModelConfig) -> str:
    dtype = getattr(model_config, "dtype", "")
    return str(dtype).replace("torch.", "")


def _get_quantization(model_config: ModelConfig) -> str:
    return str(getattr(model_config, "quantization", "") or "")


def _get_revision(model_config: ModelConfig) -> str:
    override = os.environ.get("MX_MODEL_REVISION", "")
    if override:
        return override
    return str(getattr(model_config, "revision", "") or "")


def _get_parallel_size(name: str) -> int:
    try:
        from sglang.srt import distributed

        return int(getattr(distributed, name)())
    except Exception:
        return 1


def _get_sglang_worker_rank(load_config: LoadConfig) -> int:
    """Return the SGLang model-parallel shard key, excluding DP replicas."""
    try:
        from sglang.srt import distributed

        tp_rank = int(distributed.get_tensor_model_parallel_rank())
        pp_rank = int(distributed.get_pipeline_model_parallel_rank())
        tp_size = int(distributed.get_tensor_model_parallel_world_size())
        return pp_rank * tp_size + tp_rank
    except Exception:
        return int(getattr(load_config, "tp_rank", 0) or 0)


def build_sglang_load_context(
    load_config: LoadConfig,
    model_config: ModelConfig,
    device_config: DeviceConfig,
) -> LoadContext:
    """Build a LoadContext from SGLang config objects."""

    adapter = SglangAdapter(load_config, model_config, device_config)
    worker_rank = adapter.get_worker_rank()
    global_rank = adapter.get_global_rank()
    server_url = getattr(load_config, "modelexpress_url", None)
    return LoadContext(
        model_config=model_config,
        load_config=load_config,
        target_device=adapter.get_target_device(),
        global_rank=global_rank,
        worker_rank=worker_rank,
        device_id=adapter.get_device_id(),
        identity=adapter.build_identity(),
        mx_client=create_metadata_client(
            worker_rank=worker_rank,
            server_url=server_url,
        ),
        worker_id=uuid.uuid4().hex[:8],
        adapter=adapter,
        accelerator_backend=adapter.accelerator_backend,
    )
