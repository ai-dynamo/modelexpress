# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM implementation of the ModelExpress engine adapter contract."""

from __future__ import annotations

import copy
import gc
import logging
import os
import uuid
from contextlib import nullcontext
from typing import TYPE_CHECKING, Iterator

import torch

from ...adapter import EngineAdapter
from ...load_strategy.context import LoadContext, LoadResult
from ...metadata.client_factory import create_metadata_client
from ...metadata.publish import build_source_identity
from ...rank_utils import get_global_rank
from ...tensor_utils import (
    adopt_hidden_tensors,
    capture_tensor_attrs,
    collect_module_tensors,
    debug_tensor_enabled,
    tensor_debug_summary,
)
from ...quantization_providers import (
    MANIFEST_TENSOR_OVERRIDES_ATTR,
    SOURCE_MANIFEST_TENSOR_NAMES_ATTR,
    get_quantization_provider,
)
from ...types import ManifestMismatchError

logger = logging.getLogger("modelexpress.engines.vllm.adapter")

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class VllmAdapter(EngineAdapter):
    """Adapter that maps strategy hooks onto vLLM's native loader APIs."""

    def __init__(self, vllm_config, model_config):
        self.vllm_config = vllm_config
        self.model_config = model_config
        self.load_config = vllm_config.load_config
        self.target_device = self._resolve_target_device()

    def build_identity(self):
        return build_source_identity(self.vllm_config, self.model_config)

    def get_worker_rank(self) -> int:
        return _get_vllm_worker_rank(self.vllm_config, self.target_device)

    def get_global_rank(self) -> int:
        return get_global_rank(self.target_device)

    def get_device_id(self) -> int:
        return _get_vllm_device_id(self.target_device)

    def get_target_device(self) -> torch.device:
        return self.target_device

    def is_cuda_alike(self) -> bool:
        from vllm.platforms import current_platform

        return bool(current_platform.is_cuda_alike())

    def discover_tensors(self, result: LoadResult) -> dict[str, torch.Tensor]:
        if result.model is None:
            raise RuntimeError("vLLM tensor discovery requires result.model")
        quantization = str(getattr(self.model_config, "quantization", "") or "")
        self._quantization_provider().capture_from_model(result.model)
        adopt_hidden_tensors(result.model)
        return collect_module_tensors(
            result.model,
            quantization=quantization,
        )

    def prepare_rdma_target(self, result: LoadResult) -> LoadResult:
        if result.model is None:
            raise RuntimeError("vLLM RDMA target preparation requires result.model")

        from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

        dummy_config = copy.copy(self.load_config)
        try:
            dummy_config.load_format = "dummy"
        except AttributeError:
            object.__setattr__(dummy_config, "load_format", "dummy")
        DummyModelLoader(dummy_config).load_weights(result.model, self.model_config)
        return result

    def before_rdma_receive(self, result: LoadResult) -> LoadResult:
        return self._process_weights_after_loading(result)

    def after_rdma_receive(self, result: LoadResult) -> LoadResult:
        return self._refresh_attention_weights_after_rdma(result)

    def prepare_rdma_target_from_manifest(
        self,
        result: LoadResult,
        source_tensors,
    ) -> LoadResult:
        if result.model is None:
            raise RuntimeError("vLLM RDMA target manifest preparation requires result.model")

        replaced = _apply_source_manifest_tensor_structure(
            result.model,
            source_tensors,
            self.target_device,
            provider=self._quantization_provider(),
        )
        if replaced:
            logger.info(
                "[Worker %s] Rebuilt %d RDMA target tensor structures "
                "from source manifest before registration",
                self.get_global_rank(),
                replaced,
            )
        return result

    def apply_weight_iter(
        self,
        result: LoadResult,
        weights_iter: Iterator[tuple[str, torch.Tensor]],
    ) -> LoadResult:
        if result.model is None:
            raise RuntimeError("vLLM weight iterator loading requires result.model")
        provider = self._quantization_provider()
        with provider.capture_during_load(enabled=provider.enabled(
            str(getattr(self.model_config, "quantization", "") or ""),
        )):
            result.model.load_weights(weights_iter)
        return result

    def build_model_streamer_weight_iter(
        self,
        model_uri: str,
        model: torch.nn.Module | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        from vllm.model_executor.model_loader.runai_streamer_loader import (
            RunaiModelStreamerLoader,
        )

        load_config = copy.copy(self.load_config)
        extra_config = dict(getattr(load_config, "model_loader_extra_config", None) or {})
        if self._model_streamer_distributed_enabled():
            extra_config["distributed"] = True
        _set_load_config_extra_config(load_config, extra_config)

        loader = RunaiModelStreamerLoader(load_config)
        revision = getattr(self.model_config, "revision", None)
        return loader._get_weights_iterator(model_uri, revision)

    def load_via_native(self, result: LoadResult) -> LoadResult:
        if result.model is None:
            raise RuntimeError("vLLM native loading requires result.model")

        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

        disk_config = copy.copy(self.load_config)
        try:
            disk_config.load_format = "auto"
        except AttributeError:
            object.__setattr__(disk_config, "load_format", "auto")

        provider = self._quantization_provider()
        with provider.capture_during_load(enabled=provider.enabled(
            str(getattr(self.model_config, "quantization", "") or ""),
        )):
            DefaultModelLoader(disk_config).load_weights(result.model, self.model_config)
        return result

    def after_weight_iter_load(self, result: LoadResult) -> LoadResult:
        return self._process_weights_after_loading(result)

    def after_native_load(self, result: LoadResult) -> LoadResult:
        return self._process_weights_after_loading(result)

    def reinit_for_retry(self, result: LoadResult) -> LoadResult:
        from vllm.model_executor.model_loader.utils import initialize_model

        self.release_failed_load(result)
        self._reset_compilation_state()
        logger.info(
            "[Worker %s] Re-initializing vLLM model after failed strategy",
            self.get_global_rank(),
        )
        with self.target_device:
            model = initialize_model(
                vllm_config=self.vllm_config,
                model_config=self.model_config,
            )
        return LoadResult(value=model, model=model, publishable=result.publishable)

    def release_failed_load(self, result: LoadResult) -> LoadResult:
        old_value = result.value
        old_model = result.model
        result.value = None
        result.model = None
        del old_value
        del old_model
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(self.target_device)
            except Exception as e:
                logger.debug("CUDA synchronize during failed-load cleanup failed: %s", e)
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception as e:
                logger.debug("CUDA IPC cleanup during failed-load cleanup failed: %s", e)
        return LoadResult(value=None, model=None, publishable=result.publishable)

    def _process_weights_after_loading(self, result: LoadResult) -> LoadResult:
        if result.model is None:
            raise RuntimeError("vLLM post-load processing requires result.model")

        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        provider = self._quantization_provider()
        with capture_tensor_attrs(), provider.capture_during_load(
            enabled=provider.enabled(str(getattr(self.model_config, "quantization", "") or "")),
        ):
            process_weights_after_loading(
                result.model, self.model_config, self.target_device,
            )
        provider.capture_from_model(result.model)
        return result

    def _refresh_attention_weights_after_rdma(self, result: LoadResult) -> LoadResult:
        if result.model is None:
            raise RuntimeError("vLLM RDMA post-load refresh requires result.model")

        refreshed = _refresh_vllm_attention_runtime_tensors(
            result.model,
            self.model_config,
            self.target_device,
        )
        if refreshed:
            logger.info(
                "[Worker %s] Refreshed %d vLLM attention runtime tensor owner(s) "
                "after RDMA receive",
                self.get_global_rank(),
                refreshed,
            )
        return result

    def _resolve_target_device(self) -> torch.device:
        load_device = (
            self.vllm_config.device_config.device
            if self.load_config.device is None
            else self.load_config.device
        )
        return torch.device(load_device)

    def _quantization_provider(self):
        return get_quantization_provider(
            str(getattr(self.model_config, "quantization", "") or "")
        )

    def _reset_compilation_state(self) -> None:
        compilation_config = self.vllm_config.compilation_config
        # vLLM registers attention / MLA / Mamba / FusedMoE layers into fields
        # on vllm_config.compilation_config during initialize_model(). Those
        # fields live on the config object, so they survive del model and trip
        # duplicate registration on the next initialize_model().
        compilation_config.static_forward_context.clear()
        compilation_config.static_all_moe_layers.clear()
        compilation_config.enabled_custom_ops.clear()
        compilation_config.disabled_custom_ops.clear()
        compilation_config.traced_files.clear()
        compilation_config.compilation_time = 0.0

    def _model_streamer_distributed_enabled(self) -> bool:
        tp_size = getattr(self.vllm_config.parallel_config, "tensor_parallel_size", 1)
        return (
            tp_size > 1
            and os.environ.get("MX_MS_DISTRIBUTED", "0").lower() in ("1", "true")
        )


def _apply_source_manifest_tensor_structure(
    model: torch.nn.Module,
    source_tensors,
    target_device: torch.device,
    provider=None,
) -> int:
    if provider is None:
        provider = get_quantization_provider("")
    modules = dict(model.named_modules())
    setattr(
        model,
        SOURCE_MANIFEST_TENSOR_NAMES_ATTR,
        {str(desc.name) for desc in source_tensors},
    )
    replaced = 0

    for desc in source_tensors:
        if desc.name.endswith(".__storage"):
            continue
        module_name, _, leaf = desc.name.rpartition(".")
        module = modules.get(module_name) if module_name else model
        if module is None or not leaf:
            continue

        old_tensor = None
        is_parameter = leaf in module._parameters and module._parameters[leaf] is not None
        is_buffer = leaf in module._buffers and module._buffers[leaf] is not None
        if is_parameter:
            old_tensor = module._parameters[leaf].data
        elif is_buffer:
            old_tensor = module._buffers[leaf]
        else:
            continue

        if debug_tensor_enabled(desc.name):
            logger.warning(
                "Source-manifest target prepare: source=(name=%r size=%d "
                "dtype=%r shape=%s stride=%s storage_offset=%d "
                "storage_nbytes=%d layout_kind=%r original_shape=%s "
                "original_dtype=%r original_nbytes=%d tensor_kind=%r "
                "owner_module=%r owner_class=%r quant_method=%r "
                "runtime_role=%r replace_policy=%r) local=(%s)",
                desc.name,
                desc.size,
                desc.dtype,
                list(desc.shape or []),
                list(desc.stride or []),
                desc.storage_offset,
                desc.storage_nbytes,
                desc.layout_kind,
                list(desc.original_shape or []),
                desc.original_dtype,
                desc.original_nbytes,
                getattr(desc, "tensor_kind", ""),
                getattr(desc, "owner_module", ""),
                getattr(desc, "owner_class", ""),
                getattr(desc, "quant_method", ""),
                getattr(desc, "runtime_role", ""),
                getattr(desc, "replace_policy", ""),
                tensor_debug_summary(desc.name, old_tensor),
            )

        old_nbytes = old_tensor.numel() * old_tensor.element_size()
        if old_nbytes == desc.size:
            if debug_tensor_enabled(desc.name):
                logger.warning(
                    "Source-manifest target prepare: keeping local tensor "
                    "for %r because local nbytes matches source size (%d)",
                    desc.name,
                    desc.size,
                )
            continue
        replace_policy = str(getattr(desc, "replace_policy", "") or "structural_replace")
        if replace_policy != "structural_replace":
            raise ManifestMismatchError(
                f"Cannot prepare RDMA target tensor {desc.name!r}: "
                f"source requires replace_policy={replace_policy!r} "
                f"(quant_method={getattr(desc, 'quant_method', '')!r}, "
                f"runtime_role={getattr(desc, 'runtime_role', '')!r}), "
                f"but local current nbytes={old_nbytes} and source size={desc.size}"
            )
        if desc.original_nbytes and old_nbytes != desc.original_nbytes:
            raise ManifestMismatchError(
                f"Cannot prepare RDMA target tensor {desc.name!r}: "
                f"source original_nbytes={desc.original_nbytes}, "
                f"local current nbytes={old_nbytes}"
            )

        new_tensor = _empty_tensor_from_manifest_descriptor(desc, target_device)
        if is_parameter:
            old_requires_grad = bool(module._parameters[leaf].requires_grad)
            requires_grad = old_requires_grad and (
                new_tensor.is_floating_point() or new_tensor.is_complex()
            )
            module._parameters[leaf] = torch.nn.Parameter(
                new_tensor,
                requires_grad=requires_grad,
            )
        else:
            module._buffers[leaf] = new_tensor
        provider.align_target_module_from_source(module, leaf, desc)
        manifest_overrides = getattr(module, MANIFEST_TENSOR_OVERRIDES_ATTR, None)
        if not isinstance(manifest_overrides, dict):
            manifest_overrides = {}
            setattr(module, MANIFEST_TENSOR_OVERRIDES_ATTR, manifest_overrides)
        manifest_overrides[leaf] = (
            str(getattr(desc, "runtime_role", "") or leaf),
            str(getattr(desc, "replace_policy", "") or "structural_replace"),
        )
        provider.after_target_tensor_rebuilt(module, leaf, desc)
        replaced += 1

        if debug_tensor_enabled(desc.name):
            logger.warning(
                "Source-manifest target prepare: rebuilt %r as (%s)",
                desc.name,
                tensor_debug_summary(desc.name, new_tensor),
            )

    return replaced


def _refresh_vllm_attention_runtime_tensors(
    model: torch.nn.Module,
    model_config,
    target_device: torch.device,
) -> int:
    try:
        from vllm.model_executor.model_loader.utils import device_loading_context
    except Exception:
        device_loading_context = None

    attention_class_names = {
        "Attention",
        "MLAAttention",
        "MMEncoderAttention",
    }
    refreshed = 0
    with capture_tensor_attrs():
        for _, module in model.named_modules():
            if type(module).__name__ not in attention_class_names:
                continue
            process_weights = getattr(module, "process_weights_after_loading", None)
            if process_weights is None:
                continue
            context = (
                device_loading_context(module, target_device)
                if device_loading_context is not None
                else nullcontext(module)
            )
            with context:
                process_weights(model_config.dtype)
            refreshed += 1
    return refreshed

def _empty_tensor_from_manifest_descriptor(desc, target_device: torch.device) -> torch.Tensor:
    layout_kind = str(getattr(desc, "layout_kind", ""))
    if layout_kind not in ("", "contiguous"):
        raise ManifestMismatchError(
            f"Cannot rebuild RDMA target tensor {desc.name!r}: "
            f"unsupported layout_kind={layout_kind!r}"
        )
    if getattr(desc, "storage_offset", 0):
        raise ManifestMismatchError(
            f"Cannot rebuild RDMA target tensor {desc.name!r}: "
            f"unsupported storage_offset={desc.storage_offset}"
        )
    shape = [int(dim) for dim in (desc.shape or [])]
    if not shape:
        raise ManifestMismatchError(
            f"Cannot rebuild RDMA target tensor {desc.name!r}: missing shape"
        )
    dtype = _torch_dtype_from_manifest(desc.dtype)
    tensor = torch.empty(shape, dtype=dtype, device=target_device)
    nbytes = tensor.numel() * tensor.element_size()
    if nbytes != desc.size:
        raise ManifestMismatchError(
            f"Cannot rebuild RDMA target tensor {desc.name!r}: "
            f"manifest shape/dtype create {nbytes} bytes, source expects "
            f"{desc.size} bytes"
        )
    return tensor


def _torch_dtype_from_manifest(dtype: str) -> torch.dtype:
    normalized = str(dtype).replace("torch.", "")
    torch_dtype = getattr(torch, normalized, None)
    if not isinstance(torch_dtype, torch.dtype):
        raise ManifestMismatchError(f"Unsupported tensor dtype in manifest: {dtype!r}")
    return torch_dtype


def _set_load_config_extra_config(load_config, extra_config: dict) -> None:
    try:
        load_config.model_loader_extra_config = extra_config
    except AttributeError:
        object.__setattr__(load_config, "model_loader_extra_config", extra_config)


def _get_vllm_worker_rank(
    vllm_config: VllmConfig, target_device: torch.device
) -> int:
    """Return the vLLM model-shard key (torch.distributed world rank).

    Falls back to vllm_config.parallel_config.rank when torch.distributed is
    not initialised and the target device has no index (pre-init / bare-cuda
    test paths), so workers in the same DP still get distinct keys.
    """
    worker_rank = get_global_rank(target_device)
    if worker_rank == 0 and target_device.index is None:
        worker_rank = int(vllm_config.parallel_config.rank)
    logger.debug("vLLM worker rank: %d", worker_rank)
    return worker_rank


def _get_vllm_device_id(target_device: torch.device) -> int:
    """Return the local CUDA ordinal vLLM assigned to this worker."""
    if target_device.index is not None:
        device_id = int(target_device.index)
        logger.debug("Got vLLM device id from target_device: %d", device_id)
        return device_id

    from vllm.platforms import current_platform

    device_id = int(current_platform.current_device())
    logger.debug("Got vLLM device id from current_platform: %d", device_id)
    return device_id


def build_vllm_load_context(vllm_config, model_config) -> LoadContext:
    """Build a LoadContext from vLLM config objects."""

    adapter = VllmAdapter(vllm_config, model_config)
    global_rank = adapter.get_global_rank()
    worker_rank = adapter.get_worker_rank()
    return LoadContext(
        model_config=model_config,
        load_config=vllm_config.load_config,
        target_device=adapter.get_target_device(),
        global_rank=global_rank,
        worker_rank=worker_rank,
        device_id=adapter.get_device_id(),
        identity=adapter.build_identity(),
        mx_client=create_metadata_client(worker_rank=worker_rank),
        worker_id=uuid.uuid4().hex[:8],
        adapter=adapter,
    )
