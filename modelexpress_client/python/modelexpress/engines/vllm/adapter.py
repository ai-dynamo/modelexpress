# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM implementation of the ModelExpress engine adapter contract."""

from __future__ import annotations

import copy
import logging
import os
import uuid
from typing import TYPE_CHECKING, Iterator

import torch

from ...adapter import EngineAdapter
from ...accelerator_backend import accelerator_backend_for
from ...load_strategy.context import LoadContext, LoadResult
from ...metadata.client_factory import create_metadata_client
from ...metadata.publish import build_source_identity
from ...rank_utils import get_global_rank
from ...tensor_utils import adopt_hidden_tensors, capture_tensor_attrs, collect_module_tensors

logger = logging.getLogger("modelexpress.engines.vllm.adapter")

_VLLM_POST_LOAD_FINALIZER_NAMES = (
    # DeepSeek V4 finalizes MegaMoE expert layouts from load_weights().
    # The RDMA target path uses vLLM's dummy loader, which bypasses the
    # model load_weights() method, so mirror the model-level hook here.
    "finalize_mega_moe_weights",
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class VllmAdapter(EngineAdapter):
    """Adapter that maps strategy hooks onto vLLM's native loader APIs."""

    def __init__(self, vllm_config, model_config):
        self.vllm_config = vllm_config
        self.model_config = model_config
        self.load_config = vllm_config.load_config
        self.target_device = self._resolve_target_device()
        self.accelerator_backend = accelerator_backend_for(self.target_device)

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
        adopt_hidden_tensors(result.model)
        return collect_module_tensors(result.model)

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
        # Native vLLM load_weights() runs model-specific finalizers before
        # post-load processing. RDMA targets use the dummy loader, so run
        # those hooks before receiving tensors to expose the same target
        # tensor layout and hidden buffers that the source published.
        result = self._finalize_model_specific_weights(result)
        return self._process_weights_after_loading(result)

    def apply_weight_iter(
        self,
        result: LoadResult,
        weights_iter: Iterator[tuple[str, torch.Tensor]],
    ) -> LoadResult:
        if result.model is None:
            raise RuntimeError("vLLM weight iterator loading requires result.model")
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

        DefaultModelLoader(disk_config).load_weights(result.model, self.model_config)
        return result

    def after_weight_iter_load(self, result: LoadResult) -> LoadResult:
        return self._process_weights_after_loading(result)

    def after_native_load(self, result: LoadResult) -> LoadResult:
        return self._process_weights_after_loading(result)

    def reinit_for_retry(self, result: LoadResult) -> LoadResult:
        from vllm.model_executor.model_loader.utils import initialize_model

        old_value = result.value
        result.value = None
        result.model = None
        del old_value
        self.accelerator_backend.empty_cache()
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

    def _process_weights_after_loading(
        self,
        result: LoadResult,
    ) -> LoadResult:
        if result.model is None:
            raise RuntimeError("vLLM post-load processing requires result.model")

        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        with capture_tensor_attrs():
            process_weights_after_loading(
                result.model,
                self.model_config,
                self.target_device,
            )
        return result

    def _finalize_model_specific_weights(
        self,
        result: LoadResult,
    ) -> LoadResult:
        """Run model finalizers that vLLM normally calls in load_weights()."""

        if result.model is None:
            raise RuntimeError("vLLM RDMA post-load processing requires result.model")

        finalized_prefixes: list[str] = []
        with capture_tensor_attrs():
            for name, module in result.model.named_modules():
                # Some vLLM finalizers are model-level hooks that recursively
                # transform child layers. If a parent ran one, do not call another
                # matching hook on its descendants and risk duplicate repacking.
                if any(
                    _is_same_or_descendant(name, prefix)
                    for prefix in finalized_prefixes
                ):
                    continue

                module_finalized = False
                for finalizer_name in _VLLM_POST_LOAD_FINALIZER_NAMES:
                    finalizer = getattr(module, finalizer_name, None)
                    if not callable(finalizer):
                        continue

                    logger.info(
                        "Running vLLM model-specific post-load finalizer %s on %s",
                        finalizer_name,
                        name or type(module).__name__,
                    )
                    finalizer()
                    module_finalized = True

                if module_finalized:
                    finalized_prefixes.append(name)
        return result

    def _resolve_target_device(self) -> torch.device:
        load_device = (
            self.vllm_config.device_config.device
            if self.load_config.device is None
            else self.load_config.device
        )
        return torch.device(load_device)

    def _reset_compilation_state(self) -> None:
        compilation_config = self.vllm_config.compilation_config
        # vLLM registers each attention / MLA / Mamba / FusedMoE layer into
        # fields on vllm_config.compilation_config during initialize_model().
        # Those fields live on the config object, not the model, so they survive
        # del model and trip duplicate registration on the next initialize_model().
        # Clear them so re-init starts from a clean slate. Audited against vLLM
        # 0.17.1; other versions may add init=False fields that need similar
        # treatment.
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


def _set_load_config_extra_config(load_config, extra_config: dict) -> None:
    try:
        load_config.model_loader_extra_config = extra_config
    except AttributeError:
        object.__setattr__(load_config, "model_loader_extra_config", extra_config)


def _is_same_or_descendant(name: str, prefix: str) -> bool:
    return prefix == "" or name == prefix or name.startswith(f"{prefix}.")


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
        accelerator_backend=adapter.accelerator_backend,
    )
