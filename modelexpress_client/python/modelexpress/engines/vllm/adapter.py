# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM implementation of the ModelExpress engine adapter contract."""

from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
import uuid
from enum import Enum, auto
from typing import TYPE_CHECKING, Iterator

import torch

from ... import envs
from ...adapter import EngineAdapter
from ...accelerators import accelerator_backend_for
from ...load_strategy.context import LoadContext, LoadResult
from ...metadata.client_factory import create_metadata_client
from ...rank_utils import get_global_rank
from ...tensor_utils import adopt_hidden_tensors, capture_tensor_attrs, collect_module_tensors
from .source_identity import build_source_identity

logger = logging.getLogger("modelexpress.engines.vllm.adapter")

_VLLM_PRE_RDMA_FINALIZER_NAMES = (
    # MegaMoE changes the model's tensor layout. It must run before tensor
    # discovery and RDMA registration so the target exposes the same regions
    # that the source published.
    "finalize_mega_moe_weights",
)

_VLLM_POST_RDMA_FINALIZER_NAMES = (
    # DeepSeek V4 derives this tensor from hc_attn_fn. Unlike MegaMoE, it is
    # target-local and is not sent by RDMA, so it must be built only after
    # hc_attn_fn has received its real weight values.
    "finalize_mhc_broadcast_weights",
)

# vLLM keeps accelerator-backed attention scales together with host mirrors
# consumed by backends such as FlashInfer and AITER. Only the accelerator
# tensors are part of the ModelExpress manifest, so an RDMA target must refresh
# the mirrors after the received values overwrite its dummy-loaded tensors.
_VLLM_ATTENTION_SCALE_NAMES: tuple[str, ...] = ("q", "k", "v")

# MTP draft weights live under an "mtp." prefix in the shared checkpoint. The
# draft's embedding and lm_head come from the target, so these are all it needs.
_DRAFT_WEIGHT_PREFIXES: tuple[str, ...] = ("mtp.",)

_SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"

# Registries on compilation_config that vLLM keys by layer name.
_LAYER_REGISTRY_FIELDS: tuple[str, ...] = (
    "static_forward_context",
    "static_all_moe_layers",
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _is_speculative_draft(vllm_config, model_config) -> bool:
    """True for the draft pass of a speculative load.

    vLLM gives the draft ModelConfig runner="draft" and the target "generate".
    Reading runner_type avoids the ngram/custom_class case where the draft
    config aliases the target's.
    """
    if getattr(vllm_config, "speculative_config", None) is None:
        return False
    return getattr(model_config, "runner_type", None) == "draft"


class DraftShardSelection(Enum):
    """Outcome of picking the draft's own shards out of a checkpoint."""

    SELECTED = auto()
    NO_DRAFT_WEIGHTS = auto()
    UNRESOLVED = auto()


def _read_local_safetensors_index(directory: str) -> dict | None:
    """Read model.safetensors.index.json from a local directory."""
    local_index = os.path.join(directory, _SAFETENSORS_INDEX_NAME)
    if not os.path.isfile(local_index):
        return None
    with open(local_index, encoding="utf-8") as handle:
        return json.load(handle)


def _read_safetensors_index(model_uri: str) -> dict | None:
    """Read model.safetensors.index.json from a local dir or object store.

    Returns the parsed index, or None if it cannot be read.
    """
    index = _read_local_safetensors_index(model_uri)
    if index is not None:
        return index

    from runai_model_streamer import pull_files

    with tempfile.TemporaryDirectory() as tmp:
        # runai's allow_pattern is a glob matched against the full object key,
        # so a bare filename never matches; anchor it with a leading wildcard.
        pull_files(model_uri, tmp, allow_pattern=[f"*{_SAFETENSORS_INDEX_NAME}"])
        for root, _dirs, files in os.walk(tmp):
            if _SAFETENSORS_INDEX_NAME in files:
                with open(
                    os.path.join(root, _SAFETENSORS_INDEX_NAME), encoding="utf-8"
                ) as handle:
                    return json.load(handle)
    logger.warning(
        "safetensors index %s not found under %s; draft-shard selection will "
        "fall back to streaming all shards",
        _SAFETENSORS_INDEX_NAME,
        model_uri,
    )
    return None


def _load_safetensors_index(
    model_uri: str,
    hf_weights_files: list[str],
) -> dict | None:
    """Read the checkpoint index, preferring the resolved shards' directory.

    _prepare_weights has already resolved model_uri (which may be an HF repo
    id) to local shard paths, so the index sits next to them. Fall back to
    model_uri for shards the streamer hands back as remote object-store paths.
    """
    seen: set[str] = set()
    for shard in hf_weights_files:
        directory = os.path.dirname(shard)
        if not directory or directory in seen:
            continue
        seen.add(directory)
        index = _read_local_safetensors_index(directory)
        if index is not None:
            return index
    return _read_safetensors_index(model_uri)


def _select_draft_weight_files(
    model_uri: str,
    hf_weights_files: list[str],
) -> tuple[DraftShardSelection, list[str]]:
    """Return the shards holding the draft's own weights.

    Keeps shards whose index tensors carry a draft prefix. Anything other than
    SELECTED leaves the caller streaming every shard, so a checkpoint without a
    draft head (or without a readable index) is never truncated to nothing.
    """
    try:
        index = _load_safetensors_index(model_uri, hf_weights_files)
        if not index:
            return DraftShardSelection.UNRESOLVED, []
        weight_map = index.get("weight_map") or {}
        wanted = {
            fname
            for tname, fname in weight_map.items()
            if tname.startswith(_DRAFT_WEIGHT_PREFIXES)
        }
        if not wanted:
            return DraftShardSelection.NO_DRAFT_WEIGHTS, []
        subset = [f for f in hf_weights_files if os.path.basename(f) in wanted]
        if not subset:
            return DraftShardSelection.UNRESOLVED, []
        return DraftShardSelection.SELECTED, subset
    except Exception as exc:
        logger.warning("Draft weight-file selection failed: %s", exc)
        return DraftShardSelection.UNRESOLVED, []


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
        adopt_hidden_tensors(result.model, self.accelerator_backend)
        return collect_module_tensors(result.model, self.accelerator_backend)

    def prepare_rdma_target(self, result: LoadResult) -> LoadResult:
        if result.model is None:
            raise RuntimeError("vLLM RDMA target preparation requires result.model")

        from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

        dummy_config = copy.copy(self.load_config)
        try:
            dummy_config.load_format = "dummy"
        except AttributeError:
            object.__setattr__(dummy_config, "load_format", "dummy")
        # DummyModelLoader rejects any model_loader_extra_config. The RDMA
        # target only allocates empty tensors, so strip the streamer-oriented
        # extra config (e.g. distributed / memory_limit) the source config may
        # carry. Rebind rather than mutate: the shared dict is still used by the
        # streamer fallback path.
        _set_load_config_extra_config(dummy_config, {})
        DummyModelLoader(dummy_config).load_weights(result.model, self.model_config)
        return result

    def before_rdma_receive(self, result: LoadResult) -> LoadResult:
        # Native vLLM load_weights() runs model-specific finalizers before
        # post-load processing. RDMA targets use the dummy loader, so run
        # those hooks before receiving tensors to expose the same target
        # tensor layout and hidden buffers that the source published.
        result = self._finalize_model_specific_weights(
            result, _VLLM_PRE_RDMA_FINALIZER_NAMES
        )
        return self._process_weights_after_loading(result)

    def after_rdma_receive(self, result: LoadResult) -> LoadResult:
        """Build target-local tensors derived from the received weights."""
        result = self._finalize_model_specific_weights(
            result, _VLLM_POST_RDMA_FINALIZER_NAMES
        )
        return self._refresh_host_quantization_state(result)

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

        if not _is_speculative_draft(self.vllm_config, self.model_config):
            return loader._get_weights_iterator(model_uri, revision)

        # An MTP draft shares the target's checkpoint but needs only its own
        # shards. Stream just those so we do not re-read the whole model from
        # storage for a small head. Fall back to the full set if unrecognized.
        from vllm.model_executor.model_loader.weight_utils import (
            runai_safetensors_weights_iterator,
        )

        hf_weights_files = loader._prepare_weights(model_uri, revision)
        selection, subset = _select_draft_weight_files(model_uri, hf_weights_files)
        if selection is DraftShardSelection.UNRESOLVED:
            logger.warning(
                "[draft] could not resolve draft-only shards from %s for %s; "
                "streaming all %d shards",
                _SAFETENSORS_INDEX_NAME,
                model_uri,
                len(hf_weights_files),
            )
            return loader._get_weights_iterator(model_uri, revision)
        if selection is DraftShardSelection.NO_DRAFT_WEIGHTS:
            logger.info(
                "[draft] %s for %s contains no mtp. tensors; streaming all "
                "%d shards",
                _SAFETENSORS_INDEX_NAME,
                model_uri,
                len(hf_weights_files),
            )
            return loader._get_weights_iterator(model_uri, revision)

        logger.info(
            "[draft] streaming %d of %d safetensors shards for draft weights: %s",
            len(subset),
            len(hf_weights_files),
            [os.path.basename(f) for f in subset],
        )
        return runai_safetensors_weights_iterator(
            subset,
            load_config.use_tqdm_on_load,
            loader._is_distributed,
        )

    def build_instanttensor_weight_iter(
        self,
        model: torch.nn.Module | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        if model is None:
            raise RuntimeError("vLLM InstantTensor loading requires the initialized model")

        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

        # vLLM's DefaultModelLoader selects instanttensor_weights_iterator when
        # load_format == "instanttensor"; get_all_weights() resolves the model's
        # own safetensors (and any secondary sources). The iterator handles the
        # CUDA check and TP process group internally.
        load_config = copy.copy(self.load_config)
        try:
            load_config.load_format = "instanttensor"
        except AttributeError:
            object.__setattr__(load_config, "load_format", "instanttensor")

        loader = DefaultModelLoader(load_config)
        return loader.get_all_weights(self.model_config, model)

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

        stale_value = result.value
        stale_model = result.model
        result.value = None
        result.model = None
        # Unregister before dropping the model: its registrations identify it,
        # and clearing them frees its parameters before the rebuild allocates.
        self._unregister_model_layers(stale_model)
        del stale_value
        del stale_model
        self.accelerator_backend.empty_cache()
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

        with capture_tensor_attrs(self.accelerator_backend):
            process_weights_after_loading(
                result.model,
                self.model_config,
                self.target_device,
            )
        return result

    @torch.no_grad()
    def _refresh_host_quantization_state(self, result: LoadResult) -> LoadResult:
        """Refresh vLLM host mirrors after RDMA overwrites device scales.

        vLLM's attention PWAL copies ``_q/_k/_v_scale`` into Python floats and,
        for some backends, CPU tensors whose launch APIs require host scalars.
        RDMA receives only the registered accelerator tensors, leaving those
        mirrors at values derived from the target's dummy load. Re-running PWAL
        is not a safe repair because the incoming model is already in its
        processed runtime representation; update only the non-transferable
        mirrors that the selected backend exposes.

        Compressed-tensors may use per-head scales. Match vLLM's own conversion
        by using their maximum for the scalar host mirror.
        """
        if result.model is None:
            raise RuntimeError("vLLM RDMA post-load processing requires result.model")

        float_names = tuple(
            f"_{scale_name}_scale_float"
            for scale_name in _VLLM_ATTENTION_SCALE_NAMES
        )
        cpu_names = ("_k_scale_cpu", "_v_scale_cpu")
        device_names = tuple(
            f"_{scale_name}_scale" for scale_name in _VLLM_ATTENTION_SCALE_NAMES
        )
        required_names = device_names + float_names + ("_o_scale_float",)
        flashinfer_cache_names = ("bmm1_scale", "bmm2_scale", "o_sf_scale")

        cache_config = getattr(self.vllm_config, "cache_config", None)
        configured_cache_dtype = getattr(cache_config, "cache_dtype", None)
        fp8_expected = isinstance(configured_cache_dtype, str) and (
            configured_cache_dtype.startswith("fp8")
        )

        # This is a shim over vLLM private attributes, so an upstream scale
        # contract change must fail the RDMA strategy instead of silently
        # serving with stale values. Any failure after mutation is recovered by
        # the outer strategy reinitializing the whole model before fallback.
        refreshed_values = 0
        stale_values = 0
        refreshed_modules = 0
        for module_name, module in result.model.named_modules():
            module_label = module_name or type(module).__name__
            kv_cache_dtype = getattr(module, "kv_cache_dtype", None)
            if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8"):
                fp8_expected = True

            host_names = float_names + cpu_names
            if not any(hasattr(module, name) for name in host_names):
                continue

            missing_names = [
                name for name in required_names if not hasattr(module, name)
            ]
            if missing_names:
                raise RuntimeError(
                    "Incomplete vLLM attention scale contract on "
                    f"{module_label}: missing {', '.join(missing_names)}"
                )

            for float_name in float_names:
                if not isinstance(getattr(module, float_name), float):
                    raise RuntimeError(
                        "Invalid vLLM attention host scalar on "
                        f"{module_label}: {float_name} must be a float"
                    )

            for cpu_name in cpu_names:
                if not hasattr(module, cpu_name):
                    continue
                cpu_mirror = getattr(module, cpu_name)
                if (
                    not isinstance(cpu_mirror, torch.Tensor)
                    or cpu_mirror.numel() != 1
                    or not torch.is_floating_point(cpu_mirror)
                ):
                    raise RuntimeError(
                        "Invalid vLLM attention CPU scale on "
                        f"{module_label}: {cpu_name} must be a floating-point "
                        "singleton tensor"
                    )

            impl = getattr(module, "impl", None)
            initialized_cache_names = []
            if module._o_scale_float is not None:
                initialized_cache_names.append("_o_scale_float")
            initialized_cache_names.extend(
                cache_name
                for cache_name in flashinfer_cache_names
                if hasattr(impl, cache_name) and getattr(impl, cache_name) is not None
            )
            if initialized_cache_names:
                raise RuntimeError(
                    "vLLM attention host-scale refresh requires a cold-load "
                    f"state on {module_label}; already initialized: "
                    f"{', '.join(initialized_cache_names)}"
                )

            for scale_name, tensor_name in zip(
                _VLLM_ATTENTION_SCALE_NAMES, device_names, strict=True
            ):
                scale = getattr(module, tensor_name)
                if (
                    not isinstance(scale, torch.Tensor)
                    or not self.accelerator_backend.is_accel_tensor(scale)
                    or scale.numel() == 0
                    or not torch.is_floating_point(scale)
                ):
                    raise RuntimeError(
                        "Invalid vLLM accelerator attention scale on "
                        f"{module_label}: {tensor_name} must be a nonempty "
                        "floating-point accelerator tensor"
                    )

                scale_float = scale.detach().float()
                if not bool(torch.isfinite(scale_float).all().item()) or not bool(
                    (scale_float > 0).all().item()
                ):
                    raise RuntimeError(
                        "Invalid vLLM accelerator attention scale on "
                        f"{module_label}: {tensor_name} must contain only finite, "
                        "positive values"
                    )
                value = float(scale_float.max().item())
                float_name = f"_{scale_name}_scale_float"
                previous = getattr(module, float_name)
                if previous != value:
                    stale_values += 1
                    logger.debug(
                        "Refreshing vLLM host scale %s.%s: %r -> %r",
                        module_label,
                        float_name,
                        previous,
                        value,
                    )
                setattr(module, float_name, value)

                cpu_mirror = getattr(module, f"_{scale_name}_scale_cpu", None)
                if isinstance(cpu_mirror, torch.Tensor):
                    cpu_mirror.fill_(value)

                refreshed_values += 1
            refreshed_modules += 1

        if fp8_expected and not refreshed_modules:
            raise RuntimeError(
                "FP8 KV cache requires recognizable vLLM q/k/v host scale state, "
                "but no attention module was refreshed"
            )

        if refreshed_values:
            logger.info(
                "Refreshed %d vLLM host attention scales across %d modules "
                "after RDMA receive (%d stale dummy values replaced)",
                refreshed_values,
                refreshed_modules,
                stale_values,
            )
        return result

    def _finalize_model_specific_weights(
        self,
        result: LoadResult,
        finalizer_names: tuple[str, ...],
    ) -> LoadResult:
        """Run selected model finalizers that vLLM normally calls in load_weights()."""

        if result.model is None:
            raise RuntimeError("vLLM RDMA post-load processing requires result.model")

        finalized_prefixes: list[str] = []
        with capture_tensor_attrs(self.accelerator_backend):
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
                for finalizer_name in finalizer_names:
                    finalizer = getattr(module, finalizer_name, None)
                    if not callable(finalizer):
                        continue

                    logger.info(
                        "Running vLLM model finalizer %s on %s",
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

    def _unregister_model_layers(self, stale_model: torch.nn.Module | None) -> None:
        """Remove `stale_model`'s layers from vLLM's layer registries.

        The registries live on compilation_config, so dropping the model leaves
        its entries behind and the rebuild fails vLLM's duplicate-name check.
        Clearing them is wrong: one compilation_config is shared by every model
        built from a VllmConfig, so under MTP they also hold the live target's
        layers. Remove only what this model registered, matched by its own
        modules or the `layer_name` they registered under.

        Args:
            stale_model: Model being discarded, or None to clear outright.
        """
        owned_ids: set[int] = set()
        owned_names: set[str] = set()
        for module in stale_model.modules() if stale_model is not None else ():
            owned_ids.add(id(module))
            layer_name = getattr(module, "layer_name", None)
            if isinstance(layer_name, str):
                owned_names.add(layer_name)

        for attr in _LAYER_REGISTRY_FIELDS:
            registry = getattr(self.vllm_config.compilation_config, attr, None)
            if registry is None:
                continue
            if stale_model is None:
                registry.clear()
            else:
                _drop_owned_entries(attr, registry, owned_ids, owned_names)

    def _model_streamer_distributed_enabled(self) -> bool:
        tp_size = getattr(self.vllm_config.parallel_config, "tensor_parallel_size", 1)
        return (
            tp_size > 1
            and envs.MX_MS_DISTRIBUTED
        )


def _set_load_config_extra_config(load_config, extra_config: dict) -> None:
    try:
        load_config.model_loader_extra_config = extra_config
    except AttributeError:
        object.__setattr__(load_config, "model_loader_extra_config", extra_config)


def _is_same_or_descendant(name: str, prefix: str) -> bool:
    return prefix == "" or name == prefix or name.startswith(f"{prefix}.")


def _drop_owned_entries(
    attr: str,
    registry,
    owned_ids: set[int],
    owned_names: set[str],
) -> None:
    """Remove one compilation registry's entries belonging to a single model."""

    def is_owned(entry) -> bool:
        return id(entry) in owned_ids or (isinstance(entry, str) and entry in owned_names)

    if isinstance(registry, dict):
        for key in [k for k, v in registry.items() if is_owned(k) or is_owned(v)]:
            del registry[key]
    elif isinstance(registry, set):
        registry.difference_update({e for e in registry if is_owned(e)})
    elif isinstance(registry, list):
        registry[:] = [e for e in registry if not is_owned(e)]
    else:
        # A leftover entry only fails the rebuild's duplicate-name check, while
        # clearing blind could unregister a co-owner's layers.
        logger.warning(
            "compilation_config.%s is a %s, which cannot be filtered by owner; "
            "leaving it untouched",
            attr,
            type(registry).__name__,
        )


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
        node_rank=int(getattr(vllm_config.parallel_config, "node_rank", 0)),
        head_addr=getattr(vllm_config.parallel_config, "master_addr", None),
        adapter=adapter,
        accelerator_backend=adapter.accelerator_backend,
    )
