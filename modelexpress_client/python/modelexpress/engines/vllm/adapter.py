# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM implementation of the ModelExpress engine adapter contract."""

from __future__ import annotations

import copy
import gc
import inspect
import json
import logging
import os
import tempfile
import uuid
from typing import TYPE_CHECKING, Callable, Iterator

import torch

from ... import envs
from ...adapter import EngineAdapter
from ...accelerators import accelerator_backend_for
from ...load_strategy.context import LoadContext, LoadResult
from ...metadata.client_factory import create_metadata_client
from ...rank_utils import get_global_rank
from ...tensor_utils import adopt_hidden_tensors, capture_tensor_attrs, collect_module_tensors
from .host_quantization import refresh_host_quantization_state
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


def _target_shared_draft_prefixes(model, vllm_config) -> tuple[str, ...]:
    """Tensor-name prefixes vLLM swaps for the target's after an MTP draft loads.

    Mirrors the MTP branch of vLLM's proposer ``_maybe_share_embeddings`` /
    ``_maybe_share_lm_head``: a draft without ``has_own_embed_tokens`` always
    takes the target's embedding (single pipeline stage), and the base MTP
    proposer shares the target's ``lm_head`` and each ``shared_head.head``.
    Step3p5MTP is different: its proposer keeps each layer's own head (older
    vLLM releases do not mark this with ``has_own_lm_head``). Draft copies
    replaced by the base proposer are discarded, so they are neither served
    nor received over P2P; holding them for NIXL
    would pin memory vLLM frees to size the KV cache. EAGLE drafts carry the
    ``has_own_*`` attributes and are compared by content, so nothing is
    dropped for them.
    """
    prefixes: list[str] = []
    parallel = getattr(vllm_config, "parallel_config", None)
    if (
        not hasattr(model, "has_own_embed_tokens")
        and getattr(parallel, "pipeline_parallel_size", 1) == 1
    ):
        prefixes.append("model.embed_tokens.")
    if not hasattr(model, "has_own_lm_head") and type(model).__name__ != "Step3p5MTP":
        prefixes.append("lm_head.")
        for name, module in model.named_modules():
            if name.endswith("shared_head") and hasattr(module, "head"):
                prefixes.append(f"{name}.head.")
    return tuple(prefixes)


def _read_local_json(directory: str, name: str) -> dict | None:
    """Read a JSON file by name from a local directory."""
    path = os.path.join(directory, name)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_json(model_uri: str, name: str) -> dict | None:
    """Read a JSON file by name from a local dir or object store.

    Returns the parsed JSON, or None if it cannot be read.
    """
    local = _read_local_json(model_uri, name)
    if local is not None:
        return local

    from runai_model_streamer import pull_files

    with tempfile.TemporaryDirectory() as tmp:
        # runai's allow_pattern is a glob matched against the full object key,
        # so a bare filename never matches; anchor it with a leading wildcard.
        pull_files(model_uri, tmp, allow_pattern=[f"*{name}"])
        for root, _dirs, files in os.walk(tmp):
            if name in files:
                with open(os.path.join(root, name), encoding="utf-8") as handle:
                    return json.load(handle)
    return None


def _read_local_json_near_shards(
    hf_weights_files: list[str], name: str
) -> dict | None:
    """Read a JSON file from the first resolved shard directory that has it.

    _prepare_weights has already resolved model_uri (which may be an HF repo id)
    to local shard paths, so the index sits next to them.
    """
    seen: set[str] = set()
    for shard in hf_weights_files:
        directory = os.path.dirname(shard)
        if not directory or directory in seen:
            continue
        seen.add(directory)
        found = _read_local_json(directory, name)
        if found is not None:
            return found
    return None


def _read_safetensors_index(model_uri: str) -> dict | None:
    """Read model.safetensors.index.json from a local dir or object store."""
    index = _read_json(model_uri, _SAFETENSORS_INDEX_NAME)
    if index is None:
        logger.warning(
            "safetensors index %s not found under %s; shard selection "
            "will fall back to streaming all shards",
            _SAFETENSORS_INDEX_NAME,
            model_uri,
        )
    return index


def _load_safetensors_index(
    model_uri: str,
    hf_weights_files: list[str],
) -> dict | None:
    """Read the checkpoint index, preferring the resolved shards' directory,
    then model_uri for shards the streamer hands back as object-store paths."""
    local = _read_local_json_near_shards(hf_weights_files, _SAFETENSORS_INDEX_NAME)
    return local if local is not None else _read_safetensors_index(model_uri)


def _select_remote_weight_files(
    model_uri: str,
    files: list[str],
    is_unused_weight: Callable[[str], bool],
) -> list[str]:
    """Use optional index metadata to skip wholly unused object-store shards.

    A missing, malformed, or incomplete index preserves the original file list.
    Unknown files are retained. The model-owned predicate is evaluated outside
    the metadata-error boundary so a broken model rule is never hidden.
    """
    try:
        index = _load_safetensors_index(model_uri, files)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("missing weight_map")
        names = [os.path.basename(path) for path in files]
        if len(set(names)) != len(names):
            raise ValueError("ambiguous shard basenames")
        for tensor_name, shard in weight_map.items():
            if (
                not isinstance(tensor_name, str)
                or not isinstance(shard, str)
                or not shard
                or shard != os.path.basename(shard)
                or shard in (".", "..")
            ):
                raise ValueError("invalid index entry")
        indexed = set(weight_map.values())
        if not indexed.issubset(names):
            raise ValueError("index references unresolved shards")
    except Exception as exc:
        logger.warning("Cannot select RunAI shards from checkpoint index: %s", exc)
        return files

    needed = {
        shard for name, shard in weight_map.items() if not is_unused_weight(name)
    }
    selected = [
        path
        for path in files
        if os.path.basename(path) not in indexed or os.path.basename(path) in needed
    ]
    return selected or files


def _converge_streamer_files(
    files: list[str], selected: list[str], distributed: bool
) -> list[str]:
    """Distributed RunAI ranks must stream the same union of selected files."""
    if not distributed or not torch.distributed.is_initialized():
        return selected
    from vllm.distributed import get_world_group

    group = get_world_group()
    selections = [None] * group.world_size
    torch.distributed.all_gather_object(
        selections, (files, selected), group=group.cpu_group
    )
    if any(original != files for original, _ in selections):
        return files
    needed = {path for _, subset in selections for path in subset}
    return [path for path in files if path in needed]


class VllmAdapter(EngineAdapter):
    """Adapter that maps strategy hooks onto vLLM's native loader APIs."""

    def __init__(self, vllm_config, model_config):
        self.vllm_config = vllm_config
        self.model_config = model_config
        # Resolve the MX name separately from vLLM's model-loading configuration.
        self._identity_model_config = copy.copy(model_config)
        self._identity_model_config.model = (
            envs.MX_MODEL_NAME_OVERRIDE or model_config.model
        )
        self.load_config = vllm_config.load_config
        self.target_device = self._resolve_target_device()
        self.accelerator_backend = accelerator_backend_for(self.target_device)

    def build_identity(self):
        return build_source_identity(self.vllm_config, self._identity_model_config)

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

    def all_gather_state(self, state) -> tuple[object, ...]:
        """Gather rank-local state through vLLM's CPU process group."""
        from vllm.distributed import get_world_group

        group = get_world_group()
        states = [None] * group.world_size
        torch.distributed.all_gather_object(
            states,
            state,
            group=group.cpu_group,
        )
        return tuple(states)

    def broadcast_state(self, state):
        """Broadcast rank-zero state through vLLM's world group."""
        from vllm.distributed import get_world_group

        return get_world_group().broadcast_object(state, src=0)

    def discover_tensors(self, result: LoadResult) -> dict[str, torch.Tensor]:
        if result.model is None:
            raise RuntimeError("vLLM tensor discovery requires result.model")
        adopt_hidden_tensors(result.model, self.accelerator_backend)
        tensors = collect_module_tensors(result.model, self.accelerator_backend)
        if _is_speculative_draft(self.vllm_config, self.model_config):
            prefixes = _target_shared_draft_prefixes(result.model, self.vllm_config)
            if prefixes:
                tensors = {
                    name: tensor
                    for name, tensor in tensors.items()
                    if not name.startswith(prefixes)
                }
        return tensors

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

        is_unused_weight = getattr(model, "is_unused_checkpoint_weight", None)
        if not callable(is_unused_weight):
            # Old vLLM releases and models without an owned rule keep the
            # original full-stream behavior. MX does not guess MTP names.
            return loader._get_weights_iterator(model_uri, revision)

        # Once vLLM's RunAI loader accepts the model-owned rule, delegate the
        # entire selection path (including distributed agreement) to it.
        try:
            supports_rule = "is_unused_weight" in inspect.signature(
                loader._get_weights_iterator
            ).parameters
        except (TypeError, ValueError):
            supports_rule = False
        if supports_rule:
            return loader._get_weights_iterator(
                model_uri, revision, is_unused_weight
            )

        from vllm.model_executor.model_loader.weight_utils import (
            runai_safetensors_weights_iterator,
        )
        from vllm.transformers_utils.runai_utils import is_runai_obj_uri

        try:
            from vllm.model_executor.model_loader.weight_utils import (
                filter_safetensors_files_by_weight_name,
            )
        except ImportError:
            # The model hook can exist in an engine build whose RunAI loader
            # predates the shared safetensors file filter.
            return loader._get_weights_iterator(model_uri, revision)

        files = loader._prepare_weights(model_uri, revision)
        if is_runai_obj_uri(model_uri):
            selected = _select_remote_weight_files(
                model_uri, files, is_unused_weight
            )
        else:
            selected = filter_safetensors_files_by_weight_name(
                files, is_unused_weight
            )
        selected = _converge_streamer_files(files, selected, loader._is_distributed)
        logger.info(
            "Streaming %d/%d safetensors shards selected by the vLLM model",
            len(selected),
            len(files),
        )
        return runai_safetensors_weights_iterator(
            selected, load_config.use_tqdm_on_load, loader._is_distributed
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

        stale_model = result.model
        if stale_model is None:
            raise RuntimeError("vLLM retry reinitialization requires result.model")
        result.value = None
        result.model = None
        self._unregister_model_layers(stale_model)
        # Native tensor aliases can retain child modules through weight-loader
        # callbacks beyond Python GC. Preserve shared parameterless caches,
        # such as rotary embeddings, while releasing the discarded weights.
        for module in list(stale_model.modules()):
            if module is stale_model or module._parameters:
                module.__dict__.clear()
        del module, stale_model
        gc.collect()
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

    def _refresh_host_quantization_state(self, result: LoadResult) -> LoadResult:
        if result.model is None:
            raise RuntimeError("vLLM RDMA post-load processing requires result.model")
        refresh_host_quantization_state(
            result.model, self.vllm_config, self.accelerator_backend
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

    from vllm.distributed import get_world_group

    adapter = VllmAdapter(vllm_config, model_config)
    global_rank = adapter.get_global_rank()
    worker_rank = adapter.get_worker_rank()
    return LoadContext(
        model_config=model_config,
        load_config=vllm_config.load_config,
        target_device=adapter.get_target_device(),
        global_rank=global_rank,
        worker_rank=worker_rank,
        local_rank=int(get_world_group().local_rank),
        device_id=adapter.get_device_id(),
        identity=adapter.build_identity(),
        mx_client=create_metadata_client(worker_rank=worker_rank),
        worker_id=uuid.uuid4().hex[:8],
        node_rank=int(getattr(vllm_config.parallel_config, "node_rank", 0)),
        head_addr=getattr(vllm_config.parallel_config, "master_addr", None),
        adapter=adapter,
        accelerator_backend=adapter.accelerator_backend,
    )
