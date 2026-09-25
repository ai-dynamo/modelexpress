# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental Kimi BF16 checkpoint groups; never initialize global reload."""

from __future__ import annotations

import copy
import hashlib
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from modelexpress_rl.inference.checkpoint_selection import (
    DependencyGroup,
    read_selected_tensors,
    select_checkpoint_sources,
)
from modelexpress_rl.inference.receiver import PreparedCheckpoint
from modelexpress_rl.utils import index_checkpoint_tensors

logger = logging.getLogger(__name__)

_EMBEDDING = "language_model.model.embed_tokens.weight"
_PROJECTION = "language_model.model.layers.0.self_attn.kv_b_proj.weight"
_DERIVED = ("W_UV", "W_UK_T")
_MLA_SCALES = frozenset(
    {
        "_q_scale",
        "_k_scale",
        "_v_scale",
        "_prob_scale",
        "q_range",
        "k_range",
        "v_range",
    }
)


class _Unsupported(Exception):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise _Unsupported(reason)


def _layout(tensor: torch.Tensor) -> tuple:
    return (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
    )


@dataclass(frozen=True)
class _Copy:
    owner: nn.Module
    name: str
    destination: torch.Tensor
    source: torch.Tensor
    layout: tuple

    @classmethod
    def stage(cls, owner, name, source):
        destination = getattr(owner, name)
        _require(
            destination.shape == source.shape
            and destination.dtype == source.dtype
            and destination.device == source.device,
            "derived tensor layout changed",
        )
        return cls(
            owner, name, destination, source.detach().clone(), _layout(destination)
        )


@dataclass(frozen=True)
class StagedBf16Checkpoint:
    copies: tuple[_Copy, ...]

    @torch.no_grad()
    def install(self) -> None:
        # Validate the entire write set before the first live copy.
        for item in self.copies:
            if (
                getattr(item.owner, item.name) is not item.destination
                or _layout(item.destination) != item.layout
            ):
                raise RuntimeError(
                    "partial checkpoint destination changed after staging"
                )
        for item in self.copies:
            item.destination.copy_(item.source)


def _contract():
    from vllm.model_executor.layers.attention import MLAAttention
    from vllm.model_executor.layers.attention.attention import set_default_quant_scales
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        get_and_maybe_dequant_weights,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
        VocabParallelEmbedding,
    )
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention
    from vllm.model_executor.models.kimi_k25 import KimiK25ForConditionalGeneration
    from vllm.version import __version__

    _require(__version__ == "0.19.0", "requires the audited vLLM 0.19.0 API")
    # File hashes from upstream v0.19.0; patched/nightly implementations decline.
    for implementation, digest in (
        (
            MultiHeadLatentAttentionWrapper,
            "4919d21219ee82502b16c645fb1d52333be47bd3c04fd60bd323de88c382fc63",
        ),
        (
            DeepseekV2MLAAttention,
            "c0f27b22e0023d42861ed4763e2a23fe305d049c6c45f6bd3ac05299d8e76da2",
        ),
        (
            ColumnParallelLinear.weight_loader,
            "db35d8070e734f2a5d4f705213bcc30a44cab2642c4d29cefd83a14339bef346",
        ),
        (
            VocabParallelEmbedding.weight_loader,
            "9dff6cab7c17798991cc48e854f275a6b596cc637e5694f166c20cdc68d31879",
        ),
        (
            MLAAttention.process_weights_after_loading,
            "a816328aaf1f6e6c5752eff9b27c736b46bb0b08edc5ced933b3acc89ef7ba39",
        ),
        (
            get_and_maybe_dequant_weights,
            "012206124cc41350e2ddee3b975c4eafef72a5731abd38da9ca7220c718ab171",
        ),
        (
            set_default_quant_scales,
            "2e45ac35100a1396bd03f9fe8d7f3e4dd41f3dc0a78b3bc7e5e118b536a0fe3c",
        ),
        (
            KimiK25ForConditionalGeneration,
            "1bbce9c894945a6b95818181b52cd066debb39eb8f47334ecff6d870ead5326b",
        ),
    ):
        try:
            actual = hashlib.sha256(
                Path(inspect.getfile(implementation)).read_bytes()
            ).hexdigest()
        except (OSError, TypeError) as error:
            raise _Unsupported("audited source is unavailable") from error
        _require(actual == digest, "vLLM source differs from the audited tag")
    return (
        KimiK25ForConditionalGeneration,
        VocabParallelEmbedding,
        UnquantizedEmbeddingMethod,
        ColumnParallelLinear,
        UnquantizedLinearMethod,
        MLAAttention,
        DeepseekV2MLAAttention,
        MultiHeadLatentAttentionWrapper,
    )


def _direct_tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in {
            **module._parameters,
            **module._buffers,
            **vars(module),
        }.items()
        if isinstance(value, torch.Tensor)
    }


def _projection_aliases(model, projection, attention, outer_type, wrapper_type):
    path = _PROJECTION.rsplit(".", 2)[0]
    outer = model.get_submodule(path)
    wrapper = getattr(outer, "mla_attn", None)
    _require(
        type(outer) is outer_type
        and type(wrapper) is wrapper_type
        and outer.kv_b_proj is projection
        and wrapper.kv_b_proj is projection
        and wrapper.mla_attn is attention
        and attention.kv_b_proj is projection,
        "unsupported MLA wrapper topology",
    )
    return {
        id(projection): {
            f"{path}.kv_b_proj",
            f"{path}.mla_attn.kv_b_proj",
            f"{path}.mla_attn.mla_attn.kv_b_proj",
        }
    }


def _validate_aliases(
    model: nn.Module,
    copies: list[_Copy],
    audited_paths: dict[int, set[str]] | None = None,
) -> None:
    allowed = {(id(item.owner), item.name) for item in copies}
    owner_paths: dict[int, set[str]] = {id(item.owner): set() for item in copies}
    for path, module in model.named_modules(remove_duplicate=False):
        if id(module) in owner_paths:
            owner_paths[id(module)].add(path)
    for owner, paths in owner_paths.items():
        expected = (audited_paths or {}).get(owner)
        _require(
            paths == expected if expected is not None else len(paths) == 1,
            "unaccounted aliased storage module",
        )
    storages = {
        (item.destination.device, item.destination.untyped_storage().data_ptr())
        for item in copies
    }
    for module in model.modules():
        for name, tensor in _direct_tensors(module).items():
            if tensor.device.type == "meta":
                raise _Unsupported("model contains unmaterialized runtime tensors")
            if (tensor.device, tensor.untyped_storage().data_ptr()) in storages:
                _require(
                    (id(module), name) in allowed, "unaccounted tied or aliased storage"
                )


def _stage_weight(layer, source, loader) -> nn.Module:
    weight = layer.weight
    _require(
        weight.dtype == source.dtype == torch.bfloat16
        and weight.ndim == source.ndim == 2
        and weight.is_contiguous()
        and source.shape[1] == weight.shape[1]
        and getattr(weight, "output_dim", None) == 0
        and getattr(weight, "packed_dim", None) is None,
        "requires ordinary row-sharded BF16 weights",
    )
    _require(
        {name for name, value in layer._parameters.items() if value is not None}
        == {"weight"},
        "extra layer parameters",
    )
    _require(not dict(layer.named_buffers(recurse=False)), "extra layer buffers")
    shadow = copy.copy(layer)
    shadow._parameters = {}
    shadow._buffers = {}
    shadow._modules = {}
    shadow.weight = nn.Parameter(torch.empty_like(weight), requires_grad=False)
    shadow.weight.output_dim = 0
    loader(shadow, shadow.weight, source)
    return shadow


def _stage_mla(attention, projection, mla_type) -> list[_Copy]:
    _require(
        type(attention) is mla_type
        and not attention.is_aiter_triton_fp4_bmm_enabled
        and not attention.is_aiter_triton_fp8_bmm_enabled
        and not getattr(attention, "calculate_kv_scales", False),
        "unsupported MLA backend or dynamic cache scales",
    )
    original = _direct_tensors(attention)
    _require(all(name in original for name in _DERIVED), "missing MLA derived tensors")

    def isolate(name, value):
        if name in _MLA_SCALES:
            _require(value.numel() == 1, "non-scalar MLA cache scale")
            return value.clone()
        # The audited finalizer cannot read cache/workspace contents. Meta
        # stand-ins isolate them without copying potentially huge KV caches.
        return torch.empty_strided(
            value.shape, value.stride(), dtype=value.dtype, device="meta"
        )

    shadow = copy.copy(attention)
    shadow._parameters = {
        name: nn.Parameter(isolate(name, value), requires_grad=False)
        if value is not None
        else None
        for name, value in attention._parameters.items()
    }
    shadow._buffers = {
        name: isolate(name, value) if value is not None else None
        for name, value in attention._buffers.items()
    }
    shadow._modules = dict(attention._modules)
    for name, value in vars(attention).items():
        if isinstance(value, torch.Tensor):
            setattr(shadow, name, isolate(name, value))
    isolated = _direct_tensors(shadow)
    versions = {name: value._version for name, value in isolated.items()}
    shadow.quant_config = copy.deepcopy(attention.quant_config)
    shadow.kv_b_proj = projection
    mla_type.process_weights_after_loading(shadow, torch.bfloat16)
    refreshed = _direct_tensors(shadow)
    _require(
        set(refreshed) == set(original), "MLA finalizer changed its tensor contract"
    )
    for name, value in original.items():
        if name in _MLA_SCALES:
            updated = refreshed[name]
            _require(
                updated.shape == value.shape
                and updated.dtype == value.dtype
                and torch.equal(value.detach().cpu(), updated.detach().cpu()),
                "MLA finalizer changed cache state",
            )
        elif name not in _DERIVED:
            _require(
                refreshed[name] is isolated[name]
                and refreshed[name]._version == versions[name],
                "MLA finalizer accessed unrelated runtime state",
            )

    def host_values(module):
        return {
            name: value
            for name, value in vars(module).items()
            if isinstance(value, (bool, int, float, str)) or value is None
        }

    _require(
        host_values(shadow) == host_values(attention),
        "MLA finalizer changed host state",
    )
    return [_Copy.stage(attention, name, refreshed[name]) for name in _DERIVED]


@torch.inference_mode(False)
@torch.no_grad()
def prepare_bf16_checkpoint(
    model: nn.Module,
    vllm_config,
    prepared: PreparedCheckpoint,
    *,
    serving_version: str,
) -> StagedBf16Checkpoint | None:
    """Stage the two historical Kimi BF16 groups, or decline before mutation.

    No quantized parameters, expert weights, tied destinations, graph mode,
    offloading, PP/EP, LoRA, speculative or secondary weights are supported.
    GPU correctness and latency remain unverified; use only the collective API.
    """
    try:
        (
            kimi,
            embedding_type,
            embedding_method,
            linear,
            linear_method,
            mla,
            outer_type,
            wrapper_type,
        ) = _contract()
        config = vllm_config.model_config
        parallel = vllm_config.parallel_config
        _require(
            type(model) is kimi
            and config.enforce_eager
            and config.dtype == torch.bfloat16
            and parallel.pipeline_parallel_size == 1
            and not parallel.enable_expert_parallel
            and getattr(parallel, "data_parallel_size", 1) == 1
            and getattr(parallel, "decode_context_parallel_size", 1) == 1
            and getattr(parallel, "prefill_context_parallel_size", 1) == 1
            and getattr(vllm_config.cache_config, "cpu_offload_gb", 0) == 0
            and vllm_config.cache_config.cache_dtype in ("auto", "bfloat16")
            and getattr(vllm_config, "lora_config", None) is None
            and getattr(vllm_config, "speculative_config", None) is None
            and not getattr(model, "secondary_weights", ()),
            "unsupported runtime configuration",
        )
        _, locations, _ = index_checkpoint_tensors(prepared.path)
        selection = select_checkpoint_sources(
            prepared.changes,
            serving_version=serving_version,
            target_version=prepared.target_version,
            checkpoint_names=frozenset(locations),
            groups=(
                DependencyGroup("embedding", frozenset({_EMBEDDING})),
                DependencyGroup(
                    "projection", frozenset({_PROJECTION}), frozenset(_DERIVED)
                ),
            ),
        )
        _require(
            selection.fallback_reason is None,
            selection.fallback_reason or "unsupported",
        )
        sources = read_selected_tensors(
            selection,
            checkpoint=prepared.path,
            weight_map={name: location[0].name for name, location in locations.items()},
        )
        copies = []
        audited_paths = {}
        for name, source in sources.items():
            layer = model.get_submodule(name.rsplit(".", 1)[0])
            _require(layer.weight.device.type == "cuda", "CPU/offloaded weights")
            if name == _EMBEDDING:
                _require(
                    type(layer) is embedding_type
                    and type(layer.quant_method) is embedding_method
                    and layer.num_embeddings == layer.org_vocab_size
                    and source.shape[0] == layer.org_vocab_size,
                    "unsupported embedding layout",
                )
                shadow = _stage_weight(layer, source, embedding_type.weight_loader)
            else:
                _require(
                    type(layer) is linear
                    and type(layer.quant_method) is linear_method
                    and source.shape[0] == layer.weight.shape[0] * layer.tp_size,
                    "unsupported projection layout",
                )
                shadow = _stage_weight(layer, source, linear.weight_loader)
                consumers = [
                    m
                    for m in model.modules()
                    if getattr(m, "kv_b_proj", None) is layer and type(m) is mla
                ]
                _require(len(consumers) == 1, "requires exactly one MLA consumer")
                audited_paths.update(
                    _projection_aliases(
                        model, layer, consumers[0], outer_type, wrapper_type
                    )
                )
                copies.extend(_stage_mla(consumers[0], shadow, mla))
            copies.append(_Copy.stage(layer, "weight", shadow.weight))

        _validate_aliases(model, copies, audited_paths)
        return StagedBf16Checkpoint(tuple(copies))
    except (_Unsupported, ImportError, AttributeError) as error:
        logger.info("Partial checkpoint unsupported; using full reload: %s", error)
        return None
