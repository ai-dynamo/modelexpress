# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental, pinned GLM-5 eager adapter for the PrimeRL pause lifecycle.

The caller must hold the engine paused throughout preparation, installation,
verification and release. This is a mutating DIRECT operation, not activation
of an independently staged complete version. No arbitrary loader/PWAL callback
receives a staging tensor, so receive storage cannot escape through that path.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import torch

from ...plan import PreparedDirectGroupTensors, PreparedStreamingTensors
from .direct_copy import (
    _DirectDestination,
    _dispatch_guard,
    _drain,
    _install_copy_batches,
    _require,
)
from .direct_glm_profile import MODEL_CLASS, MODULE_COUNTS, SOURCE_HASHES, VLLM_VERSION
from .direct_mla import _prepare_mla_refresh


@lru_cache(maxsize=1)
def _profile_classes():
    """Verify reviewed source files once per immutable worker process."""
    for relative, expected in SOURCE_HASHES.items():
        module_name = (
            relative.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
        )
        module = importlib.import_module(module_name)
        path = Path(inspect.getsourcefile(module))
        _require(
            hashlib.sha256(path.read_bytes()).hexdigest() == expected,
            f"GLM source differs: {relative}",
        )
    result = {}
    for name in MODULE_COUNTS:
        module_name, symbol = name.rsplit(".", 1)
        result[getattr(importlib.import_module(module_name), symbol)] = name
    from vllm.model_executor.models import ModelRegistry

    selected, _ = ModelRegistry.resolve_model_cls(
        "GlmMoeDsaForCausalLM", SimpleNamespace(model_impl="vllm")
    )
    _require(
        result.get(selected) == MODEL_CLASS,
        "GLM registry selects an unreviewed model implementation",
    )
    return result


def _class(value, name):
    module_name, symbol = name.rsplit(".", 1)
    _require(
        type(value) is getattr(importlib.import_module(module_name), symbol),
        f"unsupported GLM helper: {name}",
    )


def _check_vllm030_configuration(config):
    _require(config.attention_config.hisparse_config is None, "HiSparse KV cache")
    _require(
        not config.parallel_config.enable_elastic_ep
        and config.parallel_config.elastic_ep_max_dp_size == 1,
        "elastic expert parallelism",
    )
    _require(
        config.kernel_config.sparse_indexer_topk_backend == "auto",
        "alternate sparse indexer backend",
    )


def _check_vllm030_module(module):
    name = type(module).__name__
    if name == "DeepseekV32Attention":
        _require(module.hisparse_cache is None, "HiSparse attention state")
        group = module.impl.index_group
        _class(group, "vllm.v1.attention.backends.mla.index_group.SparseMLAIndexGroup")
        _require(
            module.impl.index_group_index == 0
            and group.has_indexer is True
            and group.num_layers == 1
            and group.logical_topk_indices is module.topk_indices_buffer
            and group.logical_topk_indices is module.indexer.topk_indices_buffer
            and group.logical_topk_indices
            is module.indexer.indexer_op.topk_indices_buffer
            and group.logical_topk_indices is module.impl.topk_indices_buffer,
            "alternate sparse index group",
        )
    elif name == "SparseAttnIndexer":
        _require(
            module.candidate_blocks is None
            and module.candidate_block_size == 0
            and module.candidate_write is False
            and module.compress_ratio == 1
            and module.use_fp4_cache is False
            and module.skip_k_cache_insert is False
            and module.topk_backend == "auto",
            "alternate sparse indexer representation",
        )
    elif name in ("VocabParallelEmbedding", "ParallelLMHead"):
        _require(module.parallel_group is None, "alternate embedding parallel group")
    elif name == "UnquantizedFusedMoEMethod":
        _require(module.moe.elastic_ep_max_dp_size == 1, "elastic MoE configuration")


def _check_moe(method):
    prefix = "vllm.model_executor.layers.fused_moe."
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoeBackend,
    )

    _class(method, prefix + "unquantized_fused_moe_method.UnquantizedFusedMoEMethod")
    _require(method.unquantized_backend is UnquantizedMoeBackend.TRITON, "MoE backend")
    config = method.moe
    _require(
        not config.has_bias
        and not config.is_lora_enabled
        and not config.moe_parallel_config.enable_eplb,
        "MoE bias/LoRA/EPLB",
    )
    quant = method.moe_quant_config
    _class(quant, prefix + "config.FusedMoEQuantConfig")
    for name in ("_a1", "_a2", "_w1", "_w2"):
        desc = getattr(quant, name)
        _class(desc, prefix + "config.FusedMoEQuantDesc")
        _require(
            all(
                getattr(desc, key) is None
                for key in ("dtype", "scale", "zp", "bias", "alpha_or_gscale")
            ),
            "MoE quantized or additional weight state",
        )
    _class(method.moe_kernel, prefix + "modular_kernel.FusedMoEKernel")
    impl = method.moe_kernel.impl
    _class(impl, prefix + "modular_kernel.FusedMoEKernelModularImpl")
    experts = impl.fused_experts
    _class(experts, prefix + "experts.triton_moe.TritonExperts")
    _require(
        experts.quant_config is quant
        and experts._lora_context is None
        and experts.quantization_emulation is False,
        "MoE expert configuration",
    )
    _class(
        impl.prepare_finalize,
        prefix + "prepare_finalize.no_dp_ep.MoEPrepareAndFinalizeNoDPEPModular",
    )
    # Routing replay belongs to monolithic experts. The pinned modular classes
    # do not declare it; even an inactive added field means unreviewed state.
    for owner in (method.moe_kernel, impl, experts, impl.prepare_finalize):
        _require(
            not hasattr(owner, "routing_replay_capture_fn")
            and not hasattr(owner, "_routing_replay_buffer"),
            "unexpected modular MoE routing replay state",
        )


def _parameter_geometry(tensor):
    from vllm.model_executor.parameter import ModelWeightParameter

    _require(
        type(tensor) in (torch.nn.Parameter, ModelWeightParameter), "parameter subclass"
    )
    _require(
        tensor.device.type == "cuda"
        and tensor.is_contiguous()
        and tensor.dtype in (torch.bfloat16, torch.float32)
        and tensor.numel() > 0,
        "GLM parameter layout/device",
    )
    storage = tensor.untyped_storage()
    nbytes = tensor.numel() * tensor.element_size()
    _require(
        tensor.storage_offset() == 0
        and storage.nbytes() == nbytes
        and tensor.data_ptr() == storage.data_ptr(),
        "GLM parameter storage coverage",
    )
    return (
        id(tensor),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        0,
        storage.data_ptr(),
        storage.nbytes(),
        tensor.data_ptr(),
        nbytes,
    )


def _inspect(model, config):
    _dispatch_guard()
    classes = _profile_classes()
    _require(
        classes.get(type(model)) == MODEL_CLASS,
        "not the pinned GLM-5 model",
    )
    _require(
        config.model_config.hf_config.model_type == "glm_moe_dsa"
        and config.model_config.dtype is torch.bfloat16,
        "not the pinned BF16 GLM architecture",
    )
    _require(
        config.quant_config is None and config.lora_config is None, "quantization/LoRA"
    )
    _require(config.speculative_config is None, "speculative decoding")
    _require(config.model_config.enforce_eager, "GLM DIRECT requires eager execution")
    parallel = config.parallel_config
    for name, expected in (
        ("tensor_parallel_size", 32),
        ("pipeline_parallel_size", 1),
        ("data_parallel_size", 1),
        ("decode_context_parallel_size", 1),
        ("prefill_context_parallel_size", 1),
    ):
        _require(getattr(parallel, name) == expected, f"unsupported {name}")
    _require(not parallel.enable_eplb and not parallel.enable_dbo, "EPLB/DBO")
    _require(not parallel.use_sequence_parallel_moe, "sequence-parallel MoE")
    _require(parallel.enable_expert_parallel, "GLM DIRECT requires expert parallelism")
    if VLLM_VERSION == "0.30.0":
        _check_vllm030_configuration(config)
    counts, mla, derived, topology = Counter(), [], set(), []
    helper_prefix = "vllm.model_executor.layers."
    for name, module in model.named_modules():
        cls = type(module)
        _require(cls in classes, f"unsupported GLM module: {name} ({cls})")
        if VLLM_VERSION == "0.30.0":
            _check_vllm030_module(module)
        counts[classes[cls]] += 1
        state = vars(module)
        _require(
            all(
                not value
                for key, value in state.items()
                if "hook" in key and isinstance(value, dict)
            ),
            f"module hook on {name}",
        )
        _require("forward" not in state, f"instance forward override on {name}")
        dispatch = state.get("_forward_method")
        if dispatch is not None:
            function = getattr(dispatch, "__func__", None)
            _require(
                getattr(dispatch, "__self__", None) is module
                and function is not None
                and any(
                    function is value
                    for base in cls.__mro__
                    for value in vars(base).values()
                ),
                f"unsupported custom-op dispatch on {name}",
            )
        quant = getattr(module, "quant_method", None)
        if quant is not None:
            _require(
                type(quant).__name__
                in (
                    "UnquantizedLinearMethod",
                    "UnquantizedEmbeddingMethod",
                    "UnquantizedFusedMoEMethod",
                ),
                f"quantized helper on {name}",
            )
            expected_module = {
                "UnquantizedLinearMethod": "linear.",
                "UnquantizedEmbeddingMethod": "vocab_parallel_embedding.",
                "UnquantizedFusedMoEMethod": "fused_moe.unquantized_fused_moe_method.",
            }[type(quant).__name__]
            _class(quant, helper_prefix + expected_module + type(quant).__name__)
            if type(quant).__name__ == "UnquantizedLinearMethod":
                from vllm.model_executor.layers.utils import default_unquantized_gemm

                _require(
                    quant._gemm_impl is default_unquantized_gemm,
                    "alternate linear dispatch",
                )
        if cls.__name__ == "DeepseekV32Attention":
            from vllm.model_executor.layers.attention.mla_attention import MLAAttention
            from vllm.v1.attention.backend import AttentionImplBase

            _require(not module.use_pcp and module.use_sparse, "alternate MLA mode")
            _require(
                module.process_weights_after_loading.__func__
                is MLAAttention.process_weights_after_loading,
                "additional MLA module weight processing",
            )
            _require(
                module.indexer is not None
                and module.skip_topk is False
                and module._index_rope_interleave is True
                and module._fp8_query is False
                and module._fp8_kv_needs_view is False
                and module.dcp_manager is None
                and module.W_UK_T_dcp_qrep is None,
                "alternate DSA attention representation",
            )
            _require(not module.is_amx_bmm_enabled, "packed CPU MLA weights")
            _require(
                module.kv_cache_dtype in ("auto", "bfloat16"), "quantized KV cache"
            )
            _class(
                module.impl,
                "vllm.v1.attention.backends.mla.flashattn_mla_sparse.FlashAttnMLASparseImpl",
            )
            _require(
                module.impl.process_weights_after_loading.__func__
                is AttentionImplBase.process_weights_after_loading,
                "additional MLA backend weight processing",
            )
            plan = _prepare_mla_refresh(module)
            mla.append(plan)
            derived.update((id(module.W_UK_T), id(module.W_UV)))
        elif cls.__name__ == "UnquantizedFusedMoEMethod":
            _check_moe(module)
        elif cls.__name__ == "DeepseekV2MoE":
            _require(
                not module.is_fused_shared_expert_enabled,
                "fused shared-expert layout",
            )
        elif cls.__name__ == "DeepseekV32Model":
            _require(
                not module.replicated_embed and module.embed_tokens.tp_size == 32,
                "replicated input embedding",
            )
            _require(not module.use_sequence_parallel, "sequence-parallel backbone")
        elif cls.__name__ == "DeepseekV32DecoderLayer":
            _require(
                not module.use_sequence_parallel and not module.use_mha,
                "alternate decoder layout",
            )
        elif cls.__name__ == "SharedExperts":
            _require(
                module.enable_dbo is False and module._output == [None, None],
                "shared-expert work outstanding",
            )
        elif cls.__name__ == "MoERunner":
            _require(
                module._combined_gate_weight is None
                and not module._fse_fuse_gate
                and module.routed_input_transform is None
                and module.routed_output_transform is None,
                "additional MoE weight transform",
            )
            _class(
                module.router,
                helper_prefix
                + "fused_moe.router.grouped_topk_router.GroupedTopKRouter",
            )
            bias = module.gate.e_score_correction_bias
            _require(
                bias is module.router.e_score_correction_bias
                and bias is module.routed_experts.e_score_correction_bias,
                "shared router bias alias differs",
            )
        topology.append(
            (
                name,
                id(module),
                tuple((key, id(value)) for key, value in module._modules.items()),
                tuple((key, id(value)) for key, value in module._parameters.items()),
            )
        )
    _require(dict(counts) == MODULE_COUNTS, "GLM module census differs")
    destinations = tuple(
        _DirectDestination(name, value, _parameter_geometry(value))
        for name, value in model.named_parameters()
        if id(value) not in derived
    )
    _require(
        len(destinations) == 1395 and len(derived) == 156,
        "GLM parameter census differs",
    )
    intervals = sorted(
        (str(d.geometry[4]), d.geometry[8], d.geometry[8] + d.geometry[9])
        for d in destinations
    )
    _require(
        all(a[0] != b[0] or a[2] <= b[1] for a, b in pairwise(intervals)),
        "overlapping live GLM destinations",
    )
    signature = (
        tuple(topology),
        tuple((d.name, d.geometry) for d in destinations),
        tuple(p.signature for p in mla),
    )
    return destinations, tuple(mla), signature


@dataclass(frozen=True)
class _GlmDirectPlan:
    model: object
    config: object
    version_id: str
    destinations: tuple
    batch_names: tuple[frozenset[str], ...]
    mla: tuple
    signature: tuple


def prepare_glm_direct(
    model, config, *, version_id, source, batch_names, parameter_layout
):
    """Fail before opening READ on any unsupported campaign configuration."""
    _require(type(source) is PreparedStreamingTensors, "GLM streaming source")
    destinations, mla, signature = _inspect(model, config)
    expected = {d.name: (d.geometry[1], d.geometry[3]) for d in destinations}
    _require(
        parameter_layout == expected,
        "captured GLM load layout differs from live destinations",
    )
    _require(
        source.parameter_names == frozenset(expected), "incomplete GLM source coverage"
    )
    _require(type(batch_names) is tuple and bool(batch_names), "missing GLM batches")
    seen = set()
    for names in batch_names:
        _require(
            type(names) is frozenset
            and names
            and not names & seen
            and names <= expected.keys(),
            "invalid GLM batch coverage",
        )
        seen.update(names)
    _require(seen == expected.keys(), "incomplete GLM batches")
    plan = _GlmDirectPlan(
        model, config, version_id, destinations, batch_names, mla, signature
    )
    return PreparedDirectGroupTensors(version_id=version_id, plan=plan, source=source)


def install_glm_direct(prepared, *, model):
    plan = prepared.plan
    _require(
        type(plan) is _GlmDirectPlan
        and plan.model is model
        and prepared.version_id == plan.version_id,
        "GLM model/version differs",
    )
    started = time.perf_counter()

    def drain():
        try:
            _drain(plan)
        except BaseException:
            prepared.ownership.drain_failed = True
            raise

    # The framework pause has completed. Drain auxiliary CUDA streams as well
    # before checking SharedExperts outputs and touching live parameters.
    drain()
    _, _, signature = _inspect(model, plan.config)
    _require(signature == plan.signature, "GLM bindings changed after preparation")
    metrics = prepared.source.transfer_metrics
    metrics["direct_guard_s"] = time.perf_counter() - started

    def finish():
        refreshed = time.perf_counter()
        for mla in plan.mla:
            mla.refresh()
        drain()
        metrics["derived_refresh_s"] = time.perf_counter() - refreshed

    # No model callbacks run between batches. This path never attaches a receive
    # tensor, calls layerwise reload or calls the general post-load dispatcher.
    _install_copy_batches(
        prepared, plan=plan, validate_batch=_dispatch_guard, finish=finish
    )
    metrics.update(
        glm_direct_install=1,
        reload_s=0.0,
        retention_scan_s=0.0,
        retention_batch_scan_s=0.0,
        retention_final_scan_s=0.0,
        retention_arena_setup_s=0.0,
        retention_batch_scans=0,
        retention_final_scans=0,
    )
