# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from types import SimpleNamespace

import pytest
import torch
from modelexpress_rl.inference.engines.vllm import partial_checkpoint
from modelexpress_rl.inference.engines.vllm.partial_checkpoint import (
    StagedVllmCheckpoint,
    _Copy,
    _projection_aliases,
    _resolve_bindings,
    _stage_mla,
    _stage_weight,
    _Unsupported,
    _validate_aliases,
)
from modelexpress_rl.inference.receiver import PreparedCheckpoint
from torch import nn


class Projection(nn.Module):
    def __init__(self, rank, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 2, dtype=dtype), requires_grad=False)
        self.weight.output_dim = 0
        self.tp_rank = rank

    def loader(self, param, source):
        param.data.copy_(
            source.narrow(0, self.tp_rank * param.shape[0], param.shape[0])
        )


class Attention(nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.kv_b_proj = projection
        self.quant_config = None
        self.is_aiter_triton_fp4_bmm_enabled = False
        self.is_aiter_triton_fp8_bmm_enabled = False
        self.calculate_kv_scales = False
        self.register_buffer("_q_scale", torch.ones(1))
        # Strided, shared storage with the projection, as in unquantized MLA.
        self.process_weights_after_loading(projection.weight.dtype)

    def process_weights_after_loading(self, dtype):
        weight = self.kv_b_proj.weight.to(dtype).T.view(2, 2, 2)
        key, value = weight.split([1, 1], dim=-1)
        self.W_UV = value.transpose(0, 1)
        self.W_UK_T = key.permute(1, 2, 0)


@pytest.mark.parametrize("rank", [0, 1])
def test_partial_install_stages_tp_and_mla_without_mutating_unselected_tensors(rank):
    layer = Projection(rank)
    attention = Attention(layer)
    untouched = Projection(1 - rank)
    unrelated = Attention(untouched)
    source = torch.arange(16, dtype=torch.bfloat16).reshape(8, 2)
    destinations = [layer.weight, attention.W_UV, attention.W_UK_T]
    addresses = [(t.data_ptr(), t.stride(), t.storage_offset()) for t in destinations]
    old_unrelated = [
        t.clone() for t in (untouched.weight, unrelated.W_UV, unrelated.W_UK_T)
    ]

    with torch.no_grad():
        shadow = _stage_weight(layer, source, Projection.loader)
        copies = _stage_mla(attention, shadow, Attention)
        copies.append(_Copy.stage(layer, "weight", shadow.weight))
    assert torch.count_nonzero(layer.weight) == 0
    assert torch.count_nonzero(attention.W_UV) == 0

    model = nn.Module()
    model.projection = layer
    model.attention = attention
    _validate_aliases(model, copies, {id(layer): {"projection", "attention.kv_b_proj"}})
    StagedVllmCheckpoint(tuple(copies)).install()

    expected = source[rank * 4 : (rank + 1) * 4]
    assert torch.equal(layer.weight, expected)
    key, value = expected.T.view(2, 2, 2).split([1, 1], dim=-1)
    assert torch.equal(attention.W_UV, value.transpose(0, 1))
    assert torch.equal(attention.W_UK_T, key.permute(1, 2, 0))
    assert addresses == [
        (t.data_ptr(), t.stride(), t.storage_offset()) for t in destinations
    ]
    assert attention.W_UV is destinations[1] and layer.weight is destinations[0]
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            old_unrelated,
            (untouched.weight, unrelated.W_UV, unrelated.W_UK_T),
            strict=True,
        )
    )


def test_embedding_loader_stages_padding_without_touching_live_weight():
    layer = Projection(1)
    source = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)

    def embedding_loader(_layer, param, source):
        selected = source[4:6]
        param[:2].data.copy_(selected)
        param[2:].data.zero_()

    with torch.no_grad():
        shadow = _stage_weight(layer, source, embedding_loader)
    assert torch.count_nonzero(layer.weight) == 0
    assert torch.equal(shadow.weight[:2], source[4:6])
    assert torch.count_nonzero(shadow.weight[2:]) == 0


def test_destination_rebinding_rejects_entire_write_set_before_mutation():
    first, second = Projection(0), Projection(1)
    copies = tuple(
        _Copy.stage(layer, "weight", torch.ones_like(layer.weight))
        for layer in (first, second)
    )
    second.weight = nn.Parameter(torch.zeros_like(second.weight), requires_grad=False)
    with pytest.raises(RuntimeError, match="destination changed"):
        StagedVllmCheckpoint(copies).install()
    assert torch.count_nonzero(first.weight) == 0
    assert torch.count_nonzero(second.weight) == 0


@pytest.mark.parametrize("extra", ["parameter", "buffer", "packed"])
def test_incomplete_or_packed_group_declines_before_loading(extra):
    layer = Projection(0)
    if extra == "parameter":
        layer.other = nn.Parameter(torch.zeros(1))
    elif extra == "buffer":
        layer.register_buffer("extra", torch.ones(1))
    else:
        layer.weight.packed_dim = 0
    calls = []
    with pytest.raises(_Unsupported):
        _stage_weight(
            layer,
            torch.ones(8, 2, dtype=torch.bfloat16),
            lambda *args: calls.append(args),
        )
    assert calls == []


@pytest.mark.parametrize("change", ["scale", "host", "new_tensor", "shape"])
def test_unexpected_mla_finalizer_effect_declines_and_leaves_live_state_unchanged(
    change,
):
    class ChangedAttention(Attention):
        def process_weights_after_loading(self, dtype):
            super().process_weights_after_loading(dtype)
            if getattr(self, "initialized", False):
                if change == "scale":
                    self._q_scale.fill_(7)
                elif change == "host":
                    self.new_host_cache = 7.0
                elif change == "new_tensor":
                    self.new_tensor = torch.ones(1)
                else:
                    self.W_UV = torch.ones(1, dtype=dtype)

    layer = Projection(0)
    attention = ChangedAttention(layer)
    attention.initialized = True
    with torch.no_grad():
        shadow = _stage_weight(
            layer, torch.ones(8, 2, dtype=torch.bfloat16), Projection.loader
        )
        with pytest.raises(_Unsupported):
            _stage_mla(attention, shadow, ChangedAttention)
    assert torch.equal(attention._q_scale, torch.ones(1))
    assert not hasattr(attention, "new_host_cache")
    assert not hasattr(attention, "new_tensor")
    assert torch.count_nonzero(layer.weight) == 0


def test_loader_exception_leaves_live_parameter_unchanged():
    layer = Projection(0)

    def failed(_layer, param, _source):
        param.data.fill_(9)
        raise RuntimeError("staging failure")

    with pytest.raises(RuntimeError, match="staging failure"):
        _stage_weight(layer, torch.ones(8, 2, dtype=torch.bfloat16), failed)
    assert torch.count_nonzero(layer.weight) == 0


@pytest.mark.parametrize(
    "alias", ["tied_head", "duplicate_parameter", "buffer_view", "module_alias"]
)
def test_unaccounted_storage_alias_declines(alias):
    model = nn.Module()
    model.layer = Projection(0)
    if alias == "tied_head":
        model.head = nn.Module()
        model.head.weight = model.layer.weight
    elif alias == "duplicate_parameter":
        model.layer.second_name = model.layer.weight
    elif alias == "module_alias":
        model.head = model.layer
    else:
        model.register_buffer("view", model.layer.weight.view(-1))
    copies = [_Copy.stage(model.layer, "weight", torch.ones_like(model.layer.weight))]
    with pytest.raises(_Unsupported, match="aliased storage"):
        _validate_aliases(model, copies)
    assert torch.count_nonzero(model.layer.weight) == 0


def test_mla_staging_never_copies_unrelated_kv_cache(monkeypatch):
    layer = Projection(0)
    attention = Attention(layer)
    attention.kv_cache = torch.zeros(128, 128)
    clone = torch.Tensor.clone

    def tracked(tensor, *args, **kwargs):
        assert tensor is not attention.kv_cache
        return clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", tracked)
    with torch.no_grad():
        shadow = _stage_weight(
            layer, torch.ones(8, 2, dtype=torch.bfloat16), Projection.loader
        )
        copies = _stage_mla(attention, shadow, Attention)
    assert len(copies) == 2
    assert attention.kv_cache._version == 0
    assert torch.count_nonzero(attention.kv_cache) == 0


@pytest.mark.parametrize("unsupported", ["graph", "fp8_cache", "ep", "cpu_offload"])
def test_runtime_contract_declines_before_checkpoint_reads(
    monkeypatch, tmp_path, unsupported
):
    monkeypatch.setattr(partial_checkpoint, "_contract", lambda: (nn.Module,) * 7)
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True, dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, enable_expert_parallel=False
        ),
        cache_config=SimpleNamespace(cpu_offload_gb=0, cache_dtype="auto"),
    )
    if unsupported == "graph":
        config.model_config.enforce_eager = False
    elif unsupported == "fp8_cache":
        config.cache_config.cache_dtype = "fp8"
    elif unsupported == "ep":
        config.parallel_config.enable_expert_parallel = True
    else:
        config.cache_config.cpu_offload_gb = 1

    def unexpected_read(_path):
        pytest.fail("unsupported configuration read checkpoint metadata")

    monkeypatch.setattr(partial_checkpoint, "index_checkpoint_tensors", unexpected_read)
    assert (
        partial_checkpoint.prepare_partial_checkpoint(
            nn.Module(),
            config,
            PreparedCheckpoint("target", tmp_path, {}),
            serving_version="base",
        )
        is None
    )


@pytest.mark.parametrize("change", ["same", "value", "shape", "dtype"])
def test_recreated_mla_range_scalars_are_compared_strictly(change):
    class RangeAttention(Attention):
        def process_weights_after_loading(self, dtype):
            super().process_weights_after_loading(dtype)
            self.q_range = torch.tensor(7.0)
            if getattr(self, "initialized", False):
                if change == "value":
                    self.q_range.fill_(8)
                elif change == "shape":
                    self.q_range = self.q_range.reshape(1)
                elif change == "dtype":
                    self.q_range = self.q_range.double()

    layer = Projection(0)
    attention = RangeAttention(layer)
    attention.initialized = True
    original = attention.q_range
    with torch.no_grad():
        shadow = _stage_weight(
            layer, torch.ones(8, 2, dtype=torch.bfloat16), Projection.loader
        )
        if change == "same":
            assert len(_stage_mla(attention, shadow, RangeAttention)) == 2
        else:
            with pytest.raises(_Unsupported, match="cache state"):
                _stage_mla(attention, shadow, RangeAttention)
    assert attention.q_range is original and original.item() == 7
    assert torch.count_nonzero(layer.weight) == 0


class OuterAttention(nn.Module):
    pass


class AttentionWrapper(nn.Module):
    pass


def _wrapped_projection(outer_type=OuterAttention, wrapper_type=AttentionWrapper):
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    model.language_model.model.layers = nn.ModuleList([nn.Module()])
    outer = outer_type.__new__(outer_type)
    nn.Module.__init__(outer)
    wrapper = wrapper_type.__new__(wrapper_type)
    nn.Module.__init__(wrapper)
    outer.kv_b_proj = Projection(0)
    wrapper.kv_b_proj = outer.kv_b_proj
    wrapper.mla_attn = Attention(outer.kv_b_proj)
    outer.mla_attn = wrapper
    model.language_model.model.layers[0].self_attn = outer
    shadow = _stage_weight(
        outer.kv_b_proj, torch.ones(8, 2, dtype=torch.bfloat16), Projection.loader
    )
    copies = _stage_mla(wrapper.mla_attn, shadow, Attention)
    copies.append(_Copy.stage(outer.kv_b_proj, "weight", shadow.weight))
    return model, outer, wrapper, copies


@pytest.mark.parametrize("combined", [False, True])
def test_audited_projection_wrapper_aliases_install(combined):
    model, outer, wrapper, copies = _wrapped_projection()
    if combined:
        model.language_model.model.embed_tokens = Projection(0)
        embedding = model.language_model.model.embed_tokens
        copies.append(
            _Copy.stage(embedding, "weight", torch.ones_like(embedding.weight))
        )
    paths = _projection_aliases(
        model,
        "language_model.model.layers.0.self_attn",
        outer.kv_b_proj,
        wrapper.mla_attn,
        OuterAttention,
        AttentionWrapper,
    )
    _validate_aliases(model, copies, paths)
    StagedVllmCheckpoint(tuple(copies)).install()
    for item in copies:
        assert torch.equal(item.destination, item.source)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_projection",
        "extra_wrapper",
        "extra_outer",
        "extra_consumer",
        "tied_parameter",
        "buffer_view",
        "wrong_wrapper",
        "wrong_outer",
        "different_projection",
        "different_consumer",
        "missing_path",
    ],
)
def test_audited_projection_wrapper_rejects_unknown_topology_or_storage(mutation):
    model, outer, wrapper, copies = _wrapped_projection()
    attention = wrapper.mla_attn
    if mutation == "extra_projection":
        model.extra = outer.kv_b_proj
    elif mutation == "extra_wrapper":
        model.extra = wrapper
    elif mutation == "extra_outer":
        model.extra = outer
    elif mutation == "extra_consumer":
        model.extra = attention
    elif mutation == "tied_parameter":
        model.tied = outer.kv_b_proj.weight
    elif mutation == "buffer_view":
        model.register_buffer("view", outer.kv_b_proj.weight.view(-1))
    elif mutation == "wrong_wrapper":

        class Replacement(AttentionWrapper):
            pass

        wrapper.__class__ = Replacement
    elif mutation == "wrong_outer":

        class Replacement(OuterAttention):
            pass

        outer.__class__ = Replacement
    elif mutation == "different_projection":
        wrapper.kv_b_proj = Projection(0)
    elif mutation == "different_consumer":
        wrapper.mla_attn = Attention(outer.kv_b_proj)
    else:
        del wrapper.kv_b_proj
    with pytest.raises((_Unsupported, AttributeError)):
        paths = _projection_aliases(
            model,
            "language_model.model.layers.0.self_attn",
            outer.kv_b_proj,
            attention,
            OuterAttention,
            AttentionWrapper,
        )
        _validate_aliases(model, copies, paths)
    assert torch.count_nonzero(outer.kv_b_proj.weight) == 0


class Embedding(Projection):
    pass


class Unquantized:
    pass


def _capability_bindings(model, mapping):
    return _resolve_bindings(
        model,
        mapping,
        Embedding,
        Unquantized,
        Projection,
        Unquantized,
        Attention,
        OuterAttention,
        AttentionWrapper,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_non_kimi_mapping_supports_multiple_layers_and_floating_dtypes(dtype):
    model = nn.Module()
    model.tokens = Embedding(0, dtype)
    model.tokens.quant_method = Unquantized()
    model.tokens.num_embeddings = model.tokens.org_vocab_size = 8
    model.blocks = nn.ModuleList()
    mapping = {"checkpoint.vocabulary": "tokens"}
    for index in range(2):
        outer = OuterAttention()
        outer.kv_b_proj = Projection(0, dtype)
        outer.kv_b_proj.quant_method = Unquantized()
        outer.mla_attn = AttentionWrapper()
        outer.mla_attn.kv_b_proj = outer.kv_b_proj
        outer.mla_attn.mla_attn = Attention(outer.kv_b_proj)
        model.blocks.append(outer)
        mapping[f"checkpoint.projection.{index}"] = f"blocks.{index}.kv_b_proj"
    bindings = _capability_bindings(model, mapping)
    copies, aliases = [], {}
    for index, binding in enumerate(bindings.values(), start=1):
        source = torch.full((8, 2), index, dtype=dtype)
        shadow = _stage_weight(binding.layer, source, Projection.loader)
        if binding.attention is not None:
            copies.extend(_stage_mla(binding.attention, shadow, Attention))
            aliases.update(binding.aliases)
        copies.append(_Copy.stage(binding.layer, "weight", shadow.weight))
    _validate_aliases(model, copies, aliases)
    StagedVllmCheckpoint(tuple(copies)).install()
    for item in copies:
        assert item.destination.dtype == dtype
        assert torch.equal(item.destination, item.source)


@pytest.mark.parametrize(
    "invalid", ["duplicate", "unknown", "quantized", "added_vocab", "missing_consumer"]
)
def test_explicit_mapping_does_not_bypass_capability_checks(invalid):
    model = nn.Module()
    model.tokens = Embedding(0)
    model.tokens.quant_method = Unquantized()
    model.tokens.num_embeddings = model.tokens.org_vocab_size = 8
    mapping = {"source": "tokens"}
    if invalid == "duplicate":
        mapping["other_source"] = "tokens"
    elif invalid == "unknown":
        mapping["source"] = "missing"
    elif invalid == "quantized":
        model.tokens.quant_method = object()
    elif invalid == "added_vocab":
        model.tokens.num_embeddings = 9
    else:
        model.tokens = Projection(0)
        model.tokens.quant_method = Unquantized()
    with pytest.raises((_Unsupported, AttributeError)):
        _capability_bindings(model, mapping)
