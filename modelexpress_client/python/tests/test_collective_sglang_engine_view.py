# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sharded-destination engine views against SGLang's own weight loaders.

One Qwen3-shaped decoder layer (fused ``qkv_proj`` and ``gate_up_proj``,
row-parallel ``o_proj`` and ``down_proj``, a vocabulary-parallel embedding)
is built per TP rank twice. The stock copy loads the full HF tensors through
each parameter's own ``weight_loader`` with Qwen3's stacked shard ids; the
other copy starts zeroed and receives only this rank's plan slice through
``_engine_view``. Every live parameter must then be byte-identical.

The ``sglang`` backend uses SGLang's real layer classes. They do not import
from a plain install on CPU, so they run only when ``MX_TEST_SGLANG_PYTHON``
names a fork-base ``python/`` tree (and its dependencies are installed). The
``reproduced`` backend always runs: stand-ins that carry the same attributes
and reproduce the CUDA-path arithmetic of ``QKVParallelLinear``,
``MergedColumnParallelLinear``, ``RowParallelLinear`` and
``VocabParallelEmbedding`` weight loaders at the pinned fork base.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from dataclasses import replace
from torch import nn

import modelexpress_rl.collective.integrations._common as common
import modelexpress_rl.collective.integrations.sglang as sglang_integration
from modelexpress_rl.collective.integrations._common import (
    _derive_wire_plan,
    _expected_index,
    _stack_classes,
    _stack_trainer_buffers,
)
from modelexpress_rl.collective.integrations.sglang import SglangLoader, _engine_view
from modelexpress_rl.collective.integrations.sglang_layout import (
    SglangModelFacts,
    destination_shard_dim,
)
from modelexpress_rl.collective.types import (
    MeshSpec,
    ParamPlan,
    Placement,
    PlacementKind,
    ReshardPlan,
)

HIDDEN = 32
HEADS = 8
HEAD_DIM = 8
FFN = 48
VOCAB = 128

# HF projection -> (SGLang module path suffix, stacked shard id or None).
_TARGETS = {
    "self_attn.q_proj": ("self_attn.qkv_proj", "q"),
    "self_attn.k_proj": ("self_attn.qkv_proj", "k"),
    "self_attn.v_proj": ("self_attn.qkv_proj", "v"),
    "self_attn.o_proj": ("self_attn.o_proj", None),
    "mlp.gate_proj": ("mlp.gate_up_proj", 0),
    "mlp.up_proj": ("mlp.gate_up_proj", 1),
    "mlp.down_proj": ("mlp.down_proj", None),
}


# --- reproduced SGLang layers (layers/linear.py, vocab_parallel_embedding.py)


def _weight(module, shape, *, split_dim_attrs=True):
    weight = nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16), requires_grad=False)
    if split_dim_attrs:
        weight.input_dim = 1
        weight.output_dim = 0
    weight.weight_loader = module.weight_loader
    module.register_parameter("weight", weight)


class ReproducedQKV(nn.Module):
    def __init__(self, hidden, head_size, heads, kv_heads, *, tp_rank, tp_size):
        super().__init__()
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.kv_tp_rank, self.kv_tp_size = tp_rank, tp_size
        self.head_size = self.v_head_size = head_size
        self.total_num_heads, self.total_num_kv_heads = heads, kv_heads
        self.num_heads = heads // tp_size
        if tp_size >= kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = tp_size // kv_heads
        else:
            self.num_kv_heads = kv_heads // tp_size
            self.num_kv_head_replicas = 1
        self.use_presharded_weights = False
        rows = (self.num_heads + 2 * self.num_kv_heads) * head_size
        _weight(self, (rows, hidden))

    def weight_loader(self, param, loaded_weight, loaded_shard_id):
        # linear.py QKVParallelLinear.weight_loader, output_dim branch.
        q = self.num_heads * self.head_size
        kv = self.num_kv_heads * self.head_size
        offset, size = {"q": (0, q), "k": (q, kv), "v": (q + kv, kv)}[loaded_shard_id]
        shard = (
            self.tp_rank
            if loaded_shard_id == "q"
            else self.kv_tp_rank // self.num_kv_head_replicas
        )
        target = param.data.narrow(0, offset, size)
        target.copy_(loaded_weight.narrow(0, shard * size, size))


class ReproducedMergedColumn(nn.Module):
    def __init__(self, hidden, output_sizes, *, tp_rank, tp_size):
        super().__init__()
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.output_sizes = list(output_sizes)
        self.use_presharded_weights = False
        _weight(self, (sum(output_sizes) // tp_size, hidden))

    def weight_loader(self, param, loaded_weight, loaded_shard_id):
        # linear.py MergedColumnParallelLinear.weight_loader, output_dim branch.
        offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        size = self.output_sizes[loaded_shard_id] // self.tp_size
        target = param.data.narrow(0, offset, size)
        target.copy_(loaded_weight.narrow(0, self.tp_rank * size, size))


class ReproducedRow(nn.Module):
    def __init__(self, input_size, output_size, *, tp_rank, tp_size):
        super().__init__()
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.input_size, self.output_size = input_size, output_size
        self.use_presharded_weights = False
        _weight(self, (output_size, input_size // tp_size))

    def weight_loader(self, param, loaded_weight):
        # linear.py RowParallelLinear.weight_loader.
        size = param.data.shape[1]
        param.data.copy_(loaded_weight.narrow(1, self.tp_rank * size, size))


class ReproducedVocab(nn.Module):
    def __init__(self, vocab, hidden, *, tp_rank, tp_size, padding=64):
        super().__init__()
        self.tp_size = tp_size
        self.org_vocab_size = vocab
        self.num_added_embeddings = 0
        self.num_embeddings_padded = -(-vocab // padding) * padding
        per = self.num_embeddings_padded // tp_size
        start = min(tp_rank * per, vocab)
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=start,
            org_vocab_end_index=min(start + per, vocab),
        )
        _weight(self, (per, hidden))

    def weight_loader(self, param, loaded_weight):
        # vocab_parallel_embedding.py VocabParallelEmbedding.weight_loader.
        start = self.shard_indices.org_vocab_start_index
        size = self.shard_indices.org_vocab_end_index - start
        loaded = loaded_weight.narrow(0, start, size)
        param[: loaded.shape[0]].data.copy_(loaded)
        param[loaded.shape[0] :].data.fill_(0)


def _reproduced_layers(kv_heads, rank, tp, vocab):
    return {
        "self_attn.qkv_proj": ReproducedQKV(
            HIDDEN, HEAD_DIM, HEADS, kv_heads, tp_rank=rank, tp_size=tp
        ),
        "self_attn.o_proj": ReproducedRow(
            HEADS * HEAD_DIM, HIDDEN, tp_rank=rank, tp_size=tp
        ),
        "mlp.gate_up_proj": ReproducedMergedColumn(
            HIDDEN, [FFN, FFN], tp_rank=rank, tp_size=tp
        ),
        "mlp.down_proj": ReproducedRow(FFN, HIDDEN, tp_rank=rank, tp_size=tp),
        "embed_tokens": ReproducedVocab(vocab, HIDDEN, tp_rank=rank, tp_size=tp),
    }


# --- SGLang's real layers, from a fork-base tree --------------------------


def _sglang_layers(kv_heads, rank, tp, vocab):
    import sglang.srt.layers.vocab_parallel_embedding as vpe
    from sglang.srt.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )

    vpe.get_parallel = lambda: SimpleNamespace(
        tp_rank=rank, tp_size=tp, attn_tp_rank=rank, attn_tp_size=tp
    )
    bf16 = torch.bfloat16
    layers = {
        "self_attn.qkv_proj": QKVParallelLinear(
            HIDDEN,
            HEAD_DIM,
            HEADS,
            kv_heads,
            bias=False,
            tp_rank=rank,
            tp_size=tp,
            params_dtype=bf16,
        ),
        "self_attn.o_proj": RowParallelLinear(
            HEADS * HEAD_DIM,
            HIDDEN,
            bias=False,
            tp_rank=rank,
            tp_size=tp,
            params_dtype=bf16,
            reduce_results=False,
        ),
        "mlp.gate_up_proj": MergedColumnParallelLinear(
            HIDDEN, [FFN, FFN], bias=False, tp_rank=rank, tp_size=tp, params_dtype=bf16
        ),
        "mlp.down_proj": RowParallelLinear(
            FFN,
            HIDDEN,
            bias=False,
            tp_rank=rank,
            tp_size=tp,
            params_dtype=bf16,
            reduce_results=False,
        ),
        "embed_tokens": vpe.VocabParallelEmbedding(vocab, HIDDEN, params_dtype=bf16),
    }
    for module in layers.values():
        for param in module.parameters():
            param.data.zero_()
    return layers


def _sglang_available() -> bool:
    path = os.environ.get("MX_TEST_SGLANG_PYTHON")
    if not path:
        return False
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        import sglang.srt.layers.linear  # noqa: F401
        import sglang.srt.layers.vocab_parallel_embedding  # noqa: F401
    except Exception:
        return False
    return True


BACKENDS = [
    pytest.param(_reproduced_layers, id="reproduced"),
    pytest.param(
        _sglang_layers,
        id="sglang",
        marks=pytest.mark.skipif(
            not _sglang_available(),
            reason="set MX_TEST_SGLANG_PYTHON to a fork-base sglang python/ tree",
        ),
    ),
]


class _Namespace(nn.Module):
    pass


def _model(layers, *, kv_heads, vocab=VOCAB):
    """A Qwen3ForCausalLM-shaped tree: model.layers.0.{self_attn,mlp}.*."""
    layer = _Namespace()
    layer.self_attn = _Namespace()
    layer.mlp = _Namespace()
    layer.self_attn.qkv_proj = layers["self_attn.qkv_proj"]
    layer.self_attn.o_proj = layers["self_attn.o_proj"]
    layer.mlp.gate_up_proj = layers["mlp.gate_up_proj"]
    layer.mlp.down_proj = layers["mlp.down_proj"]
    inner = _Namespace()
    inner.embed_tokens = layers["embed_tokens"]
    inner.layers = nn.ModuleList([layer])
    model = _Namespace()
    model.model = inner
    model.config = SimpleNamespace(
        num_attention_heads=HEADS,
        num_key_value_heads=kv_heads,
        vocab_size=vocab,
        tie_word_embeddings=True,
    )
    model.loaded = []
    model.load_weights = lambda weights: model.loaded.append(list(weights))
    return model


def _hf_tensors(kv_heads, vocab=VOCAB, seed=0):
    generator = torch.Generator().manual_seed(seed)

    def tensor(*shape):
        return torch.randn(shape, generator=generator).to(torch.bfloat16)

    pre = "model.layers.0."
    return {
        pre + "self_attn.q_proj.weight": tensor(HEADS * HEAD_DIM, HIDDEN),
        pre + "self_attn.k_proj.weight": tensor(kv_heads * HEAD_DIM, HIDDEN),
        pre + "self_attn.v_proj.weight": tensor(kv_heads * HEAD_DIM, HIDDEN),
        pre + "self_attn.o_proj.weight": tensor(HIDDEN, HEADS * HEAD_DIM),
        pre + "mlp.gate_proj.weight": tensor(FFN, HIDDEN),
        pre + "mlp.up_proj.weight": tensor(FFN, HIDDEN),
        pre + "mlp.down_proj.weight": tensor(HIDDEN, FFN),
        "model.embed_tokens.weight": tensor(vocab, HIDDEN),
    }


def _stock_load(layers, tensors):
    """Qwen3ForCausalLM.load_weights: stacked shard ids, then each loader."""
    for name, tensor in tensors.items():
        if name == "model.embed_tokens.weight":
            weight = layers["embed_tokens"].weight
            weight.weight_loader(weight, tensor)
            continue
        hf_suffix = name.removeprefix("model.layers.0.").removesuffix(".weight")
        target, shard_id = _TARGETS[hf_suffix]
        weight = layers[target].weight
        if shard_id is None:
            weight.weight_loader(weight, tensor)
        else:
            weight.weight_loader(weight, tensor, shard_id)


def _facts(kv_heads, vocab=VOCAB):
    return SglangModelFacts(HEADS, kv_heads, vocab, True)


@pytest.mark.parametrize("build", BACKENDS)
@pytest.mark.parametrize("tp", [1, 2, 4])
def test_aliased_slices_match_the_stock_weight_loaders(build, tp):
    kv_heads = 4
    tensors = _hf_tensors(kv_heads)
    for rank in range(tp):
        stock = build(kv_heads, rank, tp, VOCAB)
        _stock_load(stock, tensors)
        layers = build(kv_heads, rank, tp, VOCAB)
        model = _model(layers, kv_heads=kv_heads)
        for name, tensor in tensors.items():
            dim = destination_shard_dim(name, tuple(tensor.shape), tp, _facts(kv_heads))
            if tp > 1:
                assert dim is not None, name
            else:
                # TP1 stays whole in the plan; the view itself is still exact.
                assert dim is None, name
                dim = 0 if "o_proj" not in name and "down_proj" not in name else 1
            view = _engine_view(model, name, tuple(tensor.shape), rank, tp)
            index = _expected_index(
                tuple(tensor.shape), MeshSpec((tp,)), (Placement.shard(dim),), rank
            )
            assert tuple(view.shape) == tuple(tensor[index].shape), name
            view.copy_(tensor[index])
        for key, module in stock.items():
            assert torch.equal(module.weight, layers[key].weight), (key, rank)
            assert module.weight.data_ptr() != layers[key].weight.data_ptr()


@pytest.mark.parametrize("build", BACKENDS)
def test_replicated_kv_heads_are_never_aliased(build):
    # Two KV heads over TP4: SGLang replicates each KV head on two ranks, so
    # rank r's k rows are not the plan's even slice r.
    kv_heads, tp = 2, 4
    tensors = _hf_tensors(kv_heads)
    for rank in range(tp):
        model = _model(build(kv_heads, rank, tp, VOCAB), kv_heads=kv_heads)
        for member in ("q_proj", "k_proj", "v_proj"):
            name = f"model.layers.0.self_attn.{member}.weight"
            shape = tuple(tensors[name].shape)
            assert destination_shard_dim(name, shape, tp, _facts(kv_heads)) is None
            with pytest.raises(ValueError, match="replicated"):
                _engine_view(model, name, shape, rank, tp)


@pytest.mark.parametrize("build", BACKENDS)
def test_a_padded_vocabulary_is_never_aliased(build):
    vocab, tp = 100, 2
    model = _model(build(4, 1, tp, vocab), kv_heads=4, vocab=vocab)

    assert (
        destination_shard_dim(
            "model.embed_tokens.weight", (vocab, HIDDEN), tp, _facts(4, vocab)
        )
        is None
    )
    with pytest.raises(ValueError, match="padded or extended"):
        _engine_view(model, "model.embed_tokens.weight", (vocab, HIDDEN), 1, tp)


# --- the loader over a sharded-destination plan -----------------------------


class _AsCuda:
    """CPU storage that reports an indexed CUDA device to the storage checks."""

    def __init__(self, tensor):
        self._tensor = tensor
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device = "cuda:0"

    def data_ptr(self):
        return self._tensor.data_ptr()

    def is_contiguous(self):
        return self._tensor.is_contiguous()


@pytest.fixture
def cpu_storage(monkeypatch):
    original = common._tensor_signature

    def signature(name, tensor, **kwargs):
        return original(name, _AsCuda(tensor), **kwargs)

    monkeypatch.setattr(common, "_tensor_signature", signature)
    monkeypatch.setattr(sglang_integration, "_tensor_signature", signature)


def _sharded_plan(tensors, *, tp, kv_heads, overrides=None):
    overrides = overrides or {}
    entries = []
    for name, tensor in tensors.items():
        shape = tuple(tensor.shape)
        dim = overrides.get(
            name, destination_shard_dim(name, shape, tp, _facts(kv_heads))
        )
        entries.append(
            ParamPlan(
                name=name,
                global_shape=shape,
                dtype="bfloat16",
                partition_id=0,
                src_mesh=MeshSpec((1,)),
                src_placements=(Placement.replicate(),),
                dst_mesh=MeshSpec((tp,), rank_offset=1),
                dst_placements=(
                    Placement.replicate() if dim is None else Placement.shard(dim),
                ),
                group_key="publish-group-0",
            )
        )
    entries.sort(key=lambda entry: entry.canonical())
    return ReshardPlan(bulk=entries, source_partition_count=1)


def _norm_tensors(tensors):
    tensors = dict(tensors)
    tensors["model.norm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
    return tensors


def test_the_loader_aliases_sharded_entries_and_stages_the_rest(cpu_storage):
    kv_heads, tp, rank = 4, 2, 1
    tensors = _norm_tensors(_hf_tensors(kv_heads))
    layers = _reproduced_layers(kv_heads, rank, tp, VOCAB)
    model = _model(layers, kv_heads=kv_heads)
    plan = _sharded_plan(tensors, tp=tp, kv_heads=kv_heads)
    loader = SglangLoader(
        plan=plan,
        model=model,
        device="cpu",
        layer_groups=tuple((entry.name,) for entry in plan.bulk),
        generator_index=rank,
        tp_rank=rank,
        tp_size=tp,
    )

    specs = loader.local_params()
    qkv = layers["self_attn.qkv_proj"].weight
    k_view = specs["model.layers.0.self_attn.k_proj.weight"].base
    assert k_view.untyped_storage().data_ptr() == qkv.untyped_storage().data_ptr()
    norm = specs["model.norm.weight"].base
    assert tuple(norm.shape) == (HIDDEN,)

    # Receive: write each plan slice into the loader's buffers, then install.
    loader.start_new_round("v1")
    for group_id, entry in enumerate(plan.bulk):
        name = entry.name
        index = _expected_index(
            entry.global_shape, entry.dst_mesh, entry.dst_placements, 1 + rank
        )
        specs[name].base.copy_(tensors[name][index])
        loader.install(group_id)
    loader.finish()

    stock = _reproduced_layers(kv_heads, rank, tp, VOCAB)
    _stock_load(stock, {k: v for k, v in tensors.items() if k != "model.norm.weight"})
    for key, module in stock.items():
        assert torch.equal(module.weight, layers[key].weight), key
    # Only the replicated norm went through SGLang's load_weights.
    assert [[name for name, _ in call] for call in model.loaded] == [
        ["model.norm.weight"]
    ]


def test_the_loader_fails_closed_when_sglang_rebinds_aliased_storage(cpu_storage):
    kv_heads, tp, rank = 4, 2, 0
    tensors = _hf_tensors(kv_heads)
    layers = _reproduced_layers(kv_heads, rank, tp, VOCAB)
    model = _model(layers, kv_heads=kv_heads)
    loader = SglangLoader(
        plan=_sharded_plan(tensors, tp=tp, kv_heads=kv_heads),
        model=model,
        device="cpu",
        generator_index=rank,
        tp_rank=rank,
        tp_size=tp,
    )
    loader.start_new_round("v1")
    loader.finish()

    down = layers["mlp.down_proj"]
    down.weight.data = down.weight.data.clone()

    with pytest.raises(RuntimeError, match="moved or reshaped the live storage"):
        loader.start_new_round("v2")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("kv_override", "disagree on the destination rule"),
        ("fp32", "weight is torch.float32"),
        ("wrong_rank", "is TP coordinate 1"),
    ],
)
def test_the_loader_refuses_sharded_entries_it_cannot_prove(
    cpu_storage, change, message
):
    kv_heads, tp, rank = 4, 2, 0
    tensors = _hf_tensors(kv_heads)
    layers = _reproduced_layers(kv_heads, rank, tp, VOCAB)
    overrides = {}
    generator_index = rank
    if change == "kv_override":
        # Claim a split the shared rule does not allow.
        tensors["model.layers.0.self_attn.q_norm.weight"] = torch.ones(
            HEAD_DIM * 2, dtype=torch.bfloat16
        )
        overrides["model.layers.0.self_attn.q_norm.weight"] = 0
    elif change == "fp32":
        weight = layers["mlp.down_proj"].weight
        weight.data = weight.data.float()
    else:
        generator_index = 1
    with pytest.raises(ValueError, match=message):
        SglangLoader(
            plan=_sharded_plan(tensors, tp=tp, kv_heads=kv_heads, overrides=overrides),
            model=_model(layers, kv_heads=kv_heads),
            device="cpu",
            generator_index=generator_index,
            tp_rank=rank,
            tp_size=tp,
        )


# --- equal-geometry stacks over the same live layers ------------------------


def _more_norms(tensors):
    tensors = _norm_tensors(tensors)
    for index, name in enumerate(
        (
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        )
    ):
        tensors[name] = torch.full((HIDDEN,), float(index + 2), dtype=torch.bfloat16)
    return tensors


def _with_stacks(plan, budget=1 << 20):
    keys = _stack_classes(plan.bulk, budget)
    return ReshardPlan(
        bulk=[
            replace(entry, group_key=keys[entry.name]) if entry.name in keys else entry
            for entry in plan.bulk
        ],
        source_partition_count=1,
    )


def _receive(loader, tensors, rank):
    """Receive one round the way the backend does, then install each group."""
    wire = loader.capture()
    stacks = {
        stack.name: stack for stack in _derive_wire_plan(loader._plan.capture())[1]
    }
    specs = loader.local_params()
    loader.start_new_round("v1")
    for group_id, entry in enumerate(wire.bulk):
        if entry.name in stacks:
            full = torch.stack([tensors[name] for name in stacks[entry.name].members])
        else:
            full = tensors[entry.name]
        index = _expected_index(
            entry.global_shape, entry.dst_mesh, entry.dst_placements, 1 + rank
        )
        spec = specs[entry.name]
        ctx = spec.enter()
        ctx.buf.copy_(full[index])
        spec.leave(ctx)
        loader.install(group_id)
    loader.finish()
    return wire


def _loader_for(plan, model, *, tp, rank):
    wire, _ = _derive_wire_plan(plan)
    return SglangLoader(
        plan=plan,
        model=model,
        device="cpu",
        layer_groups=tuple((entry.name,) for entry in wire.bulk),
        generator_index=rank,
        tp_rank=rank,
        tp_size=tp,
    )


@pytest.mark.parametrize("build", BACKENDS)
@pytest.mark.parametrize("tp", [1, 2, 4])
def test_a_stacked_receive_installs_exactly_what_a_per_tensor_receive_does(
    cpu_storage, build, tp
):
    kv_heads = 4
    tensors = _more_norms(_hf_tensors(kv_heads))
    for rank in range(tp):
        per_tensor = _sharded_plan(tensors, tp=tp, kv_heads=kv_heads)
        stacked = _with_stacks(per_tensor)
        wire, stacks = _derive_wire_plan(stacked)
        # k/v, gate/up and the three norms stack; everything else stays plain.
        assert sorted(len(stack.members) for stack in stacks) == [2, 2, 3]
        assert len(wire.bulk) == len(per_tensor.bulk) - 4

        ref_layers = build(kv_heads, rank, tp, VOCAB)
        ref_model = _model(ref_layers, kv_heads=kv_heads)
        _receive(_loader_for(per_tensor, ref_model, tp=tp, rank=rank), tensors, rank)

        layers = build(kv_heads, rank, tp, VOCAB)
        model = _model(layers, kv_heads=kv_heads)
        loader = _loader_for(stacked, model, tp=tp, rank=rank)
        _receive(loader, tensors, rank)

        stock = build(kv_heads, rank, tp, VOCAB)
        _stock_load(stock, {k: v for k, v in tensors.items() if "norm" not in k})
        for key, module in stock.items():
            # At TP1 every entry is staged and only recorded by this model's
            # stand-in load_weights, so the live layers are untouched; the
            # staged inputs are compared below instead.
            if tp > 1:
                assert torch.equal(module.weight, layers[key].weight), (key, rank)
            assert torch.equal(ref_layers[key].weight, layers[key].weight), (
                key,
                rank,
            )
        loaded = {
            name: tensor.clone() for call in model.loaded for name, tensor in call
        }
        reference = {
            name: tensor.clone() for call in ref_model.loaded for name, tensor in call
        }
        assert loaded.keys() == reference.keys()
        for name, tensor in loaded.items():
            assert torch.equal(tensor, reference[name]), name
            assert torch.equal(tensor, tensors[name]), name
        # The staged members of a stack go through one load_weights call.
        norm_calls = [
            [name for name, _ in call] for call in model.loaded if "norm" in call[0][0]
        ]
        assert [len(call) for call in norm_calls] == [3]
        # TP1 keeps every entry whole, so every stack is staged; at TP>1 the
        # sharded stacks are received through the scratch instead.
        assert (loader._scratch_elements > 0) == (tp > 1)


@pytest.mark.parametrize("tp", [2, 4])
def test_a_sharded_stack_never_installs_through_load_weights(cpu_storage, tp):
    kv_heads = 4
    tensors = _more_norms(_hf_tensors(kv_heads))
    plan = _with_stacks(_sharded_plan(tensors, tp=tp, kv_heads=kv_heads))
    layers = _reproduced_layers(kv_heads, 0, tp, VOCAB)
    loader = _loader_for(plan, _model(layers, kv_heads=kv_heads), tp=tp, rank=0)
    flags = {
        tuple(group): bool(loader._staged_members(index))
        for index, group in enumerate(loader.layer_groups)
    }
    stacks = {stack.name: stack for stack in _derive_wire_plan(plan)[1]}
    for (name,), reads in flags.items():
        if name in stacks and "norm" in stacks[name].members[0]:
            assert reads is True
        else:
            assert reads is False, name


def test_a_stack_whose_engine_views_overlap_is_refused(cpu_storage, monkeypatch):
    kv_heads, tp, rank = 4, 2, 0
    tensors = _more_norms(_hf_tensors(kv_heads))
    plan = _with_stacks(_sharded_plan(tensors, tp=tp, kv_heads=kv_heads))
    layers = _reproduced_layers(kv_heads, rank, tp, VOCAB)
    original = sglang_integration._engine_view

    def overlapping(model, name, global_shape, tp_rank, tp_size):
        # up_proj resolves to gate_proj's slice: two members, one memory range.
        name = name.replace("up_proj", "gate_proj")
        return original(model, name, global_shape, tp_rank, tp_size)

    monkeypatch.setattr(sglang_integration, "_engine_view", overlapping)
    with pytest.raises(ValueError, match="overlap in memory"):
        _loader_for(plan, _model(layers, kv_heads=kv_heads), tp=tp, rank=rank)


def test_a_non_bf16_sharded_stack_is_refused_with_the_stack_named(
    cpu_storage, monkeypatch
):
    kv_heads, tp, rank = 4, 2, 0
    tensors = _more_norms(_hf_tensors(kv_heads))
    plan = _with_stacks(_sharded_plan(tensors, tp=tp, kv_heads=kv_heads))
    layers = _reproduced_layers(kv_heads, rank, tp, VOCAB)
    original = sglang_integration._derive_wire_plan

    def doctored(frozen):
        # The plan's bf16 gate sits in front of this path; doctoring only the
        # sharded stacks proves the scratch path names its own failure if a
        # non-bf16 wire dtype ever arrives past it.
        wire, stacks = original(frozen)
        out = []
        for stack in stacks:
            if stack.entry.dst_placements[-1].kind is PlacementKind.SHARD:
                stack = replace(stack, entry=replace(stack.entry, dtype="float32"))
            out.append(stack)
        return wire, tuple(out)

    monkeypatch.setattr(sglang_integration, "_derive_wire_plan", doctored)
    with pytest.raises(ValueError, match="bfloat16 scratch"):
        _loader_for(plan, _model(layers, kv_heads=kv_heads), tp=tp, rank=rank)


def test_each_stream_gets_its_own_receive_scratch(cpu_storage, monkeypatch):
    kv_heads, tp, rank = 4, 2, 0
    tensors = _more_norms(_hf_tensors(kv_heads))
    plan = _with_stacks(_sharded_plan(tensors, tp=tp, kv_heads=kv_heads))
    layers = _reproduced_layers(kv_heads, rank, tp, VOCAB)
    loader = _loader_for(plan, _model(layers, kv_heads=kv_heads), tp=tp, rank=rank)
    specs = loader.local_params()
    (sharded,) = [
        specs[name]
        for name, stack in loader._stacks.items()
        if name not in loader._stack_buffers and "gate_proj" in stack.members[0]
    ]
    key = {"value": 1}
    monkeypatch.setattr(sglang_integration, "_current_stream_key", lambda: key["value"])

    first = sharded.enter().buf
    # The stream allocated at prepare claims the scratch that was made then.
    assert (
        first.untyped_storage().data_ptr()
        == loader._scratch_primary.untyped_storage().data_ptr()
    )
    assert sharded.enter().buf.untyped_storage().data_ptr() == (
        first.untyped_storage().data_ptr()
    )
    key["value"] = 2
    second = sharded.enter().buf
    assert second.untyped_storage().data_ptr() != first.untyped_storage().data_ptr()
    assert tuple(second.shape) == tuple(first.shape)


@pytest.mark.parametrize("tp", [1, 2])
def test_trainer_and_engine_agree_on_the_wire_plan_and_abi(
    cpu_storage, monkeypatch, tp
):
    import modelexpress_rl.collective.integrations.miles as miles_integration
    from modelexpress_rl.collective.integrations._common import (
        REPLICATED_DESTINATION_ABI,
        SHARDED_DESTINATION_ABI,
        STACK_ABI_SUFFIX,
    )
    from modelexpress_rl.collective.integrations.miles import (
        CollectiveTopology,
        MilesPublisher,
    )
    from modelexpress_rl.collective.plan import plan_digest

    monkeypatch.setattr(
        miles_integration, "_tensor_signature", common._tensor_signature
    )
    kv_heads, rank = 4, 0
    tensors = _more_norms(_hf_tensors(kv_heads))
    plan = _with_stacks(_sharded_plan(tensors, tp=tp, kv_heads=kv_heads))
    wire, stacks = _derive_wire_plan(plan)
    local = {name: tensor.clone() for name, tensor in tensors.items()}
    storage = _stack_trainer_buffers(stacks, local)
    members = {name for stack in stacks for name in stack.members}
    wire_tensors = {n: t for n, t in local.items() if n not in members}
    wire_tensors.update(storage)
    publisher = MilesPublisher(
        plan=wire, source_partition=0, tensors=wire_tensors, source_rank=0
    )
    layers = _reproduced_layers(kv_heads, rank, tp, VOCAB)
    loader = _loader_for(plan, _model(layers, kv_heads=kv_heads), tp=tp, rank=rank)

    abi = (SHARDED_DESTINATION_ABI if tp > 1 else REPLICATED_DESTINATION_ABI) + (
        STACK_ABI_SUFFIX
    )
    topology = CollectiveTopology(
        model_name="m",
        trainer_slots=("t0",),
        generator_slots=tuple(f"g{i}" for i in range(tp)),
        source_partition_count=1,
        m2n_abi_version=abi,
    )
    publisher.validate_topology(topology)
    loader.validate_topology(topology)
    assert plan_digest(publisher.capture(), m2n_abi_version=abi) == plan_digest(
        loader.capture(), m2n_abi_version=abi
    )
    # The trainer sends the stacks it keeps; the members are views of them.
    assert sorted(publisher.parameter_names()) == sorted(loader.parameter_names())
    for stack in stacks:
        assert tuple(storage[stack.name].shape)[0] == len(stack.members)
