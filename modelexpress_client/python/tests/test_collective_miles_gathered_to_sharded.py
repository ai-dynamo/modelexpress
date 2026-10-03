# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A bare MILES external loader feeding a sharded SGLang engine.

The bare external loader (MILES ``WeightTransferProtocol`` with no
``configure_model`` hook) never hands the adapter ``tensor_layouts``. The
trainer gathers TP, so every TP rank yields every tensor whole. With
``MX_MILES_DST_LAYOUT=sharded`` the adapter must still plan a replicated
source and per-engine-rank sharded destinations, and the SGLang loader must
install those slices through aliased engine views.
"""

from types import SimpleNamespace

import pytest
import torch

from modelexpress_rl.collective.integrations import miles_protocol
from modelexpress_rl.collective.integrations._common import _expected_index
from modelexpress_rl.collective.integrations.miles_protocol import (
    MilesCollectiveProtocolCore,
)
from modelexpress_rl.collective.integrations.sglang import SglangLoader
from modelexpress_rl.collective.types import MeshSpec, Placement
from tests.test_collective_miles_layouts import _Group
from tests.test_collective_sglang_engine_view import (
    BACKENDS,
    HEADS,
    VOCAB,
    _hf_tensors,
    _model,
    _norm_tensors,
    _reproduced_layers,
    _stock_load,
    cpu_storage,  # noqa: F401  (pytest fixture)
)

R = Placement.replicate()
TP = 2
KV_HEADS = 4
TRAINER_SLOTS = TP


def _bare_protocol(monkeypatch, tmp_path, tp_rank):
    """A TP2 trainer rank driven the way bare MILES drives it: no configure_model."""
    monkeypatch.setenv("MX_SERVER_ADDRESS", "mx:50051")
    monkeypatch.setenv("MX_MILES_DST_LAYOUT", "sharded")
    (tmp_path / "config.json").write_text(
        '{"num_attention_heads": %d, "num_key_value_heads": %d, '
        '"vocab_size": %d, "tie_word_embeddings": true}' % (HEADS, KV_HEADS, VOCAB)
    )

    def all_gather(value):
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], list):
            return [(rank, value[1]) for rank in range(TP)]
        return [value] * TP

    monkeypatch.setattr(miles_protocol, "_rank_and_world", lambda: (tp_rank, TP))
    monkeypatch.setattr(miles_protocol, "_all_gather", all_gather)
    protocol = MilesCollectiveProtocolCore(SimpleNamespace(hf_checkpoint=str(tmp_path)))
    # Deliberately no configure_model call: bare MILES has no such hook.
    assert protocol._iterator_layouts is None
    parallel = SimpleNamespace(
        pp=_Group(),
        tp=_Group(TP, tp_rank),
        ep=_Group(),
        etp=_Group(),
        cp=_Group(),
        intra_dp=_Group(1, 0),
        indep_dp=_Group(),
    )
    protocol.connect(
        [object()],
        [TP],
        [0],
        parallel,
        # The resolved placement bare MILES passes on #3868: TP gathered.
        SimpleNamespace(gather_pp=False, gather_tp=True, gather_ep=True),
        "target",
    )
    return protocol


def _plan_from_bare_trainer(monkeypatch, tmp_path, tensors):
    """Both trainer TP ranks yield the same whole tensors; return their plans."""
    plans = []
    for tp_rank in range(TP):
        protocol = _bare_protocol(monkeypatch, tmp_path, tp_rank)
        whole = [(name, tensor.clone()) for name, tensor in tensors.items()]
        protocol.begin_sync(1, lambda *, materialize, t=whole: iter([t]))
        assert {
            name: tuple(shape) for name, shape in protocol._local_shapes.items()
        } == {name: tuple(tensor.shape) for name, tensor in tensors.items()}
        plans.append(protocol)
    return plans


@pytest.mark.usefixtures("cpu_storage")
@pytest.mark.parametrize("build", BACKENDS)
def test_whole_trainer_tensors_plan_replicate_to_shard_and_install_aliased(
    build, monkeypatch, tmp_path
):
    tensors = _norm_tensors(_hf_tensors(KV_HEADS))
    protocols = _plan_from_bare_trainer(monkeypatch, tmp_path, tensors)
    plan = protocols[0]._plan

    # Both trainer ranks derive the identical wire plan.
    assert plan == protocols[1]._plan
    assert protocols[0]._topology.m2n_abi_version == (
        miles_protocol.SHARDED_DESTINATION_ABI
    )
    sharded = 0
    for entry in plan.bulk:
        assert entry.src_mesh == MeshSpec((TP,))
        assert entry.src_placements == (R,), entry.name
        assert entry.dst_mesh == MeshSpec((TP,), rank_offset=TRAINER_SLOTS)
        assert entry.global_shape == tuple(tensors[entry.name].shape)
        if entry.dst_placements != (R,):
            sharded += 1
    assert sharded >= 8
    assert dict((e.name, e.dst_placements) for e in plan.bulk)["model.norm.weight"] == (
        R,
    )

    for rank in range(TP):
        layers = build(KV_HEADS, rank, TP, VOCAB)
        model = _model(layers, kv_heads=KV_HEADS)
        loader = SglangLoader(
            plan=plan,
            model=model,
            device="cpu",
            layer_groups=tuple((entry.name,) for entry in plan.bulk),
            generator_index=rank,
            tp_rank=rank,
            tp_size=TP,
        )
        specs = loader.local_params()
        qkv = layers["self_attn.qkv_proj"].weight
        # Local shapes of the destination slices are the engine views' shapes.
        for entry in plan.bulk:
            index = _expected_index(
                entry.global_shape,
                entry.dst_mesh,
                entry.dst_placements,
                TRAINER_SLOTS + rank,
            )
            want = tuple(tensors[entry.name][index].shape)
            assert tuple(specs[entry.name].base.shape) == want, entry.name
        k_view = specs["model.layers.0.self_attn.k_proj.weight"].base
        assert k_view.untyped_storage().data_ptr() == qkv.untyped_storage().data_ptr()

        loader.start_new_round("v1")
        for group_id, entry in enumerate(plan.bulk):
            index = _expected_index(
                entry.global_shape,
                entry.dst_mesh,
                entry.dst_placements,
                TRAINER_SLOTS + rank,
            )
            specs[entry.name].base.copy_(tensors[entry.name][index])
            loader.install(group_id)
        loader.finish()

        stock = _reproduced_layers(KV_HEADS, rank, TP, VOCAB)
        _stock_load(
            stock, {k: v for k, v in tensors.items() if k != "model.norm.weight"}
        )
        for key, module in stock.items():
            assert torch.equal(module.weight, layers[key].weight), (key, rank)
        # Only the replicated norm took the staged load_weights path.
        assert [[name for name, _ in call] for call in model.loaded] == [
            ["model.norm.weight"]
        ]
