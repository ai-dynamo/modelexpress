# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gathered source's plan contract and the destinations it builds.

Plan-shape tests run in one process: the trainer-world helpers are replaced
by a stand-in that reports the other ranks' manifests, so each test sees the
plan a real DP x TP world would build. ``test_collective_miles_multirank.py``
drives the same coordination over real Gloo ranks.

Adapted from 0eab6fbd's test_collective_miles_layouts.py: MX-3a sources are
always whole tensors (MILES gathers TP), so the trainer-local layout contract
cases are dropped and a fail-closed case replaces them.
"""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from modelexpress_rl.collective.integrations import miles_protocol
from modelexpress_rl.collective.integrations.miles_protocol import (
    MilesCollectiveProtocolCore,
    TensorLayout,
)
from modelexpress_rl.collective.integrations.sglang_layout import (
    SglangModelFacts,
    destination_shard_dim,
)
from modelexpress_rl.collective.plan import plan_digest
from modelexpress_rl.collective.types import MeshSpec, Placement

R = Placement.replicate()

# Qwen3-0.6B geometry, shrunk: 16 query heads, 8 KV heads, head_dim 4.
HIDDEN = 32
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 4
FFN = 48
VOCAB = 128


def _hf_shapes(layers=1):
    shapes = {
        "model.embed_tokens.weight": ((VOCAB, HIDDEN), 0),
        "model.norm.weight": ((HIDDEN,), None),
    }
    for layer in range(layers):
        pre = f"model.layers.{layer}."
        shapes.update(
            {
                pre + "self_attn.q_proj.weight": ((HEADS * HEAD_DIM, HIDDEN), 0),
                pre + "self_attn.k_proj.weight": ((KV_HEADS * HEAD_DIM, HIDDEN), 0),
                pre + "self_attn.v_proj.weight": ((KV_HEADS * HEAD_DIM, HIDDEN), 0),
                pre + "self_attn.o_proj.weight": ((HIDDEN, HEADS * HEAD_DIM), 1),
                pre + "self_attn.q_norm.weight": ((HEAD_DIM,), None),
                pre + "self_attn.k_norm.weight": ((HEAD_DIM,), None),
                pre + "mlp.gate_proj.weight": ((FFN, HIDDEN), 0),
                pre + "mlp.up_proj.weight": ((FFN, HIDDEN), 0),
                pre + "mlp.down_proj.weight": ((HIDDEN, FFN), 1),
                pre + "input_layernorm.weight": ((HIDDEN,), None),
                pre + "post_attention_layernorm.weight": ((HIDDEN,), None),
            }
        )
    return shapes


class _Group:
    def __init__(self, size=1, rank=0):
        self.size = size
        self.rank = rank


def _parallel_state(*, dp=1, tp=1, dp_rank=0, tp_rank=0):
    return SimpleNamespace(
        pp=_Group(),
        tp=_Group(tp, tp_rank),
        ep=_Group(),
        etp=_Group(),
        cp=_Group(),
        intra_dp=_Group(dp, dp_rank),
        indep_dp=_Group(),
    )


@pytest.fixture(autouse=True)
def _server_address(monkeypatch):
    monkeypatch.setenv("MX_SERVER_ADDRESS", "mx:50051")
    monkeypatch.delenv("MX_MILES_DST_LAYOUT", raising=False)


@pytest.fixture
def checkpoint(tmp_path):
    config = {
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "vocab_size": VOCAB,
        "tie_word_embeddings": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def _fake_world(monkeypatch, world):
    """Stand in for a trainer world: this process is rank 0 of ``world``."""

    def all_gather(value):
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], list):
            return [(lane_rank, value[1]) for lane_rank in range(world)]
        return [value] * world

    monkeypatch.setattr(miles_protocol, "_rank_and_world", lambda: (0, world))
    monkeypatch.setattr(miles_protocol, "_all_gather", all_gather)


def _protocol(
    monkeypatch,
    *,
    dp=1,
    tp=1,
    engines=(2,),
    layout=None,
    checkpoint=None,
    layers=1,
):
    if layout is not None:
        monkeypatch.setenv("MX_MILES_DST_LAYOUT", layout)
    _fake_world(monkeypatch, dp * tp)
    args = SimpleNamespace(
        hf_checkpoint=None if checkpoint is None else str(checkpoint)
    )
    protocol = MilesCollectiveProtocolCore(args)
    shapes = _hf_shapes(layers)
    offsets = [sum(engines[:index]) for index in range(len(engines))]
    protocol.connect(
        [object() for _ in engines],
        list(engines),
        offsets,
        _parallel_state(dp=dp, tp=tp),
        SimpleNamespace(gather_pp=False, gather_tp=True, gather_ep=True),
        "target",
    )
    # A gathered stream: MILES delivers every tensor whole on every rank.
    tensors = [
        (name, torch.zeros(shape, dtype=torch.bfloat16))
        for name, (shape, _dim) in shapes.items()
    ]
    return protocol, tensors


@contextmanager
def _fails_on_every_rank(error_type, match):
    """A begin_sync failure in a multi-rank world fans out as a RuntimeError
    naming every rank, chained to the local error."""
    with pytest.raises(
        RuntimeError, match=f"begin_sync preparation failed: rank 0: .*{match}"
    ) as caught:
        yield caught
    assert isinstance(caught.value.__cause__, error_type)


def _entries(protocol):
    return {entry.name: entry for entry in protocol._plan.bulk}


class TestTensorLayoutContract:
    def test_tensor_layout_validates_its_fields(self):
        assert TensorLayout((4, 2), 1).global_shape == (4, 2)
        assert TensorLayout([4, 2], None).global_shape == (4, 2)
        for shape, dim in (((), None), ((0, 2), None), ((4, 2), 2), ((4,), -1)):
            with pytest.raises(ValueError):
                TensorLayout(shape, dim)
        with pytest.raises(ValueError):
            TensorLayout((4, 2), True)

    def test_without_layouts_every_tensor_is_whole_on_a_single_tp_rank(
        self, monkeypatch
    ):
        protocol, tensors = _protocol(monkeypatch)
        protocol.begin_sync(1, lambda *, materialize: iter([tensors]))

        assert {
            name: (layout.global_shape, layout.shard_dim)
            for name, layout in protocol._layouts.items()
        } == {name: (tuple(tensor.shape), None) for name, tensor in tensors}

    def test_no_layouts_at_tp2_means_a_gathered_whole_stream(self, monkeypatch):
        # MILES contract: every TP rank holds every tensor whole.
        protocol, tensors = _protocol(monkeypatch, dp=2, tp=2)

        protocol.begin_sync(1, lambda *, materialize: iter([tensors]))

        for entry in protocol._plan.bulk:
            assert entry.src_mesh == MeshSpec((2, 2))
            assert entry.src_placements == (R, R)
            assert entry.global_shape == _hf_shapes()[entry.name][0]
            assert entry.dst_placements == (R,)
        assert protocol._round_version == "1"

    def test_ranks_disagreeing_on_shapes_fail_the_plan(self, monkeypatch):
        protocol, _tensors = _protocol(monkeypatch, tp=2)
        whole = [("model.norm.weight", torch.zeros(HIDDEN, dtype=torch.bfloat16))]

        def all_gather(value):
            if isinstance(value, tuple) and len(value) == 2:
                other = [("model.norm.weight", (HIDDEN // 2,), None)]
                return [value, (1, other)]
            return [value, value]

        monkeypatch.setattr(miles_protocol, "_all_gather", all_gather)

        with pytest.raises(ValueError, match=r"ranks \[1\] differ from rank 0"):
            protocol.begin_sync(1, lambda *, materialize: iter([whole]))
        assert protocol._plan is None

    def test_an_iterator_with_tensor_layouts_fails_closed(self, monkeypatch):
        protocol, tensors = _protocol(monkeypatch, tp=2)
        protocol.configure_model(SimpleNamespace(tensor_layouts={}))

        with _fails_on_every_rank(ValueError, "trainer-local sources"):
            protocol.begin_sync(1, lambda *, materialize: iter([tensors]))
        assert protocol._plan is None


class TestReplicateDestinationPlan:
    def test_dp2_tp2_source_mesh_and_flat_generator_mesh(self, monkeypatch):
        protocol, tensors = _protocol(monkeypatch, dp=2, tp=2, engines=(2, 1))
        protocol.begin_sync(1, lambda *, materialize: iter([tensors]))

        entries = _entries(protocol)
        q_proj = entries["model.layers.0.self_attn.q_proj.weight"]
        norm = entries["model.norm.weight"]
        assert q_proj.src_mesh == MeshSpec((2, 2))
        assert q_proj.src_placements == (R, R)
        assert norm.src_placements == (R, R)
        assert q_proj.global_shape == (HEADS * HEAD_DIM, HIDDEN)
        for entry in entries.values():
            assert entry.dst_mesh == MeshSpec((3,), rank_offset=4)
            assert entry.dst_placements == (R,)
            assert entry.partition_id == 0
            assert entry.group_key == "publish-group-0"
        topology = protocol._topology
        assert len(topology.trainer_slots) == 4
        assert topology.trainer_slots[3].endswith(":trainer-3")
        assert len(topology.generator_slots) == 3
        assert topology.source_partition_count == 1
        assert topology.m2n_abi_version == miles_protocol.REPLICATED_DESTINATION_ABI
        assert [entry.name for entry in protocol._plan.bulk] == [
            entry.name
            for entry in sorted(protocol._plan.bulk, key=lambda e: e.canonical())
        ]

    def test_dp1_uses_the_one_dimensional_tp_mesh(self, monkeypatch):
        protocol, tensors = _protocol(monkeypatch, tp=2)
        protocol.begin_sync(1, lambda *, materialize: iter([tensors]))

        q_proj = _entries(protocol)["model.layers.0.self_attn.q_proj.weight"]
        assert q_proj.src_mesh == MeshSpec((2,))
        assert q_proj.src_placements == (R,)
        assert q_proj.dst_mesh == MeshSpec((2,), rank_offset=2)

    def test_a_world_that_is_not_dp_times_tp_fails(self, monkeypatch):
        protocol, tensors = _protocol(monkeypatch, tp=2)
        monkeypatch.setattr(miles_protocol, "_rank_and_world", lambda: (0, 3))

        with pytest.raises(ValueError, match="must be exactly DP x TP"):
            protocol.begin_sync(1, lambda *, materialize: iter([tensors]))


class TestShardedDestinationPlan:
    def test_one_tp2_engine_splits_by_the_shared_rule(self, monkeypatch, checkpoint):
        protocol, tensors = _protocol(
            monkeypatch, tp=2, engines=(2,), layout="sharded", checkpoint=checkpoint
        )
        protocol.begin_sync(1, lambda *, materialize: iter([tensors]))

        entries = _entries(protocol)
        expected = {
            "q_proj": 0,
            "k_proj": 0,
            "v_proj": 0,
            "gate_proj": 0,
            "up_proj": 0,
            "o_proj": 1,
            "down_proj": 1,
            "q_norm": None,
            "input_layernorm": None,
        }
        for name, entry in entries.items():
            assert entry.src_placements == (R,), name
            assert entry.dst_mesh == MeshSpec((2,), rank_offset=2)
            short = name.split(".")[-2]
            if short in expected:
                dim = expected[short]
                assert entry.dst_placements == (
                    (R,) if dim is None else (Placement.shard(dim),)
                ), name
        assert entries["model.embed_tokens.weight"].dst_placements == (
            Placement.shard(0),
        )
        assert entries["model.norm.weight"].dst_placements == (R,)
        assert protocol._topology.m2n_abi_version == (
            miles_protocol.SHARDED_DESTINATION_ABI
        )

    def test_several_engines_replicate_across_engines_and_split_within(
        self, monkeypatch, checkpoint
    ):
        protocol, tensors = _protocol(
            monkeypatch,
            dp=2,
            tp=2,
            engines=(2, 2, 2),
            layout="sharded",
            checkpoint=checkpoint,
        )
        protocol.begin_sync(1, lambda *, materialize: iter([tensors]))

        down = _entries(protocol)["model.layers.0.mlp.down_proj.weight"]
        assert down.src_mesh == MeshSpec((2, 2))
        assert down.dst_mesh == MeshSpec((3, 2), rank_offset=4)
        assert down.dst_placements == (R, Placement.shard(1))

    def test_tp1_engines_receive_every_tensor_whole(self, monkeypatch, checkpoint):
        protocol, tensors = _protocol(
            monkeypatch, tp=2, engines=(1, 1), layout="sharded", checkpoint=checkpoint
        )
        protocol.begin_sync(1, lambda *, materialize: iter([tensors]))

        for entry in protocol._plan.bulk:
            assert entry.dst_mesh == MeshSpec((2, 1), rank_offset=2)
            assert entry.dst_placements == (R, R)

    def test_the_mode_is_part_of_the_plan_digest(self, monkeypatch, checkpoint):
        digests = {}
        for layout in ("replicate", "sharded"):
            protocol, tensors = _protocol(
                monkeypatch, tp=2, engines=(1, 1), layout=layout, checkpoint=checkpoint
            )
            protocol.begin_sync(1, lambda *, materialize: iter([tensors]))
            topology = protocol._topology
            digests[layout] = plan_digest(
                protocol._plan,
                receiver_protocol=topology.receiver_protocol,
                m2n_abi_version=topology.m2n_abi_version,
            )
        assert digests["replicate"] != digests["sharded"]

    def test_sharded_mode_needs_the_hf_config(self, monkeypatch):
        protocol, tensors = _protocol(monkeypatch, tp=2, layout="sharded")

        with _fails_on_every_rank(ValueError, "pass --hf-checkpoint"):
            protocol.begin_sync(1, lambda *, materialize: iter([tensors]))

    def test_sharded_mode_needs_one_engine_tp_size(self, monkeypatch):
        with pytest.raises(ValueError, match="same TP size"):
            _protocol(monkeypatch, engines=(2, 1), layout="sharded")

    def test_an_unknown_destination_layout_fails_at_construction(self, monkeypatch):
        monkeypatch.setenv("MX_MILES_DST_LAYOUT", "diagonal")
        with pytest.raises(ValueError, match="MX_MILES_DST_LAYOUT must be one of"):
            MilesCollectiveProtocolCore(SimpleNamespace())


class TestDestinationRule:
    FACTS = SglangModelFacts(
        num_attention_heads=16,
        num_key_value_heads=8,
        vocab_size=151936,
        tie_word_embeddings=False,
    )

    @pytest.mark.parametrize(
        ("name", "shape", "tp", "expected"),
        [
            ("model.layers.3.self_attn.q_proj.weight", (2048, 1024), 2, 0),
            ("model.layers.3.self_attn.k_proj.weight", (1024, 1024), 8, 0),
            # KV heads replicate past 8 TP ranks, so the triple stays whole.
            ("model.layers.3.self_attn.k_proj.weight", (1024, 1024), 16, None),
            ("model.layers.3.self_attn.q_proj.weight", (2048, 1024), 16, None),
            ("model.layers.3.mlp.gate_proj.weight", (3072, 1024), 4, 0),
            ("model.layers.3.mlp.up_proj.weight", (3072, 1024), 5, None),
            ("model.layers.3.self_attn.o_proj.weight", (1024, 2048), 4, 1),
            ("model.layers.3.mlp.down_proj.weight", (1024, 3072), 4, 1),
            ("model.embed_tokens.weight", (151936, 1024), 2, 0),
            ("lm_head.weight", (151936, 1024), 2, 0),
            ("model.layers.3.self_attn.q_norm.weight", (128,), 2, None),
            ("model.norm.weight", (1024,), 2, None),
            ("model.layers.3.mlp.experts.0.gate_proj.weight", (768, 1024), 2, None),
            ("model.layers.3.self_attn.q_proj.weight", (2048, 1024), 1, None),
        ],
    )
    def test_destination_shard_dim(self, name, shape, tp, expected):
        assert destination_shard_dim(name, shape, tp, self.FACTS) == expected

    def test_a_padded_vocabulary_or_a_tied_head_stays_whole(self):
        padded = SglangModelFacts(16, 8, 151937, False)
        assert (
            destination_shard_dim("model.embed_tokens.weight", (151937, 8), 1, padded)
            is None
        )
        assert (
            destination_shard_dim("model.embed_tokens.weight", (151937, 8), 2, padded)
            is None
        )
        tied = SglangModelFacts(16, 8, 151936, True)
        assert destination_shard_dim("lm_head.weight", (151936, 8), 2, tied) is None
        assert (
            destination_shard_dim("model.embed_tokens.weight", (151936, 8), 2, tied)
            == 0
        )

    def test_facts_read_hf_configs_and_nested_text_configs(self, tmp_path):
        nested = {
            "text_config": {
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "vocab_size": 64,
            },
            "tie_word_embeddings": True,
        }
        facts = SglangModelFacts.from_config(nested)
        assert facts == SglangModelFacts(8, 2, 64, True)
        (tmp_path / "config.json").write_text(json.dumps(nested))
        assert SglangModelFacts.from_checkpoint(str(tmp_path)) == facts
        assert SglangModelFacts.from_config(
            SimpleNamespace(num_attention_heads=4, vocab_size=8)
        ) == SglangModelFacts(4, 4, 8, False)
        with pytest.raises(ValueError, match="vocab_size"):
            SglangModelFacts.from_config({"num_attention_heads": 4})
