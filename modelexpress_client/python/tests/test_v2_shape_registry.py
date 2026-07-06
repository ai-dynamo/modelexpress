# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for v2 shape descriptors (no GPU / no NIXL required)."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import pytest


# Direct-load shape_descriptors so we can run without the full modelexpress
# package being importable (the package init pulls in nixl_transfer which
# requires a CUDA build to import).
_HERE = Path(__file__).resolve().parent
_MOD_PATH = _HERE.parent / "modelexpress" / "shape_descriptors.py"


@pytest.fixture(scope="module")
def sd():
    spec = importlib.util.spec_from_file_location(
        "modelexpress.shape_descriptors_for_test", _MOD_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modelexpress.shape_descriptors_for_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_replicate_descriptor_round_trip(sd):
    import torch

    t = torch.randn(8, 16, dtype=torch.bfloat16)
    desc = sd.describe_tensor(name="lm_head.weight", tensor=t, rank=0, fsdp_world_size=1)
    assert desc.placement_kind == sd.PLACEMENT_REPLICATE
    assert desc.global_shape == (8, 16)
    assert desc.local_shard_range is None
    blob = sd.encode_registry([desc], version=1, trainer_world_layout="fsdp:1")
    parsed = sd.decode_registry(blob)
    assert parsed["tensors"][0].name == "lm_head.weight"
    assert parsed["tensors"][0].placement_kind == sd.PLACEMENT_REPLICATE


def test_sharded_dtensor_local_range(sd):
    import torch
    from torch.distributed.tensor.placement_types import Shard

    class FakeDT:
        def __init__(self, shape, dtype, placements):
            self.shape = torch.Size(shape)
            self.dtype = dtype
            self.placements = placements

    # Simulating an FSDP shard: rank 2 of 4 holds rows [4, 6)
    fake = FakeDT([2, 16], torch.bfloat16, [Shard(0)])
    desc = sd.describe_tensor(
        name="model.layers.0.mlp.gate_proj.weight",
        tensor=fake,
        rank=2,
        fsdp_world_size=4,
    )
    assert desc.placement_kind == sd.PLACEMENT_SHARD
    assert desc.shard_axis == 0
    assert desc.global_shape == (8, 16)
    assert desc.local_shard_range == (4, 6)


def test_moe_expert_descriptor_in_registry(sd):
    import torch
    from torch.distributed.tensor.placement_types import Shard

    class FakeDT:
        def __init__(self, shape, dtype, placements):
            self.shape = torch.Size(shape)
            self.dtype = dtype
            self.placements = placements

    fake = FakeDT([24, 4096, 12288], torch.bfloat16, [Shard(0)])
    desc = sd.describe_tensor(
        name="model.layers.5.mlp.experts.weight",
        tensor=fake,
        rank=2,
        fsdp_world_size=8,
        is_expert=True,
        expert_axis=0,
        owned_expert_ids={48, 49, 50, 51, 52, 53},
    )
    assert desc.is_expert
    assert desc.global_shape == (192, 4096, 12288)
    assert desc.local_shard_range == (48, 72)
    assert set(desc.owned_expert_ids) == {48, 49, 50, 51, 52, 53}

    blob = sd.encode_registry([desc], version=99, trainer_world_layout="fsdp:8,ep:8")
    parsed = sd.decode_registry(blob)
    parsed_desc = parsed["tensors"][0]
    assert parsed_desc.is_expert
    assert set(parsed_desc.owned_expert_ids) == {48, 49, 50, 51, 52, 53}
    assert parsed_desc.global_shape == (192, 4096, 12288)


def test_expert_owner_map_uniform(sd):
    m = sd.even_expert_owner_map(num_experts=192, ep_world_size=8)
    assert all(len(s) == 24 for s in m.values())
    assert sum(len(s) for s in m.values()) == 192
    # Coverage is exactly the union of [0..192).
    flat = set().union(*m.values())
    assert flat == set(range(192))


def test_expert_owner_map_rejects_uneven(sd):
    with pytest.raises(ValueError, match="not divisible"):
        sd.even_expert_owner_map(num_experts=190, ep_world_size=8)


def test_expert_set_codec_round_trip(sd):
    es = {3, 1, 2, 5, 4, 5, 1}  # duplicates collapse
    encoded = sd.encode_expert_set(es)
    assert encoded == "1,2,3,4,5"
    assert sd.decode_expert_set(encoded) == {1, 2, 3, 4, 5}


def test_decode_expert_set_handles_empty_and_whitespace(sd):
    assert sd.decode_expert_set("") == set()
    assert sd.decode_expert_set(None) == set()
    assert sd.decode_expert_set(" 1, 2 , ,3 ") == {1, 2, 3}


def test_registry_full_round_trip_multitensor(sd):
    """Trainer-side encode → wire → receiver-side decode preserves everything."""
    import torch
    from torch.distributed.tensor.placement_types import Shard

    class FakeDT:
        def __init__(self, shape, dtype, placements):
            self.shape = torch.Size(shape)
            self.dtype = dtype
            self.placements = placements

    descriptors = [
        sd.describe_tensor(
            name="lm_head.weight",
            tensor=torch.randn(2048, 4096, dtype=torch.bfloat16),
            rank=0,
            fsdp_world_size=1,
        ),
        sd.describe_tensor(
            name="model.layers.0.mlp.gate_up_proj.weight",
            tensor=FakeDT([2048, 4096], torch.bfloat16, [Shard(0)]),
            rank=3,
            fsdp_world_size=4,
        ),
        sd.describe_tensor(
            name="model.layers.0.mlp.experts.weight",
            tensor=FakeDT([24, 4096, 12288], torch.bfloat16, [Shard(0)]),
            rank=3,
            fsdp_world_size=8,
            is_expert=True,
            owned_expert_ids={72, 73, 74, 75, 76, 77},
        ),
    ]
    blob = sd.encode_registry(
        descriptors, version=1234, trainer_world_layout="fsdp:8,ep:8"
    )
    parsed = sd.decode_registry(blob)
    assert parsed["version"] == 1234
    assert parsed["trainer_world_layout"] == "fsdp:8,ep:8"
    assert len(parsed["tensors"]) == 3

    by_name = {t.name: t for t in parsed["tensors"]}
    # 1) lm_head: replicate
    h = by_name["lm_head.weight"]
    assert h.placement_kind == sd.PLACEMENT_REPLICATE
    assert h.global_shape == (2048, 4096)
    # 2) gate_up_proj: sharded
    g = by_name["model.layers.0.mlp.gate_up_proj.weight"]
    assert g.placement_kind == sd.PLACEMENT_SHARD
    assert g.global_shape == (8192, 4096)  # 2048 * 4
    assert g.local_shard_range == (6144, 8192)  # rank 3 of 4
    # 3) MoE expert
    e = by_name["model.layers.0.mlp.experts.weight"]
    assert e.is_expert
    assert e.global_shape == (192, 4096, 12288)
    assert e.local_shard_range == (72, 96)
    assert set(e.owned_expert_ids) == {72, 73, 74, 75, 76, 77}
