# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from modelexpress_rl.refit.reshard.rendezvous import (
    unwrap_rendezvous_blob,
)
from modelexpress_rl.train.engines.megatron import (
    MegatronAliasInput,
    MegatronTensorSpec,
    build_hf_aliases,
    build_megatron_reshard_manifest,
)


class _Manager:
    agent_name = "trainer-r3"
    nixl_metadata = b"agent-metadata"


def test_manifest_builder_reuses_registered_tensor_addresses():
    tensor = torch.zeros((8, 8), dtype=torch.bfloat16)
    published = build_hf_aliases(
        [
            MegatronTensorSpec(
                name="column",
                tensor=tensor,
                role="column",
                hf_names=("column",),
                global_shape=(16, 8),
                placement_kind="SHARD",
                shard_axis=0,
                local_shard_range=(8, 16),
            )
        ],
        agent_name="trainer-r3",
    )

    manifest = build_megatron_reshard_manifest(
        manager=_Manager(),
        published=published,
        metadata_endpoint="10.0.0.3:19003",
    )

    payload = unwrap_rendezvous_blob(manifest.blob)
    stable = json.loads(manifest.blob)
    assert stable["tensor_count"] == 1
    assert stable["total_bytes"] == tensor.numel() * tensor.element_size()
    assert "digest" not in stable["tensors"][0]["shards"][0]
    assert manifest.blob == json.dumps(
        stable, sort_keys=True, separators=(",", ":")
    ).encode()
    assert (payload.agent_metadata, payload.agent_name, payload.metadata_endpoint) == (
        b"agent-metadata",
        "trainer-r3",
        "10.0.0.3:19003",
    )
    assert payload.tensors[0].shards[0].addr == tensor.data_ptr()
    assert payload.tensors[0].shards[0].shard_offset == (8, 0)


def test_manifest_builder_rejects_duplicate_tensor_names():
    tensor = torch.zeros((8, 8), dtype=torch.bfloat16)
    published = build_hf_aliases(
        [
            MegatronTensorSpec(
                name="column",
                tensor=tensor,
                role="column",
                hf_names=("column",),
                global_shape=(8, 8),
                placement_kind="REPLICATE",
                shard_axis=None,
                local_shard_range=None,
            )
        ],
        agent_name="trainer-r3",
    )

    with pytest.raises(ValueError, match="duplicate published tensor"):
        build_megatron_reshard_manifest(
            manager=_Manager(),
            published=[published[0], published[0]],
            metadata_endpoint="10.0.0.3:19003",
        )


def test_gated_aliases_split_each_tp_shard_into_hf_gate_and_up():
    fused = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)

    gate, up = build_hf_aliases(
        [
            MegatronAliasInput(
                name="linear_fc1.weight",
                tensor=fused,
                role="gated_mlp_column",
                hf_names=("gate_proj.weight", "up_proj.weight"),
                global_shape=(16, 4),
                placement_kind="SHARD",
                shard_axis=0,
                local_shard_range=(8, 16),
                extras={"gated_mlp_order": "gate_then_up"},
            )
        ],
        agent_name="trainer-tp1",
    )

    assert gate.full_shape == up.full_shape == (8, 4)
    assert gate.shards[0].shape == up.shards[0].shape == (4, 4)
    assert gate.shards[0].shard_offset == up.shards[0].shard_offset == (4, 0)
    assert gate.shards[0].addr == fused.data_ptr()
    assert up.shards[0].addr == fused[4:].data_ptr()


def test_an_unknown_fused_gate_up_order_is_rejected():
    """The halves map to `hf_names` positionally, so an up-then-gate layout would
    publish the gate projection's bytes under the up projection's name. No digest
    gate can see that: both names receive the bytes their publisher advertised."""
    fused = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)

    with pytest.raises(ValueError, match="gated_mlp_order"):
        build_hf_aliases(
            [
                MegatronAliasInput(
                    name="linear_fc1.weight",
                    tensor=fused,
                    role="gated_mlp_column",
                    hf_names=("gate_proj.weight", "up_proj.weight"),
                    global_shape=(16, 4),
                    placement_kind="SHARD",
                    shard_axis=0,
                    local_shard_range=(8, 16),
                    extras={"gated_mlp_order": "up_then_gate"},
                )
            ],
            agent_name="trainer-tp1",
        )


def test_a_missing_fused_gate_up_order_is_rejected():
    """Absent metadata is not evidence of gate-then-up storage."""
    fused = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)

    with pytest.raises(ValueError, match="gated_mlp_order"):
        build_hf_aliases(
            [
                MegatronAliasInput(
                    name="linear_fc1.weight",
                    tensor=fused,
                    role="gated_mlp_column",
                    hf_names=("gate_proj.weight", "up_proj.weight"),
                    global_shape=(16, 4),
                    placement_kind="SHARD",
                    shard_axis=0,
                    local_shard_range=(8, 16),
                )
            ],
            agent_name="trainer-tp1",
        )


def test_gated_aliases_reject_inconsistent_declared_global_shape():
    fused = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)

    with pytest.raises(
        ValueError, match=r"linear_fc1\.weight: derived gate/up shape"
    ):
        build_hf_aliases(
            [
                MegatronAliasInput(
                    name="linear_fc1.weight",
                    tensor=fused,
                    role="gated_mlp_column",
                    hf_names=("gate_proj.weight", "up_proj.weight"),
                    global_shape=(16, 5),
                    placement_kind="SHARD",
                    shard_axis=0,
                    local_shard_range=(8, 16),
                    extras={"gated_mlp_order": "gate_then_up"},
                )
            ],
            agent_name="trainer-tp1",
        )


def test_qkv_aliases_expose_hf_head_ranges_without_copy():
    qkv = torch.arange(48, dtype=torch.bfloat16).reshape(12, 4)

    q, k, v = build_hf_aliases(
        [
            MegatronAliasInput(
                name="linear_qkv.weight",
                tensor=qkv,
                role="qkv_column",
                hf_names=("q_proj.weight", "k_proj.weight", "v_proj.weight"),
                global_shape=(24, 4),
                placement_kind="SHARD",
                shard_axis=0,
                local_shard_range=(12, 24),
                extras={
                    "num_heads_local": "4",
                    "num_kv_heads_local": "1",
                    "head_dim": "2",
                },
            )
        ],
        agent_name="trainer-tp1",
    )

    assert q.full_shape == (16, 4)
    assert k.full_shape == v.full_shape == (4, 4)
    assert q.shards[0].shape == (8, 4)
    assert k.shards[0].shape == v.shards[0].shape == (2, 4)
    assert q.shards[0].shard_offset == (8, 0)
    assert k.shards[0].shard_offset == v.shards[0].shard_offset == (2, 0)
    assert q.shards[0].addr == qkv.data_ptr()
    assert k.shards[0].addr == qkv[8:].data_ptr()
    assert v.shards[0].addr == qkv[10:].data_ptr()


def test_qkv_aliases_report_missing_extras_with_tensor_name():
    qkv = torch.arange(48, dtype=torch.bfloat16).reshape(12, 4)

    with pytest.raises(
        ValueError,
        match=r"linear_qkv\.weight: QKV aliasing requires extras",
    ):
        build_hf_aliases(
            [
                MegatronAliasInput(
                    name="linear_qkv.weight",
                    tensor=qkv,
                    role="qkv_column",
                    hf_names=("q_proj.weight", "k_proj.weight", "v_proj.weight"),
                    global_shape=(24, 4),
                    placement_kind="SHARD",
                    shard_axis=0,
                    local_shard_range=(12, 24),
                    extras={"head_dim": "2"},
                )
            ],
            agent_name="trainer-tp1",
        )


def test_a_shard_range_beyond_the_global_extent_is_rejected():
    """A range of (16, 24) against a global extent of 16 clears every other
    check: it is the right width, it divides evenly, and it yields source rank 2
    of a 2-rank group. The alias would then describe bytes past the end of the
    tensor it claims to be part of."""
    fused = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)

    with pytest.raises(ValueError, match="outside the global extent"):
        build_hf_aliases(
            [
                MegatronAliasInput(
                    name="linear_fc1.weight",
                    tensor=fused,
                    role="gated_mlp_column",
                    hf_names=("gate_proj.weight", "up_proj.weight"),
                    global_shape=(16, 4),
                    placement_kind="SHARD",
                    shard_axis=0,
                    local_shard_range=(16, 24),
                    extras={"gated_mlp_order": "gate_then_up"},
                )
            ],
            agent_name="trainer-tp1",
        )


def test_a_non_contiguous_qkv_tensor_is_rejected():
    """A shard is published as one address plus a shape, so a strided view would
    advertise bytes that belong to the columns it skipped."""
    strided = torch.arange(96, dtype=torch.bfloat16).reshape(12, 8)[:, :4]
    assert not strided.is_contiguous()

    with pytest.raises(ValueError, match="requires contiguous storage"):
        build_hf_aliases(
            [
                MegatronAliasInput(
                    name="linear_qkv.weight",
                    tensor=strided,
                    role="qkv_column",
                    hf_names=("q_proj.weight", "k_proj.weight", "v_proj.weight"),
                    global_shape=(24, 4),
                    placement_kind="SHARD",
                    shard_axis=0,
                    local_shard_range=(12, 24),
                    extras={
                        "num_heads_local": "4",
                        "num_kv_heads_local": "1",
                        "head_dim": "2",
                    },
                )
            ],
            agent_name="trainer-tp1",
        )


def test_a_non_contiguous_ordinary_alias_is_rejected():
    """The single-name path publishes the tensor's own address, so a transpose
    would hand a reader the untransposed bytes."""
    transposed = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8).t()
    assert not transposed.is_contiguous()

    with pytest.raises(ValueError, match="requires contiguous storage"):
        build_hf_aliases(
            [
                MegatronAliasInput(
                    name="linear_fc2.weight",
                    tensor=transposed,
                    role="row",
                    hf_names=("down_proj.weight",),
                    global_shape=(8, 4),
                    placement_kind="REPLICATE",
                    shard_axis=None,
                    local_shard_range=None,
                )
            ],
            agent_name="trainer-tp1",
        )
