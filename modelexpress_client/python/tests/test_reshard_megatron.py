# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

import pytest
import torch
from modelexpress.refit.reshard.rendezvous import (
    MxReshardRendezvous,
    unwrap_rendezvous_blob,
)
from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard.transfer_plan import SourceInfo, plan_transfer
from modelexpress_rl.inference.reshard.megatron import (
    MegatronReshardReceiver,
    MegatronTargetLayout,
    MegatronTargetSpec,
    lower_megatron_target,
)
from modelexpress_rl.train.engines.megatron import (
    MegatronAliasInput,
    MegatronPublishedTensorSpec,
    MegatronTensorSpec,
    build_hf_aliases,
    build_megatron_reshard_manifest,
    publish_megatron_reshard_view,
    publish_registered_shard_table,
)


def _bf16_sources() -> tuple[dict[str, SourceInfo], list[torch.Tensor]]:
    tensors = [
        torch.zeros((8, 8), dtype=torch.bfloat16),
        torch.zeros((8, 8), dtype=torch.bfloat16),
        torch.zeros((8, 8), dtype=torch.bfloat16),
        torch.zeros((8, 8), dtype=torch.bfloat16),
        torch.zeros((8,), dtype=torch.bfloat16),
    ]
    sources = {
        "expert_fc1": SourceInfo(
            (16, 8),
            torch.bfloat16,
            2,
            [
                Shard((0, 0), (8, 8), "tp0", tensors[0].data_ptr(), 2),
                Shard((8, 0), (8, 8), "tp1", tensors[1].data_ptr(), 2),
            ],
        ),
        "expert_fc2": SourceInfo(
            (8, 16),
            torch.bfloat16,
            2,
            [
                Shard((0, 0), (8, 8), "tp0", tensors[2].data_ptr(), 2),
                Shard((0, 8), (8, 8), "tp1", tensors[3].data_ptr(), 2),
            ],
        ),
        "norm": SourceInfo(
            (8,),
            torch.bfloat16,
            2,
            [Shard((0,), (8,), "tp0", tensors[4].data_ptr(), 2)],
        ),
    }
    return sources, tensors


def _specs() -> list[MegatronTargetSpec]:
    return [
        MegatronTargetSpec(
            "expert_fc1",
            "expert_column",
            (16, 8),
            torch.bfloat16,
            shard_axis=0,
            descriptor_extras={"expert_layout": "grouped"},
        ),
        MegatronTargetSpec(
            "expert_fc2",
            "expert_row",
            (8, 16),
            torch.bfloat16,
            shard_axis=1,
            descriptor_extras={"expert_layout": "grouped"},
        ),
        MegatronTargetSpec("norm", "replicated", (8,), torch.bfloat16),
    ]


@pytest.mark.parametrize(
    ("tp_size", "tp_rank", "expected_bytes", "expected_segments"),
    [(2, 1, 272, 3), (4, 2, 144, 10)],
)
def test_megatron_target_lowers_to_destination_owned_bytes(
    tp_size: int,
    tp_rank: int,
    expected_bytes: int,
    expected_segments: int,
):
    sources, keepalive = _bf16_sources()
    capture, layouts = lower_megatron_target(
        _specs(), MegatronTargetLayout(tp_size=tp_size, tp_rank=tp_rank)
    )

    plan = plan_transfer(capture, sources)

    assert plan.fallback == []
    assert plan.bytes_planned() == expected_bytes
    assert len(plan.segments) == expected_segments
    assert layouts["expert_fc1"][0] == (16 // tp_size, 8)
    assert layouts["expert_fc2"][0] == (8, 16 // tp_size)
    assert layouts["norm"][0] == (8,)
    assert all(tensor.data_ptr() for tensor in keepalive)


def test_leading_axis_experts_preserve_expert_dimension():
    capture, layouts = lower_megatron_target(
        [
            MegatronTargetSpec(
                "fc1",
                "expert_column",
                (4, 16, 8),
                torch.bfloat16,
                descriptor_extras={"expert_layout": "leading_axis"},
            ),
            MegatronTargetSpec(
                "fc2",
                "expert_row",
                (4, 8, 16),
                torch.bfloat16,
                descriptor_extras={"expert_layout": "leading_axis"},
            ),
        ],
        MegatronTargetLayout(tp_size=4, tp_rank=3),
    )

    assert layouts == {
        "fc1": ((4, 4, 8), torch.bfloat16),
        "fc2": ((4, 8, 4), torch.bfloat16),
    }
    assert capture.copies[0].op_chain == (("narrow", (1, 12, 4), ()),)
    assert capture.copies[1].op_chain == (("narrow", (2, 12, 4), ()),)


def test_non_divisible_target_geometry_fails_closed():
    with pytest.raises(ValueError, match="not divisible"):
        lower_megatron_target(
            [
                MegatronTargetSpec(
                    "column",
                    "column",
                    (10, 8),
                    torch.bfloat16,
                )
            ],
            MegatronTargetLayout(tp_size=4, tp_rank=0),
        )


@pytest.mark.parametrize("extent", [4.5, True])
def test_target_spec_rejects_non_integral_global_shape(extent):
    with pytest.raises(ValueError, match="positive integer extents"):
        MegatronTargetSpec("column", "column", (extent, 8), torch.bfloat16)


@pytest.mark.parametrize(
    ("tp_size", "tp_rank", "message"),
    [
        (2.5, 0, "tp_size must be an integer"),
        (True, 0, "tp_size must be an integer"),
        (2, 0.5, "tp_rank must be an integer"),
        (2, False, "tp_rank must be an integer"),
    ],
)
def test_target_layout_rejects_non_integral_geometry(tp_size, tp_rank, message):
    with pytest.raises(ValueError, match=message):
        MegatronTargetLayout(tp_size=tp_size, tp_rank=tp_rank)


def test_legacy_megatron_publisher_imports_remain_available():
    legacy_aliases = import_module(
        "modelexpress.refit.reshard.megatron_aliases"
    )
    legacy_publisher = import_module(
        "modelexpress.refit.reshard.megatron_publisher"
    )

    assert legacy_aliases.MegatronAliasInput is MegatronAliasInput
    assert legacy_aliases.build_hf_aliases is build_hf_aliases
    assert (
        legacy_publisher.publish_registered_shard_table
        is publish_registered_shard_table
    )


def test_receiver_seam_validates_manifest_and_invokes_installer():
    installed = []
    receiver = object.__new__(MegatronReshardReceiver)
    receiver._target_specs = _specs()
    receiver._target_layout = MegatronTargetLayout(tp_size=4, tp_rank=0)
    receiver._install_native = installed.append
    manifest = [
        ("expert_fc1", torch.bfloat16, (16, 8)),
        ("expert_fc2", torch.bfloat16, (8, 16)),
        ("norm", torch.bfloat16, (8,)),
    ]

    capture, layouts = receiver._capture(manifest)
    buffers = {
        name: torch.empty(shape, dtype=dtype)
        for name, (shape, dtype) in layouts.items()
    }
    receiver._install(buffers)

    assert len(capture.copies) == 3
    assert installed == [buffers]


def test_receiver_seam_rejects_stale_manifest_geometry():
    receiver = object.__new__(MegatronReshardReceiver)
    receiver._target_specs = _specs()
    receiver._target_layout = MegatronTargetLayout(tp_size=2, tp_rank=0)

    with pytest.raises(RuntimeError, match="disagrees"):
        receiver._capture(
            [
                ("expert_fc1", torch.bfloat16, (8, 8)),
                ("expert_fc2", torch.bfloat16, (8, 16)),
                ("norm", torch.bfloat16, (8,)),
            ]
        )


class _Manager:
    agent_name = "trainer-r3"
    nixl_metadata = b"agent-metadata"


class _PublishClient:
    def __init__(self):
        self.worker = None

    def publish_metadata(self, _identity, worker, _worker_id):
        self.worker = worker
        return "source-id"

    def update_status(self, **_kwargs):
        return True


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
    assert (payload.agent_metadata, payload.agent_name, payload.metadata_endpoint) == (
        b"agent-metadata",
        "trainer-r3",
        "10.0.0.3:19003",
    )
    assert payload.tensors[0].shards[0].addr == tensor.data_ptr()
    assert payload.tensors[0].shards[0].shard_offset == (8, 0)


def test_existing_reshard_publisher_remains_compatible():
    client = _PublishClient()
    rendezvous = MxReshardRendezvous(
        client,
        role="trainer",
        rank=3,
        model_name="model",
        worker_id="worker-3",
    )
    tensor = torch.zeros((8, 8), dtype=torch.bfloat16)

    try:
        source_id = publish_megatron_reshard_view(
            manager=_Manager(),
            rendezvous=rendezvous,
            tensors={"column": tensor},
            specs=[
                MegatronPublishedTensorSpec(
                    name="column",
                    global_shape=(16, 8),
                    shard_axis=0,
                    local_shard_range=(8, 16),
                )
            ],
            metadata_endpoint="10.0.0.3:19003",
        )
    finally:
        rendezvous.close()

    assert source_id == "source-id"
    payload = unwrap_rendezvous_blob(client.worker.nixl_metadata)
    assert payload.tensors[0].shards[0].addr == tensor.data_ptr()


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
