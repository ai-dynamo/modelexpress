# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from modelexpress import p2p_pb2
from modelexpress.refit.reshard.megatron import (
    MegatronTargetLayout,
    MegatronTargetSpec,
    lower_megatron_target,
)
from modelexpress.refit.reshard.megatron_aliases import (
    MegatronAliasInput,
    build_hf_aliases,
)
from modelexpress.refit.reshard.megatron_publisher import (
    MegatronPublishedTensorSpec,
    publish_megatron_reshard_view,
)
from modelexpress.refit.reshard.megatron_receiver import MegatronReshardReceiver
from modelexpress.refit.reshard.rendezvous import (
    MxReshardRendezvous,
    unwrap_rendezvous_blob,
)
from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard.transfer_plan import SourceInfo, plan_transfer


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


class _PublishClient:
    """Records the published worker record and accepts heartbeat status updates."""

    def __init__(self):
        self.worker = None
        self.status_updates = []

    def publish_metadata(self, _identity, worker, _worker_id):
        self.worker = worker
        return "source-id"

    def update_status(self, **kwargs):
        self.status_updates.append(kwargs)
        return True


class _Manager:
    agent_name = "trainer-r3"
    nixl_metadata = b"agent-metadata"


def _trainer_rendezvous(client):
    return MxReshardRendezvous(
        client, role="trainer", rank=3, model_name="model", worker_id="worker-3"
    )


def test_publisher_seam_reuses_registered_tensor_addresses():
    client = _PublishClient()
    rendezvous = _trainer_rendezvous(client)
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
        # The caller owns the rendezvous precisely so its heartbeat can be stopped.
        rendezvous.close()

    assert source_id == "source-id"
    assert client.worker.status > 0
    agent_metadata, agent_name, endpoint, published = unwrap_rendezvous_blob(
        client.worker.nixl_metadata
    )
    assert (agent_metadata, agent_name, endpoint) == (
        b"agent-metadata",
        "trainer-r3",
        "10.0.0.3:19003",
    )
    assert published[0].shards[0].addr == tensor.data_ptr()
    assert published[0].shards[0].shard_offset == (8, 0)


def test_publishing_leaves_the_heartbeat_with_its_owner():
    """Publishing starts the source's READY heartbeat. A rendezvous built inside the
    seam would leave that thread running with no handle to stop it, so the source
    would only be marked stale at interpreter exit."""
    client = _PublishClient()
    rendezvous = _trainer_rendezvous(client)

    publish_megatron_reshard_view(
        manager=_Manager(),
        rendezvous=rendezvous,
        tensors={"column": torch.zeros((8, 8), dtype=torch.bfloat16)},
        specs=[MegatronPublishedTensorSpec(name="column", global_shape=(8, 8))],
        metadata_endpoint="10.0.0.3:19003",
    )
    rendezvous.close()

    assert client.status_updates[-1]["status"] == p2p_pb2.SOURCE_STATUS_STALE
    assert client.status_updates[-1]["worker_id"] == "worker-3"


def test_publisher_seam_rejects_duplicate_spec_names():
    """Last-writer-wins would publish one spec's shard description under a name the
    other spec owns, and comparing key sets against the tensors cannot see it."""

    class Client:
        def publish_metadata(self, *_args, **_kwargs):
            raise AssertionError("publication must not run")

    spec = MegatronPublishedTensorSpec(name="column", global_shape=(8, 8))

    with pytest.raises(ValueError, match="duplicate Megatron publish spec"):
        publish_megatron_reshard_view(
            manager=_Manager(),
            rendezvous=_trainer_rendezvous(Client()),
            tensors={"column": torch.zeros((8, 8), dtype=torch.bfloat16)},
            specs=[spec, spec],
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
