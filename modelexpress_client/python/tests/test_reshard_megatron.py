# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from modelexpress.refit.reshard.megatron import (
    MegatronTargetLayout,
    MegatronTargetSpec,
    lower_megatron_target,
)
from modelexpress.refit.reshard.megatron_receiver import MegatronReshardReceiver
from modelexpress.refit.reshard import megatron_aliases
from modelexpress.refit.reshard.megatron_aliases import (
    MegatronAliasInput,
    build_hf_aliases,
)
from modelexpress.refit.reshard.megatron_publisher import (
    MegatronPublishedTensorSpec,
    publish_megatron_reshard_view,
)
from modelexpress.refit.reshard.rendezvous import unwrap_rendezvous_blob
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
    buffers = {name: torch.empty(shape, dtype=dtype) for name, (shape, dtype) in layouts.items()}
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


def test_publisher_seam_reuses_registered_tensor_addresses():
    class Manager:
        agent_name = "trainer-r3"
        nixl_metadata = b"agent-metadata"

    class Client:
        def __init__(self):
            self.worker = None

        def publish_metadata(self, _identity, worker, _worker_id):
            self.worker = worker
            return "source-id"

    client = Client()
    tensor = torch.zeros((8, 8), dtype=torch.bfloat16)

    source_id = publish_megatron_reshard_view(
        manager=Manager(),
        client=client,
        model_name="model",
        worker_rank=3,
        worker_id="worker-3",
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


def _alias_items(fused, qkv, plain):
    """One item per alias construction path: gated split, qkv split, simple."""
    return [
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
        ),
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
        ),
        MegatronAliasInput(
            name="norm.weight",
            tensor=plain,
            role="replicated",
            hf_names=("norm.weight",),
            global_shape=(4,),
            placement_kind="REPLICATE",
            shard_axis=None,
            local_shard_range=None,
        ),
    ]


def test_every_alias_path_publishes_a_digest_when_verify_is_on(monkeypatch):
    """Regression: the digest was added to megatron_publisher only, so aliased
    tensors published none. On a fused-QKV gated-MLP MoE that is most of the
    model, and the receiver's verify gate reported skipped_no_digest=98304 -
    reporting *no evidence* rather than failing, which is why it went unnoticed.
    """
    monkeypatch.setattr(megatron_aliases, "VERIFY", True)
    published = build_hf_aliases(
        _alias_items(
            torch.arange(32, dtype=torch.bfloat16).reshape(8, 4),
            torch.arange(48, dtype=torch.bfloat16).reshape(12, 4),
            torch.arange(4, dtype=torch.bfloat16),
        ),
        agent_name="trainer-tp1",
    )

    assert len(published) == 6  # gate, up, q, k, v, norm
    for tensor in published:
        for shard in tensor.shards:
            assert shard.digest is not None, f"{tensor.name} published no digest"


def test_alias_digests_are_taken_over_the_view_not_the_parent(monkeypatch):
    """q/k/v are narrow() views of one fused tensor. Digesting the parent would
    give all three the same value and silently accept a q/k/v mix-up."""
    monkeypatch.setattr(megatron_aliases, "VERIFY", True)
    qkv = torch.arange(48, dtype=torch.bfloat16).reshape(12, 4)

    q, k, v = build_hf_aliases(
        [_alias_items(torch.zeros(8, 4), qkv, torch.zeros(4))[1]],
        agent_name="trainer-tp1",
    )

    digests = {q.shards[0].digest, k.shards[0].digest, v.shards[0].digest}
    assert len(digests) == 3, "q/k/v share a digest, so the parent was digested"


def test_no_digest_is_published_when_verify_is_off(monkeypatch):
    """The digest costs a reduction per tensor, so it stays opt-in, and the blob
    must stay byte-identical to the pre-digest schema for a mixed fleet."""
    monkeypatch.setattr(megatron_aliases, "VERIFY", False)
    published = build_hf_aliases(
        _alias_items(
            torch.arange(32, dtype=torch.bfloat16).reshape(8, 4),
            torch.arange(48, dtype=torch.bfloat16).reshape(12, 4),
            torch.arange(4, dtype=torch.bfloat16),
        ),
        agent_name="trainer-tp1",
    )

    for tensor in published:
        for shard in tensor.shards:
            assert shard.digest is None
