# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
import torch

from modelexpress import p2p_pb2
from modelexpress.integrations.verl_checkpoint_engine import (
    _ModelExpressCheckpointEngineMixin,
    _model_version_from_global_steps,
    _normalize_source_role_policy,
    _topology_from_metadata,
)
from modelexpress.rl_metadata import RlSourceRole
from modelexpress.rl_transfer_lease import RlTransferLeaseInventory
from modelexpress.rl_transfer_report import RlTransferAttempt, RlTransferReport


async def _collect_weights(weights):
    return [item async for item in weights]


class _FakeCudaTensor:
    is_cuda = True

    def detach(self):
        return self

    def is_contiguous(self):
        return True


class _FakePublishingTransfer:
    def __init__(self):
        self.tensors = None
        self.kwargs = None

    def publish_tensors(self, tensors, **kwargs):
        self.tensors = tensors
        self.kwargs = kwargs


def test_model_version_prefers_global_steps_and_falls_back_to_counter():
    assert _model_version_from_global_steps(42, current_version=7) == 42
    assert _model_version_from_global_steps(None, current_version=7) == 8


def test_build_topology_matches_verl_checkpoint_engine_shape():
    trainer_kwargs, rollout_kwargs = _ModelExpressCheckpointEngineMixin.build_topology(
        trainer_world_size=3,
        rollout_world_size=2,
        metadata=[],
    )

    assert trainer_kwargs == {
        "rank": [0, -1, -1],
        "world_size": [3, 3, 3],
        "is_trainer": [True, True, True],
        "receiver_rank": [None, None, None],
        "source_world_size": [1, 1, 1],
    }
    assert rollout_kwargs == {
        "rank": [1, 2],
        "world_size": [3, 3],
        "is_trainer": [False, False],
        "receiver_rank": [0, 1],
        "source_world_size": [1, 1],
    }


def test_build_topology_supports_rank_local_mode():
    trainer_kwargs, rollout_kwargs = _ModelExpressCheckpointEngineMixin.build_topology(
        trainer_world_size=2,
        rollout_world_size=2,
        metadata=[{"modelexpress_topology": "rank_local"}],
    )

    assert trainer_kwargs == {
        "rank": [0, 1],
        "world_size": [2, 2],
        "is_trainer": [True, True],
        "receiver_rank": [None, None],
        "source_world_size": [2, 2],
    }
    assert rollout_kwargs == {
        "rank": [0, 1],
        "world_size": [2, 2],
        "is_trainer": [False, False],
        "receiver_rank": [0, 1],
        "source_world_size": [2, 2],
    }


def test_rank_local_topology_requires_matching_world_sizes():
    with pytest.raises(ValueError, match="requires equal trainer and rollout world sizes"):
        _ModelExpressCheckpointEngineMixin.build_topology(
            trainer_world_size=2,
            rollout_world_size=3,
            metadata=[{"modelexpress_topology": "rank_local"}],
        )


def test_topology_metadata_defaults_to_broadcast():
    assert _topology_from_metadata([]) == "broadcast"


def test_topology_metadata_accepts_tree_fanout():
    assert _topology_from_metadata([{"modelexpress_topology": "tree_fanout"}]) == (
        "tree_fanout"
    )


def test_topology_metadata_rejects_conflicting_values():
    with pytest.raises(ValueError, match="conflicting ModelExpress veRL topologies"):
        _topology_from_metadata(
            [
                {"modelexpress_topology": "broadcast"},
                {"modelexpress_topology": "rank_local"},
            ]
        )


def test_topology_metadata_rejects_unknown_values():
    with pytest.raises(ValueError, match="unsupported ModelExpress veRL topology"):
        _topology_from_metadata([{"modelexpress_topology": "unknown"}])


def test_init_requires_model_name():
    with pytest.raises(ValueError, match="requires model_name"):
        _ModelExpressCheckpointEngineMixin(bucket_size=1)


def test_init_builds_source_identity_and_transfer_session():
    mx_client = object()

    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        backend_framework="sglang",
        tensor_parallel_size=2,
        mx_client=mx_client,
    )

    assert engine.base_identity.model_name == "test-model"
    assert engine.base_identity.backend_framework == p2p_pb2.BACKEND_FRAMEWORK_SGLANG
    assert engine.base_identity.tensor_parallel_size == 2
    assert engine._transfer.mx_client is mx_client
    assert engine._transfer.base_identity is engine.base_identity


def test_init_process_group_records_rank_local_roles():
    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        topology="rank_local",
        mx_client=object(),
    )

    engine.init_process_group(
        rank=1,
        world_size=2,
        is_trainer=False,
        receiver_rank=1,
        source_world_size=2,
    )

    assert engine._is_trainer is False
    assert engine._receiver_rank == 1
    assert engine._source_world_size == 2
    assert engine.same_rank_only is True
    assert engine._replica_world_size_for_publish() == 2


def test_init_rejects_unknown_topology():
    with pytest.raises(ValueError, match="unsupported ModelExpress veRL topology"):
        _ModelExpressCheckpointEngineMixin(
            bucket_size=1,
            model_name="test-model",
            topology="unknown",
        )


def test_tree_fanout_defaults_to_republish_received():
    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        topology="tree_fanout",
        tree_fanout=3,
        mx_client=object(),
    )

    assert engine.republish_received is True
    assert engine.same_rank_only is False
    assert engine.prepare() == {"modelexpress_topology": "tree_fanout"}


def test_tree_fanout_requires_republish_received():
    with pytest.raises(ValueError, match="requires republish_received"):
        _ModelExpressCheckpointEngineMixin(
            bucket_size=1,
            model_name="test-model",
            topology="tree_fanout",
            republish_received=False,
            mx_client=object(),
        )


def test_source_role_policy_defaults_to_trainer_with_replica_fallback(monkeypatch):
    monkeypatch.delenv("MX_RL_SOURCE_ROLE_POLICY", raising=False)

    assert _normalize_source_role_policy(None) == (
        RlSourceRole.TRAINER,
        RlSourceRole.INFERENCE_REPLICA,
    )


def test_source_role_policy_accepts_replica_first():
    assert _normalize_source_role_policy("replica_first") == (
        RlSourceRole.INFERENCE_REPLICA,
        RlSourceRole.TRAINER,
    )


def test_source_role_policy_rejects_unknown_values():
    with pytest.raises(ValueError, match="unsupported ModelExpress veRL source role policy"):
        _normalize_source_role_policy("unknown")


def test_receive_weights_without_global_steps_requests_latest_and_republishes():
    class _FakeTransfer:
        def __init__(self):
            self.republish_kwargs = None

        async def receive_tensors_and_publish_replica(self, **kwargs):
            self.republish_kwargs = kwargs
            return [("w", torch.zeros(1))]

        async def receive_tensors(self, **kwargs):
            raise AssertionError("receive_tensors should not run when republish is enabled")

    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        republish_received=True,
        mx_client=object(),
    )
    fake_transfer = _FakeTransfer()
    engine._transfer = fake_transfer
    engine.init_process_group(
        rank=2,
        world_size=4,
        is_trainer=False,
        receiver_rank=1,
        source_world_size=1,
    )

    weights = asyncio.run(_collect_weights(engine.receive_weights(global_steps=None)))

    assert weights[0][0] == "w"
    assert fake_transfer.republish_kwargs == {
        "model_version": None,
        "receiver_rank": 1,
        "same_rank_only": False,
        "roles": (RlSourceRole.TRAINER, RlSourceRole.INFERENCE_REPLICA),
        "replica_world_size": 3,
    }


def test_receive_weights_runs_lifecycle_hooks_around_verl_refit():
    events = []

    class _FakeTransfer:
        async def receive_tensors(self, **kwargs):
            events.append(("receive", kwargs))
            return [("w", torch.zeros(1))]

    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        pause_generation=lambda: events.append("pause"),
        flush_cache=lambda: events.append("flush"),
        resume_generation=lambda: events.append("resume"),
        mx_client=object(),
    )
    engine._transfer = _FakeTransfer()
    engine.init_process_group(
        rank=2,
        world_size=4,
        is_trainer=False,
        receiver_rank=1,
        source_world_size=1,
    )

    weights = asyncio.run(_collect_weights(engine.receive_weights(global_steps=None)))

    assert weights[0][0] == "w"
    assert events == [
        "pause",
        (
            "receive",
            {
                "model_version": None,
                "receiver_rank": 1,
                "same_rank_only": False,
                "roles": (RlSourceRole.TRAINER, RlSourceRole.INFERENCE_REPLICA),
            },
        ),
        "flush",
        "resume",
    ]


def test_transfer_lease_summary_uses_receive_report_and_inventory():
    report = RlTransferReport(
        requested_model_version=None,
        resolved_model_version=7,
        receiver_rank=1,
        attempts=(
            RlTransferAttempt(
                mx_source_id="source-a",
                worker_id="worker-a",
                worker_rank=0,
                role=RlSourceRole.TRAINER,
                model_version=7,
                success=True,
                lease_id="lease-a",
            ),
        ),
    )

    class _FakeTransfer:
        def __init__(self):
            self.last_receive_report = report
            self.list_kwargs = None

        def list_target_transfer_leases(self, **kwargs):
            self.list_kwargs = kwargs
            return RlTransferLeaseInventory(
                target_worker_id="verl-worker",
                leases=(
                    p2p_pb2.TransferLease(
                        lease_id="lease-a",
                        status=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
                    ),
                ),
            )

    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        mx_client=object(),
    )
    fake_transfer = _FakeTransfer()
    engine._transfer = fake_transfer

    summary = engine.transfer_lease_summary(
        mx_source_id="source-a",
        statuses=(p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,),
    )

    assert engine.last_receive_report is report
    assert fake_transfer.list_kwargs == {
        "mx_source_id": "source-a",
        "statuses": (p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,),
    }
    assert summary.report is report
    assert [lease.lease_id for lease in summary.matching_leases] == ["lease-a"]


def test_finalize_retains_sources_by_default_for_verl_manager_finalize():
    class _FakeTransfer:
        def __init__(self):
            self.finalize_called = False
            self.finalize_receive_state_called = False

        def finalize(self):
            self.finalize_called = True

        def finalize_receive_state(self):
            self.finalize_receive_state_called = True

    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        mx_client=object(),
    )
    fake_transfer = _FakeTransfer()
    engine._transfer = fake_transfer

    engine.finalize()

    assert fake_transfer.finalize_receive_state_called
    assert not fake_transfer.finalize_called


def test_finalize_can_stale_sources_when_configured():
    class _FakeTransfer:
        def __init__(self):
            self.finalize_called = False
            self.finalize_receive_state_called = False

        def finalize(self):
            self.finalize_called = True

        def finalize_receive_state(self):
            self.finalize_receive_state_called = True

    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        retain_sources_on_finalize=False,
        mx_client=object(),
    )
    fake_transfer = _FakeTransfer()
    engine._transfer = fake_transfer

    engine.finalize()

    assert fake_transfer.finalize_called
    assert not fake_transfer.finalize_receive_state_called


def test_mark_current_source_stale_delegates_to_transfer():
    class _FakeTransfer:
        def __init__(self):
            self.marked_stale = False

        def mark_current_source_stale(self):
            self.marked_stale = True

    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        mx_client=object(),
    )
    fake_transfer = _FakeTransfer()
    engine._transfer = fake_transfer

    engine.mark_current_source_stale()

    assert fake_transfer.marked_stale


def test_receive_weights_validates_before_pause_hook():
    events = []
    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        pause_generation=lambda: events.append("pause"),
        resume_generation=lambda: events.append("resume"),
        mx_client=object(),
    )

    with pytest.raises(RuntimeError, match="init_process_group"):
        asyncio.run(_collect_weights(engine.receive_weights(global_steps=None)))

    assert events == []


def test_receive_weights_uses_explicit_replica_world_size():
    class _FakeTransfer:
        def __init__(self):
            self.republish_kwargs = None

        async def receive_tensors_and_publish_replica(self, **kwargs):
            self.republish_kwargs = kwargs
            return [("w", torch.zeros(1))]

    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        republish_received="true",
        topology="rank_local",
        mx_client=object(),
    )
    fake_transfer = _FakeTransfer()
    engine._transfer = fake_transfer
    engine.init_process_group(
        rank=1,
        world_size=2,
        is_trainer=False,
        receiver_rank=1,
        source_world_size=2,
        replica_world_size=4,
    )

    asyncio.run(_collect_weights(engine.receive_weights(global_steps=8)))

    assert fake_transfer.republish_kwargs["replica_world_size"] == 4
    assert fake_transfer.republish_kwargs["model_version"] == 8
    assert fake_transfer.republish_kwargs["roles"] == (
        RlSourceRole.TRAINER,
        RlSourceRole.INFERENCE_REPLICA,
    )


def test_receive_weights_can_prefer_replica_sources():
    class _FakeTransfer:
        def __init__(self):
            self.receive_kwargs = None

        async def receive_tensors(self, **kwargs):
            self.receive_kwargs = kwargs
            return [("w", torch.zeros(1))]

    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        source_role_policy="replica_first",
        mx_client=object(),
    )
    fake_transfer = _FakeTransfer()
    engine._transfer = fake_transfer
    engine.init_process_group(
        rank=2,
        world_size=4,
        is_trainer=False,
        receiver_rank=1,
        source_world_size=1,
    )

    asyncio.run(_collect_weights(engine.receive_weights(global_steps=8)))

    assert fake_transfer.receive_kwargs == {
        "model_version": 8,
        "receiver_rank": 1,
        "same_rank_only": False,
        "roles": (RlSourceRole.INFERENCE_REPLICA, RlSourceRole.TRAINER),
    }


def test_tree_fanout_receive_uses_parent_replica_policy():
    class _FakeTransfer:
        def __init__(self):
            self.republish_kwargs = None

        async def receive_tensors_and_publish_replica(self, **kwargs):
            self.republish_kwargs = kwargs
            return [("w", torch.zeros(1))]

    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        topology="tree_fanout",
        tree_fanout=2,
        mx_client=object(),
    )
    fake_transfer = _FakeTransfer()
    engine._transfer = fake_transfer
    engine.init_process_group(
        rank=3,
        world_size=5,
        is_trainer=False,
        receiver_rank=2,
        source_world_size=1,
    )

    asyncio.run(_collect_weights(engine.receive_weights(global_steps=None)))

    assert fake_transfer.republish_kwargs == {
        "model_version": None,
        "receiver_rank": 2,
        "same_rank_only": False,
        "roles": (RlSourceRole.INFERENCE_REPLICA,),
        "source_ranks_by_role": {RlSourceRole.INFERENCE_REPLICA: (0,)},
        "require_complete_version": False,
        "replica_world_size": 4,
    }


def test_tree_fanout_first_wave_receives_from_trainer_root():
    class _FakeTransfer:
        def __init__(self):
            self.republish_kwargs = None

        async def receive_tensors_and_publish_replica(self, **kwargs):
            self.republish_kwargs = kwargs
            return [("w", torch.zeros(1))]

    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        topology="tree_fanout",
        tree_fanout=2,
        mx_client=object(),
    )
    fake_transfer = _FakeTransfer()
    engine._transfer = fake_transfer
    engine.init_process_group(
        rank=1,
        world_size=5,
        is_trainer=False,
        receiver_rank=0,
        source_world_size=1,
    )

    asyncio.run(_collect_weights(engine.receive_weights(global_steps=8)))

    assert fake_transfer.republish_kwargs == {
        "model_version": 8,
        "receiver_rank": 0,
        "same_rank_only": False,
        "roles": (RlSourceRole.TRAINER,),
        "source_ranks_by_role": {RlSourceRole.TRAINER: (0,)},
        "require_complete_version": True,
        "replica_world_size": 4,
    }


def test_send_weights_passes_tensor_metadata_to_transfer():
    tensor_metadata = {
        "experts.w": {
            "expert_ids": [0, 1],
            "expert_axis": 0,
        }
    }
    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        tensor_metadata=tensor_metadata,
        mx_client=object(),
    )
    fake_transfer = _FakePublishingTransfer()
    engine._transfer = fake_transfer
    engine.init_process_group(rank=0, world_size=1, is_trainer=True)

    asyncio.run(engine.send_weights([("experts.w", _FakeCudaTensor())], global_steps=9))

    assert fake_transfer.tensors is not None
    assert list(fake_transfer.tensors) == ["experts.w"]
    assert fake_transfer.kwargs == {
        "model_version": 9,
        "worker_rank": 0,
        "source_world_size": 1,
        "tensor_metadata": tensor_metadata,
    }


def test_send_weights_accepts_inline_tensor_metadata():
    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        mx_client=object(),
    )
    fake_transfer = _FakePublishingTransfer()
    engine._transfer = fake_transfer
    engine.init_process_group(rank=0, world_size=1, is_trainer=True)

    asyncio.run(
        engine.send_weights(
            [
                (
                    "experts.w",
                    _FakeCudaTensor(),
                    {"expert_ids": [0, 1]},
                )
            ],
            global_steps=9,
        )
    )

    assert fake_transfer.kwargs["tensor_metadata"] == {
        "experts.w": {"expert_ids": [0, 1]}
    }


def test_send_weights_merges_configured_and_inline_tensor_metadata():
    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        tensor_metadata={
            "experts.w": {
                "global_shape": [4, 2],
                "shard_offsets": [0, 0],
            }
        },
        mx_client=object(),
    )
    fake_transfer = _FakePublishingTransfer()
    engine._transfer = fake_transfer
    engine.init_process_group(rank=0, world_size=1, is_trainer=True)

    asyncio.run(
        engine.send_weights(
            [
                (
                    "experts.w",
                    _FakeCudaTensor(),
                    {"expert_ids": [0, 1]},
                )
            ],
            global_steps=9,
        )
    )

    assert fake_transfer.kwargs["tensor_metadata"] == {
        "experts.w": {
            "global_shape": [4, 2],
            "shard_offsets": [0, 0],
            "expert_ids": [0, 1],
        }
    }


def test_send_weights_rejects_invalid_inline_tensor_metadata():
    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        mx_client=object(),
    )
    fake_transfer = _FakePublishingTransfer()
    engine._transfer = fake_transfer
    engine.init_process_group(rank=0, world_size=1, is_trainer=True)

    with pytest.raises(ValueError, match="per-tensor metadata"):
        asyncio.run(
            engine.send_weights(
                [("experts.w", object(), "bad")],
                global_steps=9,
            )
        )
    assert fake_transfer.kwargs is None
