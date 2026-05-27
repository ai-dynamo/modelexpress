# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
import torch

from modelexpress import p2p_pb2
from modelexpress.integrations.verl_checkpoint_engine import (
    _ModelExpressCheckpointEngineMixin,
    _model_version_from_global_steps,
    _topology_from_metadata,
)


async def _collect_weights(weights):
    return [item async for item in weights]


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


def test_receive_weights_can_republish_received_replica():
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
        "model_version": 0,
        "receiver_rank": 1,
        "same_rank_only": False,
        "replica_world_size": 3,
    }


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
