# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from modelexpress import p2p_pb2
from modelexpress.integrations.verl_checkpoint_engine import (
    _ModelExpressCheckpointEngineMixin,
    _model_version_from_global_steps,
    _topology_from_metadata,
)


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


def test_init_rejects_unknown_topology():
    with pytest.raises(ValueError, match="unsupported ModelExpress veRL topology"):
        _ModelExpressCheckpointEngineMixin(
            bucket_size=1,
            model_name="test-model",
            topology="unknown",
        )
