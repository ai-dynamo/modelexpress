# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from modelexpress import p2p_pb2
from modelexpress.integrations.verl_checkpoint_engine import (
    _ModelExpressCheckpointEngineMixin,
    _model_version_from_global_steps,
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
    }
    assert rollout_kwargs == {
        "rank": [1, 2],
        "world_size": [3, 3],
    }


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
