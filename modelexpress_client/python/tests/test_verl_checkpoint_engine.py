# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
import torch

from modelexpress import p2p_pb2
from modelexpress.integrations.verl_checkpoint_engine import (
    _ModelExpressCheckpointEngineMixin,
    _allocate_tensors_from_shape_registry,
    _backend_framework_value,
    _build_base_identity,
    _identity_matches_base,
    _model_version_from_global_steps,
    _shape_registry_from_tensors,
    _torch_dtype_from_string,
)
from modelexpress.rl_metadata import RlSourceMetadata, RlSourceRole, with_rl_source_metadata


class _FakeMxClient:
    def __init__(self, response):
        self.responses = list(response) if isinstance(response, list) else [response]
        self.list_call_count = 0
        self.status_updates = []

    def list_sources(self, identity=None, status_filter=None):
        self.identity = identity
        self.status_filter = status_filter
        self.list_call_count += 1
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]

    def update_status(self, mx_source_id, worker_id, worker_rank, status):
        self.status_updates.append((mx_source_id, worker_id, worker_rank, status))
        return True


def _base_identity(**overrides):
    values = {
        "model_name": "test-model",
        "mx_version": "0.3.0",
        "backend_framework": "vllm",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 0,
        "dtype": "bfloat16",
        "quantization": "",
        "revision": "",
    }
    values.update(overrides)
    return _build_base_identity(**values)


def test_shape_registry_round_trip_allocates_tensors_on_requested_device():
    tensors = {
        "w1": torch.zeros((2, 3), dtype=torch.bfloat16),
        "w2": torch.ones((1,), dtype=torch.float32),
    }

    registry = _shape_registry_from_tensors(tensors)
    allocated = _allocate_tensors_from_shape_registry(registry, device="cpu")

    assert allocated["w1"].shape == (2, 3)
    assert allocated["w1"].dtype == torch.bfloat16
    assert allocated["w1"].device.type == "cpu"
    assert allocated["w2"].shape == (1,)
    assert allocated["w2"].dtype == torch.float32


def test_torch_dtype_from_string_accepts_torch_prefix():
    assert _torch_dtype_from_string("torch.bfloat16") is torch.bfloat16
    assert _torch_dtype_from_string("float16") is torch.float16


def test_torch_dtype_from_string_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="unsupported tensor dtype"):
        _torch_dtype_from_string("not_a_dtype")


def test_model_version_prefers_global_steps_and_falls_back_to_counter():
    assert _model_version_from_global_steps(42, current_version=7) == 42
    assert _model_version_from_global_steps(None, current_version=7) == 8


def test_build_base_identity_uses_expected_backend_framework():
    identity = _base_identity(tensor_parallel_size=2, revision="abc123")

    assert identity.model_name == "test-model"
    assert identity.backend_framework == p2p_pb2.BACKEND_FRAMEWORK_VLLM
    assert identity.tensor_parallel_size == 2
    assert identity.revision == "abc123"


def test_build_base_identity_requires_model_name():
    with pytest.raises(ValueError, match="requires model_name"):
        _build_base_identity(
            model_name="",
            mx_version="0.3.0",
            backend_framework="vllm",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            expert_parallel_size=0,
            dtype="bfloat16",
            quantization="",
            revision="",
        )


def test_backend_framework_value_rejects_unknown_framework():
    assert _backend_framework_value("trtllm") == p2p_pb2.BACKEND_FRAMEWORK_TRT_LLM

    with pytest.raises(ValueError, match="unsupported backend_framework"):
        _backend_framework_value("unknown")


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


def test_select_source_uses_broad_list_sources_and_requested_version():
    base_identity = _base_identity()
    identity = with_rl_source_metadata(
        base_identity,
        RlSourceMetadata(
            model_version=5,
            role=RlSourceRole.TRAINER,
            world_size=1,
            shape_registry={"w": {"shape": [1], "dtype": "torch.float32"}},
        ),
    )
    response = p2p_pb2.ListSourcesResponse(
        instances=[
            p2p_pb2.SourceInstanceRef(
                mx_source_id="source-v5",
                worker_id="worker-v5",
                model_name="test-model",
                worker_rank=0,
                identity=identity,
            )
        ]
    )
    fake_client = _FakeMxClient(response)
    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        mx_client=fake_client,
    )
    engine.init_process_group(rank=2, world_size=3)

    candidate = engine._select_source(5)

    assert candidate.mx_source_id == "source-v5"
    assert candidate.worker_id == "worker-v5"
    assert fake_client.identity is None
    assert fake_client.status_filter == p2p_pb2.SOURCE_STATUS_READY


def test_select_source_ignores_different_base_identity():
    base_identity = _base_identity()
    wrong_identity = with_rl_source_metadata(
        _base_identity(dtype="float16"),
        RlSourceMetadata(
            model_version=5,
            role=RlSourceRole.TRAINER,
            world_size=1,
            shape_registry={"w": {"shape": [1], "dtype": "torch.float32"}},
        ),
    )
    right_identity = with_rl_source_metadata(
        base_identity,
        RlSourceMetadata(
            model_version=5,
            role=RlSourceRole.TRAINER,
            world_size=1,
            shape_registry={"w": {"shape": [1], "dtype": "torch.float32"}},
        ),
    )
    response = p2p_pb2.ListSourcesResponse(
        instances=[
            p2p_pb2.SourceInstanceRef(
                mx_source_id="source-wrong",
                worker_id="worker-a",
                model_name="test-model",
                worker_rank=0,
                identity=wrong_identity,
            ),
            p2p_pb2.SourceInstanceRef(
                mx_source_id="source-right",
                worker_id="worker-z",
                model_name="test-model",
                worker_rank=0,
                identity=right_identity,
            ),
        ]
    )
    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        mx_client=_FakeMxClient(response),
    )
    engine.init_process_group(rank=2, world_size=3)

    candidate = engine._select_source(5)

    assert candidate.mx_source_id == "source-right"
    assert _identity_matches_base(right_identity, base_identity)
    assert not _identity_matches_base(wrong_identity, base_identity)


def test_wait_for_source_retries_until_source_is_published():
    identity = with_rl_source_metadata(
        _base_identity(),
        RlSourceMetadata(
            model_version=5,
            role=RlSourceRole.TRAINER,
            world_size=1,
            shape_registry={"w": {"shape": [1], "dtype": "torch.float32"}},
        ),
    )
    empty = p2p_pb2.ListSourcesResponse()
    ready = p2p_pb2.ListSourcesResponse(
        instances=[
            p2p_pb2.SourceInstanceRef(
                mx_source_id="source-v5",
                worker_id="worker-v5",
                model_name="test-model",
                worker_rank=0,
                identity=identity,
            )
        ]
    )
    fake_client = _FakeMxClient([empty, ready])
    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        mx_client=fake_client,
        timeout_seconds=1.0,
    )
    engine.init_process_group(rank=1, world_size=2)

    candidate = asyncio.run(engine._wait_for_source(5))

    assert candidate.mx_source_id == "source-v5"
    assert fake_client.list_call_count == 2


def test_finalize_marks_published_source_stale():
    fake_client = _FakeMxClient(p2p_pb2.ListSourcesResponse())
    engine = _ModelExpressCheckpointEngineMixin(
        bucket_size=1,
        model_name="test-model",
        mx_client=fake_client,
    )
    engine._mx_source_id = "source-v5"

    engine.finalize()

    assert fake_client.status_updates == [
        ("source-v5", engine._worker_id, 0, p2p_pb2.SOURCE_STATUS_STALE)
    ]
    assert engine._mx_source_id is None
