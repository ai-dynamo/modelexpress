# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
import torch

from modelexpress import p2p_pb2
from modelexpress.rl_metadata import RlSourceMetadata, RlSourceRole, with_rl_source_metadata
from modelexpress.rl_transfer import (
    RlNixlWeightTransfer,
    allocate_tensors_from_shape_registry,
    backend_framework_value,
    build_rl_base_identity,
    identity_matches_base,
    shape_registry_from_tensors,
    torch_dtype_from_string,
)


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
    return build_rl_base_identity(**values)


def _transfer(mx_client):
    return RlNixlWeightTransfer(
        mx_client=mx_client,
        base_identity=_base_identity(),
        worker_id="worker-local",
    )


def _source_ref(
    mx_source_id: str,
    worker_id: str,
    *,
    model_version: int = 5,
    role: RlSourceRole = RlSourceRole.TRAINER,
    worker_rank: int = 0,
):
    identity = with_rl_source_metadata(
        _base_identity(),
        RlSourceMetadata(
            model_version=model_version,
            role=role,
            world_size=1,
            shape_registry={"w": {"shape": [1], "dtype": "torch.float32"}},
        ),
    )
    return p2p_pb2.SourceInstanceRef(
        mx_source_id=mx_source_id,
        worker_id=worker_id,
        model_name="test-model",
        worker_rank=worker_rank,
        identity=identity,
    )


def test_shape_registry_round_trip_allocates_tensors_on_requested_device():
    tensors = {
        "w1": torch.zeros((2, 3), dtype=torch.bfloat16),
        "w2": torch.ones((1,), dtype=torch.float32),
    }

    registry = shape_registry_from_tensors(tensors)
    allocated = allocate_tensors_from_shape_registry(registry, device="cpu")

    assert allocated["w1"].shape == (2, 3)
    assert allocated["w1"].dtype == torch.bfloat16
    assert allocated["w1"].device.type == "cpu"
    assert allocated["w2"].shape == (1,)
    assert allocated["w2"].dtype == torch.float32


def test_torch_dtype_from_string_accepts_torch_prefix():
    assert torch_dtype_from_string("torch.bfloat16") is torch.bfloat16
    assert torch_dtype_from_string("float16") is torch.float16


def test_torch_dtype_from_string_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="unsupported tensor dtype"):
        torch_dtype_from_string("not_a_dtype")


def test_build_base_identity_uses_expected_backend_framework():
    identity = _base_identity(tensor_parallel_size=2, revision="abc123")

    assert identity.model_name == "test-model"
    assert identity.backend_framework == p2p_pb2.BACKEND_FRAMEWORK_VLLM
    assert identity.tensor_parallel_size == 2
    assert identity.revision == "abc123"


def test_build_base_identity_requires_model_name():
    with pytest.raises(ValueError, match="requires model_name"):
        build_rl_base_identity(
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
    assert backend_framework_value("trtllm") == p2p_pb2.BACKEND_FRAMEWORK_TRT_LLM

    with pytest.raises(ValueError, match="unsupported backend_framework"):
        backend_framework_value("unknown")


def test_select_source_uses_broad_list_sources_and_requested_version():
    response = p2p_pb2.ListSourcesResponse(
        instances=[_source_ref("source-v5", "worker-v5")]
    )
    fake_client = _FakeMxClient(response)

    candidate = _transfer(fake_client).select_source(
        model_version=5,
        receiver_rank=1,
    )

    assert candidate.mx_source_id == "source-v5"
    assert candidate.worker_id == "worker-v5"
    assert fake_client.identity is None
    assert fake_client.status_filter == p2p_pb2.SOURCE_STATUS_READY


def test_select_sources_returns_ordered_candidates_for_retry():
    response = p2p_pb2.ListSourcesResponse(
        instances=[
            _source_ref("source-b", "worker-b"),
            _source_ref("source-a", "worker-a"),
        ]
    )

    candidates = _transfer(_FakeMxClient(response)).select_sources(
        model_version=5,
        receiver_rank=0,
    )

    assert [candidate.mx_source_id for candidate in candidates] == [
        "source-a",
        "source-b",
    ]


def test_select_source_uses_latest_visible_version_when_unspecified():
    response = p2p_pb2.ListSourcesResponse(
        instances=[
            _source_ref("source-v5", "worker-v5", model_version=5),
            _source_ref("source-v8", "worker-v8", model_version=8),
        ]
    )

    candidate = _transfer(_FakeMxClient(response)).select_source(
        model_version=None,
        receiver_rank=0,
    )

    assert candidate.mx_source_id == "source-v8"
    assert candidate.metadata.model_version == 8


def test_select_source_prefers_inference_replica_for_same_latest_version():
    response = p2p_pb2.ListSourcesResponse(
        instances=[
            _source_ref(
                "trainer-v8",
                "trainer-worker",
                model_version=8,
                role=RlSourceRole.TRAINER,
            ),
            _source_ref(
                "replica-v8",
                "replica-worker",
                model_version=8,
                role=RlSourceRole.INFERENCE_REPLICA,
            ),
        ]
    )

    candidate = _transfer(_FakeMxClient(response)).select_source(
        model_version=None,
        receiver_rank=0,
    )

    assert candidate.mx_source_id == "replica-v8"
    assert candidate.metadata.role == RlSourceRole.INFERENCE_REPLICA


def test_publish_tensors_rejects_empty_tensor_set():
    with pytest.raises(RuntimeError, match="no tensors to publish"):
        _transfer(_FakeMxClient(p2p_pb2.ListSourcesResponse())).publish_tensors(
            {},
            model_version=1,
        )


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

    candidate = _transfer(_FakeMxClient(response)).select_source(
        model_version=5,
        receiver_rank=1,
    )

    assert candidate.mx_source_id == "source-right"
    assert identity_matches_base(right_identity, base_identity)
    assert not identity_matches_base(wrong_identity, base_identity)


def test_wait_for_source_retries_until_source_is_published():
    empty = p2p_pb2.ListSourcesResponse()
    ready = p2p_pb2.ListSourcesResponse(
        instances=[_source_ref("source-v5", "worker-v5")]
    )
    fake_client = _FakeMxClient([empty, ready])
    transfer = RlNixlWeightTransfer(
        mx_client=fake_client,
        base_identity=_base_identity(),
        worker_id="worker-local",
        timeout_seconds=1.0,
    )

    candidate = asyncio.run(
        transfer.wait_for_source(model_version=5, receiver_rank=0)
    )

    assert candidate.mx_source_id == "source-v5"
    assert fake_client.list_call_count == 2


def test_receive_tensors_retries_next_candidate_after_failure():
    class _RetryTransfer(RlNixlWeightTransfer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.attempted_sources = []
            self.model_versions = []

        def _receive_from_candidate(self, candidate, model_version, target_tensors=None):
            del target_tensors
            self.attempted_sources.append(candidate.mx_source_id)
            self.model_versions.append(model_version)
            if candidate.mx_source_id == "source-a":
                raise RuntimeError("boom")
            return [("w", torch.zeros(1))]

    response = p2p_pb2.ListSourcesResponse(
        instances=[
            _source_ref("source-a", "worker-a", model_version=7),
            _source_ref("source-b", "worker-b", model_version=7),
        ]
    )
    transfer = _RetryTransfer(
        mx_client=_FakeMxClient(response),
        base_identity=_base_identity(),
        worker_id="worker-local",
    )

    tensors = asyncio.run(
        transfer.receive_tensors(model_version=None, receiver_rank=0)
    )

    assert transfer.attempted_sources == ["source-a", "source-b"]
    assert transfer.model_versions == [7, 7]
    assert tensors[0][0] == "w"


def test_receive_into_tensors_passes_caller_owned_targets():
    class _IntoTransfer(RlNixlWeightTransfer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.target_tensors = None

        def _receive_from_candidate(self, candidate, model_version, target_tensors=None):
            del candidate
            del model_version
            self.target_tensors = target_tensors
            return list(target_tensors.items())

    response = p2p_pb2.ListSourcesResponse(
        instances=[_source_ref("source-a", "worker-a")]
    )
    transfer = _IntoTransfer(
        mx_client=_FakeMxClient(response),
        base_identity=_base_identity(),
        worker_id="worker-local",
    )
    target_tensors = {"w": torch.zeros(1)}

    tensors = asyncio.run(
        transfer.receive_into_tensors(
            target_tensors,
            model_version=5,
            receiver_rank=0,
        )
    )

    assert transfer.target_tensors is target_tensors
    assert tensors[0][0] == "w"
    assert tensors[0][1] is target_tensors["w"]


def test_receive_into_tensors_rejects_empty_target_set():
    with pytest.raises(RuntimeError, match="no target tensors"):
        asyncio.run(
            _transfer(_FakeMxClient(p2p_pb2.ListSourcesResponse())).receive_into_tensors(
                {},
                model_version=5,
                receiver_rank=0,
            )
        )


def test_finalize_marks_published_source_stale():
    fake_client = _FakeMxClient(p2p_pb2.ListSourcesResponse())
    transfer = _transfer(fake_client)
    transfer._mx_source_id = "source-v5"
    transfer._worker_rank = 3

    transfer.finalize()

    assert fake_client.status_updates == [
        ("source-v5", "worker-local", 3, p2p_pb2.SOURCE_STATUS_STALE)
    ]
    assert transfer._mx_source_id is None
