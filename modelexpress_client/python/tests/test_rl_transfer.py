# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
import torch

import modelexpress.rl_transfer as rl_transfer_module
from modelexpress import p2p_pb2
from modelexpress.rl_fanin_transfer import (
    DenseFanInReceiveResult,
    DenseFanInSourceResult,
)
from modelexpress.rl_metadata import (
    RlSourceMetadata,
    RlSourceRole,
    get_rl_source_metadata,
    with_rl_source_metadata,
)
from modelexpress.rl_reshard import TensorReceiveSpec
from modelexpress.rl_transfer import (
    RlNixlWeightTransfer,
    _ReceiveCandidateResult,
    backend_framework_value,
    build_rl_base_identity,
    identity_matches_base,
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


class _LeaseRecordingMxClient(_FakeMxClient):
    def __init__(self, response):
        super().__init__(response)
        self.lease_begins = []
        self.lease_completes = []
        self._lease_counts = {}

    def begin_transfer_lease(self, **kwargs):
        self.lease_begins.append(kwargs)
        source_id = kwargs["mx_source_id"]
        count = self._lease_counts.get(source_id, 0) + 1
        self._lease_counts[source_id] = count
        return p2p_pb2.TransferLease(lease_id=f"lease-{source_id}-{count}")

    def renew_transfer_lease(self, lease_id, *, ttl_millis=0):
        return p2p_pb2.TransferLease(lease_id=lease_id)

    def complete_transfer_lease(self, lease_id, *, status, error_message=""):
        self.lease_completes.append((lease_id, status, error_message))
        return p2p_pb2.TransferLease(lease_id=lease_id, status=status)


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
    source_world_size: int = 1,
    shape_registry=None,
):
    if shape_registry is None:
        shape_registry = {"w": {"shape": [1], "dtype": "torch.float32"}}
    identity = with_rl_source_metadata(
        _base_identity(),
        RlSourceMetadata(
            model_version=model_version,
            role=role,
            world_size=source_world_size,
            shape_registry=shape_registry,
        ),
    )
    return p2p_pb2.SourceInstanceRef(
        mx_source_id=mx_source_id,
        worker_id=worker_id,
        model_name="test-model",
        worker_rank=worker_rank,
        identity=identity,
    )


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


def test_select_sources_excludes_current_worker_after_replica_publish():
    response = p2p_pb2.ListSourcesResponse(
        instances=[
            _source_ref(
                "self-replica",
                "worker-local",
                model_version=5,
                role=RlSourceRole.INFERENCE_REPLICA,
            ),
            _source_ref(
                "peer-replica",
                "worker-peer",
                model_version=5,
                role=RlSourceRole.INFERENCE_REPLICA,
            ),
            _source_ref(
                "trainer-source",
                "worker-trainer",
                model_version=5,
                role=RlSourceRole.TRAINER,
            ),
        ]
    )

    candidates = _transfer(_FakeMxClient(response)).select_sources(
        model_version=5,
        receiver_rank=0,
    )

    assert [candidate.worker_id for candidate in candidates] == [
        "worker-peer",
        "worker-trainer",
    ]


def test_select_sources_can_target_incomplete_replica_parent_rank():
    response = p2p_pb2.ListSourcesResponse(
        instances=[
            _source_ref(
                "replica-r0",
                "worker-r0",
                model_version=5,
                role=RlSourceRole.INFERENCE_REPLICA,
                worker_rank=0,
                source_world_size=3,
            ),
            _source_ref(
                "replica-r1",
                "worker-r1",
                model_version=5,
                role=RlSourceRole.INFERENCE_REPLICA,
                worker_rank=1,
                source_world_size=3,
            ),
            _source_ref(
                "trainer-r0",
                "worker-trainer",
                model_version=5,
                role=RlSourceRole.TRAINER,
                worker_rank=0,
            ),
        ]
    )

    candidates = _transfer(_FakeMxClient(response)).select_sources(
        model_version=5,
        receiver_rank=2,
        roles=(RlSourceRole.INFERENCE_REPLICA,),
        same_rank_only=False,
        require_complete_version=False,
        source_ranks_by_role={RlSourceRole.INFERENCE_REPLICA: (0,)},
    )

    assert [candidate.mx_source_id for candidate in candidates] == ["replica-r0"]


def test_select_sources_reports_missing_when_only_current_worker_matches():
    response = p2p_pb2.ListSourcesResponse(
        instances=[
            _source_ref(
                "self-replica",
                "worker-local",
                model_version=5,
                role=RlSourceRole.INFERENCE_REPLICA,
            )
        ]
    )

    with pytest.raises(RuntimeError, match="No ModelExpress RL source found"):
        _transfer(_FakeMxClient(response)).select_sources(
            model_version=5,
            receiver_rank=0,
        )


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


def test_select_source_uses_latest_complete_version_when_unspecified():
    response = p2p_pb2.ListSourcesResponse(
        instances=[
            _source_ref(
                "source-v7",
                "worker-v7-r0",
                model_version=7,
                worker_rank=0,
                source_world_size=2,
            ),
            _source_ref(
                "source-v7",
                "worker-v7-r1",
                model_version=7,
                worker_rank=1,
                source_world_size=2,
            ),
            _source_ref(
                "source-v8",
                "worker-v8-r0",
                model_version=8,
                worker_rank=0,
                source_world_size=2,
            ),
        ]
    )

    candidate = _transfer(_FakeMxClient(response)).select_source(
        model_version=None,
        receiver_rank=1,
    )

    assert candidate.mx_source_id == "source-v7"
    assert candidate.metadata.model_version == 7


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


def test_publish_tensors_starts_heartbeat_and_finalize_stops_it(monkeypatch):
    class _FakeNixlManager:
        def __init__(self, *args, **kwargs):
            self.nixl_metadata = b"nixl"
            self.tensor_descriptors = []
            self.shutdown_called = False

        def initialize(self):
            pass

        def register_tensors(self, tensors):
            self.tensor_descriptors = [
                p2p_pb2.TensorDescriptor(
                    name=name,
                    addr=1,
                    size=tensor.numel() * tensor.element_size(),
                    device_id=0,
                    dtype=str(tensor.dtype),
                )
                for name, tensor in tensors.items()
            ]

        def shutdown(self):
            self.shutdown_called = True

    class _FakeHeartbeat:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False
            self.stop_calls = []
            type(self).instances.append(self)

        def start(self):
            self.started = True

        def stop(self, *, mark_stale=True):
            self.stop_calls.append(mark_stale)

    class _PublishMxClient(_FakeMxClient):
        def __init__(self):
            super().__init__(p2p_pb2.ListSourcesResponse())
            self.published = []

        def publish_metadata(self, identity, worker, worker_id):
            self.published.append((identity, worker, worker_id))
            return "source-v5"

    monkeypatch.setattr(rl_transfer_module, "NixlTransferManager", _FakeNixlManager)
    monkeypatch.setattr(rl_transfer_module, "HeartbeatThread", _FakeHeartbeat)
    fake_client = _PublishMxClient()
    transfer = RlNixlWeightTransfer(
        mx_client=fake_client,
        base_identity=_base_identity(),
        worker_id="worker-local",
    )

    source_id = transfer.publish_tensors(
        {"w": torch.zeros(1, dtype=torch.float32)},
        model_version=5,
        worker_rank=2,
        tensor_metadata={"w": {"expert_ids": [0], "expert_axis": 0}},
    )

    assert source_id == "source-v5"
    assert get_rl_source_metadata(fake_client.published[0][0]).shape_registry == {
        "w": {
            "shape": [1],
            "dtype": "torch.float32",
            "expert_ids": [0],
            "expert_axis": 0,
        }
    }
    assert fake_client.status_updates == [
        ("source-v5", "worker-local", 2, p2p_pb2.SOURCE_STATUS_READY)
    ]
    assert len(_FakeHeartbeat.instances) == 1
    heartbeat = _FakeHeartbeat.instances[0]
    assert heartbeat.started
    assert heartbeat.kwargs["mx_source_id"] == "source-v5"
    assert heartbeat.kwargs["worker_rank"] == 2
    assert heartbeat.kwargs["initially_ready"] is True

    transfer.finalize()

    assert heartbeat.stop_calls == [False]
    assert fake_client.status_updates[-1] == (
        "source-v5",
        "worker-local",
        2,
        p2p_pb2.SOURCE_STATUS_STALE,
    )
    assert transfer._heartbeat is None


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

        def _receive_from_candidate(self, candidate, model_version, **kwargs):
            del kwargs
            self.attempted_sources.append(candidate.mx_source_id)
            self.model_versions.append(model_version)
            if candidate.mx_source_id == "source-a":
                raise RuntimeError("boom")
            return _ReceiveCandidateResult(
                tensors=[("w", torch.zeros(1))],
                bytes_transferred=4,
                tensor_count=1,
                duration_seconds=0.01,
            )

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
    assert transfer.last_receive_report is not None
    assert transfer.last_receive_report.success
    assert transfer.last_receive_report.retry_count == 1
    assert transfer.last_receive_report.resolved_model_version == 7
    assert transfer.last_receive_report.source_worker_id == "worker-b"
    assert [attempt.success for attempt in transfer.last_receive_report.attempts] == [
        False,
        True,
    ]
    assert transfer.last_receive_report.attempts[1].bytes_transferred == 4


def test_receive_tensors_reports_transfer_lease_ids_across_retries():
    class _LeasedRetryTransfer(RlNixlWeightTransfer):
        def _receive_from_candidate_unleased(self, candidate, model_version, **kwargs):
            del model_version
            del kwargs
            if candidate.mx_source_id == "source-a":
                raise RuntimeError("boom")
            return _ReceiveCandidateResult(
                tensors=[("w", torch.zeros(1))],
                bytes_transferred=4,
                tensor_count=1,
                duration_seconds=0.01,
            )

    response = p2p_pb2.ListSourcesResponse(
        instances=[
            _source_ref("source-a", "worker-a", model_version=7),
            _source_ref("source-b", "worker-b", model_version=7),
        ]
    )
    fake_client = _LeaseRecordingMxClient(response)
    transfer = _LeasedRetryTransfer(
        mx_client=fake_client,
        base_identity=_base_identity(),
        worker_id="worker-local",
    )

    tensors = asyncio.run(
        transfer.receive_tensors(model_version=None, receiver_rank=0)
    )

    assert tensors[0][0] == "w"
    assert transfer.last_receive_report is not None
    assert [
        (attempt.mx_source_id, attempt.success, attempt.lease_id, attempt.error)
        for attempt in transfer.last_receive_report.attempts
    ] == [
        ("source-a", False, "lease-source-a-1", "boom"),
        ("source-b", True, "lease-source-b-1", None),
    ]
    assert fake_client.lease_completes == [
        (
            "lease-source-a-1",
            p2p_pb2.TRANSFER_LEASE_STATUS_FAILED,
            "boom",
        ),
        (
            "lease-source-b-1",
            p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
            "",
        ),
    ]


def test_receive_tensors_reports_empty_lease_id_when_leases_are_disabled():
    class _LeaselessTransfer(RlNixlWeightTransfer):
        def _receive_from_candidate_unleased(self, candidate, model_version, **kwargs):
            del candidate
            del model_version
            del kwargs
            return _ReceiveCandidateResult(
                tensors=[("w", torch.zeros(1))],
                bytes_transferred=4,
                tensor_count=1,
                duration_seconds=0.01,
            )

    response = p2p_pb2.ListSourcesResponse(
        instances=[_source_ref("source-a", "worker-a", model_version=7)]
    )
    transfer = _LeaselessTransfer(
        mx_client=_FakeMxClient(response),
        base_identity=_base_identity(),
        worker_id="worker-local",
    )

    asyncio.run(transfer.receive_tensors(model_version=None, receiver_rank=0))

    assert transfer.last_receive_report is not None
    assert transfer.last_receive_report.attempts[0].lease_id == ""


def test_receive_tensors_reports_dense_fanin_transfer_lease_ids():
    class _LeasedFanInTransfer(RlNixlWeightTransfer):
        def _receive_from_candidate_unleased(self, candidate, model_version, **kwargs):
            del model_version
            del kwargs
            raise RuntimeError(f"single-source incomplete: {candidate.mx_source_id}")

        def _receive_from_candidate_group_unleased(self, candidates, **kwargs):
            del kwargs
            return DenseFanInReceiveResult(
                tensors=[("w", torch.zeros(4))],
                source_results=tuple(
                    DenseFanInSourceResult(
                        candidate=candidate,
                        bytes_transferred=8,
                        tensor_count=1,
                        duration_seconds=0.01,
                    )
                    for candidate in candidates
                ),
            )

    response = p2p_pb2.ListSourcesResponse(
        instances=[
            _source_ref(
                "source-r0",
                "worker-r0",
                model_version=7,
                worker_rank=0,
                source_world_size=2,
            ),
            _source_ref(
                "source-r1",
                "worker-r1",
                model_version=7,
                worker_rank=1,
                source_world_size=2,
            ),
        ]
    )
    fake_client = _LeaseRecordingMxClient(response)
    transfer = _LeasedFanInTransfer(
        mx_client=fake_client,
        base_identity=_base_identity(),
        worker_id="worker-local",
    )

    tensors = asyncio.run(
        transfer.receive_tensors(model_version=None, receiver_rank=0)
    )

    assert tensors[0][0] == "w"
    assert transfer.last_receive_report is not None
    assert [
        (attempt.mx_source_id, attempt.success, attempt.lease_id)
        for attempt in transfer.last_receive_report.attempts
    ] == [
        ("source-r0", False, "lease-source-r0-1"),
        ("source-r1", False, "lease-source-r1-1"),
        ("source-r0", True, "lease-source-r0-2"),
        ("source-r1", True, "lease-source-r1-2"),
    ]
    completed_statuses = [
        (lease_id, status) for lease_id, status, _error in fake_client.lease_completes
    ]
    assert completed_statuses[:2] == [
        ("lease-source-r0-1", p2p_pb2.TRANSFER_LEASE_STATUS_FAILED),
        ("lease-source-r1-1", p2p_pb2.TRANSFER_LEASE_STATUS_FAILED),
    ]
    assert set(completed_statuses[2:]) == {
        ("lease-source-r0-2", p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED),
        ("lease-source-r1-2", p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED),
    }


def test_receive_tensors_records_failed_transfer_report():
    class _FailTransfer(RlNixlWeightTransfer):
        def _receive_from_candidate(self, candidate, model_version, **kwargs):
            del candidate
            del model_version
            del kwargs
            raise RuntimeError("transfer failed")

    response = p2p_pb2.ListSourcesResponse(
        instances=[_source_ref("source-a", "worker-a", model_version=7)]
    )
    transfer = _FailTransfer(
        mx_client=_FakeMxClient(response),
        base_identity=_base_identity(),
        worker_id="worker-local",
    )

    with pytest.raises(RuntimeError, match="No ModelExpress RL source transfer succeeded"):
        asyncio.run(transfer.receive_tensors(model_version=None, receiver_rank=0))

    assert transfer.last_receive_report is not None
    assert not transfer.last_receive_report.success
    assert transfer.last_receive_report.resolved_model_version is None
    assert transfer.last_receive_report.retry_count == 1
    assert transfer.last_receive_report.attempts[0].error == "transfer failed"


def test_receive_tensors_clears_stale_report_when_discovery_fails():
    transfer = RlNixlWeightTransfer(
        mx_client=_FakeMxClient(p2p_pb2.ListSourcesResponse()),
        base_identity=_base_identity(),
        worker_id="worker-local",
        timeout_seconds=0.0,
    )
    transfer.last_receive_report = object()

    with pytest.raises(RuntimeError, match="No ModelExpress RL source found"):
        asyncio.run(transfer.receive_tensors(model_version=5, receiver_rank=0))

    assert transfer.last_receive_report is None


def test_receive_into_tensors_applies_exact_plan_before_nixl(monkeypatch):
    class _FakeNixlManager:
        received_tensor_names = None
        registered_tensor_names = None

        def __init__(self, *args, **kwargs):
            pass

        @property
        def nixl_metadata(self):
            return b""

        @property
        def tensor_descriptors(self):
            return []

        def initialize(self):
            pass

        def register_tensors(self, tensors):
            type(self).registered_tensor_names = sorted(tensors)

        def receive_from_source(self, source_metadata, source_tensors, timeout_seconds):
            del source_metadata
            del timeout_seconds
            type(self).received_tensor_names = [tensor.name for tensor in source_tensors]
            return 4, len(source_tensors), 0.0

    class _MetadataMxClient(_FakeMxClient):
        def get_metadata(self, mx_source_id, worker_id):
            del mx_source_id
            del worker_id
            return p2p_pb2.GetMetadataResponse(
                found=True,
                worker=p2p_pb2.WorkerMetadata(
                    worker_rank=0,
                    nixl_metadata=b"source",
                    tensors=[
                        p2p_pb2.TensorDescriptor(
                            name="w",
                            addr=1,
                            size=4,
                            device_id=0,
                            dtype="torch.float32",
                        ),
                        p2p_pb2.TensorDescriptor(
                            name="unused",
                            addr=2,
                            size=4,
                            device_id=0,
                            dtype="torch.float32",
                        ),
                    ],
                ),
            )

    monkeypatch.setattr(rl_transfer_module, "NixlTransferManager", _FakeNixlManager)
    response = p2p_pb2.ListSourcesResponse(
        instances=[_source_ref("source-a", "worker-a")]
    )
    transfer = RlNixlWeightTransfer(
        mx_client=_MetadataMxClient(response),
        base_identity=_base_identity(),
        worker_id="worker-local",
    )

    tensors = asyncio.run(
        transfer.receive_into_tensors(
            {"w": torch.zeros(1, dtype=torch.float32)},
            model_version=5,
            receiver_rank=0,
        )
    )

    assert tensors[0][0] == "w"
    assert _FakeNixlManager.registered_tensor_names == ["w"]
    assert _FakeNixlManager.received_tensor_names == ["w"]


def test_receive_into_tensors_rejects_incomplete_exact_plan():
    class _MetadataMxClient(_FakeMxClient):
        def get_metadata(self, mx_source_id, worker_id):
            del mx_source_id
            del worker_id
            return p2p_pb2.GetMetadataResponse(
                found=True,
                worker=p2p_pb2.WorkerMetadata(
                    worker_rank=0,
                    nixl_metadata=b"source",
                    tensors=[
                        p2p_pb2.TensorDescriptor(
                            name="experts.w",
                            addr=1,
                            size=4,
                            device_id=0,
                            dtype="torch.float32",
                        )
                    ],
                ),
            )

    response = p2p_pb2.ListSourcesResponse(
        instances=[
            _source_ref(
                "source-a",
                "worker-a",
                shape_registry={
                    "experts.w": {
                        "shape": [1],
                        "dtype": "torch.float32",
                        "expert_ids": [0],
                    }
                },
            )
        ]
    )
    transfer = RlNixlWeightTransfer(
        mx_client=_MetadataMxClient(response),
        base_identity=_base_identity(),
        worker_id="worker-local",
    )

    with pytest.raises(RuntimeError, match="incomplete RL reshard plan"):
        asyncio.run(
            transfer.receive_into_tensors(
                {"experts.w": torch.zeros(1, dtype=torch.float32)},
                model_version=5,
                receiver_rank=0,
                target_specs=[
                    TensorReceiveSpec(
                        name="experts.w",
                        receiver_rank=0,
                        shape=(1,),
                        dtype="torch.float32",
                        expert_ids=frozenset({1}),
                    )
                ],
            )
        )


def test_receive_into_tensors_validates_descriptors_before_nixl(monkeypatch):
    class _UnexpectedNixlManager:
        def __init__(self, *args, **kwargs):
            raise AssertionError("NIXL should not initialize before descriptor validation")

    class _MetadataMxClient(_FakeMxClient):
        def get_metadata(self, mx_source_id, worker_id):
            del mx_source_id
            del worker_id
            return p2p_pb2.GetMetadataResponse(
                found=True,
                worker=p2p_pb2.WorkerMetadata(
                    worker_rank=0,
                    nixl_metadata=b"source",
                    tensors=[],
                ),
            )

    monkeypatch.setattr(rl_transfer_module, "NixlTransferManager", _UnexpectedNixlManager)
    response = p2p_pb2.ListSourcesResponse(
        instances=[_source_ref("source-a", "worker-a")]
    )
    transfer = RlNixlWeightTransfer(
        mx_client=_MetadataMxClient(response),
        base_identity=_base_identity(),
        worker_id="worker-local",
    )

    with pytest.raises(RuntimeError, match="source descriptors missing planned entries"):
        asyncio.run(
            transfer.receive_into_tensors(
                {"w": torch.zeros(1, dtype=torch.float32)},
                model_version=5,
                receiver_rank=0,
            )
        )


def test_receive_into_tensors_passes_caller_owned_targets():
    class _IntoTransfer(RlNixlWeightTransfer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.target_tensors = None

        def _receive_from_candidate(self, candidate, model_version, **kwargs):
            del candidate
            del model_version
            target_tensors = kwargs["target_tensors"]
            self.target_tensors = target_tensors
            return _ReceiveCandidateResult(
                tensors=list(target_tensors.items()),
                bytes_transferred=4,
                tensor_count=len(target_tensors),
                duration_seconds=0.01,
            )

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


def test_receive_and_publish_replica_uses_received_version_and_receiver_rank():
    class _ReplicaTransfer(RlNixlWeightTransfer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.publish_calls = []

        def _receive_from_candidate(self, candidate, model_version, **kwargs):
            del candidate
            del model_version
            tensors = list(kwargs["target_tensors"].items())
            return _ReceiveCandidateResult(
                tensors=tensors,
                bytes_transferred=4,
                tensor_count=len(tensors),
                duration_seconds=0.01,
            )

        def publish_tensors(
            self,
            tensors,
            *,
            model_version,
            role=RlSourceRole.TRAINER,
            worker_rank=0,
            source_world_size=1,
            tensor_metadata=None,
        ):
            self.publish_calls.append(
                {
                    "tensors": tensors,
                    "model_version": model_version,
                    "role": role,
                    "worker_rank": worker_rank,
                    "source_world_size": source_world_size,
                    "tensor_metadata": tensor_metadata,
                }
            )
            return "replica-source"

    response = p2p_pb2.ListSourcesResponse(
        instances=[_source_ref("source-v9", "worker-v9", model_version=9)]
    )
    transfer = _ReplicaTransfer(
        mx_client=_FakeMxClient(response),
        base_identity=_base_identity(),
        worker_id="worker-local",
    )
    target_tensors = {"w": torch.zeros(1)}

    tensors = asyncio.run(
        transfer.receive_and_publish_replica(
            target_tensors,
            model_version=None,
            receiver_rank=3,
            target_specs=[
                TensorReceiveSpec(
                    name="w",
                    receiver_rank=3,
                    shape=(1,),
                    dtype="torch.float32",
                    global_shape=(2,),
                    shard_offsets=(1,),
                    expert_ids=(1,),
                    expert_axis=0,
                )
            ],
            replica_world_size=4,
        )
    )

    assert tensors == list(target_tensors.items())
    assert transfer.publish_calls == [
        {
            "tensors": target_tensors,
            "model_version": 9,
            "role": RlSourceRole.INFERENCE_REPLICA,
            "worker_rank": 3,
            "source_world_size": 4,
            "tensor_metadata": {
                "w": {
                    "global_shape": [2],
                    "shard_offsets": [1],
                    "tensor_parallel_rank": 0,
                    "pipeline_parallel_rank": 0,
                    "expert_ids": [1],
                    "expert_axis": 0,
                }
            },
        }
    ]


def test_receive_tensors_and_publish_replica_uses_allocated_receive():
    class _ReplicaTransfer(RlNixlWeightTransfer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.publish_calls = []

        def _receive_from_candidate(self, candidate, model_version, **kwargs):
            del candidate
            del model_version
            del kwargs
            return _ReceiveCandidateResult(
                tensors=[("w", torch.zeros(1))],
                bytes_transferred=4,
                tensor_count=1,
                duration_seconds=0.01,
            )

        def publish_tensors(
            self,
            tensors,
            *,
            model_version,
            role=RlSourceRole.TRAINER,
            worker_rank=0,
            source_world_size=1,
            tensor_metadata=None,
        ):
            self.publish_calls.append(
                {
                    "tensors": tensors,
                    "model_version": model_version,
                    "role": role,
                    "worker_rank": worker_rank,
                    "source_world_size": source_world_size,
                    "tensor_metadata": tensor_metadata,
                }
            )
            return "replica-source"

    response = p2p_pb2.ListSourcesResponse(
        instances=[_source_ref("source-v11", "worker-v11", model_version=11)]
    )
    transfer = _ReplicaTransfer(
        mx_client=_FakeMxClient(response),
        base_identity=_base_identity(),
        worker_id="worker-local",
    )

    tensors = asyncio.run(
        transfer.receive_tensors_and_publish_replica(
            model_version=None,
            receiver_rank=2,
            replica_world_size=3,
        )
    )

    assert tensors[0][0] == "w"
    assert transfer.publish_calls == [
        {
            "tensors": {"w": tensors[0][1]},
            "model_version": 11,
            "role": RlSourceRole.INFERENCE_REPLICA,
            "worker_rank": 2,
            "source_world_size": 3,
            "tensor_metadata": None,
        }
    ]


def test_receive_tensors_and_publish_replica_preserves_allocated_metadata(monkeypatch):
    class _FakeNixlManager:
        def __init__(self, *args, **kwargs):
            pass

        def initialize(self):
            pass

        def register_tensors(self, tensors):
            self.tensors = tensors

        def receive_from_source(self, source_metadata, source_tensors, timeout_seconds):
            del source_metadata
            del timeout_seconds
            return 8, len(source_tensors), 0.0

    class _MetadataMxClient(_FakeMxClient):
        def get_metadata(self, mx_source_id, worker_id):
            del mx_source_id
            del worker_id
            return p2p_pb2.GetMetadataResponse(
                found=True,
                worker=p2p_pb2.WorkerMetadata(
                    worker_rank=0,
                    nixl_metadata=b"source",
                    tensors=[
                        p2p_pb2.TensorDescriptor(
                            name="experts.w",
                            addr=1,
                            size=8,
                            device_id=0,
                            dtype="torch.float32",
                        )
                    ],
                ),
            )

    class _ReplicaTransfer(RlNixlWeightTransfer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.publish_calls = []

        def publish_tensors(
            self,
            tensors,
            *,
            model_version,
            role=RlSourceRole.TRAINER,
            worker_rank=0,
            source_world_size=1,
            tensor_metadata=None,
        ):
            self.publish_calls.append(
                {
                    "tensors": tensors,
                    "model_version": model_version,
                    "role": role,
                    "worker_rank": worker_rank,
                    "source_world_size": source_world_size,
                    "tensor_metadata": tensor_metadata,
                }
            )
            return "replica-source"

    shape_registry = {
        "experts.w": {
            "shape": [1, 2],
            "dtype": "torch.float32",
            "global_shape": [4, 2],
            "shard_offsets": [1, 0],
            "tensor_parallel_rank": 2,
            "pipeline_parallel_rank": 3,
            "expert_ids": [1],
            "expert_axis": 0,
        }
    }
    monkeypatch.setattr(rl_transfer_module, "NixlTransferManager", _FakeNixlManager)
    monkeypatch.setattr(
        rl_transfer_module,
        "allocate_tensors_from_shape_registry",
        lambda _shape_registry, *, device: {
            "experts.w": torch.zeros((1, 2), dtype=torch.float32)
        },
    )
    transfer = _ReplicaTransfer(
        mx_client=_MetadataMxClient(
            p2p_pb2.ListSourcesResponse(
                instances=[
                    _source_ref(
                        "source-v12",
                        "worker-v12",
                        model_version=12,
                        shape_registry=shape_registry,
                    )
                ]
            )
        ),
        base_identity=_base_identity(),
        worker_id="worker-local",
    )

    tensors = asyncio.run(
        transfer.receive_tensors_and_publish_replica(
            model_version=None,
            receiver_rank=2,
            replica_world_size=3,
        )
    )

    assert tensors[0][0] == "experts.w"
    assert transfer.publish_calls[0]["tensor_metadata"] == {
        "experts.w": {
            "global_shape": [4, 2],
            "shard_offsets": [1, 0],
            "tensor_parallel_rank": 2,
            "pipeline_parallel_rank": 3,
            "expert_ids": [1],
            "expert_axis": 0,
        }
    }


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
