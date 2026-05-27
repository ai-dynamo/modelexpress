# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import time
import uuid

import grpc
import pytest
import torch

from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from modelexpress.rl_fanout import RlTreeFanoutPolicy
from modelexpress.rl_metadata import (
    RlSourceMetadata,
    RlSourceRole,
    get_rl_source_metadata,
    with_rl_source_metadata,
)
from modelexpress.rl_reshard import TensorReceiveSpec
from modelexpress.rl_transfer import RlNixlWeightTransfer, build_rl_base_identity
from modelexpress.rl_transfer_lease import (
    RlTransferLeaseCoordinator,
    summarize_report_leases,
)

_LIVE_SERVER_ENV = "MX_LIVE_SERVER_URL"

pytestmark = pytest.mark.skipif(
    not os.environ.get(_LIVE_SERVER_ENV),
    reason=f"{_LIVE_SERVER_ENV} is not set",
)


def _base_identity(model_name: str):
    return build_rl_base_identity(
        model_name=model_name,
        mx_version="0.3.0",
        backend_framework="vllm",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=0,
        dtype="float32",
        quantization="",
        revision="",
    )


def _begin_transfer_lease_or_skip(client: MxClient, **kwargs):
    try:
        return client.begin_transfer_lease(**kwargs)
    except grpc.RpcError as exc:
        if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
            pytest.skip("server does not expose transfer lease RPCs")
        raise


def test_live_server_transfer_lease_contract():
    client = MxClient(server_url=os.environ[_LIVE_SERVER_ENV])
    lease_id = f"lease-{uuid.uuid4().hex}"
    other_lease_id = f"lease-{uuid.uuid4().hex}"
    other_source_lease_id = f"lease-{uuid.uuid4().hex}"
    mx_source_id = f"live-lease-source-{uuid.uuid4().hex[:8]}"
    target_worker_id = f"target-worker-{uuid.uuid4().hex[:8]}"

    try:
        lease = _begin_transfer_lease_or_skip(
            client,
            lease_id=lease_id,
            mx_source_id=mx_source_id,
            source_worker_id="source-worker",
            target_worker_id=target_worker_id,
            target_worker_rank=1,
            model_version=17,
            ttl_millis=5_000,
            metadata={"contract": "live"},
        )

        assert lease.lease_id == lease_id
        assert lease.status == p2p_pb2.TRANSFER_LEASE_STATUS_ACTIVE
        assert lease.mx_source_id == mx_source_id
        assert lease.source_worker_id == "source-worker"
        assert lease.target_worker_id == target_worker_id
        assert lease.target_worker_rank == 1
        assert lease.model_version == 17
        assert lease.metadata["contract"] == "live"

        renewed = client.renew_transfer_lease(lease_id, ttl_millis=5_000)
        assert renewed.status == p2p_pb2.TRANSFER_LEASE_STATUS_ACTIVE
        assert renewed.expires_at >= lease.expires_at

        completed = client.complete_transfer_lease(
            lease_id,
            status=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
        )
        assert completed.status == p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED

        other_lease = _begin_transfer_lease_or_skip(
            client,
            lease_id=other_lease_id,
            mx_source_id=mx_source_id,
            source_worker_id="source-worker",
            target_worker_id=target_worker_id,
            target_worker_rank=1,
            model_version=18,
            ttl_millis=5_000,
            metadata={"contract": "live-other-version"},
        )
        assert other_lease.model_version == 18
        client.complete_transfer_lease(
            other_lease_id,
            status=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
        )

        other_source_lease = _begin_transfer_lease_or_skip(
            client,
            lease_id=other_source_lease_id,
            mx_source_id=mx_source_id,
            source_worker_id="source-worker-other",
            target_worker_id=target_worker_id,
            target_worker_rank=1,
            model_version=17,
            ttl_millis=5_000,
            metadata={"contract": "live-other-source"},
        )
        assert other_source_lease.source_worker_id == "source-worker-other"
        assert other_source_lease.model_version == 17
        client.complete_transfer_lease(
            other_source_lease_id,
            status=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
        )

        fetched = client.get_transfer_lease(lease_id)
        assert fetched.found
        assert fetched.lease.lease_id == lease_id
        assert fetched.lease.status == p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED

        listed = client.list_transfer_leases(
            mx_source_id=mx_source_id,
            target_worker_id=target_worker_id,
            status_filter=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
        )
        assert [lease.lease_id for lease in listed.leases] == [
            lease_id,
            other_lease_id,
            other_source_lease_id,
        ]
        listed_v17 = client.list_transfer_leases(
            mx_source_id=mx_source_id,
            target_worker_id=target_worker_id,
            status_filter=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
            model_version_filter=17,
        )
        assert [lease.lease_id for lease in listed_v17.leases] == [
            lease_id,
            other_source_lease_id,
        ]
        listed_source_worker = client.list_transfer_leases(
            mx_source_id=mx_source_id,
            target_worker_id=target_worker_id,
            status_filter=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
            model_version_filter=17,
            source_worker_id="source-worker",
        )
        assert [lease.lease_id for lease in listed_source_worker.leases] == [lease_id]

        inventory = RlTransferLeaseCoordinator(
            mx_client=client,
            target_worker_id=target_worker_id,
            ttl_seconds=5,
        ).list_target_leases(
            mx_source_id=mx_source_id,
            statuses=(p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,),
            model_version=17,
            source_worker_id="source-worker",
        )
        assert inventory.discovery_supported
        assert [lease.lease_id for lease in inventory.leases] == [lease_id]
        assert inventory.latest_model_version() == 17
        assert [lease.lease_id for lease in inventory.latest_attempts()] == [lease_id]
    finally:
        client.close()


def test_live_server_transfer_lease_expires_abandoned_attempts():
    client = MxClient(server_url=os.environ[_LIVE_SERVER_ENV])
    lease_id = f"lease-expire-{uuid.uuid4().hex}"
    mx_source_id = f"live-lease-expire-source-{uuid.uuid4().hex[:8]}"
    target_worker_id = f"target-expire-{uuid.uuid4().hex[:8]}"

    try:
        lease = _begin_transfer_lease_or_skip(
            client,
            lease_id=lease_id,
            mx_source_id=mx_source_id,
            source_worker_id="source-worker",
            target_worker_id=target_worker_id,
            target_worker_rank=2,
            model_version=23,
            ttl_millis=1,
            metadata={"contract": "expire"},
        )

        assert lease.status == p2p_pb2.TRANSFER_LEASE_STATUS_ACTIVE

        deadline = time.monotonic() + 3.0
        while True:
            fetched = client.get_transfer_lease(lease_id)
            assert fetched.found
            if fetched.lease.status == p2p_pb2.TRANSFER_LEASE_STATUS_EXPIRED:
                break
            if time.monotonic() >= deadline:
                pytest.fail("transfer lease did not expire within the live contract deadline")
            time.sleep(0.1)

        assert fetched.lease.error_message == "transfer lease expired"
        assert fetched.lease.target_worker_id == target_worker_id
        assert fetched.lease.model_version == 23

        expired = client.list_transfer_leases(
            mx_source_id=mx_source_id,
            target_worker_id=target_worker_id,
            status_filter=p2p_pb2.TRANSFER_LEASE_STATUS_EXPIRED,
        )
        assert [lease.lease_id for lease in expired.leases] == [lease_id]

        active = client.list_transfer_leases(
            mx_source_id=mx_source_id,
            target_worker_id=target_worker_id,
            status_filter=p2p_pb2.TRANSFER_LEASE_STATUS_ACTIVE,
        )
        assert active.leases == []

        with pytest.raises(RuntimeError, match="is not active"):
            client.renew_transfer_lease(lease_id, ttl_millis=5_000)
    finally:
        client.close()


def test_live_server_lists_rl_identity_and_selects_latest_retained_version():
    client = MxClient(server_url=os.environ[_LIVE_SERVER_ENV])
    model_name = f"mx-live-contract-{uuid.uuid4().hex[:8]}"
    base_identity = _base_identity(model_name)
    published: list[tuple[str, str]] = []

    try:
        for version in (1, 2):
            metadata = RlSourceMetadata(
                model_version=version,
                role=RlSourceRole.TRAINER,
                world_size=1,
                retain_latest_k=2,
                shape_registry={
                    "w": {
                        "shape": [1],
                        "dtype": "torch.float32",
                    }
                },
            )
            identity = with_rl_source_metadata(base_identity, metadata)
            worker = p2p_pb2.WorkerMetadata(
                worker_rank=0,
                nixl_metadata=b"contract-test",
                tensors=[
                    p2p_pb2.TensorDescriptor(
                        name="w",
                        addr=version,
                        size=4,
                        device_id=0,
                        dtype="torch.float32",
                    )
                ],
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
            )
            worker_id = f"worker-v{version}-{uuid.uuid4().hex[:6]}"
            source_id = client.publish_metadata(identity, worker, worker_id)
            published.append((source_id, worker_id))
            assert client.update_status(
                source_id,
                worker_id,
                0,
                p2p_pb2.SOURCE_STATUS_READY,
            )

        response = client.list_sources(
            identity=None,
            status_filter=p2p_pb2.SOURCE_STATUS_READY,
        )
        versions = sorted(
            get_rl_source_metadata(ref.identity).model_version
            for ref in response.instances
            if ref.HasField("identity") and ref.identity.model_name == model_name
        )
        assert versions == [1, 2]

        transfer = RlNixlWeightTransfer(
            mx_client=client,
            base_identity=base_identity,
            worker_id="receiver-live-contract",
            timeout_seconds=0.0,
        )
        selected = transfer.select_source(model_version=None, receiver_rank=0)

        assert selected.metadata.model_version == 2
        assert selected.worker_id.startswith("worker-v2-")
    finally:
        for source_id, worker_id in published:
            client.update_status(
                source_id,
                worker_id,
                0,
                p2p_pb2.SOURCE_STATUS_STALE,
            )
        client.close()


def test_live_server_transfers_latest_retained_cuda_version():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("nixl._api")

    client = MxClient(server_url=os.environ[_LIVE_SERVER_ENV])
    model_name = f"mx-live-transfer-{uuid.uuid4().hex[:8]}"
    base_identity = _base_identity(model_name)
    publisher = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"publisher-{uuid.uuid4().hex[:8]}",
        retain_latest_k=2,
        device_id=0,
        timeout_seconds=20.0,
    )
    receiver = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"receiver-{uuid.uuid4().hex[:8]}",
        device_id=0,
        timeout_seconds=20.0,
    )

    try:
        source_tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
        publisher.publish_tensors({"w": source_tensor}, model_version=1)
        source_tensor.add_(10.0)
        publisher.publish_tensors({"w": source_tensor}, model_version=2)

        received = torch.zeros_like(source_tensor)
        tensors = asyncio.run(
            receiver.receive_into_tensors(
                {"w": received},
                model_version=None,
                receiver_rank=0,
            )
        )
        torch.cuda.synchronize(0)

        assert tensors == [("w", received)]
        assert received.detach().cpu().tolist() == [11.0, 12.0, 13.0]
        assert receiver.last_receive_report is not None
        assert receiver.last_receive_report.resolved_model_version == 2
        attempt = receiver.last_receive_report.attempts[0]
        assert attempt.bytes_transferred == 12
        if not attempt.lease_id:
            pytest.skip("server does not expose transfer lease RPCs")
        fetched = client.get_transfer_lease(attempt.lease_id)
        assert fetched.found
        assert fetched.lease.status == p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED
        assert fetched.lease.target_worker_id == receiver.worker_id
    finally:
        receiver.finalize()
        publisher.finalize()
        client.close()


def test_live_server_transfers_moe_expert_axis_slice_cuda_version():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("nixl._api")

    client = MxClient(server_url=os.environ[_LIVE_SERVER_ENV])
    model_name = f"mx-live-moe-{uuid.uuid4().hex[:8]}"
    base_identity = _base_identity(model_name)
    publisher = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"publisher-{uuid.uuid4().hex[:8]}",
        device_id=0,
        timeout_seconds=20.0,
    )
    receiver = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"receiver-{uuid.uuid4().hex[:8]}",
        device_id=0,
        timeout_seconds=20.0,
    )

    try:
        source_tensor = torch.arange(
            8,
            dtype=torch.float32,
            device="cuda:0",
        ).reshape(4, 2)
        publisher.publish_tensors(
            {"experts.w": source_tensor},
            model_version=7,
            tensor_metadata={
                "experts.w": {
                    "global_shape": [4, 2],
                    "shard_offsets": [0, 0],
                    "expert_ids": [0, 1, 2, 3],
                    "expert_axis": 0,
                }
            },
        )

        received = torch.zeros((2, 2), dtype=torch.float32, device="cuda:0")
        tensors = asyncio.run(
            receiver.receive_into_tensors(
                {"experts.w": received},
                model_version=None,
                receiver_rank=0,
                target_specs=[
                    TensorReceiveSpec(
                        name="experts.w",
                        receiver_rank=0,
                        shape=(2, 2),
                        dtype="torch.float32",
                        global_shape=(4, 2),
                        shard_offsets=(0, 0),
                        expert_ids=(1, 3),
                        expert_axis=0,
                    )
                ],
            )
        )
        torch.cuda.synchronize(0)

        assert tensors == [("experts.w", received)]
        assert received.detach().cpu().tolist() == [[2.0, 3.0], [6.0, 7.0]]
        assert receiver.last_receive_report is not None
        assert receiver.last_receive_report.resolved_model_version == 7
        assert receiver.last_receive_report.attempts[0].bytes_transferred == 16
        assert receiver.last_receive_report.attempts[0].tensor_count == 2
    finally:
        receiver.finalize()
        publisher.finalize()
        client.close()


def test_live_server_transfers_multi_source_dense_fanin_cuda_version():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < 3:
        pytest.skip("dense fan-in live contract needs at least 3 CUDA devices")
    pytest.importorskip("nixl._api")

    client = MxClient(server_url=os.environ[_LIVE_SERVER_ENV])
    model_name = f"mx-live-fanin-{uuid.uuid4().hex[:8]}"
    base_identity = _base_identity(model_name)
    publisher_r0 = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"publisher-r0-{uuid.uuid4().hex[:8]}",
        device_id=0,
        timeout_seconds=20.0,
    )
    publisher_r1 = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"publisher-r1-{uuid.uuid4().hex[:8]}",
        device_id=1,
        timeout_seconds=20.0,
    )
    receiver = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"receiver-{uuid.uuid4().hex[:8]}",
        device_id=2,
        timeout_seconds=20.0,
    )

    try:
        publisher_r0.publish_tensors(
            {"w": torch.tensor([1.0, 2.0], dtype=torch.float32, device="cuda:0")},
            model_version=11,
            worker_rank=0,
            source_world_size=2,
            tensor_metadata={
                "w": {
                    "global_shape": [4],
                    "shard_offsets": [0],
                }
            },
        )
        publisher_r1.publish_tensors(
            {"w": torch.tensor([3.0, 4.0], dtype=torch.float32, device="cuda:1")},
            model_version=11,
            worker_rank=1,
            source_world_size=2,
            tensor_metadata={
                "w": {
                    "global_shape": [4],
                    "shard_offsets": [2],
                }
            },
        )

        received = torch.zeros((4,), dtype=torch.float32, device="cuda:2")
        tensors = asyncio.run(
            receiver.receive_into_tensors(
                {"w": received},
                model_version=None,
                receiver_rank=0,
                target_specs=[
                    TensorReceiveSpec(
                        name="w",
                        receiver_rank=0,
                        shape=(4,),
                        dtype="torch.float32",
                        global_shape=(4,),
                        shard_offsets=(0,),
                    )
                ],
            )
        )
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        torch.cuda.synchronize(2)

        assert tensors == [("w", received)]
        assert received.detach().cpu().tolist() == [1.0, 2.0, 3.0, 4.0]
        assert receiver.last_receive_report is not None
        assert receiver.last_receive_report.resolved_model_version == 11
        assert receiver.last_receive_report.retry_count == 0
        assert [
            (attempt.worker_rank, attempt.bytes_transferred, attempt.tensor_count)
            for attempt in receiver.last_receive_report.attempts
        ] == [(0, 8, 1), (1, 8, 1)]

        allocated_tensors = asyncio.run(
            receiver.receive_tensors(
                model_version=11,
                receiver_rank=0,
            )
        )
        torch.cuda.synchronize(2)

        assert allocated_tensors[0][0] == "w"
        assert allocated_tensors[0][1].device.index == 2
        assert allocated_tensors[0][1].detach().cpu().tolist() == [1.0, 2.0, 3.0, 4.0]
        assert receiver.last_receive_report is not None
        assert receiver.last_receive_report.retry_count == 0
    finally:
        receiver.finalize()
        publisher_r1.finalize()
        publisher_r0.finalize()
        client.close()


def test_live_server_allocated_fanin_replica_preserves_layout_metadata():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < 3:
        pytest.skip("allocated dense fan-in replica live contract needs at least 3 CUDA devices")
    pytest.importorskip("nixl._api")

    client = MxClient(server_url=os.environ[_LIVE_SERVER_ENV])
    model_name = f"mx-live-fanin-replica-{uuid.uuid4().hex[:8]}"
    base_identity = _base_identity(model_name)
    publisher_r0 = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"publisher-r0-{uuid.uuid4().hex[:8]}",
        device_id=0,
        timeout_seconds=20.0,
    )
    publisher_r1 = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"publisher-r1-{uuid.uuid4().hex[:8]}",
        device_id=1,
        timeout_seconds=20.0,
    )
    replica = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"replica-{uuid.uuid4().hex[:8]}",
        device_id=2,
        timeout_seconds=20.0,
    )
    restarted_receiver = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"restart-{uuid.uuid4().hex[:8]}",
        device_id=2,
        timeout_seconds=20.0,
    )

    try:
        publisher_r0.publish_tensors(
            {
                "experts.w": torch.tensor(
                    [[1.0, 2.0]],
                    dtype=torch.float32,
                    device="cuda:0",
                )
            },
            model_version=19,
            worker_rank=0,
            source_world_size=2,
            tensor_metadata={
                "experts.w": {
                    "global_shape": [2, 2],
                    "shard_offsets": [0, 0],
                    "expert_ids": [0],
                    "expert_axis": 0,
                }
            },
        )
        publisher_r1.publish_tensors(
            {
                "experts.w": torch.tensor(
                    [[3.0, 4.0]],
                    dtype=torch.float32,
                    device="cuda:1",
                )
            },
            model_version=19,
            worker_rank=1,
            source_world_size=2,
            tensor_metadata={
                "experts.w": {
                    "global_shape": [2, 2],
                    "shard_offsets": [1, 0],
                    "expert_ids": [1],
                    "expert_axis": 0,
                }
            },
        )

        replica_tensors = asyncio.run(
            replica.receive_tensors_and_publish_replica(
                model_version=None,
                receiver_rank=0,
                roles=(RlSourceRole.TRAINER,),
                replica_world_size=1,
            )
        )
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        torch.cuda.synchronize(2)

        assert replica_tensors[0][0] == "experts.w"
        assert replica_tensors[0][1].device.index == 2
        assert replica_tensors[0][1].detach().cpu().tolist() == [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
        refs = client.list_sources(
            identity=None,
            status_filter=p2p_pb2.SOURCE_STATUS_READY,
        )
        replica_metadata = []
        for ref in refs.instances:
            if not ref.HasField("identity") or ref.identity.model_name != model_name:
                continue
            metadata = get_rl_source_metadata(ref.identity)
            if metadata.role == RlSourceRole.INFERENCE_REPLICA:
                replica_metadata.append(metadata)
        assert len(replica_metadata) == 1
        assert replica_metadata[0].shape_registry["experts.w"] == {
            "shape": [2, 2],
            "dtype": "torch.float32",
            "global_shape": [2, 2],
            "shard_offsets": [0, 0],
            "tensor_parallel_rank": 0,
            "pipeline_parallel_rank": 0,
            "expert_ids": [0, 1],
            "expert_axis": 0,
        }

        publisher_r1.finalize()
        publisher_r0.finalize()

        expert_one = torch.zeros((1, 2), dtype=torch.float32, device="cuda:2")
        recovered = asyncio.run(
            restarted_receiver.receive_into_tensors(
                {"experts.w": expert_one},
                model_version=None,
                receiver_rank=0,
                roles=(RlSourceRole.INFERENCE_REPLICA,),
                target_specs=[
                    TensorReceiveSpec(
                        name="experts.w",
                        receiver_rank=0,
                        shape=(1, 2),
                        dtype="torch.float32",
                        global_shape=(2, 2),
                        shard_offsets=(0, 0),
                        expert_ids=(1,),
                        expert_axis=0,
                    )
                ],
            )
        )
        torch.cuda.synchronize(2)

        assert recovered == [("experts.w", expert_one)]
        assert expert_one.detach().cpu().tolist() == [[3.0, 4.0]]
        assert restarted_receiver.last_receive_report is not None
        assert restarted_receiver.last_receive_report.resolved_model_version == 19
        assert restarted_receiver.last_receive_report.attempts[0].role == (
            RlSourceRole.INFERENCE_REPLICA
        )
    finally:
        restarted_receiver.finalize()
        replica.finalize()
        publisher_r1.finalize()
        publisher_r0.finalize()
        client.close()


def test_live_server_summarizes_failed_transfer_lease_cuda_version():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("nixl._api")

    client = MxClient(server_url=os.environ[_LIVE_SERVER_ENV])
    model_name = f"mx-live-failed-lease-{uuid.uuid4().hex[:8]}"
    base_identity = _base_identity(model_name)
    publisher = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"publisher-{uuid.uuid4().hex[:8]}",
        device_id=0,
        timeout_seconds=20.0,
    )
    receiver = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"receiver-{uuid.uuid4().hex[:8]}",
        device_id=0,
        timeout_seconds=20.0,
    )

    try:
        publisher.publish_tensors(
            {"w": torch.tensor([1.0], dtype=torch.float32, device="cuda:0")},
            model_version=13,
        )

        with pytest.raises(
            RuntimeError,
            match="No ModelExpress RL source transfer succeeded",
        ):
            asyncio.run(
                receiver.receive_into_tensors(
                    {"missing": torch.zeros((1,), dtype=torch.float32, device="cuda:0")},
                    model_version=None,
                    receiver_rank=0,
                    target_specs=[
                        TensorReceiveSpec(
                            name="missing",
                            receiver_rank=0,
                            shape=(1,),
                            dtype="torch.float32",
                        )
                    ],
                )
            )

        assert receiver.last_receive_report is not None
        attempt = receiver.last_receive_report.attempts[0]
        assert not attempt.success
        if not attempt.lease_id:
            pytest.skip("server does not expose transfer lease RPCs")

        inventory = receiver.list_target_transfer_leases(
            statuses=(p2p_pb2.TRANSFER_LEASE_STATUS_FAILED,),
        )
        summary = summarize_report_leases(receiver.last_receive_report, inventory)

        assert summary.report_lease_ids == (attempt.lease_id,)
        assert summary.missing_lease_ids == ()
        assert [lease.lease_id for lease in summary.matching_leases] == [
            attempt.lease_id
        ]
        assert [
            lease.status for lease in summary.non_completed_matching_leases
        ] == [p2p_pb2.TRANSFER_LEASE_STATUS_FAILED]
    finally:
        receiver.finalize()
        publisher.finalize()
        client.close()


def test_live_server_recovers_from_republished_inference_replica():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("nixl._api")

    client = MxClient(server_url=os.environ[_LIVE_SERVER_ENV])
    model_name = f"mx-live-replica-{uuid.uuid4().hex[:8]}"
    base_identity = _base_identity(model_name)
    publisher = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"publisher-{uuid.uuid4().hex[:8]}",
        device_id=0,
        timeout_seconds=20.0,
    )
    replica = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"replica-{uuid.uuid4().hex[:8]}",
        device_id=0,
        timeout_seconds=20.0,
    )
    restarted_receiver = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"restart-{uuid.uuid4().hex[:8]}",
        device_id=0,
        timeout_seconds=20.0,
    )

    try:
        source_tensor = torch.tensor([4.0, 5.0, 6.0], device="cuda:0")
        publisher.publish_tensors({"w": source_tensor}, model_version=4)

        replica_tensor = torch.zeros_like(source_tensor)
        first_receive = asyncio.run(
            replica.receive_and_publish_replica(
                {"w": replica_tensor},
                model_version=None,
                receiver_rank=0,
                replica_world_size=1,
            )
        )
        torch.cuda.synchronize(0)
        assert first_receive == [("w", replica_tensor)]
        assert replica_tensor.detach().cpu().tolist() == [4.0, 5.0, 6.0]

        publisher.finalize()

        recovered = asyncio.run(
            restarted_receiver.receive_tensors(model_version=None, receiver_rank=0)
        )
        torch.cuda.synchronize(0)

        assert recovered[0][0] == "w"
        assert recovered[0][1].detach().cpu().tolist() == [4.0, 5.0, 6.0]
        assert restarted_receiver.last_receive_report is not None
        assert restarted_receiver.last_receive_report.resolved_model_version == 4
        assert (
            restarted_receiver.last_receive_report.attempts[0].role
            == RlSourceRole.INFERENCE_REPLICA
        )
    finally:
        restarted_receiver.finalize()
        replica.finalize()
        publisher.finalize()
        client.close()


def test_live_server_tree_fanout_uses_parent_replica_source():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("nixl._api")

    client = MxClient(server_url=os.environ[_LIVE_SERVER_ENV])
    model_name = f"mx-live-tree-{uuid.uuid4().hex[:8]}"
    base_identity = _base_identity(model_name)
    publisher = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"publisher-{uuid.uuid4().hex[:8]}",
        device_id=0,
        timeout_seconds=20.0,
    )
    parent = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"parent-{uuid.uuid4().hex[:8]}",
        device_id=0,
        timeout_seconds=20.0,
    )
    child = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"child-{uuid.uuid4().hex[:8]}",
        device_id=0,
        timeout_seconds=20.0,
    )

    try:
        source_tensor = torch.tensor([7.0, 8.0, 9.0], device="cuda:0")
        publisher.publish_tensors({"w": source_tensor}, model_version=9)

        root_policy = RlTreeFanoutPolicy(
            receiver_rank=0,
            replica_world_size=3,
            fanout=2,
        )
        parent_tensor = torch.zeros_like(source_tensor)
        asyncio.run(
            parent.receive_and_publish_replica(
                {"w": parent_tensor},
                model_version=None,
                receiver_rank=0,
                roles=root_policy.roles,
                same_rank_only=False,
                source_ranks_by_role=root_policy.source_ranks_by_role,
                require_complete_version=root_policy.parent_replica_rank is None,
                replica_world_size=3,
            )
        )
        torch.cuda.synchronize(0)
        assert parent_tensor.detach().cpu().tolist() == [7.0, 8.0, 9.0]

        child_policy = RlTreeFanoutPolicy(
            receiver_rank=2,
            replica_world_size=3,
            fanout=2,
        )
        child_tensor = torch.zeros_like(source_tensor)
        received = asyncio.run(
            child.receive_into_tensors(
                {"w": child_tensor},
                model_version=None,
                receiver_rank=2,
                roles=child_policy.roles,
                same_rank_only=False,
                source_ranks_by_role=child_policy.source_ranks_by_role,
                require_complete_version=child_policy.parent_replica_rank is None,
            )
        )
        torch.cuda.synchronize(0)

        assert received == [("w", child_tensor)]
        assert child_tensor.detach().cpu().tolist() == [7.0, 8.0, 9.0]
        assert child.last_receive_report is not None
        attempt = child.last_receive_report.attempts[0]
        assert attempt.role == RlSourceRole.INFERENCE_REPLICA
        assert attempt.worker_rank == 0
    finally:
        child.finalize()
        parent.finalize()
        publisher.finalize()
        client.close()
