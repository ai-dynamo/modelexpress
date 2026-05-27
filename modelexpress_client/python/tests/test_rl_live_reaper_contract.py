# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import uuid

import grpc
import pytest

from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from modelexpress.rl_transfer import build_rl_base_identity

_LIVE_SERVER_ENV = "MX_LIVE_SERVER_URL"
_LIVE_TRANSFER_LEASE_GC_ENV = "MX_LIVE_TRANSFER_LEASE_GC_EXPECTED"
_LIVE_WORKER_REAPER_GC_ENV = "MX_LIVE_WORKER_REAPER_GC_EXPECTED"
_LIVE_WORKER_REAPER_STALE_ENV = "MX_LIVE_WORKER_REAPER_STALE_EXPECTED"

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


def test_live_server_reaper_garbage_collects_terminal_transfer_leases():
    if not os.environ.get(_LIVE_TRANSFER_LEASE_GC_ENV):
        pytest.skip(
            f"{_LIVE_TRANSFER_LEASE_GC_ENV} is not set; "
            "requires a server started with a short MX_TRANSFER_LEASE_GC_TIMEOUT_SECS"
        )

    client = MxClient(server_url=os.environ[_LIVE_SERVER_ENV])
    lease_id = f"lease-terminal-gc-{uuid.uuid4().hex}"
    mx_source_id = f"live-lease-gc-source-{uuid.uuid4().hex[:8]}"
    target_worker_id = f"target-gc-{uuid.uuid4().hex[:8]}"

    try:
        lease = _begin_transfer_lease_or_skip(
            client,
            lease_id=lease_id,
            mx_source_id=mx_source_id,
            source_worker_id="source-worker",
            target_worker_id=target_worker_id,
            target_worker_rank=3,
            model_version=29,
            ttl_millis=5_000,
            metadata={"contract": "terminal-gc"},
        )
        assert lease.status == p2p_pb2.TRANSFER_LEASE_STATUS_ACTIVE

        completed = client.complete_transfer_lease(
            lease_id,
            status=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
        )
        assert completed.status == p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED

        deadline = time.monotonic() + 5.0
        while True:
            listed = client.list_transfer_leases(
                mx_source_id=mx_source_id,
                target_worker_id=target_worker_id,
                status_filter=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
            )
            if listed.leases == []:
                break
            if time.monotonic() >= deadline:
                pytest.fail(
                    "terminal transfer lease was not garbage-collected "
                    "within the live contract deadline"
                )
            time.sleep(0.1)

        fetched = client.get_transfer_lease(lease_id)
        assert not fetched.found
    finally:
        client.close()


def test_live_server_reaper_garbage_collects_stale_workers():
    if not os.environ.get(_LIVE_WORKER_REAPER_GC_ENV):
        pytest.skip(
            f"{_LIVE_WORKER_REAPER_GC_ENV} is not set; "
            "requires a server started with a short MX_GC_TIMEOUT_SECS"
        )

    client = MxClient(server_url=os.environ[_LIVE_SERVER_ENV])
    model_name = f"mx-live-worker-gc-{uuid.uuid4().hex[:8]}"
    base_identity = _base_identity(model_name)
    worker_id = f"stale-worker-{uuid.uuid4().hex[:8]}"
    old_updated_at = int(time.time() * 1000) - 7_200_000

    try:
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=0,
            nixl_metadata=b"worker-gc-contract",
            tensors=[],
            status=p2p_pb2.SOURCE_STATUS_STALE,
            updated_at=old_updated_at,
        )
        source_id = client.publish_metadata(base_identity, worker, worker_id)

        fetched = client.get_metadata(source_id, worker_id)
        assert fetched.found
        assert fetched.worker.status == p2p_pb2.SOURCE_STATUS_STALE
        assert fetched.worker.updated_at == old_updated_at

        stale = client.list_sources(
            identity=base_identity,
            status_filter=p2p_pb2.SOURCE_STATUS_STALE,
        )
        assert any(ref.worker_id == worker_id for ref in stale.instances)

        deadline = time.monotonic() + 5.0
        while True:
            fetched = client.get_metadata(source_id, worker_id)
            if not fetched.found:
                break
            if time.monotonic() >= deadline:
                pytest.fail(
                    "stale worker metadata was not garbage-collected "
                    "within the live contract deadline"
                )
            time.sleep(0.1)

        stale = client.list_sources(
            identity=base_identity,
            status_filter=p2p_pb2.SOURCE_STATUS_STALE,
        )
        assert all(ref.worker_id != worker_id for ref in stale.instances)
    finally:
        client.close()


def test_live_server_reaper_marks_workers_stale_after_heartbeat_timeout():
    if not os.environ.get(_LIVE_WORKER_REAPER_STALE_ENV):
        pytest.skip(
            f"{_LIVE_WORKER_REAPER_STALE_ENV} is not set; "
            "requires a server started with a short MX_HEARTBEAT_TIMEOUT_SECS"
        )

    client = MxClient(server_url=os.environ[_LIVE_SERVER_ENV])
    model_name = f"mx-live-worker-stale-{uuid.uuid4().hex[:8]}"
    base_identity = _base_identity(model_name)
    worker_id = f"ready-worker-{uuid.uuid4().hex[:8]}"
    old_updated_at = int(time.time() * 1000) - 120_000

    try:
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=0,
            nixl_metadata=b"worker-stale-contract",
            tensors=[],
            status=p2p_pb2.SOURCE_STATUS_READY,
            updated_at=old_updated_at,
        )
        source_id = client.publish_metadata(base_identity, worker, worker_id)

        fetched = client.get_metadata(source_id, worker_id)
        assert fetched.found
        assert fetched.worker.status == p2p_pb2.SOURCE_STATUS_READY
        assert fetched.worker.updated_at == old_updated_at

        ready = client.list_sources(
            identity=base_identity,
            status_filter=p2p_pb2.SOURCE_STATUS_READY,
        )
        ready_ref = next(ref for ref in ready.instances if ref.worker_id == worker_id)
        assert ready_ref.status == p2p_pb2.SOURCE_STATUS_READY
        assert ready_ref.updated_at == old_updated_at

        deadline = time.monotonic() + 5.0
        while True:
            fetched = client.get_metadata(source_id, worker_id)
            assert fetched.found
            if fetched.worker.status == p2p_pb2.SOURCE_STATUS_STALE:
                break
            if time.monotonic() >= deadline:
                pytest.fail(
                    "ready worker metadata was not marked stale "
                    "within the live contract deadline"
                )
            time.sleep(0.1)

        assert fetched.worker.updated_at > old_updated_at

        stale = client.list_sources(
            identity=base_identity,
            status_filter=p2p_pb2.SOURCE_STATUS_STALE,
        )
        stale_ref = next(ref for ref in stale.instances if ref.worker_id == worker_id)
        assert stale_ref.status == p2p_pb2.SOURCE_STATUS_STALE
        assert stale_ref.updated_at == fetched.worker.updated_at

        ready = client.list_sources(
            identity=base_identity,
            status_filter=p2p_pb2.SOURCE_STATUS_READY,
        )
        assert all(ref.worker_id != worker_id for ref in ready.instances)
    finally:
        client.close()
