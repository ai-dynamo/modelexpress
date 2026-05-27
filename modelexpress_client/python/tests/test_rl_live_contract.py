# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import uuid

import pytest
import torch

from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from modelexpress.rl_metadata import (
    RlSourceMetadata,
    RlSourceRole,
    get_rl_source_metadata,
    with_rl_source_metadata,
)
from modelexpress.rl_transfer import RlNixlWeightTransfer, build_rl_base_identity

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
        assert receiver.last_receive_report.attempts[0].bytes_transferred == 12
    finally:
        receiver.finalize()
        publisher.finalize()
        client.close()
