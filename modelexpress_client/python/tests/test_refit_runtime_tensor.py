# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from modelexpress import p2p_pb2

from modelexpress_rl import WeightPayloadFormat
from modelexpress_rl.inference.methods import (
    LoadTimeTensorNixlUpdateMethod,
    RuntimeTensorNixlUpdateMethod,
)
from modelexpress_rl.inference.plan import (
    GeneratorPeerUpdateSource,
    PreparedRuntimeTensors,
    TrainerUpdateSource,
    WeightSource,
)


class _Transfer:
    def __init__(self):
        self.prepared_tensors = None
        self.receive_tensors = None
        self.fail_before_start = False
        self.lease = type(
            "Lease",
            (),
            {
                "closed": False,
                "close": lambda lease: setattr(lease, "closed", True),
            },
        )()
        self.closed = False

    def prepare_peer_read(
        self, *, source, mx_source_id, worker_id, destination_tensors
    ):
        self.mx_source_id = mx_source_id
        self.worker_id = worker_id
        self.prepared_tensors = destination_tensors
        return self.lease

    def receive_peer(
        self, *, tensor_read, destination_tensors, on_transfer_start
    ):
        assert tensor_read is self.lease
        self.receive_tensors = destination_tensors
        if self.fail_before_start:
            raise RuntimeError("peer disappeared before transfer")
        on_transfer_start()
        return {"bytes_received": 16, "wire_s": 0.25}

    def close(self):
        self.closed = True


def test_load_time_and_runtime_methods_have_disjoint_source_contracts():
    load_time = LoadTimeTensorNixlUpdateMethod(
        transfer=_Transfer(),
        capture_layout=lambda manifest: manifest,
    )
    runtime = RuntimeTensorNixlUpdateMethod(
        transfer=_Transfer(),
        runtime_tensors={},
    )

    assert load_time.capabilities.sources == frozenset({WeightSource.TRAINER})
    assert runtime.capabilities.sources == frozenset({WeightSource.GENERATOR})


def test_runtime_method_defers_direct_receive_until_installation():
    transfer = _Transfer()
    runtime_tensors = {
        "model.weight": torch.empty((2, 3), dtype=torch.bfloat16),
        "model._mx_runtime_buffer": torch.empty(4, dtype=torch.float32),
    }
    method = RuntimeTensorNixlUpdateMethod(
        transfer=transfer,
        runtime_tensors=runtime_tensors,
    )
    source = GeneratorPeerUpdateSource(
        worker=p2p_pb2.WorkerMetadata(),
        mx_source_id="source-1",
        worker_id="worker-1",
    )

    prepared = method.prepare(version=object(), source=source)

    assert isinstance(prepared, PreparedRuntimeTensors)
    assert transfer.prepared_tensors is runtime_tensors
    assert transfer.receive_tensors is None
    assert transfer.lease.closed is False
    assert transfer.mx_source_id == "source-1"
    assert transfer.worker_id == "worker-1"
    assert method.capabilities.payload_formats == frozenset(
        {WeightPayloadFormat.FULL_TENSOR}
    )

    with method.installation_context(prepared):
        assert transfer.receive_tensors is runtime_tensors
        assert method.mutated_during_installation_context(prepared) is True

    assert prepared.metrics["bytes_received"] == 16
    assert transfer.lease.closed is True


def test_runtime_method_keeps_pretransfer_failure_recoverable():
    transfer = _Transfer()
    transfer.fail_before_start = True
    method = RuntimeTensorNixlUpdateMethod(
        transfer=transfer,
        runtime_tensors={"model.weight": torch.empty(1)},
    )
    prepared = method.prepare(
        version=object(),
        source=GeneratorPeerUpdateSource(
            worker=p2p_pb2.WorkerMetadata(),
            mx_source_id="source-1",
            worker_id="worker-1",
        ),
    )

    with pytest.raises(RuntimeError, match="before transfer"):
        with method.installation_context(prepared):
            pass

    assert method.mutated_during_installation_context(prepared) is False
    assert transfer.lease.closed is True


def test_runtime_method_releases_reserved_peer_without_apply():
    transfer = _Transfer()
    method = RuntimeTensorNixlUpdateMethod(
        transfer=transfer,
        runtime_tensors={"model.weight": torch.empty(1)},
    )
    prepared = method.prepare(
        version=object(),
        source=GeneratorPeerUpdateSource(
            worker=p2p_pb2.WorkerMetadata(),
            mx_source_id="source-1",
            worker_id="worker-1",
        ),
    )

    method.release(prepared)

    assert transfer.lease.closed is True


def test_runtime_method_rejects_a_trainer_source():
    method = RuntimeTensorNixlUpdateMethod(
        transfer=_Transfer(),
        runtime_tensors={},
    )
    source = TrainerUpdateSource(inputs=object())

    try:
        method.prepare(version=object(), source=source)
    except TypeError as error:
        assert "generator" in str(error).lower()
    else:
        raise AssertionError("runtime tensor method accepted a trainer source")
