# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
        self.peer_layout = None
        self.closed = False

    def stage_peer(self, *, source, parameter_layout):
        self.peer_layout = parameter_layout
        return type("Staged", (), {"tensors": {}, "metrics": {}})()

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


def test_runtime_method_stages_the_complete_post_pwal_layout():
    transfer = _Transfer()
    runtime_tensors = {
        "model.weight": torch.empty((2, 3), dtype=torch.bfloat16),
        "model._mx_runtime_buffer": torch.empty(4, dtype=torch.float32),
    }
    method = RuntimeTensorNixlUpdateMethod(
        transfer=transfer,
        runtime_tensors=runtime_tensors,
    )
    source = GeneratorPeerUpdateSource(worker=p2p_pb2.WorkerMetadata())

    prepared = method.prepare(version=object(), source=source)

    assert isinstance(prepared, PreparedRuntimeTensors)
    assert transfer.peer_layout == {
        "model.weight": ((2, 3), torch.bfloat16),
        "model._mx_runtime_buffer": ((4,), torch.float32),
    }
    assert method.capabilities.payload_formats == frozenset(
        {WeightPayloadFormat.FULL_TENSOR}
    )


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
