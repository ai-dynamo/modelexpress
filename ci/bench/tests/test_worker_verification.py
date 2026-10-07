# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The worker verifies installed weights through the engine's shard loader."""

import importlib
import json
import struct
import sys
from types import SimpleNamespace

import pytest


class Tensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)
        self.dtype = "bfloat16"
        self.output_dim = 0

    def detach(self):
        return self

    cpu = detach
    contiguous = detach
    numpy = detach

    def view(self, dtype):
        return self

    def tobytes(self):
        return struct.pack(f"{len(self.values)}f", *self.values)


@pytest.mark.parametrize("installed,verified", [([3, 4, 0], True), ([3, 9, 0], False)])
def test_runtime_tensor_uses_engine_sharding_without_mutating_weight(
    monkeypatch, tmp_path, installed, verified
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "weights.safetensors"}})
    )
    live = Tensor(installed)
    loader_calls = []

    class VocabParallelEmbedding:
        weight = live

        def weight_loader(self, parameter, raw):
            assert parameter is not self.weight
            assert parameter.output_dim == self.weight.output_dim
            loader_calls.append(raw.values)
            parameter.values = raw.values[2:4] + [0]

    class CheckpointStore:
        def __init__(self, **kwargs):
            pass

        def checkpoint_path(self, version_id):
            assert version_id == "arbitrary-target"
            return checkpoint

    class SafeFile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_tensor(self, name):
            assert name == "model.embed_tokens.weight"
            return Tensor([1, 2, 3, 4])

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            nn=SimpleNamespace(Parameter=lambda tensor, **kwargs: tensor),
            empty_like=lambda tensor, **kwargs: Tensor([0] * len(tensor.values)),
            equal=lambda left, right: left.values == right.values,
            uint8="uint8",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "modelexpress_rl.inference.checkpoint_store",
        SimpleNamespace(LocalCheckpointStore=CheckpointStore),
    )
    monkeypatch.setitem(
        sys.modules,
        "safetensors",
        SimpleNamespace(safe_open=lambda *a, **k: SafeFile()),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.vocab_parallel_embedding",
        SimpleNamespace(VocabParallelEmbedding=VocabParallelEmbedding),
    )
    monkeypatch.setenv("BENCH_MODEL", "test/model")
    monkeypatch.delitem(sys.modules, "engines.vllm.worker", raising=False)
    module = importlib.import_module("engines.vllm.worker")
    worker = module.RefitWorkerExtension()
    worker.model_runner = SimpleNamespace(
        get_model=lambda: SimpleNamespace(
            get_submodule=lambda name: VocabParallelEmbedding()
        )
    )

    def record(self, phase, body):
        self._last_record = dict(body, phase=phase)

    monkeypatch.setattr(module.RefitWorkerExtension, "_record", record)
    result = worker.verify_runtime_tensor(
        "arbitrary-target", "model.embed_tokens.weight"
    )
    assert result["phase"] == "runtime-tensor-verified"
    assert result["verified"] is verified
    assert (result["actual_sha256"] == result["expected_sha256"]) is verified
    assert loader_calls == [[1, 2, 3, 4]]
    assert live.values == installed

    live.packed_dim = 0
    failed = worker.verify_runtime_tensor(
        "arbitrary-target", "model.embed_tokens.weight"
    )
    assert failed["phase"] == "runtime-tensor-verified-failed"
    assert failed["verified"] is False
    assert "packed embedding" in failed["error"]
