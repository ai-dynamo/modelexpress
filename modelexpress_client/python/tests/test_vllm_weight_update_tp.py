# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent / "modelexpress"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def weight_update():
    pkg = types.ModuleType("modelexpress")
    pkg.__path__ = [str(_PKG_ROOT)]
    sys.modules["modelexpress"] = pkg
    _load("modelexpress.refit_timing", _PKG_ROOT / "refit_timing.py")

    base = types.ModuleType("vllm.distributed.weight_transfer.base")
    base.WeightTransferInitInfo = type("WeightTransferInitInfo", (), {})
    base.WeightTransferUpdateInfo = type("WeightTransferUpdateInfo", (), {})
    sys.modules["vllm"] = types.ModuleType("vllm")
    sys.modules["vllm.distributed"] = types.ModuleType("vllm.distributed")
    sys.modules["vllm.distributed.weight_transfer"] = types.ModuleType(
        "vllm.distributed.weight_transfer"
    )
    sys.modules["vllm.distributed.weight_transfer.base"] = base
    return _load(
        "modelexpress.engines.vllm.weight_update",
        _PKG_ROOT / "engines" / "vllm" / "weight_update.py",
    )


def _descriptor(name, shape, placement="REPLICATE", axis=0, local_range=None):
    return types.SimpleNamespace(
        name=name,
        global_shape=shape,
        placement_kind=placement,
        shard_axis=axis,
        local_shard_range=local_range,
    )


def test_dtensor_receive_restores_embedding_lm_head_and_tp_shapes(weight_update):
    names = {
        "model.embed_tokens.weight": (10, 4),
        "lm_head.weight": (10, 4),
        "model.layers.0.mlp.down_proj.weight": (4, 8),
    }

    class Scratch:
        def receive_weights_scratch(self, _ref, *, tensor_shapes, **_kwargs):
            assert tensor_shapes == names
            for name, shape in tensor_shapes.items():
                yield name, torch.arange(torch.tensor(shape).prod()).view(shape)

    updater = weight_update.MxVllmWeightUpdater()
    updater._receiver = types.SimpleNamespace(_receiver=Scratch())
    candidate = types.SimpleNamespace(
        ref=object(),
        registry={
            "tensors": [
                _descriptor(name, shape) for name, shape in names.items()
            ]
        },
    )
    update = weight_update.MxUpdateInfo(timeout_seconds=1)

    received = dict(updater._receive_dtensor([candidate], update))
    assert {name: tuple(tensor.shape) for name, tensor in received.items()} == names


def test_dtensor_receive_prunes_wire_manifest_for_subset(weight_update):
    names = {
        "model.embed_tokens.weight": (10, 4),
        "model.layers.0.mlp.down_proj.weight": (4, 8),
        "model.layers.1.mlp.down_proj.weight": (4, 8),
        "model.layers.5.mlp.down_proj.weight": (4, 8),
        "model.layers.6.mlp.down_proj.weight": (4, 8),
    }
    captured = {}

    class Scratch:
        def receive_weights_scratch(
            self, _ref, *, tensor_shapes, include_names=None, **_kwargs
        ):
            captured["tensor_shapes"] = tensor_shapes
            captured["include_names"] = include_names
            for name in include_names or tensor_shapes:
                yield name, torch.empty(tensor_shapes[name])

    updater = weight_update.MxVllmWeightUpdater()
    updater._receiver = types.SimpleNamespace(_receiver=Scratch())
    candidate = types.SimpleNamespace(
        ref=object(),
        registry={
            "tensors": [
                _descriptor(name, shape) for name, shape in names.items()
            ]
        },
    )
    subset = weight_update.WeightSubset(layers=[0, 1, 5])

    received = dict(
        updater._receive_dtensor(
            [candidate],
            weight_update.MxUpdateInfo(timeout_seconds=1),
            subset=subset,
        )
    )
    expected = {
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
        "model.layers.5.mlp.down_proj.weight",
    }
    assert captured["tensor_shapes"] == names
    assert captured["include_names"] == expected
    assert set(received) == expected


def test_dtensor_receive_does_not_filter_wire_without_subset(weight_update):
    names = {
        "model.embed_tokens.weight": (10, 4),
        "model.layers.0.mlp.down_proj.weight": (4, 8),
    }
    captured = {}

    class Scratch:
        def receive_weights_scratch(
            self, _ref, *, tensor_shapes, include_names=None, **_kwargs
        ):
            captured["include_names"] = include_names
            for name, shape in tensor_shapes.items():
                yield name, torch.empty(shape)

    updater = weight_update.MxVllmWeightUpdater()
    updater._receiver = types.SimpleNamespace(_receiver=Scratch())
    candidate = types.SimpleNamespace(
        ref=object(),
        registry={
            "tensors": [
                _descriptor(name, shape) for name, shape in names.items()
            ]
        },
    )

    received = dict(
        updater._receive_dtensor(
            [candidate], weight_update.MxUpdateInfo(timeout_seconds=1)
        )
    )
    assert captured["include_names"] is None
    assert set(received) == set(names)


def test_dtensor_receive_empty_subset_skips_wire(weight_update):
    called = False

    class Scratch:
        def receive_weights_scratch(self, *_args, **_kwargs):
            nonlocal called
            called = True
            yield from ()

    updater = weight_update.MxVllmWeightUpdater()
    updater._receiver = types.SimpleNamespace(_receiver=Scratch())
    candidate = types.SimpleNamespace(
        ref=object(),
        registry={
            "tensors": [
                _descriptor(
                    "model.layers.0.mlp.down_proj.weight", (4, 8)
                )
            ]
        },
    )

    received = updater._receive_dtensor(
        [candidate],
        weight_update.MxUpdateInfo(timeout_seconds=1),
        subset=weight_update.WeightSubset(layers=[99]),
    )
    assert received == []
    assert called is False


def test_dtensor_receive_uses_local_wire_shape_for_sharded_tensor(weight_update):
    descriptor = _descriptor(
        "model.layers.0.self_attn.q_proj.weight",
        (8, 8),
        placement="SHARD",
        axis=0,
        local_range=(4, 8),
    )

    class Scratch:
        def receive_weights_scratch(self, _ref, *, tensor_shapes, **_kwargs):
            assert tensor_shapes[descriptor.name] == (4, 8)
            yield descriptor.name, torch.empty(4, 8)

    updater = weight_update.MxVllmWeightUpdater()
    updater._receiver = types.SimpleNamespace(_receiver=Scratch())
    candidate = types.SimpleNamespace(
        ref=object(), registry={"tensors": [descriptor]}
    )
    received = updater._receive_dtensor(
        [candidate], weight_update.MxUpdateInfo(timeout_seconds=1)
    )
    assert tuple(received[0][1].shape) == (4, 8)


class _FakeVocab(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.org_vocab_size = 10
        self.shard_indices = types.SimpleNamespace(
            org_vocab_start_index=6,
            org_vocab_end_index=10,
        )
        self.weight = torch.nn.Parameter(torch.empty(6, 4))
        self.weight.output_dim = 0


class _FakeTpModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = _FakeVocab()
        self.lm_head = _FakeVocab()
        self.proj = torch.nn.Module()
        self.proj.weight = torch.nn.Parameter(torch.empty(4, 8))
        self.proj.weight.output_dim = 0


class _FakeReplicatedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(4, 8))
        self.weight.output_dim = 0


def test_stock_tp_validation_covers_vocab_lm_head_and_ordinary_tensor(
    weight_update,
):
    model = _FakeTpModel()
    weights = [
        ("model.embed_tokens.weight", torch.empty(10, 4)),
        ("lm_head.weight", torch.empty(10, 4)),
        ("proj.weight", torch.empty(8, 8)),
    ]
    update = weight_update.MxUpdateInfo(tp_world_size=2, tp_rank=1)
    weight_update.MxVllmWeightUpdater._validate_stock_tp_weights(
        weights, update, model
    )


def test_stock_tp_validation_rejects_already_local_ordinary_tensor(weight_update):
    model = _FakeTpModel()
    update = weight_update.MxUpdateInfo(tp_world_size=2, tp_rank=1)
    with pytest.raises(RuntimeError, match="already TP-local"):
        weight_update.MxVllmWeightUpdater._validate_stock_tp_weights(
            [("proj.weight", torch.empty(4, 8))], update, model
        )


def test_stock_tp_validation_accepts_replicated_tensor(weight_update):
    model = _FakeTpModel()
    model.replicated = _FakeReplicatedLinear()
    update = weight_update.MxUpdateInfo(tp_world_size=2, tp_rank=1)
    weight_update.MxVllmWeightUpdater._validate_stock_tp_weights(
        [("replicated.weight", torch.empty(4, 8))], update, model
    )
