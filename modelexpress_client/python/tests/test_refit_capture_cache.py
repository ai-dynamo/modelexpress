# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import sys
from contextlib import nullcontext
from types import ModuleType

from uuid import uuid4

import pytest
import torch
from modelexpress_rl.inference.engines.vllm import installer as module
from torch import nn


@pytest.fixture
def capture_installer(monkeypatch):
    names = (
        "vllm",
        "vllm.config",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.reload",
        "vllm.model_executor.model_loader.reload.layerwise",
    )
    modules = {name: ModuleType(name) for name in names}
    modules["vllm.config"].set_current_vllm_config = lambda config: nullcontext()
    layerwise = modules["vllm.model_executor.model_loader.reload.layerwise"]
    layerwise.LAYERWISE_INFO = {}
    layerwise.initialize_layerwise_reload = lambda model: None
    layerwise._get_original_loader = lambda parameter: None
    layerwise._place_kernel_tensors = lambda layer, info: None
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    weight_utils = ModuleType("vllm.model_executor.model_loader.weight_utils")
    weight_utils.default_weight_loader = lambda parameter, weight: parameter.data.copy_(
        weight
    )
    monkeypatch.setitem(sys.modules, weight_utils.__name__, weight_utils)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, 2))
            self.capture_calls = 0

        def load_weights(self, weights):
            self.capture_calls += 1
            for name, weight in weights:
                self.weight.weight_loader(self.weight, weight[:2, :2])

    model = Model()
    installer = module._VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    return installer, model


def test_capture_cache_isolates_slicing_metadata_and_retains_invalidation(
    capture_installer,
):
    installer, model = capture_installer
    manifest = [("weight", torch.float32, (3, 3))]
    cold, _ = installer.capture(manifest)
    assert cold.copies[0].op_chain[0][0] == "__getitem__"
    original = copy.deepcopy(cold)
    cold.copies[0].dest_offset = 55
    first, layout = installer.capture(manifest)
    assert first == original and model.capture_calls == 1
    first.copies[0].op_chain = (("view", ([4],), ()),)
    first.copies.clear()
    first.destination_layouts.clear()
    first.replay_id = uuid4()
    first.source_replay = False
    first.generator_fingerprint = None
    layout.clear()
    assert installer.capture(manifest)[0] == original
    assert model.capture_calls == 1
    installer.capture([("weight", torch.float32, (4, 4))])
    assert model.capture_calls == 2
    model.weight = nn.Parameter(torch.zeros(1, 2))
    changed, changed_layout = installer.capture(manifest)
    assert model.capture_calls == 3
    assert changed.copies[0].dest_shape == (1, 2)
    assert changed_layout == {"weight": ((1, 2), torch.float32)}
    assert installer.capture(manifest) == (changed, changed_layout)
    assert model.capture_calls == 3


def test_failed_fresh_capture_discards_cache(capture_installer, monkeypatch):
    installer, _ = capture_installer
    installer.capture([("weight", torch.float32, (3, 3))])

    def fail(*args, **kwargs):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(module, "capture_weights", fail)
    with pytest.raises(RuntimeError, match="capture failed"):
        installer.capture([("weight", torch.float32, (4, 4))])
    assert installer.source_install_plan is None


def test_real_capture_cache_preserves_duplicate_record_identity(
    capture_installer, monkeypatch
):
    installer, _ = capture_installer
    capture = module.capture_weights

    def duplicate(*args, **kwargs):
        result = capture(*args, **kwargs)
        result.copies.append(result.copies[0])
        return result

    monkeypatch.setattr(module, "capture_weights", duplicate)
    manifest = [("weight", torch.float32, (3, 3))]
    installer.capture(manifest)
    warm, _ = installer.capture(manifest)
    assert warm.copies[0] is warm.copies[1]


def test_real_capture_cache_nested_mutable_payload_remains_isolated(
    capture_installer, monkeypatch
):
    installer, _ = capture_installer
    capture = module.capture_weights

    def mutable(*args, **kwargs):
        result = capture(*args, **kwargs)
        result.copies[0].op_chain = (("view", ([4],), ()),)
        return result

    monkeypatch.setattr(module, "capture_weights", mutable)
    manifest = [("weight", torch.float32, (3, 3))]
    installer.capture(manifest)
    first, _ = installer.capture(manifest)
    first.copies[0].op_chain[0][1][0][0] = 99
    assert installer.capture(manifest)[0].copies[0].op_chain[0][1][0][0] == 4


@pytest.mark.parametrize("tensor_index", [False, True])
def test_integer_index_payload_deepcopy_remains_isolated(
    capture_installer, monkeypatch, tensor_index
):
    installer, _ = capture_installer
    capture = module.capture_weights
    index = torch.tensor([2, 0, 2]) if tensor_index else [2, 0, 2]

    def indexed(*args, **kwargs):
        result = capture(*args, **kwargs)
        result.copies[0].op_chain = (("__getitem__", (index,), ()),)
        return result

    monkeypatch.setattr(module, "capture_weights", indexed)
    manifest = [("weight", torch.float32, (3, 3))]
    installer.capture(manifest)
    index[0] = 1
    first, _ = installer.capture(manifest)
    first_index = first.copies[0].op_chain[0][1][0]
    assert list(first_index) == [2, 0, 2]
    first_index[0] = 0
    second, _ = installer.capture(manifest)
    assert list(second.copies[0].op_chain[0][1][0]) == [2, 0, 2]
