# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from modelexpress.engines.vllm.refit import MdlLoader
from modelexpress.engines.vllm.mdl import MdlLoader as CompatibilityMdlLoader
from modelexpress.refit import RefitTimingRecorder, use_refit_timing


class _Model:
    def named_parameters(self):
        return []


def test_legacy_mdl_import_is_compatible():
    assert CompatibilityMdlLoader is MdlLoader


def _loader(rank=1):
    loader = MdlLoader(_Model())
    loader._tp_size = 2
    loader._tp_rank = rank
    return loader


def test_tp_local_tensor_axis_zero():
    tensor = torch.arange(24).reshape(6, 4)
    local = _loader(rank=1)._tp_local_tensor(tensor, (3, 4))
    assert torch.equal(local, tensor[3:6])
    assert local.is_contiguous()


def test_tp_local_tensor_axis_one():
    tensor = torch.arange(24).reshape(4, 6)
    local = _loader(rank=1)._tp_local_tensor(tensor, (4, 3))
    assert torch.equal(local, tensor[:, 3:6])
    assert local.is_contiguous()


def test_tp_shape_compatibility_requires_one_sharded_dimension():
    loader = _loader()
    assert loader._tp_shape_compatible((6, 4), (3, 4))
    assert loader._tp_shape_compatible((4, 6), (4, 3))
    assert not loader._tp_shape_compatible((6, 8), (3, 4))
    assert not loader._tp_shape_compatible((5, 4), (3, 4))


def test_mdl_reports_cold_then_warm_to_active_timing(monkeypatch):
    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(2))
            self.load_weights_calls = 0

        def named_parameters(self):
            return [("weight", self.weight)]

        def load_weights(self, *, weights):
            self.load_weights_calls += 1
            for _name, value in weights:
                self.weight.data.copy_(value)

    monkeypatch.setenv("MX_LOAD_MODE", "direct")
    model = Model()
    loader = MdlLoader(model)

    cold = RefitTimingRecorder(backend="test", version=1)
    with use_refit_timing(cold):
        loader.load_weights([("weight", torch.ones(2))])
    assert cold.as_dict()["cold_warm"] == "cold"

    warm = RefitTimingRecorder(backend="test", version=2)
    with use_refit_timing(warm):
        loader.load_weights([("weight", torch.full((2,), 2.0))])
    assert warm.as_dict()["cold_warm"] == "warm"

    model.weight = torch.nn.Parameter(torch.zeros(2))
    rebuilt = RefitTimingRecorder(backend="test", version=3)
    with use_refit_timing(rebuilt):
        loader.load_weights([("weight", torch.full((2,), 3.0))])
    assert rebuilt.as_dict()["cold_warm"] == "cold"
    assert model.load_weights_calls == 2
    assert torch.equal(model.weight, torch.full((2,), 3.0))


def test_layout_signature_does_not_depend_on_incremental_map_state():
    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(2))

        def named_parameters(self):
            return [("weight", self.weight)]

        @staticmethod
        def get_expert_mapping():
            return [("experts.w2_", "experts.0.down_proj.", 0, "w2")]

    loader = MdlLoader(Model())
    before = loader._layout_signature()
    loader._expert["experts.0.down_proj.weight"] = torch.zeros(2)
    after = loader._layout_signature()
    assert after == before


def test_layout_signature_handles_temporarily_unavailable_expert_mapping():
    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(2))
            self.mapping_available = False

        def named_parameters(self):
            return [("weight", self.weight)]

        def get_expert_mapping(self):
            if not self.mapping_available:
                raise RuntimeError("mapping is not initialized")
            return [("experts.w2_", "experts.0.down_proj.", 0, "w2")]

    model = Model()
    loader = MdlLoader(model)
    unavailable = loader._layout_signature()
    assert unavailable == loader._layout_signature()

    model.mapping_available = True
    assert loader._layout_signature() != unavailable


def test_warm_stock_fallback_runs_without_grad(monkeypatch):
    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(2))
            self.load_grad_states: list[bool] = []

        def named_parameters(self):
            return [("weight", self.weight)]

        def load_weights(self, *, weights):
            self.load_grad_states.append(torch.is_grad_enabled())
            for name, value in weights:
                if name == "weight":
                    self.weight.data.copy_(value)

    monkeypatch.setenv("MX_LOAD_MODE", "direct")
    model = Model()
    loader = MdlLoader(model)
    loader.load_weights([("weight", torch.ones(2))])
    loader.load_weights(
        [
            ("weight", torch.full((2,), 2.0)),
            ("unsupported", torch.ones(1)),
        ]
    )

    assert model.load_grad_states[-1] is False
    assert torch.equal(model.weight, torch.full((2,), 2.0))


def test_unrecognized_stock_loader_error_is_reraised(monkeypatch):
    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(2))

        def named_parameters(self):
            return [("weight", self.weight)]

        def load_weights(self, *, weights):
            del weights
            raise ValueError("invalid checkpoint")

    monkeypatch.setenv("MX_LOAD_MODE", "direct")
    loader = MdlLoader(Model())

    with pytest.raises(ValueError, match="invalid checkpoint"):
        loader.load_weights([("weight", torch.ones(2))])
    assert loader._loaderless is False
