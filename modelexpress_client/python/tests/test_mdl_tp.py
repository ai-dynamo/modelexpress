# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from modelexpress.engines.vllm.mdl import MdlLoader
from modelexpress.refit_timing import RefitTimingRecorder, use_refit_timing


class _Model:
    def named_parameters(self):
        return []


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

        def named_parameters(self):
            return [("weight", self.weight)]

        def load_weights(self, *, weights):
            for _name, value in weights:
                self.weight.data.copy_(value)

    monkeypatch.setenv("MX_LOAD_MODE", "direct")
    loader = MdlLoader(Model())

    cold = RefitTimingRecorder(backend="test", version=1)
    with use_refit_timing(cold):
        loader.load_weights([("weight", torch.ones(2))])
    assert cold.as_dict()["cold_warm"] == "cold"

    warm = RefitTimingRecorder(backend="test", version=2)
    with use_refit_timing(warm):
        loader.load_weights([("weight", torch.full((2,), 2.0))])
    assert warm.as_dict()["cold_warm"] == "warm"
