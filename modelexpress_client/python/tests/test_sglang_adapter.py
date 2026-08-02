# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SGLang adapter's RDMA post-receive derivation.

Covers ``SglangAdapter.after_rdma_receive``: the receive path must reset
value-derived caches (computed pre-transfer from random-init weights) and
re-run the engine's model-level post-load derivation from the received
weights, without touching quantized packing (``_process_weights_after_loading``
must not re-run on the already post-processed tensors).
"""

from types import SimpleNamespace

import pytest
import torch

from modelexpress.engines.sglang.adapter import SglangAdapter
from modelexpress.load_strategy.context import LoadResult


def _adapter() -> SglangAdapter:
    return SglangAdapter(
        load_config=SimpleNamespace(),
        model_config=SimpleNamespace(),
        device_config=SimpleNamespace(device="cuda"),
    )


class _ResidualModel(torch.nn.Module):
    """Stands in for an engine that derives value caches from weights."""

    def __init__(self):
        super().__init__()
        self.lm = torch.nn.Linear(4, 4, bias=False)
        self.res_proj = torch.nn.Linear(4, 4, bias=False)
        # Cache keyed by weight content, computed pre-transfer from
        # random-init weights (cf. SGLang get_cw/_attn_res_cw_cache).
        self.res_proj._attn_res_cw_cache = {"fp32": self.lm.weight.detach().clone()}
        self.post_load_calls = 0
        self.quant_process_calls = 0

    def post_load_weights(self):
        self.post_load_calls += 1


def test_after_rdma_receive_resets_derived_caches_and_rederives():
    model = _ResidualModel()
    fresh = torch.zeros_like(model.lm.weight.detach())
    with torch.no_grad():  # simulate the manifest write landing
        model.lm.weight.copy_(fresh)

    result = _adapter().after_rdma_receive(LoadResult(value=model, model=model))

    assert result.model is model
    assert model.res_proj._attn_res_cw_cache == {}
    assert model.post_load_calls == 1


def test_after_rdma_receive_requires_a_model():
    with pytest.raises(RuntimeError, match="requires result.model"):
        _adapter().after_rdma_receive(LoadResult(value=None, model=None))


def test_after_rdma_receive_tolerates_models_without_caches():
    model = torch.nn.Linear(4, 4, bias=False)  # no caches, no post_load_weights
    result = _adapter().after_rdma_receive(LoadResult(value=model, model=model))
    assert result.model is model
