# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SGLang's RDMA post-receive processing."""

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


class _DeepseekV4Model(torch.nn.Module):
    """Stands in for SGLang's DeepSeek-V4 post-load behavior."""

    def __init__(self):
        super().__init__()
        self.post_load_calls = 0
        self.weight_scale_inv = torch.nn.Parameter(
            torch.ones(1),
            requires_grad=False,
        )

    def post_load_weights(self):
        self.post_load_calls += 1
        self.weight_scale_inv.data = self.weight_scale_inv.data.contiguous() * 2


def test_after_rdma_receive_preserves_deepseek_v4_scale_storage(
    monkeypatch,
):
    model = _DeepseekV4Model()
    adapter = _adapter()
    monkeypatch.setattr(
        SglangAdapter,
        "_process_weights_after_loading",
        lambda self, result: result,
    )

    adapter.before_rdma_receive(LoadResult(value=model, model=model))
    assert model.post_load_calls == 1
    scale_ptr = model.weight_scale_inv.data_ptr()
    assert torch.equal(model.weight_scale_inv, torch.full((1,), 2))

    with torch.no_grad():
        model.weight_scale_inv.fill_(3)

    result = adapter.after_rdma_receive(LoadResult(value=model, model=model))

    assert result.model is model
    assert model.post_load_calls == 1
    assert model.weight_scale_inv.data_ptr() == scale_ptr
    assert torch.equal(model.weight_scale_inv, torch.full((1,), 3))


class _Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))


class _KimiK3Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_attn_residuals = True
        self.self_attention_res_proj = torch.nn.Linear(4, 1, bias=False)
        self.self_attention_res_norm = _Norm()
        self.mlp_res_proj = torch.nn.Linear(4, 1, bias=False)
        self.mlp_res_norm = _Norm()


class _KimiK3Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_KimiK3Layer()])
        self.post_load_calls = 0

    def post_load_weights(self):
        self.post_load_calls += 1
        layer = self.model.layers[0]
        for proj, norm in (
            (layer.self_attention_res_proj, layer.self_attention_res_norm),
            (layer.mlp_res_proj, layer.mlp_res_norm),
        ):
            combined = (
                norm.weight.float() * proj.weight.squeeze().float()
            ).contiguous()
            proj._attn_res_cw_cache = {
                torch.float32: combined.to(torch.float32),
                torch.bfloat16: combined.to(torch.bfloat16),
            }


_KimiK3Model.__module__ = "sglang.srt.models.kimi_k3"


def test_after_rdma_receive_eagerly_refreshes_kimi_k3_caches(monkeypatch):
    model = _KimiK3Model()
    adapter = _adapter()
    monkeypatch.setattr(
        SglangAdapter,
        "_process_weights_after_loading",
        lambda self, result: result,
    )

    adapter.before_rdma_receive(LoadResult(value=model, model=model))
    layer = model.model.layers[0]
    cache = layer.self_attention_res_proj._attn_res_cw_cache
    cache_ptrs = {dtype: tensor.data_ptr() for dtype, tensor in cache.items()}

    received = torch.arange(1, 5, dtype=torch.float32).reshape(1, 4)
    with torch.no_grad():
        layer.self_attention_res_proj.weight.copy_(received)

    result = adapter.after_rdma_receive(LoadResult(value=model, model=model))

    assert result.model is model
    assert model.post_load_calls == 1
    assert layer.self_attention_res_proj._attn_res_cw_cache is cache
    expected = received.squeeze()
    for dtype, cached in cache.items():
        assert cached.data_ptr() == cache_ptrs[dtype]
        assert torch.equal(cached, expected.to(dtype))


def test_after_rdma_receive_requires_a_model():
    with pytest.raises(RuntimeError, match=r"requires result\.model"):
        _adapter().after_rdma_receive(LoadResult(value=None, model=None))


def test_after_rdma_receive_tolerates_models_without_caches():
    model = torch.nn.Linear(4, 4, bias=False)
    result = _adapter().after_rdma_receive(LoadResult(value=model, model=model))
    assert result.model is model


def test_after_rdma_receive_leaves_non_kimi_caches_unchanged():
    model = torch.nn.Linear(4, 4, bias=False)
    cache = {torch.float32: torch.ones(4)}
    model._attn_res_cw_cache = cache

    _adapter().after_rdma_receive(LoadResult(value=model, model=model))

    assert model._attn_res_cw_cache is cache
    assert torch.equal(cache[torch.float32], torch.ones(4))
