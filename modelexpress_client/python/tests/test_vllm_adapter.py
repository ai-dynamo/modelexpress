# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the vLLM engine adapter."""

import sys
from types import SimpleNamespace

import torch

from modelexpress.engines.vllm.adapter import (
    _get_vllm_device_id,
    _get_vllm_worker_rank,
    build_vllm_load_context,
)


def _vllm_config(*, rank: int, tp_size: int, pp_size: int):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            rank=rank,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
        )
    )


def test_worker_rank_uses_vllm_model_parallel_rank():
    # vLLM rank layout is DP x PP x TP. Modulo TP*PP keeps PP/TP shard
    # identity and drops the DP component.
    config = _vllm_config(rank=6, tp_size=4, pp_size=2)

    assert _get_vllm_worker_rank(config) == 6


def test_worker_rank_ignores_dp_component():
    dp0 = _vllm_config(rank=5, tp_size=4, pp_size=2)
    dp1 = _vllm_config(rank=13, tp_size=4, pp_size=2)

    assert _get_vllm_worker_rank(dp0) == 5
    assert _get_vllm_worker_rank(dp1) == 5


def test_vllm_device_id_uses_current_platform_device(monkeypatch):
    fake_platforms = SimpleNamespace(
        current_platform=SimpleNamespace(
            current_device=lambda: 2,
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)

    assert _get_vllm_device_id(torch.device("cuda")) == 2


def test_build_vllm_load_context_uses_current_platform_for_bare_cuda(monkeypatch):
    _stub_vllm_current_device(monkeypatch, current_device=2)
    _stub_metadata_client(monkeypatch)
    vllm_config = _context_config(load_device=None)

    ctx = build_vllm_load_context(vllm_config, _model_config())

    assert ctx.target_device == torch.device("cuda")
    assert ctx.target_device.index is None
    assert ctx.device_id == 2


def test_build_vllm_load_context_keeps_explicit_cuda_index(monkeypatch):
    _stub_vllm_current_device(monkeypatch, current_device=2)
    _stub_metadata_client(monkeypatch)
    vllm_config = _context_config(load_device="cuda:3")

    ctx = build_vllm_load_context(vllm_config, _model_config())

    assert ctx.target_device == torch.device("cuda:3")
    assert ctx.target_device.index == 3
    assert ctx.device_id == ctx.target_device.index


def _stub_vllm_current_device(monkeypatch, *, current_device: int) -> None:
    fake_platforms = SimpleNamespace(
        current_platform=SimpleNamespace(
            current_device=lambda: current_device,
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)


def _stub_metadata_client(monkeypatch) -> None:
    monkeypatch.setattr(
        "modelexpress.engines.vllm.adapter.create_metadata_client",
        lambda worker_rank: object(),
    )


def _context_config(*, load_device):
    return SimpleNamespace(
        device_config=SimpleNamespace(device="cuda"),
        load_config=SimpleNamespace(device=load_device),
        parallel_config=SimpleNamespace(
            rank=0,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        ),
    )


def _model_config():
    return SimpleNamespace(
        dtype=torch.bfloat16,
        model="test-model",
        quantization=None,
        revision=None,
    )
