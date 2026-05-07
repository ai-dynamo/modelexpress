# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the vLLM engine adapter."""

import sys
from types import SimpleNamespace

import torch

from modelexpress.engines.vllm.adapter import VllmAdapter, _get_vllm_worker_rank


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
    vllm_config = SimpleNamespace(
        device_config=SimpleNamespace(device="cuda"),
        load_config=SimpleNamespace(device=None),
    )
    adapter = VllmAdapter(vllm_config, SimpleNamespace())

    assert adapter.get_device_id() == 2
