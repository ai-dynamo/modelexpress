# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the vLLM engine adapter."""

from types import SimpleNamespace

import torch

from modelexpress.engines.vllm.adapter import _get_vllm_worker_rank
from modelexpress.rank_utils import get_device_id


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


def test_device_id_uses_local_cuda_ordinal():
    assert get_device_id(torch.device("cuda:3")) == 3
