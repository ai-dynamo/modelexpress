# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from modelexpress.engines.vllm.adapter import VllmAdapter


def _make_adapter(extra_config):
    load_config = SimpleNamespace(
        device=None,
        load_format="modelexpress",
        model_loader_extra_config=extra_config,
    )
    vllm_config = MagicMock()
    vllm_config.load_config = load_config
    vllm_config.device_config.device = "cuda"
    vllm_config.parallel_config.tensor_parallel_size = 8
    model_config = SimpleNamespace(revision="main")
    return VllmAdapter(vllm_config, model_config), load_config


def _patch_dummy_loader():
    loader_instance = MagicMock()
    loader_cls = MagicMock(return_value=loader_instance)
    module = SimpleNamespace(DummyModelLoader=loader_cls)
    return (
        patch.dict(
            sys.modules,
            {"vllm.model_executor.model_loader.dummy_loader": module},
        ),
        loader_cls,
    )


def test_prepare_rdma_target_strips_extra_config_for_dummy_loader():
    # The RDMA target allocates empty tensors via the dummy loader, which
    # rejects any model_loader_extra_config. Streamer-oriented extra config
    # (distributed / memory_limit) on the source config must not leak into it,
    # otherwise the dummy load raises and the whole RDMA strategy is skipped.
    adapter, source_config = _make_adapter(
        {"distributed": True, "memory_limit": 10_000_000_000}
    )
    patcher, loader_cls = _patch_dummy_loader()
    result = SimpleNamespace(model=torch.nn.Module())

    with patcher:
        adapter.prepare_rdma_target(result)

    dummy_config = loader_cls.call_args.args[0]
    assert dummy_config.load_format == "dummy"
    assert dummy_config.model_loader_extra_config == {}
    # The source config is untouched, so the streamer fallback still gets it.
    assert source_config.model_loader_extra_config == {
        "distributed": True,
        "memory_limit": 10_000_000_000,
    }


def test_prepare_rdma_target_handles_empty_extra_config():
    # Default MX usage has no extra config; behaviour must be unchanged.
    adapter, _source_config = _make_adapter({})
    patcher, loader_cls = _patch_dummy_loader()
    result = SimpleNamespace(model=torch.nn.Module())

    with patcher:
        adapter.prepare_rdma_target(result)

    dummy_config = loader_cls.call_args.args[0]
    assert dummy_config.load_format == "dummy"
    assert dummy_config.model_loader_extra_config == {}
