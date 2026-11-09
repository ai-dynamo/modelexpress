# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A model with parameters outside accelerator memory must not serve P2P.

Tensor discovery only collects accelerator tensors, so host-resident
parameters (DeepSeek V4.1 Engram tables in pinned host memory) were never
transferred. The target reported a complete transfer and then served its
dummy values.

Run: pytest tests/test_host_resident_parameters_guard.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from modelexpress.adapter import StrategyFailed
from modelexpress.load_strategy import base, rdma_strategy
from modelexpress.load_strategy.context import LoadResult
from modelexpress.load_strategy.rdma_strategy import RdmaStrategy


def _ctx(backend):
    return SimpleNamespace(
        p2p_enabled=True,
        p2p_role="main",
        shared_nixl_manager=None,
        global_rank=0,
        worker_rank=0,
        device_id=0,
        tensors={},
        nixl_manager=None,
        vmm_arena=None,
        adapter=MagicMock(),
        accelerator_backend=backend,
        mx_client=MagicMock(),
    )


def _model(*, host_table: bool):
    model = nn.Module()
    model.attn = nn.Parameter(torch.empty(4, device="meta"))
    if host_table:
        model.engram = nn.Parameter(torch.empty(4))
    return model


@pytest.fixture
def backend(mock_accelerator_backend_cls):
    return mock_accelerator_backend_cls(torch_device_type="meta")


@pytest.fixture
def patched(monkeypatch):
    """Take the environment checks out of the picture; they are not under test."""
    monkeypatch.setattr(base, "is_nixl_available", lambda: True)
    monkeypatch.setattr(base, "_metadata_publication_configured", lambda ctx: True)


def test_source_with_host_parameters_does_not_register_or_publish(patched, backend):
    ctx = _ctx(backend)
    model = _model(host_table=True)

    with patch.object(base, "_init_nixl_manager") as init:
        base.register_tensors(LoadResult(value=model, model=model), ctx)
    with patch.object(base, "publish_metadata_and_ready") as publish:
        base.publish_metadata(ctx)

    ctx.adapter.discover_tensors.assert_not_called()
    init.assert_not_called()
    assert ctx.nixl_manager is None
    publish.assert_not_called()


def test_source_with_accelerator_parameters_registers(patched, backend):
    ctx = _ctx(backend)
    model = _model(host_table=False)
    ctx.adapter.discover_tensors.return_value = {"attn": model.attn}
    manager = MagicMock()
    manager.tensor_descriptors = []

    with patch.object(base, "_init_nixl_manager", return_value=manager):
        base.register_tensors(LoadResult(value=model, model=model), ctx)

    assert ctx.nixl_manager is manager
    manager.register_tensors.assert_called_once_with({"attn": model.attn})


def test_target_with_host_parameters_falls_back_before_receiving(
    patched, backend, monkeypatch
):
    """A source that still advertises such a model must not be received from."""
    ctx = _ctx(backend)
    ctx.is_reload = False
    model = _model(host_table=True)
    monkeypatch.setattr(rdma_strategy, "worker_tensor_descriptors", lambda _w: [])

    with patch.object(base, "_init_nixl_manager") as init:
        with pytest.raises(StrategyFailed, match="NIXL registration failed") as excinfo:
            RdmaStrategy()._receive_from_peer(
                LoadResult(value=model, model=model),
                ctx,
                SimpleNamespace(worker_grpc_endpoint="", nixl_metadata=b""),
                "mx-source",
                "worker",
            )

    assert excinfo.value.mutated is True
    ctx.adapter.discover_tensors.assert_not_called()
    init.assert_not_called()
