# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest.mock as mock

import pytest
import torch.nn as nn

import modelexpress.load_strategy as chain
import modelexpress.load_strategy.default_strategy as default_strategy
import modelexpress.load_strategy.rdma_strategy as rdma_strategy
import modelexpress.p2p_pb2 as p2p_pb2


@pytest.mark.parametrize("outcome", ["success", "unavailable", "miss", "partial", "unexpected"])
def test_rdma_only_never_attempts_storage(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """Exercise the real executor without a storage fallback.

    Args:
        monkeypatch: Fixture for the policy and strategy constructors.
        outcome: Successful transfer, unavailability, or transfer failure.
    """
    monkeypatch.setenv("MX_LOAD_STRATEGY_CHAIN", "RDMA_ONLY")
    monkeypatch.delenv("MX_LOAD_PLUGIN", raising=False)
    model = nn.Linear(1, 1)
    ctx = mock.Mock(
        spec=chain.LoadContext, global_rank=0, is_reload=True,
        p2p_enabled=False, adapter=None,
        identity=p2p_pb2.SourceIdentity(model_name="test"),
    )
    rdma = mock.Mock(spec=chain.LoadStrategy)
    rdma.name = "rdma"
    rdma.is_available.return_value = outcome != "unavailable"
    rdma.load.return_value = chain.LoadResult(value=model, model=model)
    if outcome in ("miss", "partial"):
        rdma.load.side_effect = chain.StrategyFailed(
            "peer unavailable", mutated=outcome == "partial"
        )
    elif outcome == "unexpected":
        rdma.load.side_effect = RuntimeError("transport failure")
    monkeypatch.setattr(rdma_strategy, "RdmaStrategy", lambda: rdma)
    storage = mock.Mock(side_effect=AssertionError("storage was constructed"))
    monkeypatch.setattr(default_strategy, "DefaultStrategy", storage)
    publish = mock.Mock()
    monkeypatch.setattr(chain, "publish_source_if_supported", publish)
    if outcome == "success":
        assert chain.run_load_strategy_chain(model, ctx) is model
        publish.assert_called_once()
    else:
        with pytest.raises(RuntimeError, match="No loading strategy succeeded"):
            chain.run_load_strategy_chain(model, ctx)
        publish.assert_not_called()
        if outcome != "unavailable":
            rdma.rollback.assert_called_once_with(ctx)
    storage.assert_not_called()


def test_rdma_only_rejects_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a conflicting plugin before loading.

    Args:
        monkeypatch: Fixture for the conflicting policy settings.
    """
    monkeypatch.setenv("MX_LOAD_STRATEGY_CHAIN", "RDMA_ONLY")
    monkeypatch.setenv("MX_LOAD_PLUGIN", "custom")
    with pytest.raises(ValueError, match="RDMA_ONLY.*MX_LOAD_PLUGIN"):
        chain.run_load_strategy_chain(nn.Linear(1, 1), mock.Mock(spec=chain.LoadContext))
