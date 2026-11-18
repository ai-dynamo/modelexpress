# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine ranks leave every strategy the same way.

Each rank runs the strategy chain on its own, but later strategies
(InstantTensor, distributed ModelStreamer) and the engine's post-load barrier
are collective. After each strategy the chain gathers every rank's outcome:

    all loaded           -> every rank publishes and returns
    any rank failed      -> every rank falls back to the next strategy together
    any rank fatal       -> every rank raises StrategyRecoveryError

Run: pytest tests/test_rank_agreement.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modelexpress.adapter import StrategyFailed, StrategyRecoveryError
from modelexpress.load_strategy import execute_load_strategies
from modelexpress.load_strategy.base import LoadStrategy


class _PeerRanks:
    """Adapter stand-in: this rank plus peers whose outcome is fixed per strategy."""

    def __init__(self, peer_outcomes, *, peer_strategy=None):
        self.peer_outcomes = peer_outcomes
        self.peer_strategy = peer_strategy
        self.gathered = []
        self.reinit_for_retry = MagicMock(side_effect=lambda result: result)

    def all_gather_state(self, state):
        strategy, _outcome = state
        self.gathered.append(state)
        peer = (self.peer_strategy or strategy, self.peer_outcomes[strategy])
        return (state, peer)


class _Strategy(LoadStrategy):
    def __init__(self, name, *, error=None):
        self.name = name
        self.error = error
        self.rolled_back = 0

    def is_available(self, ctx):
        return True

    def prepare(self, result, ctx):
        pass

    def load(self, result, ctx):
        if self.error is not None:
            raise self.error
        return result

    def finalize(self, result, ctx):
        pass

    def abort(self, result, ctx):
        pass

    def rollback(self, ctx):
        self.rolled_back += 1


def _ctx(adapter, *, is_reload=False):
    return SimpleNamespace(
        global_rank=0,
        is_reload=is_reload,
        p2p_enabled=False,
        adapter=adapter,
        identity=SimpleNamespace(model_name="m"),
    )


@pytest.fixture
def published(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "modelexpress.load_strategy.publish_source_if_supported",
        lambda result, ctx: calls.append(result.value),
    )
    return calls


def test_all_ranks_loaded_publishes_and_returns(published):
    adapter = _PeerRanks({"rdma": "loaded"})
    model = MagicMock()

    assert execute_load_strategies(model, _ctx(adapter), [_Strategy("rdma")]) is model

    assert published == [model]
    adapter.reinit_for_retry.assert_not_called()


def test_a_rank_that_loaded_falls_back_when_another_rank_failed(published):
    """The #901/#966 split: this rank's RDMA read finished, a peer's did not."""
    adapter = _PeerRanks({"rdma": "failed", "disk": "loaded"})
    rdma, disk = _Strategy("rdma"), _Strategy("disk")
    model = MagicMock()

    assert execute_load_strategies(model, _ctx(adapter), [rdma, disk]) is model

    assert [state for state in adapter.gathered] == [("rdma", "loaded"), ("disk", "loaded")]
    # The finished rank undoes its RDMA state and rebuilds the model it wrote,
    # and publishes only once every rank has loaded.
    assert rdma.rolled_back == 1
    adapter.reinit_for_retry.assert_called_once()
    assert published == [model]


def test_a_failed_rank_falls_back_with_a_rank_that_loaded(published):
    adapter = _PeerRanks({"rdma": "loaded", "disk": "loaded"})
    rdma = _Strategy("rdma", error=StrategyFailed("source died", mutated=True))
    disk = _Strategy("disk")

    execute_load_strategies(MagicMock(), _ctx(adapter), [rdma, disk])

    assert adapter.gathered == [("rdma", "failed"), ("disk", "loaded")]
    adapter.reinit_for_retry.assert_called_once()
    assert len(published) == 1


def test_an_unmutated_miss_is_not_reinitialized(published):
    adapter = _PeerRanks({"rdma": "loaded", "disk": "loaded"})
    rdma = _Strategy("rdma", error=StrategyFailed("no source", mutated=False))

    execute_load_strategies(MagicMock(), _ctx(adapter), [rdma, _Strategy("disk")])

    adapter.reinit_for_retry.assert_not_called()


def test_a_reload_falls_back_together_without_reinitializing(published):
    adapter = _PeerRanks({"rdma": "failed", "disk": "loaded"})
    rdma = _Strategy("rdma")

    execute_load_strategies(
        MagicMock(), _ctx(adapter, is_reload=True), [rdma, _Strategy("disk")]
    )

    assert rdma.rolled_back == 1
    adapter.reinit_for_retry.assert_not_called()


def test_every_rank_fails_closed_when_another_rank_is_fatal(published):
    adapter = _PeerRanks({"rdma": "fatal"})
    disk = _Strategy("disk")

    with pytest.raises(StrategyRecoveryError, match="Another rank could not recover"):
        execute_load_strategies(MagicMock(), _ctx(adapter), [_Strategy("rdma"), disk])

    assert adapter.gathered == [("rdma", "loaded")]
    assert published == []


def test_a_fatal_rank_still_joins_the_agreement_before_raising(published):
    """Otherwise the other ranks would wait in the gather forever."""
    adapter = _PeerRanks({"rdma": "loaded"})
    error = StrategyRecoveryError("model cannot be restored")
    rdma = _Strategy("rdma", error=error)

    with pytest.raises(StrategyRecoveryError) as raised:
        execute_load_strategies(MagicMock(), _ctx(adapter), [rdma, _Strategy("disk")])

    assert raised.value is error
    assert adapter.gathered == [("rdma", "fatal")]
    assert rdma.rolled_back == 1


def test_ranks_on_different_strategies_fail_closed(published):
    adapter = _PeerRanks({"rdma": "loaded"}, peer_strategy="disk")

    with pytest.raises(StrategyRecoveryError, match="Ranks disagree on the loading strategy"):
        execute_load_strategies(MagicMock(), _ctx(adapter), [_Strategy("rdma")])

    assert published == []


def test_no_strategy_succeeds_when_every_rank_fails_everywhere(published):
    adapter = _PeerRanks({"rdma": "failed", "disk": "failed"})
    disk = _Strategy("disk", error=StrategyFailed("disk read failed"))

    with pytest.raises(RuntimeError, match="No loading strategy succeeded"):
        execute_load_strategies(MagicMock(), _ctx(adapter), [_Strategy("rdma"), disk])

    assert published == []
