# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for client-side P2P source selection.

Covers the selector module (policies, registry, config fallback, determinism)
and the RdmaStrategy integration (rank filtering, max-retry slicing,
metadata-miss fallback, and the no-retry-after-transfer-start rule).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modelexpress import p2p_pb2
from modelexpress.adapter import StrategyFailed
from modelexpress.load_strategy.rdma_strategy import MAX_SOURCE_RETRIES, RdmaStrategy
from modelexpress.source_selection import (
    ENV_SELECTOR,
    RandomSelector,
    RendezvousHashSelector,
    configured_policy_label,
    get_configured_selector,
    get_selector,
    register_selector,
)


def _ctx(worker_id="target-0", worker_rank=0, model_name="m"):
    # Selectors read only these fields off the live LoadContext, so a tiny
    # duck-typed stand-in is enough; no need to build a full LoadContext.
    return SimpleNamespace(
        worker_rank=worker_rank,
        global_rank=worker_rank,
        worker_id=worker_id,
        identity=SimpleNamespace(model_name=model_name),
    )


def _ref(mx_source_id, worker_id, worker_rank=0, model_name="m"):
    return p2p_pb2.SourceInstanceRef(
        mx_source_id=mx_source_id,
        worker_id=worker_id,
        model_name=model_name,
        worker_rank=worker_rank,
    )


def _sources(n, worker_rank=0):
    return [_ref(f"src{i:04x}aaaaaaaaaa", f"w{i}", worker_rank) for i in range(n)]


# ---------------------------------------------------------------------------
# Registry / config resolution
# ---------------------------------------------------------------------------


def test_default_is_random(monkeypatch):
    monkeypatch.delenv(ENV_SELECTOR, raising=False)
    assert get_configured_selector().name == "random"


def test_env_selects_rendezvous_hash(monkeypatch):
    monkeypatch.setenv(ENV_SELECTOR, "rendezvous_hash")
    assert get_configured_selector().name == "rendezvous_hash"


def test_unknown_name_falls_back_to_random(caplog):
    with caplog.at_level(logging.WARNING, logger="modelexpress.source_selection"):
        sel = get_selector("does-not-exist")
    assert sel.name == "random"
    assert any("Unknown P2P source selector" in r.message for r in caplog.records)


def test_invalid_env_falls_back_to_random(monkeypatch):
    monkeypatch.setenv(ENV_SELECTOR, "garbage")
    assert get_configured_selector().name == "random"


def test_factory_failure_falls_back_to_random(caplog):
    def _boom():
        raise RuntimeError("broken factory")

    register_selector("broken", _boom)
    try:
        with caplog.at_level(logging.WARNING, logger="modelexpress.source_selection"):
            sel = get_selector("broken")
        assert sel.name == "random"
        assert any("Failed to construct" in r.message for r in caplog.records)
    finally:
        from modelexpress.source_selection import SELECTORS

        SELECTORS.pop("broken", None)


def test_register_custom_selector():
    sentinel = RandomSelector()
    register_selector("custom-x", lambda: sentinel)
    try:
        assert get_selector("custom-x") is sentinel
    finally:
        from modelexpress.source_selection import SELECTORS

        SELECTORS.pop("custom-x", None)


# ---------------------------------------------------------------------------
# Random policy
# ---------------------------------------------------------------------------


def test_random_preserves_candidate_set():
    cands = _sources(6)
    out = RandomSelector().order(cands, _ctx())
    assert len(out) == len(cands)
    assert {c.worker_id for c in out} == {c.worker_id for c in cands}


def test_random_uses_local_rng_not_global(monkeypatch):
    import random as _random

    calls = []
    orig_shuffle = _random.shuffle
    monkeypatch.setattr(_random, "shuffle", lambda *a, **k: calls.append(a))
    try:
        RandomSelector().order(_sources(4), _ctx())
    finally:
        monkeypatch.setattr(_random, "shuffle", orig_shuffle)
    # The policy must not touch process-global random.shuffle.
    assert calls == []


def test_empty_candidates():
    assert RandomSelector().order([], _ctx()) == []
    assert RendezvousHashSelector().order([], _ctx()) == []


# ---------------------------------------------------------------------------
# Rendezvous hash policy
# ---------------------------------------------------------------------------


def test_rendezvous_hash_deterministic():
    cands = _sources(8)
    sel = RendezvousHashSelector()
    a = [c.worker_id for c in sel.order(cands, _ctx())]
    b = [c.worker_id for c in sel.order(cands, _ctx())]
    assert a == b


def test_rendezvous_hash_order_independent_of_input_order():
    cands = _sources(8)
    sel = RendezvousHashSelector()
    forward = [c.worker_id for c in sel.order(cands, _ctx())]
    reverse = [c.worker_id for c in sel.order(list(reversed(cands)), _ctx())]
    assert forward == reverse


def test_rendezvous_hash_stable_across_processes():
    # Pinned blake2b value guards against an accidental switch to Python's
    # process-salted hash(). key = "m|t|0|s|cw|0".
    score = RendezvousHashSelector().score(
        _ref("s", "cw", 0), _ctx(worker_id="t", worker_rank=0, model_name="m")
    )
    assert score == 3844933907942436947


def test_rendezvous_hash_spreads_first_choices():
    sources = _sources(4)
    sel = RendezvousHashSelector()
    first_choices = {
        sel.order(sources, _ctx(worker_id=f"target-{t}"))[0].worker_id
        for t in range(40)
    }
    # Different targets must not all converge on the same source.
    assert len(first_choices) > 1


def test_rendezvous_hash_removing_source_preserves_relative_order():
    # Each candidate's score is independent of the others, so dropping one
    # leaves the relative order of the rest unchanged (only a fraction of
    # rankings is perturbed when the set changes).
    cands = _sources(8)
    sel = RendezvousHashSelector()
    full = [c.worker_id for c in sel.order(cands, _ctx())]
    dropped = full[3]
    remaining = [c for c in cands if c.worker_id != dropped]
    after = [c.worker_id for c in sel.order(remaining, _ctx())]
    assert after == [w for w in full if w != dropped]


# ---------------------------------------------------------------------------
# RdmaStrategy integration
# ---------------------------------------------------------------------------


def _rdma_ctx(instances):
    ctx = MagicMock()
    ctx.global_rank = 0
    ctx.worker_rank = 0
    ctx.worker_id = "target-0"
    ctx.identity = p2p_pb2.SourceIdentity(model_name="m")
    ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
        instances=instances
    )
    return ctx


def test_find_source_instances_filters_by_worker_rank(monkeypatch):
    monkeypatch.setenv(ENV_SELECTOR, "rendezvous_hash")
    instances = [
        _ref("s0aaaaaaaaaaaaaa", "w0", worker_rank=0),
        _ref("s1aaaaaaaaaaaaaa", "w1", worker_rank=1),
        _ref("s2aaaaaaaaaaaaaa", "w2", worker_rank=0),
    ]
    ctx = _rdma_ctx(instances)
    out = RdmaStrategy()._find_source_instances(ctx)
    assert {c.worker_id for c in out} == {"w0", "w2"}


def test_find_source_instances_empty_on_list_error():
    ctx = _rdma_ctx([])
    ctx.mx_client.list_sources.side_effect = RuntimeError("grpc down")
    assert RdmaStrategy()._find_source_instances(ctx) == []


def test_load_slices_to_max_source_retries():
    strat = RdmaStrategy()
    strat._find_source_instances = MagicMock(return_value=_sources(5))
    strat._fetch_worker_metadata = MagicMock(return_value=None)
    strat._load_as_target = MagicMock()
    ctx = MagicMock(global_rank=0)

    with pytest.raises(StrategyFailed) as ei:
        strat.load(MagicMock(), ctx)

    assert ei.value.mutated is False
    assert strat._fetch_worker_metadata.call_count == MAX_SOURCE_RETRIES
    strat._load_as_target.assert_not_called()


def test_load_metadata_miss_tries_next_candidate():
    strat = RdmaStrategy()
    cands = _sources(3)
    strat._find_source_instances = MagicMock(return_value=cands)
    worker = MagicMock()
    # First candidate misses metadata (None); second returns a worker.
    strat._fetch_worker_metadata = MagicMock(side_effect=[None, worker])
    strat._load_as_target = MagicMock(return_value="loaded")
    ctx = MagicMock(global_rank=0)
    ctx.accelerator_backend.name = ""  # unknown target -> accelerator gate accepts

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "modelexpress.load_strategy.rdma_strategy.worker_tensor_count",
            lambda w: 1,
        )
        result = strat.load(MagicMock(), ctx)

    assert result == "loaded"
    assert strat._fetch_worker_metadata.call_count == 2
    # _load_as_target invoked with the second candidate's identifiers.
    args = strat._load_as_target.call_args.args
    assert cands[1].mx_source_id in args
    assert cands[1].worker_id in args


def test_load_transfer_failure_does_not_try_next_source():
    strat = RdmaStrategy()
    cands = _sources(3)
    strat._find_source_instances = MagicMock(return_value=cands)
    strat._fetch_worker_metadata = MagicMock(return_value=MagicMock())
    strat._load_as_target = MagicMock(
        side_effect=StrategyFailed("receive failed", mutated=True)
    )
    ctx = MagicMock(global_rank=0)
    ctx.accelerator_backend.name = ""  # unknown target -> accelerator gate accepts

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "modelexpress.load_strategy.rdma_strategy.worker_tensor_count",
            lambda w: 1,
        )
        with pytest.raises(StrategyFailed) as ei:
            strat.load(MagicMock(), ctx)

    assert ei.value.mutated is True
    # Exactly one source was attempted; no retry after transfer start.
    assert strat._fetch_worker_metadata.call_count == 1
    assert strat._load_as_target.call_count == 1


# ---------------------------------------------------------------------------
# configured_policy_label (metric/label resolution)
# ---------------------------------------------------------------------------


def test_configured_policy_label_default(monkeypatch):
    monkeypatch.delenv(ENV_SELECTOR, raising=False)
    assert configured_policy_label() == "random"


def test_configured_policy_label_valid(monkeypatch):
    monkeypatch.setenv(ENV_SELECTOR, "rendezvous_hash")
    assert configured_policy_label() == "rendezvous_hash"


def test_configured_policy_label_unknown_falls_back(monkeypatch):
    # Must match get_selector's fallback so labels never claim a policy that
    # did not actually run.
    monkeypatch.setenv(ENV_SELECTOR, "garbage")
    assert configured_policy_label() == "random"


def test_configured_policy_label_failing_factory_falls_back(monkeypatch):
    # A registered-but-raising factory resolves to "random" at runtime, so the
    # label must too (else load() metrics would be split from selection metrics).
    def _boom():
        raise RuntimeError("broken factory")

    register_selector("broken-label", _boom)
    monkeypatch.setenv(ENV_SELECTOR, "broken-label")
    try:
        assert configured_policy_label() == "random"
    finally:
        from modelexpress.source_selection import SELECTORS

        SELECTORS.pop("broken-label", None)


def test_rendezvous_hash_stable_on_score_tie():
    # Identical hash-key fields (mx_source_id/worker_id/worker_rank) tie on
    # score; model_name is not hashed, so it distinguishes order. sorted() is
    # stable, so a tie preserves input order.
    sel = RendezvousHashSelector()
    a = _ref("samesrc0000aaaa", "samew", 0, model_name="A")
    b = _ref("samesrc0000aaaa", "samew", 0, model_name="B")
    assert sel.score(a, _ctx()) == sel.score(b, _ctx())
    assert [c.model_name for c in sel.order([a, b], _ctx())] == ["A", "B"]
    assert [c.model_name for c in sel.order([b, a], _ctx())] == ["B", "A"]


# ---------------------------------------------------------------------------
# RdmaStrategy.load() -> selection metrics
# ---------------------------------------------------------------------------


def _patched_metrics(monkeypatch):
    m = MagicMock()
    monkeypatch.setattr("modelexpress.load_strategy.rdma_strategy.selection_metrics", m)
    monkeypatch.setattr(
        "modelexpress.load_strategy.rdma_strategy.worker_tensor_count", lambda w: 1
    )
    monkeypatch.delenv(ENV_SELECTOR, raising=False)  # policy label -> "random"
    return m


def test_load_records_success_metrics(monkeypatch):
    m = _patched_metrics(monkeypatch)
    strat = RdmaStrategy()
    cands = _sources(1)
    strat._find_source_instances = MagicMock(return_value=cands)
    strat._fetch_worker_metadata = MagicMock(return_value=MagicMock())
    strat._load_as_target = MagicMock(return_value="loaded")
    ctx = MagicMock(global_rank=0)
    ctx.accelerator_backend.name = ""  # unknown target -> accelerator gate accepts

    assert strat.load(MagicMock(), ctx) == "loaded"
    m.record_selection.assert_called_once_with("random", cands[0].worker_id)
    m.record_attempt.assert_any_call("random", "success")
    assert m.observe_transfer_seconds.call_args.args[:2] == ("random", "success")


def test_load_records_transfer_fallback_metrics(monkeypatch):
    m = _patched_metrics(monkeypatch)
    strat = RdmaStrategy()
    strat._find_source_instances = MagicMock(return_value=_sources(2))
    strat._fetch_worker_metadata = MagicMock(return_value=MagicMock())
    strat._load_as_target = MagicMock(
        side_effect=StrategyFailed("receive failed", mutated=True)
    )
    ctx = MagicMock(global_rank=0)
    ctx.accelerator_backend.name = ""  # unknown target -> accelerator gate accepts

    with pytest.raises(StrategyFailed):
        strat.load(MagicMock(), ctx)
    m.record_attempt.assert_any_call("random", "transfer_fallback")
    assert m.observe_transfer_seconds.call_args.args[:2] == ("random", "fallback")


def test_load_records_metadata_miss_metric(monkeypatch):
    m = _patched_metrics(monkeypatch)
    strat = RdmaStrategy()
    strat._find_source_instances = MagicMock(return_value=_sources(3))
    strat._fetch_worker_metadata = MagicMock(return_value=None)
    strat._load_as_target = MagicMock()

    with pytest.raises(StrategyFailed):
        strat.load(MagicMock(), MagicMock(global_rank=0))
    assert m.record_attempt.call_count == MAX_SOURCE_RETRIES
    m.record_attempt.assert_called_with("random", "metadata_miss")
    m.record_selection.assert_not_called()
