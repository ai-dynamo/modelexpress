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
from modelexpress.load_strategy.base import LoadResult
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


def _ref(mx_source_id, worker_id, worker_rank=0, model_name="m", accelerator=""):
    return p2p_pb2.SourceInstanceRef(
        mx_source_id=mx_source_id,
        worker_id=worker_id,
        model_name=model_name,
        worker_rank=worker_rank,
        accelerator=accelerator,
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


def test_find_source_instances_filters_incompatible_accelerator():
    # A compatible source ranked last must survive incompatible ones ranked
    # first, since filtering happens before the MAX_SOURCE_RETRIES slice.
    # "other" is a placeholder for any non-matching accelerator family; the
    # filter compares the strings and does not enumerate known backends.
    instances = [
        _ref("s0aaaaaaaaaaaaaa", "match-0", accelerator="cuda"),
        _ref("s1aaaaaaaaaaaaaa", "other-0", accelerator="other"),
        _ref("s2aaaaaaaaaaaaaa", "other-1", accelerator="other"),
        _ref("s3aaaaaaaaaaaaaa", "other-2", accelerator="other"),
    ]
    ctx = _rdma_ctx(instances)
    ctx.accelerator_backend.name = "cuda"
    out = RdmaStrategy()._find_source_instances(ctx)
    assert {c.worker_id for c in out} == {"match-0"}


def test_find_source_instances_empty_accelerator_is_compatible():
    # Empty source accelerator (records that predate the field) is treated as
    # unknown and accepted; a populated mismatch is still filtered.
    instances = [
        _ref("s0aaaaaaaaaaaaaa", "legacy", accelerator=""),
        _ref("s1aaaaaaaaaaaaaa", "other-0", accelerator="other"),
    ]
    ctx = _rdma_ctx(instances)
    ctx.accelerator_backend.name = "cuda"
    out = RdmaStrategy()._find_source_instances(ctx)
    assert {c.worker_id for c in out} == {"legacy"}


def test_find_source_instances_unknown_target_accepts_all():
    # Empty target accelerator (unknown) accepts every source regardless of
    # the source's published accelerator.
    instances = [
        _ref("s0aaaaaaaaaaaaaa", "match-0", accelerator="cuda"),
        _ref("s1aaaaaaaaaaaaaa", "other-0", accelerator="other"),
    ]
    ctx = _rdma_ctx(instances)
    ctx.accelerator_backend.name = ""
    out = RdmaStrategy()._find_source_instances(ctx)
    assert {c.worker_id for c in out} == {"match-0", "other-0"}


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


def test_load_generation_mismatch_tries_next_candidate(monkeypatch):
    strat = RdmaStrategy()
    cands = _sources(2)
    strat._find_source_instances = MagicMock(return_value=cands)
    workers = [
        p2p_pb2.WorkerMetadata(worker_grpc_endpoint="source:6555"),
        p2p_pb2.WorkerMetadata(
            worker_grpc_endpoint="source:6556",
            tensor_source=p2p_pb2.TensorSourceMetadata(
                tensors=[p2p_pb2.TensorDescriptor(name="stale")],
            ),
        ),
    ]
    strat._load_as_target = MagicMock(return_value="loaded")
    manifest = [p2p_pb2.TensorDescriptor(name="weight")]
    fetch_manifest = MagicMock(
        side_effect=[RuntimeError("worker_id mismatch"), (manifest, 10)],
    )
    monkeypatch.setattr(
        "modelexpress.metadata.worker_server.fetch_tensor_manifest",
        fetch_manifest,
    )
    monkeypatch.setattr(
        "modelexpress.load_strategy.rdma_strategy.worker_tensor_count",
        lambda worker: 1,
    )

    ctx = MagicMock(global_rank=0)
    ctx.mx_client.get_metadata.side_effect = [
        p2p_pb2.GetMetadataResponse(found=True, worker=worker)
        for worker in workers
    ]

    result = strat.load(MagicMock(), ctx)

    assert result == "loaded"
    assert fetch_manifest.call_args_list[0].kwargs == {
        "endpoint": "source:6555",
        "mx_source_id": cands[0].mx_source_id,
        "worker_id": cands[0].worker_id,
    }
    assert fetch_manifest.call_args_list[1].kwargs == {
        "endpoint": "source:6556",
        "mx_source_id": cands[1].mx_source_id,
        "worker_id": cands[1].worker_id,
    }
    assert strat._load_as_target.call_count == 1
    ctx.adapter.reinit_for_retry.assert_not_called()
    selected_worker = strat._load_as_target.call_args.args[2]
    assert [tensor.name for tensor in selected_worker.tensor_source.tensors] == [
        "weight"
    ]


def test_fetch_worker_metadata_prefetches_legacy_endpoint(monkeypatch):
    strat = RdmaStrategy()
    worker = p2p_pb2.WorkerMetadata(worker_grpc_endpoint="source:6555")
    ctx = MagicMock(global_rank=0)
    ctx.mx_client.get_metadata.return_value = p2p_pb2.GetMetadataResponse(
        found=True,
        worker=worker,
    )
    fetch_manifest = MagicMock(return_value=([], 0))
    monkeypatch.setattr(
        "modelexpress.metadata.worker_server.fetch_tensor_manifest",
        fetch_manifest,
    )

    result = strat._fetch_worker_metadata(ctx, "source-123", "")

    assert result is not None
    fetch_manifest.assert_called_once_with(
        endpoint="source:6555",
        mx_source_id="source-123",
        worker_id="",
    )


def test_fetch_worker_metadata_reuses_empty_worker_id_manifest(monkeypatch):
    strat = RdmaStrategy()
    worker = p2p_pb2.WorkerMetadata(
        worker_grpc_endpoint="service:6555",
        tensor_source=p2p_pb2.TensorSourceMetadata(
            tensors=[p2p_pb2.TensorDescriptor(name="weight")],
        ),
    )
    ctx = MagicMock(global_rank=0)
    ctx.mx_client.get_metadata.return_value = p2p_pb2.GetMetadataResponse(
        found=True,
        worker=worker,
    )
    fetch_manifest = MagicMock()
    monkeypatch.setattr(
        "modelexpress.metadata.worker_server.fetch_tensor_manifest",
        fetch_manifest,
    )

    result = strat._fetch_worker_metadata(ctx, "source-123", "")

    assert [tensor.name for tensor in result.tensor_source.tensors] == ["weight"]
    fetch_manifest.assert_not_called()


def test_load_transfer_failure_reinitializes_and_tries_next_source():
    strat = RdmaStrategy()
    cands = _sources(3)
    strat._find_source_instances = MagicMock(return_value=cands)
    strat._fetch_worker_metadata = MagicMock(return_value=MagicMock())
    strat._load_as_target = MagicMock(
        side_effect=[StrategyFailed("receive failed", mutated=True), "loaded"]
    )
    original_result = MagicMock(name="original-result")
    retry_result = MagicMock(name="retry-result")
    ctx = MagicMock(global_rank=0)
    ctx.accelerator_backend.name = ""  # unknown target -> accelerator gate accepts
    ctx.adapter.reinit_for_retry.return_value = retry_result

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "modelexpress.load_strategy.rdma_strategy.worker_tensor_count",
            lambda w: 1,
        )
        result = strat.load(original_result, ctx)

    assert result == "loaded"
    assert strat._fetch_worker_metadata.call_count == 2
    assert strat._load_as_target.call_count == 2
    ctx.adapter.reinit_for_retry.assert_called_once()
    assert ctx.adapter.reinit_for_retry.call_args.args[0].value is original_result
    assert strat._load_as_target.call_args_list[1].args[0] is retry_result


def test_load_clean_transfer_failure_tries_next_source_without_reinit():
    strat = RdmaStrategy()
    strat._find_source_instances = MagicMock(return_value=_sources(2))
    strat._fetch_worker_metadata = MagicMock(return_value=MagicMock())
    strat._load_as_target = MagicMock(
        side_effect=[StrategyFailed("clean failure", mutated=False), "loaded"]
    )
    ctx = MagicMock(global_rank=0)
    ctx.accelerator_backend.name = ""

    assert strat.load(MagicMock(), ctx) == "loaded"
    ctx.adapter.reinit_for_retry.assert_not_called()


def test_load_requires_outer_reinit_after_reinit_then_metadata_miss():
    strat = RdmaStrategy()
    strat._find_source_instances = MagicMock(return_value=_sources(2))
    strat._fetch_worker_metadata = MagicMock(side_effect=[MagicMock(), None])
    strat._load_as_target = MagicMock(
        side_effect=StrategyFailed("receive failed", mutated=True)
    )
    ctx = MagicMock(global_rank=0)
    ctx.accelerator_backend.name = ""
    original_model = MagicMock(name="original-model")
    original_result = LoadResult(value=original_model, model=original_model)

    def reinit(result):
        result.value = None
        result.model = None
        retry_model = MagicMock(name="retry-model")
        return LoadResult(value=retry_model, model=retry_model)

    ctx.adapter.reinit_for_retry.side_effect = reinit

    with pytest.raises(StrategyFailed) as exc:
        strat.load(original_result, ctx)

    assert exc.value.mutated is True
    assert original_result.model is None
    ctx.adapter.reinit_for_retry.assert_called_once()


def test_load_requires_outer_reinit_after_reinit_then_clean_failure():
    strat = RdmaStrategy()
    strat._find_source_instances = MagicMock(return_value=_sources(2))
    strat._fetch_worker_metadata = MagicMock(return_value=MagicMock())
    strat._load_as_target = MagicMock(
        side_effect=[
            StrategyFailed("mutated failure", mutated=True),
            StrategyFailed("clean failure", mutated=False),
        ]
    )
    ctx = MagicMock(global_rank=0)
    ctx.accelerator_backend.name = ""
    ctx.adapter.reinit_for_retry.return_value = MagicMock(name="retry-result")

    with pytest.raises(StrategyFailed, match="clean failure") as exc:
        strat.load(MagicMock(), ctx)

    assert exc.value.mutated is True
    ctx.adapter.reinit_for_retry.assert_called_once()


def test_load_reinit_failure_remains_mutated():
    strat = RdmaStrategy()
    strat._find_source_instances = MagicMock(return_value=_sources(2))
    strat._fetch_worker_metadata = MagicMock(return_value=MagicMock())
    strat._load_as_target = MagicMock(
        side_effect=StrategyFailed("receive failed", mutated=True)
    )
    ctx = MagicMock(global_rank=0)
    ctx.accelerator_backend.name = ""
    ctx.adapter.reinit_for_retry.side_effect = RuntimeError("reinit failed")

    with pytest.raises(StrategyFailed, match="reinit failed") as exc:
        strat.load(MagicMock(), ctx)

    assert exc.value.mutated is True


def test_load_cleanup_failure_aborts_before_reinit():
    strat = RdmaStrategy()
    strat._find_source_instances = MagicMock(return_value=_sources(2))
    strat._fetch_worker_metadata = MagicMock(return_value=MagicMock())
    strat._load_as_target = MagicMock(
        side_effect=StrategyFailed("receive failed", mutated=True)
    )
    ctx = MagicMock(global_rank=0)
    ctx.accelerator_backend.name = ""
    manager = MagicMock()
    manager.shutdown.side_effect = RuntimeError("shutdown failed")
    ctx.nixl_manager = manager

    with pytest.raises(StrategyFailed, match="shutdown failed") as exc:
        strat.load(MagicMock(), ctx)

    assert exc.value.mutated is True
    assert strat._load_as_target.call_count == 1
    assert ctx.nixl_manager is manager
    ctx.adapter.reinit_for_retry.assert_not_called()


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
    strat._find_source_instances = MagicMock(return_value=_sources(1))
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


def test_load_records_transfer_retry_metrics(monkeypatch):
    m = _patched_metrics(monkeypatch)
    strat = RdmaStrategy()
    strat._find_source_instances = MagicMock(return_value=_sources(2))
    strat._fetch_worker_metadata = MagicMock(return_value=MagicMock())
    strat._load_as_target = MagicMock(
        side_effect=[StrategyFailed("receive failed", mutated=True), "loaded"]
    )
    ctx = MagicMock(global_rank=0)
    ctx.accelerator_backend.name = ""
    ctx.adapter.reinit_for_retry.return_value = MagicMock(name="retry-result")

    assert strat.load(MagicMock(), ctx) == "loaded"
    m.record_attempt.assert_any_call("random", "transfer_retry")
    assert m.observe_transfer_seconds.call_args_list[0].args[:2] == (
        "random",
        "retry",
    )


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
