# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for S3RdmaStrategy gating, rank agreement, and fallback semantics."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from modelexpress.adapter import EngineAdapter, StrategyFailed, StrategyRecoveryError
from modelexpress.load_strategy import LoadContext, LoadStrategy, execute_load_strategies
from modelexpress.load_strategy.context import LoadResult
from modelexpress.load_strategy.s3_rdma_strategy import S3RdmaStrategy

URI = "s3://models/llama"


class RecordingAdapter(EngineAdapter):
    def __init__(
        self,
        events,
        *,
        apply_error=None,
        after_error=None,
        consume=True,
        peers=None,
        gather_error=None,
    ):
        self.events = events
        self.apply_error = apply_error
        self.after_error = after_error
        self.consume = consume
        self.peers = peers or {}
        self.gather_error = gather_error

    def apply_weight_iter(self, result, weights_iter):
        self.events.append("apply")
        if self.consume:
            for _ in weights_iter:
                pass
        else:
            next(weights_iter)
        if self.apply_error is not None:
            raise self.apply_error
        return result

    def after_weight_iter_load(self, result):
        self.events.append("after")
        if self.after_error is not None:
            raise self.after_error
        return result

    def all_gather_state(self, state):
        self.events.append(("gather", state[0]))
        if self.gather_error is not None:
            raise self.gather_error
        return (state, *self.peers.get(state[0], ()))

    def reinit_for_retry(self, result):
        self.events.append("reinit")
        return result


def fake_loader_cls(events, *, prepare_error=None, iter_error=None, fingerprint="fp"):
    class FakeLoader:
        def __init__(self, backend, device_id, config):
            events.append("init")
            self.finished = False

        def prepare(self, uri):
            events.append(("prepare", uri))
            if prepare_error is not None:
                raise prepare_error
            return SimpleNamespace(fingerprint=fingerprint)

        def iter_tensors(self):
            events.append("iter")

            def generate():
                try:
                    yield "w", torch.zeros(2)
                    yield "b", torch.ones(1)
                    if iter_error is not None:
                        raise iter_error
                    self.finished = True
                finally:
                    events.append("iter_closed")

            return generate()

        def release(self):
            events.append("release")

        def shutdown(self):
            events.append("shutdown")

        def log_timing(self):
            events.append("log_timing")

    return FakeLoader


def make_ctx(adapter, backend):
    return LoadContext(
        model_config=MagicMock(),
        load_config=MagicMock(),
        target_device=torch.device("cpu"),
        global_rank=0,
        worker_rank=0,
        local_rank=0,
        device_id=0,
        identity=MagicMock(),
        mx_client=MagicMock(),
        worker_id="test-worker",
        adapter=adapter,
        accelerator_backend=backend,
    )


@pytest.fixture
def s3_rdma_env(monkeypatch):
    monkeypatch.setenv("MX_S3_RDMA_ENABLED", "1")
    monkeypatch.setenv("MX_MODEL_URI", URI)
    monkeypatch.delenv("MX_VMM_ARENA", raising=False)


@pytest.fixture
def run(s3_rdma_env, mock_accelerator_backend_cls):
    """Run S3RdmaStrategy.load with a fake loader; returns (events, register_mock, outcome)."""

    def _run(adapter_kwargs=None, loader_kwargs=None):
        events: list = []
        adapter = RecordingAdapter(events, **(adapter_kwargs or {}))
        ctx = make_ctx(adapter, mock_accelerator_backend_cls(s3_rdma=True))
        with patch(
            "modelexpress.s3_rdma_loader.MxS3RdmaLoader", fake_loader_cls(events, **(loader_kwargs or {}))
        ), patch("modelexpress.load_strategy.s3_rdma_strategy.register_tensors") as register:
            try:
                outcome = S3RdmaStrategy().load(MagicMock(), ctx)
            except Exception as exc:
                outcome = exc
        return events, register, outcome

    return _run


class TestIsAvailable:
    @pytest.fixture
    def ctx(self, mock_accelerator_backend_cls):
        return make_ctx(RecordingAdapter([]), mock_accelerator_backend_cls(s3_rdma=True))

    @pytest.fixture(autouse=True)
    def _s3_rdma_present(self):
        with patch("modelexpress.s3_rdma_transfer.is_s3_rdma_available", return_value=True):
            yield

    def test_available(self, s3_rdma_env, ctx):
        assert S3RdmaStrategy().is_available(ctx) is True

    def test_disabled_by_default(self, s3_rdma_env, ctx, monkeypatch):
        monkeypatch.delenv("MX_S3_RDMA_ENABLED")
        assert S3RdmaStrategy().is_available(ctx) is False

    @pytest.mark.parametrize("uri", ["", "gs://models/llama", "/models/llama"])
    def test_requires_s3_uri(self, s3_rdma_env, ctx, monkeypatch, uri):
        monkeypatch.setenv("MX_MODEL_URI", uri)
        assert S3RdmaStrategy().is_available(ctx) is False

    def test_requires_apply_weight_iter(self, s3_rdma_env, mock_accelerator_backend_cls):
        class NoIter(EngineAdapter):
            pass

        ctx = make_ctx(NoIter(), mock_accelerator_backend_cls(s3_rdma=True))
        assert S3RdmaStrategy().is_available(ctx) is False

    def test_vmm_arena_excluded(self, s3_rdma_env, ctx, monkeypatch):
        monkeypatch.setenv("MX_VMM_ARENA", "1")
        assert S3RdmaStrategy().is_available(ctx) is False

    def test_backend_without_s3_rdma_skips_probe(self, s3_rdma_env, mock_accelerator_backend_cls):
        ctx = make_ctx(RecordingAdapter([]), mock_accelerator_backend_cls(s3_rdma=False))
        with patch(
            "modelexpress.s3_rdma_transfer.is_s3_rdma_available",
            side_effect=AssertionError("system probe should not run"),
        ):
            assert S3RdmaStrategy().is_available(ctx) is False

    def test_multi_rank_requires_gather(self, s3_rdma_env, mock_accelerator_backend_cls):
        class IterOnly(EngineAdapter):
            def apply_weight_iter(self, result, weights_iter):
                return result

        ctx = make_ctx(IterOnly(), mock_accelerator_backend_cls(s3_rdma=True))
        with patch("modelexpress.load_strategy.s3_rdma_strategy._world_size", return_value=2):
            assert S3RdmaStrategy().is_available(ctx) is False
        with patch("modelexpress.load_strategy.s3_rdma_strategy._world_size", return_value=1):
            assert S3RdmaStrategy().is_available(ctx) is True

    def test_nixl_obj_missing(self, s3_rdma_env, ctx):
        with patch("modelexpress.s3_rdma_transfer.is_s3_rdma_available", return_value=False):
            assert S3RdmaStrategy().is_available(ctx) is False


class TestLoad:
    def test_success_order(self, run):
        events, register, outcome = run()
        assert isinstance(outcome, LoadResult)
        assert events == [
            "init",
            ("prepare", URI),
            ("gather", "prepared"),
            "iter",
            "apply",
            "iter_closed",
            "release",
            ("gather", "streamed"),
            "after",
            ("gather", "finalized"),
            "shutdown",
            "log_timing",
        ]
        register.assert_called_once()

    def test_local_prepare_failure_is_clean(self, run):
        events, register, outcome = run(loader_kwargs={"prepare_error": RuntimeError("no fabric")})
        assert isinstance(outcome, StrategyFailed) and outcome.mutated is False
        assert "no fabric" in str(outcome)
        assert "apply" not in events and "shutdown" in events
        register.assert_not_called()

    def test_peer_prepare_failure_is_clean(self, run):
        peers = {"prepared": [("prepared", False, "S3RdmaBackendUnavailable: down", None)]}
        events, _, outcome = run(adapter_kwargs={"peers": peers})
        assert isinstance(outcome, StrategyFailed) and outcome.mutated is False
        assert "rank 1" in str(outcome)
        assert "apply" not in events

    def test_fingerprint_mismatch(self, run):
        peers = {"prepared": [("prepared", True, "", "other")]}
        _, _, outcome = run(adapter_kwargs={"peers": peers})
        assert isinstance(outcome, StrategyFailed) and outcome.mutated is False
        assert "different checkpoint" in str(outcome)

    def test_apply_failure_is_mutated_and_skips_post_processing(self, run):
        events, _, outcome = run(adapter_kwargs={"apply_error": RuntimeError("bad shape")})
        assert isinstance(outcome, StrategyFailed) and outcome.mutated is True
        assert "after" not in events
        assert events.index("release") < events.index(("gather", "streamed"))

    def test_stream_error_is_mutated(self, run):
        _, _, outcome = run(loader_kwargs={"iter_error": RuntimeError("short read")})
        assert isinstance(outcome, StrategyFailed) and outcome.mutated is True

    def test_partial_consumption_is_failure(self, run):
        events, _, outcome = run(adapter_kwargs={"consume": False})
        assert isinstance(outcome, StrategyFailed) and outcome.mutated is True
        assert "stopped consuming" in str(outcome)
        assert "iter_closed" in events

    def test_peer_stream_failure_skips_post_processing(self, run):
        peers = {"streamed": [("streamed", False, "S3RdmaIntegrityError: tail", None)]}
        events, _, outcome = run(adapter_kwargs={"peers": peers})
        assert isinstance(outcome, StrategyFailed) and outcome.mutated is True
        assert "after" not in events

    def test_after_failure_is_mutated(self, run):
        _, register, outcome = run(adapter_kwargs={"after_error": RuntimeError("quant")})
        assert isinstance(outcome, StrategyFailed) and outcome.mutated is True
        register.assert_not_called()

    def test_gather_error_aborts_chain(self, run):
        _, _, outcome = run(adapter_kwargs={"gather_error": RuntimeError("pg down")})
        assert isinstance(outcome, StrategyRecoveryError)

    @pytest.mark.parametrize(
        "peer",
        [("prepared", True), ("streamed", True, "", None), "garbage"],
    )
    def test_malformed_or_mismatched_peer_state(self, run, peer):
        _, _, outcome = run(adapter_kwargs={"peers": {"prepared": [peer]}})
        assert isinstance(outcome, StrategyRecoveryError)


class _NextStrategy(LoadStrategy):
    name = "next"

    def __init__(self, events):
        self.events = events

    def load(self, result, ctx):
        self.events.append("next")
        return result


def test_mutated_failure_reinitializes_before_next_strategy(s3_rdma_env, mock_accelerator_backend_cls):
    events: list = []
    adapter = RecordingAdapter(events, apply_error=RuntimeError("bad"))
    ctx = make_ctx(adapter, mock_accelerator_backend_cls(s3_rdma=True))
    with patch("modelexpress.s3_rdma_transfer.is_s3_rdma_available", return_value=True), patch(
        "modelexpress.s3_rdma_loader.MxS3RdmaLoader", fake_loader_cls(events)
    ), patch("modelexpress.load_strategy.publish_source_if_supported"):
        execute_load_strategies(MagicMock(), ctx, [S3RdmaStrategy(), _NextStrategy(events)])
    assert events.index("reinit") < events.index("next")
    assert events.index("shutdown") < events.index("reinit")
