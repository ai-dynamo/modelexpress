# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The chain sequences an attempt's phases; the strategy decides what they do.

The chain always runs:

    prepare -> load -> finalize                 on success
    prepare -> load -> abort                    when load fails

What each phase does, including how it differs on a reload, belongs to the
strategy. The base class gives streaming strategies their reload behaviour:
the engine's layerwise reload is entered in prepare() and left in finalize()
or abort(), and post-processing only runs on a cold load. Every phase writes
into the model, so a strategy reports a phase failure as a mutated one.

Registration has to see the model in its final layout, so the default
finalize() ends with it rather than load(). RDMA overrides finalize() without
it: its target buffers must carry registrations before the source can write
into them, so it registers mid-transfer.

Run: pytest tests/test_load_phases.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modelexpress.adapter import StrategyFailed
from modelexpress.load_strategy import _strategy_phases
from modelexpress.load_strategy.base import LoadStrategy
from modelexpress.load_strategy.context import LoadResult


class _RecordingStrategy(LoadStrategy):
    """Records the order its phases ran in."""

    name = "recording"

    def __init__(self, calls, *, fail_in=None):
        self.calls = calls
        self.fail_in = fail_in

    def _record(self, phase):
        self.calls.append(phase)
        if self.fail_in == phase:
            raise RuntimeError(f"{phase} blew up")

    def prepare(self, result, ctx):
        self._record("prepare")

    def load(self, result, ctx):
        self.calls.append("load")
        if self.fail_in == "load":
            raise StrategyFailed("load blew up", mutated=False)
        return result

    def finalize(self, result, ctx):
        self._record("finalize")

    def abort(self, result, ctx):
        self._record("abort")


class _StreamingStrategy(LoadStrategy):
    """Keeps the base class's prepare/finalize/abort, as streaming strategies do."""

    name = "streaming"

    def __init__(self, *, fail=False, fail_post_process=False):
        self.fail = fail
        self.fail_post_process = fail_post_process
        self.post_processed = False

    def load(self, result, ctx):
        if self.fail:
            raise StrategyFailed("load blew up", mutated=False)
        return result

    def post_process(self, result, ctx):
        if self.fail_post_process:
            raise RuntimeError("post_process blew up")
        self.post_processed = True


def _ctx(*, is_reload=False):
    return SimpleNamespace(
        global_rank=0,
        is_reload=is_reload,
        p2p_enabled=False,
        adapter=MagicMock(),
        identity=SimpleNamespace(model_name="m"),
    )


def _result():
    model = MagicMock()
    return LoadResult(value=model, model=model)


def _attempt(strategy, ctx):
    result = _result()
    with _strategy_phases(strategy, result, ctx):
        strategy.load(result, ctx)


def test_a_successful_attempt_runs_prepare_then_load_then_finalize():
    calls = []
    _attempt(_RecordingStrategy(calls), _ctx())
    assert calls == ["prepare", "load", "finalize"]


def test_a_failed_load_runs_abort_instead_of_finalize():
    calls = []
    with pytest.raises(StrategyFailed, match="load blew up"):
        _attempt(_RecordingStrategy(calls, fail_in="load"), _ctx())
    assert calls == ["prepare", "load", "abort"]


def test_a_failing_abort_propagates_in_place_of_the_load_failure():
    calls = []
    strategy = _RecordingStrategy(calls, fail_in="abort")
    strategy.load = MagicMock(side_effect=StrategyFailed("load blew up"))

    with pytest.raises(RuntimeError, match="abort blew up"):
        _attempt(strategy, _ctx())

    assert calls == ["prepare", "abort"]


@pytest.mark.parametrize("phase", ["prepare", "finalize"])
def test_the_chain_does_not_reinterpret_a_phase_failure(phase):
    """Reporting a phase failure as mutated is the strategy's job."""
    with pytest.raises(RuntimeError, match=f"{phase} blew up"):
        _attempt(_RecordingStrategy([], fail_in=phase), _ctx())


def test_streaming_cold_load_post_processes_without_layerwise_reload():
    strategy = _StreamingStrategy()
    ctx = _ctx()

    _attempt(strategy, ctx)

    assert strategy.post_processed is True
    ctx.adapter.begin_streaming_reload.assert_not_called()
    ctx.adapter.end_streaming_reload.assert_not_called()


def test_streaming_reload_runs_inside_layerwise_reload_instead():
    """Layerwise reload processes each layer as it materializes."""
    strategy = _StreamingStrategy()
    ctx = _ctx(is_reload=True)

    _attempt(strategy, ctx)

    assert strategy.post_processed is False
    ctx.adapter.begin_streaming_reload.assert_called_once()
    ctx.adapter.end_streaming_reload.assert_called_once()


def test_streaming_post_process_failure_is_reported_as_mutated():
    """Post-processing writes into the model, so a fallback needs a reinit."""
    with pytest.raises(StrategyFailed, match="post_process blew up") as exc:
        _attempt(_StreamingStrategy(fail_post_process=True), _ctx())

    assert exc.value.mutated is True


@pytest.mark.parametrize("hook", ["begin_streaming_reload", "end_streaming_reload"])
def test_streaming_reload_hook_failure_is_reported_as_mutated(hook):
    """A model left partly deferred to meta needs a reinit before a fallback."""
    ctx = _ctx(is_reload=True)
    getattr(ctx.adapter, hook).side_effect = RuntimeError(f"{hook} blew up")

    with pytest.raises(StrategyFailed, match=f"{hook} blew up") as exc:
        _attempt(_StreamingStrategy(), ctx)

    assert exc.value.mutated is True


def test_streaming_abort_failure_is_reported_as_mutated():
    ctx = _ctx(is_reload=True)
    ctx.adapter.end_streaming_reload.side_effect = RuntimeError("end blew up")

    with pytest.raises(StrategyFailed, match="end blew up") as exc:
        _attempt(_StreamingStrategy(fail=True), ctx)

    assert exc.value.mutated is True


def test_layerwise_reload_is_left_even_when_the_attempt_fails():
    """A fallback must not start against a half-materialized model."""
    ctx = _ctx(is_reload=True)

    with pytest.raises(StrategyFailed):
        _attempt(_StreamingStrategy(fail=True), ctx)

    ctx.adapter.end_streaming_reload.assert_called_once()


def test_finalize_registers_once_the_model_is_post_processed():
    """Discovery only means anything once the model is in its final layout."""
    strategy = _StreamingStrategy()

    with patch("modelexpress.load_strategy.base.register_tensors") as register:
        register.side_effect = lambda *a, **k: (
            None if strategy.post_processed else pytest.fail("registered too early")
        )
        _attempt(strategy, _ctx())

    register.assert_called_once()


def test_finalize_registers_after_leaving_layerwise_reload():
    ctx = _ctx(is_reload=True)
    events = []
    ctx.adapter.end_streaming_reload.side_effect = lambda r: events.append("end")

    with patch("modelexpress.load_strategy.base.register_tensors") as register:
        register.side_effect = lambda *a, **k: events.append("register")
        _attempt(_StreamingStrategy(), ctx)

    assert events == ["end", "register"]


def test_an_unpublishable_model_is_not_registered():
    strategy = _StreamingStrategy()
    result = _result()
    result.publishable = False
    ctx = _ctx()

    with patch("modelexpress.load_strategy.base.register_tensors") as register:
        with _strategy_phases(strategy, result, ctx):
            strategy.load(result, ctx)

    register.assert_not_called()


def test_a_failed_attempt_is_not_registered():
    with patch("modelexpress.load_strategy.base.register_tensors") as register:
        with pytest.raises(StrategyFailed):
            _attempt(_StreamingStrategy(fail=True), _ctx())

    register.assert_not_called()


def test_rdma_finalize_does_not_register_again():
    """RDMA already registered its target buffers to receive into them."""
    from modelexpress.load_strategy.rdma_strategy import RdmaStrategy

    with patch("modelexpress.load_strategy.base.register_tensors") as register:
        RdmaStrategy().finalize(_result(), _ctx())

    register.assert_not_called()


@pytest.mark.parametrize(
    "strategy_cls, delivery, hook",
    [
        ("instant_tensor_strategy.InstantTensorStrategy", "apply_weight_iter", "after_weight_iter_load"),
        ("model_streamer_strategy.ModelStreamerStrategy", "apply_weight_iter", "after_weight_iter_load"),
        ("gds_strategy.GdsStrategy", "apply_weight_iter", "after_weight_iter_load"),
        ("default_strategy.DefaultStrategy", "load_via_native", "after_native_load"),
        ("server_cache_strategy.ServerCacheStrategy", "load_via_native", "after_native_load"),
    ],
)
def test_post_processing_follows_how_the_strategy_loads(strategy_cls, delivery, hook):
    """The base class a strategy picks fixes both its delivery and its hook."""
    import importlib

    from modelexpress.adapter import EngineAdapter

    module, cls = strategy_cls.split(".")
    strategy = getattr(importlib.import_module(f"modelexpress.load_strategy.{module}"), cls)()
    ctx = _ctx()

    strategy.post_process(_result(), ctx)

    assert getattr(EngineAdapter, delivery) in strategy.requires
    getattr(ctx.adapter, hook).assert_called_once()
