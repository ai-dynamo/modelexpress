# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L2: the per-strategy timing recorded from the chain loop.

The invariant these tests exist to hold is that a strategy's absence from the
histogram has exactly one meaning. There are three ways a strategy can record
no attempt -- filtered out before the loop, eligible but never reached because
an earlier one succeeded, or tried and recorded -- and only the first is a skip.
Conflating the first two is the failure mode that makes the panel unreadable,
so it is asserted directly rather than inferred.
"""

import re

import pytest
import torch
from unittest.mock import MagicMock, patch

from modelexpress.adapter import StrategyFailed, StrategyRecoveryError
from modelexpress import p2p_pb2
from modelexpress.load_strategy import LoadContext, LoadStrategyChain
from modelexpress.metrics import (
    LOAD_STRATEGIES,
    LOAD_STRATEGY_OUTCOMES,
    LOAD_STRATEGY_SKIP_REASONS,
    MetricsCollector,
)

_BASE = "modelexpress.load_strategy"
_CLASSES = {
    "rdma": f"{_BASE}.rdma_strategy.RdmaStrategy",
    "server-cache": f"{_BASE}.server_cache_strategy.ServerCacheStrategy",
    "instant_tensor": f"{_BASE}.instant_tensor_strategy.InstantTensorStrategy",
    "model_streamer": f"{_BASE}.model_streamer_strategy.ModelStreamerStrategy",
    "gds": f"{_BASE}.gds_strategy.GdsStrategy",
    "default": f"{_BASE}.default_strategy.DefaultStrategy",
}


def _ctx(engine="vllm", model="org/model"):
    return LoadContext(
        engine=engine,
        model_config=MagicMock(),
        load_config=MagicMock(),
        target_device=torch.device("cpu"),
        global_rank=0,
        worker_rank=0,
        device_id=0,
        identity=p2p_pb2.SourceIdentity(model_name=model),
        mx_client=MagicMock(),
        worker_id="w",
        adapter=MagicMock(),
    )


@pytest.fixture
def collector(monkeypatch):
    """A collector on a private registry, wired into the chain."""
    from prometheus_client import CollectorRegistry

    monkeypatch.setenv("MX_METRICS_ENABLED", "1")
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    c = MetricsCollector(registry=CollectorRegistry())
    monkeypatch.setattr(f"{_BASE}.load_metrics", c)
    return c


def _exposition(collector):
    from prometheus_client import generate_latest

    return generate_latest(collector._exposition_registry()).decode()


def _attempts(text):
    """{(strategy, outcome): count} from mx_load_strategy_seconds_count."""
    out = {}
    for line in text.splitlines():
        m = re.match(
            r'mx_load_strategy_seconds_count\{.*?outcome="([^"]+)".*?'
            r'strategy="([^"]+)".*?\} (\S+)',
            line,
        )
        if m:
            out[(m.group(2), m.group(1))] = float(m.group(3))
    return out


def _skips(text):
    """{(strategy, reason): count} from mx_load_strategy_skipped_total."""
    out = {}
    for line in text.splitlines():
        m = re.match(
            r'mx_load_strategy_skipped_total\{.*?reason="([^"]+)".*?'
            r'strategy="([^"]+)".*?\} (\S+)',
            line,
        )
        if m:
            out[(m.group(2), m.group(1))] = float(m.group(3))
    return out


def _run(eligible, loads, ctx=None):
    """Run the chain with a chosen eligibility set and per-strategy load behavior.

    ``eligible`` maps a strategy name to None (runnable) or a skip reason. A
    strategy the caller does not name is skipped, not made eligible: defaulting
    the other way lets an unpatched real ``load`` run inside a unit test, which
    is how this helper was wrong the first time.
    """
    ctx = ctx or _ctx()
    model = MagicMock()
    stack = []
    for name, path in _CLASSES.items():
        stack.append(patch(f"{path}.skip_reason", return_value=eligible.get(name, "other")))
        behavior = loads.get(name)
        if behavior is not None:
            stack.append(patch(f"{path}.load", **behavior))
        stack.append(patch(f"{path}.rollback", return_value=None))
    for p in stack:
        p.start()
    try:
        return LoadStrategyChain.run(model, ctx)
    finally:
        for p in reversed(stack):
            p.stop()


def _succeed():
    result = MagicMock()
    result.value = MagicMock()
    return {"return_value": result}


# ---------------------------------------------------------------------------
# The three ways a strategy can record nothing
# ---------------------------------------------------------------------------


def test_a_skipped_strategy_is_counted_as_skipped_and_never_timed(collector):
    with patch(f"{_BASE}.publish_source_if_supported"):
        _run(
            eligible={
                "rdma": "nixl_unavailable",
                "server-cache": "prefetch_disabled",
                "instant_tensor": "package_missing",
                "model_streamer": "package_missing",
                "gds": "driver_unavailable",
                "default": None,
            },
            loads={"default": _succeed()},
        )
    text = _exposition(collector)
    assert _skips(text) == {
        ("rdma", "nixl_unavailable"): 1.0,
        ("server-cache", "prefetch_disabled"): 1.0,
        ("instant_tensor", "package_missing"): 1.0,
        ("model_streamer", "package_missing"): 1.0,
        ("gds", "driver_unavailable"): 1.0,
    }
    # A skipped strategy is not an attempt of zero duration. It is not an
    # attempt, and a zero would sit in the bottom bucket looking like a fast one.
    assert _attempts(text) == {("default", "success"): 1.0}


def test_an_eligible_strategy_the_chain_never_reached_records_neither(collector):
    """The distinction the skip counter exists to preserve.

    rdma succeeds, so the five strategies behind it are eligible and simply
    never tried. Counting those as skips would make ``default`` look
    permanently skipped on every healthy load, which is the opposite of true.
    """
    with patch(f"{_BASE}.publish_source_if_supported"):
        _run(
            eligible=dict.fromkeys(_CLASSES, None),
            loads={"rdma": _succeed()},
        )
    text = _exposition(collector)
    assert _attempts(text) == {("rdma", "success"): 1.0}
    assert _skips(text) == {}


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "side_effect,expected",
    [
        (StrategyFailed("clean miss", mutated=False), "fallback"),
        (StrategyFailed("dirty", mutated=True), "fallback_dirty"),
        (RuntimeError("unexpected"), "error"),
    ],
)
def test_each_miss_records_its_own_outcome(collector, side_effect, expected):
    ctx = _ctx()
    ctx.adapter.reinit_for_retry = MagicMock(return_value=MagicMock())
    with patch(f"{_BASE}.publish_source_if_supported"):
        _run(
            eligible={"rdma": None, "default": None},
            loads={"rdma": {"side_effect": side_effect}, "default": _succeed()},
            ctx=ctx,
        )
    attempts = _attempts(_exposition(collector))
    assert attempts[("rdma", expected)] == 1.0
    assert attempts[("default", "success")] == 1.0


def test_a_failed_recovery_is_recorded_before_the_chain_fails_closed(collector):
    """The outcome that only exists because the timing is in a finally.

    StrategyRecoveryError re-raises out of the loop, so an except-clause
    recorder would never run for it -- and this is the case most worth having,
    since the whole chain aborts here.
    """
    with patch(f"{_BASE}.publish_source_if_supported"):
        with pytest.raises(StrategyRecoveryError):
            _run(
                eligible={"rdma": None, "default": None},
                loads={"rdma": {"side_effect": StrategyRecoveryError("no way back")}},
            )
    attempts = _attempts(_exposition(collector))
    assert attempts == {("rdma", "recovery_error"): 1.0}


def test_every_eligible_strategy_tried_records_exactly_one_observation(collector):
    with patch(f"{_BASE}.publish_source_if_supported"):
        _run(
            eligible=dict.fromkeys(_CLASSES, None),
            loads={
                name: {"side_effect": StrategyFailed("miss", mutated=False)}
                for name in _CLASSES
                if name != "default"
            }
            | {"default": _succeed()},
        )
    attempts = _attempts(_exposition(collector))
    assert sum(attempts.values()) == len(_CLASSES)
    assert all(v == 1.0 for v in attempts.values())


# ---------------------------------------------------------------------------
# Label domains
# ---------------------------------------------------------------------------


def test_an_unrecognized_strategy_is_dropped_but_an_unknown_outcome_clamps(collector):
    """The asymmetry between the two recorders, asserted so it stays deliberate.

    The histogram partitions the chain phase, so folding a stray name into an
    existing strategy would inflate that strategy while the sum still looked
    sound. The counter partitions nothing, so a catch-all costs a distinction
    rather than a wrong total.
    """
    collector.observe_load_strategy_seconds("vllm", "m", "not_a_strategy", "success", 1.0)
    collector.observe_load_strategy_seconds("vllm", "m", "rdma", "not_an_outcome", 1.0)
    collector.record_strategy_skipped("vllm", "not_a_strategy", "not_a_reason")

    text = _exposition(collector)
    assert ("not_a_strategy", "success") not in _attempts(text)
    assert _attempts(text) == {("rdma", "error"): 1.0}
    assert _skips(text) == {("other", "other"): 1.0}


def test_an_out_of_tree_engine_clamps_without_opening_the_label_domain(collector):
    collector.observe_load_strategy_seconds("someone_elses_engine", "m", "rdma", "success", 1.0)
    collector.record_strategy_skipped("someone_elses_engine", "gds", "driver_unavailable")
    text = _exposition(collector)
    assert 'engine="other"' in text
    assert "someone_elses_engine" not in text


# ---------------------------------------------------------------------------
# The contract that keeps the label domains honest
# ---------------------------------------------------------------------------


def test_every_strategy_overrides_skip_reason_and_none_overrides_is_available():
    """The chain calls skip_reason, so an is_available override would be dead.

    This is the drift that would not fail loudly: a strategy overriding
    is_available still passes its own unit tests, while the chain silently
    stops honoring it. One definition of is_available, on the base class, is
    what makes that impossible.
    """
    from modelexpress.load_strategy.base import LoadStrategy
    from modelexpress.load_strategy.default_strategy import DefaultStrategy
    from modelexpress.load_strategy.gds_strategy import GdsStrategy
    from modelexpress.load_strategy.instant_tensor_strategy import InstantTensorStrategy
    from modelexpress.load_strategy.model_streamer_strategy import ModelStreamerStrategy
    from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
    from modelexpress.load_strategy.server_cache_strategy import ServerCacheStrategy

    classes = [
        RdmaStrategy,
        ServerCacheStrategy,
        InstantTensorStrategy,
        ModelStreamerStrategy,
        GdsStrategy,
        DefaultStrategy,
    ]
    for cls in classes:
        assert "skip_reason" in vars(cls), f"{cls.__name__} must override skip_reason"
        assert "is_available" not in vars(cls), (
            f"{cls.__name__} overrides is_available, which the chain no longer calls"
        )
    assert "is_available" in vars(LoadStrategy)


def test_the_strategy_label_domain_matches_the_chain_and_the_names_are_the_tracers():
    """LOAD_STRATEGIES is closed by construction, so prove it against the source.

    ``server-cache`` keeps its hyphen here because that is the value the tracer
    already records as weight_loading_strategy. Normalizing it in the metric
    would desynchronize the two for no gain, so the shape is pinned.
    """
    from modelexpress.load_strategy.default_strategy import DefaultStrategy
    from modelexpress.load_strategy.gds_strategy import GdsStrategy
    from modelexpress.load_strategy.instant_tensor_strategy import InstantTensorStrategy
    from modelexpress.load_strategy.model_streamer_strategy import ModelStreamerStrategy
    from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
    from modelexpress.load_strategy.server_cache_strategy import ServerCacheStrategy

    built = (
        RdmaStrategy.name,
        ServerCacheStrategy.name,
        InstantTensorStrategy.name,
        ModelStreamerStrategy.name,
        GdsStrategy.name,
        DefaultStrategy.name,
    )
    assert built == LOAD_STRATEGIES
    assert "server-cache" in LOAD_STRATEGIES


def test_every_skip_reason_a_strategy_can_return_is_in_the_label_domain():
    """Grep the reasons out of the source rather than trusting the enum.

    A reason that is not in LOAD_STRATEGY_SKIP_REASONS silently clamps to
    ``other``, so the panel loses the distinction without anything failing.
    """
    import pathlib

    pkg = pathlib.Path(__import__("modelexpress").__file__).parent
    returned = set()
    for path in (pkg / "load_strategy").glob("*.py"):
        src = path.read_text()
        body = re.search(r"def skip_reason\(.*?\n(?=    def |\Z)", src, re.S)
        if body:
            returned |= set(re.findall(r'return "([a-z_]+)"', body.group(0)))
    assert returned, "no skip reasons found -- the scan is broken, not the code"
    assert returned <= set(LOAD_STRATEGY_SKIP_REASONS), (
        f"unlabelled skip reasons: {sorted(returned - set(LOAD_STRATEGY_SKIP_REASONS))}"
    )


def test_the_chain_records_only_outcomes_that_are_in_the_label_domain():
    import pathlib

    src = (
        pathlib.Path(__import__("modelexpress").__file__).parent
        / "load_strategy"
        / "__init__.py"
    ).read_text()
    assigned = set(re.findall(r'outcome = "([a-z_]+)"', src))
    assigned |= set(re.findall(r'"([a-z_]+)" if e\.mutated else "([a-z_]+)"', src)[0])
    assert assigned <= set(LOAD_STRATEGY_OUTCOMES), (
        f"unlabelled outcomes: {sorted(assigned - set(LOAD_STRATEGY_OUTCOMES))}"
    )
