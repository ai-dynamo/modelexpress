# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the opt-in ModelExpress metrics collector.

Two groups. The first locks the guarantees the module has always promised:
every recording call is a no-op when metrics are disabled, and nothing ever
raises into the load path.

The second is a regression test per defect in the exposition path. Each one
fails against the previous implementation, which is the point — every defect
produced plausible-looking but wrong output rather than an error, so only a test
that asserts the *corrected* shape can tell them apart:

* **D2** one rank bound the port and N-1 gave up, so the endpoint served one
  rank's data while presenting as the pod's.
* **D3** the Pushgateway grouping key was the hostname, and the push is a PUT
  that replaces the whole group, so ranks on a host erased each other.
* **D4** initialization was reachable only from a recording call, so a run that
  skipped P2P produced output byte-identical to ``MX_METRICS_ENABLED=0``.
* **D6** the push ran from ``atexit``, which does not run on SIGKILL.
* **D7** ``source_worker_id`` was a per-process uuid, so its label domain grew
  without bound.
* **D8** ``mx_p2p_candidates{stage="listed"}`` could never observe zero, and a
  ListSources RPC failure recorded nothing at all.

D1 (the Helm annotation pointed at the gRPC port) is a deployment artifact,
covered by ``test_metrics_deployment.py`` and by the server integration test in
``modelexpress_server/tests/in_process_server.rs``.
"""

from __future__ import annotations

import errno
import os
import re
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from modelexpress.metrics import (
    ENV_ENABLED,
    LIST_SOURCES_RESULTS,
    _XSLOW_BUCKETS,
    MetricsCollector,
    enable_metrics,
    push_metrics_if_enabled,
    reset_multiproc_dir,
)

_RECORDERS = [
    ("record_selection", ("random", "a1b2c3d4")),
    ("record_attempt", ("random", "success")),
    ("record_metadata_failure", ("random",)),
    ("record_list_sources", ("random", "ok")),
    ("observe_candidates", ("random", "listed", 2)),
    ("observe_selection_seconds", ("random", 0.001)),
    ("observe_transfer_seconds", ("random", "success", 1.0)),
    ("record_nixl_error", ("timeout",)),
    ("record_nixl_receive", ("complete",)),
    ("observe_load_seconds", ("vllm", "Qwen/Qwen2.5-0.5B-Instruct", "main", "success", 12.0)),
    ("observe_load_phase_seconds", ("vllm", "Qwen/Qwen2.5-0.5B-Instruct", "chain", 4.0)),
]

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _fresh_collector(monkeypatch, **env):
    """An initialized collector on a private registry.

    The registry is injected so the real families can be built without polluting
    the process-global one. That is not sufficient on its own: ``_exposition()``
    reads back through ``_exposition_registry()``, which under multiprocess mode
    ignores both registries and merges every ``.db`` file in
    ``PROMETHEUS_MULTIPROC_DIR``. So the variable is *deleted* here unless a test
    asks for one — inheriting the ambient value would make every assertion read
    process-external state that survives across pytest sessions.
    """
    from prometheus_client import CollectorRegistry

    monkeypatch.setenv(ENV_ENABLED, "1")
    env.setdefault("PROMETHEUS_MULTIPROC_DIR", None)
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    return MetricsCollector(registry=CollectorRegistry())


def _exposition(collector) -> str:
    from prometheus_client import generate_latest

    return generate_latest(collector._exposition_registry()).decode()


@pytest.fixture
def squatted_port():
    """A port already held by a listener, so the collector's bind must fail.

    Bound on ``0.0.0.0`` because that is where ``start_http_server`` binds:
    squatting only ``127.0.0.1`` leaves the wildcard address free and the bind
    quietly succeeds, which would leave a real HTTP server running for the rest
    of the session and silently invert what the test asserts.
    """
    squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    squatter.bind(("0.0.0.0", 0))
    squatter.listen(1)
    try:
        yield squatter.getsockname()[1]
    finally:
        squatter.close()


# ---------------------------------------------------------------------------
# Invariants: never on, never fatal
# ---------------------------------------------------------------------------


def test_recorders_are_noop_when_disabled(monkeypatch):
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    m = MetricsCollector()
    for name, args in _RECORDERS:
        assert getattr(m, name)(*args) is None
    # Disabled => collectors are never constructed and the server never starts.
    assert m._ready is False
    assert m._init_attempted is True
    assert not hasattr(m, "selections")
    assert m._server_started is False


def test_recorders_never_raise_into_load_path(monkeypatch):
    # Force the "enabled + initialized" state without touching the global
    # prometheus registry, then make a collector blow up on use.
    m = MetricsCollector()
    m._ready = True
    m._init_attempted = True
    boom = MagicMock()
    boom.labels.side_effect = RuntimeError("collector exploded")
    m.selections = boom
    m.attempts = boom
    m.metadata_failures = boom
    m.list_sources = boom
    m.candidates = boom
    m.selection_seconds = boom
    m.transfer_seconds = boom
    m.nixl_errors = boom
    m.nixl_receives = boom
    # None of these may propagate the RuntimeError.
    for name, args in _RECORDERS:
        getattr(m, name)(*args)


def test_push_is_noop_when_disabled(monkeypatch):
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    # No exception, no network call.
    push_metrics_if_enabled()


def test_enable_is_a_noop_and_never_raises_when_disabled(monkeypatch):
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    assert enable_metrics() is False


# ---------------------------------------------------------------------------
# D4: initialization must not depend on a recording call
# ---------------------------------------------------------------------------


def test_enable_brings_up_the_exporter_with_zero_recordings(monkeypatch):
    """A run that records nothing must still be distinguishable from disabled.

    Before, ``_ensure()`` was reachable only from the six recorders, whose only
    caller was the RDMA strategy. A load that fell back to a local or
    HuggingFace path started no endpoint and pushed nothing, so the run you most
    need to diagnose produced output byte-identical to MX_METRICS_ENABLED=0.
    """
    collector = _fresh_collector(monkeypatch, MX_METRICS_PORT=None, MX_METRICS_PUSHGATEWAY=None)
    monkeypatch.setattr("modelexpress.metrics.metrics", collector)

    assert enable_metrics() is True

    assert collector._ready is True
    assert collector._server_started is True
    body = _exposition(collector)
    # Proof of life without a single recorder having been called.
    assert "mx_build_info" in body


def test_build_info_is_a_gauge_set_to_one(monkeypatch):
    """Never an Info: under multiprocess mode an Info exports nothing silently.

    An Info would pass its own health check while being invisible, and would
    silently empty every ``group_left`` join that uses it to attach the process
    constants.
    """
    collector = _fresh_collector(monkeypatch)
    assert collector._ensure() is True

    body = _exposition(collector)
    assert "# TYPE mx_build_info gauge" in body
    series = [
        line for line in body.splitlines() if line.startswith("mx_build_info{")
    ]
    assert len(series) == 1, body
    assert series[0].endswith(" 1.0"), series[0]
    # ``_info`` is prometheus_client's Info suffix; its presence would mean the
    # wrong representation was used.
    assert "mx_build_info_info" not in body


# ---------------------------------------------------------------------------
# D7: no unbounded label
# ---------------------------------------------------------------------------


def test_selection_counter_carries_no_per_process_worker_id(monkeypatch):
    """``source_worker_id`` is ``uuid4().hex[:8]``, minted fresh per process.

    Its domain is bounded by process count over time rather than by cluster
    size, so a long-lived Prometheus watching a fleet with pod churn grew series
    without bound. A 16-node dev cluster masks this; production does not.

    The call site still passes the peer id — the collector is what drops it — so
    this asserts on the exposition rather than on the call.
    """
    collector = _fresh_collector(monkeypatch, MX_METRICS_SOURCE_ID_LABEL=None)
    assert collector._ensure() is True

    collector.record_selection("random", "a1b2c3d4")
    collector.record_selection("random", "e5f6a7b8")

    body = _exposition(collector)
    assert "source_worker_id" not in body
    assert "a1b2c3d4" not in body
    selections = [
        line
        for line in body.splitlines()
        if line.startswith("mx_p2p_source_selections_total{")
    ]
    # Two selections of two different peers collapse to one series.
    assert len(selections) == 1, body
    assert selections[0].endswith(" 2.0"), selections[0]


def test_source_id_label_can_be_opted_into_for_benchmarks(monkeypatch):
    """The escape hatch for selection-policy comparison runs.

    Comparing `random` against `rendezvous_hash` or `load_aware` needs the
    per-peer breakdown to answer "is one peer picked disproportionately". A
    benchmark Prometheus is ephemeral and small, so the cardinality that makes
    this label unsafe on a fleet is harmless there.
    """
    collector = _fresh_collector(monkeypatch, MX_METRICS_SOURCE_ID_LABEL="1")
    assert collector._ensure() is True

    collector.record_selection("random", "a1b2c3d4")
    collector.record_selection("random", "a1b2c3d4")
    collector.record_selection("random", "e5f6a7b8")

    body = _exposition(collector)
    selections = [
        line
        for line in body.splitlines()
        if line.startswith("mx_p2p_source_selections_total{")
    ]
    assert len(selections) == 2, body
    assert any('source_worker_id="a1b2c3d4"' in line and line.endswith(" 2.0") for line in selections)
    assert any('source_worker_id="e5f6a7b8"' in line and line.endswith(" 1.0") for line in selections)


def test_source_id_label_survives_a_caller_that_passes_nothing(monkeypatch):
    """Label arity is fixed at construction, so a missing id must not raise.

    The family's label set is decided once. A recorder that passed two values to
    a three-label family would raise inside the load path — which the blanket
    except would swallow, silently dropping every selection.
    """
    collector = _fresh_collector(monkeypatch, MX_METRICS_SOURCE_ID_LABEL="1")
    assert collector._ensure() is True

    collector.record_selection("random")

    body = _exposition(collector)
    assert 'source_worker_id="unknown"' in body, body


def test_source_id_label_is_latched_not_re_read_per_call(monkeypatch):
    """Flipping the env after the family exists must not break recording.

    prometheus_client fixes a family's label set at construction. Reading the
    variable per call would make every `.labels()` mismatch after a flip, and
    the recorder's blanket except would turn that into silence rather than an
    error.
    """
    collector = _fresh_collector(monkeypatch, MX_METRICS_SOURCE_ID_LABEL=None)
    assert collector._ensure() is True

    monkeypatch.setenv("MX_METRICS_SOURCE_ID_LABEL", "1")
    collector.record_selection("random", "a1b2c3d4")

    body = _exposition(collector)
    assert "source_worker_id" not in body
    selections = [
        line
        for line in body.splitlines()
        if line.startswith("mx_p2p_source_selections_total{")
    ]
    assert len(selections) == 1, body
    assert selections[0].endswith(" 1.0"), "the selection was silently dropped"


# ---------------------------------------------------------------------------
# D8: the funnel must be able to observe zero, and an RPC failure must not be
# silence
# ---------------------------------------------------------------------------


def test_candidate_funnel_can_observe_zero(monkeypatch):
    """The zero bucket is what separates "no peers" from "all filtered out"."""
    collector = _fresh_collector(monkeypatch)
    assert collector._ensure() is True

    collector.observe_candidates("random", "listed", 0)

    body = _exposition(collector)
    zero_bucket = [
        line
        for line in body.splitlines()
        if line.startswith("mx_p2p_candidates_bucket{")
        and 'stage="listed"' in line
        and 'le="0.0"' in line
    ]
    assert zero_bucket, body
    assert zero_bucket[0].endswith(" 1.0"), zero_bucket[0]


def test_list_sources_outcomes_are_a_closed_enum(monkeypatch):
    """A backend outage must be distinguishable from a cluster with no peers."""
    collector = _fresh_collector(monkeypatch)
    assert collector._ensure() is True

    for result in LIST_SOURCES_RESULTS:
        collector.record_list_sources("random", result)
    # An unrecognized value is clamped rather than minting a new series.
    collector.record_list_sources("random", "totally-unexpected-" + "x" * 64)

    body = _exposition(collector)
    series = [
        line
        for line in body.splitlines()
        if line.startswith("mx_p2p_list_sources_total{")
    ]
    assert len(series) == len(LIST_SOURCES_RESULTS), body
    assert any('result="error"' in line and line.endswith(" 2.0") for line in series)


# ---------------------------------------------------------------------------
# D2: losing the bind is the normal case, and ownership can migrate
# ---------------------------------------------------------------------------


def test_losing_the_metrics_bind_is_not_fatal(monkeypatch, tmp_path, squatted_port):
    """N-1 ranks lose the bind by design; the winner serves them all.

    The bind loss must not disable the collector: this rank's counters still
    reach the endpoint, through the winner's MultiProcessCollector.
    """
    collector = _fresh_collector(
        monkeypatch,
        MX_METRICS_PORT=str(squatted_port),
        PROMETHEUS_MULTIPROC_DIR=str(tmp_path),
        MX_METRICS_BIND_RETRY_SECS="3600",
    )
    assert collector._ensure() is True

    assert collector._ready is True, "a lost bind must not disable recording"
    assert collector._bind_owner is False
    # A retry thread is armed so the endpoint can migrate when the current owner
    # exits; without it a pod whose bind winner exits first runs on with no
    # endpoint at all while staying perfectly healthy.
    assert collector._retry_thread is not None
    assert collector._retry_thread.daemon is True


def test_bind_failure_without_multiproc_is_warned_not_silent(
    monkeypatch, caplog, squatted_port
):
    """Without a shared directory a lost bind really does lose that rank's data."""
    collector = _fresh_collector(
        monkeypatch,
        MX_METRICS_PORT=str(squatted_port),
        PROMETHEUS_MULTIPROC_DIR=None,
    )
    with caplog.at_level("WARNING", logger="modelexpress.metrics"):
        assert collector._ensure() is True
    assert any(
        "PROMETHEUS_MULTIPROC_DIR" in record.message for record in caplog.records
    ), caplog.text


def test_invalid_metrics_port_does_not_raise(monkeypatch):
    collector = _fresh_collector(monkeypatch, MX_METRICS_PORT="not-a-port")
    assert collector._ensure() is True
    assert collector._bind_owner is False


@pytest.mark.parametrize("port", ["0", "-1", "not-a-port", "   "])
def test_an_unusable_port_does_not_suppress_the_push(monkeypatch, port):
    """"No endpoint" must not read as "a scrape endpoint is configured".

    ``0`` is this module's own documented disable value and every one of these is
    a truthy string. Gating the push/scrape exclusion on the raw environment
    string rather than the parsed port made a batch pod that turned the endpoint
    off with ``0`` get neither: the exclusion dropped the push, and the bind was
    then skipped because the port did not parse.
    """
    collector = _fresh_collector(
        monkeypatch,
        MX_METRICS_PORT=port,
        MX_METRICS_PUSHGATEWAY="http://pushgateway:9091",
    )
    assert collector._ensure() is True

    assert collector._bind_owner is False, "an unusable port must not bind"
    assert collector._push_registered is True, (
        "the push is the only exposition left; it must not be disabled by a port "
        "that does not exist"
    )


@pytest.mark.parametrize("port", ["0", "-1", "not-a-port"])
def test_push_still_runs_when_the_port_is_unusable(monkeypatch, port):
    """The same rule on the manual-push path."""
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    monkeypatch.setenv("MX_METRICS_PUSHGATEWAY", "http://pushgateway:9091")
    monkeypatch.setenv("MX_METRICS_PORT", port)

    pushes = []
    import prometheus_client

    monkeypatch.setattr(
        prometheus_client, "push_to_gateway", lambda *a, **kw: pushes.append(kw)
    )

    push_metrics_if_enabled()
    assert len(pushes) == 1, f"MX_METRICS_PORT={port!r} wrongly suppressed the push"


def test_an_unbindable_port_is_reported_as_an_error_not_a_lost_race(
    monkeypatch, tmp_path, caplog
):
    """EACCES is not EADDRINUSE, and must not be reported as one.

    A port the container may not bind fails identically on every rank, so no rank
    ever serves and no retry can succeed. Branching on the presence of a
    multiprocess directory alone reported it with the reassuring "another rank in
    this pod owns the endpoint" line and armed a retry loop that logged nothing —
    the pod is permanently unscrapeable and every rank says it is fine.
    """
    collector = _fresh_collector(
        monkeypatch,
        MX_METRICS_PORT="443",
        PROMETHEUS_MULTIPROC_DIR=str(tmp_path),
    )

    def denied(*args, **kwargs):
        raise PermissionError(errno.EACCES, "Permission denied")

    import prometheus_client

    monkeypatch.setattr(prometheus_client, "start_http_server", denied)

    with caplog.at_level("DEBUG", logger="modelexpress.metrics"):
        assert collector._ensure() is True

    assert collector._bind_owner is False
    assert collector._retry_thread is None, "a retry cannot fix EACCES"
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert errors, caplog.text
    # Not errors[0]: a multiprocess dir under pytest also (correctly) logs that
    # multiprocess mode is inactive, since the value class latched
    # single-process at import.
    assert any("not a lost race" in r.message for r in errors), caplog.text
    assert not any(
        "held by another rank" in r.message for r in caplog.records
    ), caplog.text


def test_exposition_starts_from_the_merged_registry(monkeypatch, tmp_path):
    """The call site, not just the helper.

    ``_exposition_registry`` returning a merged registry is worth nothing if
    ``_try_bind`` hands ``start_http_server`` the local one instead — that is D2
    exactly, and it is invisible to a test that only exercises the helper.
    """
    collector = _fresh_collector(
        monkeypatch,
        MX_METRICS_PORT="19999",
        PROMETHEUS_MULTIPROC_DIR=str(tmp_path),
    )

    captured = {}

    def fake_start(port, registry=None, **kwargs):
        captured["port"] = port
        captured["registry"] = registry
        return None

    import prometheus_client

    monkeypatch.setattr(prometheus_client, "start_http_server", fake_start)

    assert collector._ensure() is True

    assert captured["port"] == 19999
    served = captured["registry"]
    assert served is not collector._registry, "served the local registry, not the union"
    assert any(
        type(c).__name__ == "MultiProcessCollector" for c in served._collector_to_names
    ), served._collector_to_names


def test_a_recorder_before_enable_never_raises(monkeypatch, tmp_path):
    """The invariant, exercised on the path that actually initializes.

    ``test_recorders_never_raise_into_load_path`` pre-sets ``_ready``, so it
    never runs initialization. Here the first recorder call is what builds the
    families and starts exposition, and the thread spawn inside the bind-failure
    handler raises — the shape seen under thread or FD exhaustion on a large-TP
    worker.
    """
    collector = _fresh_collector(
        monkeypatch,
        MX_METRICS_PORT="19998",
        PROMETHEUS_MULTIPROC_DIR=str(tmp_path),
    )

    def in_use(*args, **kwargs):
        raise OSError(errno.EADDRINUSE, "Address already in use")

    def no_threads(self):
        raise RuntimeError("can't start new thread")

    import prometheus_client

    monkeypatch.setattr(prometheus_client, "start_http_server", in_use)
    monkeypatch.setattr(threading.Thread, "start", no_threads)

    # Must not propagate, and must still have recorded.
    collector.record_selection("random", "a1b2c3d4")

    assert collector._ready is True
    # Read the local registry, not _exposition(): with a multiprocess directory
    # set, exposition merges .db files, and pytest latched prometheus_client's
    # single-process value class at import so none exist. The point here is that
    # the recording survived the failed exposition, not how it is served.
    from prometheus_client import generate_latest

    body = generate_latest(collector._registry).decode()
    assert "mx_p2p_source_selections_total" in body


# ---------------------------------------------------------------------------
# D3 and D6: the push is pod-scoped, and scrape and push are exclusive
# ---------------------------------------------------------------------------


def test_push_grouping_key_is_the_pod_not_the_host(monkeypatch):
    """Every rank on a node shares a hostname, and the push is a PUT.

    Keying on the hostname therefore had each rank replace the whole group the
    previous one wrote: whichever rank exited last was the only survivor, and
    the result looked complete. Keying on the pod, with a merged registry, means
    concurrent pushes carry the same payload instead of erasing data.
    """
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setenv("MX_METRICS_PUSHGATEWAY", "http://pushgateway:9091")
    monkeypatch.delenv("MX_METRICS_PORT", raising=False)
    monkeypatch.setenv("POD_UID", "pod-uid-1234")

    captured = {}

    def fake_push(gateway, job, grouping_key, registry):
        captured.update(
            gateway=gateway, job=job, grouping_key=grouping_key, registry=registry
        )

    import prometheus_client

    monkeypatch.setattr(prometheus_client, "push_to_gateway", fake_push)

    push_metrics_if_enabled()

    assert captured["grouping_key"] == {"pod": "pod-uid-1234"}
    assert "instance" not in captured["grouping_key"]


def test_push_and_scrape_are_mutually_exclusive(monkeypatch):
    """Running both double-counts every series, so it is refused in code."""
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setenv("MX_METRICS_PUSHGATEWAY", "http://pushgateway:9091")
    monkeypatch.setenv("MX_METRICS_PORT", "9402")

    called = []

    def fake_push(*args, **kwargs):
        called.append(kwargs)

    import prometheus_client

    monkeypatch.setattr(prometheus_client, "push_to_gateway", fake_push)

    push_metrics_if_enabled()
    assert called == [], "the push must stand down when a scrape endpoint exists"


def test_push_is_not_armed_when_a_scrape_port_is_configured(monkeypatch, squatted_port):
    collector = _fresh_collector(
        monkeypatch,
        MX_METRICS_PORT=str(squatted_port),
        MX_METRICS_PUSHGATEWAY="http://pushgateway:9091",
    )
    assert collector._ensure() is True
    assert collector._push_registered is False


# ---------------------------------------------------------------------------
# Multiprocess directory handling
# ---------------------------------------------------------------------------


def test_enable_never_wipes_the_multiproc_directory(monkeypatch, tmp_path):
    """A late-starting rank must not unlink an early rank's mmapped files.

    Ranks start staggered. If rank 5's ``enable_metrics()`` wiped the directory, rank 0
    would keep writing to an unlinked inode and vanish from every subsequent
    scrape — permanently, silently, and only under the staggered start that a
    single-rank test would never reproduce.
    """
    existing = tmp_path / "counter_1234.db"
    existing.write_bytes(b"pretend this is rank 0's mmap")

    collector = _fresh_collector(
        monkeypatch, PROMETHEUS_MULTIPROC_DIR=str(tmp_path), MX_METRICS_PORT=None
    )
    monkeypatch.setattr("modelexpress.metrics.metrics", collector)
    assert enable_metrics() is True

    assert existing.exists(), "enable_metrics() must never unlink another rank's files"


def test_reset_multiproc_dir_clears_stale_files(tmp_path):
    """The entrypoint helper: an emptyDir survives container restarts."""
    stale = tmp_path / "counter_999.db"
    stale.write_bytes(b"from the previous container start")
    keep = tmp_path / "not-a-metrics-file.txt"
    keep.write_text("untouched")

    reset_multiproc_dir(str(tmp_path))

    assert not stale.exists()
    assert keep.exists()


def test_reset_multiproc_dir_is_a_noop_without_a_directory(monkeypatch):
    # The variable is deleted explicitly: reset_multiproc_dir(None) falls back to
    # the ambient PROMETHEUS_MULTIPROC_DIR, so on a developer machine that
    # exports it this "no-op" test would unlink every .db file in a directory
    # that, in the documented deployment shape, belongs to a live process.
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    reset_multiproc_dir(None)  # must not raise


def test_inactive_multiprocess_mode_is_reported(tmp_path, caplog):
    """The silent failure that has no other symptom.

    ``prometheus_client.values.get_value_class()`` latches at module import. If
    ``PROMETHEUS_MULTIPROC_DIR`` is assigned after the engine has imported
    prometheus_client, no ``.db`` files are written, the merged endpoint returns
    200 with nothing in it, and nothing raises.

    Detected from the empty directory rather than from
    ``values.ValueClass._multiprocess``: that attribute is private and the
    dependency has no upper version bound, so keying on it would make a rename
    upstream report healthy pods as broken.
    """
    from modelexpress.metrics import _warn_if_multiprocess_inactive

    with caplog.at_level("ERROR", logger="modelexpress.metrics"):
        _warn_if_multiprocess_inactive(str(tmp_path))

    assert any(
        "no .db files were written" in record.message for record in caplog.records
    ), caplog.text


def test_active_multiprocess_mode_is_silent(tmp_path, caplog):
    """A directory with files must not produce the warning.

    This is the false-alarm case the private-attribute probe could not avoid: a
    healthy pod whose prometheus_client renamed its internals.
    """
    from modelexpress.metrics import _warn_if_multiprocess_inactive

    (tmp_path / "counter_1234.db").write_bytes(b"pretend this is a rank's mmap")

    with caplog.at_level("ERROR", logger="modelexpress.metrics"):
        _warn_if_multiprocess_inactive(str(tmp_path))

    assert not caplog.records, caplog.text


# ---------------------------------------------------------------------------
# D2 + D6 end to end: one endpoint, every rank, including hard-killed ones
# ---------------------------------------------------------------------------

_MULTIRANK_SCRIPT = textwrap.dedent(
    """
    import os
    import signal
    import sys

    sys.path.insert(0, sys.argv[1])
    ranks = int(sys.argv[2])

    import modelexpress.metrics as mx

    # prometheus_client is imported lazily inside enable_metrics(), so the value class
    # latches per child, after the fork, with the directory already set.
    children = []
    for rank in range(ranks):
        pid = os.fork()
        if pid == 0:
            try:
                mx.enable_metrics()
                for _ in range(rank + 1):
                    mx.metrics.record_attempt("random", "success")
            except BaseException:
                os._exit(1)
            if rank == 0:
                os._exit(0)                      # hard exit: no atexit hook
            os.kill(os.getpid(), signal.SIGKILL)  # OOM-kill simulation
        children.append(pid)

    failures = 0
    for pid in children:
        _, status = os.waitpid(pid, 0)
        if os.WIFEXITED(status) and os.WEXITSTATUS(status) != 0:
            failures += 1
    print("failures", failures)
    """
)


@pytest.mark.parametrize("ranks", [2, 8])
def test_merged_endpoint_serves_every_rank_including_hard_killed(tmp_path, ranks):
    """The phase-1 exit criterion, at TP=2 and TP=8.

    One directory, N ranks, and a merged registry that reports the sum of all of
    them — including ranks that were SIGKILLed and so ran no exit hook. The
    previous implementation lost N-1 ranks on the pull path (one bind winner)
    and N-1 on the push path (the grouping key collided), and lost a hard-killed
    rank entirely on both.

    TP=8 is not decorative: D2 and D3 scale with ranks per pod, and 8 is the
    benchmarked configuration in ``docs/BENCHMARKS.md``.
    """
    multiproc_dir = tmp_path / "mx-metrics"
    multiproc_dir.mkdir()

    env = dict(os.environ)
    env["MX_METRICS_ENABLED"] = "1"
    env["PROMETHEUS_MULTIPROC_DIR"] = str(multiproc_dir)
    env.pop("MX_METRICS_PORT", None)
    env.pop("MX_METRICS_PUSHGATEWAY", None)

    result = subprocess.run(
        [sys.executable, "-c", _MULTIRANK_SCRIPT, str(_PACKAGE_ROOT), str(ranks)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "failures 0" in result.stdout, result.stdout + result.stderr

    db_files = sorted(p.name for p in multiproc_dir.glob("*.db"))
    assert db_files, "no rank wrote an mmap file"

    from prometheus_client import CollectorRegistry, generate_latest, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=str(multiproc_dir))
    body = generate_latest(registry).decode()

    attempts = [
        line
        for line in body.splitlines()
        if line.startswith("mx_p2p_source_attempts_total{")
    ]
    assert len(attempts) == 1, body
    total = float(attempts[0].rsplit(" ", 1)[1])
    expected = ranks * (ranks + 1) / 2  # rank r recorded r+1 attempts
    assert total == expected, (
        f"merged total {total} != {expected}; a rank's data was lost. "
        f"files={db_files}\n{body}"
    )

    # No pid label may leak into the exposition: that is how multiprocess mode
    # silently turns a bounded family into an unbounded one.
    assert "pid=" not in body, body

    # mx_build_info must merge to exactly 1, not to the rank count. This is what
    # pins the gauge's multiprocess_mode: `all`/`liveall` are caught by the pid
    # assertion above, but a summing mode (`livesum`, `sum`) is not — it would
    # report N here, multiply every group_left join against it, and, because
    # mark_process_dead never runs on a SIGKILL, wedge a dead rank's
    # contribution into the sum forever.
    build_info = [
        line for line in body.splitlines() if line.startswith("mx_build_info{")
    ]
    assert len(build_info) == 1, body
    assert build_info[0].endswith(" 1.0"), (
        f"mx_build_info merged to {build_info[0].rsplit(' ', 1)[1]} across {ranks} "
        f"ranks; a summing multiprocess_mode has been introduced"
    )


# ---------------------------------------------------------------------------
# NIXL data-plane families
# ---------------------------------------------------------------------------


def test_nixl_labels_are_closed_enums(monkeypatch):
    """An unrecognized value must clamp rather than mint a new series.

    The underlying `_data_plane_error` is formatted free text, so an
    unclassified value reaching the label would grow the domain with the wording
    of the message.

    Built through ``_fresh_collector`` like every other test here, and not on the
    process-global registry: ``_ensure()`` also runs ``_start_exposition()``, so
    a bare ``MetricsCollector()`` under an ambient ``MX_METRICS_PORT`` binds a
    real listener that outlives the pytest session, and a second collector on the
    global registry fails with "Duplicated timeseries ... mx_build_info" —
    an error that names the wrong metric and would land on whichever test ran
    second.
    """
    m = _fresh_collector(monkeypatch, MX_METRICS_PORT=None, MX_METRICS_PUSHGATEWAY=None)
    assert m._ensure() is True

    m.record_nixl_error("timeout")
    m.record_nixl_error("status_error")
    m.record_nixl_error("QP 3 wedged on device mlx5_2")
    m.record_nixl_receive("complete")
    m.record_nixl_receive("partial")
    m.record_nixl_receive("something new")

    kinds = _label_values(m.nixl_errors, "kind")
    assert kinds == {"timeout", "status_error"}, kinds
    m.record_nixl_receive("rejected")
    results = _label_values(m.nixl_receives, "result")
    assert results == {"complete", "partial", "empty", "rejected"}, results


def test_a_nixl_recording_counts_exactly_one_event(monkeypatch):
    """The label set is not the metric; the value is.

    Every query these two families exist for is a function of the sample value —
    ``rate(mx_nixl_data_plane_errors_total[5m])`` for the agent-health signal,
    ``partial / sum(mx_nixl_receive_total)`` for the manifest-drift ratio. A
    recorder that incremented by anything other than one per event would pass a
    label-domain assertion unremarked while inflating both.

    The clamped call is counted too, on the ``status_error`` series: an
    unclassified failure is still a failure, and dropping it would understate the
    rate exactly when the fabric is misbehaving in a way nobody has classified
    yet.
    """
    collector = _fresh_collector(
        monkeypatch, MX_METRICS_PORT=None, MX_METRICS_PUSHGATEWAY=None
    )
    assert collector._ensure() is True

    collector.record_nixl_error("timeout")
    collector.record_nixl_error("timeout")
    collector.record_nixl_error("QP 3 wedged on device mlx5_2")
    collector.record_nixl_receive("complete")
    collector.record_nixl_receive("complete")
    collector.record_nixl_receive("partial")

    assert _label_counts(collector.nixl_errors, "kind") == {
        "timeout": 2.0,
        "status_error": 1.0,
    }
    assert _label_counts(collector.nixl_receives, "result") == {
        "complete": 2.0,
        "partial": 1.0,
    }

    # The family names are the dashboard's API: assert them as exposed text
    # rather than through the attribute, which a rename would carry along.
    body = _exposition(collector)
    assert 'mx_nixl_data_plane_errors_total{kind="timeout"' in body, body
    assert 'mx_nixl_receive_total{result="partial"' in body, body


@pytest.mark.parametrize(
    "recorder,args,family,label",
    [
        ("record_nixl_error", ("timeout",), "nixl_errors", "kind"),
        ("record_nixl_receive", ("partial",), "nixl_receives", "result"),
    ],
)
def test_a_nixl_event_can_be_the_first_metrics_call_of_a_process(
    monkeypatch, recorder, args, family, label
):
    """D4 for the two NIXL recorders: they must route through ``_ensure()``.

    A process that never selects a P2P source can still hit the data plane, so a
    NIXL failure or a degraded receive may be the first metrics event of the run.
    A recorder that touched its family without ``_ensure()`` would raise
    ``AttributeError`` on a collector that had not initialized yet, and the
    blanket ``except`` in the recorder turns that into silence: no families, no
    endpoint, no error.

    One collector per recorder, because the first call to initialize hides the
    omission in the other: sharing a collector would let a recorder that skips
    ``_ensure()`` free-ride on its sibling.
    """
    collector = _fresh_collector(
        monkeypatch, MX_METRICS_PORT=None, MX_METRICS_PUSHGATEWAY=None
    )
    assert collector._ready is False, "the fixture must not pre-initialize"

    getattr(collector, recorder)(*args)

    assert collector._ready is True, "the recorder did not initialize the collector"
    assert _label_counts(getattr(collector, family), label) == {args[0]: 1.0}


def _label_values(collector, label):
    """Distinct values seen for one label of a collector."""
    values = set()
    for metric in collector.collect():
        for sample in metric.samples:
            if label in sample.labels:
                values.add(sample.labels[label])
    return values


def _label_counts(collector, label):
    """Counter value per distinct value of one label.

    Only ``_total`` samples: a Counter also emits a ``_created`` timestamp under
    the same labels, and summing that in would swamp the counts with epoch
    seconds.
    """
    counts = {}
    for metric in collector.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total") and label in sample.labels:
                counts[sample.labels[label]] = (
                    counts.get(sample.labels[label], 0.0) + sample.value
                )
    return counts


# ---------------------------------------------------------------------------
# Cross-check with the alerting rules the Helm chart ships
# ---------------------------------------------------------------------------

_REPO_ROOT = _PACKAGE_ROOT.parents[1]
_RUST_CROSS_CHECK = _REPO_ROOT / "workspace-tests" / "tests" / "helm_alert_rules.rs"


def _client_families_claimed_by_rust() -> list[str]:
    """The ``CLIENT_FAMILIES`` list from the Rust-side alert-rule check.

    Parsed rather than duplicated so the two lists cannot drift apart while both
    suites stay green.
    """
    source = _RUST_CROSS_CHECK.read_text()
    block = re.search(r"const CLIENT_FAMILIES: &\[&str\] = &\[(.*?)\];", source, re.S)
    assert block, f"CLIENT_FAMILIES not found in {_RUST_CROSS_CHECK}"
    return re.findall(r'"([^"]+)"', block.group(1))


def _exported_series_names(exposition: str) -> set[str]:
    """Exact series names from the exposition, one per sample line.

    Not a substring search over the whole text. A removed family whose name is a
    prefix of a surviving one -- or which still appears in a HELP line -- would
    match anywhere in the blob and the check would pass while the series was
    gone. These are the names a query would actually have to use.
    """
    names = set()
    for line in exposition.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        names.add(line.split("{", 1)[0].split(" ", 1)[0])
    return names


def test_alert_rule_client_families_exist(monkeypatch):
    """Every client family the alert rules rely on is really exported.

    ``helm_alert_rules.rs`` proves each ``mx_*`` name in the rules is either a
    server family or one of these; this closes the other half, because the Rust
    side cannot enumerate a Python registry and would otherwise accept anything
    written into that list.

    Without this pairing a client-side rename produces no failure anywhere: the
    Rust check keeps passing on the stale name because it is in the allowlist,
    and the alert quietly stops matching anything.
    """
    collector = _fresh_collector(monkeypatch)
    for name, args in _RECORDERS:
        getattr(collector, name)(*args)
    exposition = _exposition(collector)

    claimed = _client_families_claimed_by_rust()
    assert claimed, "parsed an empty CLIENT_FAMILIES; the regex no longer matches"

    exported = _exported_series_names(exposition)
    missing = [family for family in claimed if family not in exported]
    assert not missing, (
        f"the alert rules name client families that are not exported: {missing}\n"
        f"Exported: {sorted(exported)}"
    )


# ---------------------------------------------------------------------------
# L0 / L1 load timing
# ---------------------------------------------------------------------------


def _sum_for(exposition: str, family: str, **labels) -> float:
    """The ``_sum`` of one histogram child, or 0.0 if that child has no series."""
    import re

    want = {k: v for k, v in labels.items()}
    total = 0.0
    for line in exposition.splitlines():
        if not line.startswith(f"{family}_sum{{"):
            continue
        got = dict(re.findall(r'([a-zA-Z_]\w*)="([^"]*)"', line))
        if all(got.get(k) == v for k, v in want.items()):
            total += float(line.rsplit(" ", 1)[1])
    return total


def test_a_load_records_one_observation_with_its_outcome(monkeypatch):
    collector = _fresh_collector(monkeypatch)
    with collector.time_load("vllm", "test-model", "main"):
        pass

    exposition = _exposition(collector)
    assert (
        'mx_load_seconds_count{engine="vllm",model="test-model",model_role="main",outcome="success",scheme=""} 1.0'
        in exposition
    ), exposition


def test_a_failed_load_is_recorded_and_the_exception_still_propagates(monkeypatch):
    """The load that raises is the one worth timing.

    Recording only successes would drop exactly the observations someone goes
    looking for, and swallowing the exception would turn a metrics concern into
    a correctness one.
    """
    collector = _fresh_collector(monkeypatch)
    with pytest.raises(RuntimeError, match="boom"):
        with collector.time_load("vllm", "test-model", "main"):
            raise RuntimeError("boom")

    exposition = _exposition(collector)
    assert 'outcome="error"' in exposition, exposition
    assert (
        'mx_load_seconds_count{engine="vllm",model="test-model",model_role="main",outcome="error",scheme=""} 1.0'
        in exposition
    ), exposition


def test_a_phase_that_raises_is_still_recorded(monkeypatch):
    """The elapsed time was really spent inside the load.

    Dropping it would make the phases stop summing to the whole on precisely the
    failed loads being investigated.
    """
    collector = _fresh_collector(monkeypatch)
    with pytest.raises(RuntimeError):
        with collector.time_load_phase("vllm", "test-model", "model_init"):
            raise RuntimeError("boom")

    exposition = _exposition(collector)
    assert (
        'mx_load_phase_seconds_count{engine="vllm",model="test-model",phase="model_init",scheme=""} 1.0'
        in exposition
    ), exposition


def test_phases_partition_the_load(monkeypatch):
    """The L1 invariant: sum(phases) <= L0, on real timings.

    This is the property that makes "which part was slow" answerable without the
    numbers contradicting each other. It holds by construction here because the
    phases are disjoint intervals inside the L0 span -- the test exists so that
    a future phase recorded from a second site fails rather than quietly
    double-counting.
    """
    collector = _fresh_collector(monkeypatch)
    with collector.time_load("vllm", "test-model", "main"):
        for phase in ("artifact_install", "model_init", "chain", "publish"):
            with collector.time_load_phase("vllm", "test-model", phase):
                time.sleep(0.002)

    exposition = _exposition(collector)
    total = _sum_for(exposition, "mx_load_seconds", engine="vllm")
    phases = _sum_for(exposition, "mx_load_phase_seconds", engine="vllm")

    assert phases > 0, exposition
    assert phases <= total, (
        f"phases summed to {phases:.6f}s but the load took {total:.6f}s; "
        "a phase is being recorded outside the load span or from two sites"
    )


def test_load_labels_are_closed_enums(monkeypatch):
    """An out-of-tree engine clamps to `other` rather than opening the domain."""
    collector = _fresh_collector(monkeypatch)
    collector.observe_load_seconds("some-fork", "m", "speculative", "weird", 1.0)

    exposition = _exposition(collector)
    assert 'engine="other"' in exposition, exposition
    assert 'model_role="other"' in exposition, exposition
    assert 'outcome="error"' in exposition, exposition
    assert "some-fork" not in exposition, exposition


def test_an_unknown_phase_is_dropped_not_clamped(monkeypatch):
    """Folding a stray phase into a real one inflates that phase's total.

    The sum would still look sound, which is worse than the observation going
    missing: a wrong number that reads as a right one.
    """
    collector = _fresh_collector(monkeypatch)
    collector.observe_load_phase_seconds("vllm", "m", "not_a_phase", 5.0)

    exposition = _exposition(collector)
    assert "not_a_phase" not in exposition, exposition
    assert "mx_load_phase_seconds_count" not in exposition, exposition


def test_the_model_label_is_clamped_by_length_not_by_enum(monkeypatch):
    """The one load label whose domain code cannot close.

    Engine and phase are closed enums; a model id is whatever the deployment
    serves. So the guard is a length cap, and the absent case gets a value
    rather than an empty label -- an empty string would render as
    ``model=""`` and read as a bug in the exporter.
    """
    from modelexpress.metrics import _MODEL_LABEL_MAX

    collector = _fresh_collector(monkeypatch)
    collector.observe_load_seconds("vllm", "x" * 500, "main", "success", 1.0)
    collector.observe_load_seconds("vllm", None, "main", "success", 1.0)
    collector.observe_load_seconds("vllm", "   ", "main", "success", 1.0)

    exposition = _exposition(collector)
    longest = max(
        (len(m) for m in re.findall(r'model="([^"]*)"', exposition)), default=0
    )
    assert longest <= _MODEL_LABEL_MAX, exposition
    assert 'model="unknown"' in exposition, exposition


def test_the_model_label_separates_two_models_in_one_process(monkeypatch):
    """Two models must not merge into one series.

    A pod serves one model, so this is not the production shape -- but the label
    is only worth adding if it actually partitions, and a test that records one
    model cannot tell a working label from a constant.
    """
    collector = _fresh_collector(monkeypatch)
    collector.observe_load_seconds("vllm", "org/small", "main", "success", 2.0)
    collector.observe_load_seconds("vllm", "org/large", "main", "success", 90.0)

    exposition = _exposition(collector)
    assert 'model="org/small"' in exposition, exposition
    assert 'model="org/large"' in exposition, exposition
    counts = re.findall(r"mx_load_seconds_count\{[^}]*\} (\d+\.\d+)", exposition)
    assert counts == ["1.0", "1.0"], exposition


def test_load_timers_never_raise_into_the_load_path(monkeypatch):
    """Same guarantee the other recorders carry, on the new entry points."""
    collector = _fresh_collector(monkeypatch)
    collector._ready = True
    collector.load_seconds = None  # force an AttributeError inside the recorder
    collector.load_phase_seconds = None

    with collector.time_load("vllm", "test-model", "main"):
        with collector.time_load_phase("vllm", "test-model", "chain"):
            pass

    collector.observe_load_seconds("vllm", "m", "main", "success", 1.0)
    collector.observe_load_phase_seconds("vllm", "m", "chain", 1.0)


def test_load_buckets_match_the_rust_xslow_band():
    """A server download and a client load are compared on one dashboard.

    Quantiles from differently-bucketed histograms are not comparable, so the
    two bands are pinned to each other here rather than trusted to stay in step.
    """
    rust = (
        Path(__file__).resolve().parents[3]
        / "modelexpress_server"
        / "src"
        / "metrics"
        / "buckets.rs"
    )
    source = rust.read_text(encoding="utf-8")
    match = re.search(r"pub const XSLOW: \[f64; \d+\] = \[(.*?)\];", source, re.S)
    assert match, "could not find XSLOW in buckets.rs"
    rust_buckets = tuple(
        float(v.strip()) for v in match.group(1).split(",") if v.strip()
    )
    assert rust_buckets == tuple(float(b) for b in _XSLOW_BUCKETS), (
        f"Rust XSLOW {rust_buckets} != Python _XSLOW_BUCKETS {_XSLOW_BUCKETS}"
    )
