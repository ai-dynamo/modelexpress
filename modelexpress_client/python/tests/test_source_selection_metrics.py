# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the opt-in P2P source-selection metrics layer.

Locks the two load-bearing guarantees the module promises: every recording
call is a no-op when metrics are disabled, and nothing ever raises into the
load path. Uses fresh _Metrics() instances and avoids creating real
prometheus_client collectors, so the global registry is never polluted.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from modelexpress.source_selection_metrics import (
    ENV_ENABLED,
    _Metrics,
    push_metrics_if_enabled,
)

_RECORDERS = [
    ("record_selection", ("random", "w1")),
    ("record_attempt", ("random", "success")),
    ("record_metadata_failure", ("random",)),
    ("observe_candidates", ("random", "listed", 2)),
    ("observe_selection_seconds", ("random", 0.001)),
    ("observe_transfer_seconds", ("random", "success", 1.0)),
]


def test_recorders_are_noop_when_disabled(monkeypatch):
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    m = _Metrics()
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
    m = _Metrics()
    m._ready = True
    m._init_attempted = True
    boom = MagicMock()
    boom.labels.side_effect = RuntimeError("collector exploded")
    m.selections = boom
    m.attempts = boom
    m.metadata_failures = boom
    m.candidates = boom
    m.selection_seconds = boom
    m.transfer_seconds = boom
    # None of these may propagate the RuntimeError.
    for name, args in _RECORDERS:
        getattr(m, name)(*args)


def test_push_is_noop_when_disabled(monkeypatch):
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    # No exception, no network call.
    push_metrics_if_enabled()
