# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Edge cases for the shared refit throughput floor."""

import json
import logging

import pytest

from modelexpress_rl.refit.reshard import throughput


@pytest.mark.parametrize(
    ("value", "warning"),
    [
        ("nan", "Non-finite MX_RESHARD_MIN_GBPS"),
        ("NaN", "Non-finite MX_RESHARD_MIN_GBPS"),
        ("inf", "Non-finite MX_RESHARD_MIN_GBPS"),
        ("-inf", "Non-finite MX_RESHARD_MIN_GBPS"),
        ("-50", "is negative"),
        ("-0.5", "is negative"),
        ("not-a-number", "Invalid MX_RESHARD_MIN_GBPS"),
    ],
)
def test_invalid_floor_disables_slow_transfer_warning(
    monkeypatch, caplog, value, warning
):
    monkeypatch.setenv("MX_RESHARD_MIN_GBPS", value)
    log = logging.getLogger(__name__)
    with caplog.at_level(logging.WARNING):
        assert throughput.min_gbps(log) == 0
        throughput.warn_if_below_floor(
            wire_bytes=1_000_000_000, wire_seconds=1, log=log
        )

    assert warning in caplog.text
    assert throughput.SLOW_THROUGHPUT_MARKER not in caplog.text


@pytest.mark.parametrize(("wire_seconds", "warns"), [(0.16, False), (0.32, True)])
def test_warning_requires_rate_below_floor(monkeypatch, caplog, wire_seconds, warns):
    monkeypatch.setenv("MX_RESHARD_MIN_GBPS", "50")
    with caplog.at_level(logging.WARNING):
        throughput.warn_if_below_floor(
            wire_bytes=1_000_000_000,
            wire_seconds=wire_seconds,
            log=logging.getLogger(__name__),
        )

    records = [m for m in caplog.messages if throughput.SLOW_THROUGHPUT_MARKER in m]
    assert bool(records) is warns
    if warns:
        payload = json.loads(records[0].split(" ", 1)[1])
        assert payload["implied_gbps"] == 25
        assert payload["shortfall_x"] == 2


@pytest.mark.parametrize(("wire_bytes", "wire_seconds"), [(16, 0), (0, 0.5)])
def test_empty_transfer_or_zero_duration_is_silent(
    monkeypatch, caplog, wire_bytes, wire_seconds
):
    monkeypatch.setenv("MX_RESHARD_MIN_GBPS", "50")
    with caplog.at_level(logging.WARNING):
        throughput.warn_if_below_floor(
            wire_bytes=wire_bytes,
            wire_seconds=wire_seconds,
            log=logging.getLogger(__name__),
        )

    assert not caplog.records
