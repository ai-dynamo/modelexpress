# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the MX_RDMA_NIC_PIN=stripe multi-NIC mode.

Surfaced by ai-dynamo/modelexpress#449: single-NIC pinning leaves
75% of allocated NIC bandwidth idle on 4-NIC GB200 / 8-NIC GB300
pods, accounting for most of the MX-vs-NCCL gap in the
16-receiver Llama 3.1 8B refit benchmark.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from modelexpress import ucx_utils


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Make sure each test starts from a clean env."""
    for var in (
        "MX_RDMA_NIC_PIN",
        "MX_RDMA_NIC_PIN_MIN_RATE_GBPS",
        "UCX_NET_DEVICES",
        "UCX_MAX_RMA_RAILS",
    ):
        monkeypatch.delenv(var, raising=False)


def _fake_nics(*names_rates_paths):
    """Return a list shaped like _list_compute_ib_nics: (name, numa, rate, path)."""
    return [(name, 0, rate, path) for name, rate, path in names_rates_paths]


def test_stripe_lists_all_compute_nics(monkeypatch):
    """stripe mode returns comma-joined NIC list with :1 suffix on each."""
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "stripe")
    monkeypatch.setattr(
        ucx_utils,
        "_list_compute_ib_nics",
        lambda min_rate_gbps=None: _fake_nics(
            ("mlx5_0", 400.0, ["a", "b"]),
            ("mlx5_1", 400.0, ["a", "c"]),
            ("mlx5_2", 400.0, ["a", "d"]),
            ("mlx5_3", 400.0, ["a", "e"]),
        ),
    )
    ucx_utils.apply_nic_pin_for_device(0)
    assert os.environ["UCX_NET_DEVICES"] == "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1"


def test_stripe_bumps_max_rma_rails_to_nic_count(monkeypatch):
    """Multi-NIC stripe bumps UCX_MAX_RMA_RAILS to NIC count if not already set."""
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "stripe")
    monkeypatch.setattr(
        ucx_utils,
        "_list_compute_ib_nics",
        lambda min_rate_gbps=None: _fake_nics(
            ("mlx5_0", 400.0, []),
            ("mlx5_1", 400.0, []),
            ("mlx5_2", 400.0, []),
            ("mlx5_3", 400.0, []),
        ),
    )
    ucx_utils.apply_nic_pin_for_device(0)
    assert os.environ["UCX_MAX_RMA_RAILS"] == "4"


def test_stripe_respects_explicit_max_rma_rails(monkeypatch):
    """If user already set UCX_MAX_RMA_RAILS, don't overwrite."""
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "stripe")
    monkeypatch.setenv("UCX_MAX_RMA_RAILS", "2")
    monkeypatch.setattr(
        ucx_utils,
        "_list_compute_ib_nics",
        lambda min_rate_gbps=None: _fake_nics(
            ("mlx5_0", 400.0, []),
            ("mlx5_1", 400.0, []),
            ("mlx5_2", 400.0, []),
            ("mlx5_3", 400.0, []),
        ),
    )
    ucx_utils.apply_nic_pin_for_device(0)
    assert os.environ["UCX_MAX_RMA_RAILS"] == "2"


def test_stripe_single_nic_skips_rails_bump(monkeypatch):
    """1-NIC pod: stripe degenerates to single-NIC, don't set rails."""
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "stripe")
    monkeypatch.setattr(
        ucx_utils,
        "_list_compute_ib_nics",
        lambda min_rate_gbps=None: _fake_nics(("mlx5_0", 400.0, [])),
    )
    ucx_utils.apply_nic_pin_for_device(0)
    assert os.environ["UCX_NET_DEVICES"] == "mlx5_0:1"
    assert "UCX_MAX_RMA_RAILS" not in os.environ


def test_stripe_no_nics_visible(monkeypatch):
    """Topologically NIC-less host: stripe is a no-op, no env vars set."""
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "stripe")
    monkeypatch.setattr(
        ucx_utils, "_list_compute_ib_nics", lambda min_rate_gbps=None: []
    )
    ucx_utils.apply_nic_pin_for_device(0)
    assert "UCX_NET_DEVICES" not in os.environ
    assert "UCX_MAX_RMA_RAILS" not in os.environ


def test_stripe_alias_all(monkeypatch):
    """'all' is an accepted alias of 'stripe' for ergonomic ops use."""
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "all")
    monkeypatch.setattr(
        ucx_utils,
        "_list_compute_ib_nics",
        lambda min_rate_gbps=None: _fake_nics(
            ("mlx5_0", 400.0, []),
            ("mlx5_1", 400.0, []),
        ),
    )
    ucx_utils.apply_nic_pin_for_device(0)
    assert os.environ["UCX_NET_DEVICES"] == "mlx5_0:1,mlx5_1:1"
    assert os.environ["UCX_MAX_RMA_RAILS"] == "2"


def test_auto_mode_does_not_set_rails(monkeypatch):
    """Existing 'auto' mode preserves its single-NIC behavior — no rails bump."""
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "auto")
    monkeypatch.setattr(
        ucx_utils,
        "probe_nic_pin_for_device",
        lambda device_id, min_rate_gbps=None: "mlx5_0:1",
    )
    ucx_utils.apply_nic_pin_for_device(0)
    assert os.environ["UCX_NET_DEVICES"] == "mlx5_0:1"
    assert "UCX_MAX_RMA_RAILS" not in os.environ


def test_min_rate_filter_passed_to_stripe(monkeypatch):
    """MX_RDMA_NIC_PIN_MIN_RATE_GBPS is forwarded to the stripe lister."""
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "stripe")
    monkeypatch.setenv("MX_RDMA_NIC_PIN_MIN_RATE_GBPS", "200")
    captured = {}

    def fake_list(min_rate_gbps=None):
        captured["min_rate"] = min_rate_gbps
        return _fake_nics(("mlx5_0", 400.0, []))

    monkeypatch.setattr(ucx_utils, "_list_compute_ib_nics", fake_list)
    ucx_utils.apply_nic_pin_for_device(0)
    assert captured["min_rate"] == 200.0


def test_off_mode_unchanged(monkeypatch):
    """Sanity check: 'off' / unset still does nothing."""
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "off")
    ucx_utils.apply_nic_pin_for_device(0)
    assert "UCX_NET_DEVICES" not in os.environ
    assert "UCX_MAX_RMA_RAILS" not in os.environ
