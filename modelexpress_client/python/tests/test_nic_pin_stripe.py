# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MX_RDMA_NIC_PIN=stripe (multi-NIC compute-fabric striping).

Auto mode picks ONE NIC per rank (best PCIe affinity). Stripe mode hands
UCX every compute-rate NIC via a comma-separated ``UCX_NET_DEVICES``
value AND bumps ``UCX_MAX_RMA_RAILS`` so UCX actually uses the extra
rails. This lets a single receiver saturate multiple NICs simultaneously,
which is where the multi-NIC bandwidth win comes from on GB200 (4x
mlx5 per compute node).

We can't validate the bandwidth win in unit tests — that requires real
RDMA hardware — but we CAN pin down the string format + the
UCX_MAX_RMA_RAILS side-effect + the fallback path when no NICs are
visible.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from modelexpress.ucx_utils import _resolve_nic_pin


class TestStripeMode:
    def test_stripe_with_multiple_nics(self, monkeypatch):
        """The core happy path: 4 compute NICs -> comma-joined stripe
        list + UCX_MAX_RMA_RAILS auto-set to 4.
        """
        monkeypatch.setenv("MX_RDMA_NIC_PIN", "stripe")
        monkeypatch.delenv("UCX_MAX_RMA_RAILS", raising=False)
        monkeypatch.delenv("MX_RDMA_NIC_PIN_MIN_RATE_GBPS", raising=False)

        fake_nics = [
            ("mlx5_0", 0, 400.0, ["0000:00:00.0"]),
            ("mlx5_1", 0, 400.0, ["0000:01:00.0"]),
            ("mlx5_2", 1, 400.0, ["0000:02:00.0"]),
            ("mlx5_3", 1, 400.0, ["0000:03:00.0"]),
        ]
        with patch(
            "modelexpress.ucx_utils._list_compute_ib_nics",
            return_value=fake_nics,
        ):
            result = _resolve_nic_pin(device_id=0)

        assert result == "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1"
        import os
        assert os.environ.get("UCX_MAX_RMA_RAILS") == "4"

    def test_stripe_single_nic_still_works(self, monkeypatch):
        """1 NIC visible -> stripe list of 1 -> rails=1. Degrades to
        auto-mode's behavior without falling back to it.
        """
        monkeypatch.setenv("MX_RDMA_NIC_PIN", "stripe")
        monkeypatch.delenv("UCX_MAX_RMA_RAILS", raising=False)

        with patch(
            "modelexpress.ucx_utils._list_compute_ib_nics",
            return_value=[("mlx5_0", 0, 400.0, ["0000:00:00.0"])],
        ):
            result = _resolve_nic_pin(device_id=0)

        assert result == "mlx5_0:1"
        import os
        assert os.environ.get("UCX_MAX_RMA_RAILS") == "1"

    def test_stripe_no_nics_returns_none(self, monkeypatch):
        """Non-RDMA host (empty /sys/class/infiniband) -> stripe becomes
        a no-op. Callers treat None as ``leave UCX selection alone``.
        """
        monkeypatch.setenv("MX_RDMA_NIC_PIN", "stripe")

        with patch(
            "modelexpress.ucx_utils._list_compute_ib_nics",
            return_value=[],
        ):
            result = _resolve_nic_pin(device_id=0)

        assert result is None

    def test_stripe_does_not_clobber_preset_max_rma_rails(self, monkeypatch):
        """If the caller pre-set UCX_MAX_RMA_RAILS (e.g. to test caps
        or match a NIXL constraint), stripe mode leaves it alone.
        """
        monkeypatch.setenv("MX_RDMA_NIC_PIN", "stripe")
        monkeypatch.setenv("UCX_MAX_RMA_RAILS", "2")

        fake_nics = [
            ("mlx5_0", 0, 400.0, []),
            ("mlx5_1", 0, 400.0, []),
            ("mlx5_2", 1, 400.0, []),
            ("mlx5_3", 1, 400.0, []),
        ]
        with patch(
            "modelexpress.ucx_utils._list_compute_ib_nics",
            return_value=fake_nics,
        ):
            _resolve_nic_pin(device_id=0)

        import os
        assert os.environ.get("UCX_MAX_RMA_RAILS") == "2"

    def test_stripe_is_case_insensitive(self, monkeypatch):
        """MX_RDMA_NIC_PIN=STRIPE / Stripe / stripe all activate the
        stripe branch. Matches the (existing) case-insensitive
        handling of "off"/"true"/etc.
        """
        for form in ("stripe", "Stripe", "STRIPE"):
            monkeypatch.setenv("MX_RDMA_NIC_PIN", form)
            monkeypatch.delenv("UCX_MAX_RMA_RAILS", raising=False)
            with patch(
                "modelexpress.ucx_utils._list_compute_ib_nics",
                return_value=[("mlx5_0", 0, 400.0, [])],
            ):
                result = _resolve_nic_pin(device_id=0)
            assert result == "mlx5_0:1", f"form={form!r} did not stripe"

    def test_stripe_respects_min_rate_override(self, monkeypatch):
        """MX_RDMA_NIC_PIN_MIN_RATE_GBPS should get passed through to
        _list_compute_ib_nics as the rate filter. Explicit lower bound
        overrides the default max-rate auto-detect.
        """
        monkeypatch.setenv("MX_RDMA_NIC_PIN", "stripe")
        monkeypatch.setenv("MX_RDMA_NIC_PIN_MIN_RATE_GBPS", "200")
        monkeypatch.delenv("UCX_MAX_RMA_RAILS", raising=False)

        captured: dict[str, float | None] = {}
        def fake_list(min_rate_gbps=None):
            captured["min_rate"] = min_rate_gbps
            return [("mlx5_0", 0, 400.0, [])]

        with patch(
            "modelexpress.ucx_utils._list_compute_ib_nics",
            side_effect=fake_list,
        ):
            _resolve_nic_pin(device_id=0)

        assert captured["min_rate"] == pytest.approx(200.0)


class TestNonStripeStillWorks:
    """Regression: adding the stripe branch must not affect the other
    modes.
    """

    def test_off_still_returns_none(self, monkeypatch):
        monkeypatch.setenv("MX_RDMA_NIC_PIN", "off")
        assert _resolve_nic_pin(device_id=0) is None

    def test_unset_still_returns_none(self, monkeypatch):
        monkeypatch.delenv("MX_RDMA_NIC_PIN", raising=False)
        assert _resolve_nic_pin(device_id=0) is None

    def test_explicit_list_still_indexes_by_device(self, monkeypatch):
        monkeypatch.setenv("MX_RDMA_NIC_PIN", "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1")
        assert _resolve_nic_pin(device_id=2) == "mlx5_2:1"
