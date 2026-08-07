# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the source-side RDMA NIC utilization sampler."""

from __future__ import annotations

from modelexpress.nic_metrics import (
    NicUtilizationSampler,
    SourceLoadSampler,
    _parse_rate_bytes_per_sec,
    make_source_load_provider,
)


def test_parse_rate():
    assert _parse_rate_bytes_per_sec("400 Gb/sec (4X HDR)") == 400e9 / 8
    assert _parse_rate_bytes_per_sec("100 Gb/sec (4X EDR)") == 100e9 / 8
    assert _parse_rate_bytes_per_sec("garbage") is None


class _FakeCounters:
    """Injectable sysfs reader returning scripted xmit/rcv word counts."""

    def __init__(self, xmit_words, rcv_words):
        self.xmit = xmit_words
        self.rcv = rcv_words

    def __call__(self, path):
        if path.endswith("port_xmit_data"):
            return self.xmit
        if path.endswith("port_rcv_data"):
            return self.rcv
        return None


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _sampler(counters, clock, link_bps=400e9 / 8):
    return NicUtilizationSampler(
        "mlx5_0",
        _reader=counters,
        _clock=clock,
        _link_bytes_per_sec=link_bps,
    )


def test_first_sample_is_zero_baseline():
    s = _sampler(_FakeCounters(0, 0), _Clock())
    assert s.sample() == 0.0


def test_utilization_from_delta():
    # link = 400 Gb/s = 50 GB/s. Over 1s, tx grows by 25 GB (in 4-byte words)
    # -> 25/50 = 0.5 utilization.
    counters = _FakeCounters(0, 0)
    clock = _Clock()
    s = _sampler(counters, clock)
    assert s.sample() == 0.0  # baseline
    link_bps = 400e9 / 8  # bytes/s
    bytes_in_1s = link_bps * 0.5  # half the link
    counters.xmit = int(bytes_in_1s / 4)  # counter is in 4-byte words
    counters.rcv = 0
    clock.t = 1.0
    assert abs(s.sample() - 0.5) < 1e-6


def test_uses_busier_direction():
    counters = _FakeCounters(0, 0)
    clock = _Clock()
    s = _sampler(counters, clock)
    s.sample()
    link_bps = 400e9 / 8
    counters.xmit = int((link_bps * 0.2) / 4)  # 20% TX
    counters.rcv = int((link_bps * 0.7) / 4)  # 70% RX
    clock.t = 1.0
    assert abs(s.sample() - 0.7) < 1e-6  # max of the two directions


def test_clamped_to_one():
    counters = _FakeCounters(0, 0)
    clock = _Clock()
    s = _sampler(counters, clock)
    s.sample()
    counters.xmit = 10**18  # absurd
    clock.t = 1.0
    assert s.sample() == 1.0


def test_counter_reset_does_not_go_negative():
    counters = _FakeCounters(10**9, 10**9)
    clock = _Clock()
    s = _sampler(counters, clock)
    s.sample()
    counters.xmit = 0  # counter reset/wrap
    counters.rcv = 0
    clock.t = 1.0
    assert s.sample() == 0.0


def test_no_device_returns_zero():
    s = NicUtilizationSampler(None)
    assert s.sample() == 0.0


def test_unreadable_counters_return_zero():
    s = _sampler(lambda path: None, _Clock())
    assert s.sample() == 0.0


def test_missing_link_rate_returns_zero():
    s = NicUtilizationSampler(
        "mlx5_0", _reader=_FakeCounters(0, 0), _clock=_Clock(), _link_bytes_per_sec=None
    )
    # link rate could not be read (device sysfs absent), so always 0.
    assert s.sample() == 0.0


# ---------------------------------------------------------------------------
# SourceLoadSampler (affine + fallback) and the provider seam
# ---------------------------------------------------------------------------


def test_source_load_sampler_uses_affine_device():
    calls = []

    def fake_factory(dev):
        calls.append(dev)
        return type("S", (), {"sample": lambda self: 0.4})()

    s = SourceLoadSampler(
        0,
        _resolver=lambda did: "mlx5_0",
        _lister=lambda: ["mlx5_0", "mlx5_1", "mlx5_2"],
        _sampler_factory=fake_factory,
    )
    # Only the affine device is sampled, not every node NIC.
    assert calls == ["mlx5_0"]
    assert s.sample() == 0.4


def test_source_load_sampler_falls_back_to_busiest_node_nic():
    loads = {"mlx5_0": 0.1, "mlx5_1": 0.9, "mlx5_2": 0.3}

    def fake_factory(dev):
        return type("S", (), {"sample": lambda self, d=dev: loads[d]})()

    s = SourceLoadSampler(
        0,
        _resolver=lambda did: None,  # affine unresolvable
        _lister=lambda: list(loads),
        _sampler_factory=fake_factory,
    )
    # Falls back to max across all node NICs.
    assert abs(s.sample() - 0.9) < 1e-9


def test_source_load_sampler_no_nic_returns_zero():
    s = SourceLoadSampler(
        0,
        _resolver=lambda did: None,
        _lister=lambda: [],
        _sampler_factory=lambda d: None,
    )
    assert s.sample() == 0.0


def test_make_source_load_provider_returns_callable():
    provider = make_source_load_provider(0)
    val = provider()
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0
