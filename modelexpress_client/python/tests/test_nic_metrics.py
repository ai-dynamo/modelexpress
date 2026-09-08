# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the source-side RDMA NIC utilization sampler."""

from __future__ import annotations

import struct

import pytest

from modelexpress import nic_metrics
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


def test_first_sample_is_none_baseline():
    s = _sampler(_FakeCounters(0, 0), _Clock())
    assert s.sample() is None


def test_utilization_from_delta():
    # link = 400 Gb/s = 50 GB/s. Over 1s, tx grows by 25 GB (in 4-byte words)
    # -> 25/50 = 0.5 utilization.
    counters = _FakeCounters(0, 0)
    clock = _Clock()
    s = _sampler(counters, clock)
    assert s.sample() is None  # baseline
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


def test_no_device_returns_none():
    s = NicUtilizationSampler(None)
    assert s.sample() is None


def test_unreadable_counters_return_none():
    s = _sampler(lambda path: None, _Clock())
    assert s.sample() is None


def test_missing_link_rate_returns_none():
    s = NicUtilizationSampler(
        "mlx5_0", _reader=_FakeCounters(0, 0), _clock=_Clock(), _link_bytes_per_sec=None
    )
    # link rate could not be read (device sysfs absent), so always 0.
    assert s.sample() is None


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


def test_source_load_sampler_no_nic_returns_none():
    s = SourceLoadSampler(
        0,
        _resolver=lambda did: None,
        _lister=lambda: [],
        _sampler_factory=lambda d: None,
    )
    assert s.sample() is None


def test_make_source_load_provider_returns_callable():
    provider = make_source_load_provider(0)
    val = provider()
    assert val is None or isinstance(val, float)
    assert val is None or 0.0 <= val <= 1.0


class TestNetlinkVfFallback:
    """SR-IOV VF pods have no `counters/` sysfs; the reader falls back to netlink."""

    @staticmethod
    def _no_sysfs(_path):
        return None

    def test_falls_back_to_netlink_when_sysfs_counters_absent(self):
        counters = {"rdma_tx_bytes": 0, "rdma_rx_bytes": 0}

        def nl(_dev, _port):
            return dict(counters)

        clock = iter([0.0, 1.0])
        s = nic_metrics.NicUtilizationSampler(
            "mlx5_0",
            _reader=self._no_sysfs,
            _clock=lambda: next(clock),
            _link_bytes_per_sec=100.0,
            _nl_counters=nl,
        )
        assert s.sample() is None  # baseline
        counters["rdma_tx_bytes"] = 50  # 50 B over 1 s against 100 B/s = 0.5
        assert s.sample() == pytest.approx(0.5)

    def test_netlink_bytes_are_not_scaled_by_the_sysfs_word_size(self):
        """sysfs counters are in 4-octet words; netlink values are already bytes."""
        counters = {"rdma_tx_bytes": 0, "rdma_rx_bytes": 0}
        clock = iter([0.0, 1.0])
        s = nic_metrics.NicUtilizationSampler(
            "mlx5_0",
            _reader=self._no_sysfs,
            _clock=lambda: next(clock),
            _link_bytes_per_sec=1000.0,
            _nl_counters=lambda _d, _p: dict(counters),
        )
        s.sample()
        counters["rdma_rx_bytes"] = 1000
        # Scaling by 4 would overflow to a clamped 1.0 instead of exactly 1.0/1.
        assert s.sample() == pytest.approx(1.0)

    def test_default_hwcounter_set_without_byte_totals_yields_none(self):
        """The real VF case: netlink works, but byte counters are not enabled."""
        default_set = {"rx_write_requests": 12345, "rx_read_requests": 678, "out_of_buffer": 0}
        clock = iter([0.0, 1.0, 2.0])
        s = nic_metrics.NicUtilizationSampler(
            "mlx5_0",
            _reader=self._no_sysfs,
            _clock=lambda: next(clock),
            _link_bytes_per_sec=100.0,
            _nl_counters=lambda _d, _p: dict(default_set),
        )
        assert s.sample() is None
        assert s.sample() is None

    def test_netlink_failure_degrades_to_zero(self):
        def boom(_dev, _port):
            raise OSError("no netlink in this sandbox")

        s = nic_metrics.NicUtilizationSampler(
            "mlx5_0",
            _reader=self._no_sysfs,
            _link_bytes_per_sec=100.0,
            _nl_counters=boom,
        )
        with pytest.raises(OSError):
            s._read_bytes_netlink()

    def test_sysfs_still_wins_when_present(self):
        """On a PF host the sysfs path must not be displaced by the fallback."""
        def nl(_dev, _port):
            raise AssertionError("netlink must not be consulted when sysfs works")

        clock = iter([0.0, 1.0])
        s = nic_metrics.NicUtilizationSampler(
            "mlx5_0",
            _reader=lambda p: 25 if p.endswith("port_xmit_data") else 0,
            _clock=lambda: next(clock),
            _link_bytes_per_sec=100.0,
            _nl_counters=nl,
        )
        assert s.sample() is None
        assert s.sample() == 0.0  # counters static -> no traffic, no netlink call


class TestNetlinkWireEncoding:
    """The attribute codec, pinned against values read from <rdma/rdma_netlink.h>."""

    def test_message_types_match_uapi_header(self):
        assert nic_metrics._NL_MSG_STAT_GET == 5137
        assert nic_metrics._NL_MSG_DEV_GET == 5121
        assert nic_metrics._NETLINK_RDMA == 20

    def test_attr_roundtrip_with_padding(self):
        blob = nic_metrics._nl_attr(nic_metrics._NLA_DEV_INDEX, struct.pack("=I", 7))
        assert len(blob) % 4 == 0
        parsed = nic_metrics._nl_parse_attrs(blob)
        assert struct.unpack("=I", parsed[nic_metrics._NLA_DEV_INDEX][0][:4])[0] == 7

    def test_parse_attrs_masks_nested_flag(self):
        # NLA_F_NESTED (0x8000) must not leak into the type key.
        payload = b"\x00\x00\x00\x00"
        blob = struct.pack("=HH", 8, nic_metrics._NLA_HWCOUNTERS | 0x8000) + payload
        assert nic_metrics._NLA_HWCOUNTERS in nic_metrics._nl_parse_attrs(blob)

    def test_parse_attrs_stops_on_truncated_attribute(self):
        assert nic_metrics._nl_parse_attrs(struct.pack("=HH", 999, 1)) == {}


class TestNoReadingIsNotIdle:
    """None (no reading) and 0.0 (measured idle) are different answers."""

    def test_source_load_sampler_ignores_nics_without_a_reading(self):
        class _S:
            def __init__(self, v): self._v = v
            def sample(self): return self._v
        s = nic_metrics.SourceLoadSampler(
            0, _resolver=lambda _d: None, _lister=lambda: ["a", "b", "c"],
            _sampler_factory=lambda d: _S({"a": None, "b": 0.4, "c": None}[d]),
        )
        assert s.sample() == pytest.approx(0.4)

    def test_source_load_sampler_all_unknown_is_none(self):
        class _S:
            def sample(self): return None
        s = nic_metrics.SourceLoadSampler(
            0, _resolver=lambda _d: None, _lister=lambda: ["a", "b"],
            _sampler_factory=lambda d: _S(),
        )
        assert s.sample() is None

    def test_static_counters_are_a_measured_zero_not_unknown(self):
        clock = iter([0.0, 1.0])
        s = nic_metrics.NicUtilizationSampler(
            "mlx5_0", _reader=lambda p: 100, _clock=lambda: next(clock),
            _link_bytes_per_sec=1000.0,
        )
        assert s.sample() is None   # baseline: no interval yet
        assert s.sample() == 0.0    # zero delta over a real interval
