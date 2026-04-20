# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for mx_peer_discovery.static."""

from mx_peer_discovery.static import (
    DEFAULT_ENV_VAR,
    endpoints_from_env,
    parse_endpoints,
)


# -- parse_endpoints: basic --------------------------------------------------


def test_parse_single_endpoint():
    assert parse_endpoints("10.0.0.1:4001") == [("10.0.0.1", 4001)]


def test_parse_multiple_endpoints():
    assert parse_endpoints("10.0.0.1:4001,10.0.0.2:4002,10.0.0.3:4003") == [
        ("10.0.0.1", 4001),
        ("10.0.0.2", 4002),
        ("10.0.0.3", 4003),
    ]


def test_parse_hostname_not_ip():
    assert parse_endpoints("worker-01.cluster.local:4001") == [
        ("worker-01.cluster.local", 4001),
    ]


def test_parse_preserves_order():
    assert parse_endpoints("c:1,a:2,b:3") == [("c", 1), ("a", 2), ("b", 3)]


# -- parse_endpoints: whitespace ---------------------------------------------


def test_parse_whitespace_around_entries():
    assert parse_endpoints("  10.0.0.1:4001  ,  10.0.0.2:4002  ") == [
        ("10.0.0.1", 4001),
        ("10.0.0.2", 4002),
    ]


def test_parse_empty_string():
    assert parse_endpoints("") == []


def test_parse_whitespace_only():
    assert parse_endpoints("   ") == []


def test_parse_trailing_comma():
    assert parse_endpoints("10.0.0.1:4001,") == [("10.0.0.1", 4001)]


def test_parse_double_comma():
    assert parse_endpoints("10.0.0.1:4001,,10.0.0.2:4002") == [
        ("10.0.0.1", 4001),
        ("10.0.0.2", 4002),
    ]


# -- parse_endpoints: IPv6 ---------------------------------------------------


def test_parse_ipv6_bracketed():
    assert parse_endpoints("[::1]:4001") == [("::1", 4001)]


def test_parse_ipv6_full_address():
    assert parse_endpoints("[fe80::1234:5678]:4001") == [
        ("fe80::1234:5678", 4001),
    ]


def test_parse_ipv6_mixed_with_ipv4():
    assert parse_endpoints("10.0.0.1:4001,[::1]:4002") == [
        ("10.0.0.1", 4001),
        ("::1", 4002),
    ]


# -- parse_endpoints: malformed entries skipped ------------------------------


def test_parse_missing_port_skipped():
    # First entry has no port, second is valid
    assert parse_endpoints("10.0.0.1,10.0.0.2:4002") == [("10.0.0.2", 4002)]


def test_parse_non_numeric_port_skipped():
    assert parse_endpoints("10.0.0.1:abc,10.0.0.2:4002") == [("10.0.0.2", 4002)]


def test_parse_port_zero_skipped():
    assert parse_endpoints("10.0.0.1:0,10.0.0.2:4002") == [("10.0.0.2", 4002)]


def test_parse_port_too_high_skipped():
    assert parse_endpoints("10.0.0.1:65536,10.0.0.2:4002") == [("10.0.0.2", 4002)]


def test_parse_negative_port_skipped():
    assert parse_endpoints("10.0.0.1:-1,10.0.0.2:4002") == [("10.0.0.2", 4002)]


def test_parse_ipv6_missing_close_bracket():
    assert parse_endpoints("[::1:4001,10.0.0.2:4002") == [("10.0.0.2", 4002)]


def test_parse_ipv6_missing_port():
    assert parse_endpoints("[::1],10.0.0.2:4002") == [("10.0.0.2", 4002)]


def test_parse_missing_host():
    # `:4001` has no host
    assert parse_endpoints(":4001,10.0.0.2:4002") == [("10.0.0.2", 4002)]


def test_parse_all_malformed_returns_empty():
    assert parse_endpoints("nope,also-nope,:4001") == []


def test_parse_port_boundary_values():
    assert parse_endpoints("10.0.0.1:1,10.0.0.2:65535") == [
        ("10.0.0.1", 1),
        ("10.0.0.2", 65535),
    ]


# -- endpoints_from_env ------------------------------------------------------


def test_env_reads_default_var(monkeypatch):
    monkeypatch.setenv(DEFAULT_ENV_VAR, "10.0.0.1:4001,10.0.0.2:4002")
    assert endpoints_from_env() == [
        ("10.0.0.1", 4001),
        ("10.0.0.2", 4002),
    ]


def test_env_reads_custom_var(monkeypatch):
    monkeypatch.setenv("CUSTOM_PEERS", "host1:1234")
    assert endpoints_from_env("CUSTOM_PEERS") == [("host1", 1234)]


def test_env_unset_returns_empty(monkeypatch):
    monkeypatch.delenv(DEFAULT_ENV_VAR, raising=False)
    assert endpoints_from_env() == []


def test_env_empty_returns_empty(monkeypatch):
    monkeypatch.setenv(DEFAULT_ENV_VAR, "")
    assert endpoints_from_env() == []


def test_default_env_var_name():
    assert DEFAULT_ENV_VAR == "MX_PEER_ENDPOINTS"
