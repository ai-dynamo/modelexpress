# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for readiness-URL resolution across multi-node topologies.

Regression coverage for NVBug 6518216: a pod-local loopback readiness URL is
unsatisfiable on the headless (non-head) nodes of a multi-node engine, and the
previous StatefulSet-shaped fallback derived an unresolvable hostname under
LeaderWorkerSet, so removing the env var did not help either.
"""

import pytest

from modelexpress.metadata import artifact_lifecycle as al

VLLM_DEFAULT = "http://127.0.0.1:8000/health"
NON_DEFAULT_HEALTH = "http://127.0.0.1:9090/health"

# Real LeaderWorkerSet naming: leader pod <lws>-<group>, worker pods
# <lws>-<group>-<index>, both under the headless service <lws>. So a worker's
# leader is its own name minus the last segment, under a shorter subdomain.
LWS_LEADER = "mx-vllm-vllmworker-0-0.mx-vllm-vllmworker-0.hwoo"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("LWS_LEADER_ADDRESS", raising=False)


def test_headless_worker_probes_head_and_keeps_port_and_path():
    """The core fix: rewrite only the host, never the port or path.

    The old fallback hardcoded port 8000, so any deployment serving health on
    a different port probed the wrong one even once the host was right.
    """
    assert (
        al.resolve_health_url(NON_DEFAULT_HEALTH, VLLM_DEFAULT, LWS_LEADER)
        == f"http://{LWS_LEADER}:9090/health"
    )


def test_lws_worker_no_longer_gets_the_broken_statefulset_hostname():
    """Guards the exact regression: `<pod>-0.<pod>` is wrong under LWS."""
    resolved = al.resolve_health_url(NON_DEFAULT_HEALTH, VLLM_DEFAULT, LWS_LEADER)
    # What the previous rpartition("-") derivation produced from
    # HOSTNAME=mx-vllm-vllmworker-0-0-1, which does not resolve.
    assert "mx-vllm-vllmworker-0-0-0" not in resolved


def test_single_node_engine_keeps_loopback():
    """vLLM defaults master_addr to 127.0.0.1, and single-node launches pass
    no --master-addr, so nothing should be rewritten."""
    assert (
        al.resolve_health_url(NON_DEFAULT_HEALTH, VLLM_DEFAULT, "127.0.0.1")
        == NON_DEFAULT_HEALTH
    )


def test_loopback_engine_address_beats_lws_env(monkeypatch):
    """A single-node engine inside an LWS group must keep its local probe.

    LWS injects LWS_LEADER_ADDRESS into every pod of the group, so it is set
    even when the engine is single-node. Reaching past a loopback master_addr
    to that value would gate this pod's publication on a different pod's
    health.
    """
    monkeypatch.setenv("LWS_LEADER_ADDRESS", LWS_LEADER)
    assert (
        al.resolve_health_url(NON_DEFAULT_HEALTH, VLLM_DEFAULT, "127.0.0.1")
        == NON_DEFAULT_HEALTH
    )


def test_lws_env_used_when_engine_exposes_no_head(monkeypatch):
    """Covers SGLang, whose LoadContext carries no master address."""
    monkeypatch.setenv("LWS_LEADER_ADDRESS", LWS_LEADER)
    assert (
        al.resolve_health_url(NON_DEFAULT_HEALTH, VLLM_DEFAULT, None)
        == f"http://{LWS_LEADER}:9090/health"
    )


def test_engine_head_addr_wins_over_lws_env(monkeypatch):
    monkeypatch.setenv("LWS_LEADER_ADDRESS", "from-env")
    assert (
        al.resolve_health_url(NON_DEFAULT_HEALTH, VLLM_DEFAULT, "from-engine")
        == "http://from-engine:9090/health"
    )


def test_explicit_remote_host_is_never_rewritten():
    assert (
        al.resolve_health_url("http://shared-head:9090/health", VLLM_DEFAULT, LWS_LEADER)
        == "http://shared-head:9090/health"
    )


def test_ipv6_head_is_bracketed():
    assert (
        al.resolve_health_url(NON_DEFAULT_HEALTH, VLLM_DEFAULT, "fd00::1")
        == "http://[fd00::1]:9090/health"
    )


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.1.2.3", True),
        ("localhost", True),
        ("0.0.0.0", True),
        ("::1", True),
        ("", True),
        (None, True),
        ("mx-vllm-0.mx-vllm.ns", False),
        ("10.0.6.238", False),
    ],
)
def test_is_loopback(host, expected):
    assert al._is_loopback(host) is expected
