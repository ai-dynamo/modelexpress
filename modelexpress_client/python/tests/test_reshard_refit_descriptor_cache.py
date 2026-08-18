# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Reusing the read descriptors across steps - CPU only, no NIXL.

A read descriptor is a (session, src_addr, dst_addr, nbytes) tuple derived from
the transfer plan and the registered buffer addresses. Neither changes between
steps, so building the lists once per plan rather than once per step removes work
that scales with the descriptor count: a real MoE refit issues hundreds of
thousands of them, and re-deriving that list in Python cost more than the local
re-slice it precedes.

Caching addresses is only safe while those addresses are still the plan's, so the
tests that matter are the ones that would catch a stale cache:

  * a second refit through the cache moves the same bytes as the first, and the
    same bytes an uncached refit moves;
  * rebuilding the plan drops the cache, so a refit never RDMAs into the previous
    plan's addresses;
  * switching the fused/phased arm rebuilds rather than reusing, because the
    phased arm does not build the exact descriptors at all;
  * the build is reported as its own stage, so it cannot silently return to being
    unattributed time.

Run: pytest tests/test_reshard_refit_descriptor_cache.py
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from modelexpress.refit.reshard import receiver as receiver_mod
from modelexpress.refit.reshard.transfer_plan import TransferPlan
from modelexpress.refit.reshard.types import CaptureResult
from tests.test_reshard_refit_fused_wire import _build, _RecordingTransport


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """These refits run on CPU, where the stage syncs have nothing to wait on."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)


def _refit(harness, step):
    """One refit, returning the metrics and a copy of every receive buffer."""
    metrics = harness.update_weights(step=step)
    buffers = {
        name: buffer.detach().clone()
        for name, buffer in harness._recv_buffers.items()
    }
    return metrics, buffers


def _zero(harness):
    for buffer in harness._recv_buffers.values():
        buffer.zero_()


def test_cached_and_uncached_refits_move_identical_bytes(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_CACHE_DESCRIPTORS", "0")
    uncached_harness, uncached_keepalive = _build(_RecordingTransport())
    _, uncached = _refit(uncached_harness, 1)

    monkeypatch.setenv("MX_RESHARD_CACHE_DESCRIPTORS", "1")
    cached_harness, cached_keepalive = _build(_RecordingTransport())
    _, cached = _refit(cached_harness, 1)

    assert set(cached) == set(uncached)
    for name in cached:
        assert torch.equal(cached[name], uncached[name]), name
    assert uncached_keepalive and cached_keepalive


def test_second_refit_through_the_cache_repeats_the_first(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_CACHE_DESCRIPTORS", "1")
    harness, keepalive = _build(_RecordingTransport())

    first_metrics, first = _refit(harness, 1)
    # Zeroed so a cache that quietly moved nothing the second time cannot pass by
    # leaving the first refit's bytes in place.
    _zero(harness)
    second_metrics, second = _refit(harness, 2)

    for name in first:
        assert torch.equal(second[name], first[name]), name
    assert second_metrics["bytes_received"] == first_metrics["bytes_received"]
    assert second_metrics["segments"] == first_metrics["segments"]
    assert keepalive


def test_rebuilding_the_plan_drops_the_cache(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_CACHE_DESCRIPTORS", "1")
    harness, keepalive = _build(_RecordingTransport())
    _refit(harness, 1)
    assert harness._cached_descriptors is not None

    new_plan = TransferPlan()
    monkeypatch.setattr(
        receiver_mod,
        "gather_sources",
        lambda *_args, **_kwargs: ({}, {}, {}, {}),
    )
    monkeypatch.setattr(receiver_mod, "plan_transfer", lambda *_args: new_plan)
    monkeypatch.setattr(
        receiver_mod, "handshake_endpoints_for_plan", lambda *_args: {}
    )
    monkeypatch.setattr(receiver_mod, "handshake_with_peers", lambda *_args: None)
    monkeypatch.setattr(
        receiver_mod, "NixlReshardTransport", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(receiver_mod, "classic_cuda_alloc", nullcontext)
    harness._num_trainer_sources = 0
    harness._mx_client = object()
    harness._model_name = "model"
    harness._global_rank = 0
    harness._manager = SimpleNamespace(register_tensors=lambda *_args: None)
    harness._capture = lambda _manifest: (CaptureResult(), {})
    harness._log_coverage = lambda *_args: None

    harness._prepare(timeout=1.0)

    assert harness._plan is new_plan
    assert harness._cached_descriptors is None
    assert keepalive


def test_switching_the_wire_arm_rebuilds_the_cache(monkeypatch):
    """A cache filled under one wire arm must not be served to the other.

    The phased arm hands the plan to execute_transfer instead of building exact
    descriptors, so its cache entry holds None for them. Serving that to the fused
    arm would read the exact segments not at all rather than into the wrong place,
    which is the quiet kind of wrong: fewer bytes, no error.
    """
    monkeypatch.setenv("MX_RESHARD_CACHE_DESCRIPTORS", "1")
    monkeypatch.setenv("MX_RESHARD_FUSED_WIRE", "0")
    harness, keepalive = _build(_RecordingTransport())
    phased_metrics, phased = _refit(harness, 1)
    assert harness._cached_descriptors[0] is False

    monkeypatch.setenv("MX_RESHARD_FUSED_WIRE", "1")
    _zero(harness)
    fused_metrics, fused = _refit(harness, 2)
    assert harness._cached_descriptors[0] is True

    for name in phased:
        assert torch.equal(fused[name], phased[name]), name
    assert fused_metrics["bytes_received"] == phased_metrics["bytes_received"]
    assert keepalive


def test_descriptor_build_is_reported_as_a_stage(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_CACHE_DESCRIPTORS", "1")
    harness, keepalive = _build(_RecordingTransport())
    metrics, _ = _refit(harness, 1)
    # Present and non-negative rather than above a threshold: this asserts the
    # stage is accounted for, and a timing floor would be a flake on shared CI.
    assert "descriptor_build_s" in metrics
    assert metrics["descriptor_build_s"] >= 0.0
    assert keepalive
