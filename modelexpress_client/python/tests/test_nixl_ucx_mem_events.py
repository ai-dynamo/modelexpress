# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the opt-in MX_UCX_DISABLE_MEM_EVENTS -> UCX_MEM_EVENTS wiring.

UCX_MEM_EVENTS is process-wide (registration-cache invalidation for every
UCP context, not just this agent's) and the assignment in
NixlTransferManager.initialize() is permanent, so these tests pin down the
opt-in-only, UCX-only, never-override-the-operator contract from the PR
review discussion rather than relying on default-on behavior.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from modelexpress import nixl_transfer
from modelexpress.nixl_transfer import NixlTransferManager


@pytest.fixture
def fake_nixl(monkeypatch):
    """Stand in for the real NIXL bindings so initialize() can run without hardware.

    initialize() writes UCX_MEM_EVENTS to os.environ directly (permanent,
    matching production behavior), which monkeypatch.setenv/delenv calls made
    before that write don't know to restore. Snapshot and restore it here so
    a test that triggers the write doesn't leak it to the next test.
    """
    monkeypatch.setattr(nixl_transfer, "NIXL_AVAILABLE", True)
    monkeypatch.setattr(nixl_transfer, "NixlAgent", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(nixl_transfer, "nixl_agent_config", None)
    had_value = "UCX_MEM_EVENTS" in os.environ
    prior_value = os.environ.get("UCX_MEM_EVENTS")
    yield
    if had_value:
        os.environ["UCX_MEM_EVENTS"] = prior_value
    else:
        os.environ.pop("UCX_MEM_EVENTS", None)


def _manager(monkeypatch, backend: str) -> NixlTransferManager:
    monkeypatch.setenv("MX_NIXL_BACKEND", backend)
    return NixlTransferManager(
        agent_name="test",
        device_id=0,
        accelerator_backend=MagicMock(),
    )


class TestUcxMemEventsOptIn:
    def test_off_by_default_on_ucx(self, monkeypatch, fake_nixl):
        monkeypatch.delenv("MX_UCX_DISABLE_MEM_EVENTS", raising=False)
        monkeypatch.delenv("UCX_MEM_EVENTS", raising=False)
        mgr = _manager(monkeypatch, "UCX")
        mgr.initialize()
        assert "UCX_MEM_EVENTS" not in os.environ

    def test_opt_in_sets_it_on_ucx(self, monkeypatch, fake_nixl):
        monkeypatch.setenv("MX_UCX_DISABLE_MEM_EVENTS", "1")
        monkeypatch.delenv("UCX_MEM_EVENTS", raising=False)
        mgr = _manager(monkeypatch, "UCX")
        mgr.initialize()

        assert os.environ["UCX_MEM_EVENTS"] == "n"

    def test_opt_in_is_noop_on_libfabric(self, monkeypatch, fake_nixl):
        monkeypatch.setenv("MX_UCX_DISABLE_MEM_EVENTS", "1")
        monkeypatch.delenv("UCX_MEM_EVENTS", raising=False)
        mgr = _manager(monkeypatch, "LIBFABRIC")
        mgr.initialize()
        assert "UCX_MEM_EVENTS" not in os.environ

    def test_never_overrides_operator_set_value(self, monkeypatch, fake_nixl):
        monkeypatch.setenv("MX_UCX_DISABLE_MEM_EVENTS", "1")
        monkeypatch.setenv("UCX_MEM_EVENTS", "y")
        mgr = _manager(monkeypatch, "UCX")
        mgr.initialize()

        assert os.environ["UCX_MEM_EVENTS"] == "y"
