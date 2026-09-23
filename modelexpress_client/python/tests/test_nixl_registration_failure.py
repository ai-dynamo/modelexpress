# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A failed NIXL registration must not leave a half-initialized manager behind.

Regression cover for a restored replica that could never serve P2P. Observed on a
CRIU-restored vLLM worker: ``register_tensors`` hit ``ibv_reg_mr ... Bad address``
because one tensor still pointed into an unbacked ``cuMemAddressReserve`` range
(PROT_NONE) that wake-up had not re-materialized. The exception was swallowed --
correctly, P2P is best-effort -- but the agent created moments earlier stayed
alive, and with it the metadata listener socket on MX_METADATA_PORT + device_id.

That single leak produced both halves of the failure:

  * the re-registration that runs after restore died on "Address already in use",
    so one recoverable failure became permanent for the life of the process; and
  * ``publish_metadata`` treats a non-None manager as "ready to serve", so the
    worker advertised itself to the MX server anyway and peers selected a source
    that could not transfer a byte.

Run: pytest tests/test_nixl_registration_failure.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modelexpress.load_strategy import base


class FakeManager:
    """Stand-in for NixlTransferManager recording teardown."""

    def __init__(self, shutdown_raises: bool = False):
        self.tensor_descriptors: list[object] = []
        self.shutdown_calls = 0
        self.shutdown_raises = shutdown_raises

    def register_tensors(self, tensors):
        raise RuntimeError("NIXL_ERR_BACKEND")

    def register_arena(self, arena, tensors):
        raise RuntimeError("NIXL_ERR_BACKEND")

    def shutdown(self):
        self.shutdown_calls += 1
        if self.shutdown_raises:
            raise RuntimeError("agent already gone")


def _ctx():
    return SimpleNamespace(
        p2p_enabled=True,
        global_rank=0,
        worker_rank=0,
        device_id=0,
        tensors={"w": object()},
        nixl_manager=None,
        vmm_arena=None,
        adapter=MagicMock(),
        accelerator_backend=SimpleNamespace(name="cuda"),
        mx_client=MagicMock(),
        identity=MagicMock(),
        worker_id="w0",
        source_ready_fn=None,
    )


@pytest.fixture
def patched(monkeypatch):
    """Take the environment checks out of the picture; they are not under test."""
    monkeypatch.setattr(base, "is_nixl_available", lambda: True)
    monkeypatch.setattr(base, "_metadata_publication_configured", lambda ctx: True)


def test_failed_registration_tears_down_the_manager(patched):
    """The leaked agent is what wedges the port, so it must be shut down."""
    manager = FakeManager()
    ctx = _ctx()

    with patch.object(base, "_init_nixl_manager", return_value=manager):
        base.register_tensors(MagicMock(), ctx, reuse_discovered=True)

    assert manager.shutdown_calls == 1, "agent must be released, not leaked"
    assert ctx.nixl_manager is None


def test_failed_registration_does_not_publish(patched):
    """A source that cannot transfer must not be advertised to the MX server."""
    manager = FakeManager()
    ctx = _ctx()

    with patch.object(base, "_init_nixl_manager", return_value=manager):
        base.register_tensors(MagicMock(), ctx, reuse_discovered=True)

    with patch.object(base, "publish_metadata_and_ready") as publish:
        base.publish_metadata(ctx)

    publish.assert_not_called()


def test_retry_after_failure_can_build_a_fresh_manager(patched):
    """The point of the teardown: the next attempt is not poisoned by the first.

    Previously ctx.nixl_manager stayed set, so the retry either reused a dead
    manager or raced a still-bound port. Clearing it lets _init_nixl_manager run
    again and bind cleanly.
    """
    ctx = _ctx()
    with patch.object(base, "_init_nixl_manager", return_value=FakeManager()):
        base.register_tensors(MagicMock(), ctx, reuse_discovered=True)
    assert ctx.nixl_manager is None

    healthy = MagicMock()
    healthy.tensor_descriptors = []
    with patch.object(base, "_init_nixl_manager", return_value=healthy) as init:
        base.register_tensors(MagicMock(), ctx, reuse_discovered=True)

    init.assert_called_once()
    assert ctx.nixl_manager is healthy
    healthy.register_tensors.assert_called_once()


def test_shutdown_failure_still_clears_the_manager(patched):
    """Teardown is best-effort, but the reference must go regardless.

    If shutdown() itself fails the port may stay bound -- nothing we can do -- but
    keeping the manager would additionally advertise a dead source.
    """
    manager = FakeManager(shutdown_raises=True)
    ctx = _ctx()

    with patch.object(base, "_init_nixl_manager", return_value=manager):
        base.register_tensors(MagicMock(), ctx, reuse_discovered=True)

    assert ctx.nixl_manager is None
