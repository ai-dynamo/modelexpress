# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``modelexpress.lifecycle`` (``pause_serving`` / ``resume_serving``).

These cover the orchestration contract — ordering, ctx mutation, and the
swallow-failures semantics — without exercising the underlying NIXL,
MxClient, or metadata-server code paths (which have their own test files).
"""

from unittest.mock import MagicMock, patch

import torch

from modelexpress import p2p_pb2
from modelexpress.lifecycle import pause_serving, resume_serving


def _make_load_context(**overrides):
    """Return a real LoadContext with mocked dependencies (matches the
    convention used by ``test_vllm_loader.py`` / ``test_model_streamer_strategy.py``).

    Defaults wire ``mx_client`` and ``nixl_manager`` to fresh ``MagicMock``
    instances so the common happy-path tests don't need to override them.
    """
    from modelexpress.load_strategy import LoadContext

    defaults = dict(
        model_config=MagicMock(),
        load_config=MagicMock(),
        target_device=torch.device("cpu"),
        global_rank=0,
        worker_rank=0,
        device_id=0,
        identity=p2p_pb2.SourceIdentity(model_name="test-model"),
        mx_client=MagicMock(),
        worker_id="pre-pause-id",
        adapter=MagicMock(),
        nixl_manager=MagicMock(),
        tensors={},
    )
    defaults.update(overrides)
    return LoadContext(**defaults)


# ---------------------------------------------------------------------------
# pause_serving
# ---------------------------------------------------------------------------


def test_pause_full_teardown():
    """Happy path: unpublish, close client, shutdown nixl, null both fields,
    retain tensors."""
    fake_tensor = MagicMock()
    ctx = _make_load_context(tensors={"layer.weight": fake_tensor})
    client = ctx.mx_client
    nixl = ctx.nixl_manager

    with patch("modelexpress.lifecycle.unpublish_metadata") as mock_unpub:
        pause_serving(ctx)

    mock_unpub.assert_called_once_with(ctx)
    client.close.assert_called_once()
    nixl.shutdown.assert_called_once()
    assert ctx.mx_client is None
    assert ctx.nixl_manager is None
    assert ctx.tensors == {"layer.weight": fake_tensor}


def test_pause_handles_already_none_fields():
    """No AttributeError when mx_client / nixl_manager are already None."""
    ctx = _make_load_context(mx_client=None, nixl_manager=None)
    with patch("modelexpress.lifecycle.unpublish_metadata"):
        pause_serving(ctx)  # no raise
    assert ctx.mx_client is None
    assert ctx.nixl_manager is None


def test_pause_swallows_failures_and_completes_both_steps():
    """A client-close failure must not skip nixl shutdown; both still nulled."""
    ctx = _make_load_context()
    client = ctx.mx_client
    nixl = ctx.nixl_manager
    client.close.side_effect = RuntimeError("client boom")
    nixl.shutdown.side_effect = RuntimeError("nixl boom")

    with patch("modelexpress.lifecycle.unpublish_metadata"):
        pause_serving(ctx)  # must not raise

    client.close.assert_called_once()
    nixl.shutdown.assert_called_once()
    assert ctx.mx_client is None
    assert ctx.nixl_manager is None


# ---------------------------------------------------------------------------
# resume_serving
# ---------------------------------------------------------------------------


def test_resume_full_rebuild():
    """Happy path: fresh client, fresh worker_id, register with reuse_discovered,
    publish."""
    ctx = _make_load_context(worker_rank=7, worker_id="pre-pause-id")
    new_client = MagicMock()
    model = MagicMock()

    with patch(
        "modelexpress.lifecycle.create_metadata_client", return_value=new_client,
    ) as mock_create, patch(
        "modelexpress.lifecycle.register_tensors",
    ) as mock_register, patch(
        "modelexpress.lifecycle.publish_metadata",
    ) as mock_publish:
        resume_serving(ctx, model)

    mock_create.assert_called_once_with(worker_rank=7)
    assert ctx.mx_client is new_client
    assert ctx.worker_id != "pre-pause-id"
    assert len(ctx.worker_id) == 8
    assert all(c in "0123456789abcdef" for c in ctx.worker_id)
    mock_register.assert_called_once_with(model, ctx, reuse_discovered=True)
    mock_publish.assert_called_once_with(ctx)


def test_resume_honors_explicit_worker_id():
    """Caller-supplied worker_id wins over the default fresh-mint."""
    ctx = _make_load_context(worker_id="pre-pause-id")
    with patch("modelexpress.lifecycle.create_metadata_client"), patch(
        "modelexpress.lifecycle.register_tensors",
    ), patch("modelexpress.lifecycle.publish_metadata"):
        resume_serving(ctx, MagicMock(), worker_id="caller-supplied")
    assert ctx.worker_id == "caller-supplied"


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


def test_pause_resume_cycle_preserves_tensors_and_rotates_worker_id():
    """Full pause -> resume: tensors retained, mx_client rebuilt, worker_id changes."""
    fake_tensor = MagicMock()
    ctx = _make_load_context(
        tensors={"layer.weight": fake_tensor}, worker_id="pre-pause-id",
    )
    pre_pause_client = ctx.mx_client
    pre_pause_nixl = ctx.nixl_manager
    new_client = MagicMock()

    with patch("modelexpress.lifecycle.unpublish_metadata"), patch(
        "modelexpress.lifecycle.create_metadata_client", return_value=new_client,
    ), patch("modelexpress.lifecycle.register_tensors"), patch(
        "modelexpress.lifecycle.publish_metadata",
    ):
        pause_serving(ctx)
        assert ctx.mx_client is None
        assert ctx.nixl_manager is None
        assert ctx.tensors == {"layer.weight": fake_tensor}

        resume_serving(ctx, MagicMock())
        assert ctx.mx_client is new_client
        assert ctx.tensors == {"layer.weight": fake_tensor}
        assert ctx.worker_id != "pre-pause-id"

    pre_pause_client.close.assert_called_once()
    pre_pause_nixl.shutdown.assert_called_once()
