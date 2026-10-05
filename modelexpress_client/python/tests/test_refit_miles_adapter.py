# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

pytest.importorskip("miles")

from modelexpress_rl.train.frameworks.miles import modelexpress as mx


def test_rank_zero_failure_preserves_cause_after_broadcast(monkeypatch):
    monkeypatch.setattr(mx.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(mx, "get_gloo_group", lambda: None)
    broadcast = Mock()
    monkeypatch.setattr(mx.dist, "broadcast_object_list", broadcast)
    original_error = ValueError("receiver install failed")

    def install():
        raise original_error

    with pytest.raises(RuntimeError, match="receiver install failed") as caught:
        mx.UpdateWeightFromModelExpress._rank_zero_call(None, install)

    broadcast.assert_called_once_with(
        [None, "receiver install failed"], src=0, group=None
    )
    assert caught.value.__cause__ is original_error
    assert original_error.__traceback__ is not None


def test_peer_reports_broadcast_failure_without_running_action(monkeypatch):
    monkeypatch.setattr(mx.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(mx, "get_gloo_group", lambda: None)
    monkeypatch.setattr(
        mx.dist,
        "broadcast_object_list",
        lambda result, **kwargs: result.__setitem__(1, "receiver install failed"),
    )
    action = Mock()

    with pytest.raises(RuntimeError, match="receiver install failed") as caught:
        mx.UpdateWeightFromModelExpress._rank_zero_call(None, action)

    action.assert_not_called()
    assert caught.value.__cause__ is None
