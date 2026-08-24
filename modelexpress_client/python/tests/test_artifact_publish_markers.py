# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for restart-safe artifact publication markers."""

import json
from types import SimpleNamespace

from modelexpress import p2p_pb2
from modelexpress.metadata import artifact_lifecycle as al
from modelexpress.metadata.artifact_transfer import ArtifactCacheRoot


def _transfer(tmp_path):
    return SimpleNamespace(
        name="triton_cache",
        roots=(
            ArtifactCacheRoot(
                name="primary",
                source_root=tmp_path / "cache",
                target_root=tmp_path / "cache",
            ),
        ),
    )


def _identity():
    return p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        model_name="test-model",
    )


def test_publish_marker_skips_a_live_owner(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)
    monkeypatch.setattr(al.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(al, "_process_starttime", lambda pid: "100")
    ctx = SimpleNamespace(global_rank=0)

    marker_path = al.mark_publish_scheduled(ctx, _transfer(tmp_path), _identity())

    assert marker_path is not None
    assert al.mark_publish_scheduled(ctx, _transfer(tmp_path), _identity()) is None


def test_publish_marker_skips_a_live_owner_without_starttime(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)
    monkeypatch.setattr(al.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(al, "_process_starttime", lambda pid: None)
    ctx = SimpleNamespace(global_rank=0)

    marker_path = al.mark_publish_scheduled(ctx, _transfer(tmp_path), _identity())

    assert marker_path is not None
    assert json.loads(marker_path.read_text())["starttime"] is None
    assert al.mark_publish_scheduled(ctx, _transfer(tmp_path), _identity()) is None


def test_publish_marker_skips_an_owner_without_pid_permission(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)

    def denied_pid(pid, signal):
        raise PermissionError

    monkeypatch.setattr(al.os, "kill", denied_pid)
    transfer = _transfer(tmp_path)
    identity = _identity()
    marker_path = al.artifact_marker_path(transfer, identity, "publish-scheduled")
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(
        json.dumps({"version": 1, "pid": 1234, "starttime": "100", "worker_rank": 0})
    )

    assert al.mark_publish_scheduled(SimpleNamespace(global_rank=1), transfer, identity) is None
    assert json.loads(marker_path.read_text())["pid"] == 1234


def test_publish_marker_reclaims_a_dead_owner(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)

    def dead_pid(pid, signal):
        raise ProcessLookupError

    monkeypatch.setattr(al.os, "kill", dead_pid)
    monkeypatch.setattr(al, "_process_starttime", lambda pid: "200")
    transfer = _transfer(tmp_path)
    identity = _identity()
    marker_path = al.artifact_marker_path(transfer, identity, "publish-scheduled")
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(
        json.dumps({"version": 1, "pid": 1234, "starttime": "100", "worker_rank": 0})
    )

    assert al.mark_publish_scheduled(SimpleNamespace(global_rank=1), transfer, identity) == marker_path
    assert json.loads(marker_path.read_text()) == {
        "pid": 4242,
        "starttime": "200",
        "version": 1,
        "worker_rank": 1,
    }


def test_publish_marker_reclaims_a_reused_pid(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)
    monkeypatch.setattr(al.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(al, "_process_starttime", lambda pid: "new-starttime")
    transfer = _transfer(tmp_path)
    identity = _identity()
    marker_path = al.artifact_marker_path(transfer, identity, "publish-scheduled")
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(
        json.dumps(
            {"version": 1, "pid": 1234, "starttime": "old-starttime", "worker_rank": 0}
        )
    )

    assert al.mark_publish_scheduled(SimpleNamespace(global_rank=1), transfer, identity) == marker_path
    assert json.loads(marker_path.read_text())["starttime"] == "new-starttime"


def test_publish_marker_reclaims_legacy_rank_only_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)
    monkeypatch.setattr(al, "_process_starttime", lambda pid: "100")
    transfer = _transfer(tmp_path)
    identity = _identity()
    marker_path = al.artifact_marker_path(transfer, identity, "publish-scheduled")
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text("0")

    assert al.mark_publish_scheduled(SimpleNamespace(global_rank=1), transfer, identity) == marker_path
    assert json.loads(marker_path.read_text())["pid"] == 4242
    assert json.loads(marker_path.read_text())["starttime"] == "100"


def test_publish_marker_reclaims_a_boolean_pid(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)
    monkeypatch.setattr(al, "_process_starttime", lambda pid: "100")
    transfer = _transfer(tmp_path)
    identity = _identity()
    marker_path = al.artifact_marker_path(transfer, identity, "publish-scheduled")
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(
        json.dumps({"version": 1, "pid": True, "starttime": "100", "worker_rank": 0})
    )

    assert al.mark_publish_scheduled(SimpleNamespace(global_rank=1), transfer, identity) == marker_path
    assert json.loads(marker_path.read_text())["pid"] == 4242


def test_publish_marker_reclaims_invalid_version_or_worker_rank(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)
    monkeypatch.setattr(al, "_process_starttime", lambda pid: "100")
    transfer = _transfer(tmp_path)
    identity = _identity()
    marker_path = al.artifact_marker_path(transfer, identity, "publish-scheduled")
    marker_path.parent.mkdir(parents=True)

    for invalid_marker in (
        {"version": True, "pid": 1234, "starttime": "100", "worker_rank": 0},
        {"version": 1, "pid": 1234, "starttime": "100"},
        {"version": 1, "pid": 1234, "starttime": "100", "worker_rank": True},
        {"version": 1, "pid": 1234, "starttime": "100", "worker_rank": -1},
    ):
        marker_path.write_text(json.dumps(invalid_marker))

        assert (
            al.mark_publish_scheduled(SimpleNamespace(global_rank=1), transfer, identity)
            == marker_path
        )
        assert json.loads(marker_path.read_text()) == {
            "pid": 4242,
            "starttime": "100",
            "version": 1,
            "worker_rank": 1,
        }
