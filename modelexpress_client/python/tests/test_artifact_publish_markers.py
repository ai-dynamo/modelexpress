# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for restart-safe artifact publication leases."""

import multiprocessing
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from modelexpress import p2p_pb2
from modelexpress.metadata import artifact_lifecycle as al
from modelexpress.metadata.artifact_transfer import ArtifactCacheRoot


def _transfer(tmp_path):
    return SimpleNamespace(
        name="triton_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
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


@pytest.fixture(autouse=True)
def _clear_publish_leases():
    yield
    for lease in al._publish_leases.values():
        lease.close()
    al._publish_leases.clear()


def _schedule_publish(tmp_dir, rank, results):
    al.tempfile.gettempdir = lambda: tmp_dir
    al._artifact_transfer_enabled = lambda: True
    al._p2p_metadata_enabled_for_artifacts = lambda ctx, engine, log: True
    al._metadata_publication_configured = lambda ctx: True
    scheduled_publishers = {}
    al.schedule_artifact_publish(
        SimpleNamespace(
            global_rank=rank,
            worker_rank=rank,
            worker_id=f"worker-{rank}",
            device_id=rank,
            mx_client=object(),
            nixl_manager=object(),
        ),
        lambda: [(_transfer(Path(tmp_dir)), _identity())],
        engine_label="test",
        ready_fn_factory=lambda roots: lambda: True,
        artifact_publish_fn=lambda transfer, identity: SimpleNamespace(
            endpoint=SimpleNamespace(mx_source_id=f"source-{rank}")
        ),
        scheduled_publishers=scheduled_publishers,
    )
    if not scheduled_publishers:
        results.put("blocked")
        return
    publisher = next(iter(scheduled_publishers.values()))
    publisher._thread.join(timeout=5)
    results.put("published" if publisher.mx_source_id is not None else "failed")
    time.sleep(60)


def test_publish_lease_skips_a_live_owner(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    ctx = SimpleNamespace(global_rank=0)

    lease_path = al.mark_publish_scheduled(ctx, _transfer(tmp_path), _identity())

    assert lease_path is not None
    assert al.mark_publish_scheduled(ctx, _transfer(tmp_path), _identity()) is None


def test_unlocked_legacy_marker_does_not_block_publish(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    transfer = _transfer(tmp_path)
    identity = _identity()
    lease_path = al.artifact_marker_path(transfer, identity, "publish-scheduled")
    lease_path.parent.mkdir(parents=True)
    lease_path.write_text("0")

    assert (
        al.mark_publish_scheduled(SimpleNamespace(global_rank=0), transfer, identity)
        == lease_path
    )


def test_failed_publish_releases_lease(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    ctx = SimpleNamespace(global_rank=0)

    lease_path = al.mark_publish_scheduled(ctx, _transfer(tmp_path), _identity())
    al.clear_publish_scheduled(SimpleNamespace(mx_source_id=None), lease_path)

    assert (
        al.mark_publish_scheduled(ctx, _transfer(tmp_path), _identity()) == lease_path
    )


def test_successful_publish_retains_lease(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    ctx = SimpleNamespace(global_rank=0)

    lease_path = al.mark_publish_scheduled(ctx, _transfer(tmp_path), _identity())
    al.clear_publish_scheduled(SimpleNamespace(mx_source_id="source-id"), lease_path)

    assert al.mark_publish_scheduled(ctx, _transfer(tmp_path), _identity()) is None


def test_scheduled_publisher_lease_reclaimed_after_owner_terminates(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    context = multiprocessing.get_context("fork")
    results = context.Queue()
    owner = context.Process(
        target=_schedule_publish,
        args=(str(tmp_path), 0, results),
    )
    processes = [owner]
    owner.start()
    try:
        assert results.get(timeout=5) == "published"

        contender = context.Process(
            target=_schedule_publish,
            args=(str(tmp_path), 1, results),
        )
        processes.append(contender)
        contender.start()
        assert results.get(timeout=5) == "blocked"
        contender.join(timeout=5)
        assert not contender.is_alive()

        owner.terminate()
        owner.join(timeout=5)
        assert not owner.is_alive()

        replacement = context.Process(
            target=_schedule_publish,
            args=(str(tmp_path), 2, results),
        )
        processes.append(replacement)
        replacement.start()
        assert results.get(timeout=5) == "published"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
