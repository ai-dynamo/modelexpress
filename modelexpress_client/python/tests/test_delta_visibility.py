# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import threading
import time
from unittest import mock

import pytest

from modelexpress.rl import delta_visibility


@pytest.fixture(autouse=True)
def visibility_env(monkeypatch):
    monkeypatch.setenv("MX_S3_ENDPOINT", "http://mx-minio:9000")
    monkeypatch.setenv("MX_S3_BUCKET", "weight-delta")
    monkeypatch.setenv("MX_DELTA_STREAM_PREFIX", "test-run")


def _index(shard_name="model-00000-of-00001.safetensors"):
    return json.dumps(
        {
            "metadata": {"version": "000001", "base_version": "000000"},
            "weight_map": {"model.weight": shard_name},
        }
    )


def _configure(monkeypatch, download, expected_size=8):
    fake_s3 = mock.MagicMock()
    fake_s3.head_object.return_value = {"ContentLength": expected_size}
    monkeypatch.setattr(
        delta_visibility, "s3_client", mock.Mock(return_value=fake_s3)
    )
    monkeypatch.setattr(
        delta_visibility, "list_versions", mock.Mock(return_value=[1])
    )
    download_mock = mock.Mock(side_effect=download)
    monkeypatch.setattr(delta_visibility, "download_version_dir", download_mock)
    return fake_s3, download_mock


def _temp_dir(dest_parent, name="download"):
    path = os.path.join(dest_parent, name)
    os.mkdir(path)
    return path


def test_missing_index_raises(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"

    def download(s3, bucket, prefix, version, dest_parent):
        temp_dir = _temp_dir(dest_parent)
        with open(os.path.join(temp_dir, "model-00000-of-00001.safetensors"), "wb") as file:
            file.write(b"complete")
        return temp_dir

    _configure(monkeypatch, download)

    with pytest.raises(RuntimeError, match="missing index"):
        delta_visibility.ensure_visible(str(source_dir), 1)
    assert not (source_dir / "weight_v000001").exists()


def test_weight_map_missing_shard_raises(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"

    def download(s3, bucket, prefix, version, dest_parent):
        temp_dir = _temp_dir(dest_parent)
        with open(os.path.join(temp_dir, "model.safetensors.index.json"), "w") as file:
            file.write(_index("absent.safetensors"))
        return temp_dir

    _configure(monkeypatch, download)

    with pytest.raises(RuntimeError, match="missing shard absent.safetensors"):
        delta_visibility.ensure_visible(str(source_dir), 1)
    assert not (source_dir / "weight_v000001").exists()


def test_truncated_shard_raises(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"

    def download(s3, bucket, prefix, version, dest_parent):
        temp_dir = _temp_dir(dest_parent)
        with open(os.path.join(temp_dir, "model.safetensors.index.json"), "w") as file:
            file.write(_index())
        with open(os.path.join(temp_dir, "model-00000-of-00001.safetensors"), "wb") as file:
            file.write(b"short")
        return temp_dir

    _configure(monkeypatch, download, expected_size=8)

    with pytest.raises(RuntimeError, match="has size 5, expected 8"):
        delta_visibility.ensure_visible(str(source_dir), 1)
    assert not (source_dir / "weight_v000001").exists()


def test_complete_dir_is_installed_and_skipped_on_reentry(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"

    def download(s3, bucket, prefix, version, dest_parent):
        temp_dir = _temp_dir(dest_parent)
        with open(os.path.join(temp_dir, "model.safetensors.index.json"), "w") as file:
            file.write(_index())
        with open(os.path.join(temp_dir, "model-00000-of-00001.safetensors"), "wb") as file:
            file.write(b"complete")
        return temp_dir

    _, download_mock = _configure(monkeypatch, download)

    delta_visibility.ensure_visible(str(source_dir), 1)
    delta_visibility.ensure_visible(str(source_dir), 1)

    assert download_mock.call_count == 1
    assert (
        source_dir / "weight_v000001" / "model-00000-of-00001.safetensors"
    ).read_bytes() == b"complete"


def test_get_path_logs_cache_miss_then_hit(tmp_path, monkeypatch, caplog):
    """The GET workflow is observable: first entry logs cache=miss + install-from-S3,
    the re-entry on an already-complete dir logs cache=hit with no download."""
    source_dir = tmp_path / "source"

    def download(s3, bucket, prefix, version, dest_parent):
        temp_dir = _temp_dir(dest_parent)
        with open(os.path.join(temp_dir, "model.safetensors.index.json"), "w") as file:
            file.write(_index())
        with open(os.path.join(temp_dir, "model-00000-of-00001.safetensors"), "wb") as file:
            file.write(b"complete")
        return temp_dir

    _, download_mock = _configure(monkeypatch, download)

    with caplog.at_level("INFO", logger="modelexpress.rl.delta_visibility"):
        delta_visibility.ensure_visible(str(source_dir), 1)  # cache miss -> download
        delta_visibility.ensure_visible(str(source_dir), 1)  # cache hit -> skip

    messages = [r.getMessage() for r in caplog.records]
    assert any("version=1 cache=miss downloading from S3" in m for m in messages)
    assert any("version=1 complete shards=1 (installed from S3)" in m for m in messages)
    assert any("version=1 cache=hit" in m and "no download" in m for m in messages)
    assert download_mock.call_count == 1


def test_concurrent_staging_never_exposes_partial_final_dir(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    partial_written = threading.Event()
    finish_write = threading.Event()

    def download(s3, bucket, prefix, version, dest_parent):
        temp_dir = _temp_dir(dest_parent)
        with open(os.path.join(temp_dir, "model.safetensors.index.json"), "w") as file:
            file.write(_index())
        shard_path = os.path.join(temp_dir, "model-00000-of-00001.safetensors")
        with open(shard_path, "wb") as file:
            file.write(b"half")
            file.flush()
            partial_written.set()
            assert finish_write.wait(timeout=5)
            file.write(b"done")
        return temp_dir

    _, download_mock = _configure(monkeypatch, download)
    errors = []

    def stage():
        try:
            delta_visibility.ensure_visible(str(source_dir), 1)
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=stage)
    second = threading.Thread(target=stage)
    first.start()
    assert partial_written.wait(timeout=5)
    second.start()

    final_shard = source_dir / "weight_v000001" / "model-00000-of-00001.safetensors"
    for _ in range(20):
        if final_shard.exists():
            assert final_shard.read_bytes() == b"halfdone"
        time.sleep(0.005)
    finish_write.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not errors
    assert not first.is_alive()
    assert not second.is_alive()
    assert download_mock.call_count == 1
    assert final_shard.read_bytes() == b"halfdone"
