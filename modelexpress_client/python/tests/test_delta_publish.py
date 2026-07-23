# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from modelexpress.rl import delta_publish
from modelexpress.rl.s3_delta import s3_client, upload_version_dir


@pytest.fixture
def publish_case(tmp_path, monkeypatch):
    delta_dir = tmp_path / "delta"
    version_dir = delta_dir / "weight_v000001"
    version_dir.mkdir(parents=True)
    (version_dir / "model-00000-of-00001.safetensors").write_bytes(b"delta")
    (version_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "version": "000001",
                    "base_version": "000000",
                },
                "weight_map": {
                    "model.weight": "model-00000-of-00001.safetensors",
                },
            }
        )
    )
    args = SimpleNamespace(
        update_weight_disk_dir=str(delta_dir),
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        world_size=4,
        model_name="Qwen/Qwen3-4B",
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MX_S3_ENDPOINT", "http://mx-minio:9000")
    monkeypatch.setenv("MX_S3_BUCKET", "CaseSensitiveBucket")
    monkeypatch.setenv("MX_DELTA_STREAM_PREFIX", "Runs/CaseSensitiveStream")
    monkeypatch.delenv("MX_DELTA_POC_EVICT_LOCAL", raising=False)
    monkeypatch.setattr(delta_publish, "_baseline_failure", None)
    return args, version_dir


def _mock_upload(monkeypatch):
    fake_s3 = mock.MagicMock()
    monkeypatch.setattr(delta_publish, "s3_client", mock.Mock(return_value=fake_s3))
    upload = mock.Mock(return_value=2)
    monkeypatch.setattr(delta_publish, "upload_version_dir", upload)
    return fake_s3, upload


def test_rank_one_is_noop(publish_case, monkeypatch):
    args, version_dir = publish_case
    monkeypatch.setenv("RANK", "1")
    s3 = mock.Mock()
    monkeypatch.setattr(delta_publish, "s3_client", s3)

    delta_publish.publish_delta(args, str(version_dir))

    s3.assert_not_called()


def test_rank_zero_uploads_to_stream_and_version_prefix(publish_case, monkeypatch):
    args, version_dir = publish_case
    fake_s3, upload = _mock_upload(monkeypatch)

    delta_publish.publish_delta(args, str(version_dir))

    upload.assert_called_once_with(
        fake_s3,
        "CaseSensitiveBucket",
        "Runs/CaseSensitiveStream",
        str(version_dir),
    )


def test_evict_local_removes_version_after_upload(publish_case, monkeypatch):
    args, version_dir = publish_case
    _mock_upload(monkeypatch)
    monkeypatch.setenv("MX_DELTA_POC_EVICT_LOCAL", "1")

    delta_publish.publish_delta(args, str(version_dir))

    assert not version_dir.exists()


def test_baseline_call_records_and_does_not_upload(publish_case, monkeypatch):
    """When version_dir is the shared disk_dir itself (baseline v0), record
    lineage and return without any S3 upload."""
    args, _ = publish_case
    _fake_s3, upload = _mock_upload(monkeypatch)

    delta_publish.publish_delta(args, args.update_weight_disk_dir)

    upload.assert_not_called()
    assert delta_publish._baseline_failure is None


def test_upload_helper_skips_tmp_and_keys_under_stream_and_version(tmp_path):
    version_dir = tmp_path / "weight_v000007"
    version_dir.mkdir()
    (version_dir / "model.safetensors").write_bytes(b"complete")
    (version_dir / "model.safetensors.tmp").write_bytes(b"partial")
    fake_s3 = mock.MagicMock()

    count = upload_version_dir(fake_s3, "bucket", "run-id", str(version_dir))

    assert count == 1
    assert fake_s3.upload_file.call_args.args[2] == (
        "run-id/weight_v000007/model.safetensors"
    )


def test_s3_client_uses_path_style_and_standard_retries(monkeypatch):
    monkeypatch.setenv("MX_S3_ENDPOINT", "http://mx-minio:9000")
    with mock.patch("boto3.client") as boto_client:
        s3_client()

    _, kwargs = boto_client.call_args
    assert kwargs["endpoint_url"] == "http://mx-minio:9000"
    assert kwargs["config"].s3["addressing_style"] == "path"
    assert kwargs["config"].retries == {
        "max_attempts": 5,
        "mode": "standard",
    }


def test_bandwidth_mb_s_edges_and_value():
    from modelexpress.rl.s3_delta import _bandwidth_mb_s, _mb

    assert _bandwidth_mb_s(0, 0) == 0.0          # no bytes, no time
    assert _bandwidth_mb_s(1_000_000, 0) == 0.0  # guard against div-by-zero
    assert _bandwidth_mb_s(0, 1.0) == 0.0        # no bytes
    # decimal MB: 1e6 bytes == 1.0 MB
    assert abs(_mb(1_000_000) - 1.0) < 1e-9
    # 1 MB in 1 s == 1.0 MB/s
    assert abs(_bandwidth_mb_s(1_000_000, 1.0) - 1.0) < 1e-9
    # 2 MB in 0.5 s == 4.0 MB/s
    assert abs(_bandwidth_mb_s(2_000_000, 0.5) - 4.0) < 1e-9
