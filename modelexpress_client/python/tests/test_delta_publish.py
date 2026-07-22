# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from types import SimpleNamespace
from unittest import mock

import pytest

from modelexpress import p2p_pb2
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
        hf_checkpoint="/Models/CaseSensitive/Qwen3-4B",
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MX_S3_ENDPOINT", "http://mx-minio:9000")
    monkeypatch.setenv("MX_S3_BUCKET", "CaseSensitiveBucket")
    monkeypatch.setenv("MX_DELTA_STREAM_PREFIX", "Runs/CaseSensitiveStream")
    monkeypatch.setenv("MODEL_EXPRESS_URL", "mx-server:8001")
    monkeypatch.delenv("MX_DELTA_POC_EVICT_LOCAL", raising=False)
    monkeypatch.setattr(delta_publish, "_baseline_failure", None)
    return args, version_dir


def _mock_publish(monkeypatch, acknowledged=True):
    fake_s3 = mock.MagicMock()
    fake_client = mock.MagicMock()
    fake_client.publish_metadata.return_value = "source-id"
    fake_client.update_status.return_value = acknowledged
    monkeypatch.setattr(delta_publish, "s3_client", mock.Mock(return_value=fake_s3))
    upload = mock.Mock(return_value=2)
    monkeypatch.setattr(delta_publish, "upload_version_dir", upload)
    client_cls = mock.Mock(return_value=fake_client)
    monkeypatch.setattr(delta_publish, "MxClient", client_cls)
    return fake_s3, upload, client_cls, fake_client


def test_rank_one_is_noop(publish_case, monkeypatch):
    args, version_dir = publish_case
    monkeypatch.setenv("RANK", "1")
    s3 = mock.Mock()
    client = mock.Mock()
    monkeypatch.setattr(delta_publish, "s3_client", s3)
    monkeypatch.setattr(delta_publish, "MxClient", client)

    delta_publish.publish_delta(args, str(version_dir))

    s3.assert_not_called()
    client.assert_not_called()


def test_rank_zero_uploads_then_publishes_checked_ready(publish_case, monkeypatch):
    args, version_dir = publish_case
    fake_s3, upload, client_cls, fake_client = _mock_publish(monkeypatch)

    delta_publish.publish_delta(args, str(version_dir))

    upload.assert_called_once_with(
        fake_s3,
        "CaseSensitiveBucket",
        "Runs/CaseSensitiveStream",
        str(version_dir),
    )
    client_cls.assert_called_once_with("mx-server:8001")
    identity, worker, worker_id = fake_client.publish_metadata.call_args.args
    assert identity.extra_parameters == {
        "training_step": "1",
        "base_version": "0",
        "layout_signature": "miles-disk-delta-v1",
    }
    assert re.fullmatch(r"[0-9a-f]{64}", identity.revision)
    assert identity.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_WEIGHTS
    assert worker.worker_rank == 0
    fake_client.update_status.assert_called_once_with(
        "source-id",
        worker_id,
        0,
        p2p_pb2.SOURCE_STATUS_READY,
    )
    serialized = identity.SerializeToString()
    assert b"CaseSensitiveBucket" not in serialized
    assert b"Runs/CaseSensitiveStream" not in serialized


def test_false_update_status_raises(publish_case, monkeypatch):
    args, version_dir = publish_case
    _mock_publish(monkeypatch, acknowledged=False)

    with pytest.raises(RuntimeError, match="did not acknowledge"):
        delta_publish.publish_delta(args, str(version_dir))


def test_evict_local_removes_version_after_verified_publish(
    publish_case, monkeypatch
):
    args, version_dir = publish_case
    _mock_publish(monkeypatch)
    monkeypatch.setenv("MX_DELTA_POC_EVICT_LOCAL", "1")

    delta_publish.publish_delta(args, str(version_dir))

    assert not version_dir.exists()


def test_no_model_express_url_uploads_without_mx_publish(publish_case, monkeypatch):
    """When MODEL_EXPRESS_URL is unset (weight-sync-only POC), upload happens and
    MxClient is never constructed; eviction still works."""
    args, version_dir = publish_case
    fake_s3, upload, client_cls, fake_client = _mock_publish(monkeypatch)
    monkeypatch.delenv("MODEL_EXPRESS_URL", raising=False)
    monkeypatch.setenv("MX_DELTA_POC_EVICT_LOCAL", "1")

    delta_publish.publish_delta(args, str(version_dir))

    upload.assert_called_once()
    client_cls.assert_not_called()
    fake_client.publish_metadata.assert_not_called()
    assert not version_dir.exists()


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
