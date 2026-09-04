# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelExpress RL-specific environment variables."""

import os

import pytest

from modelexpress_rl import envs


def test_defaults_when_unset(monkeypatch):
    for name in envs.environment_variables:
        monkeypatch.delenv(name, raising=False)

    assert envs.MX_REFIT_METADATA_PORT == 7555
    assert envs.MX_TRAINER_STAGING_MODE == "IN_PLACE"
    assert envs.MX_WEIGHT_PAYLOAD_FORMAT == "FULL_TENSOR"
    assert envs.MX_REFIT_CHECKSUM_FORMAT == "adler32"
    assert envs.MX_REFIT_DELTA_BUCKET_BYTES == 512 * 1024**2
    assert envs.MX_REFIT_DELTA_WORKERS == min(32, os.cpu_count() or 8)
    assert envs.MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES == 4 * 1024**3
    assert envs.MX_S3_MULTIPART_THRESHOLD_BYTES == 100 * 1024**2
    assert envs.MX_S3_UPLOAD_PART_BYTES == 16 * 1024**2
    assert envs.MX_S3_UPLOAD_WORKERS == 8
    assert envs.MX_S3_DOWNLOAD_RANGE_THRESHOLD_BYTES == 100 * 1024**2
    assert envs.MX_S3_DOWNLOAD_RANGE_BYTES == 8 * 1024**2
    assert envs.MX_S3_DOWNLOAD_IO_CHUNK_BYTES == 1024**2
    assert envs.MX_S3_DOWNLOAD_MAX_IN_MEMORY_CHUNKS == 16
    assert envs.MX_S3_DOWNLOAD_WORKERS == 16
    assert envs.MX_S3_MAX_POOL_CONNECTIONS == 32
    assert envs.MX_S3_MAX_ATTEMPTS == 5
    assert envs.MX_S3_TCP_KEEPALIVE is True


def test_values_are_normalized_and_read_live(monkeypatch):
    monkeypatch.setenv("MX_REFIT_METADATA_PORT", "8000")
    monkeypatch.setenv("MX_TRAINER_STAGING_MODE", " copy_to_device ")
    monkeypatch.setenv("MX_WEIGHT_PAYLOAD_FORMAT", " xor_delta ")
    monkeypatch.setenv("MX_REFIT_CHECKSUM_FORMAT", " ADLER32 ")
    monkeypatch.setenv("MX_REFIT_DELTA_BUCKET_BYTES", "1024")
    monkeypatch.setenv("MX_REFIT_DELTA_WORKERS", "3")
    monkeypatch.setenv("MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES", "8192")
    monkeypatch.setenv("MX_S3_MULTIPART_THRESHOLD_BYTES", "2048")
    monkeypatch.setenv("MX_S3_UPLOAD_PART_BYTES", str(5 * 1024**2))
    monkeypatch.setenv("MX_S3_UPLOAD_WORKERS", "4")
    monkeypatch.setenv("MX_S3_DOWNLOAD_RANGE_THRESHOLD_BYTES", "4096")
    monkeypatch.setenv("MX_S3_DOWNLOAD_RANGE_BYTES", "1024")
    monkeypatch.setenv("MX_S3_DOWNLOAD_IO_CHUNK_BYTES", "512")
    monkeypatch.setenv("MX_S3_DOWNLOAD_MAX_IN_MEMORY_CHUNKS", "6")
    monkeypatch.setenv("MX_S3_DOWNLOAD_WORKERS", "5")
    monkeypatch.setenv("MX_S3_MAX_POOL_CONNECTIONS", "6")
    monkeypatch.setenv("MX_S3_MAX_ATTEMPTS", "7")
    monkeypatch.setenv("MX_S3_TCP_KEEPALIVE", "off")

    assert envs.MX_REFIT_METADATA_PORT == 8000
    assert envs.MX_TRAINER_STAGING_MODE == "COPY_TO_DEVICE"
    assert envs.MX_WEIGHT_PAYLOAD_FORMAT == "XOR_DELTA"
    assert envs.MX_REFIT_CHECKSUM_FORMAT == "adler32"
    assert envs.MX_REFIT_DELTA_BUCKET_BYTES == 1024
    assert envs.MX_REFIT_DELTA_WORKERS == 3
    assert envs.MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES == 8192
    assert envs.MX_S3_MULTIPART_THRESHOLD_BYTES == 2048
    assert envs.MX_S3_UPLOAD_PART_BYTES == 5 * 1024**2
    assert envs.MX_S3_UPLOAD_WORKERS == 4
    assert envs.MX_S3_DOWNLOAD_RANGE_THRESHOLD_BYTES == 4096
    assert envs.MX_S3_DOWNLOAD_RANGE_BYTES == 1024
    assert envs.MX_S3_DOWNLOAD_IO_CHUNK_BYTES == 512
    assert envs.MX_S3_DOWNLOAD_MAX_IN_MEMORY_CHUNKS == 6
    assert envs.MX_S3_DOWNLOAD_WORKERS == 5
    assert envs.MX_S3_MAX_POOL_CONNECTIONS == 6
    assert envs.MX_S3_MAX_ATTEMPTS == 7
    assert envs.MX_S3_TCP_KEEPALIVE is False


@pytest.mark.parametrize(
    "name",
    [
        "MX_REFIT_DELTA_BUCKET_BYTES",
        "MX_REFIT_DELTA_WORKERS",
        "MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES",
        "MX_S3_MULTIPART_THRESHOLD_BYTES",
        "MX_S3_UPLOAD_PART_BYTES",
        "MX_S3_UPLOAD_WORKERS",
        "MX_S3_DOWNLOAD_RANGE_THRESHOLD_BYTES",
        "MX_S3_DOWNLOAD_RANGE_BYTES",
        "MX_S3_DOWNLOAD_IO_CHUNK_BYTES",
        "MX_S3_DOWNLOAD_MAX_IN_MEMORY_CHUNKS",
        "MX_S3_DOWNLOAD_WORKERS",
        "MX_S3_MAX_POOL_CONNECTIONS",
        "MX_S3_MAX_ATTEMPTS",
    ],
)
def test_positive_integer_settings_reject_zero(monkeypatch, name):
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValueError, match=f"{name} must be positive"):
        getattr(envs, name)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        _ = envs.NOT_A_REAL_ENV_VAR


def test_refit_metadata_port_must_be_positive(monkeypatch):
    monkeypatch.setenv("MX_REFIT_METADATA_PORT", "0")

    with pytest.raises(ValueError, match="MX_REFIT_METADATA_PORT must be positive"):
        _ = envs.MX_REFIT_METADATA_PORT


def test_dir_lists_registered_names():
    assert set(envs.environment_variables).issubset(dir(envs))


@pytest.mark.parametrize("value", [0, -1])
def test_require_positive_int_rejects_non_positive_values(value):
    with pytest.raises(ValueError, match="count must be positive"):
        envs.require_positive_int(value, "count")


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_require_positive_float_rejects_non_positive_or_non_finite_values(value):
    with pytest.raises(ValueError, match="timeout must be finite and positive"):
        envs.require_positive_float(value, "timeout")


def test_s3_tcp_keepalive_rejects_invalid_boolean(monkeypatch):
    monkeypatch.setenv("MX_S3_TCP_KEEPALIVE", "sometimes")
    with pytest.raises(ValueError, match="MX_S3_TCP_KEEPALIVE must be a boolean"):
        _ = envs.MX_S3_TCP_KEEPALIVE
