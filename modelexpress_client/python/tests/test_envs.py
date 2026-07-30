# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the centralized environment-variable registry."""

import pytest

from modelexpress import envs


def test_defaults_when_unset(monkeypatch):
    for name in (
        "MX_NIXL_BACKEND",
        "MX_METADATA_PORT",
        "MX_WORKER_GRPC_PORT",
        "MX_POOL_REG",
        "MX_VMM_ARENA",
        "MX_MS_DISTRIBUTED",
        "MX_INSTANT_TENSOR",
        "VLLM_ATTENTION_BACKEND",
        "SGLANG_CACHE_DIR",
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
        "MODEL_EXPRESS_URL",
        "MX_SERVER_ADDRESS",
        "MX_GDS_TIMEOUT",
        "MX_HEARTBEAT_INTERVAL_SECS",
    ):
        monkeypatch.delenv(name, raising=False)

    assert envs.MX_NIXL_BACKEND == "UCX"
    assert envs.MX_METADATA_PORT == 5555
    assert envs.MX_WORKER_GRPC_PORT == 6555
    assert envs.MX_POOL_REG is False
    assert envs.MX_VMM_ARENA is False
    assert envs.MX_MS_DISTRIBUTED is True
    assert envs.MX_INSTANT_TENSOR is True
    assert envs.VLLM_ATTENTION_BACKEND == "auto"
    assert envs.SGLANG_CACHE_DIR is None
    assert envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR is None
    assert envs.MODEL_EXPRESS_URL is None
    assert envs.MX_SERVER_ADDRESS is None
    assert envs.MX_GDS_TIMEOUT == pytest.approx(120.0)
    assert envs.MX_HEARTBEAT_INTERVAL_SECS == 30


def test_int_and_float_parsing(monkeypatch):
    monkeypatch.setenv("MX_METADATA_PORT", "1234")
    monkeypatch.setenv("MX_GDS_TIMEOUT", "1.5")
    monkeypatch.setenv("MX_SOURCE_QUERY_TIMEOUT", "42")
    assert envs.MX_METADATA_PORT == 1234
    assert envs.MX_GDS_TIMEOUT == pytest.approx(1.5)
    assert envs.MX_SOURCE_QUERY_TIMEOUT == 42


def test_invalid_int_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MX_METADATA_PORT", "not-a-number")
    assert envs.MX_METADATA_PORT == 5555


def test_bool_parsing(monkeypatch):
    monkeypatch.setenv("MX_POOL_REG", "1")
    assert envs.MX_POOL_REG is True
    monkeypatch.setenv("MX_POOL_REG", "0")
    assert envs.MX_POOL_REG is False

    monkeypatch.setenv("MX_MS_DISTRIBUTED", "true")
    assert envs.MX_MS_DISTRIBUTED is True
    monkeypatch.setenv("MX_MS_DISTRIBUTED", "no")
    assert envs.MX_MS_DISTRIBUTED is False

    monkeypatch.setenv("MX_INSTANT_TENSOR", "0")
    assert envs.MX_INSTANT_TENSOR is False
    monkeypatch.setenv("MX_INSTANT_TENSOR", "true")
    assert envs.MX_INSTANT_TENSOR is True

    monkeypatch.setenv("MX_VMM_ARENA", "1")
    assert envs.MX_VMM_ARENA is True

    for truthy in ("1", "TRUE", "yes", "On"):
        monkeypatch.setenv("MX_ARTIFACT_TRANSFER", truthy)
        assert envs.MX_ARTIFACT_TRANSFER is True
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "maybe")
    assert envs.MX_ARTIFACT_TRANSFER is False


def test_normalization(monkeypatch):
    monkeypatch.setenv("MX_NIXL_BACKEND", "  libfabric ")
    assert envs.MX_NIXL_BACKEND == "LIBFABRIC"
    monkeypatch.setenv("MX_METADATA_BACKEND", "  Redis  ")
    assert envs.MX_METADATA_BACKEND == "redis"
    monkeypatch.setenv("MODEL_EXPRESS_LOG_LEVEL", "debug")
    assert envs.MODEL_EXPRESS_LOG_LEVEL == "DEBUG"


def test_raw_optional_passthrough(monkeypatch, tmp_path):
    monkeypatch.delenv("MX_MODEL_URI", raising=False)
    assert envs.MX_MODEL_URI is None
    monkeypatch.setenv("MX_MODEL_URI", "s3://bucket/model")
    assert envs.MX_MODEL_URI == "s3://bucket/model"
    # Raw string; artifact_manifest owns the parse/validation.
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER_CHUNK_SIZE", "12345")
    assert envs.MX_ARTIFACT_TRANSFER_CHUNK_SIZE == "12345"
    autotune_root = tmp_path / "flashinfer-autotune"
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(autotune_root))
    assert envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR == str(autotune_root)
    sglang_cache_root = tmp_path / "sglang-cache"
    monkeypatch.setenv("SGLANG_CACHE_DIR", str(sglang_cache_root))
    assert envs.SGLANG_CACHE_DIR == str(sglang_cache_root)


def test_is_set(monkeypatch):
    monkeypatch.delenv("MX_WORKER_HOST", raising=False)
    assert envs.is_set("MX_WORKER_HOST") is False
    monkeypatch.setenv("MX_WORKER_HOST", "")
    assert envs.is_set("MX_WORKER_HOST") is True


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        _ = envs.NOT_A_REAL_ENV_VAR


def test_dir_lists_registered_names():
    names = dir(envs)
    assert "MX_NIXL_BACKEND" in names
    assert "VLLM_ATTENTION_BACKEND" in names
    assert names == sorted(names)
