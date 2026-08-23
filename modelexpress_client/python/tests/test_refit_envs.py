# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelExpress RL-specific environment variables."""

import os

import pytest
from modelexpress_rl import envs


def test_defaults_when_unset(monkeypatch):
    for name in envs.environment_variables:
        monkeypatch.delenv(name, raising=False)

    assert envs.MX_TRAINER_ENGINE == "MEGATRON"
    assert envs.MX_TRAINER_STAGING_MODE == "IN_PLACE"
    assert envs.MX_WEIGHT_PAYLOAD_FORMAT == "FULL_TENSOR"
    assert envs.MX_REFIT_DELTA_BUCKET_BYTES == 512 * 1024**2
    assert envs.MX_REFIT_DELTA_WORKERS == min(32, os.cpu_count() or 8)


def test_values_are_normalized_and_read_live(monkeypatch):
    monkeypatch.setenv("MX_TRAINER_ENGINE", " megatron ")
    monkeypatch.setenv("MX_TRAINER_STAGING_MODE", " copy_to_device ")
    monkeypatch.setenv("MX_WEIGHT_PAYLOAD_FORMAT", " xor_delta ")
    monkeypatch.setenv("MX_REFIT_DELTA_BUCKET_BYTES", "1024")
    monkeypatch.setenv("MX_REFIT_DELTA_WORKERS", "3")

    assert envs.MX_TRAINER_ENGINE == "MEGATRON"
    assert envs.MX_TRAINER_STAGING_MODE == "COPY_TO_DEVICE"
    assert envs.MX_WEIGHT_PAYLOAD_FORMAT == "XOR_DELTA"
    assert envs.MX_REFIT_DELTA_BUCKET_BYTES == 1024
    assert envs.MX_REFIT_DELTA_WORKERS == 3


@pytest.mark.parametrize(
    "name",
    [
        "MX_REFIT_DELTA_BUCKET_BYTES",
        "MX_REFIT_DELTA_WORKERS",
    ],
)
def test_positive_integer_settings_reject_zero(monkeypatch, name):
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValueError, match=f"{name} must be positive"):
        getattr(envs, name)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        _ = envs.NOT_A_REAL_ENV_VAR


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
