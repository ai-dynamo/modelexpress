# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelExpress RL-specific environment variables."""

import pytest
from modelexpress_rl import envs


def test_defaults_when_unset(monkeypatch):
    for name in envs.environment_variables:
        monkeypatch.delenv(name, raising=False)

    assert envs.MX_REFIT_METADATA_PORT == 7555
    assert envs.MX_TRAINER_ENGINE == "MEGATRON"
    assert envs.MX_TRAINER_STAGING_MODE == "IN_PLACE"
    assert envs.MX_WEIGHT_PAYLOAD_FORMAT == "FULL_TENSOR"


def test_values_are_normalized_and_read_live(monkeypatch):
    monkeypatch.setenv("MX_REFIT_METADATA_PORT", "8000")
    monkeypatch.setenv("MX_TRAINER_ENGINE", " megatron ")
    monkeypatch.setenv("MX_TRAINER_STAGING_MODE", " copy_to_device ")
    monkeypatch.setenv("MX_WEIGHT_PAYLOAD_FORMAT", " xor_delta ")

    assert envs.MX_REFIT_METADATA_PORT == 8000
    assert envs.MX_TRAINER_ENGINE == "MEGATRON"
    assert envs.MX_TRAINER_STAGING_MODE == "COPY_TO_DEVICE"
    assert envs.MX_WEIGHT_PAYLOAD_FORMAT == "XOR_DELTA"


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
