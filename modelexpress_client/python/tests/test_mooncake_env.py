# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Mooncake MX_MC_* -> MC_* environment promotion.

``modelexpress.mooncake_env`` is a leaf module that only uses the standard
library, so the promotion logic itself has no torch/mooncake dependency.
Importing the module still executes ``modelexpress/__init__.py``, so running
these tests requires the client dependencies (torch, grpc) to be installed.
"""

from __future__ import annotations

import os

import pytest

from modelexpress.mooncake_env import mx_mc_env_override

_MOONCAKE_ENV_PREFIXES = ("MX_MC_", "MC_")


@pytest.fixture
def clean_mooncake_env(monkeypatch):
    """Remove every MX_MC_* / MC_* variable for the duration of the test."""
    for name in list(os.environ):
        if name.startswith(_MOONCAKE_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    yield


class TestMxMcEnvOverride:
    def test_promotes_and_restores_existing_variables(
        self, monkeypatch, clean_mooncake_env
    ):
        monkeypatch.setenv("MC_MASTER_SERVER", "A")
        monkeypatch.setenv("MC_PROTOCOL", "tcp")
        monkeypatch.setenv("MX_MC_MASTER_SERVER", "B")
        monkeypatch.setenv("MX_MC_MTU", "1024")
        monkeypatch.setenv("NOT_MOONCAKE", "keep")

        with mx_mc_env_override():
            assert os.environ["MC_MASTER_SERVER"] == "B"
            assert os.environ["MC_MTU"] == "1024"
            # MX_MC_PROTOCOL unset: native value must survive untouched.
            assert os.environ["MC_PROTOCOL"] == "tcp"
            # Unrelated variables are never touched.
            assert os.environ["NOT_MOONCAKE"] == "keep"

        assert os.environ["MC_MASTER_SERVER"] == "A"
        assert os.environ["MC_PROTOCOL"] == "tcp"
        assert os.environ["NOT_MOONCAKE"] == "keep"
        # MC_MTU did not exist before; it must be removed again.
        assert "MC_MTU" not in os.environ

    def test_empty_mx_values_do_not_override(
        self, monkeypatch, clean_mooncake_env
    ):
        monkeypatch.setenv("MC_PROTOCOL", "tcp")
        monkeypatch.setenv("MX_MC_PROTOCOL", "")
        monkeypatch.setenv("MX_MC_DEVICE_NAME", "")

        with mx_mc_env_override():
            assert os.environ["MC_PROTOCOL"] == "tcp"
            assert "MC_DEVICE_NAME" not in os.environ

    def test_restores_on_exception(self, monkeypatch, clean_mooncake_env):
        monkeypatch.setenv("MC_MASTER_SERVER", "A")
        monkeypatch.setenv("MX_MC_MASTER_SERVER", "B")
        monkeypatch.setenv("MX_MC_METADATA_ADDR", "mx-metadata")

        with pytest.raises(RuntimeError):
            with mx_mc_env_override():
                assert os.environ["MC_MASTER_SERVER"] == "B"
                raise RuntimeError("boom")

        assert os.environ["MC_MASTER_SERVER"] == "A"
        assert "MC_METADATA_ADDR" not in os.environ
