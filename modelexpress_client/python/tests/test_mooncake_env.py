# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Mooncake and etcd environment promotion.

``modelexpress.mooncake_env`` is a leaf module that only uses the standard
library, so the promotion logic itself has no torch/mooncake dependency.
Importing the module still executes ``modelexpress/__init__.py``, so running
these tests requires the client dependencies (torch, grpc) to be installed.
"""

from __future__ import annotations

import os

import pytest

from modelexpress.mooncake_env import mx_mc_env_override

_MOONCAKE_ENV_PREFIXES = ("MX_MC_", "MC_", "MX_ETCD_", "ETCD_")


@pytest.fixture
def clean_mooncake_env(monkeypatch):
    """Remove every promoted Mooncake / etcd variable during the test."""
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

    def test_promotes_and_restores_etcd_variables(
        self, monkeypatch, clean_mooncake_env
    ):
        monkeypatch.setenv("ETCD_USERNAME", "native-user")
        monkeypatch.setenv("ETCD_PASSWORD", "native-password")
        monkeypatch.setenv("MX_ETCD_USERNAME", "artifact-user")
        monkeypatch.setenv("MX_ETCD_PASSWORD", "artifact-password")
        monkeypatch.setenv("MX_ETCD_CA_CERT", "/secrets/artifact-ca.pem")

        with mx_mc_env_override():
            assert os.environ["ETCD_USERNAME"] == "artifact-user"
            assert os.environ["ETCD_PASSWORD"] == "artifact-password"
            assert os.environ["ETCD_CA_CERT"] == "/secrets/artifact-ca.pem"

        assert os.environ["ETCD_USERNAME"] == "native-user"
        assert os.environ["ETCD_PASSWORD"] == "native-password"
        assert "ETCD_CA_CERT" not in os.environ

    def test_restores_on_exception(self, monkeypatch, clean_mooncake_env):
        monkeypatch.setenv("MC_MASTER_SERVER", "A")
        monkeypatch.setenv("ETCD_USERNAME", "native-user")
        monkeypatch.setenv("MX_MC_MASTER_SERVER", "B")
        monkeypatch.setenv("MX_MC_METADATA_ADDR", "mx-metadata")
        monkeypatch.setenv("MX_ETCD_USERNAME", "artifact-user")
        monkeypatch.setenv("MX_ETCD_CA_CERT", "/secrets/artifact-ca.pem")

        with pytest.raises(RuntimeError):
            with mx_mc_env_override():
                assert os.environ["MC_MASTER_SERVER"] == "B"
                assert os.environ["ETCD_USERNAME"] == "artifact-user"
                assert os.environ["ETCD_CA_CERT"] == "/secrets/artifact-ca.pem"
                raise RuntimeError("boom")

        assert os.environ["MC_MASTER_SERVER"] == "A"
        assert "MC_METADATA_ADDR" not in os.environ
        assert os.environ["ETCD_USERNAME"] == "native-user"
        assert "ETCD_CA_CERT" not in os.environ
