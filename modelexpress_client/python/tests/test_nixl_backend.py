# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MX_NIXL_BACKEND resolution in NixlTransferManager."""

import pytest

from modelexpress.nixl_transfer import (
    DEFAULT_NIXL_BACKEND,
    NIXL_ACCELERATOR_MEM_TYPE,
    NixlTransferManager,
    SUPPORTED_NIXL_BACKENDS,
    _resolve_nixl_backend,
)


class TestResolveNixlBackend:
    def test_default_is_ucx(self, monkeypatch):
        monkeypatch.delenv("MX_NIXL_BACKEND", raising=False)
        assert _resolve_nixl_backend() == "UCX"
        assert DEFAULT_NIXL_BACKEND == "UCX"

    def test_libfabric_explicit(self, monkeypatch):
        monkeypatch.setenv("MX_NIXL_BACKEND", "LIBFABRIC")
        assert _resolve_nixl_backend() == "LIBFABRIC"

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("MX_NIXL_BACKEND", "libfabric")
        assert _resolve_nixl_backend() == "LIBFABRIC"

    def test_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("MX_NIXL_BACKEND", "  ucx  ")
        assert _resolve_nixl_backend() == "UCX"

    def test_unknown_backend_raises(self, monkeypatch):
        monkeypatch.setenv("MX_NIXL_BACKEND", "GDS_MT")
        with pytest.raises(ValueError, match="MX_NIXL_BACKEND="):
            _resolve_nixl_backend()

    def test_supported_backends_contains_both(self):
        assert "UCX" in SUPPORTED_NIXL_BACKENDS
        assert "LIBFABRIC" in SUPPORTED_NIXL_BACKENDS


def test_raw_accelerator_descriptors_use_nixl_vram_segment():
    assert NIXL_ACCELERATOR_MEM_TYPE == "VRAM"


class TestNixlTransferManagerBackend:
    """Verify the manager picks up the env var at construction time."""

    def test_default_backend_on_manager(self, monkeypatch):
        monkeypatch.delenv("MX_NIXL_BACKEND", raising=False)
        mgr = NixlTransferManager(agent_name="test", device_id=0)
        assert mgr._backend == "UCX"
        assert mgr._backends == ["UCX"]

    def test_libfabric_backend_on_manager(self, monkeypatch):
        monkeypatch.setenv("MX_NIXL_BACKEND", "LIBFABRIC")
        mgr = NixlTransferManager(agent_name="test", device_id=0)
        assert mgr._backend == "LIBFABRIC"
        assert mgr._backends == ["LIBFABRIC"]

    def test_invalid_value_fails_construction(self, monkeypatch):
        monkeypatch.setenv("MX_NIXL_BACKEND", "bogus")
        with pytest.raises(ValueError, match="MX_NIXL_BACKEND="):
            NixlTransferManager(agent_name="test", device_id=0)
