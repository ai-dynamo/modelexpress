# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the snapshot_download prefetch hook."""

import importlib

import pytest

from modelexpress import model_prefetch
from modelexpress.engines.vllm.patches.patch_hf_snapshot_prefetch import (
    _PATCH_TARGETS,
    patch_hf_snapshot_prefetch,
)

REPO = "org/model"


# Captured at import time, before anything here can patch it. Restoring from a
# fixture-local snapshot is not enough: the patch mutates huggingface_hub
# process-wide, and a stub that survives teardown silently reroutes every later
# test that resolves a model.
_PRISTINE = importlib.import_module("huggingface_hub._snapshot_download").snapshot_download

# Which namespaces bind the name varies by version: 1.8 has HfApi import it
# inside the method body, older releases bind it at hf_api module level.
_BOUND_MODULES = [
    module
    for module in (importlib.import_module(name) for name in _PATCH_TARGETS)
    if getattr(module, "snapshot_download", None) is _PRISTINE
]


def _modules():
    return _BOUND_MODULES


@pytest.fixture(autouse=True)
def restore_snapshot_download():
    yield
    for module in _BOUND_MODULES:
        module.snapshot_download = _PRISTINE


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("MODEL_EXPRESS_NO_SHARED_STORAGE", "MODEL_EXPRESS_URL", "MX_SERVER_ADDRESS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("MX_DISABLE_PATCHES", raising=False)
    model_prefetch.reset()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_NO_SHARED_STORAGE", "1")
    monkeypatch.setenv("MODEL_EXPRESS_URL", "http://mx:8001")


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_ensure(repo_id, revision=None):
        calls.append((repo_id, revision))
        return None

    monkeypatch.setattr(model_prefetch, "ensure_metadata", fake_ensure)
    return calls


@pytest.fixture
def stub_download():
    """Replace the real downloader so patched calls stay offline.

    Deliberately not monkeypatch: restore_snapshot_download is the single
    owner of this global, and two undo mechanisms racing over the same
    attribute is what let a stub escape into the rest of the suite.
    """
    calls = []

    def fake_snapshot_download(*args, **kwargs):
        calls.append((args, kwargs))
        return "/local/snapshot"

    for module in _BOUND_MODULES:
        module.snapshot_download = fake_snapshot_download
    return calls


class TestPatchGating:
    def test_no_op_when_disabled(self, stub_download):
        assert patch_hf_snapshot_prefetch() is False
        for module in _modules():
            assert getattr(module.snapshot_download, "__modelexpress_patched__", False) is False

    def test_applies_when_enabled(self, enabled, stub_download):
        assert patch_hf_snapshot_prefetch() is True
        for module in _modules():
            assert module.snapshot_download.__modelexpress_patched__ is True

    def test_is_idempotent(self, enabled, stub_download):
        assert patch_hf_snapshot_prefetch() is True
        assert patch_hf_snapshot_prefetch() is False

    def test_disable_patches_env_skips_it(self, enabled, stub_download, monkeypatch):
        from modelexpress.patches import apply_patches

        monkeypatch.setenv("MX_DISABLE_PATCHES", "1")
        apply_patches([patch_hf_snapshot_prefetch])
        for module in _modules():
            assert getattr(module.snapshot_download, "__modelexpress_patched__", False) is False


class TestPatchedBehavior:
    def test_prefetches_then_delegates(self, enabled, stub_download, recorded):
        patch_hf_snapshot_prefetch()
        import huggingface_hub

        result = huggingface_hub.snapshot_download(REPO, revision="abc")

        assert recorded == [(REPO, "abc")]
        assert result == "/local/snapshot"

    def test_reaches_the_hf_api_call_path(self, enabled, stub_download, recorded):
        """The path vLLM actually takes: HfApi().snapshot_download.

        This is the call in the issue's traceback, and it resolves the name
        differently across huggingface_hub versions -- module-level in older
        releases, a local import inside the method in 1.8. Patching the source
        module has to cover both.
        """
        patch_hf_snapshot_prefetch()
        from huggingface_hub import HfApi

        HfApi().snapshot_download(repo_id=REPO)

        assert recorded == [(REPO, None)]

    def test_repo_id_as_keyword(self, enabled, stub_download, recorded):
        patch_hf_snapshot_prefetch()
        import huggingface_hub

        huggingface_hub.snapshot_download(repo_id=REPO)

        assert recorded == [(REPO, None)]

    def test_skips_non_model_repos(self, enabled, stub_download, recorded):
        patch_hf_snapshot_prefetch()
        import huggingface_hub

        huggingface_hub.snapshot_download(REPO, repo_type="dataset")

        assert recorded == []

    def test_prefetch_failure_keeps_original_semantics(self, enabled, stub_download, monkeypatch):
        def failing_ensure(repo_id, revision=None):
            raise RuntimeError("server unreachable")

        monkeypatch.setattr(model_prefetch, "ensure_metadata", failing_ensure)
        patch_hf_snapshot_prefetch()
        import huggingface_hub

        assert huggingface_hub.snapshot_download(REPO) == "/local/snapshot"
        assert len(stub_download) == 1

    def test_arguments_are_forwarded_untouched(self, enabled, stub_download, recorded):
        patch_hf_snapshot_prefetch()
        import huggingface_hub

        huggingface_hub.snapshot_download(
            REPO, revision="abc", local_files_only=True, allow_patterns=["*.json"]
        )

        args, kwargs = stub_download[0]
        assert args == (REPO,)
        assert kwargs["revision"] == "abc"
        assert kwargs["local_files_only"] is True
        assert kwargs["allow_patterns"] == ["*.json"]
