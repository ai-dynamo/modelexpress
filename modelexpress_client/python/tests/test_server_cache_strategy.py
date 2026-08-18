# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ServerCacheStrategy."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from modelexpress import model_prefetch, p2p_pb2
from modelexpress.adapter import EngineAdapter, StrategyFailed
from modelexpress.load_strategy.context import LoadResult
from modelexpress.load_strategy.server_cache_strategy import ServerCacheStrategy

REPO = "org/model"
COMMIT = "a" * 40


class _FakeAdapter(EngineAdapter):
    """Adapter implementing the native-load capability the strategy requires."""

    def __init__(self, *, native_error=None):
        self.native_error = native_error
        self.native_calls = 0
        self.post_calls = 0

    def discover_tensors(self, result: LoadResult):
        return {}

    def load_via_native(self, result: LoadResult) -> LoadResult:
        self.native_calls += 1
        if self.native_error is not None:
            raise self.native_error
        return result

    def after_native_load(self, result: LoadResult) -> LoadResult:
        self.post_calls += 1
        return result


class _NoNativeAdapter(EngineAdapter):
    """Adapter without load_via_native, so the strategy must be ineligible."""

    def discover_tensors(self, result: LoadResult):
        return {}


class FakeClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.error = None
        FakeClient.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def install_weight_files(self, repo_id, snapshot_path, *args, **kwargs):
        self.calls.append((repo_id, snapshot_path))
        if self.error is not None:
            raise self.error


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    model_prefetch.reset()
    FakeClient.instances = []
    for name in ("MODEL_EXPRESS_NO_SHARED_STORAGE", "MODEL_EXPRESS_URL", "MX_SERVER_ADDRESS"):
        monkeypatch.delenv(name, raising=False)
    yield
    model_prefetch.reset()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_NO_SHARED_STORAGE", "1")
    monkeypatch.setenv("MODEL_EXPRESS_URL", "http://mx:8001")


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "models--org--model" / "snapshots" / COMMIT
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    return path


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setattr("modelexpress.model_client.ModelCacheClient", FakeClient)
    return FakeClient


def _make_context(model_name, *, adapter=None, model_path=None, revision=None):
    from modelexpress.load_strategy import LoadContext

    return LoadContext(
        model_config=SimpleNamespace(model=model_path, revision=revision),
        load_config=MagicMock(),
        target_device=torch.device("cpu"),
        global_rank=0,
        worker_rank=0,
        device_id=0,
        identity=p2p_pb2.SourceIdentity(model_name=model_name, tensor_parallel_size=1),
        mx_client=MagicMock(),
        worker_id="test-worker",
        adapter=adapter if adapter is not None else _FakeAdapter(),
    )


class TestIsAvailable:
    def test_unavailable_when_switch_is_off(self):
        assert ServerCacheStrategy().is_available(_make_context(REPO)) is False

    def test_unavailable_without_server_address(self, monkeypatch):
        monkeypatch.setenv("MODEL_EXPRESS_NO_SHARED_STORAGE", "1")
        assert ServerCacheStrategy().is_available(_make_context(REPO)) is False

    def test_unavailable_without_native_load_capability(self, enabled):
        ctx = _make_context(REPO, adapter=_NoNativeAdapter())
        assert ServerCacheStrategy().is_available(ctx) is False

    def test_available_for_a_repo_id(self, enabled):
        assert ServerCacheStrategy().is_available(_make_context(REPO)) is True

    def test_available_for_a_registered_snapshot_path(self, enabled, snapshot):
        model_prefetch._snapshot_to_repo_id[str(snapshot)] = REPO
        assert ServerCacheStrategy().is_available(_make_context(str(snapshot))) is True

    def test_unavailable_for_an_unknown_local_path(self, enabled):
        ctx = _make_context("/opt/models/llama")
        assert ServerCacheStrategy().is_available(ctx) is False

    def test_available_in_a_process_that_never_ran_the_prefetch(self, enabled, snapshot):
        """The EngineCore process has no prefetch record; the path must suffice.

        vLLM rewrites ModelConfig.model with the resolved snapshot path and
        loads weights in a separate process, so is_available() has to recover
        the repo id from the cache layout alone.
        """
        model_prefetch.reset()
        ctx = _make_context(str(snapshot))
        assert ServerCacheStrategy().is_available(ctx) is True


class TestLoad:
    def test_installs_weights_then_loads_natively(self, enabled, snapshot, fake_client):
        adapter = _FakeAdapter()
        ctx = _make_context(REPO, adapter=adapter, model_path=str(snapshot))
        result = LoadResult(value=MagicMock(), model=MagicMock())

        with patch(
            "modelexpress.load_strategy.server_cache_strategy.register_tensors"
        ) as register:
            out = ServerCacheStrategy().load(result, ctx)

        assert FakeClient.instances[0].calls == [(REPO, snapshot)]
        assert adapter.native_calls == 1
        assert adapter.post_calls == 1
        assert register.call_count == 1
        assert out is result

    def test_uses_the_snapshot_the_engine_resolved(self, enabled, snapshot, fake_client):
        """model_config.model is the path the engine is already reading from."""
        model_prefetch._snapshot_to_repo_id[str(snapshot)] = REPO
        ctx = _make_context(str(snapshot), model_path=str(snapshot))

        with patch("modelexpress.load_strategy.server_cache_strategy.register_tensors"):
            ServerCacheStrategy().load(LoadResult(value=MagicMock()), ctx)

        assert FakeClient.instances[0].calls == [(REPO, snapshot)]

    def test_installs_metadata_when_no_snapshot_exists(self, enabled, snapshot, fake_client):
        ctx = _make_context(REPO, model_path=None)

        with patch.object(model_prefetch, "ensure_metadata", return_value=snapshot) as ensure:
            with patch("modelexpress.load_strategy.server_cache_strategy.register_tensors"):
                ServerCacheStrategy().load(LoadResult(value=MagicMock()), ctx)

        assert ensure.call_count == 1
        assert FakeClient.instances[0].calls == [(REPO, snapshot)]

    def test_server_failure_is_a_clean_miss(self, enabled, snapshot, monkeypatch):
        def failing_factory(**kwargs):
            client = FakeClient(**kwargs)
            client.error = RuntimeError("server unreachable")
            return client

        monkeypatch.setattr("modelexpress.model_client.ModelCacheClient", failing_factory)
        adapter = _FakeAdapter()
        ctx = _make_context(REPO, adapter=adapter, model_path=str(snapshot))

        with pytest.raises(StrategyFailed) as excinfo:
            ServerCacheStrategy().load(LoadResult(value=MagicMock()), ctx)

        assert excinfo.value.mutated is False
        assert adapter.native_calls == 0

    def test_native_load_failure_reports_a_mutated_model(self, enabled, snapshot, fake_client):
        adapter = _FakeAdapter(native_error=RuntimeError("bad checkpoint"))
        ctx = _make_context(REPO, adapter=adapter, model_path=str(snapshot))

        with pytest.raises(StrategyFailed) as excinfo:
            ServerCacheStrategy().load(LoadResult(value=MagicMock()), ctx)

        assert excinfo.value.mutated is True

    def test_missing_snapshot_is_a_clean_miss(self, enabled, fake_client):
        ctx = _make_context(REPO, model_path=None)

        with patch.object(model_prefetch, "ensure_metadata", return_value=None):
            with pytest.raises(StrategyFailed) as excinfo:
                ServerCacheStrategy().load(LoadResult(value=MagicMock()), ctx)

        assert excinfo.value.mutated is False
        assert FakeClient.instances == []


class TestChainOrder:
    def test_sits_between_rdma_and_local_strategies(self):
        import inspect

        from modelexpress.load_strategy import LoadStrategyChain

        source = inspect.getsource(LoadStrategyChain.run)
        order = [
            name
            for name in (
                "RdmaStrategy()",
                "ServerCacheStrategy()",
                "InstantTensorStrategy()",
                "DefaultStrategy()",
            )
            if name in source
        ]
        assert order == [
            "RdmaStrategy()",
            "ServerCacheStrategy()",
            "InstantTensorStrategy()",
            "DefaultStrategy()",
        ]
        assert source.index("RdmaStrategy()") < source.index("ServerCacheStrategy()")
        assert source.index("ServerCacheStrategy()") < source.index("InstantTensorStrategy()")
