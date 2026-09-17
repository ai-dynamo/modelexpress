# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ServerCacheStrategy."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import grpc
import pytest
import torch

from modelexpress import model_pb2, model_prefetch, p2p_pb2
from modelexpress.adapter import EngineAdapter, StrategyFailed
from modelexpress.load_strategy.context import LoadResult
from modelexpress.load_strategy.server_cache_strategy import ServerCacheStrategy
from modelexpress.model_client import ModelCacheClient, ModelCacheError
from modelexpress.model_snapshot import ModelSnapshotCache

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


class FakeClient(ModelCacheClient):
    instances = []

    def __init__(self, **kwargs):
        super().__init__(server_url="localhost:1", **kwargs)
        self.kwargs = kwargs
        self.calls = []
        self.error = None
        FakeClient.instances.append(self)

    @property
    def stub(self):
        raise AssertionError("Metadata fixtures must not require an RPC")

    def install_weight_files(self, repo_id, snapshot_path, *args, **kwargs):
        self.calls.append((repo_id, snapshot_path))
        if self.error is not None:
            raise self.error


class _LegacyModelService:
    def __init__(self, *, pin_rpc_error=False, stream_commit=COMMIT):
        self.pin_rpc_error = pin_rpc_error
        self.stream_commit = stream_commit
        self.download_requests = []
        self.list_requests = []
        self.stream_requests = []

    def EnsureModelDownloaded(self, request):
        self.download_requests.append(request)
        if self.pin_rpc_error and request.HasField("revision"):
            raise grpc.RpcError("Pinned revision lookup is unavailable")
        return iter([
            model_pb2.ModelStatusUpdate(
                model_name=REPO, status=model_pb2.DOWNLOADED
            )
        ])

    def ListModelFiles(self, request):
        self.list_requests.append(request)
        return model_pb2.ModelFileList(
            model_name=REPO,
            files=[model_pb2.ModelFileInfo(relative_path="model.safetensors", size=7)],
            total_size=7,
        )

    def StreamModelFiles(self, request):
        self.stream_requests.append(request)
        return iter([
            model_pb2.FileChunk(
                relative_path="model.safetensors",
                data=b"weights",
                offset=0,
                total_size=7,
                is_last_chunk=True,
                is_last_file=True,
                commit_hash=self.stream_commit,
            )
        ])


def _use_legacy_service(monkeypatch, *, pin_rpc_error, stream_commit=COMMIT):
    service = _LegacyModelService(
        pin_rpc_error=pin_rpc_error, stream_commit=stream_commit
    )

    def client_factory(**kwargs):
        client = ModelCacheClient(server_url="localhost:1", **kwargs)
        client._stub = service
        return client

    monkeypatch.setattr("modelexpress.model_client.ModelCacheClient", client_factory)
    return service


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
    cache = ModelSnapshotCache(REPO, tmp_path)
    with cache.lock():
        path = cache.snapshot_path(COMMIT)
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        cache._write_metadata_inventory(COMMIT, {"config.json": 2})
    return path


@pytest.fixture
def legacy_snapshot(tmp_path):
    path = ModelSnapshotCache(REPO, tmp_path).snapshot_path(COMMIT)
    path.mkdir(parents=True)
    (path / "config.json").write_bytes(b"{}")
    return path


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setattr("modelexpress.model_client.ModelCacheClient", FakeClient)
    return FakeClient


def _make_context(model_name, *, adapter=None, model_path=None, revision=None):
    """Build a load context for server-cache strategy tests."""
    from modelexpress.load_strategy import LoadContext

    return LoadContext(
        model_config=SimpleNamespace(model=model_path, revision=revision),
        load_config=MagicMock(),
        target_device=torch.device("cpu"),
        global_rank=0,
        worker_rank=0,
        local_rank=0,
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

        assert FakeClient.instances[-1].calls == [(REPO, snapshot)]
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

        assert FakeClient.instances[-1].calls == [(REPO, snapshot)]

    def test_installs_metadata_when_no_snapshot_exists(self, enabled, snapshot, fake_client):
        ctx = _make_context(REPO, model_path=None)

        with patch.object(model_prefetch, "ensure_metadata", return_value=snapshot) as ensure:
            with patch("modelexpress.load_strategy.server_cache_strategy.register_tensors"):
                ServerCacheStrategy().load(LoadResult(value=MagicMock()), ctx)

        assert ensure.call_count == 1
        assert FakeClient.instances[-1].calls == [(REPO, snapshot)]

    def test_partial_snapshot_is_prepared_before_weights(
        self, enabled, snapshot, fake_client, monkeypatch
    ):
        (snapshot / "config.json").unlink()
        root = snapshot.parent.parent.parent
        ctx = _make_context(REPO, model_path=str(snapshot), revision="moving-branch")
        events = []

        def install_metadata(repo_id, revision, cache_directory):
            assert repo_id == REPO
            assert revision == COMMIT
            assert cache_directory == root
            events.append("metadata")
            cache = ModelSnapshotCache(repo_id, cache_directory)
            with cache.lock():
                (snapshot / "config.json").write_bytes(b"{}")
                cache._write_metadata_inventory(COMMIT, {"config.json": 2})
            return snapshot

        original_weights = FakeClient.install_weight_files

        def install_weights(client, repo_id, path, *args, **kwargs):
            assert ModelSnapshotCache(repo_id, root)._ready_metadata(COMMIT) == snapshot
            events.append("weights")
            return original_weights(client, repo_id, path, *args, **kwargs)

        monkeypatch.setattr(model_prefetch, "_ensure_metadata_snapshot", install_metadata)
        monkeypatch.setattr(FakeClient, "install_weight_files", install_weights)
        with patch("modelexpress.load_strategy.server_cache_strategy.register_tensors"):
            ServerCacheStrategy().load(LoadResult(value=MagicMock()), ctx)

        assert events == ["metadata", "weights"]

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


class TestCacheRoot:
    """Weights have to land in the snapshot root the engine will read from.

    The engine loads from the path in ``ModelConfig``, which the frontend node
    resolved against its own cache. A worker that installs under its own
    default root instead produces a complete snapshot the engine never opens.
    """

    def _run(self, ctx):
        with patch("modelexpress.load_strategy.server_cache_strategy.register_tensors"):
            ServerCacheStrategy().load(LoadResult(value=MagicMock()), ctx)

    def test_existing_snapshot_pins_the_client_to_its_own_root(
        self, enabled, snapshot, fake_client, monkeypatch, tmp_path
    ):
        default_root = tmp_path / "worker-default"
        monkeypatch.setenv("MODEL_EXPRESS_CACHE_DIRECTORY", str(default_root))
        ctx = _make_context(REPO, model_path=str(snapshot))

        self._run(ctx)

        client = FakeClient.instances[-1]
        assert client.kwargs["cache_directory"] == snapshot.parent.parent.parent
        assert client.calls == [(REPO, snapshot)]
        assert all(
            client.kwargs["cache_directory"] == snapshot.parent.parent.parent
            for client in FakeClient.instances
        )
        assert not default_root.exists()

    def test_missing_snapshot_installs_under_the_paths_root(
        self, enabled, tmp_path, fake_client, monkeypatch
    ):
        """The multi-node case: EngineCore gets a path this node has nothing at."""
        monkeypatch.setenv("MODEL_EXPRESS_CACHE_DIRECTORY", str(tmp_path / "cache-a"))
        other_root = tmp_path / "cache-b"
        engine_path = other_root / "models--org--model" / "snapshots" / COMMIT
        ctx = _make_context(REPO, model_path=str(engine_path), revision=None)

        def install(repo_id, revision, cache_directory):
            assert cache_directory == other_root
            # The commit comes from the directory name, not from ModelConfig:
            # a local path leaves revision unresolved, and the server's default
            # would be a different snapshot than the one the engine reads.
            assert revision == COMMIT
            cache = ModelSnapshotCache(repo_id, cache_directory)
            with cache.lock():
                engine_path.mkdir(parents=True)
                (engine_path / "config.json").write_bytes(b"{}")
                cache._write_metadata_inventory(COMMIT, {"config.json": 2})
            return engine_path

        with patch.object(model_prefetch, "_ensure_metadata_snapshot", side_effect=install):
            self._run(ctx)

        client = FakeClient.instances[-1]
        assert client.kwargs["cache_directory"] == other_root
        assert client.calls == [(REPO, engine_path)]

    def test_metadata_failure_on_a_missing_snapshot_is_a_clean_miss(
        self, enabled, tmp_path, fake_client
    ):
        """An old server that will not confirm the commit must not install its own."""
        engine_path = tmp_path / "models--org--model" / "snapshots" / COMMIT
        ctx = _make_context(REPO, model_path=str(engine_path))

        with patch.object(
            model_prefetch,
            "_ensure_metadata_snapshot",
            side_effect=ModelCacheError("Server did not confirm revision"),
        ):
            with pytest.raises(StrategyFailed) as excinfo:
                self._run(ctx)

        assert excinfo.value.mutated is False
        assert "did not confirm revision" in str(excinfo.value)
        assert FakeClient.instances == []

    def test_disabled_resolved_preparation_is_a_clean_miss(
        self, enabled, tmp_path, fake_client
    ):
        engine_path = tmp_path / "models--org--model" / "snapshots" / COMMIT
        ctx = _make_context(REPO, model_path=str(engine_path))

        with patch.object(model_prefetch, "_ensure_resolved_metadata", return_value=None):
            with pytest.raises(StrategyFailed, match="did not apply") as excinfo:
                self._run(ctx)

        assert excinfo.value.mutated is False
        assert FakeClient.instances == []

    def test_a_path_outside_the_cache_layout_keeps_the_old_behaviour(
        self, enabled, tmp_path, fake_client
    ):
        plain = tmp_path / "checkpoints" / "model"
        plain.mkdir(parents=True)
        model_prefetch._snapshot_to_repo_id[str(plain)] = REPO
        ctx = _make_context(str(plain), model_path=str(plain))

        with patch.object(
            model_prefetch,
            "_ensure_resolved_metadata",
            side_effect=AssertionError("Ordinary local paths must not fetch metadata"),
        ):
            self._run(ctx)

        client = FakeClient.instances[-1]
        assert client.kwargs["cache_directory"] is None
        assert client.calls == [(REPO, plain)]

    def test_a_repo_id_keeps_the_old_behaviour(self, enabled, snapshot, fake_client):
        ctx = _make_context(REPO, model_path=None, revision="main")

        with patch.object(
            model_prefetch, "ensure_metadata", return_value=snapshot
        ) as ensure:
            self._run(ctx)

        assert ensure.call_args.kwargs["cache_directory"] is None
        assert ensure.call_args.args[1] == "main"
        assert FakeClient.instances[0].kwargs["cache_directory"] is None


class TestLegacySnapshotCompatibility:
    @pytest.mark.parametrize(
        "pin_rpc_error", [False, True], ids=["unconfirmed-revision", "pin-rpc-error"]
    )
    def test_existing_snapshot_without_inventory_keeps_the_weight_path(
        self, enabled, legacy_snapshot, monkeypatch, pin_rpc_error
    ):
        service = _use_legacy_service(monkeypatch, pin_rpc_error=pin_rpc_error)
        adapter = _FakeAdapter()
        ctx = _make_context(REPO, adapter=adapter, model_path=str(legacy_snapshot))

        with patch("modelexpress.load_strategy.server_cache_strategy.register_tensors"):
            ServerCacheStrategy().load(LoadResult(value=MagicMock()), ctx)

        assert (legacy_snapshot / "model.safetensors").read_bytes() == b"weights"
        assert (legacy_snapshot / "config.json").read_bytes() == b"{}"
        assert adapter.native_calls == 1
        assert adapter.post_calls == 1
        assert len(service.download_requests) == (2 if pin_rpc_error else 1)
        assert service.download_requests[0].revision == COMMIT
        assert all(not request.ignore_weights for request in service.download_requests)
        if pin_rpc_error:
            assert not service.download_requests[1].HasField("revision")
        assert not service.list_requests[0].HasField("revision")
        assert len(service.stream_requests) == 1
        assert not model_prefetch._revision_snapshots
        cache = ModelSnapshotCache(REPO, legacy_snapshot.parent.parent.parent)
        assert cache._ready_metadata(COMMIT) is None
        assert not cache._metadata_inventory_path(COMMIT).exists()

    @pytest.mark.parametrize("pin_rpc_error", [False, True])
    def test_legacy_weight_fallback_still_rejects_a_different_stream_commit(
        self, enabled, legacy_snapshot, monkeypatch, pin_rpc_error
    ):
        service = _use_legacy_service(
            monkeypatch, pin_rpc_error=pin_rpc_error, stream_commit="b" * 40
        )
        adapter = _FakeAdapter()
        ctx = _make_context(REPO, adapter=adapter, model_path=str(legacy_snapshot))

        with pytest.raises(StrategyFailed, match="refusing to mix revisions") as excinfo:
            ServerCacheStrategy().load(LoadResult(value=MagicMock()), ctx)

        assert excinfo.value.mutated is False
        assert adapter.native_calls == 0
        assert not (legacy_snapshot / "model.safetensors").exists()
        assert len(service.stream_requests) == 1
        assert all(not request.ignore_weights for request in service.download_requests)

    @pytest.mark.parametrize("pin_rpc_error", [False, True])
    @pytest.mark.parametrize(
        "state",
        ["missing-directory", "corrupt-inventory", "symlink-inventory", "missing-metadata"],
    )
    def test_missing_or_known_invalid_snapshot_cannot_use_legacy_directory_fallback(
        self, enabled, tmp_path, monkeypatch, pin_rpc_error, state
    ):
        cache = ModelSnapshotCache(REPO, tmp_path)
        snapshot = cache.snapshot_path(COMMIT)
        if state != "missing-directory":
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_bytes(b"{}")
            if state == "missing-metadata":
                with cache.lock():
                    cache._write_metadata_inventory(COMMIT, {"config.json": 2})
                (snapshot / "config.json").unlink()
            else:
                inventory = cache._metadata_inventory_path(COMMIT)
                inventory.parent.mkdir()
                if state == "corrupt-inventory":
                    inventory.write_text("{")
                else:
                    inventory.symlink_to("missing-inventory")
        service = _use_legacy_service(monkeypatch, pin_rpc_error=pin_rpc_error)
        adapter = _FakeAdapter()
        ctx = _make_context(REPO, adapter=adapter, model_path=str(snapshot))

        with pytest.raises(StrategyFailed) as excinfo:
            ServerCacheStrategy().load(LoadResult(value=MagicMock()), ctx)

        assert excinfo.value.mutated is False
        assert adapter.native_calls == 0
        assert len(service.download_requests) == 1
        assert service.download_requests[0].ignore_weights is True
        assert not service.list_requests
        assert not service.stream_requests
        assert not (snapshot / "model.safetensors").exists()


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
