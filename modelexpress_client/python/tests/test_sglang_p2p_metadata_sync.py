# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SGLang P2P metadata sync plugin."""

from __future__ import annotations

from pathlib import Path
import socket
import sys
from types import ModuleType, SimpleNamespace

import pytest

from modelexpress import p2p_pb2
from modelexpress.engines.sglang import p2p_metadata_sync
from modelexpress.engines.sglang import registration as sglang_registration
from modelexpress.metadata.source_id import compute_mx_source_id


class _InMemoryMxStore:
    """In-memory ModelExpress P2P coordinator store for unit tests."""

    def __init__(self) -> None:
        self.records: dict[
            tuple[str, str],
            tuple[p2p_pb2.SourceIdentity, p2p_pb2.WorkerMetadata],
        ] = {}

    def make_client(self, worker_rank: int = 0, server_url: str | None = None):
        store = self

        class _Client:
            def publish_metadata(self, identity, worker, worker_id):
                mx_source_id = compute_mx_source_id(identity)
                copied = p2p_pb2.WorkerMetadata()
                copied.CopyFrom(worker)
                copied.status = p2p_pb2.SOURCE_STATUS_READY
                store.records[(mx_source_id, worker_id)] = (identity, copied)
                return mx_source_id

            def update_status(
                self, mx_source_id, worker_id, worker_rank, status, source_load=None
            ):
                key = (mx_source_id, worker_id)
                if key not in store.records:
                    return False
                identity, worker = store.records[key]
                worker.status = status
                store.records[key] = (identity, worker)
                return True

            def list_sources(self, identity=None, status_filter=None):
                target_source_id = (
                    compute_mx_source_id(identity)
                    if identity is not None and identity.model_name
                    else None
                )
                instances = []
                for (mx_source_id, worker_id), (rec_id, worker) in store.records.items():
                    if target_source_id and mx_source_id != target_source_id:
                        continue
                    if (
                        status_filter is not None
                        and worker.status != status_filter
                    ):
                        continue
                    instances.append(
                        p2p_pb2.SourceInstanceRef(
                            mx_source_id=mx_source_id,
                            worker_id=worker_id,
                            model_name=rec_id.model_name,
                            worker_rank=worker.worker_rank,
                            accelerator=worker.accelerator,
                        )
                    )
                return p2p_pb2.ListSourcesResponse(instances=instances)

            def get_metadata(self, mx_source_id, worker_id):
                key = (mx_source_id, worker_id)
                if key not in store.records:
                    return p2p_pb2.GetMetadataResponse(found=False)
                identity, worker = store.records[key]
                return p2p_pb2.GetMetadataResponse(
                    found=True,
                    worker=worker,
                    mx_source_id=mx_source_id,
                    worker_id=worker_id,
                    identity=identity,
                )

            def close(self):
                pass

        return _Client()


@pytest.fixture(autouse=True)
def reset_plugin_state(monkeypatch):
    monkeypatch.delenv("SGLANG_META_PORT", raising=False)
    monkeypatch.delenv("MX_DISABLE_PATCHES", raising=False)
    monkeypatch.delenv("MX_SGLANG_P2P_METADATA_SYNC", raising=False)
    monkeypatch.delenv("MX_SERVER_ADDRESS", raising=False)
    monkeypatch.delenv("MODEL_EXPRESS_URL", raising=False)
    monkeypatch.setenv("MX_WORKER_HOST", "127.0.0.1")
    monkeypatch.setattr(sglang_registration, "_registered", False)
    p2p_metadata_sync.reset_state_for_test()
    yield
    p2p_metadata_sync.reset_state_for_test()


def _create_source_model_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}')
    (root / "tokenizer_config.json").write_text('{"tokenizer_class": "LlamaTokenizer"}')
    (root / "tokenizer.json").write_text('{"version": "1.0"}')
    sub = root / "chat_template"
    sub.mkdir()
    (sub / "template.jinja").write_text("{{ messages }}")
    # Weight files that must be excluded from the metadata archive
    (root / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * 64)
    (root / "pytorch_model.bin").write_bytes(b"\x00" * 64)
    (root / "weights.pt").write_bytes(b"\x00" * 64)
    return root


def test_build_and_extract_metadata_tar_excludes_weights(tmp_path):
    source_dir = _create_source_model_dir(tmp_path / "source")
    target_dir = tmp_path / "target"

    tar_bytes = p2p_metadata_sync.build_metadata_tar_bytes(source_dir)
    p2p_metadata_sync.extract_metadata_tar_bytes(tar_bytes, target_dir)

    assert (target_dir / "config.json").read_text() == (
        '{"architectures": ["LlamaForCausalLM"]}'
    )
    assert (target_dir / "tokenizer_config.json").exists()
    assert (target_dir / "tokenizer.json").exists()
    assert (target_dir / "chat_template" / "template.jinja").read_text() == (
        "{{ messages }}"
    )
    assert not (target_dir / "model-00001-of-00002.safetensors").exists()
    assert not (target_dir / "pytorch_model.bin").exists()
    assert not (target_dir / "weights.pt").exists()


def test_source_and_target_end_to_end_p2p_discovery_and_target_starts_metaserver(
    monkeypatch, tmp_path
):
    store = _InMemoryMxStore()
    monkeypatch.setattr(
        p2p_metadata_sync, "create_metadata_client", store.make_client
    )
    monkeypatch.setattr(p2p_metadata_sync, "_get_worker_host", lambda: "127.0.0.1")
    monkeypatch.setenv("MX_SERVER_ADDRESS", "modelexpress-server:8001")
    monkeypatch.setenv("SGLANG_META_PORT", "0")

    source_dir = _create_source_model_dir(tmp_path / "source_pod" / "model")
    target_dir = tmp_path / "target_pod" / "model"

    # 1. Source Pod starts HTTP server and publishes to ModelExpress P2P service
    p2p_metadata_sync.sync_or_serve_metadata(str(source_dir))
    assert p2p_metadata_sync._http_server is not None
    source_server = p2p_metadata_sync._http_server
    bound_port = source_server.server_port
    assert bound_port > 0
    assert len(store.records) == 1

    # Simulate switching to a fresh Target Pod process while keeping Source Pod's
    # HTTP server alive in the background.
    for pub in list(p2p_metadata_sync._publisher_threads.values()):
        pub._stop_event.set()
    p2p_metadata_sync._publisher_threads.clear()
    p2p_metadata_sync._published_paths.clear()
    p2p_metadata_sync._http_server = None
    p2p_metadata_sync._server_started = False
    p2p_metadata_sync._worker_id = "meta-target01"

    # 2. Target Pod discovers Source Pod via ModelExpress P2P, pulls metadata,
    # and automatically starts its own metaserver + publishes itself!
    try:
        p2p_metadata_sync.sync_or_serve_metadata(str(target_dir))

        assert (target_dir / "config.json").read_text() == (
            '{"architectures": ["LlamaForCausalLM"]}'
        )
        assert (target_dir / "tokenizer_config.json").exists()
        assert (target_dir / "tokenizer.json").exists()
        assert (target_dir / "chat_template" / "template.jinja").exists()
        assert not (target_dir / "model-00001-of-00002.safetensors").exists()

        # Verify Target Pod automatically started its own metaserver and published
        assert p2p_metadata_sync._http_server is not None
        assert p2p_metadata_sync._http_server is not source_server
        assert p2p_metadata_sync._http_server.server_port > 0
        assert len(store.records) == 2
    finally:
        source_server.shutdown()
        source_server.server_close()


def test_target_discovers_source_from_weight_worker_metadata(monkeypatch, tmp_path):
    store = _InMemoryMxStore()
    monkeypatch.setattr(
        p2p_metadata_sync, "create_metadata_client", store.make_client
    )
    monkeypatch.setattr(p2p_metadata_sync, "_get_worker_host", lambda: "127.0.0.1")
    monkeypatch.setenv("MX_SERVER_ADDRESS", "modelexpress-server:8001")

    source_dir = _create_source_model_dir(tmp_path / "source" / "model")
    target_dir = tmp_path / "target" / "model"

    # Start Source Pod's HTTP server without publishing metadata sync identity,
    # and instead publish a standard P2P weight worker entry (with NIXL port 5555).
    source_server = p2p_metadata_sync.start_meta_server(str(source_dir), port=0)
    bound_meta_port = source_server.server_port
    store.records.clear()

    weight_identity = p2p_pb2.SourceIdentity(
        mx_version="0.4.0",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=str(source_dir),
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        tensor_parallel_size=8,
        dtype="bfloat16",
    )
    weight_worker = p2p_pb2.WorkerMetadata(
        worker_rank=0,
        metadata_endpoint="127.0.0.1:5555",
        worker_grpc_endpoint="127.0.0.1:6555",
        status=p2p_pb2.SOURCE_STATUS_READY,
    )
    store.make_client().publish_metadata(weight_identity, weight_worker, "weight-src-0")

    # Reset module state to simulate Target Pod where SGLANG_META_PORT matches Source's meta port
    p2p_metadata_sync._http_server = None
    p2p_metadata_sync._server_started = False
    p2p_metadata_sync._worker_id = "target-worker"
    monkeypatch.setenv("SGLANG_META_PORT", str(bound_meta_port))

    try:
        urls = p2p_metadata_sync.discover_metadata_source_urls(str(target_dir))
        assert urls == [f"http://127.0.0.1:{bound_meta_port}"]

        p2p_metadata_sync.sync_or_serve_metadata(str(target_dir))
        assert (target_dir / "config.json").exists()
    finally:
        source_server.shutdown()
        source_server.server_close()


def test_multi_path_source_serves_draft_and_main_models(monkeypatch, tmp_path):
    store = _InMemoryMxStore()
    monkeypatch.setattr(
        p2p_metadata_sync, "create_metadata_client", store.make_client
    )
    monkeypatch.setattr(p2p_metadata_sync, "_get_worker_host", lambda: "127.0.0.1")
    monkeypatch.setenv("MX_SERVER_ADDRESS", "modelexpress-server:8001")
    monkeypatch.setenv("SGLANG_META_PORT", "0")

    main_source = _create_source_model_dir(tmp_path / "source" / "main")
    draft_source = tmp_path / "source" / "draft"
    draft_source.mkdir(parents=True)
    (draft_source / "config.json").write_text('{"architectures": ["DraftModel"]}')
    (draft_source / "draft.safetensors").write_bytes(b"\x00" * 32)

    source_args = SimpleNamespace(
        model_path=str(main_source),
        tokenizer_path=str(main_source),
        speculative_draft_model_path=str(draft_source),
    )
    p2p_metadata_sync.sync_server_args_metadata(source_args)
    source_server = p2p_metadata_sync._http_server

    # Register target paths on the source map so path-based queries resolve
    main_target = tmp_path / "target" / "main"
    draft_target = tmp_path / "target" / "draft"
    p2p_metadata_sync._served_paths[str(main_target)] = str(main_source)
    p2p_metadata_sync._served_paths[str(draft_target)] = str(draft_source)

    # Simulate Target Pod process
    for pub in list(p2p_metadata_sync._publisher_threads.values()):
        pub._stop_event.set()
    p2p_metadata_sync._publisher_threads.clear()
    p2p_metadata_sync._published_paths.clear()
    p2p_metadata_sync._http_server = None
    p2p_metadata_sync._server_started = False
    p2p_metadata_sync._worker_id = "meta-target02"

    try:
        target_args = SimpleNamespace(
            model_path=str(main_target),
            tokenizer_path=None,
            speculative_draft_model_path=str(draft_target),
        )
        p2p_metadata_sync.before_server_args_post_init(target_args)

        assert (main_target / "config.json").read_text() == (
            '{"architectures": ["LlamaForCausalLM"]}'
        )
        assert (draft_target / "config.json").read_text() == (
            '{"architectures": ["DraftModel"]}'
        )
        assert not (draft_target / "draft.safetensors").exists()
        # Target Pod also started its own metaserver
        assert p2p_metadata_sync._http_server is not None
    finally:
        source_server.shutdown()
        source_server.server_close()


def test_start_meta_server_ignores_port_in_use_in_subprocesses(monkeypatch, tmp_path):
    source_dir = _create_source_model_dir(tmp_path / "source")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("0.0.0.0", 0))
        occupied.listen(1)
        busy_port = occupied.getsockname()[1]

        server = p2p_metadata_sync.start_meta_server(str(source_dir), port=busy_port)
        assert server is None
        assert p2p_metadata_sync._server_started is True


def test_sync_skips_dummy_and_uri_paths(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(
        p2p_metadata_sync,
        "discover_metadata_source_urls",
        lambda path, server_url=None: called.append(path) or [],
    )

    p2p_metadata_sync.sync_or_serve_metadata(None)
    p2p_metadata_sync.sync_or_serve_metadata("dummy")
    p2p_metadata_sync.sync_or_serve_metadata("none")
    p2p_metadata_sync.sync_or_serve_metadata("s3://bucket/model")
    assert called == []


def test_sglang_plugin_registration_registers_hooks(monkeypatch):
    hook_calls = []
    resolution_hooks = {}

    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.plugins",
        "sglang.srt.arg_groups",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    hooks_mod = ModuleType("sglang.srt.plugins.hook_registry")
    hooks_mod.HookType = SimpleNamespace(BEFORE="before")
    hooks_mod.HookRegistry = SimpleNamespace(
        register=lambda *args, **kwargs: hook_calls.append((args, kwargs))
    )
    monkeypatch.setitem(sys.modules, hooks_mod.__name__, hooks_mod)

    res_mod = ModuleType("sglang.srt.arg_groups.resolution_hooks")

    def fake_register_resolution_hook(name):
        def decorator(fn):
            resolution_hooks[name] = fn
            return fn

        return decorator

    res_mod.register_resolution_hook = fake_register_resolution_hook
    monkeypatch.setitem(sys.modules, res_mod.__name__, res_mod)

    # Disabled by default when MX_SGLANG_P2P_METADATA_SYNC is unset
    sglang_registration.register()
    assert sglang_registration._registered is False
    assert hook_calls == []

    monkeypatch.setenv("MX_SGLANG_P2P_METADATA_SYNC", "1")
    sglang_registration.register()
    # Second call should be a no-op
    sglang_registration.register()

    assert hook_calls == [
        (
            (
                "sglang.srt.server_args.ServerArgs.__post_init__",
                p2p_metadata_sync.before_server_args_post_init,
            ),
            {"hook_type": "before"},
        )
    ]
    assert "handle_model_source_paths" in resolution_hooks


def test_sglang_plugin_registration_respects_disable_patches(monkeypatch):
    monkeypatch.setenv("MX_SGLANG_P2P_METADATA_SYNC", "1")
    monkeypatch.setenv("MX_DISABLE_PATCHES", "1")
    sglang_registration.register()
    assert sglang_registration._registered is False
