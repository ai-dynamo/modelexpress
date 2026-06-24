# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for vLLM cache artifact integration."""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modelexpress import p2p_pb2
from modelexpress.engines.vllm import artifacts


def test_install_vllm_cache_artifacts_is_default_off(monkeypatch):
    monkeypatch.delenv("MX_ARTIFACT_TRANSFER", raising=False)

    with patch(
        "modelexpress.engines.vllm.artifacts.is_nixl_available",
    ) as is_nixl_available:
        artifacts.install_vllm_cache_artifacts(SimpleNamespace(global_rank=0))

    is_nixl_available.assert_not_called()


def test_install_vllm_cache_artifacts_warns_when_p2p_metadata_disabled(
    monkeypatch,
    caplog,
):
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    monkeypatch.setenv("MX_P2P_METADATA", "0")
    ctx = SimpleNamespace(global_rank=0, mx_client=object())

    with caplog.at_level(
        logging.WARNING,
        logger="modelexpress.engines.vllm.artifacts",
    ), patch(
        "modelexpress.engines.vllm.artifacts.is_nixl_available",
    ) as is_nixl_available:
        artifacts.install_vllm_cache_artifacts(ctx)

    is_nixl_available.assert_not_called()
    assert "MX_P2P_METADATA is disabled" in caplog.text


def test_install_vllm_cache_artifacts_skips_when_nixl_init_fails(monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    monkeypatch.setenv("MX_P2P_METADATA", "1")
    ctx = SimpleNamespace(global_rank=0, device_id=0, nixl_manager=None, mx_client=object())

    with patch(
        "modelexpress.engines.vllm.artifacts._metadata_publication_configured",
        return_value=True,
    ), patch(
        "modelexpress.engines.vllm.artifacts.is_nixl_available",
        return_value=True,
    ), patch(
        "modelexpress.engines.vllm.artifacts._init_nixl_manager",
        side_effect=RuntimeError("NIXL_ERR_BACKEND"),
    ), patch(
        "modelexpress.engines.vllm.artifacts._vllm_artifact_transfers",
    ) as transfers:
        artifacts.install_vllm_cache_artifacts(ctx)

    transfers.assert_not_called()


def test_torch_compile_artifact_identity_uses_model_cache_criteria(monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_COMPILE_CONFIG_DIGEST", "compile-digest")
    monkeypatch.setattr(artifacts, "_vllm_version", lambda: "0.17.1")
    monkeypatch.setattr(artifacts, "_triton_version", lambda: "3.4.0")
    monkeypatch.setattr(artifacts, "_triton_key", lambda: "triton-key")
    monkeypatch.setattr(artifacts, "_gpu_arch", lambda device_id: f"sm90-{device_id}")
    base_extra_parameters = {"weight_only": "not-artifact"}
    ctx = SimpleNamespace(
        device_id=2,
        identity=p2p_pb2.SourceIdentity(
            mx_version="0.5.0",
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
            model_name="test/model",
            backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
            tensor_parallel_size=4,
            dtype="bfloat16",
            revision="abc123",
            extra_parameters=base_extra_parameters,
        ),
    )

    identity = artifacts._artifact_identity(
        ctx,
        p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )

    assert identity.model_name == "test/model"
    assert identity.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE
    assert identity.mx_version == ""
    assert identity.tensor_parallel_size == 4
    assert identity.dtype == "bfloat16"
    assert identity.revision == "abc123"
    assert identity.backend_framework_version == "0.17.1"
    assert identity.triton_version == "3.4.0"
    assert identity.gpu_arch == "sm90-2"
    assert identity.compile_config_digest == "compile-digest"
    assert identity.extra_parameters["triton_key"] == "triton-key"
    assert "weight_only" not in identity.extra_parameters


def test_triton_artifact_identity_uses_runtime_cache_criteria(monkeypatch):
    monkeypatch.setattr(artifacts, "_triton_version", lambda: "3.4.0")
    monkeypatch.setattr(artifacts, "_triton_key", lambda: "triton-key")
    monkeypatch.setattr(artifacts, "_gpu_arch", lambda device_id: "sm90")
    monkeypatch.setattr(artifacts.torch.version, "cuda", "12.8")
    ctx = SimpleNamespace(
        device_id=0,
        identity=p2p_pb2.SourceIdentity(
            mx_version="0.5.0",
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
            model_name="test/model",
            tensor_parallel_size=8,
            dtype="bfloat16",
            revision="abc123",
            extra_parameters={"weight_only": "not-artifact"},
        ),
    )

    identity = artifacts._artifact_identity(ctx, p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE)

    assert identity.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE
    assert identity.backend_framework == p2p_pb2.BACKEND_FRAMEWORK_VLLM
    assert identity.cuda_version == "12.8"
    assert identity.triton_version == "3.4.0"
    assert identity.gpu_arch == "sm90"
    assert identity.extra_parameters["triton_key"] == "triton-key"
    assert identity.model_name == ""
    assert identity.tensor_parallel_size == 0
    assert identity.dtype == ""
    assert identity.revision == ""
    assert identity.backend_framework_version == ""
    assert identity.torch_version == ""
    assert "weight_only" not in identity.extra_parameters


def test_deep_gemm_artifact_identity_uses_deep_gemm_cache_criteria(monkeypatch):
    monkeypatch.setattr(artifacts, "_gpu_arch", lambda device_id: "sm90")
    monkeypatch.setattr(artifacts, "_deep_gemm_jit_key", lambda: "deep-gemm-key")
    monkeypatch.setattr(artifacts.torch.version, "cuda", "12.8")
    ctx = SimpleNamespace(
        device_id=0,
        identity=p2p_pb2.SourceIdentity(
            mx_version="0.5.0",
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
            model_name="test/model",
            backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
            tensor_parallel_size=8,
            dtype="bfloat16",
            revision="abc123",
            extra_parameters={"weight_only": "not-artifact"},
        ),
    )

    identity = artifacts._artifact_identity(
        ctx,
        p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE,
    )

    assert identity.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE
    assert identity.backend_framework == p2p_pb2.BACKEND_FRAMEWORK_VLLM
    assert identity.cuda_version == "12.8"
    assert identity.gpu_arch == "sm90"
    assert identity.extra_parameters["deep_gemm_jit_key"] == "deep-gemm-key"
    assert identity.model_name == ""
    assert identity.tensor_parallel_size == 0
    assert identity.dtype == ""
    assert identity.revision == ""
    assert identity.backend_framework_version == ""
    assert identity.torch_version == ""
    assert identity.triton_version == ""
    assert "weight_only" not in identity.extra_parameters


def test_triton_artifact_identity_omits_internal_key_when_unavailable(monkeypatch):
    monkeypatch.setattr(artifacts, "_triton_version", lambda: "3.4.0")
    monkeypatch.setattr(artifacts, "_triton_key", lambda: "")
    monkeypatch.setattr(artifacts, "_gpu_arch", lambda device_id: "sm90")
    ctx = SimpleNamespace(
        device_id=0,
        identity=p2p_pb2.SourceIdentity(),
    )

    identity = artifacts._artifact_identity(ctx, p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE)

    assert identity.triton_version == "3.4.0"
    assert "triton_key" not in identity.extra_parameters


def test_deep_gemm_artifact_identity_omits_jit_key_when_unavailable(monkeypatch):
    monkeypatch.setattr(artifacts, "_gpu_arch", lambda device_id: "sm90")
    monkeypatch.setattr(artifacts, "_deep_gemm_jit_key", lambda: "")
    ctx = SimpleNamespace(
        device_id=0,
        identity=p2p_pb2.SourceIdentity(),
    )

    identity = artifacts._artifact_identity(
        ctx,
        p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE,
    )

    assert "deep_gemm_jit_key" not in identity.extra_parameters


def test_vllm_artifact_transfers_use_distinct_cache_source_types(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm-cache"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton-cache"))
    monkeypatch.delenv("DG_JIT_CACHE_DIR", raising=False)
    monkeypatch.delenv("DEEP_GEMM_CACHE_DIR", raising=False)
    monkeypatch.setenv("MX_ARTIFACT_BUNDLE_ROOT", str(tmp_path / "bundles"))
    monkeypatch.setattr(artifacts, "_vllm_version", lambda: "0.17.1")
    monkeypatch.setattr(artifacts, "_triton_key", lambda: "triton-key")
    monkeypatch.setattr(artifacts, "_gpu_arch", lambda device_id: f"sm90-{device_id}")
    ctx = SimpleNamespace(
        worker_rank=1,
        worker_id="worker-a",
        device_id=0,
        identity=p2p_pb2.SourceIdentity(
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
            model_name="test/model",
        ),
    )

    transfers = artifacts._vllm_artifact_transfers(ctx)

    assert [
        (transfer.name, identity.mx_source_type, transfer.source_root, transfer.bundle_root)
        for transfer, identity in transfers
    ] == [
        (
            "torch_compile_cache",
            p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
            Path(tmp_path / "vllm-cache" / "torch_compile_cache"),
            Path(tmp_path / "bundles" / "rank-1" / "torch_compile_cache"),
        ),
        (
            "triton_cache",
            p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
            Path(tmp_path / "triton-cache"),
            Path(tmp_path / "bundles" / "rank-1" / "triton_cache"),
        ),
        (
            "deep_gemm_cache",
            p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE,
            Path(tmp_path / "vllm-cache" / "deep_gemm"),
            Path(tmp_path / "bundles" / "rank-1" / "deep_gemm_cache"),
        ),
    ]


def test_deep_gemm_cache_root_honors_dg_jit_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm-cache"))
    monkeypatch.setenv("DEEP_GEMM_CACHE_DIR", str(tmp_path / "legacy-cache"))
    monkeypatch.setenv("DG_JIT_CACHE_DIR", str(tmp_path / "deep-gemm-cache"))

    assert artifacts._deep_gemm_cache_root() == tmp_path / "deep-gemm-cache"


def test_publish_vllm_cache_artifact_uses_ephemeral_worker_port(tmp_path):
    source_root = tmp_path / "cache"
    source_root.mkdir()
    (source_root / "kernel.bin").write_bytes(b"compiled")
    transfer = SimpleNamespace(
        name="torch_compile_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        source_root=source_root,
        prepare_source=MagicMock(
            return_value=SimpleNamespace(
                artifact_id="artifact-id",
                manifest=p2p_pb2.ArtifactManifest(
                    files=[p2p_pb2.ArtifactManifestFile(size=8)],
                ),
            )
        ),
    )
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        model_name="test/model",
    )
    ctx = SimpleNamespace(
        global_rank=0,
        device_id=0,
        worker_rank=1,
        worker_id="worker-a",
        mx_client=object(),
        nixl_manager=object(),
    )
    published = SimpleNamespace(endpoint=SimpleNamespace(mx_source_id="source-id"))
    worker_server = object()

    with patch(
        "modelexpress.engines.vllm.artifacts._get_worker_server",
        return_value=worker_server,
    ), patch(
        "modelexpress.engines.vllm.artifacts.publish_artifact_source",
        return_value=published,
    ) as publish:
        assert artifacts._publish_vllm_cache_artifact(ctx, transfer, identity) is published

    publish.assert_called_once()
    assert publish.call_args.kwargs["worker_id"] == "worker-a"
    assert publish.call_args.kwargs["worker_grpc_server"] is worker_server
    artifacts._published_sources.pop(
        (ctx.device_id, transfer.mx_source_type),
        None,
    )


def test_install_vllm_cache_artifact_once_skips_after_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts.tempfile, "gettempdir", lambda: str(tmp_path))
    target_root = tmp_path / "cache"
    transfer = SimpleNamespace(
        name="deep_gemm_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE,
        source_root=target_root,
        target_root=target_root,
        discover_and_transfer=MagicMock(
            return_value=p2p_pb2.GetArtifactManifestHeaderResponse(
                artifact_id="artifact-id",
                total_size=8,
            )
        ),
        install=MagicMock(),
    )
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE,
        model_name="test/model",
    )
    ctx = SimpleNamespace(mx_client=object(), nixl_manager=object())

    first = artifacts._install_vllm_cache_artifact_once(ctx, transfer, identity)
    second = artifacts._install_vllm_cache_artifact_once(ctx, transfer, identity)

    assert first is not None
    assert second is None
    transfer.discover_and_transfer.assert_called_once_with(
        ctx.mx_client,
        identity,
        ctx.nixl_manager,
        worker_rank=None,
    )
    transfer.install.assert_called_once_with(first)


def test_schedule_vllm_cache_artifact_publish_starts_readiness_gated_publisher(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    monkeypatch.setenv("MX_P2P_METADATA", "1")
    monkeypatch.setattr(artifacts.tempfile, "gettempdir", lambda: str(tmp_path))
    source_root = tmp_path / "torch-cache"
    transfer = SimpleNamespace(
        name="torch_compile_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        source_root=source_root,
        target_root=source_root,
    )
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        model_name="test/model",
    )
    ctx = SimpleNamespace(
        global_rank=0,
        worker_rank=1,
        worker_id="worker-a",
        device_id=0,
        mx_client=object(),
        nixl_manager=object(),
    )
    other_ctx = SimpleNamespace(
        global_rank=1,
        worker_rank=2,
        worker_id="worker-b",
        device_id=1,
        mx_client=object(),
        nixl_manager=object(),
    )
    publisher = MagicMock()
    with patch(
        "modelexpress.engines.vllm.artifacts._metadata_publication_configured",
        return_value=True,
    ), patch(
        "modelexpress.engines.vllm.artifacts._vllm_artifact_transfers",
        return_value=[(transfer, identity)],
    ), patch(
        "modelexpress.engines.vllm.artifacts._publish_vllm_cache_artifact",
        return_value=SimpleNamespace(endpoint=SimpleNamespace(mx_source_id="source-id")),
    ) as publish_one, patch(
        "modelexpress.engines.vllm.artifacts.PublisherThread",
        return_value=publisher,
    ) as publisher_cls:
        artifacts.schedule_vllm_cache_artifact_publish(ctx)
        artifacts.schedule_vllm_cache_artifact_publish(other_ctx)
        publisher_cls.assert_called_once()
        kwargs = publisher_cls.call_args.kwargs
        assert kwargs["mx_client"] is ctx.mx_client
        assert kwargs["worker_id"] == "worker-a"
        assert kwargs["worker_rank"] == 1
        assert kwargs["nixl_manager"] is ctx.nixl_manager
        assert kwargs["heartbeat_after_publish"] is False
        assert kwargs["ready_fn"] is not artifacts._vllm_health_ready
        assert kwargs["publish_fn"]() == "source-id"
        publish_one.assert_called_once_with(ctx, transfer, identity)
        publisher.start.assert_called_once()
        publisher.mx_source_id = None
        kwargs["cleanup_fn"]()
        artifacts.schedule_vllm_cache_artifact_publish(other_ctx)
        assert publisher_cls.call_count == 2
    artifacts._scheduled_publishers.clear()


def test_vllm_artifact_ready_fn_waits_for_health_and_stable_cache(
    monkeypatch,
    tmp_path,
):
    cache_root = tmp_path / "torch_compile_cache"
    ready = artifacts._vllm_artifact_ready_fn(cache_root)
    health_ready = False
    now = 100.0
    monkeypatch.setattr(artifacts, "_vllm_health_ready", lambda: health_ready)
    monkeypatch.setattr(artifacts.time, "monotonic", lambda: now)

    assert ready() is False

    health_ready = True
    assert ready() is False

    cache_root.mkdir()
    cache_file = cache_root / "compiled.so"
    cache_file.write_bytes(b"compiled")
    assert ready() is False

    now += artifacts._CACHE_SETTLE_SECS - 1
    assert ready() is False

    now += 1
    assert ready() is True

    cache_file.write_bytes(b"compiled-again")
    assert ready() is False


def test_vllm_health_url_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("MX_ARTIFACT_READY_URL", raising=False)
    monkeypatch.delenv("HOSTNAME", raising=False)
    monkeypatch.delenv("POD_NAMESPACE", raising=False)

    assert artifacts._vllm_health_url() == "http://127.0.0.1:8000/health"


def test_vllm_health_url_honors_non_default_config(monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_READY_URL", "http://vllm-head:8000/health")
    monkeypatch.setenv("HOSTNAME", "mx-vllm-1")
    monkeypatch.setenv("POD_NAMESPACE", "test-ns")

    assert artifacts._vllm_health_url() == "http://vllm-head:8000/health"


def test_vllm_health_url_rejects_non_http_config(monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_READY_URL", "file:///tmp/health")
    monkeypatch.delenv("HOSTNAME", raising=False)

    assert artifacts._vllm_health_url() == "http://127.0.0.1:8000/health"


def test_vllm_health_url_uses_statefulset_head_from_worker_pod(monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_READY_URL", "http://127.0.0.1:8000/health")
    monkeypatch.setenv("HOSTNAME", "mx-vllm-1")
    monkeypatch.setenv("POD_NAMESPACE", "test-ns")

    assert artifacts._vllm_health_url() == "http://mx-vllm-0.mx-vllm.test-ns.svc:8000/health"
