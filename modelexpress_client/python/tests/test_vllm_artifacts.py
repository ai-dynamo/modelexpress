# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for vLLM cache artifact integration."""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modelexpress import p2p_pb2
from modelexpress.engines.vllm import artifacts
from modelexpress.metadata.artifact_transfer import ArtifactCacheRoot


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
    assert identity.model_name == "test/model"
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
    assert identity.model_name == "test/model"
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
        identity=p2p_pb2.SourceIdentity(model_name="test/model"),
    )

    identity = artifacts._artifact_identity(ctx, p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE)

    assert identity.triton_version == "3.4.0"
    assert "triton_key" not in identity.extra_parameters


def test_deep_gemm_artifact_identity_omits_jit_key_when_unavailable(monkeypatch):
    monkeypatch.setattr(artifacts, "_gpu_arch", lambda device_id: "sm90")
    monkeypatch.setattr(artifacts, "_deep_gemm_jit_key", lambda: "")
    ctx = SimpleNamespace(
        device_id=0,
        identity=p2p_pb2.SourceIdentity(model_name="test/model"),
    )

    identity = artifacts._artifact_identity(
        ctx,
        p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE,
    )

    assert "deep_gemm_jit_key" not in identity.extra_parameters


def test_tilelang_artifact_identity_uses_tilelang_cache_criteria(monkeypatch):
    monkeypatch.setattr(artifacts, "_gpu_arch", lambda device_id: "sm100")
    monkeypatch.setattr(artifacts, "_tilelang_version", lambda: "0.1.11")
    monkeypatch.setattr(artifacts.torch.version, "cuda", "13.0")
    ctx = SimpleNamespace(
        device_id=0,
        identity=p2p_pb2.SourceIdentity(
            model_name="deepseek-ai/DeepSeek-V4-Pro",
            tensor_parallel_size=8,
            revision="abc123",
        ),
    )

    identity = artifacts._artifact_identity(
        ctx,
        p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE,
    )

    assert identity.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE
    assert identity.model_name == "deepseek-ai/DeepSeek-V4-Pro"
    assert identity.backend_framework == p2p_pb2.BACKEND_FRAMEWORK_VLLM
    assert identity.cuda_version == "13.0"
    assert identity.gpu_arch == "sm100"
    assert identity.extra_parameters["tilelang_version"] == "0.1.11"
    assert identity.tensor_parallel_size == 0
    assert identity.revision == ""


def test_cute_dsl_artifact_identity_uses_compiler_criteria(monkeypatch):
    monkeypatch.setattr(artifacts, "_cutlass_dsl_version", lambda: "4.5.2")
    monkeypatch.setattr(artifacts, "_gpu_arch", lambda device_id: "sm100")
    monkeypatch.setattr(artifacts.torch.version, "cuda", "13.0")
    ctx = SimpleNamespace(
        device_id=0,
        identity=p2p_pb2.SourceIdentity(model_name="test/model"),
    )

    identity = artifacts._artifact_identity(
        ctx,
        p2p_pb2.MX_SOURCE_TYPE_CUTE_DSL_CACHE,
    )

    assert identity.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_CUTE_DSL_CACHE
    assert identity.model_name == "test/model"
    assert identity.cuda_version == "13.0"
    assert identity.gpu_arch == "sm100"
    assert identity.extra_parameters["cutlass_dsl_version"] == "4.5.2"
    assert identity.torch_version == ""


def test_flashinfer_artifact_identity_uses_runtime_criteria(monkeypatch):
    monkeypatch.setattr(artifacts, "_flashinfer_version", lambda: "0.6.12")
    monkeypatch.setattr(artifacts, "_gpu_arch", lambda device_id: "sm100")
    monkeypatch.setattr(artifacts.torch.version, "cuda", "13.0")
    ctx = SimpleNamespace(
        device_id=0,
        identity=p2p_pb2.SourceIdentity(model_name="test/model"),
    )

    identity = artifacts._artifact_identity(
        ctx,
        p2p_pb2.MX_SOURCE_TYPE_FLASHINFER_CACHE,
    )

    assert identity.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_FLASHINFER_CACHE
    assert identity.model_name == "test/model"
    assert identity.torch_version == artifacts.torch.__version__
    assert identity.cuda_version == "13.0"
    assert identity.gpu_arch == "sm100"
    assert identity.extra_parameters["flashinfer_version"] == "0.6.12"


def test_vllm_artifact_transfers_use_distinct_cache_source_types(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm-cache"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton-cache"))
    monkeypatch.delenv("DG_JIT_CACHE_DIR", raising=False)
    monkeypatch.delenv("DEEP_GEMM_CACHE_DIR", raising=False)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "tilelang-cache"))
    monkeypatch.setenv("CUTE_DSL_CACHE_DIR", str(tmp_path / "cute-dsl-cache"))
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(tmp_path / "flashinfer"))
    monkeypatch.setenv(
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
        str(tmp_path / "flashinfer-autotune-cache"),
    )
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
        (
            transfer.name,
            identity.mx_source_type,
            transfer.roots[0].source_root,
            transfer.bundle_root,
        )
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
        (
            "tilelang_cache",
            p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE,
            Path(tmp_path / "tilelang-cache"),
            Path(tmp_path / "bundles" / "rank-1" / "tilelang_cache"),
        ),
        (
            "cute_dsl_cache",
            p2p_pb2.MX_SOURCE_TYPE_CUTE_DSL_CACHE,
            Path(tmp_path / "cute-dsl-cache"),
            Path(tmp_path / "bundles" / "rank-1" / "cute_dsl_cache"),
        ),
        (
            "flashinfer_cache",
            p2p_pb2.MX_SOURCE_TYPE_FLASHINFER_CACHE,
            Path(tmp_path / "flashinfer" / ".cache" / "flashinfer"),
            Path(tmp_path / "bundles" / "rank-1" / "flashinfer_cache"),
        ),
    ]
    flashinfer_transfer = transfers[-1][0]
    assert tuple(root.source_root for root in flashinfer_transfer.roots) == (
        Path(tmp_path / "flashinfer" / ".cache" / "flashinfer"),
        Path(tmp_path / "flashinfer-autotune-cache"),
    )


def test_deep_gemm_cache_root_honors_dg_jit_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm-cache"))
    monkeypatch.setenv("DEEP_GEMM_CACHE_DIR", str(tmp_path / "legacy-cache"))
    monkeypatch.setenv("DG_JIT_CACHE_DIR", str(tmp_path / "deep-gemm-cache"))

    assert artifacts._deep_gemm_cache_root() == tmp_path / "deep-gemm-cache"


def test_tilelang_cache_root_uses_tilelang_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "tilelang-cache"))

    assert artifacts._tilelang_cache_root() == tmp_path / "tilelang-cache"


def test_cute_dsl_cache_root_uses_cute_dsl_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CUTE_DSL_CACHE_DIR", str(tmp_path / "cute-dsl-cache"))

    assert artifacts._cute_dsl_cache_root() == tmp_path / "cute-dsl-cache"


@pytest.mark.parametrize("error", [KeyError(), OSError()])
def test_cute_dsl_cache_root_falls_back_to_uid(monkeypatch, error):
    monkeypatch.delenv("CUTE_DSL_CACHE_DIR", raising=False)
    monkeypatch.setattr(artifacts, "getuser", MagicMock(side_effect=error))
    monkeypatch.setattr(artifacts.os, "getuid", lambda: 12345)

    assert artifacts._cute_dsl_cache_root() == (
        Path(artifacts.tempfile.gettempdir()) / "12345" / "cutlass_python_cache"
    )


def test_flashinfer_cache_root_uses_workspace_base(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(tmp_path / "flashinfer"))

    assert artifacts._flashinfer_cache_root() == (
        tmp_path / "flashinfer" / ".cache" / "flashinfer"
    )


def test_flashinfer_autotune_cache_root_uses_override(monkeypatch, tmp_path):
    configured = tmp_path / "configured-autotune-cache"
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(configured))

    assert artifacts._flashinfer_autotune_cache_root() == configured


def test_flashinfer_autotune_cache_root_uses_vllm_cache_root(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", raising=False)
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm-cache"))

    assert artifacts._flashinfer_autotune_cache_root() == (
        tmp_path / "vllm-cache" / "flashinfer_autotune_cache"
    )
    assert artifacts.envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR is None


def test_publish_vllm_cache_artifact_uses_ephemeral_worker_port(tmp_path):
    source_root = tmp_path / "cache"
    source_root.mkdir()
    (source_root / "kernel.bin").write_bytes(b"compiled")
    transfer = SimpleNamespace(
        name="torch_compile_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        roots=(
            ArtifactCacheRoot(
                name="primary",
                source_root=source_root,
                target_root=source_root,
            ),
        ),
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
        node_rank=2,
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
    assert publish.call_args.kwargs["node_rank"] == 2
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
        roots=(
            ArtifactCacheRoot(
                name="primary",
                source_root=target_root,
                target_root=target_root,
            ),
        ),
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
    ctx = SimpleNamespace(mx_client=object(), nixl_manager=object(), node_rank=2)

    first = artifacts._install_vllm_cache_artifact_once(ctx, transfer, identity)
    second = artifacts._install_vllm_cache_artifact_once(ctx, transfer, identity)

    assert first is not None
    assert second is None
    transfer.discover_and_transfer.assert_called_once_with(
        ctx.mx_client,
        identity,
        ctx.nixl_manager,
        worker_rank=None,
        node_rank=2,
    )
    transfer.install.assert_called_once_with(first)


def test_install_vllm_cache_artifact_once_does_not_retry_after_failure(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(artifacts.tempfile, "gettempdir", lambda: str(tmp_path))
    transfer = SimpleNamespace(
        name="triton_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        roots=(
            ArtifactCacheRoot(
                name="primary",
                source_root=tmp_path / "cache",
                target_root=tmp_path / "cache",
            ),
        ),
        discover_and_transfer=MagicMock(side_effect=RuntimeError("transfer failed")),
        install=MagicMock(),
    )
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        model_name="test/model",
    )
    ctx = SimpleNamespace(mx_client=object(), nixl_manager=object(), node_rank=2)

    with pytest.raises(RuntimeError, match="transfer failed"):
        artifacts._install_vllm_cache_artifact_once(ctx, transfer, identity)

    assert artifacts._install_vllm_cache_artifact_once(ctx, transfer, identity) is None
    transfer.discover_and_transfer.assert_called_once()
    transfer.install.assert_not_called()


def test_schedule_vllm_cache_artifact_publish_starts_readiness_gated_publisher(
    monkeypatch,
    tmp_path,
    caplog,
):
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    monkeypatch.setenv("MX_P2P_METADATA", "1")
    monkeypatch.setattr(artifacts.tempfile, "gettempdir", lambda: str(tmp_path))
    source_root = tmp_path / "torch-cache"
    autotune_root = tmp_path / "autotune-cache"
    transfer = SimpleNamespace(
        name="torch_compile_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        roots=(
            ArtifactCacheRoot(
                name="primary",
                source_root=source_root,
                target_root=source_root,
            ),
            ArtifactCacheRoot(
                name="autotune",
                source_root=autotune_root,
                target_root=autotune_root,
            ),
        ),
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
        with caplog.at_level(
            logging.INFO,
            logger="modelexpress.engines.vllm.artifacts",
        ):
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

    assert f"roots=['{source_root}', '{autotune_root}']" in caplog.text
    artifacts._scheduled_publishers.clear()


def test_vllm_artifact_ready_fn_waits_for_health_and_stable_cache(
    monkeypatch,
    tmp_path,
):
    cache_root = tmp_path / "torch_compile_cache"
    autotune_root = tmp_path / "autotune-cache"
    ready = artifacts._vllm_artifact_ready_fn((cache_root, autotune_root))
    health_check = MagicMock(side_effect=[False, True])
    now = 100.0
    monkeypatch.setattr(artifacts, "_vllm_health_ready", health_check)
    monkeypatch.setattr(artifacts.time, "monotonic", lambda: now)

    assert ready() is False

    assert ready() is False

    cache_root.mkdir()
    cache_file = cache_root / "compiled.so"
    cache_file.write_bytes(b"compiled")
    assert ready() is False

    now += artifacts._CACHE_SETTLE_SECS - 1
    assert ready() is False

    now += 1
    assert ready() is True

    autotune_root.mkdir()
    (autotune_root / "configs.json").write_text("{}")
    assert ready() is False

    now += artifacts._CACHE_SETTLE_SECS
    assert ready() is True

    cache_file.write_bytes(b"compiled-again")
    assert ready() is False
    assert health_check.call_count == 2


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
