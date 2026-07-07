# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SGLang cache artifact integration."""

from pathlib import Path
from types import SimpleNamespace

from modelexpress import p2p_pb2
from modelexpress.engines.sglang import artifacts


def test_sglang_torch_compile_artifact_identity_uses_sglang_criteria(monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_COMPILE_CONFIG_DIGEST", "compile-digest")
    monkeypatch.setattr(artifacts, "_sglang_version", lambda: "0.5.13")
    monkeypatch.setattr(artifacts, "_triton_version", lambda: "3.4.0")
    monkeypatch.setattr(artifacts, "_triton_key", lambda: "triton-key")
    monkeypatch.setattr(artifacts, "_gpu_arch", lambda device_id: f"sm90-{device_id}")
    ctx = SimpleNamespace(
        device_id=2,
        identity=p2p_pb2.SourceIdentity(
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
            model_name="test/model",
            backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            expert_parallel_size=1,
            dtype="bfloat16",
            quantization="fp8",
            revision="abc123",
            extra_parameters={"weight_only": "not-artifact"},
        ),
    )

    identity = artifacts._artifact_identity(
        ctx,
        p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )

    assert identity.model_name == "test/model"
    assert identity.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE
    assert identity.backend_framework == p2p_pb2.BACKEND_FRAMEWORK_SGLANG
    assert identity.tensor_parallel_size == 4
    assert identity.pipeline_parallel_size == 2
    assert identity.expert_parallel_size == 1
    assert identity.dtype == "bfloat16"
    assert identity.quantization == "fp8"
    assert identity.revision == "abc123"
    assert identity.backend_framework_version == "0.5.13"
    assert identity.triton_version == "3.4.0"
    assert identity.gpu_arch == "sm90-2"
    assert identity.compile_config_digest == "compile-digest"
    assert identity.extra_parameters["triton_key"] == "triton-key"
    assert "weight_only" not in identity.extra_parameters


def test_sglang_artifact_transfers_use_sglang_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "torchinductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton-cache"))
    monkeypatch.setenv("MX_ARTIFACT_BUNDLE_ROOT", str(tmp_path / "bundles"))
    monkeypatch.setattr(artifacts, "_sglang_version", lambda: "0.5.13")
    monkeypatch.setattr(artifacts, "_triton_key", lambda: "triton-key")
    monkeypatch.setattr(artifacts, "_gpu_arch", lambda device_id: "sm90")
    ctx = SimpleNamespace(
        worker_rank=1,
        worker_id="worker-a",
        device_id=0,
        identity=p2p_pb2.SourceIdentity(
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
            model_name="test/model",
            backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        ),
    )

    transfers = artifacts._sglang_artifact_transfers(ctx)

    assert [
        (
            transfer.name,
            identity.mx_source_type,
            identity.backend_framework,
            transfer.roots[0].source_root,
            transfer.bundle_root,
        )
        for transfer, identity in transfers[:2]
    ] == [
        (
            "torch_compile_cache",
            p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
            p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
            Path(tmp_path / "torchinductor"),
            Path(tmp_path / "bundles" / "rank-1" / "torch_compile_cache"),
        ),
        (
            "triton_cache",
            p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
            p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
            Path(tmp_path / "triton-cache"),
            Path(tmp_path / "bundles" / "rank-1" / "triton_cache"),
        ),
    ]
