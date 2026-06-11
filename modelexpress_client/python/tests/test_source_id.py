# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Python compute_mx_source_id helper.

The pinned hash values here are cross-checked by matching assertions in
modelexpress_server/src/source_identity.rs - if the canonical JSON
encoding or hashing scheme ever diverges between Python and Rust, both
sides' tests will fail together and catch it.
"""

from __future__ import annotations

from modelexpress import p2p_pb2
from modelexpress.metadata.source_id import compute_mx_source_id


def _base_identity() -> p2p_pb2.SourceIdentity:
    return p2p_pb2.SourceIdentity(
        mx_version="0.5.0",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name="deepseek-ai/DeepSeek-V3",
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel_size=0,
        dtype="bfloat16",
        quantization="",
        revision="",
    )


def test_id_is_16_hex_chars():
    sid = compute_mx_source_id(_base_identity())
    assert len(sid) == 16
    assert all(c in "0123456789abcdef" for c in sid)


def test_deterministic():
    assert compute_mx_source_id(_base_identity()) == compute_mx_source_id(_base_identity())


def test_case_insensitive():
    upper = _base_identity()
    upper.model_name = "DEEPSEEK-AI/DEEPSEEK-V3"
    upper.dtype = "BFLOAT16"
    assert compute_mx_source_id(_base_identity()) == compute_mx_source_id(upper)


def test_different_tp_gives_different_id():
    tp4 = _base_identity()
    tp4.tensor_parallel_size = 4
    assert compute_mx_source_id(_base_identity()) != compute_mx_source_id(tp4)


def test_different_revision_gives_different_id():
    pinned = _base_identity()
    pinned.revision = "abc123def4567890"
    assert compute_mx_source_id(_base_identity()) != compute_mx_source_id(pinned)


def test_empty_artifact_fields_preserve_existing_id():
    assert compute_mx_source_id(_base_identity()) == "5a5f555570065064"


def test_artifact_compatibility_fields_affect_id():
    artifact = _base_identity()
    artifact.mx_source_type = p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE
    artifact.backend_framework_version = "0.10.0"
    artifact.torch_version = "2.8.0+cu128"
    artifact.cuda_version = "12.8"
    artifact.triton_version = "3.4.0"
    artifact.gpu_arch = "SM90"
    artifact.compile_config_digest = "abc123"

    different_torch = p2p_pb2.SourceIdentity()
    different_torch.CopyFrom(artifact)
    different_torch.torch_version = "2.9.0+cu128"

    assert compute_mx_source_id(artifact) != compute_mx_source_id(different_torch)


def test_deep_gemm_cache_is_separate_artifact_source_type():
    torch_compile = _base_identity()
    torch_compile.mx_source_type = p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE
    torch_compile.gpu_arch = "SM90"
    torch_compile.compile_config_digest = "kernel-cache-a"

    deep_gemm = _base_identity()
    deep_gemm.mx_source_type = p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE
    deep_gemm.gpu_arch = "SM90"
    deep_gemm.compile_config_digest = "kernel-cache-a"

    assert compute_mx_source_id(torch_compile) != compute_mx_source_id(deep_gemm)


def test_artifact_compatibility_fields_are_case_insensitive():
    upper = _base_identity()
    upper.mx_source_type = p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE
    upper.backend_framework_version = "VLLM-0.10.0"
    upper.gpu_arch = "SM90"

    lower = _base_identity()
    lower.mx_source_type = p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE
    lower.backend_framework_version = "vllm-0.10.0"
    lower.gpu_arch = "sm90"

    assert compute_mx_source_id(upper) == compute_mx_source_id(lower)


def test_extra_parameters_sorted():
    a = _base_identity()
    a.extra_parameters["z_key"] = "val"
    a.extra_parameters["a_key"] = "val"
    b = _base_identity()
    b.extra_parameters["a_key"] = "val"
    b.extra_parameters["z_key"] = "val"
    assert compute_mx_source_id(a) == compute_mx_source_id(b)


# ---------------------------------------------------------------------------
# Pinned hashes (cross-checked by Rust; see source_identity.rs
# test_python_cross_check_*).
# ---------------------------------------------------------------------------

def test_pinned_hash_base_identity():
    assert compute_mx_source_id(_base_identity()) == "5a5f555570065064"


def test_pinned_hash_with_revision():
    pinned = _base_identity()
    pinned.revision = "abc123def4567890"
    assert compute_mx_source_id(pinned) == "d0c184b2a9a34c82"


def test_case_colliding_extra_parameters_are_deterministic():
    # Case-colliding keys (Foo vs foo) with different values. The
    # normalization rule is: sort original keys (ASCII order, so "Foo"
    # before "foo"), lowercase, keep the first value. "Foo"="a" survives
    # over "foo"="b" regardless of insertion order. Cross-checked against
    # source_identity.rs::test_python_cross_check_case_colliding_extra.
    a = _base_identity()
    a.extra_parameters["Foo"] = "a"
    a.extra_parameters["foo"] = "b"
    b = _base_identity()
    b.extra_parameters["foo"] = "b"
    b.extra_parameters["Foo"] = "a"
    assert compute_mx_source_id(a) == compute_mx_source_id(b)
    assert compute_mx_source_id(a) == "bf71fb9340cd940a"
