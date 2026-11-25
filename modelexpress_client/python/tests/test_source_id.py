# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Python compute_mx_source_id helper.

The pinned ASCII and Unicode hashes have matching assertions in
modelexpress_server/src/p2p/source_identity.rs.
"""

from __future__ import annotations

import pytest

from modelexpress import p2p_pb2
from modelexpress.metadata.source_id import compute_mx_source_id


def _base_identity() -> p2p_pb2.SourceIdentity:
    return p2p_pb2.SourceIdentity(
        mx_version="0.8.0",
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


@pytest.mark.parametrize(
    ("field", "first", "second"),
    [
        ("model_name", "s3://weights/ModelA", "s3://weights/modela"),
        ("model_name", "oss://weights/Qwen/Model", "oss://weights/qwen/model"),
        ("model_name", "/models/ModelA", "/models/modela"),
        ("model_name", "模型/Café", "模型/café"),
        ("revision", "Release-A", "release-a"),
    ],
)
def test_case_sensitive_identifiers_have_distinct_source_ids(field, first, second):
    a = _base_identity()
    b = _base_identity()
    setattr(a, field, first)
    setattr(b, field, second)
    assert compute_mx_source_id(a) != compute_mx_source_id(b)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"path": "s3://weights/ModelA"}, {"path": "s3://weights/modela"}),
        ({"Option": "value"}, {"option": "value"}),
        ({"Foo": "a", "foo": "b"}, {"Foo": "a", "foo": "c"}),
    ],
)
def test_extra_parameters_preserve_case(first, second):
    a = _base_identity()
    b = _base_identity()
    a.extra_parameters.update(first)
    b.extra_parameters.update(second)
    assert compute_mx_source_id(a) != compute_mx_source_id(b)


def test_lowercase_identity_preserves_existing_source_id():
    identity = _base_identity()
    identity.model_name = "deepseek-ai/deepseek-v3"
    assert compute_mx_source_id(identity) == "e0d1bc7459cb4cde"


def test_compatibility_labels_are_case_insensitive():
    upper = _base_identity()
    upper.dtype = "BFLOAT16"
    upper.quantization = "FP8"
    lower = _base_identity()
    lower.quantization = "fp8"
    assert compute_mx_source_id(lower) == compute_mx_source_id(upper)


def test_different_tp_gives_different_id():
    tp4 = _base_identity()
    tp4.tensor_parallel_size = 4
    assert compute_mx_source_id(_base_identity()) != compute_mx_source_id(tp4)


def test_different_revision_gives_different_id():
    pinned = _base_identity()
    pinned.revision = "abc123def4567890"
    assert compute_mx_source_id(_base_identity()) != compute_mx_source_id(pinned)


def test_empty_artifact_fields_use_case_preserving_id():
    assert compute_mx_source_id(_base_identity()) == "c4fef23b039e398e"


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


def test_tilelang_cache_is_separate_artifact_source_type():
    deep_gemm = _base_identity()
    deep_gemm.mx_source_type = p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE

    tilelang = _base_identity()
    tilelang.mx_source_type = p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE

    assert compute_mx_source_id(deep_gemm) != compute_mx_source_id(tilelang)


def test_tvm_ffi_cache_is_separate_artifact_source_type():
    triton = _base_identity()
    triton.mx_source_type = p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE

    tvm_ffi = _base_identity()
    tvm_ffi.mx_source_type = p2p_pb2.MX_SOURCE_TYPE_TVM_FFI_CACHE

    assert compute_mx_source_id(triton) != compute_mx_source_id(tvm_ffi)


def test_cute_dsl_and_flashinfer_have_separate_artifact_source_types():
    cute_dsl = _base_identity()
    cute_dsl.mx_source_type = p2p_pb2.MX_SOURCE_TYPE_CUTE_DSL_CACHE

    flashinfer = _base_identity()
    flashinfer.mx_source_type = p2p_pb2.MX_SOURCE_TYPE_FLASHINFER_CACHE

    assert compute_mx_source_id(cute_dsl) != compute_mx_source_id(flashinfer)


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
    assert compute_mx_source_id(_base_identity()) == "c4fef23b039e398e"


def test_pinned_hash_with_revision():
    pinned = _base_identity()
    pinned.revision = "abc123def4567890"
    assert compute_mx_source_id(pinned) == "e53adec2a591e2d4"


def test_case_distinct_extra_parameters_are_order_independent():
    # Both Foo and foo survive regardless of insertion order. Matched by
    # source_identity.rs::test_python_cross_check_case_distinct_extra.
    a = _base_identity()
    a.extra_parameters["Foo"] = "a"
    a.extra_parameters["foo"] = "b"
    b = _base_identity()
    b.extra_parameters["foo"] = "b"
    b.extra_parameters["Foo"] = "a"
    assert compute_mx_source_id(a) == compute_mx_source_id(b)
    assert compute_mx_source_id(a) == "c5e4c884a17fb6d7"


def test_pinned_hash_with_unicode_identity():
    identity = _base_identity()
    identity.model_name = "模型/Café"
    identity.extra_parameters["标签"] = "Crème 𐐀"
    identity.compile_config_digest = "编译-Ä"
    # Cross-checked with Rust compute_mx_source_id, which emits UTF-8 JSON.
    assert compute_mx_source_id(identity) == "26b0bb79a5869efe"
