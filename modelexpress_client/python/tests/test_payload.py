# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the accelerator compatibility policy."""

import pytest

from modelexpress import p2p_pb2
from modelexpress.metadata.payload import accelerators_compatible

WEIGHTS = p2p_pb2.MX_SOURCE_TYPE_WEIGHTS
LORA = p2p_pb2.MX_SOURCE_TYPE_LORA

ARTIFACT_TYPES = [
    p2p_pb2.MX_SOURCE_TYPE_CUDA_GRAPH,
    p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
    p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE,
    p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE,
    p2p_pb2.MX_SOURCE_TYPE_CUTE_DSL_CACHE,
    p2p_pb2.MX_SOURCE_TYPE_FLASHINFER_CACHE,
]


# ---------------------------------------------------------------------------
# Backward compatibility: empty values are unknown and accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mx_source_type", [None, WEIGHTS, LORA, *ARTIFACT_TYPES])
def test_empty_source_is_compatible(mx_source_type):
    # Unknown (empty) accelerator on a legacy/rolling source is accepted for a
    # declared-unquantized identity. Quantization must be declared: an unknown
    # accelerator could belong to a different vendor, so a quantized identity
    # would fail closed here (see the empty-accelerator fail-closed tests).
    assert accelerators_compatible(
        "cuda", "", mx_source_type=mx_source_type, quantization="", dtype="bfloat16"
    )


@pytest.mark.parametrize("mx_source_type", [None, WEIGHTS, LORA, *ARTIFACT_TYPES])
def test_empty_target_is_compatible(mx_source_type):
    assert accelerators_compatible(
        "", "xpu", mx_source_type=mx_source_type, quantization="", dtype="bfloat16"
    )


# ---------------------------------------------------------------------------
# Exact match: always compatible regardless of source type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mx_source_type", [None, WEIGHTS, LORA, *ARTIFACT_TYPES])
def test_exact_match_is_compatible(mx_source_type):
    assert accelerators_compatible("cuda", "cuda", mx_source_type=mx_source_type)


def test_same_accelerator_lora_is_compatible():
    # Same-family LoRA rides the generic exact-match rule, not the hetero path.
    assert accelerators_compatible("cuda", "cuda", mx_source_type=LORA)


# ---------------------------------------------------------------------------
# Heterogeneous weights: cuda <-> xpu both directions, unquantized only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quantization", ["", "none", "None", " NONE "])
def test_weights_cuda_to_xpu_unquantized_allowed(quantization):
    assert accelerators_compatible(
        "cuda",
        "xpu",
        mx_source_type=WEIGHTS,
        quantization=quantization,
        dtype="bfloat16",
    )


@pytest.mark.parametrize("quantization", ["", "none"])
def test_weights_xpu_to_cuda_unquantized_allowed(quantization):
    assert accelerators_compatible(
        "xpu",
        "cuda",
        mx_source_type=WEIGHTS,
        quantization=quantization,
        dtype="float16",
    )


# ---------------------------------------------------------------------------
# Heterogeneous weights: quantized layouts are hardware-specific, rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quantization", ["fp8", "FP8", "awq", "gptq", "nvfp4", "modelopt_fp4"]
)
def test_weights_cross_family_quantized_rejected(quantization):
    # DeepSeek-V3 FP8 and friends: quantization field is set, dtype is bf16.
    assert not accelerators_compatible(
        "cuda",
        "xpu",
        mx_source_type=WEIGHTS,
        quantization=quantization,
        dtype="bfloat16",
    )
    assert not accelerators_compatible(
        "xpu",
        "cuda",
        mx_source_type=WEIGHTS,
        quantization=quantization,
        dtype="bfloat16",
    )


@pytest.mark.parametrize(
    "dtype", ["float8_e4m3fn", "float8_e5m2", "fp8", "nvfp4", "mxfp8", "mxfp4"]
)
def test_weights_cross_family_quantized_dtype_rejected(dtype):
    # Empty quantization but a quantized storage dtype must still be rejected:
    # covers backends that express quantized weights through dtype alone.
    assert not accelerators_compatible(
        "cuda", "xpu", mx_source_type=WEIGHTS, quantization="", dtype=dtype
    )


@pytest.mark.parametrize("known,unknown", [("cuda", ""), ("", "xpu")])
def test_weights_unknown_accelerator_quantized_fails_closed_when_not_deferred(
    known, unknown
):
    # On the authoritative post-fetch check (defer_unknown=False), an empty
    # (unknown) accelerator could belong to a different vendor whose kernels
    # expect a different quantized layout. Trusting "empty means accept" for a
    # quantized weight identity would silently corrupt inference, so quantized
    # weights fail closed unless the family is a verified match.
    assert not accelerators_compatible(
        known,
        unknown,
        mx_source_type=WEIGHTS,
        quantization="fp8",
        dtype="bfloat16",
        defer_unknown=False,
    )
    # Same hole via a quantized storage dtype with empty quantization.
    assert not accelerators_compatible(
        known,
        unknown,
        mx_source_type=WEIGHTS,
        quantization="",
        dtype="float8_e4m3fn",
        defer_unknown=False,
    )


@pytest.mark.parametrize("known,unknown", [("cuda", ""), ("", "xpu")])
def test_weights_unknown_accelerator_quantized_deferred_by_default(known, unknown):
    # On the pre-fetch path (default defer_unknown=True), an unknown
    # accelerator is deferred to the post-fetch check rather than rejected: the
    # lightweight ref may legitimately omit the accelerator (k8s-service
    # synthetic ref). The authoritative post-fetch check does the rejection.
    assert accelerators_compatible(
        known, unknown, mx_source_type=WEIGHTS, quantization="fp8", dtype="bfloat16"
    )


def test_weights_same_family_quantized_survives_empty_check():
    # The empty-accelerator fail-closed rule must not block the primary
    # same-family quantized path (identical post-processing on both ends).
    assert accelerators_compatible(
        "cuda", "cuda", mx_source_type=WEIGHTS, quantization="fp8", dtype="bfloat16"
    )


def test_weights_cross_family_empty_dtype_fails_closed():
    # Explicitly-unquantized only: an empty dtype is unknown, not a proof of
    # bf16/fp16, so a cross-family weight transfer with an unset dtype fails
    # closed even when quantization is empty.
    assert not accelerators_compatible(
        "cuda", "xpu", mx_source_type=WEIGHTS, quantization="", dtype=""
    )
    assert not accelerators_compatible(
        "xpu", "cuda", mx_source_type=WEIGHTS, quantization="none", dtype="  "
    )


def test_weights_cross_family_omitted_quantization_fails_closed():
    # A call site that does not declare the identity's quantization must never
    # get a cross-family weight transfer. Guards future callers from silently
    # reopening the quantized cross-vendor hole.
    assert not accelerators_compatible("cuda", "xpu", mx_source_type=WEIGHTS)
    assert not accelerators_compatible(
        "cuda", "xpu", mx_source_type=WEIGHTS, quantization=""
    )
    assert not accelerators_compatible(
        "cuda", "xpu", mx_source_type=WEIGHTS, dtype="bfloat16"
    )


# ---------------------------------------------------------------------------
# Same-family transfer is unconditional: quantization does not gate it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quantization", ["fp8", "awq", "nvfp4"])
def test_same_family_quantized_allowed(quantization):
    assert accelerators_compatible(
        "cuda",
        "cuda",
        mx_source_type=WEIGHTS,
        quantization=quantization,
        dtype="bfloat16",
    )


# ---------------------------------------------------------------------------
# Narrow boundary: only cuda/xpu, no arbitrary future backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target,source",
    [
        ("cuda", "rocm"),
        ("xpu", "rocm"),
        ("cuda", "neuron"),
        ("rocm", "xpu"),
    ],
)
def test_weights_unproven_pairs_rejected(target, source):
    assert not accelerators_compatible(
        target, source, mx_source_type=WEIGHTS, quantization="", dtype="bfloat16"
    )


# ---------------------------------------------------------------------------
# Fail-closed default: unspecified source type stays strict same-family
# ---------------------------------------------------------------------------


def test_unspecified_source_type_rejects_cross_family():
    assert not accelerators_compatible("cuda", "xpu")


# ---------------------------------------------------------------------------
# LoRA boundary: not enabled until a live LoRA transfer path exists
# ---------------------------------------------------------------------------


def test_lora_cross_accelerator_not_enabled_until_live_path_exists():
    assert not accelerators_compatible("cuda", "xpu", mx_source_type=LORA)


# ---------------------------------------------------------------------------
# Artifacts: accelerator/arch-specific, never cross families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mx_source_type", ARTIFACT_TYPES)
def test_artifacts_never_cross_family(mx_source_type):
    assert not accelerators_compatible("cuda", "xpu", mx_source_type=mx_source_type)
    assert not accelerators_compatible("xpu", "cuda", mx_source_type=mx_source_type)
