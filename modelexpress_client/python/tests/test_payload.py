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
    assert accelerators_compatible("cuda", "", mx_source_type=mx_source_type)


@pytest.mark.parametrize("mx_source_type", [None, WEIGHTS, LORA, *ARTIFACT_TYPES])
def test_empty_target_is_compatible(mx_source_type):
    assert accelerators_compatible("", "xpu", mx_source_type=mx_source_type)


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
# Heterogeneous weights: cuda <-> xpu both directions
# ---------------------------------------------------------------------------


def test_weights_cuda_to_xpu_allowed():
    assert accelerators_compatible("cuda", "xpu", mx_source_type=WEIGHTS)


def test_weights_xpu_to_cuda_allowed():
    assert accelerators_compatible("xpu", "cuda", mx_source_type=WEIGHTS)


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
    assert not accelerators_compatible(target, source, mx_source_type=WEIGHTS)


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
