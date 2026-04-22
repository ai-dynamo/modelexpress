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
from modelexpress.source_id import compute_mx_source_id


def _base_identity() -> p2p_pb2.SourceIdentity:
    return p2p_pb2.SourceIdentity(
        mx_version="0.3.0",
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
    assert compute_mx_source_id(_base_identity()) == "b0c2c67edeaefc20"


def test_pinned_hash_with_revision():
    pinned = _base_identity()
    pinned.revision = "abc123def4567890"
    assert compute_mx_source_id(pinned) == "40704b34e4b7deaa"
