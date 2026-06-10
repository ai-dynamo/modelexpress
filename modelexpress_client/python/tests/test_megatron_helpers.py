# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI cross-check for the vendored Megatron-Bridge helpers.

The functions in :mod:`modelexpress.megatron_helpers` are byte-identical
ports of standalone helpers in
``megatron.bridge.models.conversion.param_mapping``. This test module:

1. Exercises the vendored helpers' MHA + GQA round-trips standalone (no
   Bridge installed) — covers the Phase C receiver path's correctness.
2. When ``megatron-bridge`` IS importable in the test environment, also
   asserts byte-identity against Bridge's authoritative implementation.
   This is the cross-check that catches drift between our vendored copy
   and Bridge upstream — runs in CI when Bridge is part of the test deps.

Bridge is a *test-time-only* dependency. The runtime receiver does not
import Bridge.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


# Load megatron_helpers as a standalone module so the test runs without
# triggering the rest of the modelexpress package init (which pulls in NIXL).
_PKG_ROOT = Path(__file__).resolve().parent.parent / "modelexpress"


def _load_helpers():
    spec = importlib.util.spec_from_file_location(
        "modelexpress_megatron_helpers",
        _PKG_ROOT / "megatron_helpers.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modelexpress_megatron_helpers"] = mod
    spec.loader.exec_module(mod)
    return mod


HELPERS = _load_helpers()
MTC = HELPERS.MegatronTransformerConfig

# Try importing Bridge for the optional cross-check.
try:
    from megatron.bridge.models.conversion.param_mapping import (
        merge_qkv_weights as bridge_merge_qkv_weights,
        split_qkv_weights as bridge_split_qkv_weights,
        merge_qkv_biases as bridge_merge_qkv_biases,
        split_qkv_biases as bridge_split_qkv_biases,
    )
    BRIDGE_AVAILABLE = True
except Exception:
    BRIDGE_AVAILABLE = False


def _bridge_config(num_attention_heads, num_query_groups, kv_channels, hidden_size):
    """Bridge expects a TransformerConfig-shaped object with these attrs."""
    from types import SimpleNamespace
    return SimpleNamespace(
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        kv_channels=kv_channels,
        hidden_size=hidden_size,
    )


# ---------------------------------------------------------------------------
# Standalone correctness: round-trip MHA + GQA without Bridge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_dim, hidden",
    [
        (8, 8, 64, 512),     # MHA, head_dim from kv_channels
        (8, 4, 64, 512),     # GQA 2:1
        (32, 4, 128, 4096),  # Llama-3-8B-style attention
        (40, 8, 128, 5120),  # GQA 5:1
        (32, 32, 128, 4096), # MHA at Llama-3 scale
    ],
)
def test_qkv_weights_roundtrip(num_heads, num_kv_heads, head_dim, hidden):
    cfg = MTC(
        num_attention_heads=num_heads,
        num_query_groups=num_kv_heads,
        kv_channels=head_dim,
        hidden_size=hidden,
    )
    torch.manual_seed(0xABCDEF)
    q = torch.randn(num_heads * head_dim, hidden, dtype=torch.float32)
    k = torch.randn(num_kv_heads * head_dim, hidden, dtype=torch.float32)
    v = torch.randn(num_kv_heads * head_dim, hidden, dtype=torch.float32)

    qkv = HELPERS.merge_qkv_weights(cfg, q, k, v)
    expected_rows = (num_heads + 2 * num_kv_heads) * head_dim
    assert qkv.shape == (expected_rows, hidden)

    q2, k2, v2 = HELPERS.split_qkv_weights(cfg, qkv)
    assert torch.allclose(q, q2, atol=0, rtol=0), "Q mismatch"
    assert torch.allclose(k, k2, atol=0, rtol=0), "K mismatch"
    assert torch.allclose(v, v2, atol=0, rtol=0), "V mismatch"


@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_dim",
    [
        (8, 8, 64),
        (8, 4, 64),
        (32, 4, 128),
    ],
)
def test_qkv_biases_roundtrip(num_heads, num_kv_heads, head_dim):
    cfg = MTC(
        num_attention_heads=num_heads,
        num_query_groups=num_kv_heads,
        kv_channels=head_dim,
        hidden_size=num_heads * head_dim,
    )
    torch.manual_seed(0xCAFE)
    q = torch.randn(num_heads * head_dim, dtype=torch.float32)
    k = torch.randn(num_kv_heads * head_dim, dtype=torch.float32)
    v = torch.randn(num_kv_heads * head_dim, dtype=torch.float32)

    qkv = HELPERS.merge_qkv_biases(cfg, q, k, v)
    q2, k2, v2 = HELPERS.split_qkv_biases(cfg, qkv)
    assert torch.allclose(q, q2)
    assert torch.allclose(k, k2)
    assert torch.allclose(v, v2)


def test_split_qkv_rejects_fp8_scale_shape():
    """v0 vendored helper rejects FP8 blockwise scale tensors explicitly."""
    cfg = MTC(num_attention_heads=8, num_query_groups=4,
              kv_channels=64, hidden_size=512)
    qkv = torch.randn((8 + 2 * 4) * 64, 32)  # last dim 32 != hidden 512
    with pytest.raises(NotImplementedError, match="FP8"):
        HELPERS.split_qkv_weights(cfg, qkv)


def test_gated_mlp_roundtrip():
    torch.manual_seed(7)
    intermediate, hidden = 1024, 256
    gate = torch.randn(intermediate, hidden)
    up = torch.randn(intermediate, hidden)
    fused = HELPERS.merge_gated_mlp(gate, up)
    assert fused.shape == (2 * intermediate, hidden)
    g2, u2 = HELPERS.split_gated_mlp(fused)
    assert torch.allclose(gate, g2)
    assert torch.allclose(up, u2)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_split_gated_mlp_tp_uninterleaves_per_rank_chunks(tp):
    """Receiver-side un-interleave: per-rank chunks are [gate_local;
    up_local]; concat_dim0 across TP ranks produces
    [gate_r0; up_r0; gate_r1; up_r1; ...]. split_gated_mlp_tp must
    recover ([gate_r0; gate_r1; ...], [up_r0; up_r1; ...]) — byte-
    identical to what each rank held + concatenated.
    """
    torch.manual_seed(42 + tp)
    inter_per_rank, hidden = 256, 128
    intermediate = inter_per_rank * tp
    # Build per-rank gates + ups, then the interleaved global the way
    # an axis-0 concat of merge_gated_mlp(gate_rank, up_rank) would
    # look on the wire.
    gate_ranks = [torch.randn(inter_per_rank, hidden) for _ in range(tp)]
    up_ranks = [torch.randn(inter_per_rank, hidden) for _ in range(tp)]
    interleaved = torch.cat(
        [HELPERS.merge_gated_mlp(g, u) for g, u in zip(gate_ranks, up_ranks)],
        dim=0,
    )
    assert interleaved.shape == (2 * intermediate, hidden)

    gate, up = HELPERS.split_gated_mlp_tp(interleaved, tp=tp)
    assert gate.shape == (intermediate, hidden)
    assert up.shape == (intermediate, hidden)
    expected_gate = torch.cat(gate_ranks, dim=0)
    expected_up = torch.cat(up_ranks, dim=0)
    assert torch.equal(gate, expected_gate)
    assert torch.equal(up, expected_up)


def test_split_gated_mlp_tp_equals_split_gated_mlp_when_tp_eq_1():
    """tp=1 should produce the same output as split_gated_mlp."""
    torch.manual_seed(11)
    intermediate, hidden = 512, 64
    gate = torch.randn(intermediate, hidden)
    up = torch.randn(intermediate, hidden)
    fused = HELPERS.merge_gated_mlp(gate, up)
    g_v1, u_v1 = HELPERS.split_gated_mlp(fused)
    g_tp, u_tp = HELPERS.split_gated_mlp_tp(fused, tp=1)
    assert torch.equal(g_v1, g_tp)
    assert torch.equal(u_v1, u_tp)


def test_split_gated_mlp_tp_rejects_bad_divisibility():
    fused = torch.randn(7, 8)  # 7 % (2 * tp=2) != 0
    with pytest.raises(ValueError, match="not divisible"):
        HELPERS.split_gated_mlp_tp(fused, tp=2)


def test_megatron_transformer_config_dict_roundtrip():
    cfg = MTC(num_attention_heads=8, num_query_groups=4,
              kv_channels=64, hidden_size=512)
    cfg2 = MTC.from_dict(cfg.to_dict())
    assert cfg == cfg2
    cfg3 = MTC(num_attention_heads=8, num_query_groups=8,
               kv_channels=None, hidden_size=512)
    cfg4 = MTC.from_dict(cfg3.to_dict())
    assert cfg3 == cfg4
    assert cfg4.head_size == 64  # derived from hidden / heads


# ---------------------------------------------------------------------------
# Cross-check vs Bridge's authoritative implementation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="megatron-bridge not installed")
@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_dim, hidden",
    [
        (8, 8, 64, 512),
        (8, 4, 64, 512),
        (32, 4, 128, 4096),
        (40, 8, 128, 5120),
    ],
)
def test_merge_qkv_weights_byte_identity_with_bridge(
    num_heads, num_kv_heads, head_dim, hidden,
):
    cfg_ours = MTC(
        num_attention_heads=num_heads, num_query_groups=num_kv_heads,
        kv_channels=head_dim, hidden_size=hidden,
    )
    cfg_bridge = _bridge_config(num_heads, num_kv_heads, head_dim, hidden)
    torch.manual_seed(123)
    q = torch.randn(num_heads * head_dim, hidden, dtype=torch.float32)
    k = torch.randn(num_kv_heads * head_dim, hidden, dtype=torch.float32)
    v = torch.randn(num_kv_heads * head_dim, hidden, dtype=torch.float32)

    ours = HELPERS.merge_qkv_weights(cfg_ours, q, k, v)
    theirs = bridge_merge_qkv_weights(cfg_bridge, q, k, v)
    assert torch.equal(ours, theirs), \
        "Vendored merge_qkv_weights drifted from Bridge upstream"


@pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="megatron-bridge not installed")
@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_dim, hidden",
    [
        (8, 8, 64, 512),
        (8, 4, 64, 512),
        (32, 4, 128, 4096),
    ],
)
def test_split_qkv_weights_byte_identity_with_bridge(
    num_heads, num_kv_heads, head_dim, hidden,
):
    cfg_ours = MTC(
        num_attention_heads=num_heads, num_query_groups=num_kv_heads,
        kv_channels=head_dim, hidden_size=hidden,
    )
    cfg_bridge = _bridge_config(num_heads, num_kv_heads, head_dim, hidden)
    # Build the interleaved QKV via Bridge so we know it matches its
    # convention exactly, then run our split against it.
    torch.manual_seed(456)
    q = torch.randn(num_heads * head_dim, hidden, dtype=torch.float32)
    k = torch.randn(num_kv_heads * head_dim, hidden, dtype=torch.float32)
    v = torch.randn(num_kv_heads * head_dim, hidden, dtype=torch.float32)
    qkv = bridge_merge_qkv_weights(cfg_bridge, q, k, v)

    q1, k1, v1 = HELPERS.split_qkv_weights(cfg_ours, qkv)
    q2, k2, v2 = bridge_split_qkv_weights(cfg_bridge, qkv)
    assert torch.equal(q1, q2)
    assert torch.equal(k1, k2)
    assert torch.equal(v1, v2)
    # And the pulled values match the originals.
    assert torch.equal(q1, q)
    assert torch.equal(k1, k)
    assert torch.equal(v1, v)


@pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="megatron-bridge not installed")
def test_split_qkv_biases_byte_identity_with_bridge():
    num_heads, num_kv_heads, head_dim = 32, 4, 128
    hidden = num_heads * head_dim
    cfg_ours = MTC(
        num_attention_heads=num_heads, num_query_groups=num_kv_heads,
        kv_channels=head_dim, hidden_size=hidden,
    )
    cfg_bridge = _bridge_config(num_heads, num_kv_heads, head_dim, hidden)
    torch.manual_seed(789)
    q = torch.randn(num_heads * head_dim, dtype=torch.float32)
    k = torch.randn(num_kv_heads * head_dim, dtype=torch.float32)
    v = torch.randn(num_kv_heads * head_dim, dtype=torch.float32)
    qkv_b = bridge_merge_qkv_biases(cfg_bridge, q, k, v)
    q1, k1, v1 = HELPERS.split_qkv_biases(cfg_ours, qkv_b)
    q2, k2, v2 = bridge_split_qkv_biases(cfg_bridge, qkv_b)
    assert torch.equal(q1, q2)
    assert torch.equal(k1, k2)
    assert torch.equal(v1, v2)
