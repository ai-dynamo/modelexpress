# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for transfer safety checks."""

import json
import os
from unittest.mock import MagicMock

import pytest
import torch

from modelexpress.transfer_safety import (
    ALLOWED_MODEL_TYPES,
    TransferFingerprint,
    _compute_manifest_hash,
    check_transfer_allowed,
    detect_model_features,
)


class FakeHfConfig:
    """Minimal fake for hf_text_config that returns None for missing attributes."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        return None


class FakeConfig:
    """Minimal fake for model_config."""
    def __init__(self, model_type="llama", dtype=torch.bfloat16, quantization=None, **kwargs):
        self.dtype = dtype
        self.quantization = quantization
        self.hf_text_config = FakeHfConfig(model_type=model_type, **kwargs)


# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------

class TestDetectModelFeatures:
    def test_llama_standard_attention(self):
        config = FakeConfig(
            model_type="llama",
            num_key_value_heads=8,
            num_attention_heads=32,
        )
        features = detect_model_features(config)
        assert features["model_type"] == "llama"
        assert features["attention"] == "standard"

    def test_deepseek_v2_mla(self):
        config = FakeConfig(
            model_type="deepseek_v2",
            kv_lora_rank=512,
            num_key_value_heads=1,
            num_attention_heads=16,
        )
        features = detect_model_features(config)
        assert features["attention"] == "mla"

    def test_fp8_quantization(self):
        config = FakeConfig(
            model_type="llama",
            quantization="fp8",
        )
        features = detect_model_features(config)
        assert features["quantization"] == "fp8"

    def test_moe_detection(self):
        config = FakeConfig(
            model_type="llama",
            num_key_value_heads=8,
            num_attention_heads=32,
            n_routed_experts=64,
        )
        features = detect_model_features(config)
        assert features["moe"] == "64"

    def test_unknown_model_type(self):
        config = FakeConfig(
            model_type="some_new_architecture",
        )
        features = detect_model_features(config)
        assert features["model_type"] == "some_new_architecture"


# ---------------------------------------------------------------------------
# Allow-list
# ---------------------------------------------------------------------------

class TestCheckTransferAllowed:
    def test_llama_bf16_allowed(self):
        config = FakeConfig(
            model_type="llama",
            num_key_value_heads=8,
            num_attention_heads=32,
        )
        allowed, reason = check_transfer_allowed(config)
        assert allowed

    def test_deepseek_mla_denied(self):
        config = FakeConfig(
            model_type="deepseek_v2",
            kv_lora_rank=512,
        )
        allowed, reason = check_transfer_allowed(config)
        assert not allowed
        assert "mla" in reason.lower()

    def test_fp8_llama_allowed(self):
        config = FakeConfig(
            model_type="llama",
            num_key_value_heads=8,
            num_attention_heads=32,
            quantization="fp8",
        )
        allowed, reason = check_transfer_allowed(config)
        assert allowed

    def test_unknown_model_type_denied(self):
        config = FakeConfig(
            model_type="brand_new_architecture",
            num_key_value_heads=8,
            num_attention_heads=32,
        )
        allowed, reason = check_transfer_allowed(config)
        assert not allowed
        assert "brand_new_architecture" in reason

    def test_unknown_quantization_denied(self):
        config = FakeConfig(
            model_type="llama",
            num_key_value_heads=8,
            num_attention_heads=32,
            quantization="some_new_quant",
        )
        allowed, reason = check_transfer_allowed(config)
        assert not allowed
        assert "some_new_quant" in reason

    def test_bypass_env_var(self):
        config = FakeConfig(
            model_type="deepseek_v2",
            kv_lora_rank=512,
        )
        os.environ["MX_SKIP_FEATURE_CHECK"] = "1"
        try:
            allowed, reason = check_transfer_allowed(config)
            assert allowed
            assert reason == "bypassed"
        finally:
            del os.environ["MX_SKIP_FEATURE_CHECK"]


# ---------------------------------------------------------------------------
# Transfer fingerprint
# ---------------------------------------------------------------------------

class TestTransferFingerprint:
    def test_round_trip_json(self):
        fp = TransferFingerprint(
            vllm_version="0.17.1",
            torch_version="2.10.0",
            cuda_version="12.9",
            deep_gemm_version="2.3.0",
            attention_backend="FLASHINFER_MLA",
            manifest_hash="abc123",
            tensor_count=729,
        )
        json_str = fp.to_json()
        fp2 = TransferFingerprint.from_json(json_str)
        assert fp2.vllm_version == "0.17.1"
        assert fp2.tensor_count == 729
        assert fp2.manifest_hash == "abc123"

    def test_validate_matching(self):
        fp1 = TransferFingerprint(
            vllm_version="0.12.0",
            cuda_version="12.9",
            attention_backend="CUTLASS_MLA",
            deep_gemm_version="2.1.0",
            manifest_hash="abc",
            tensor_count=729,
        )
        fp2 = TransferFingerprint(
            vllm_version="0.12.0",
            cuda_version="12.9",
            attention_backend="CUTLASS_MLA",
            deep_gemm_version="2.1.0",
            manifest_hash="abc",
            tensor_count=729,
        )
        compatible, mismatches = fp2.validate_against(fp1)
        assert compatible
        assert len(mismatches) == 0

    def test_validate_version_mismatches_ignored(self):
        # vLLM, CUDA, torch, and deep_gemm version mismatches are caught at
        # source-selection time via extra_parameters in SourceIdentity, not
        # by the fingerprint.
        source = TransferFingerprint(vllm_version="0.17.1", cuda_version="12.9",
                                     torch_version="2.10.0",
                                     attention_backend="X", deep_gemm_version="1.0")
        target = TransferFingerprint(vllm_version="0.12.0", cuda_version="12.8",
                                     torch_version="2.9.0",
                                     attention_backend="X", deep_gemm_version="2.0")
        compatible, mismatches = target.validate_against(source)
        assert compatible
        assert len(mismatches) == 0

    def test_validate_backend_mismatch(self):
        source = TransferFingerprint(vllm_version="0.12.0", cuda_version="12.9",
                                     attention_backend="CUTLASS_MLA", deep_gemm_version="1.0")
        target = TransferFingerprint(vllm_version="0.12.0", cuda_version="12.9",
                                     attention_backend="FLASHINFER_MLA", deep_gemm_version="1.0")
        compatible, mismatches = target.validate_against(source)
        assert not compatible
        assert any("attention" in m for m in mismatches)

    def test_validate_manifest_mismatch(self):
        source = TransferFingerprint(vllm_version="0.12.0", cuda_version="12.9",
                                     attention_backend="X", deep_gemm_version="1.0",
                                     manifest_hash="aaa", tensor_count=729)
        target = TransferFingerprint(vllm_version="0.12.0", cuda_version="12.9",
                                     attention_backend="X", deep_gemm_version="1.0",
                                     manifest_hash="bbb", tensor_count=648)
        compatible, mismatches = target.validate_against(source)
        assert not compatible
        assert any("manifest" in m for m in mismatches)


# ---------------------------------------------------------------------------
# Manifest hash
# ---------------------------------------------------------------------------

class TestManifestHash:
    def test_same_tensors_same_hash(self):
        t1 = {"a": torch.zeros(10, dtype=torch.float32), "b": torch.zeros(20, dtype=torch.bfloat16)}
        t2 = {"a": torch.ones(10, dtype=torch.float32), "b": torch.ones(20, dtype=torch.bfloat16)}
        # Same shapes and dtypes, different values -> same hash
        assert _compute_manifest_hash(t1) == _compute_manifest_hash(t2)

    def test_different_shapes_different_hash(self):
        t1 = {"a": torch.zeros(10, dtype=torch.float32)}
        t2 = {"a": torch.zeros(20, dtype=torch.float32)}
        assert _compute_manifest_hash(t1) != _compute_manifest_hash(t2)

    def test_different_names_different_hash(self):
        t1 = {"a": torch.zeros(10, dtype=torch.float32)}
        t2 = {"b": torch.zeros(10, dtype=torch.float32)}
        assert _compute_manifest_hash(t1) != _compute_manifest_hash(t2)

    def test_different_dtypes_different_hash(self):
        t1 = {"a": torch.zeros(10, dtype=torch.float32)}
        t2 = {"a": torch.zeros(10, dtype=torch.float16)}
        assert _compute_manifest_hash(t1) != _compute_manifest_hash(t2)
