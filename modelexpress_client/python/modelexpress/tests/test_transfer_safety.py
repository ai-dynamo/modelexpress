# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for transfer safety checks."""

import torch

from modelexpress.transfer_safety import (
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

    def test_trtllm_model_config_shape(self):
        config = type(
            "TrtllmConfig",
            (),
            {
                "pretrained_config": FakeHfConfig(model_type="llama"),
                "dtype": None,
                "torch_dtype": torch.bfloat16,
                "quant_config": FakeHfConfig(quant_algo="FP8"),
            },
        )()

        features = detect_model_features(config)

        assert features == {
            "model_type": "llama",
            "dtype": "bfloat16",
            "quantization": "FP8",
            "attention": "standard",
        }


# ---------------------------------------------------------------------------
# Feature checks
# ---------------------------------------------------------------------------

class TestCheckTransferAllowed:
    def test_llama_allowed(self):
        config = FakeConfig(
            model_type="llama",
            num_key_value_heads=8,
            num_attention_heads=32,
        )
        allowed, reason = check_transfer_allowed(config)
        assert allowed

    def test_unknown_model_type_allowed(self):
        config = FakeConfig(
            model_type="brand_new_architecture",
            num_key_value_heads=8,
            num_attention_heads=32,
        )
        allowed, reason = check_transfer_allowed(config)
        assert allowed

    def test_fp8_llama_allowed(self):
        config = FakeConfig(
            model_type="llama",
            num_key_value_heads=8,
            num_attention_heads=32,
            quantization="fp8",
        )
        allowed, reason = check_transfer_allowed(config)
        assert allowed

    def test_deepseek_mla_allowed(self):
        config = FakeConfig(
            model_type="deepseek_v2",
            kv_lora_rank=512,
        )
        allowed, _ = check_transfer_allowed(config)
        assert allowed

    def test_kimi_mla_allowed(self):
        config = FakeConfig(
            model_type="kimi_k25",
            kv_lora_rank=512,
        )
        allowed, _ = check_transfer_allowed(config)
        assert allowed

    def test_no_kv_lora_rank_allowed(self):
        config = FakeConfig(
            model_type="deepseek_v2",
        )
        allowed, _ = check_transfer_allowed(config)
        assert allowed
