# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Transfer safety checks for ModelExpress GPU-to-GPU weight transfer.

Feature detection surfaces model features (attention, quantization, MoE)
in the loader log so unexpected combinations are visible during QA.
Currently no feature combination blocks P2P transfer.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("modelexpress.transfer_safety")


def detect_model_features(model_config) -> dict[str, str]:
    """Detect model features relevant to transfer safety.

    Returns a dict of feature name -> value for logging and validation.
    """
    hf_config = (
        getattr(model_config, "hf_text_config", None)
        or getattr(model_config, "pretrained_config", None)
    )
    features: dict[str, str] = {}

    features["model_type"] = getattr(hf_config, "model_type", "unknown")
    dtype = getattr(model_config, "dtype", None)
    if dtype is None:
        dtype = getattr(model_config, "torch_dtype", None)
    features["dtype"] = str(dtype).replace("torch.", "")
    quantization = getattr(model_config, "quantization", None)
    if quantization is None:
        quant_config = getattr(model_config, "quant_config", None)
        quantization = getattr(quant_config, "quant_algo", None)
    features["quantization"] = str(quantization or "none")

    # MLA detection via HF config attribute
    kv_lora_rank = getattr(hf_config, "kv_lora_rank", None)
    has_mla = isinstance(kv_lora_rank, int)
    features["attention"] = "mla" if has_mla else "standard"

    # MoE
    num_experts = (
        getattr(hf_config, "n_routed_experts", None)
        or getattr(hf_config, "num_local_experts", None)
    )
    if isinstance(num_experts, int) and num_experts > 1:
        features["moe"] = str(num_experts)

    return features


def check_transfer_allowed(model_config) -> tuple[bool, str]:
    """Check if P2P weight transfer is allowed for this model.

    No feature combination is currently blocked; the function logs detected
    features and always returns allowed. Kept as a hook so future safety
    gates can be added in one place.

    Returns (allowed, reason).
    """
    features = detect_model_features(model_config)
    logger.info(f"[Transfer Safety] P2P transfer allowed. Features: {features}")
    return True, "allowed"
