# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Transfer safety checks for ModelExpress GPU-to-GPU weight transfer.

Two independent safety mechanisms:

1. Feature allow-list: checks model config before attempting P2P transfer.
   Models with unsupported post-processing features (MLA, FP8 TMA scales)
   are rejected early and fall back to disk loading.

2. Transfer fingerprint: captures the runtime environment and tensor
   manifest structure. Source publishes its fingerprint; target computes
   its own and rejects the transfer if they don't match.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field

import torch

logger = logging.getLogger("modelexpress.transfer_safety")

# ---------------------------------------------------------------------------
# Feature allow-list
# ---------------------------------------------------------------------------

# Model features that are known to produce correct results via P2P weight transfer.
# Any feature not on this list causes automatic fallback to disk loading.
# Add features here only after validating RDMA correctness end-to-end.
# Model architectures validated for P2P weight transfer.
# Only models whose model_type is in this set are allowed.
# Any unknown architecture falls back to disk loading.
ALLOWED_MODEL_TYPES: set[str] = {
    "llama",
    "mistral",
    "qwen2",
    "gemma",
    "gemma2",
    "phi3",
    "phi",
    "starcoder2",
    "gpt_neox",
    "gpt2",
    "opt",
    "falcon",
    "codellama",
}

# Model types known to be unsafe for P2P transfer.
# Listed for documentation; anything not in ALLOWED_MODEL_TYPES is denied.
BLOCKED_MODEL_TYPES: set[str] = {
    "deepseek_v2",   # MLA
    "deepseek_v3",   # MLA
    "deepseek_v32",  # MLA
    "deepseek_mtp",  # MLA
    "AXK1",          # MLA variant
    "kimi_k2",       # MLA (Moonshot Kimi K2)
    "kimi_k25",      # MLA (Moonshot Kimi K2.5)
}

# Quantization methods validated for P2P transfer.
ALLOWED_QUANTIZATIONS: set[str | None] = {
    None,            # No quantization (BF16/FP16/FP32)
    "",              # Empty string (same as None)
    "fp8",           # FP8 block quantization (scales are pure data)
    "awq",           # AWQ quantization (packed int4 weights + scales)
    "gptq",          # GPTQ quantization (packed int4 weights + scales + zeros)
    "gptq_marlin",   # GPTQ with Marlin kernel (same weights, different runtime)
    "awq_marlin",    # AWQ with Marlin kernel (same weights, different runtime)
}

# Weight dtypes validated for P2P transfer.
ALLOWED_DTYPES: set[str] = {
    "bfloat16",
    "float16",
    "float32",
    "float8_e4m3fn",  # FP8 quantized weights
}


def detect_model_features(model_config) -> dict[str, str]:
    """Detect model features relevant to transfer safety.

    Returns a dict of feature name -> value for logging and validation.
    """
    hf_config = model_config.hf_text_config
    features: dict[str, str] = {}

    features["model_type"] = getattr(hf_config, "model_type", "unknown")
    features["dtype"] = str(model_config.dtype).replace("torch.", "")
    features["quantization"] = model_config.quantization or "none"

    # MLA detection
    mla_model_types = {"deepseek_v2", "deepseek_v3", "deepseek_v32", "deepseek_mtp", "AXK1", "kimi_k2", "kimi_k25"}
    has_mla = (
        features["model_type"] in mla_model_types
        or getattr(hf_config, "kv_lora_rank", None) is not None
    )
    features["attention"] = "mla" if has_mla else "standard"

    # MoE
    num_experts = (
        getattr(hf_config, "n_routed_experts", None)
        or getattr(hf_config, "num_local_experts", None)
    )
    if num_experts and num_experts > 1:
        features["moe"] = str(num_experts)

    return features


def check_transfer_allowed(model_config) -> tuple[bool, str]:
    """Check if P2P weight transfer is allowed for this model.

    Uses a strict allow-list approach: only explicitly validated model
    architectures, quantization methods, and dtypes are permitted.
    Everything else falls back to disk loading.

    Returns (allowed, reason). If not allowed, reason explains why.
    """
    if os.environ.get("MX_SKIP_FEATURE_CHECK", "0") == "1":
        logger.warning("MX_SKIP_FEATURE_CHECK=1: bypassing feature allow-list")
        return True, "bypassed"

    features = detect_model_features(model_config)
    reasons: list[str] = []

    model_type = features["model_type"]
    if model_type not in ALLOWED_MODEL_TYPES:
        reasons.append(f"model_type '{model_type}' is not validated for P2P transfer")

    dtype = features["dtype"]
    if dtype not in ALLOWED_DTYPES:
        reasons.append(f"dtype '{dtype}' is not validated for P2P transfer")

    quant = model_config.quantization
    if quant not in ALLOWED_QUANTIZATIONS:
        reasons.append(f"quantization '{quant}' is not validated for P2P transfer")

    if features["attention"] == "mla":
        reasons.append("MLA attention creates derived tensors incompatible with P2P transfer")

    if reasons:
        reason_str = "; ".join(reasons)
        logger.warning(
            f"[Transfer Safety] P2P transfer denied: {reason_str}. "
            f"Features: {features}. "
            f"Set MX_SKIP_FEATURE_CHECK=1 to bypass."
        )
        return False, reason_str

    logger.info(f"[Transfer Safety] P2P transfer allowed. Features: {features}")
    return True, "allowed"


# ---------------------------------------------------------------------------
# Transfer fingerprint
# ---------------------------------------------------------------------------

def _get_vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except ImportError:
        return "unknown"


def _get_torch_version() -> str:
    return torch.__version__


def _get_cuda_version() -> str:
    return getattr(torch.version, "cuda", "unknown") or "unknown"


def _get_attention_backend() -> str:
    """Get the selected attention backend name."""
    try:
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        if hasattr(config, "model_config") and config.model_config is not None:
            cache_config = getattr(config, "cache_config", None)
            if cache_config is not None:
                return getattr(cache_config, "attention_backend", "unknown") or "unknown"
    except Exception:
        pass
    return os.environ.get("VLLM_ATTENTION_BACKEND", "auto")


def _get_deep_gemm_version() -> str:
    try:
        from importlib.metadata import version
        return version("deep-gemm")
    except Exception:
        return "unknown"


@dataclass
class TransferFingerprint:
    """Captures runtime environment and tensor manifest structure.

    Published by the source alongside tensor metadata. The target computes
    its own fingerprint and compares. Mismatches cause transfer rejection.
    """
    vllm_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    deep_gemm_version: str = ""
    attention_backend: str = ""
    manifest_hash: str = ""
    tensor_count: int = 0

    @classmethod
    def from_environment(cls, tensors: dict[str, torch.Tensor] | None = None) -> TransferFingerprint:
        """Build a fingerprint from the current runtime environment."""
        fp = cls(
            vllm_version=_get_vllm_version(),
            torch_version=_get_torch_version(),
            cuda_version=_get_cuda_version(),
            deep_gemm_version=_get_deep_gemm_version(),
            attention_backend=_get_attention_backend(),
        )
        if tensors is not None:
            fp.manifest_hash = _compute_manifest_hash(tensors)
            fp.tensor_count = len(tensors)
        return fp

    def to_json(self) -> str:
        """Serialize to JSON string for inclusion in proto metadata."""
        return json.dumps({
            "vllm_version": self.vllm_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "deep_gemm_version": self.deep_gemm_version,
            "attention_backend": self.attention_backend,
            "manifest_hash": self.manifest_hash,
            "tensor_count": self.tensor_count,
        }, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> TransferFingerprint:
        """Deserialize from JSON string."""
        d = json.loads(s)
        return cls(**d)

    def validate_against(self, source: TransferFingerprint) -> tuple[bool, list[str]]:
        """Compare this (target) fingerprint against a source fingerprint.

        Returns (compatible, mismatches). Mismatches is a list of human-readable
        strings describing each incompatibility.
        """
        mismatches: list[str] = []

        # Hard requirements: these MUST match
        if self.vllm_version != source.vllm_version:
            mismatches.append(
                f"vLLM version: source={source.vllm_version}, target={self.vllm_version}"
            )
        if self.cuda_version != source.cuda_version:
            mismatches.append(
                f"CUDA version: source={source.cuda_version}, target={self.cuda_version}"
            )
        if self.attention_backend != source.attention_backend:
            mismatches.append(
                f"attention backend: source={source.attention_backend}, target={self.attention_backend}"
            )
        if self.deep_gemm_version != source.deep_gemm_version:
            mismatches.append(
                f"DeepGemm version: source={source.deep_gemm_version}, target={self.deep_gemm_version}"
            )

        # Manifest structure: tensor names, sizes, dtypes must match
        if self.manifest_hash and source.manifest_hash:
            if self.manifest_hash != source.manifest_hash:
                mismatches.append(
                    f"tensor manifest hash: source={source.manifest_hash[:12]}..., "
                    f"target={self.manifest_hash[:12]}... "
                    f"(source={source.tensor_count} tensors, target={self.tensor_count} tensors)"
                )

        return len(mismatches) == 0, mismatches


def _compute_manifest_hash(tensors: dict[str, torch.Tensor]) -> str:
    """Compute a SHA256 hash of the tensor manifest structure.

    Hashes tensor names, sizes, and dtypes (NOT addresses, which are
    machine-specific). Two identical models with identical post-processing
    should produce the same hash regardless of GPU memory layout.
    """
    entries = []
    for name in sorted(tensors.keys()):
        t = tensors[name]
        entries.append(f"{name}:{list(t.shape)}:{t.dtype}:{t.numel() * t.element_size()}")
    manifest_str = "\n".join(entries)
    return hashlib.sha256(manifest_str.encode()).hexdigest()
