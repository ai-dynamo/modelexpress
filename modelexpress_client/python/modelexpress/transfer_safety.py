# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Transfer safety checks for ModelExpress GPU-to-GPU weight transfer.

Two independent safety mechanisms:

1. Feature allow-list: checks model config before attempting P2P transfer.
   Models with unsupported features are rejected early and fall back to
   disk loading. MLA attention models (DeepSeek, Kimi) are blocked from
   P2P transfers due to unresolved weight corruption after RDMA receive.
   The bytes transfer correctly but inference diverges - root cause is
   under investigation.

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
# Feature detection and blocking
# ---------------------------------------------------------------------------

# MLA (Multi-head Latent Attention) is the only feature currently blocked.
# Bytes transfer correctly via RDMA but inference diverges on all tested
# vLLM versions (0.12.0 through 0.17.1). Root cause under investigation.
# Detection is config-based: kv_lora_rank presence in the HF config.
# Bypass with MX_SKIP_FEATURE_CHECK=1 to test MLA transfers anyway.



def detect_model_features(model_config) -> dict[str, str]:
    """Detect model features relevant to transfer safety.

    Returns a dict of feature name -> value for logging and validation.
    """
    hf_config = model_config.hf_text_config
    features: dict[str, str] = {}

    features["model_type"] = getattr(hf_config, "model_type", "unknown")
    features["dtype"] = str(model_config.dtype).replace("torch.", "")
    features["quantization"] = model_config.quantization or "none"

    # MLA detection via HF config attribute
    has_mla = getattr(hf_config, "kv_lora_rank", None) is not None
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

    Currently blocks only MLA attention models (detected via kv_lora_rank
    in the HF config). All other models are allowed.

    Returns (allowed, reason). If not allowed, reason explains why.
    """
    if os.environ.get("MX_SKIP_FEATURE_CHECK", "0") == "1":
        logger.warning("MX_SKIP_FEATURE_CHECK=1: bypassing feature checks")
        return True, "bypassed"

    features = detect_model_features(model_config)

    if features["attention"] == "mla":
        reason = (
            "MLA attention models are blocked from P2P transfer due to "
            "unresolved weight corruption (NVBug 6066010)"
        )
        logger.warning(
            f"[Transfer Safety] P2P transfer denied: {reason}. "
            f"Features: {features}. "
            f"Set MX_SKIP_FEATURE_CHECK=1 to bypass."
        )
        return False, reason

    logger.info(f"[Transfer Safety] P2P transfer allowed. Features: {features}")
    return True, "allowed"


# ---------------------------------------------------------------------------
# Transfer fingerprint
# ---------------------------------------------------------------------------

def get_vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except ImportError:
        return "unknown"


def get_torch_version() -> str:
    return torch.__version__


def get_cuda_version() -> str:
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


def get_deep_gemm_version() -> str:
    try:
        from importlib.metadata import version
        return version("deep-gemm")
    except Exception:
        return "unknown"


def get_gpu_capability() -> str:
    """Get the GPU compute capability as 'major.minor' (e.g. '9.0', '10.0').

    Different GPU architectures can trigger different post-processing paths
    (e.g. DeepGemm TMA scale packing differs between SM90 and SM100).
    Including this in SourceIdentity ensures heterogeneous clusters don't
    attempt cross-architecture transfers.

    Returns 'unknown' if CUDA is not available.
    """
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return f"{props.major}.{props.minor}"
    except Exception:
        pass
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
    gpu_capability: str = ""
    attention_backend: str = ""
    manifest_hash: str = ""
    tensor_count: int = 0

    @classmethod
    def from_environment(cls, tensors: dict[str, torch.Tensor] | None = None) -> TransferFingerprint:
        """Build a fingerprint from the current runtime environment."""
        fp = cls(
            vllm_version=get_vllm_version(),
            torch_version=get_torch_version(),
            cuda_version=get_cuda_version(),
            deep_gemm_version=get_deep_gemm_version(),
            gpu_capability=get_gpu_capability(),
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
            "gpu_capability": self.gpu_capability,
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

        # vllm_version, cuda_version, torch_version, and deep_gemm_version
        # are checked at source-selection time via extra_parameters in
        # SourceIdentity (hashed into mx_source_id).
        # Only runtime-resolved properties are validated here.
        if self.attention_backend != source.attention_backend:
            mismatches.append(
                f"attention backend: source={source.attention_backend}, target={self.attention_backend}"
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
