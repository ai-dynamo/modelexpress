# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-specific raw-weight selection for SGLang speculative drafts."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

import torch


class DraftWeightAdapter(Protocol):
    """Model-specific decisions; SGLang still performs the actual load."""

    compatibility_tag: str

    def supports_main(self, class_name: str) -> bool: ...

    def supports_draft(self, class_name: str) -> bool: ...

    def includes(self, name: str) -> bool: ...

    def transferable_tensors(
        self, model: torch.nn.Module, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]: ...


class Qwen35MtpWeights:
    """Mirror Qwen3.5's MTP branch selection, not a generic MX naming rule."""

    compatibility_tag = "qwen35-mtp-v1"

    _MAIN_MODELS = frozenset(
        {
            "Qwen3_5ForCausalLM",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        }
    )

    def supports_main(self, class_name: str) -> bool:
        return class_name in self._MAIN_MODELS

    def supports_draft(self, class_name: str) -> bool:
        return class_name == "Qwen3_5ForCausalLMMTP"

    def includes(self, name: str) -> bool:
        # Matches Qwen3_5ForCausalLMMTP.load_weights in SGLang v0.5.16.
        return "mtp" in name

    def transferable_tensors(
        self, model: torch.nn.Module, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        get_shared = getattr(model, "get_embed_and_head", None)
        if not callable(get_shared):
            raise TypeError("Qwen3.5 draft does not expose get_embed_and_head")
        embed, head = get_shared()
        shared = {
            t.untyped_storage().data_ptr() for t in (embed, head) if t.numel() > 0
        }
        return {
            name: tensor
            for name, tensor in tensors.items()
            if tensor.numel() == 0 or tensor.untyped_storage().data_ptr() not in shared
        }


_ADAPTERS: tuple[DraftWeightAdapter, ...] = (Qwen35MtpWeights(),)


def draft_weight_adapter_for(
    class_name: str, *, role: str
) -> DraftWeightAdapter | None:
    for adapter in _ADAPTERS:
        if role == "main" and adapter.supports_main(class_name):
            return adapter
        if role == "draft" and adapter.supports_draft(class_name):
            return adapter
    return None


def draft_tensor_namespace(
    identity,
    adapter: DraftWeightAdapter,
    model_class: str,
    sglang_version: str,
    model_uri: str,
) -> str | None:
    """Isolate draft manifests without changing the main SourceIdentity."""
    if not identity.revision or not sglang_version:
        return None
    signature = json.dumps(
        {
            "identity": identity.SerializeToString(deterministic=True).hex(),
            "adapter": adapter.compatibility_tag,
            "draft_model": model_class,
            "sglang": sglang_version,
            "weights_uri": model_uri,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"mx_draft::{hashlib.sha256(signature).hexdigest()[:32]}::"
