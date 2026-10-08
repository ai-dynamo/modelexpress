# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-specific raw-weight selection for SGLang speculative drafts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

import torch


class DraftWeightAdapter(Protocol):
    """Model-specific decisions; SGLang still performs the actual load."""

    compatibility_tag: str

    def supports_main(self, class_name: str) -> bool: ...

    def supports_draft(self, class_name: str) -> bool: ...

    def includes(self, name: str) -> bool: ...

    def uses_checkpoint_tensor(self, name: str, role: str) -> bool: ...

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

    def uses_checkpoint_tensor(self, name: str, role: str) -> bool:
        is_draft_weight = self.includes(name)
        return is_draft_weight if role == "draft" else not is_draft_weight

    def transferable_tensors(
        self, model: torch.nn.Module, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return _without_shared_embed_and_head(model, tensors)


def _without_shared_embed_and_head(
    model: torch.nn.Module, tensors: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    get_shared = getattr(model, "get_embed_and_head", None)
    if not callable(get_shared):
        raise TypeError("MTP draft does not expose get_embed_and_head")
    embed, head = get_shared()
    shared = {
        t.untyped_storage().data_ptr() for t in (embed, head) if t.numel() > 0
    }
    return {
        name: tensor
        for name, tensor in tensors.items()
        if tensor.numel() == 0 or tensor.untyped_storage().data_ptr() not in shared
    }


@dataclass(frozen=True)
class ExtraLayerNextNWeights:
    """DeepSeek/GLM NextN checkpoints keep the draft in an extra decoder layer."""

    family: str
    main_models: frozenset[str]
    draft_model: str
    base_layer: int

    @property
    def compatibility_tag(self) -> str:
        return f"{self.family}-nextn-layer-{self.base_layer}-v1"

    def supports_main(self, class_name: str) -> bool:
        return class_name in self.main_models

    def supports_draft(self, class_name: str) -> bool:
        return class_name == self.draft_model

    def _layer(self, name: str) -> int | None:
        parts = name.split(".", 3)
        if len(parts) < 4 or parts[:2] != ["model", "layers"]:
            return None
        return int(parts[2]) if parts[2].isdecimal() else None

    def includes(self, name: str) -> bool:
        return self._layer(name) == self.base_layer and not (
            "shared_head.head" in name or "embed_tokens" in name
        )

    def uses_checkpoint_tensor(self, name: str, role: str) -> bool:
        if role == "draft":
            return self.includes(name)
        layer = self._layer(name)
        return layer is None or layer < self.base_layer

    def transferable_tensors(
        self, model: torch.nn.Module, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return _without_shared_embed_and_head(model, tensors)


_ADAPTERS: tuple[DraftWeightAdapter, ...] = (Qwen35MtpWeights(),)
_NEXTN_FAMILIES = (
    (
        "deepseek-v3",
        frozenset({"DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM"}),
        "DeepseekV3ForCausalLMNextN",
    ),
    ("glm4-moe", frozenset({"Glm4MoeForCausalLM"}), "Glm4MoeForCausalLMNextN"),
    (
        "glm4-moe-lite",
        frozenset({"Glm4MoeLiteForCausalLM"}),
        "Glm4MoeLiteForCausalLMNextN",
    ),
    (
        "glm-moe-dsa",
        frozenset({"GlmMoeDsaForCausalLM"}),
        "GlmMoeDsaForCausalLMNextN",
    ),
)


def draft_weight_adapter_for(
    class_name: str, *, role: str, model: object | None = None
) -> DraftWeightAdapter | None:
    for adapter in _ADAPTERS:
        if role == "main" and adapter.supports_main(class_name):
            return adapter
        if role == "draft" and adapter.supports_draft(class_name):
            return adapter
    config = getattr(model, "config", None)
    base = getattr(config, "num_hidden_layers", None)
    nextn_layers = getattr(config, "num_nextn_predict_layers", None)
    if (
        isinstance(base, bool)
        or not isinstance(base, int)
        or base <= 1
        or nextn_layers != 1
        or isinstance(nextn_layers, bool)
    ):
        return None
    for family, main_models, draft_model in _NEXTN_FAMILIES:
        adapter = ExtraLayerNextNWeights(family, main_models, draft_model, base)
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
