# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-specific raw-weight selection for SGLang speculative drafts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import ClassVar, Protocol

import torch


class DraftSelection(Protocol):
    """Only the decisions needed after exact model-class recognition."""

    def covers(self, name: str, role: str) -> bool: ...

    def compatible_main(self, main: DraftSelection) -> bool: ...

    def publishable(
        self, model: torch.nn.Module, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]: ...


class _SharedDraftTensors:
    def publishable(
        self, model: torch.nn.Module, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return _without_shared_embed_and_head(model, tensors)


class _SglangModelWeightRule(_SharedDraftTensors):
    role: ClassVar[str]
    compatible_main_types: ClassVar[tuple[type[_SglangModelWeightRule], ...]] = ()

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        raise NotImplementedError

    def covers(self, name: str, role: str) -> bool:
        return role == self.role and self.should_read_checkpoint_tensor(name)

    def compatible_main(self, main: DraftSelection) -> bool:
        return (
            self.role == "draft"
            and type(main) in self.compatible_main_types
            and getattr(main, "base_layer", None) == getattr(self, "base_layer", None)
        )


class _Qwen3_5WeightRule(_SglangModelWeightRule):
    pass


class _Qwen3_5TextMainWeightRule(_Qwen3_5WeightRule):
    role = "main"

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return not any(key in name for key in ("rotary_emb.inv_freq", "mtp", "visual"))


class Qwen3_5ForCausalLMWeightRule(_Qwen3_5TextMainWeightRule):
    """Qwen3_5ForCausalLM.load_weights."""


class Qwen3_5MoeForCausalLMWeightRule(_Qwen3_5TextMainWeightRule):
    """Qwen3_5MoeForCausalLM.load_weights."""


class _Qwen3_5VisionMainWeightRule(_Qwen3_5WeightRule):
    role = "main"

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return not any(key in name for key in ("rotary_emb.inv_freq", "mtp"))


class Qwen3_5ForConditionalGenerationWeightRule(_Qwen3_5VisionMainWeightRule):
    """Qwen3_5ForConditionalGeneration.load_weights."""


class Qwen3_5MoeForConditionalGenerationWeightRule(_Qwen3_5VisionMainWeightRule):
    """Qwen3_5MoeForConditionalGeneration.load_weights."""


class InternS2PreviewForConditionalGenerationWeightRule(_Qwen3_5VisionMainWeightRule):
    """InternS2Preview inherits Qwen3.5's conditional-generation loader."""


class InternS2MobiusForConditionalGenerationWeightRule(_Qwen3_5VisionMainWeightRule):
    """InternS2Mobius skips its own MTP and cached rotary tensors."""

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return not name.startswith("mtp.") and not name.endswith(
            (".rotary_emb.inv_freq", ".rotary_emb.cos_cached", ".rotary_emb.sin_cached")
        )


class Qwen3_5ForCausalLMMTPWeightRule(_Qwen3_5WeightRule):
    """Qwen3_5ForCausalLMMTP.load_weights, SGLang main."""

    role = "draft"
    compatible_main_types = (
        Qwen3_5ForCausalLMWeightRule,
        Qwen3_5MoeForCausalLMWeightRule,
        Qwen3_5ForConditionalGenerationWeightRule,
        Qwen3_5MoeForConditionalGenerationWeightRule,
        InternS2PreviewForConditionalGenerationWeightRule,
        InternS2MobiusForConditionalGenerationWeightRule,
    )

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return name in (
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        ) or ("rotary_emb.inv_freq" not in name and "mtp" in name)


@dataclass(frozen=True)
class Qwen4ExpForConditionalGenerationWeightRule(_Qwen3_5VisionMainWeightRule):
    """Qwen4ExpForConditionalGeneration.load_weights."""

    language_model_only: bool = False

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return super().should_read_checkpoint_tensor(name) and not (
            self.language_model_only and "visual" in name
        )


class Qwen4ExpForCausalLMMTPWeightRule(Qwen3_5ForCausalLMMTPWeightRule):
    """Qwen4ExpForCausalLMMTP inherits Qwen3.5's draft load loop."""

    compatible_main_types = (Qwen4ExpForConditionalGenerationWeightRule,)


class _MtpSubstringMainSelection(_SglangModelWeightRule):
    role = "main"

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return "mtp" not in name and "rotary_emb.inv_freq" not in name


class _MtpSubstringDraftSelection(_SglangModelWeightRule):
    role = "draft"

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return "mtp" in name and "rotary_emb.inv_freq" not in name


class Qwen3MoeForCausalLMWeightRule(_MtpSubstringMainSelection):
    """Qwen3MoeForCausalLM.load_weights with is_mtp=False."""


class Qwen3MoeForCausalLMMTPWeightRule(_MtpSubstringDraftSelection):
    """Qwen3MoeForCausalLMMTP.load_weights with is_mtp=True."""

    compatible_main_types = (Qwen3MoeForCausalLMWeightRule,)


class Qwen3NextForCausalLMWeightRule(_MtpSubstringMainSelection):
    """Qwen3NextForCausalLM.load_weights with is_mtp=False."""


class Qwen3NextForCausalLMMTPWeightRule(_MtpSubstringDraftSelection):
    """Qwen3NextForCausalLMMTP.load_weights with is_mtp=True."""

    compatible_main_types = (Qwen3NextForCausalLMWeightRule,)


class ExaoneMoEForCausalLMWeightRule(_MtpSubstringMainSelection):
    """ExaoneMoEForCausalLM.load_weights with is_mtp=False."""


class ExaoneMoeForCausalLMWeightRule(_MtpSubstringMainSelection):
    """ExaoneMoeForCausalLM uses ExaoneMoE's load loop."""


class ExaoneMoEForCausalLMMTPWeightRule(_MtpSubstringDraftSelection):
    """ExaoneMoEForCausalLMMTP.load_weights with is_mtp=True."""

    compatible_main_types = (
        ExaoneMoEForCausalLMWeightRule,
        ExaoneMoeForCausalLMWeightRule,
    )


class MiMoForCausalLMWeightRule(_SglangModelWeightRule):
    """MiMoForCausalLM skips the separate mtp_layers in its load loop."""

    role = "main"

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return not any(
            item in name for item in ("mtp_layers", "rotary_emb.inv_freq", "projector")
        )


class MiMoMTPWeightRule(_SglangModelWeightRule):
    """MiMoMTP remaps mtp_layers and reads shared embed/head tensors."""

    role = "draft"
    compatible_main_types = (MiMoForCausalLMWeightRule,)

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        if "rotary_emb.inv_freq" in name or "projector" in name:
            return False
        return "mtp_layers" in name or any(
            item in name for item in ("embed_tokens", "lm_head")
        )


class MiMoV2ForCausalLMWeightRule(_MtpSubstringMainSelection):
    """MiMoV2ForCausalLM skips draft-only MTP weights."""


class MiMoV2FlashForCausalLMWeightRule(_MtpSubstringMainSelection):
    """MiMoV2FlashForCausalLM inherits the MiMoV2 load loop."""


class MiMoV2MTPWeightRule(_SglangModelWeightRule):
    """MiMoV2MTP reads MTP block and shared embed/head names."""

    role = "draft"
    compatible_main_types = (
        MiMoV2ForCausalLMWeightRule,
        MiMoV2FlashForCausalLMWeightRule,
    )

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        if "rotary_emb.inv_freq" in name or "projector" in name:
            return False
        return "mtp" in name or any(
            item in name for item in ("embed_tokens", "lm_head")
        )


class HYV4ForCausalLMWeightRule(_SglangModelWeightRule):
    """HYV4ForCausalLM.load_weights excludes model.mtp_layers."""

    role = "main"

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return not name.startswith("model.mtp_layers.")


class HYV4ForCausalLMNextNWeightRule(_SglangModelWeightRule):
    """HYV4ForCausalLMNextN.load_weights reads model.mtp_layers.0."""

    role = "draft"
    compatible_main_types = (HYV4ForCausalLMWeightRule,)

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return name.startswith("model.mtp_layers.0.")


class LongcatFlashForCausalLMWeightRule(_MtpSubstringMainSelection):
    """LongcatFlashForCausalLM.load_weights skips MTP tensors."""


class LongcatFlashForCausalLMNextNWeightRule(_SglangModelWeightRule):
    """Longcat NextN loads mapped model.mtp tensors, not target layers."""

    role = "draft"
    compatible_main_types = (LongcatFlashForCausalLMWeightRule,)

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return (
            ".mtp." in name
            and "embed_tokens" not in name
            and "shared_head.head" not in name
            and "rotary_emb.inv_freq" not in name
        )


@dataclass(frozen=True)
class _Step3p5Selection(_SglangModelWeightRule):
    base_layer: int

    def _spec_layer(self, name: str) -> int | None:
        match = re.match(r"model\.layers\.(\d+)\.", name)
        return int(match.group(1)) if match else None


class Step3p5ForCausalLMWeightRule(_Step3p5Selection):
    """Step3p5 main omits appended prediction layers."""

    role = "main"

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        layer = self._spec_layer(name)
        return layer is None or layer < self.base_layer


class Step3p7ForConditionalGenerationWeightRule(Step3p5ForCausalLMWeightRule):
    """Step3p7 delegates language weights to Step3p5 and retains vision."""

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return super().should_read_checkpoint_tensor(
            name.replace("language_model.", "", 1)
        )


@dataclass(frozen=True)
class Step3p5MTPWeightRule(_Step3p5Selection):
    """Step3p5 MTP reads one appended layer and its own embedding."""

    draft_model_idx: int = 0
    role = "draft"
    compatible_main_types = (
        Step3p5ForCausalLMWeightRule,
        Step3p7ForConditionalGenerationWeightRule,
    )

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        if "rotary_emb.inv_freq" in name:
            return False
        return self._spec_layer(name) == self.base_layer + self.draft_model_idx or (
            "embed_tokens" in name
        )

    def publishable(
        self, model: torch.nn.Module, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return tensors


class InklingForConditionalGenerationWeightRule(_SglangModelWeightRule):
    """Inkling main ignores MTP sub-model tensors."""

    role = "main"

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return "mtp." not in name and "rotary_emb.inv_freq" not in name


@dataclass(frozen=True)
class InklingForConditionalGenerationMTPWeightRule(_SglangModelWeightRule):
    """Inkling draft reads its MTP block and shared embed/head inputs."""

    draft_model_idx: int = 0
    role = "draft"
    compatible_main_types = (InklingForConditionalGenerationWeightRule,)

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        if "rotary_emb.inv_freq" in name:
            return False
        match = re.search(r"(?:^|\.)mtp\.layers\.(\d+)\.", name)
        if match:
            return int(match.group(1)) == self.draft_model_idx
        return "mtp.chain_norm." in name or name.endswith(
            (
                "embed.weight",
                "embed_tokens.weight",
                "unembed.weight",
                "lm_head.weight",
                "embed_norm.weight",
            )
        )


@dataclass(frozen=True)
class _NextNSelection(_SglangModelWeightRule):
    """Shared predicates for identical SGLang NextN load_weights branches."""

    base_layer: int

    def _draft_covers(self, name: str) -> bool:
        if "rotary_emb.inv_freq" in name:
            return False
        return name.startswith(f"model.layers.{self.base_layer}") and not (
            "shared_head.head" in name or "embed_tokens" in name
        )

    def _main_covers(self, name: str) -> bool:
        if "rotary_emb.inv_freq" in name:
            return False
        if name.startswith("model.layers"):
            parts = name.split(".")
            if len(parts) >= 3 and parts[2].isdecimal():
                return int(parts[2]) < self.base_layer
        return True


class _NextNMainSelection(_NextNSelection):
    role = "main"

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return self._main_covers(name)


class _NextNDraftSelection(_NextNSelection):
    role = "draft"

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return self._draft_covers(name)


class BailingMoeForCausalLMWeightRule(_NextNMainSelection):
    """Bailing V1 main does not own the appended NextN layer."""


class BailingMoeV2ForCausalLMWeightRule(BailingMoeForCausalLMWeightRule):
    """Bailing V2 delegates to the V1 weight-loading path."""


class _BailingMtpMainSelection(_NextNMainSelection):
    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return not name.startswith("model.mtp") and self._main_covers(name)


class BailingMoeV2_5ForCausalLMWeightRule(_BailingMtpMainSelection):
    """Bailing V2.5 skips model.mtp and its appended layer."""


class BailingMoeV3ForCausalLMWeightRule(_BailingMtpMainSelection):
    """Bailing V3 skips model.mtp and its appended layer."""


class BailingMoeForCausalLMNextNWeightRule(_NextNDraftSelection):
    """Bailing NextN delegates to the configured V1, V2.5, or V3 loader."""

    compatible_main_types = (
        BailingMoeForCausalLMWeightRule,
        BailingMoeV2ForCausalLMWeightRule,
        BailingMoeV2_5ForCausalLMWeightRule,
        BailingMoeV3ForCausalLMWeightRule,
    )

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return not name.startswith("model.mtp") and self._draft_covers(name)


class Ernie4_5_MoeForCausalLMWeightRule(_SglangModelWeightRule):
    """Ernie4.5 MoE skips separately loaded model.mtp_ weights."""

    role = "main"

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return not name.startswith("model.mtp_")


@dataclass(frozen=True)
class Ernie4_5_MoeForCausalLMMTPWeightRule(_SglangModelWeightRule):
    """Ernie4.5 MTP loads only its configured head index."""

    mtp_layer_id: int
    role = "draft"
    compatible_main_types = (Ernie4_5_MoeForCausalLMWeightRule,)

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return any(
            f"{prefix}.{self.mtp_layer_id}" in name
            for prefix in (
                "mtp_block",
                "mtp_emb_norm",
                "mtp_hidden_norm",
                "mtp_linear_proj",
            )
        )


class _NemotronHMainSelection(_SglangModelWeightRule):
    role = "main"

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return "mtp" not in name


class NemotronHForCausalLMWeightRule(_NemotronHMainSelection):
    """Nemotron-H target load."""


class NemotronHPuzzleForCausalLMWeightRule(_NemotronHMainSelection):
    """Nemotron-H Puzzle inherits the target load."""


class NemotronH_Omni_Reasoning_V3WeightRule(_NemotronHMainSelection):
    """Omni forwards language_model tensors to Nemotron-H."""


class NemotronHForCausalLMMTPWeightRule(_SglangModelWeightRule):
    """Nemotron-H draft can own a standalone, differently quantized head."""

    role = "draft"
    compatible_main_types = (
        NemotronHForCausalLMWeightRule,
        NemotronHPuzzleForCausalLMWeightRule,
        NemotronH_Omni_Reasoning_V3WeightRule,
    )

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return any(
            part in name for part in ("mtp", "embed_tokens", "embeddings", "lm_head")
        )

    def publishable(
        self, model: torch.nn.Module, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return _without_shared_embed_and_head(
            model, tensors, shared_head=not model._owns_lm_head
        )


@dataclass(frozen=True)
class _Dots3Selection(_SglangModelWeightRule):
    base_layer: int
    num_nextn_layers: int

    def _nextn_layer(self, name: str) -> bool:
        return any(
            name.startswith(f"model.layers.{layer}.")
            for layer in range(self.base_layer, self.base_layer + self.num_nextn_layers)
        )


class Dots3NoteForCausalLMWeightRule(_Dots3Selection):
    """Dots3 main load excludes appended NextN layers and model.mtp."""

    role = "main"

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return (
            "rotary_emb.inv_freq" not in name
            and not self._nextn_layer(name)
            and not name.startswith("model.mtp.")
        )


class Dots3NoteForCausalLMNextNWeightRule(_Dots3Selection):
    """Dots3 NextN keeps its own MTP embedding when checkpoint-provided."""

    role = "draft"
    compatible_main_types = (Dots3NoteForCausalLMWeightRule,)

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return "rotary_emb.inv_freq" not in name and (
            self._nextn_layer(name) or name == "model.mtp.embed_tokens.weight"
        )

    def compatible_main(self, main: DraftSelection) -> bool:
        return super().compatible_main(main) and (
            self.num_nextn_layers == main.num_nextn_layers
        )

    def publishable(
        self, model: torch.nn.Module, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return _without_shared_embed_and_head(
            model, tensors, shared_embed=not model._mtp_loaded_embed
        )


class HYV3ForCausalLMWeightRule(_NextNMainSelection):
    """HYV3ForCausalLM.load_weights omits appended decoder layers."""


class HYV3ForCausalLMNextNWeightRule(_NextNDraftSelection):
    """HYV3ForCausalLMNextN reads appended layer and shared-head norm."""

    compatible_main_types = (HYV3ForCausalLMWeightRule,)

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return super().should_read_checkpoint_tensor(name) or (
            name == "model.shared_head.norm.weight"
        )


class DeepseekV4ForCausalLMWeightRule(_NextNMainSelection):
    """DeepseekV4ForCausalLM remaps raw layers.* and omits mtp.*."""

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        if name.startswith("mtp."):
            return False
        normalized = f"model.{name}" if name.startswith("layers.") else name
        return self._main_covers(normalized)


class DeepseekV4ForCausalLMNextNWeightRule(_NextNDraftSelection):
    """DeepseekV4ForCausalLMNextN reads its raw MTP or extra layer."""

    compatible_main_types = (DeepseekV4ForCausalLMWeightRule,)

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        if name.startswith("mtp."):
            return not any(part in name for part in (".emb.tok_emb", ".head."))
        normalized = f"model.{name}" if name.startswith("layers.") else name
        return self._draft_covers(normalized)


class DeepseekV4ForCausalLMDSparkWeightRule(_NextNSelection):
    """DeepseekV4ForCausalLMDSpark reads the bundled mtp.* stages."""

    role = "draft"
    compatible_main_types = (DeepseekV4ForCausalLMWeightRule,)

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return name.startswith("mtp.") and len(name.split(".", 2)) == 3

    def publishable(
        self, model: torch.nn.Module, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return tensors


class Glm5NextForConditionalGenerationWeightRule(_NextNMainSelection):
    """GLM5 main loads language_model.* or model.* decoder names."""

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return self._main_covers(name.replace("language_model.", "", 1))


class Glm5NextForConditionalGenerationNextNWeightRule(_NextNDraftSelection):
    """GLM5 draft first keeps only its extra layer from either raw prefix."""

    compatible_main_types = (Glm5NextForConditionalGenerationWeightRule,)

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        normalized = name.replace("model.language_model.", "model.", 1)
        return self._draft_covers(normalized) and "hc_head" not in name


class GlmOcrForConditionalGenerationWeightRule(_NextNMainSelection):
    """GLM-OCR main remaps model.language_model before the layer check."""

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return self._main_covers(name.replace("model.language_model.", "model.", 1))


class GlmOcrForConditionalGenerationNextNWeightRule(_NextNDraftSelection):
    """GLM-OCR draft delegates to its main load loop with is_nextn=True."""

    compatible_main_types = (GlmOcrForConditionalGenerationWeightRule,)

    def should_read_checkpoint_tensor(self, name: str) -> bool:
        return self._draft_covers(name.replace("model.language_model.", "model.", 1))


class GigaChat35ForCausalLMWeightRule(_NextNMainSelection):
    """GigaChat35ForCausalLM delegates to DeepseekV2WeightLoaderMixin."""


class GigaChat35ForCausalLMNextNWeightRule(_NextNDraftSelection):
    """GigaChat35ForCausalLMNextN delegates to the same mixin."""

    compatible_main_types = (GigaChat35ForCausalLMWeightRule,)


def _without_shared_embed_and_head(
    model: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
    *,
    shared_embed: bool = True,
    shared_head: bool = True,
) -> dict[str, torch.Tensor]:
    get_shared = getattr(model, "get_embed_and_head", None)
    if not callable(get_shared):
        raise TypeError("MTP draft does not expose get_embed_and_head")
    embed, head = get_shared()
    shared = {
        tensor.untyped_storage().data_ptr()
        for tensor, is_shared in ((embed, shared_embed), (head, shared_head))
        if is_shared and tensor.numel() > 0
    }
    return {
        name: tensor
        for name, tensor in tensors.items()
        if tensor.numel() == 0 or tensor.untyped_storage().data_ptr() not in shared
    }


class DeepseekV3ForCausalLMWeightRule(_NextNMainSelection):
    """DeepseekV3ForCausalLM, via DeepseekV2WeightLoaderMixin."""


class DeepseekV32ForCausalLMWeightRule(_NextNMainSelection):
    """DeepseekV32ForCausalLM uses the DeepSeek V2 weight-loader mixin."""


class DeepseekV3ForCausalLMNextNWeightRule(_NextNDraftSelection):
    """DeepseekV3ForCausalLMNextN delegates to the same mixin with is_nextn."""

    compatible_main_types = (
        DeepseekV3ForCausalLMWeightRule,
        DeepseekV32ForCausalLMWeightRule,
    )


class Glm4MoeForCausalLMWeightRule(_NextNMainSelection):
    """Glm4MoeForCausalLM.load_weights."""


class Glm4MoeForCausalLMNextNWeightRule(_NextNDraftSelection):
    """Glm4MoeForCausalLMNextN delegates with is_nextn=True."""

    compatible_main_types = (Glm4MoeForCausalLMWeightRule,)


class Glm4MoeLiteForCausalLMWeightRule(_NextNMainSelection):
    """Glm4MoeLiteForCausalLM.load_weights."""


class Glm4MoeLiteForCausalLMNextNWeightRule(_NextNDraftSelection):
    """Glm4MoeLiteForCausalLMNextN delegates with is_nextn=True."""

    compatible_main_types = (Glm4MoeLiteForCausalLMWeightRule,)


class GlmMoeDsaForCausalLMWeightRule(_NextNMainSelection):
    """GlmMoeDsaForCausalLM inherits the DeepSeek weight-loader mixin."""


class GlmMoeDsaForCausalLMNextNWeightRule(_NextNDraftSelection):
    """GlmMoeDsaForCausalLMNextN inherits the DeepSeek NextN loader."""

    compatible_main_types = (GlmMoeDsaForCausalLMWeightRule,)


def draft_weight_adapter_for(
    class_name: str, *, role: str, model: object | None = None
) -> DraftSelection | None:
    if role not in {"main", "draft"}:
        return None
    fixed_rules = {
        "Qwen3_5ForCausalLM": Qwen3_5ForCausalLMWeightRule,
        "Qwen3_5MoeForCausalLM": Qwen3_5MoeForCausalLMWeightRule,
        "Qwen3_5ForConditionalGeneration": Qwen3_5ForConditionalGenerationWeightRule,
        "Qwen3_5MoeForConditionalGeneration": Qwen3_5MoeForConditionalGenerationWeightRule,
        "InternS2PreviewForConditionalGeneration": InternS2PreviewForConditionalGenerationWeightRule,
        "InternS2MobiusForConditionalGeneration": InternS2MobiusForConditionalGenerationWeightRule,
        "Qwen3_5ForCausalLMMTP": Qwen3_5ForCausalLMMTPWeightRule,
        "Qwen4ExpForConditionalGeneration": Qwen4ExpForConditionalGenerationWeightRule,
        "Qwen4ExpForCausalLMMTP": Qwen4ExpForCausalLMMTPWeightRule,
        "Qwen3MoeForCausalLM": Qwen3MoeForCausalLMWeightRule,
        "Qwen3MoeForCausalLMMTP": Qwen3MoeForCausalLMMTPWeightRule,
        "Qwen3NextForCausalLM": Qwen3NextForCausalLMWeightRule,
        "Qwen3NextForCausalLMMTP": Qwen3NextForCausalLMMTPWeightRule,
        "ExaoneMoEForCausalLM": ExaoneMoEForCausalLMWeightRule,
        "ExaoneMoeForCausalLM": ExaoneMoeForCausalLMWeightRule,
        "ExaoneMoEForCausalLMMTP": ExaoneMoEForCausalLMMTPWeightRule,
        "MiMoForCausalLM": MiMoForCausalLMWeightRule,
        "MiMoMTP": MiMoMTPWeightRule,
        "MiMoV2ForCausalLM": MiMoV2ForCausalLMWeightRule,
        "MiMoV2FlashForCausalLM": MiMoV2FlashForCausalLMWeightRule,
        "MiMoV2MTP": MiMoV2MTPWeightRule,
        "HYV4ForCausalLM": HYV4ForCausalLMWeightRule,
        "HYV4ForCausalLMNextN": HYV4ForCausalLMNextNWeightRule,
        "LongcatFlashForCausalLM": LongcatFlashForCausalLMWeightRule,
        "LongcatFlashForCausalLMNextN": LongcatFlashForCausalLMNextNWeightRule,
        "InklingForConditionalGeneration": InklingForConditionalGenerationWeightRule,
        "Ernie4_5_MoeForCausalLM": Ernie4_5_MoeForCausalLMWeightRule,
        "NemotronHForCausalLM": NemotronHForCausalLMWeightRule,
        "NemotronHPuzzleForCausalLM": NemotronHPuzzleForCausalLMWeightRule,
        "NemotronH_Omni_Reasoning_V3": NemotronH_Omni_Reasoning_V3WeightRule,
        "NemotronHForCausalLMMTP": NemotronHForCausalLMMTPWeightRule,
    }
    fixed_rule = fixed_rules.get(class_name)
    if fixed_rule is not None:
        rule = (
            fixed_rule(
                language_model_only=bool(getattr(model, "language_model_only", False))
            )
            if fixed_rule is Qwen4ExpForConditionalGenerationWeightRule
            else fixed_rule()
        )
        return rule if rule.role == role else None
    if class_name == "Ernie4_5_MoeForCausalLMMTP" and role == "draft":
        layer_id = getattr(model, "mtp_layer_id", None)
        if (
            isinstance(layer_id, int)
            and not isinstance(layer_id, bool)
            and layer_id >= 0
        ):
            return Ernie4_5_MoeForCausalLMMTPWeightRule(layer_id)
        return None
    if class_name == "InklingForConditionalGenerationMTP" and role == "draft":
        draft_idx = getattr(model, "draft_model_idx", None)
        if (
            isinstance(draft_idx, int)
            and not isinstance(draft_idx, bool)
            and draft_idx >= 0
        ):
            return InklingForConditionalGenerationMTPWeightRule(draft_idx)
        return None
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    base = getattr(text_config, "num_hidden_layers", None)
    nextn_layers = getattr(config, "num_nextn_predict_layers", None)
    if nextn_layers is None:
        nextn_layers = getattr(text_config, "num_nextn_predict_layers", None)
    if (
        class_name
        in {"Step3p5ForCausalLM", "Step3p7ForConditionalGeneration", "Step3p5MTP"}
        and isinstance(base, int)
        and not isinstance(base, bool)
        and base > 1
        and nextn_layers == 1
    ):
        if class_name == "Step3p5ForCausalLM" and role == "main":
            return Step3p5ForCausalLMWeightRule(base)
        if class_name == "Step3p7ForConditionalGeneration" and role == "main":
            return Step3p7ForConditionalGenerationWeightRule(base)
        if class_name == "Step3p5MTP" and role == "draft":
            draft_idx = getattr(model, "draft_model_idx", None)
            if draft_idx == 0 and not isinstance(draft_idx, bool):
                return Step3p5MTPWeightRule(base, draft_idx)
        return None
    if (
        class_name in {"Dots3NoteForCausalLM", "Dots3NoteForCausalLMNextN"}
        and isinstance(base, int)
        and not isinstance(base, bool)
        and base > 1
        and isinstance(nextn_layers, int)
        and not isinstance(nextn_layers, bool)
        and nextn_layers > 0
    ):
        rule_type = (
            Dots3NoteForCausalLMWeightRule
            if class_name == "Dots3NoteForCausalLM"
            else Dots3NoteForCausalLMNextNWeightRule
        )
        rule = rule_type(base, nextn_layers)
        return rule if rule.role == role else None
    if (
        isinstance(base, bool)
        or not isinstance(base, int)
        or base <= 1
        or nextn_layers != 1
        or isinstance(nextn_layers, bool)
    ):
        return None
    if role == "main":
        bailing_main = {
            "BailingMoeForCausalLM": BailingMoeForCausalLMWeightRule,
            "BailingMoeV2ForCausalLM": BailingMoeV2ForCausalLMWeightRule,
            "BailingMoeV2_5ForCausalLM": BailingMoeV2_5ForCausalLMWeightRule,
            "BailingMoeV3ForCausalLM": BailingMoeV3ForCausalLMWeightRule,
        }.get(class_name)
        if bailing_main is not None:
            return bailing_main(base)
        if class_name == "DeepseekV4ForCausalLM":
            return DeepseekV4ForCausalLMWeightRule(base)
        if class_name == "Glm5NextForConditionalGeneration":
            return Glm5NextForConditionalGenerationWeightRule(base)
        if class_name == "GlmOcrForConditionalGeneration":
            return GlmOcrForConditionalGenerationWeightRule(base)
        if class_name == "GigaChat35ForCausalLM":
            return GigaChat35ForCausalLMWeightRule(base)
        if class_name == "HYV3ForCausalLM":
            return HYV3ForCausalLMWeightRule(base)
        if class_name == "DeepseekV3ForCausalLM":
            return DeepseekV3ForCausalLMWeightRule(base)
        if class_name == "DeepseekV32ForCausalLM":
            return DeepseekV32ForCausalLMWeightRule(base)
        if class_name == "Glm4MoeForCausalLM":
            return Glm4MoeForCausalLMWeightRule(base)
        if class_name == "Glm4MoeLiteForCausalLM":
            return Glm4MoeLiteForCausalLMWeightRule(base)
        if class_name == "GlmMoeDsaForCausalLM":
            return GlmMoeDsaForCausalLMWeightRule(base)
    else:
        if class_name == "BailingMoeForCausalLMNextN":
            return BailingMoeForCausalLMNextNWeightRule(base)
        if class_name == "DeepseekV4ForCausalLMNextN":
            return DeepseekV4ForCausalLMNextNWeightRule(base)
        if class_name == "DeepseekV4ForCausalLMDSpark":
            return DeepseekV4ForCausalLMDSparkWeightRule(base)
        if class_name == "Glm5NextForConditionalGenerationNextN":
            return Glm5NextForConditionalGenerationNextNWeightRule(base)
        if class_name == "GlmOcrForConditionalGenerationNextN":
            return GlmOcrForConditionalGenerationNextNWeightRule(base)
        if class_name == "GigaChat35ForCausalLMNextN":
            return GigaChat35ForCausalLMNextNWeightRule(base)
        if class_name == "HYV3ForCausalLMNextN":
            return HYV3ForCausalLMNextNWeightRule(base)
        if class_name == "DeepseekV3ForCausalLMNextN":
            return DeepseekV3ForCausalLMNextNWeightRule(base)
        if class_name == "Glm4MoeForCausalLMNextN":
            return Glm4MoeForCausalLMNextNWeightRule(base)
        if class_name == "Glm4MoeLiteForCausalLMNextN":
            return Glm4MoeLiteForCausalLMNextNWeightRule(base)
        if class_name == "GlmMoeDsaForCausalLMNextN":
            return GlmMoeDsaForCausalLMNextNWeightRule(base)
    return None


def draft_tensor_namespace(
    identity,
    main_adapter: DraftSelection,
    draft_adapter: DraftSelection,
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
            "main_selection": type(main_adapter).__name__,
            "draft_selection": type(draft_adapter).__name__,
            "draft_base_layer": getattr(draft_adapter, "base_layer", None),
            "draft_nextn_layers": getattr(draft_adapter, "num_nextn_layers", None),
            "draft_model_idx": getattr(draft_adapter, "draft_model_idx", None),
            "mtp_layer_id": getattr(draft_adapter, "mtp_layer_id", None),
            "draft_model": model_class,
            "sglang": sglang_version,
            "weights_uri": model_uri,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"mx_draft::{hashlib.sha256(signature).hexdigest()[:32]}::"
