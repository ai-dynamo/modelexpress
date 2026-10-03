# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which canonical tensors an SGLang engine can receive pre-split.

Trainer and receiver share ``destination_shard_dim``: one rule on both sides
keeps a ``Shard`` destination from ever landing in whole-tensor storage. The
rule is a fact about SGLang's dense Qwen2/Qwen3 layers at the pinned fork
base, not a property of the transport. Nothing here imports torch or SGLang.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: SGLang's DEFAULT_VOCAB_PADDING_SIZE: a padded table carries rows the
#: checkpoint does not, so it cannot be received pre-split.
VOCAB_PADDING = 64

_LAYER = r"model\.layers\.\d+\."
_ATTENTION_RE = re.compile(_LAYER + r"self_attn\.(q_proj|k_proj|v_proj)\.weight")
_GATE_UP_RE = re.compile(_LAYER + r"mlp\.(gate_proj|up_proj)\.weight")
_ROW_RE = re.compile(_LAYER + r"(self_attn\.o_proj|mlp\.down_proj)\.weight")
_VOCAB_NAMES = ("model.embed_tokens.weight", "lm_head.weight")


@dataclass(frozen=True)
class SglangModelFacts:
    """The model-config facts the destination rule depends on."""

    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    tie_word_embeddings: bool

    def __post_init__(self) -> None:
        for label in ("num_attention_heads", "num_key_value_heads", "vocab_size"):
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer, got {value!r}")

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | Any) -> SglangModelFacts:
        """Read the facts from an HF ``config.json`` mapping or config object."""

        def read(key: str, default: Any = None) -> Any:
            if isinstance(config, Mapping):
                text = config.get("text_config")
                source = text if isinstance(text, Mapping) else config
                return source.get(key, config.get(key, default))
            text = getattr(config, "text_config", None)
            source = text if text is not None else config
            return getattr(source, key, getattr(config, key, default))

        heads = read("num_attention_heads")
        return cls(
            num_attention_heads=heads,
            num_key_value_heads=read("num_key_value_heads", heads),
            vocab_size=read("vocab_size"),
            tie_word_embeddings=bool(read("tie_word_embeddings", False)),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str) -> SglangModelFacts:
        path = os.path.join(checkpoint_dir, "config.json")
        with open(path, encoding="utf-8") as handle:
            return cls.from_config(json.load(handle))


def destination_shard_dim(
    name: str,
    global_shape: tuple[int, ...],
    engine_tp: int,
    facts: SglangModelFacts,
) -> int | None:
    """The tensor dim an SGLang engine of ``engine_tp`` ranks splits ``name`` on.

    None means the engine needs the whole tensor, which the receiver installs
    through SGLang's own ``load_weights``. Details: the integrations README's
    supported envelope.
    """
    if engine_tp <= 0:
        raise ValueError(f"engine_tp must be positive, got {engine_tp}")
    if engine_tp == 1 or len(global_shape) != 2:
        return None
    rows, cols = global_shape
    if _ATTENTION_RE.fullmatch(name):
        heads = facts.num_attention_heads
        kv_heads = facts.num_key_value_heads
        if heads % engine_tp or kv_heads % engine_tp:
            return None
        per_head = heads if name.endswith("q_proj.weight") else kv_heads
        if rows % per_head:
            return None
        return 0
    if _GATE_UP_RE.fullmatch(name):
        return 0 if rows % engine_tp == 0 else None
    if _ROW_RE.fullmatch(name):
        return 1 if cols % engine_tp == 0 else None
    if name in _VOCAB_NAMES:
        if name == "lm_head.weight" and facts.tie_word_embeddings:
            return None
        vocab = facts.vocab_size
        if rows != vocab or vocab % VOCAB_PADDING or vocab % engine_tp:
            return None
        return 0
    return None


__all__ = ["SglangModelFacts", "VOCAB_PADDING", "destination_shard_dim"]
