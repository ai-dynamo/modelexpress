# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audited automatic checkpoint mappings; other architectures bind explicitly."""

import hashlib
import inspect
from pathlib import Path

from .partial_checkpoint import _require


def default_checkpoint_mapping(model) -> dict[str, str]:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention
    from vllm.model_executor.models.kimi_k25 import KimiK25ForConditionalGeneration

    _require(
        type(model) is KimiK25ForConditionalGeneration,
        "architecture requires an explicit checkpoint tensor mapping",
    )
    try:
        digest = hashlib.sha256(
            Path(inspect.getfile(KimiK25ForConditionalGeneration)).read_bytes()
        ).hexdigest()
    except (OSError, TypeError):
        digest = None
    _require(
        digest == "1bbce9c894945a6b95818181b52cd066debb39eb8f47334ecff6d870ead5326b",
        "automatic checkpoint mapping source differs from the audited tag",
    )
    embedding = "language_model.model.embed_tokens"
    mapping = {f"{embedding}.weight": embedding}
    for path, module in model.named_modules():
        if type(module) is DeepseekV2MLAAttention:
            projection = f"{path}.kv_b_proj"
            mapping[f"{projection}.weight"] = projection
    return mapping
