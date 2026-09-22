# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patch vLLM's object-storage load-format validation."""

from __future__ import annotations

from ._version import installed_vllm_minor

_PLUGIN_LOAD_FORMATS = ("modelexpress", "mx")

# Native ModelExpress object-storage validation shipped in vLLM 0.22.0.
_LATEST_AFFECTED_MINOR = (0, 21)


def patch_object_storage_format_check() -> bool:
    """Allow ModelExpress load formats for object storage on older vLLM.

    Upstream fix: https://github.com/vllm-project/vllm/pull/43105
    """
    try:
        from vllm.config import VllmConfig
        from vllm.transformers_utils.runai_utils import is_runai_obj_uri
    except ImportError:
        return False

    installed_minor = installed_vllm_minor()
    if installed_minor is None or installed_minor > _LATEST_AFFECTED_MINOR:
        return False

    original = VllmConfig.try_verify_and_update_config
    if getattr(original, "__modelexpress_patched__", False):
        return False

    def patched(self: VllmConfig) -> None:
        if (
            self.load_config.load_format in _PLUGIN_LOAD_FORMATS
            and hasattr(self.model_config, "model_weights")
            and is_runai_obj_uri(self.model_config.model_weights)
        ):
            saved = self.model_config.model_weights
            del self.model_config.model_weights
            try:
                original(self)
            finally:
                self.model_config.model_weights = saved
        else:
            original(self)

    patched.__modelexpress_patched__ = True
    VllmConfig.try_verify_and_update_config = patched
    return True
