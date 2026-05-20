# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin registration for vLLM versions without native ModelExpress support."""

from __future__ import annotations

import logging

from vllm.model_executor.model_loader import register_model_loader

logger = logging.getLogger(__name__)


_PLUGIN_LOAD_FORMATS = ("modelexpress", "mx")


def _patch_vllm_s3_format_check() -> None:
    """Allow ModelExpress load formats for object storage on older vLLM."""
    try:
        from vllm.config import VllmConfig
        from vllm.transformers_utils.runai_utils import is_runai_obj_uri
    except ImportError:
        return

    original = VllmConfig.try_verify_and_update_config
    if getattr(original, "__modelexpress_patched__", False):
        return

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
    logger.debug(
        "Patched VllmConfig.try_verify_and_update_config to allow ModelExpress "
        "for object storage URIs"
    )


def register_plugin_model_loader() -> None:
    """Register ModelExpress loaders through vLLM's plugin registry."""
    import vllm.model_executor.model_loader as model_loader

    # Older vLLM still needs the plugin registration side effects that used to
    # happen when importing loader.py directly.
    _patch_vllm_s3_format_check()

    from .loader import MxModelLoader

    for load_format in _PLUGIN_LOAD_FORMATS:
        if load_format in model_loader._LOAD_FORMAT_TO_MODEL_LOADER:
            logger.debug(
                "vLLM already provides '%s' loader registration", load_format
            )
            continue
        register_model_loader(load_format)(MxModelLoader)
