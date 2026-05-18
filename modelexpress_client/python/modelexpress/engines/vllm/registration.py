# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin registration for vLLM versions without native ModelExpress support."""

from __future__ import annotations

import logging

from vllm.model_executor.model_loader import register_model_loader

logger = logging.getLogger(__name__)


def _patch_vllm_s3_format_check() -> None:
    """Allow 'mx' as a valid load format for object storage on older vLLM."""
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
            self.load_config.load_format == "mx"
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
        "Patched VllmConfig.try_verify_and_update_config to allow 'mx' "
        "for object storage URIs"
    )


def register_plugin_model_loader() -> None:
    """Register `mx` through vLLM's plugin registry when vLLM lacks native MX."""
    import vllm.model_executor.model_loader as model_loader

    if "mx" in model_loader._LOAD_FORMAT_TO_MODEL_LOADER:
        # Native vLLM integration owns `mx` in newer versions; the plugin path
        # stays active only for older vLLM releases.
        logger.debug("vLLM already provides native 'mx' loader registration")
        return

    # Older vLLM still needs the plugin registration side effects that used to
    # happen when importing loader.py directly.
    _patch_vllm_s3_format_check()

    from .loader import MxModelLoader

    register_model_loader("mx")(MxModelLoader)
