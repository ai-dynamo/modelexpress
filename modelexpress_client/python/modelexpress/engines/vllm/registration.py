# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin registration for vLLM versions without native ModelExpress support."""

from __future__ import annotations

import logging

from vllm.model_executor.model_loader import register_model_loader

logger = logging.getLogger(__name__)


_PLUGIN_LOAD_FORMATS = ("modelexpress", "mx")


def register_plugin_model_loader() -> None:
    """Register ModelExpress loaders through vLLM's plugin registry."""
    import vllm.model_executor.model_loader as model_loader

    from .loader import MxModelLoader

    for load_format in _PLUGIN_LOAD_FORMATS:
        if load_format in model_loader._LOAD_FORMAT_TO_MODEL_LOADER:
            logger.debug(
                "vLLM already provides '%s' loader registration", load_format
            )
            continue
        register_model_loader(load_format)(MxModelLoader)
