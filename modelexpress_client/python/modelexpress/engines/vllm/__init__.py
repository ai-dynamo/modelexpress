# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM compatibility integration for ModelExpress."""

from .adapter import VllmAdapter, build_vllm_load_context

_loaders_registered = False


def register_modelexpress_loaders() -> None:
    """Register ModelExpress's vLLM loader for plugin-based vLLM integration."""
    global _loaders_registered
    if _loaders_registered:
        return
    from .registration import register_plugin_model_loader

    # Needed for older vLLM versions before native ModelExpress loader
    # registration is available.
    register_plugin_model_loader()

    _loaders_registered = True


__all__ = [
    "VllmAdapter",
    "build_vllm_load_context",
    "register_modelexpress_loaders",
]
