# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM compatibility integration for ModelExpress."""

from .adapter import VllmAdapter, build_vllm_load_context

_loaders_registered = False


def register_modelexpress_loaders() -> None:
    """Register ModelExpress's vLLM loader."""
    global _loaders_registered
    if _loaders_registered:
        return
    from . import loader  # noqa: F401

    _loaders_registered = True


__all__ = [
    "VllmAdapter",
    "build_vllm_load_context",
    "register_modelexpress_loaders",
]
