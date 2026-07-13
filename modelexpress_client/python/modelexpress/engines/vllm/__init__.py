# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM compatibility integration for ModelExpress."""

from .adapter import VllmAdapter, build_vllm_load_context

_loaders_registered = False


def register_modelexpress_loaders() -> None:
    """Register ModelExpress's vLLM integrations for plugin-based vLLM.

    Invoked automatically at vLLM startup via the ``vllm.general_plugins``
    entry point, so simply installing modelexpress makes both integrations
    available with no import line in the host application:
      * the ``mx`` / ``modelexpress`` model **load format**, and
      * the ``mx`` **weight-transfer backend** (``WeightTransferConfig(backend="mx")``).
    """
    global _loaders_registered
    if _loaders_registered:
        return
    from .registration import register_plugin_model_loader

    # Needed for older vLLM versions before native ModelExpress loader
    # registration is available.
    register_plugin_model_loader()

    # Register the "mx" weight-transfer backend so it's selectable via
    # WeightTransferConfig(backend="mx") without any host-app import. Guarded:
    # older vLLM without WeightTransferEngineFactory simply skips it.
    try:
        from .weight_transfer import register as _register_mx_weight_transfer

        _register_mx_weight_transfer()
    except Exception:  # noqa: BLE001
        pass

    _loaders_registered = True


__all__ = [
    "VllmAdapter",
    "build_vllm_load_context",
    "register_modelexpress_loaders",
]
