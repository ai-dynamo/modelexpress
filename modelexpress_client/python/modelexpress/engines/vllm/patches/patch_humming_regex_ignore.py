# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patch Humming's handling of compressed-tensors regex ignore entries."""

from importlib import import_module

from ._version import installed_vllm_minor

# vLLM 0.25.1 is the latest release with the broken substring matching.
_LATEST_AFFECTED_MINOR = (0, 25)


def patch_humming_regex_ignore() -> bool:
    """Backport vLLM's handling of ``re:`` Humming ignore entries.

    Upstream fix: https://github.com/vllm-project/vllm/pull/48507
    """
    installed_minor = installed_vllm_minor()
    if installed_minor is None or installed_minor > _LATEST_AFFECTED_MINOR:
        return False

    try:
        humming = import_module(
            "vllm.model_executor.layers.quantization.humming"
        )
    except ImportError:
        return False

    if getattr(humming, "HummingMethod", None) is None:
        return False

    config_cls = humming.HummingConfig
    original = config_cls.is_layer_skipped
    if getattr(original, "__modelexpress_patched__", False):
        return False

    probe_config = {"ignore": ["re:^mx_humming_probe$"]}
    probe_prefix = "mx_humming_probe"

    # Prefer behavior detection over the package version. This also recognizes
    # customer images that backported the upstream fix without changing their
    # vLLM version.
    if config_cls().is_layer_skipped(probe_config, probe_prefix):
        return False

    def patched(self, config, prefix):
        keys = ["ignored_layers", "ignore", "modules_to_not_convert"]
        ignored_layers = self.get_from_keys_or(config, keys, []) or []
        if hasattr(self, "hf_to_vllm_mapper"):
            ignored_layers = self.hf_to_vllm_mapper.apply_list(ignored_layers)

        for module_name in ignored_layers:
            if module_name.startswith("re:") and humming.re.match(
                module_name[3:], prefix
            ):
                return True

        return original(self, config, prefix)

    patched.__modelexpress_patched__ = True
    config_cls.is_layer_skipped = patched

    if not config_cls().is_layer_skipped(probe_config, probe_prefix):
        raise RuntimeError("Failed to apply the Humming regex-ignore patch")

    return True
