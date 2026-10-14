# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import sys
from types import ModuleType, SimpleNamespace

from modelexpress.engines.vllm.patches import _version as vllm_patch_version
from modelexpress.engines.vllm.patches.patch_humming_regex_ignore import (
    patch_humming_regex_ignore,
)
from modelexpress.engines.vllm.patches.patch_object_storage_format_check import (
    patch_object_storage_format_check,
)
from modelexpress.patches import apply_patches


def test_apply_patches_runs_in_order(monkeypatch):
    monkeypatch.delenv("MX_DISABLE_PATCHES", raising=False)
    calls = []

    def first():
        calls.append("first")
        return False

    def second():
        calls.append("second")
        return True

    apply_patches((first, second))

    assert calls == ["first", "second"]


def test_apply_patches_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MX_DISABLE_PATCHES", "true")
    calls = []

    def patch():
        calls.append("patch")
        return True

    apply_patches((patch,))

    assert calls == []


def test_vllm_entrypoint_configures_logging_before_patches(monkeypatch):
    from modelexpress.engines import vllm as vllm_integration
    from modelexpress.engines.vllm import registration

    calls = []
    monkeypatch.setattr(vllm_integration, "_loaders_registered", False)
    monkeypatch.setattr(
        vllm_integration,
        "configure_vllm_logging",
        lambda: calls.append(("logging", None)),
    )
    monkeypatch.setattr(
        vllm_integration,
        "apply_patches",
        lambda patches: calls.append(("patches", patches)),
    )
    monkeypatch.setattr(
        registration,
        "register_plugin_model_loader",
        lambda: calls.append(("loader", None)),
    )

    vllm_integration.register_modelexpress_loaders()

    assert calls == [
        ("logging", None),
        ("patches", vllm_integration.VLLM_PATCHES),
        ("loader", None),
    ]


def test_object_storage_patch_preserves_model_weights(monkeypatch):
    calls = []

    class VllmConfig:
        def try_verify_and_update_config(self):
            calls.append(hasattr(self.model_config, "model_weights"))

    config_module = ModuleType("vllm.config")
    config_module.VllmConfig = VllmConfig
    transformers_module = ModuleType("vllm.transformers_utils")
    runai_module = ModuleType("vllm.transformers_utils.runai_utils")
    runai_module.is_runai_obj_uri = lambda uri: uri.startswith("s3://")
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(
        sys.modules,
        transformers_module.__name__,
        transformers_module,
    )
    monkeypatch.setitem(sys.modules, runai_module.__name__, runai_module)
    monkeypatch.setattr(
        vllm_patch_version,
        "package_version",
        lambda name: "0.21.0",
    )

    assert patch_object_storage_format_check() is True

    config = VllmConfig()
    config.load_config = SimpleNamespace(load_format="modelexpress")
    config.model_config = SimpleNamespace(model_weights="s3://bucket/model")
    config.try_verify_and_update_config()

    assert calls == [False]
    assert config.model_config.model_weights == "s3://bucket/model"
    assert patch_object_storage_format_check() is False


def test_object_storage_patch_skips_unknown_version(monkeypatch):
    class VllmConfig:
        def try_verify_and_update_config(self):
            pass

    config_module = ModuleType("vllm.config")
    config_module.VllmConfig = VllmConfig
    transformers_module = ModuleType("vllm.transformers_utils")
    runai_module = ModuleType("vllm.transformers_utils.runai_utils")
    runai_module.is_runai_obj_uri = lambda uri: True
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(
        sys.modules,
        transformers_module.__name__,
        transformers_module,
    )
    monkeypatch.setitem(sys.modules, runai_module.__name__, runai_module)
    monkeypatch.setattr(
        vllm_patch_version,
        "package_version",
        lambda name: "0.22.0",
    )

    original = VllmConfig.try_verify_and_update_config
    assert patch_object_storage_format_check() is False
    assert VllmConfig.try_verify_and_update_config is original


class _BrokenHummingConfig:
    def get_from_keys_or(self, config, keys, default):
        return next((config[key] for key in keys if key in config), default)

    def is_layer_skipped(self, config, prefix):
        ignored = self.get_from_keys_or(
            config,
            ["ignored_layers", "ignore", "modules_to_not_convert"],
            [],
        )
        return any(module_name in prefix for module_name in ignored)


def _install_fake_humming(monkeypatch, version, config_cls):
    monkeypatch.setattr(
        vllm_patch_version,
        "package_version",
        lambda name: version,
    )

    humming = ModuleType("vllm.model_executor.layers.quantization.humming")
    humming.HummingConfig = config_cls
    humming.HummingMethod = object
    humming.re = re
    monkeypatch.setitem(sys.modules, humming.__name__, humming)


def test_humming_patch_adds_regex_and_preserves_substring_matching(monkeypatch):
    class HummingConfig(_BrokenHummingConfig):
        pass

    _install_fake_humming(monkeypatch, "0.25.1", HummingConfig)

    assert patch_humming_regex_ignore() is True
    config = HummingConfig()
    assert config.is_layer_skipped(
        {"ignore": ["re:^vision_tower.*"]},
        "vision_tower.encoder.0",
    )
    assert config.is_layer_skipped(
        {"modules_to_not_convert": ["lm_head"]},
        "model.lm_head",
    )
    assert patch_humming_regex_ignore() is False


def test_humming_patch_detects_customer_backport(monkeypatch):
    class HummingConfig(_BrokenHummingConfig):
        def is_layer_skipped(self, config, prefix):
            ignored = self.get_from_keys_or(config, ["ignore"], [])
            return any(
                module_name.startswith("re:")
                and re.match(module_name[3:], prefix)
                for module_name in ignored
            )

    _install_fake_humming(monkeypatch, "0.23.0", HummingConfig)

    assert patch_humming_regex_ignore() is False


def test_humming_patch_skips_when_backend_is_unavailable(monkeypatch):
    class HummingConfig:
        def __init__(self):
            raise AssertionError("Humming backend is unavailable")

    monkeypatch.setattr(
        vllm_patch_version,
        "package_version",
        lambda name: "0.23.0",
    )
    humming = ModuleType("vllm.model_executor.layers.quantization.humming")
    humming.HummingConfig = HummingConfig
    humming.HummingMethod = None
    monkeypatch.setitem(sys.modules, humming.__name__, humming)

    assert patch_humming_regex_ignore() is False


def test_humming_patch_skips_unknown_broken_version(monkeypatch):
    class HummingConfig(_BrokenHummingConfig):
        pass

    _install_fake_humming(monkeypatch, "0.26.0", HummingConfig)

    assert patch_humming_regex_ignore() is False
