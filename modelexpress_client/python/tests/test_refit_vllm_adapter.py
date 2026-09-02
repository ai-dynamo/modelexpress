# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

import torch

from modelexpress import p2p_pb2
from modelexpress_rl.inference.engines.vllm import (
    VllmGeneratorContext,
    _create_vllm_engine_runtime,
)


def test_vllm_engine_runtime_exposes_installation_and_full_tensor_geometry(
    monkeypatch,
):
    class ModelConfig:
        model = "test/model"

    class VllmConfig:
        model_config = ModelConfig()

    config_module = ModuleType("vllm.config")
    config_module.ModelConfig = ModelConfig
    config_module.VllmConfig = VllmConfig
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)

    class Engine:
        accelerator_backend = SimpleNamespace(name="cuda")

        def __init__(self, vllm_config, model_config):
            assert vllm_config is config
            assert model_config is config.model_config

        def get_device_id(self):
            return 2

        def get_target_device(self):
            return torch.device("cuda:2")

        def get_worker_rank(self):
            return 3

        def build_identity(self):
            return p2p_pb2.SourceIdentity(model_name="test/model")

    class Installer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def capture(self, manifest):
            return manifest

        def parameter_layout(self):
            return {"weight": ((4,), torch.float32)}

    adapter_module = ModuleType("modelexpress.engines.vllm.adapter")
    adapter_module.VllmAdapter = Engine
    monkeypatch.setitem(
        sys.modules, "modelexpress.engines.vllm.adapter", adapter_module
    )
    installer_module = ModuleType(
        "modelexpress_rl.inference.engines.vllm.installer"
    )
    installer_module._VllmInstaller = Installer
    monkeypatch.setitem(
        sys.modules,
        "modelexpress_rl.inference.engines.vllm.installer",
        installer_module,
    )

    model = torch.nn.Linear(4, 4)
    config = VllmConfig()
    convert_native_to_hf = lambda weights: weights
    runtime = _create_vllm_engine_runtime(
        VllmGeneratorContext(
            model,
            config,
            convert_native_to_hf=convert_native_to_hf,
        )
    )

    assert runtime.model_name == "test/model"
    assert runtime.installer.kwargs == {
        "model": model,
        "vllm_config": config,
        "model_config": config.model_config,
        "device": torch.device("cuda:2"),
        "convert_native_to_hf": convert_native_to_hf,
    }
    assert runtime.full_tensor is not None
    assert runtime.full_tensor.device_id == 2
    assert runtime.full_tensor.worker_rank == 3
    assert runtime.full_tensor.accelerator == "cuda"
    assert runtime.full_tensor.capture_layout(["manifest"]) == ["manifest"]
    assert runtime.full_tensor.parameter_layout() == {
        "weight": ((4,), torch.float32)
    }
    identity = runtime.full_tensor.build_identity("version-a")
    assert identity.model_name == "test/model"
    assert identity.revision == "version-a"
