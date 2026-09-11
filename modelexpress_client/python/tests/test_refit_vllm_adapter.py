# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from modelexpress import p2p_pb2
from modelexpress_rl.inference.engines.vllm import (
    VllmGeneratorContext,
    _create_vllm_engine_runtime,
)


@pytest.mark.parametrize(
    ("quant_config", "cache_dtype", "runtime_p2p_available"),
    [
        (None, "auto", True),
        (object(), "auto", False),
        (None, "fp8_e4m3", False),
    ],
)
def test_vllm_engine_runtime_exposes_installation_and_full_tensor_geometry(
    monkeypatch,
    quant_config,
    cache_dtype,
    runtime_p2p_available,
):
    class ModelConfig:
        model = "test/model"

    class VllmConfig:
        model_config = ModelConfig()

        def __init__(self):
            self.quant_config = quant_config
            self.cache_config = SimpleNamespace(cache_dtype=cache_dtype)

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

    adapter_module = ModuleType("modelexpress.engines.vllm.adapter")
    adapter_module.VllmAdapter = Engine
    monkeypatch.setitem(
        sys.modules, "modelexpress.engines.vllm.adapter", adapter_module
    )
    model = torch.nn.Linear(4, 4)

    class Loader:
        tensors = {"weight": model.weight}
        worker_id = "inference-worker-3"

        def unpublish_runtime_tensors(self):
            publication_events.append(("unpublish", self))

        def publish_runtime_tensors(self, version_id):
            publication_events.append(("publish", self, version_id))

    loader = Loader()
    loader_module = ModuleType("modelexpress.engines.vllm.loader")
    loader_module.get_model_loader = lambda device_id: loader
    monkeypatch.setitem(
        sys.modules, "modelexpress.engines.vllm.loader", loader_module
    )
    publication_events = []
    installer_module = ModuleType(
        "modelexpress_rl.inference.engines.vllm.installer"
    )
    installer_module._VllmInstaller = Installer
    monkeypatch.setitem(
        sys.modules,
        "modelexpress_rl.inference.engines.vllm.installer",
        installer_module,
    )

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
    assert {
        key: value
        for key, value in runtime.installer.kwargs.items()
        if key != "runtime_tensors"
    } == {
        "model": model,
        "vllm_config": config,
        "model_config": config.model_config,
        "device": torch.device("cuda:2"),
        "convert_native_to_hf": convert_native_to_hf,
    }
    expected_runtime_tensors = (
        {"weight": model.weight} if runtime_p2p_available else None
    )
    assert runtime.installer.kwargs["runtime_tensors"] == expected_runtime_tensors
    assert runtime.full_tensor is not None
    assert runtime.full_tensor.device_id == 2
    assert runtime.full_tensor.worker_rank == 3
    assert runtime.full_tensor.capture_layout(["manifest"]) == ["manifest"]
    assert (runtime.full_tensor.runtime_tensors is loader.tensors) is (
        runtime_p2p_available
    )
    assert runtime.full_tensor.source_worker_id == (
        "inference-worker-3" if runtime_p2p_available else None
    )
    if runtime_p2p_available:
        runtime.full_tensor.unpublish_runtime_tensors()
        runtime.full_tensor.publish_runtime_tensors("version-a")
        assert publication_events == [
            ("unpublish", loader),
            ("publish", loader, "version-a"),
        ]
    identity = runtime.full_tensor.build_identity("version-a")
    assert identity.model_name == "test/model"
    assert identity.revision == "version-a"
