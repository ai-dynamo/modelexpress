# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import modelexpress_rl.inference.engines.vllm.adapter as vllm_adapter_module
from modelexpress_rl import (
    GeneratorInstallationMode,
    WeightPayloadFormat,
)
from modelexpress_rl.inference import GeneratorSource, GeneratorTransferInputs
from modelexpress_rl.inference.engines.vllm import VllmGeneratorAdapter


def test_vllm_adapter_composes_transfer_and_installer_lifecycles(
    monkeypatch,
):
    events = []
    native_plan = object()
    transferred = type(
        "Transferred",
        (),
        {"tensors": {"weight": object()}, "metrics": {"bytes_received": 128}},
    )()

    class _Installer:
        def __init__(self, **kwargs):
            events.append(("installer_init", kwargs))
            self.capture = object()

        def install(self, tensors):
            events.append(("install", tensors))

    class _Transfer:
        def __init__(self, **kwargs):
            events.append(("transfer_init", kwargs))

        def prepare(self, **kwargs):
            events.append(("prepare", kwargs))
            return native_plan

        def stage(self, plan):
            events.append(("stage", plan))
            return transferred

        def close(self):
            events.append(("close",))

    class _Engine:
        def __init__(self, vllm_config, model_config):
            assert vllm_config == "vllm-config"
            assert model_config == "model-config"

        def get_device_id(self):
            return 2

        def get_target_device(self):
            return torch.device("cuda:2")

    monkeypatch.setattr(vllm_adapter_module, "VllmAdapter", _Engine)
    monkeypatch.setattr(vllm_adapter_module, "_VllmInstaller", _Installer)
    monkeypatch.setattr(vllm_adapter_module, "_NixlStagedTransfer", _Transfer)
    adapter = VllmGeneratorAdapter(
        model="model",
        vllm_config="vllm-config",
        model_config="model-config",
        worker_id="generator-0",
    )
    inputs = GeneratorTransferInputs(
        version_id="version-a",
        layout_signature="layout-a",
        payload_format=WeightPayloadFormat.FULL_TENSOR,
        sources=(
            GeneratorSource(
                source_slot_id="rank:0",
                worker_id="trainer-0",
                manifest_endpoint="trainer-0:9000",
                manifest_digest="digest",
                transport="NIXL",
                manifest=b"manifest",
            ),
        ),
    )

    assert adapter.supported_installation_modes == frozenset(
        {GeneratorInstallationMode.STAGED}
    )
    assert adapter.supported_payload_formats == frozenset(
        {WeightPayloadFormat.FULL_TENSOR}
    )
    plan = adapter.create_transfer_plan(inputs)
    assert adapter.validate_transfer_plan(plan, inputs)
    staged = adapter.stage_weight(plan)
    assert adapter.apply_weight(staged) == {"bytes_received": 128}
    adapter.release_staged_weight(staged)
    adapter.close()

    assert events == [
        (
            "installer_init",
            {
                "model": "model",
                "vllm_config": "vllm-config",
                "model_config": "model-config",
                "device": torch.device("cuda:2"),
            },
        ),
        (
            "transfer_init",
            {
                "agent_name": "mx-refit-generator-0",
                "device_id": 2,
                "device": torch.device("cuda:2"),
                "listen_port": vllm_adapter_module.envs.MX_METADATA_PORT + 2,
            },
        ),
        (
            "prepare",
            {
                "manifests": [b"manifest"],
                "capture_layout": adapter._installer.capture,
            },
        ),
        ("stage", native_plan),
        ("install", transferred.tensors),
        ("close",),
    ]
