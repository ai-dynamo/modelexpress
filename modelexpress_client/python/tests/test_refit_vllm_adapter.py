# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import modelexpress_rl.inference.engines.vllm.adapter as vllm_adapter_module
import pytest
import torch
from modelexpress_rl import WeightPayloadFormat
from modelexpress_rl.inference.adapter import (
    GeneratorSource,
    GeneratorTransferInputs,
    NixlGeneratorSource,
)
from modelexpress_rl.inference.engines.vllm import VllmGeneratorAdapter
from modelexpress_rl.inference.receiver import (
    CanonicalS3GeneratorAdapter,
    PreparedCheckpoint,
    S3GeneratorConfig,
)


def test_vllm_adapter_composes_transfer_and_installer_lifecycles(
    monkeypatch,
):
    events = []
    native_plan = type("NativePlan", (), {"plan_revision": 1})()
    transferred = type(
        "Transferred",
        (),
        {
            "tensors": {"weight": object()},
            "metrics": {"bytes_received": 128},
            "plan_revision": 1,
        },
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
        base_version_id=None,
        layout_signature="layout-a",
        payload_format=WeightPayloadFormat.FULL_TENSOR,
        sources=(
            GeneratorSource(
                source_slot_id="rank:0",
                worker_id="trainer-0",
                manifest_digest="digest",
                transport=NixlGeneratorSource(
                    manifest_endpoint="trainer-0:9000",
                    manifest=b"manifest",
                ),
            ),
        ),
    )

    assert adapter.supported_payload_formats == frozenset(
        {WeightPayloadFormat.FULL_TENSOR}
    )
    staged = adapter.stage_weight(inputs)
    assert staged is transferred
    with pytest.raises(RuntimeError, match="release staged weight"):
        adapter.stage_weight(inputs)
    assert adapter.apply_weight(staged) == {"bytes_received": 128}
    adapter.release_staged_weight(staged)

    adapter.release_staged_weight(adapter.stage_weight(inputs))
    adapter.release_staged_weight(
        adapter.stage_weight(replace(inputs, layout_signature="layout-b"))
    )

    with pytest.raises(ValueError, match="does not support XOR_DELTA"):
        adapter.stage_weight(
            replace(inputs, payload_format=WeightPayloadFormat.XOR_DELTA)
        )
    with pytest.raises(ValueError, match="supports NIXL sources only"):
        adapter.stage_weight(
            replace(inputs, sources=(replace(inputs.sources[0], transport="NCCL"),))
        )
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
        ("stage", native_plan),
        (
            "prepare",
            {
                "manifests": [b"manifest"],
                "capture_layout": adapter._installer.capture,
            },
        ),
        ("stage", native_plan),
        ("close",),
    ]


def test_vllm_adapter_uses_canonical_s3_without_creating_nixl(
    monkeypatch,
    tmp_path,
):
    events = []
    prepared = PreparedCheckpoint("target-a", tmp_path / "prepared", {})

    class _Installer:
        def __init__(self, **kwargs):
            events.append(("installer_init", kwargs))

        def install_checkpoint(self, path):
            events.append(("install_checkpoint", path))

    class _Transfer:
        def __init__(self, **_kwargs):
            pytest.fail("S3 mode must not create a NIXL transfer")

    class _Engine:
        def __init__(self, vllm_config, model_config):
            assert vllm_config == "vllm-config"
            assert model_config == "model-config"

        def get_device_id(self):
            return 2

        def get_target_device(self):
            return torch.device("cuda:2")

    def initialize_s3(self, **kwargs):
        events.append(("s3_init", kwargs))
        self._active_staged = None

    def stage_s3(self, inputs):
        events.append(("s3_stage", inputs))
        self._active_staged = prepared
        return prepared

    def apply_s3(self, staged):
        events.append(("s3_apply", staged))
        self.install_prepared_checkpoint(staged)
        return {"installed": 1.0}

    def release_s3(self, staged):
        events.append(("s3_release", staged))
        self._active_staged = None

    def close_s3(self):
        events.append(("s3_close",))

    monkeypatch.setattr(vllm_adapter_module, "VllmAdapter", _Engine)
    monkeypatch.setattr(vllm_adapter_module, "_VllmInstaller", _Installer)
    monkeypatch.setattr(vllm_adapter_module, "_NixlStagedTransfer", _Transfer)
    monkeypatch.setattr(CanonicalS3GeneratorAdapter, "__init__", initialize_s3)
    monkeypatch.setattr(CanonicalS3GeneratorAdapter, "stage_weight", stage_s3)
    monkeypatch.setattr(CanonicalS3GeneratorAdapter, "apply_weight", apply_s3)
    monkeypatch.setattr(
        CanonicalS3GeneratorAdapter,
        "release_staged_weight",
        release_s3,
    )
    monkeypatch.setattr(CanonicalS3GeneratorAdapter, "close", close_s3)

    s3 = S3GeneratorConfig(
        initial_base_version_id="base-a",
        launch_checkpoint=tmp_path / "launch",
        preparation_cache_dir=tmp_path / "cache",
    )
    adapter = VllmGeneratorAdapter(
        model="model",
        vllm_config="vllm-config",
        model_config="model-config",
        worker_id="generator-0",
        model_name="test/model",
        s3=s3,
    )
    inputs = object()

    assert adapter.supported_payload_formats == frozenset(
        {WeightPayloadFormat.XOR_DELTA}
    )
    assert adapter.stage_weight(inputs) is prepared
    assert adapter.apply_weight(prepared) == {"installed": 1.0}
    adapter.release_staged_weight(prepared)
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
        ("s3_init", {"model_name": "test/model", "config": s3}),
        ("s3_stage", inputs),
        ("s3_apply", prepared),
        ("install_checkpoint", prepared.path),
        ("s3_release", prepared),
        ("s3_close",),
    ]


def test_vllm_s3_requires_model_name():
    with pytest.raises(ValueError, match="model_name is required"):
        VllmGeneratorAdapter(
            model="model",
            vllm_config="vllm-config",
            model_config="model-config",
            worker_id="generator-0",
            s3=S3GeneratorConfig(
                initial_base_version_id="base-a",
                launch_checkpoint="launch",
                preparation_cache_dir="cache",
            ),
        )
