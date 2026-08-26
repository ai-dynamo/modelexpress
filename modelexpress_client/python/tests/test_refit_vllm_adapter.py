# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import modelexpress_rl.inference.engines.vllm.adapter as vllm_adapter_module
import pytest
import torch
from modelexpress import p2p_pb2
from modelexpress_rl import ObjectStorageType, WeightPayloadFormat
from modelexpress_rl.inference.adapter import (
    GeneratorSource,
    GeneratorTransferInputs,
    NixlGeneratorSource,
)
from modelexpress_rl.inference.engines.vllm import VllmGeneratorAdapter
from modelexpress_rl.inference.receiver import (
    CanonicalS3GeneratorAdapter,
    ObjectStorageGeneratorConfig,
    PreparedCheckpoint,
)


def test_vllm_adapter_composes_transfer_and_installer_lifecycles(
    monkeypatch,
):
    events = []
    native_plan = object()
    transferred = type(
        "Transferred",
        (),
        {
            "tensors": {"weight": object()},
            "metrics": {"bytes_received": 128},
        },
    )()

    class _Installer:
        def __init__(self, **kwargs):
            events.append(("installer_init", kwargs))
            self.capture = object()

        def parameter_layout(self):
            events.append(("parameter_layout",))
            return {"weight": ((4,), torch.float32)}

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

        def unpublish_peer(self):
            events.append(("unpublish_peer",))

        def stage_peer(self, **kwargs):
            events.append(("stage_peer", kwargs))
            return type(
                "PeerTransferred",
                (),
                {
                    "tensors": {"weight": object()},
                    "metrics": {"bytes_received": 128},
                },
            )()

        def publish_peer(self, **kwargs):
            events.append(("publish_peer", kwargs))

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

        def get_worker_rank(self):
            return 3

        def build_identity(self):
            return p2p_pb2.SourceIdentity(
                model_name="test/model",
                revision="checkpoint-revision",
            )

        accelerator_backend = SimpleNamespace(name="cuda")

    monkeypatch.setattr(vllm_adapter_module, "VllmAdapter", _Engine)
    monkeypatch.setattr(vllm_adapter_module, "_VllmInstaller", _Installer)
    monkeypatch.setattr(vllm_adapter_module, "_NixlStagedTransfer", _Transfer)
    monkeypatch.setenv("MX_METADATA_PORT", "62000")
    monkeypatch.setenv("MX_REFIT_METADATA_PORT", "61000")
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
    assert adapter.worker_rank == 3
    identity = adapter.build_p2p_identity("version-a")
    assert identity.model_name == "test/model"
    assert identity.revision == "version-a"
    plan = adapter.create_transfer_plan(inputs)
    assert plan is native_plan
    assert adapter.validate_transfer_plan(plan, inputs)
    assert not adapter.validate_transfer_plan(
        plan, replace(inputs, layout_signature="layout-b")
    )
    staged = adapter.stage_weight(plan)
    assert staged is transferred
    with pytest.raises(RuntimeError, match="release staged weight"):
        adapter.create_transfer_plan(inputs)
    assert adapter.apply_weight(staged) == {"bytes_received": 128}
    adapter.publish_weight_version(
        version_id="version-a",
        staged=staged,
        p2p_client="p2p-client",
        worker_id="generator-0",
    )
    adapter.release_staged_weight(staged)
    with pytest.raises(RuntimeError, match="no longer active"):
        adapter.publish_weight_version(
            version_id="version-a",
            staged=staged,
            p2p_client="p2p-client",
            worker_id="generator-0",
        )

    peer_source = p2p_pb2.WorkerMetadata(worker_rank=3)
    peer_staged = adapter.stage_peer_weight(peer_source)
    assert adapter.apply_weight(peer_staged) == {"bytes_received": 128}
    adapter.release_staged_weight(peer_staged)

    with pytest.raises(ValueError, match="does not support XOR_DELTA"):
        adapter.create_transfer_plan(
            replace(inputs, payload_format=WeightPayloadFormat.XOR_DELTA)
        )
    with pytest.raises(ValueError, match="supports NIXL sources only"):
        adapter.create_transfer_plan(
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
                "listen_port": 61002,
            },
        ),
        (
            "prepare",
            {
                "manifests": [b"manifest"],
                "capture_layout": adapter._installer.capture,
            },
        ),
        ("unpublish_peer",),
        ("stage", native_plan),
        ("install", transferred.tensors),
        (
            "publish_peer",
            {
                "staged": transferred,
                "identity": identity,
                "p2p_client": "p2p-client",
                "worker_rank": 3,
                "worker_id": "generator-0",
                "accelerator": "cuda",
            },
        ),
        ("parameter_layout",),
        (
            "stage_peer",
            {
                "source": peer_source,
                "parameter_layout": {"weight": ((4,), torch.float32)},
            },
        ),
        ("install", peer_staged.tensors),
        ("close",),
    ]


def test_vllm_adapter_uses_canonical_s3_without_creating_nixl(
    monkeypatch,
    tmp_path,
):
    events = []
    prepared = PreparedCheckpoint("target-a", tmp_path / "prepared", {})
    model_config = SimpleNamespace(model="config/model")
    vllm_config = SimpleNamespace(model_config=model_config)

    class _Installer:
        def __init__(self, **kwargs):
            events.append(("installer_init", kwargs))

        def install_checkpoint(self, path):
            events.append(("install_checkpoint", path))

    class _Transfer:
        def __init__(self, **_kwargs):
            pytest.fail("S3 mode must not create a NIXL transfer")

    class _Engine:
        def __init__(self, received_vllm_config, received_model_config):
            assert received_vllm_config is vllm_config
            assert received_model_config is model_config

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

    object_storage = ObjectStorageGeneratorConfig(
        storage_type=ObjectStorageType.S3,
        initial_base_version_id="base-a",
        launch_checkpoint=tmp_path / "launch",
        preparation_cache_dir=tmp_path / "cache",
    )
    adapter = VllmGeneratorAdapter(
        model="model",
        vllm_config=vllm_config,
        model_config=model_config,
        worker_id="generator-0",
        object_storage=object_storage,
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
                "vllm_config": vllm_config,
                "model_config": model_config,
                "device": torch.device("cuda:2"),
            },
        ),
        (
            "s3_init",
            {"model_name": "config/model", "config": object_storage},
        ),
        ("s3_stage", inputs),
        ("s3_apply", prepared),
        ("install_checkpoint", prepared.path),
        ("s3_release", prepared),
        ("s3_close",),
    ]
