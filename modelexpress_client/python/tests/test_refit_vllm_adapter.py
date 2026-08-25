# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import modelexpress_rl.inference.engines.vllm.adapter as vllm_adapter_module
import pytest
import torch
from modelexpress import p2p_pb2
from modelexpress_rl import WeightPayloadFormat
from modelexpress_rl.inference.adapter import GeneratorSource, GeneratorTransferInputs
from modelexpress_rl.inference.engines.vllm import VllmGeneratorAdapter


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
