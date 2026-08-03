# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from modelexpress.adapter import StrategyFailed
from modelexpress.engines.trtllm.adapter import (
    TrtllmAdapter,
    _normalize_model_identity,
    build_mx_identity,
    build_trtllm_load_context,
)
from modelexpress.load_strategy import LoadResult
from modelexpress.load_strategy.default_strategy import DefaultStrategy
from modelexpress.load_strategy.gds_strategy import GdsStrategy
from modelexpress.load_strategy.instant_tensor_strategy import (
    InstantTensorStrategy,
)
from modelexpress.load_strategy.model_streamer_strategy import (
    ModelStreamerStrategy,
)
from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
from torch import nn


def _mapping(**overrides):
    values = {
        "rank": 5,
        "local_rank": 1,
        "node_rank": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _TrtIdentity:
    model_name = "meta-llama/Llama-3.1-405B-Instruct"
    tp_size = 4
    pp_size = 2
    ep_size = -1
    dtype = "bfloat16"

    def to_dict(self):
        return {
            "format_version": 2,
            "model_name": self.model_name,
            "model_fingerprint": "model",
            "shard_fingerprint": "shard",
            "rank": 5,
        }


def _adapter_kwargs():
    return {
        "checkpoint_loader": object(),
        "checkpoint_dir": "/model",
        "native_loader_kwargs": {},
        "prepare_post_transform_receiver": lambda _model: None,
        "transform_protocol_version": 1,
    }


def test_build_identity_preserves_trtllm_authoritative_fingerprint():
    identity = build_mx_identity(_TrtIdentity())

    assert identity.model_name == _TrtIdentity.model_name
    assert identity.tensor_parallel_size == 4
    assert identity.pipeline_parallel_size == 2
    assert identity.expert_parallel_size == 1
    assert identity.dtype == "bfloat16"
    assert identity.extra_parameters["trtllm_weight_layout"] == "post_transform"
    payload = json.loads(identity.extra_parameters["trtllm_source_identity"])
    assert "model_name" not in payload
    assert payload["model_fingerprint"] == "model"
    assert payload["shard_fingerprint"] == "shard"


def test_different_trtllm_fingerprint_produces_different_mx_identity():
    left = build_mx_identity(_TrtIdentity())
    right_identity = _TrtIdentity()
    right_identity.to_dict = lambda: {
        **_TrtIdentity().to_dict(),
        "backend_fingerprint": "different",
    }
    right = build_mx_identity(right_identity)

    assert (
        left.extra_parameters["trtllm_source_identity"]
        != right.extra_parameters["trtllm_source_identity"]
    )


def test_mismatched_identity_never_prepares_rdma_receiver(monkeypatch):
    target_identity = _TrtIdentity()
    source_identity = _TrtIdentity()
    source_identity.to_dict = lambda: {
        **_TrtIdentity().to_dict(),
        "artifact_identity": "different-checkpoint",
    }
    source_mx_identity = build_mx_identity(source_identity)
    prepare_receiver = MagicMock()
    client = MagicMock()

    def list_sources(*, identity, status_filter):
        assert status_filter
        instances = [] if identity != source_mx_identity else [object()]
        return SimpleNamespace(instances=instances)

    client.list_sources.side_effect = list_sources
    monkeypatch.setattr(
        "modelexpress.engines.trtllm.adapter.create_metadata_client",
        lambda **_kwargs: client,
    )
    context = build_trtllm_load_context(
        model_config=object(),
        load_config=object(),
        checkpoint_loader=object(),
        checkpoint_dir="/model",
        native_loader_kwargs={},
        mapping=_mapping(),
        source_identity=target_identity,
        prepare_post_transform_receiver=prepare_receiver,
        transform_protocol_version=1,
        p2p_enabled=True,
        mx_server_url="modelexpress-server:8001",
    )
    model = nn.Linear(2, 2, bias=False)

    with pytest.raises(StrategyFailed, match="No RDMA source available"):
        RdmaStrategy().load(LoadResult(value=model, model=model), context)

    assert client.list_sources.call_count == 1
    prepare_receiver.assert_not_called()


def test_build_identity_uses_trtllm_transform_protocol():
    identity = build_mx_identity(_TrtIdentity(), transform_protocol_version=7)
    incompatible = build_mx_identity(_TrtIdentity(), transform_protocol_version=8)

    assert identity.extra_parameters["trtllm_transform_protocol_version"] == "7"
    assert identity != incompatible


def test_model_identity_preserves_hub_ids_and_normalizes_local_paths():
    assert (
        _normalize_model_identity("meta-llama/Llama-3.1-405B-Instruct")
        == "meta-llama/Llama-3.1-405B-Instruct"
    )
    assert _normalize_model_identity("/models/local-llama") == "local-llama"
    assert (
        _normalize_model_identity(
            "/cache/models--meta-llama--Llama-3.1-405B-Instruct/snapshots/0123456789"
        )
        == "meta-llama/Llama-3.1-405B-Instruct"
    )


def test_adapter_uses_global_rank_for_worker_and_local_rank_for_device():
    adapter = TrtllmAdapter(
        mapping=_mapping(),
        source_identity=_TrtIdentity(),
        **_adapter_kwargs(),
    )

    assert adapter.get_worker_rank() == 5
    assert adapter.get_global_rank() == 5
    assert adapter.get_device_id() == 1
    assert adapter.get_target_device() == torch.device("cuda", 1)


def test_adapter_discovers_canonical_parameters_only():
    model = nn.Module()
    model.primary = nn.Linear(2, 2, bias=False)
    model.alias = model.primary
    adapter = TrtllmAdapter(
        mapping=_mapping(),
        source_identity=_TrtIdentity(),
        **_adapter_kwargs(),
    )
    adapter.target_device = torch.device("cpu")

    tensors = adapter.discover_tensors(LoadResult(value=model, model=model))

    assert list(tensors) == ["primary.weight"]
    assert tensors["primary.weight"].data_ptr() == model.primary.weight.data_ptr()


def test_adapter_uses_trtllm_native_loader_and_lifecycle(monkeypatch):
    model = nn.Linear(2, 2, bias=False)
    calls = []

    class HfCheckpointLoader:
        @staticmethod
        def load_weights(loader, checkpoint_dir, mapping, **kwargs):
            calls.append(("native", loader, checkpoint_dir, mapping, kwargs))
            return {"weight": model.weight}

    module = ModuleType("tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader")
    module.HfCheckpointLoader = HfCheckpointLoader
    monkeypatch.setitem(sys.modules, module.__name__, module)
    checkpoint_loader = object()
    mapping = _mapping()

    adapter = TrtllmAdapter(
        checkpoint_loader=checkpoint_loader,
        checkpoint_dir="/model",
        native_loader_kwargs={"extra": True},
        mapping=mapping,
        source_identity=_TrtIdentity(),
        prepare_post_transform_receiver=lambda received: calls.append(
            ("prepare", received)
        ),
        transform_protocol_version=1,
    )
    result = LoadResult(value=model, model=model)

    adapter.before_rdma_receive(result)
    adapter.after_rdma_receive(result)

    assert calls == [("prepare", model)]
    assert adapter.rdma_loaded
    assert adapter.rdma_transform_protocol_version == 1

    fallback_adapter = TrtllmAdapter(
        checkpoint_loader=checkpoint_loader,
        checkpoint_dir="/model",
        native_loader_kwargs={"extra": True},
        mapping=mapping,
        source_identity=_TrtIdentity(),
        prepare_post_transform_receiver=lambda _model: None,
        transform_protocol_version=1,
    )
    fallback_result = LoadResult(value=model, model=model)
    retry = fallback_adapter.reinit_for_retry(fallback_result)
    loaded = fallback_adapter.load_via_native(retry)

    assert calls[-1] == (
        "native",
        checkpoint_loader,
        "/model",
        mapping,
        {"extra": True},
    )
    assert loaded.value == {"weight": model.weight}
    assert not loaded.publishable
    assert fallback_adapter.native_loaded
    assert not fallback_adapter.rdma_loaded
    assert retry is fallback_result
    assert retry.model is model
    assert fallback_adapter.requires_exact_tensor_catalog()


def test_build_load_context_passes_explicit_server_url(monkeypatch):
    client = object()
    calls = []

    def create_client(*, worker_rank, server_url):
        calls.append((worker_rank, server_url))
        return client

    monkeypatch.setattr(
        "modelexpress.engines.trtllm.adapter.create_metadata_client",
        create_client,
    )
    context = build_trtllm_load_context(
        model_config=object(),
        load_config=object(),
        checkpoint_loader=object(),
        checkpoint_dir="/model",
        native_loader_kwargs={},
        mapping=_mapping(),
        source_identity=_TrtIdentity(),
        prepare_post_transform_receiver=lambda _model: None,
        transform_protocol_version=1,
        p2p_enabled=True,
        mx_server_url="modelexpress-server:8001",
    )

    assert calls == [(5, "modelexpress-server:8001")]
    assert context.mx_server_url == "modelexpress-server:8001"
    assert context.global_rank == 5
    assert context.worker_rank == 5
    assert context.device_id == 1
    assert context.node_rank == 1
    assert context.mx_client is client
    assert context.p2p_enabled


def test_only_rdma_and_native_capabilities_are_exposed(monkeypatch):
    monkeypatch.setattr(
        "modelexpress.engines.trtllm.adapter.create_metadata_client",
        lambda **_kwargs: object(),
    )
    context = build_trtllm_load_context(
        model_config=object(),
        load_config=object(),
        checkpoint_loader=object(),
        checkpoint_dir="/model",
        native_loader_kwargs={},
        mapping=_mapping(),
        source_identity=_TrtIdentity(),
        prepare_post_transform_receiver=lambda _model: None,
        transform_protocol_version=1,
        p2p_enabled=True,
        mx_server_url="modelexpress-server:8001",
    )

    assert DefaultStrategy().is_available(context)
    assert not InstantTensorStrategy().is_available(context)
    assert not ModelStreamerStrategy().is_available(context)
    assert not GdsStrategy().is_available(context)
