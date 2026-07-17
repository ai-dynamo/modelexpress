# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SGLang ModelExpress adapter and loader entrypoint."""

import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from modelexpress import p2p_pb2
from modelexpress.engines.sglang.adapter import (
    SglangAdapter,
    build_sglang_load_context,
)
from modelexpress.engines.sglang.loader import MxModelLoader


def _load_config(**overrides):
    defaults = dict(
        tp_rank=3,
        modelexpress_url="modelexpress-server:8001",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _model_config(**overrides):
    defaults = dict(
        model_path="deepseek-ai/DeepSeek-V3",
        dtype=torch.bfloat16,
        quantization="fp8",
        revision="abc123",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _device_config(**overrides):
    defaults = dict(device="cpu", gpu_id=0)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def _stub_accelerator_backend_selection(monkeypatch, mock_accelerator_backend_cls):
    monkeypatch.setattr(
        "modelexpress.engines.sglang.adapter.accelerator_backend_for",
        lambda device: mock_accelerator_backend_cls(),
    )


def test_sglang_adapter_builds_identity_from_sglang_configs():
    adapter = SglangAdapter(_load_config(), _model_config(), _device_config())

    with patch(
        "modelexpress.engines.sglang.adapter._get_parallel_size",
        side_effect=lambda name: {
            "get_tensor_model_parallel_world_size": 8,
            "get_pipeline_model_parallel_world_size": 2,
            "get_moe_expert_parallel_world_size": 4,
        }[name],
    ):
        identity = adapter.build_identity()

    assert identity.model_name == "deepseek-ai/DeepSeek-V3"
    assert identity.backend_framework == p2p_pb2.BACKEND_FRAMEWORK_SGLANG
    assert identity.tensor_parallel_size == 8
    assert identity.pipeline_parallel_size == 2
    assert identity.expert_parallel_size == 4
    assert identity.dtype == "bfloat16"
    assert identity.quantization == "fp8"
    assert identity.revision == "abc123"


def test_sglang_context_uses_tp_rank_for_matching_and_url_override():
    ctx = build_sglang_load_context(
        _load_config(tp_rank=5, modelexpress_url="mx.example:9000"),
        _model_config(),
        _device_config(),
    )

    assert ctx.worker_rank == 5
    assert ctx.global_rank == 5
    assert ctx.mx_client.server_url == "mx.example:9000"


def test_sglang_context_separates_worker_rank_from_global_rank(monkeypatch):
    sglang_mod = ModuleType("sglang")
    srt_mod = ModuleType("sglang.srt")
    distributed_mod = ModuleType("sglang.srt.distributed")
    distributed_mod.get_tensor_model_parallel_rank = lambda: 1
    distributed_mod.get_pipeline_model_parallel_rank = lambda: 2
    distributed_mod.get_tensor_model_parallel_world_size = lambda: 4
    srt_mod.distributed = distributed_mod

    monkeypatch.setitem(sys.modules, "sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed_mod)

    with patch("torch.distributed.is_available", return_value=True), patch(
        "torch.distributed.is_initialized", return_value=True,
    ), patch("torch.distributed.get_rank", return_value=17):
        ctx = build_sglang_load_context(
            _load_config(tp_rank=5, modelexpress_url="mx.example:9000"),
            _model_config(),
            _device_config(),
        )

    assert ctx.worker_rank == 9
    assert ctx.global_rank == 17
    assert ctx.mx_client.server_url == "mx.example:9000"


def test_sglang_is_cuda_alike_uses_sglang_platform_helper(monkeypatch):
    adapter = SglangAdapter(_load_config(), _model_config(), _device_config())
    sglang_mod = ModuleType("sglang")
    srt_mod = ModuleType("sglang.srt")
    utils_mod = ModuleType("sglang.srt.utils")
    utils_mod.is_cuda_alike = lambda: True

    monkeypatch.setitem(sys.modules, "sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils_mod)

    assert adapter.is_cuda_alike() is True


def test_collect_sglang_tensors_preserves_non_contiguous_storage_names(
    mock_accelerator_backend_cls,
):
    backend = mock_accelerator_backend_cls(torch_device_type="cpu")
    adapter = SglangAdapter(_load_config(), _model_config(), _device_config())
    adapter.accelerator_backend = backend
    model = nn.Module()
    model.weight_t = nn.Parameter(torch.randn(4, 3).T)

    tensors = adapter._collect_tensors(model)

    assert "weight_t.__storage" in tensors
    assert tensors["weight_t.__storage"].dtype == torch.uint8


def test_collect_sglang_tensors_deduplicates_tied_parameters(
    mock_accelerator_backend_cls,
):
    backend = mock_accelerator_backend_cls(torch_device_type="cpu")
    adapter = SglangAdapter(_load_config(), _model_config(), _device_config())
    adapter.accelerator_backend = backend
    model = nn.Module()
    shared = nn.Parameter(torch.randn(4, 3))
    model.first = shared
    model.second = shared

    tensors = adapter._collect_tensors(model)

    assert list(tensors) == ["first"]


def test_sglang_adapter_discovery_uses_backend_predicate(
    monkeypatch,
    mock_accelerator_backend_cls,
):
    backend = mock_accelerator_backend_cls(torch_device_type="cpu")
    monkeypatch.setattr(
        "modelexpress.engines.sglang.adapter.accelerator_backend_for",
        lambda device: backend,
    )
    adapter = SglangAdapter(_load_config(), _model_config(), _device_config())
    model = nn.Module()
    model.weight = nn.Parameter(torch.randn(4, 3))

    tensors = adapter.discover_tensors(SimpleNamespace(model=model))

    assert list(tensors) == ["weight"]


def test_sglang_adapter_post_load_delegates_to_child_module():
    adapter = SglangAdapter(_load_config(), _model_config(), _device_config())

    class ChildModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.post_load_called = False

        def post_load_weights(self):
            self.post_load_called = True

    model = nn.Module()
    model.child = ChildModel()

    adapter._post_load_weights(SimpleNamespace(model=model))

    assert model.child.post_load_called


def test_sglang_adapter_post_load_prefers_top_level_hook():
    adapter = SglangAdapter(_load_config(), _model_config(), _device_config())

    class TopLevelModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = nn.Module()
            self.child.post_load_called = False
            self.post_load_called = False

        def post_load_weights(self):
            self.post_load_called = True

    def child_post_load_weights():
        model.child.post_load_called = True

    model = TopLevelModel()
    model.child.post_load_weights = child_post_load_weights

    adapter._post_load_weights(SimpleNamespace(model=model))

    assert model.post_load_called
    assert not model.child.post_load_called


def _install_sglang_runai_loader_modules(monkeypatch, loader_cls, load_format):
    sglang_mod = ModuleType("sglang")
    srt_mod = ModuleType("sglang.srt")
    configs_mod = ModuleType("sglang.srt.configs")
    load_config_mod = ModuleType("sglang.srt.configs.load_config")
    model_loader_mod = ModuleType("sglang.srt.model_loader")
    loader_mod = ModuleType("sglang.srt.model_loader.loader")

    load_config_mod.LoadFormat = load_format
    loader_mod.RunaiModelStreamerLoader = loader_cls

    monkeypatch.setitem(sys.modules, "sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.configs", configs_mod)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.configs.load_config",
        load_config_mod,
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.model_loader", model_loader_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.model_loader.loader", loader_mod)


def test_sglang_adapter_uses_native_model_streamer_loader(monkeypatch):
    tensor = torch.randn(2, 2)
    loader_instance = MagicMock()
    loader_instance._get_all_weights.return_value = iter([("w", tensor)])
    loader_cls = MagicMock(return_value=loader_instance)
    load_format = SimpleNamespace(RUNAI_STREAMER="runai_streamer")
    _install_sglang_runai_loader_modules(monkeypatch, loader_cls, load_format)

    load_config = _load_config(model_loader_extra_config={"concurrency": 4})
    model_config = _model_config()
    adapter = SglangAdapter(
        load_config,
        model_config,
        _device_config(device="cuda:2", gpu_id=2),
    )
    model = nn.Linear(2, 2)

    weights = list(
        adapter.build_model_streamer_weight_iter(
            "az://models/deepseek-ai/DeepSeek-V3",
            model=model,
        )
    )

    stream_config = loader_cls.call_args.args[0]
    assert stream_config is not load_config
    assert stream_config.load_format == "runai_streamer"
    assert stream_config.model_loader_extra_config == {"concurrency": 4}
    stream_model_config = loader_instance._get_all_weights.call_args.args[0]
    assert stream_model_config is not model_config
    assert stream_model_config.model_weights == "az://models/deepseek-ai/DeepSeek-V3"
    assert loader_instance._get_all_weights.call_args.args[1] is model
    assert loader_instance.target_device_str == "cuda:2"
    assert weights == [("w", tensor)]


def test_sglang_adapter_enables_distributed_model_streamer(monkeypatch):
    loader_instance = MagicMock()
    loader_instance._get_all_weights.return_value = iter([])
    loader_cls = MagicMock(return_value=loader_instance)
    load_format = SimpleNamespace(RUNAI_STREAMER="runai_streamer")
    _install_sglang_runai_loader_modules(monkeypatch, loader_cls, load_format)

    adapter = SglangAdapter(
        _load_config(model_loader_extra_config={"concurrency": 4}),
        _model_config(),
        _device_config(device="cuda:0", gpu_id=0),
    )

    with patch(
        "modelexpress.engines.sglang.adapter._get_parallel_size",
        return_value=8,
    ), patch.object(adapter, "is_cuda_alike", return_value=True), patch.dict(
        "os.environ", {"MX_MS_DISTRIBUTED": "1"}
    ):
        list(
            adapter.build_model_streamer_weight_iter(
                "s3://bucket/deepseek-ai/DeepSeek-V3",
                model=nn.Linear(2, 2),
            )
        )

    stream_config = loader_cls.call_args.args[0]
    assert stream_config.model_loader_extra_config == {
        "concurrency": 4,
        "distributed": True,
    }


def test_sglang_model_streamer_requires_initialized_model(monkeypatch):
    loader_cls = MagicMock()
    load_format = SimpleNamespace(RUNAI_STREAMER="runai_streamer")
    _install_sglang_runai_loader_modules(monkeypatch, loader_cls, load_format)

    adapter = SglangAdapter(_load_config(), _model_config(), _device_config())

    try:
        list(adapter.build_model_streamer_weight_iter("s3://bucket/model"))
    except RuntimeError as exc:
        assert "requires result.model" in str(exc)
    else:
        raise AssertionError("Expected missing model to fail")


def test_mx_model_loader_delegates_to_shared_strategy_chain():
    model = nn.Linear(2, 2)
    loader = MxModelLoader(_load_config(modelexpress_transport="nixl"))

    with patch.object(
        MxModelLoader,
        "_load_model_via_nixl",
        return_value=model,
    ) as load_via_nixl:
        loaded = loader.load_model(
            model=model,
            model_config=_model_config(),
            device_config=_device_config(),
        )

    assert loaded is model
    load_via_nixl.assert_called_once_with(
        model=model,
        model_config=_model_config(),
        device_config=_device_config(),
    )


def test_mx_model_loader_nixl_path_delegates_to_shared_strategy_chain():
    model = nn.Linear(2, 2)
    loader = MxModelLoader(_load_config(modelexpress_transport="nixl"))

    with patch(
        "modelexpress.engines.sglang.loader.LoadStrategyChain.run",
        return_value=model,
    ) as run, patch(
        "modelexpress.engines.sglang.loader.install_sglang_cache_artifacts",
    ) as install_artifacts, patch(
        "modelexpress.engines.sglang.loader.schedule_sglang_cache_artifact_publish",
    ) as schedule_artifacts:
        loaded = loader._load_model_via_nixl(
            model=model,
            model_config=_model_config(),
            device_config=_device_config(),
        )

    assert loaded is model
    run.assert_called_once()
    assert run.call_args.args[0] is model
    ctx = run.call_args.args[1]
    assert ctx.adapter.__class__ is SglangAdapter
    assert ctx.identity.backend_framework == p2p_pb2.BACKEND_FRAMEWORK_SGLANG
    install_artifacts.assert_called_once_with(ctx)
    schedule_artifacts.assert_called_once_with(ctx)


def test_mx_model_loader_delegates_transfer_engine_transport_in_mx_package():
    model = nn.Linear(2, 2)
    loader = MxModelLoader(_load_config(modelexpress_transport="transfer_engine"))

    with patch.object(
        MxModelLoader,
        "_load_model_via_transfer_engine",
        return_value=model,
    ) as load_via_transfer_engine:
        loaded = loader.load_model(
            model=model,
            model_config=_model_config(),
            device_config=_device_config(),
        )

    assert loaded is model
    load_via_transfer_engine.assert_called_once_with(
        model=model,
        model_config=_model_config(),
        device_config=_device_config(),
    )


def test_mx_model_loader_rejects_unknown_transport_in_mx_package():
    loader = MxModelLoader(_load_config(modelexpress_transport="unknown"))

    try:
        loader.load_model(
            model=nn.Linear(2, 2),
            model_config=_model_config(),
            device_config=_device_config(),
        )
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("Expected unsupported transport to fail")


def test_transfer_engine_registers_discovered_tensor_map():
    loader = MxModelLoader(_load_config(modelexpress_transport="transfer_engine"))
    tensor = torch.randn(2, 3)
    calls = []

    class FakeTransferEngine:
        def register_memory(self, addr, size):
            calls.append((addr, size))
            return 0

    weight_info = loader._register_transfer_engine_tensors(
        {"weight.__storage": tensor},
        FakeTransferEngine(),
    )

    assert calls == [(tensor.data_ptr(), tensor.numel() * tensor.element_size())]
    assert weight_info == {
        "weight.__storage": (
            tensor.data_ptr(),
            tensor.numel(),
            tensor.element_size(),
        )
    }


def test_transfer_engine_receive_uses_discovered_tensor_map():
    loader = MxModelLoader(_load_config(modelexpress_transport="transfer_engine"))
    tensor = torch.randn(2, 3)
    ctx = SimpleNamespace(global_rank=0)
    transferred = {}
    source_worker = p2p_pb2.WorkerMetadata(
        transfer_engine_session_id="te-session",
        tensors=[
            p2p_pb2.TensorDescriptor(
                name="weight.__storage",
                addr=1234,
                size=tensor.numel() * tensor.element_size(),
                device_id=0,
            )
        ],
    )

    class FakeTransferEngine:
        def batch_transfer_sync_read(
            self,
            session_id,
            client_ptr_list,
            seed_ptr_list,
            client_len_list,
        ):
            transferred["session_id"] = session_id
            transferred["client_ptr_list"] = client_ptr_list
            transferred["seed_ptr_list"] = seed_ptr_list
            transferred["client_len_list"] = client_len_list
            return 0

    loader._receive_via_transfer_engine(
        {"weight.__storage": tensor},
        FakeTransferEngine(),
        source_worker,
        ctx,
    )

    assert transferred == {
        "session_id": "te-session",
        "client_ptr_list": [tensor.data_ptr()],
        "seed_ptr_list": [1234],
        "client_len_list": [tensor.numel() * tensor.element_size()],
    }


def test_transfer_engine_publish_starts_non_nixl_heartbeat():
    loader = MxModelLoader(_load_config(modelexpress_transport="transfer_engine"))
    ctx = SimpleNamespace(
        global_rank=9,
        worker_rank=3,
        worker_id="worker-id",
        device_id=1,
        identity=p2p_pb2.SourceIdentity(model_name="sglang-model"),
        mx_client=SimpleNamespace(),
        accelerator_backend=SimpleNamespace(name="cuda"),
    )
    published = {}

    def publish_metadata(identity, worker, worker_id):
        published["identity"] = identity
        published["worker"] = worker
        published["worker_id"] = worker_id
        return "mx-source-id"

    def update_status(**kwargs):
        published["status"] = kwargs
        return True

    ctx.mx_client.publish_metadata = publish_metadata
    ctx.mx_client.update_status = update_status

    class FakePublisher:
        def __init__(self, **kwargs):
            published["heartbeat"] = kwargs

        def start(self):
            published["heartbeat_started"] = True

    with patch(
        "modelexpress.engines.sglang.loader.PublisherThread",
        FakePublisher,
    ):
        published_ok = loader._publish_transfer_engine_source(
            ctx=ctx,
            session_id="te-session",
            weight_info={"weight": (1000, 4, 2)},
        )

    assert published_ok
    assert published["worker"].transfer_engine_session_id == "te-session"
    assert published["worker"].accelerator == "cuda"
    assert published["status"]["status"] == p2p_pb2.SOURCE_STATUS_READY
    assert published["heartbeat"]["nixl_manager"] is None
    assert published["heartbeat_started"]


def test_transfer_engine_publish_failure_is_non_fatal():
    loader = MxModelLoader(_load_config(modelexpress_transport="transfer_engine"))
    ctx = SimpleNamespace(
        global_rank=9,
        worker_rank=3,
        worker_id="worker-id",
        device_id=1,
        identity=p2p_pb2.SourceIdentity(model_name="sglang-model"),
        accelerator_backend=SimpleNamespace(name="cuda"),
        mx_client=SimpleNamespace(
            publish_metadata=lambda *args: (_ for _ in ()).throw(
                RuntimeError("metadata down")
            )
        ),
    )

    assert not loader._publish_transfer_engine_source(
        ctx=ctx,
        session_id="te-session",
        weight_info={"weight": (1000, 4, 2)},
    )


# ---------------------------------------------------------------------------
# _find_transfer_engine_source: source-selector wiring
# ---------------------------------------------------------------------------


def _te_ref(sid, wid, rank=0):
    return p2p_pb2.SourceInstanceRef(
        mx_source_id=sid, worker_id=wid, model_name="m", worker_rank=rank
    )


def _te_ctx(instances):
    ctx = SimpleNamespace()
    ctx.global_rank = 0
    ctx.worker_rank = 0
    ctx.worker_id = "tgt-0"
    ctx.identity = SimpleNamespace(model_name="m")
    ctx.mx_client = MagicMock()
    ctx.mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
        instances=instances
    )
    return ctx


def _te_meta(found=True, transfer_engine=False):
    if not found:
        return SimpleNamespace(found=False, worker=None)
    worker = (
        p2p_pb2.WorkerMetadata(worker_rank=0, transfer_engine_session_id="te")
        if transfer_engine
        else p2p_pb2.WorkerMetadata(worker_rank=0)
    )
    return SimpleNamespace(found=True, worker=worker)


def test_te_find_source_filters_rank_and_returns_transfer_engine(monkeypatch):
    monkeypatch.delenv("MX_P2P_SOURCE_SELECTOR", raising=False)
    loader = MxModelLoader(_load_config(modelexpress_transport="transfer_engine"))
    ctx = _te_ctx(
        [
            _te_ref("s0aaaaaaaaaaaaaa", "w0", rank=0),
            _te_ref("s1aaaaaaaaaaaaaa", "w1", rank=1),  # wrong rank -> filtered out
            _te_ref("s2aaaaaaaaaaaaaa", "w2", rank=0),
        ]
    )
    ctx.mx_client.get_metadata.side_effect = lambda mx_source_id, worker_id: _te_meta(
        transfer_engine=(worker_id == "w2")
    )

    worker = loader._find_transfer_engine_source(ctx)
    assert worker is not None
    assert worker.WhichOneof("backend_metadata") == "transfer_engine_session_id"
    queried = {c.kwargs["worker_id"] for c in ctx.mx_client.get_metadata.call_args_list}
    assert "w1" not in queried  # rank-mismatched source never queried


def test_te_find_source_iterates_in_selector_order_and_none_when_no_match(monkeypatch):
    class _ReverseSelector:
        name = "reverse"

        def order(self, candidates, context):
            return list(reversed(candidates))

    monkeypatch.setattr(
        "modelexpress.engines.sglang.loader.get_configured_selector",
        lambda: _ReverseSelector(),
    )
    loader = MxModelLoader(_load_config(modelexpress_transport="transfer_engine"))
    ctx = _te_ctx([_te_ref(f"s{i}aaaaaaaaaaaaaa", f"w{i}", rank=0) for i in range(3)])
    ctx.mx_client.get_metadata.side_effect = lambda mx_source_id, worker_id: _te_meta(
        transfer_engine=False
    )

    # No transfer_engine source -> None, and iteration follows the selector order.
    assert loader._find_transfer_engine_source(ctx) is None
    order = [c.kwargs["worker_id"] for c in ctx.mx_client.get_metadata.call_args_list]
    assert order == ["w2", "w1", "w0"]


def test_te_find_source_skips_not_found(monkeypatch):
    # Force identity order so w0 (not-found) is queried first and the
    # `if not metadata.found: continue` branch is actually exercised.
    class _IdentitySelector:
        name = "identity"

        def order(self, candidates, context):
            return list(candidates)

    monkeypatch.setattr(
        "modelexpress.engines.sglang.loader.get_configured_selector",
        lambda: _IdentitySelector(),
    )
    loader = MxModelLoader(_load_config(modelexpress_transport="transfer_engine"))
    ctx = _te_ctx(
        [_te_ref("s0aaaaaaaaaaaaaa", "w0"), _te_ref("s1aaaaaaaaaaaaaa", "w1")]
    )
    ctx.mx_client.get_metadata.side_effect = lambda mx_source_id, worker_id: _te_meta(
        found=(worker_id == "w1"), transfer_engine=(worker_id == "w1")
    )
    worker = loader._find_transfer_engine_source(ctx)
    assert worker is not None
    assert worker.WhichOneof("backend_metadata") == "transfer_engine_session_id"
    assert [
        c.kwargs["worker_id"] for c in ctx.mx_client.get_metadata.call_args_list
    ] == ["w0", "w1"]
