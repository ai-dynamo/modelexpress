# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TensorRT-LLM ModelExpress loader integration."""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from modelexpress import p2p_pb2
from modelexpress.adapter import EngineAdapter
from modelexpress.engines.trtllm import TrtllmAdapter
from modelexpress.engines.trtllm import adapter as trtllm_adapter
from modelexpress.engines.trtllm import loader as trtllm
from modelexpress.load_strategy.context import LoadResult


class _FakeTensor:
    def numel(self):
        return 2

    def element_size(self):
        return 4


def _fake_trtllm_mapping(rank=0, tp_size=1, pp_size=1, cp_size=1):
    return SimpleNamespace(
        rank=rank,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
    )


@contextmanager
def _noop_rank_log_scope(global_rank):
    del global_rank
    yield


@pytest.fixture
def trtllm_loader_with_fake_trt(monkeypatch):
    """Reload the TRT-LLM loader with a minimal fake TensorRT-LLM surface."""

    def install_package(name):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, module)
        return module

    def install_module(name):
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        setattr(sys.modules[parent], child, module)
        return module

    for package in [
        "tensorrt_llm",
        "tensorrt_llm._torch",
        "tensorrt_llm._torch.models",
        "tensorrt_llm._torch.models.checkpoints",
        "tensorrt_llm._torch.models.checkpoints.hf",
    ]:
        install_package(package)

    base_config = install_module(
        "tensorrt_llm._torch.models.checkpoints.base_config_loader"
    )
    base_weight = install_module(
        "tensorrt_llm._torch.models.checkpoints.base_weight_loader"
    )
    base_mapper = install_module(
        "tensorrt_llm._torch.models.checkpoints.base_weight_mapper"
    )
    auto_mapper = install_module(
        "tensorrt_llm._torch.models.checkpoints.auto_mapper"
    )
    hf_loader = install_module(
        "tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader"
    )
    modeling_utils = install_module("tensorrt_llm._torch.models.modeling_utils")
    mapping_module = install_module("tensorrt_llm.mapping")

    class BaseConfigLoader:
        pass

    class BaseWeightLoader:
        pass

    class ConsumableWeightsDict:
        def __init__(self, weights):
            self._weights = weights

        def __len__(self):
            return len(self._weights)

        def values(self):
            return list(self._weights.values())

    class BaseWeightMapper:
        pass

    class FakeWeightMapper:
        def __init__(self):
            self.init_calls = []

        def init_model_and_config(self, model, config):
            self.init_calls.append((model, config))

    class AutoCheckpointMapper:
        calls = []

        @staticmethod
        def get(format, name=None):
            AutoCheckpointMapper.calls.append((format, name))
            return FakeWeightMapper()

    class HfCheckpointLoader:
        def __init__(self, *, weight_loader=None, weight_mapper=None, config_loader=None):
            self.weight_loader = weight_loader
            self.weight_mapper = weight_mapper
            self.config_loader = config_loader
            self.disk_loads = []

        def load_weights(self, checkpoint_dir, *, mapping, **kwargs):
            self.disk_loads.append((checkpoint_dir, mapping, kwargs))
            return {"disk.weight": _FakeTensor()}

    class Mapping:
        pass

    def register_checkpoint_loader(name):
        def decorator(cls):
            cls._registered_checkpoint_loader_name = name
            return cls

        return decorator

    base_config.BaseConfigLoader = BaseConfigLoader
    base_weight.BaseWeightLoader = BaseWeightLoader
    base_weight.ConsumableWeightsDict = ConsumableWeightsDict
    base_mapper.BaseWeightMapper = BaseWeightMapper
    auto_mapper.AutoCheckpointMapper = AutoCheckpointMapper
    hf_loader.HfCheckpointLoader = HfCheckpointLoader
    modeling_utils.register_checkpoint_loader = register_checkpoint_loader
    mapping_module.Mapping = Mapping

    reloaded = importlib.reload(trtllm)
    monkeypatch.setattr(reloaded, "_rank_log_scope", _noop_rank_log_scope)
    yield reloaded
    importlib.reload(trtllm)


def test_trtllm_adapter_inherits_engine_adapter():
    assert issubclass(TrtllmAdapter, EngineAdapter)


def test_trtllm_adapter_native_loader_feeds_default_strategy():
    fallback_weights = {"disk.weight": _FakeTensor()}
    model = object()
    adapter = TrtllmAdapter(
        model_name="Qwen/Qwen2.5-7B",
        model=model,
        native_loader=lambda: fallback_weights,
    )

    result = adapter.load_via_native(LoadResult(value=model, model=model))

    assert result.value is fallback_weights
    assert result.model is None
    assert result.publishable is False


def test_trtllm_rdma_receiver_is_not_republished_by_chain():
    model = object()
    adapter = TrtllmAdapter(model_name="Qwen/Qwen2.5-7B", model=model)
    result = LoadResult(value=model, model=model)

    result = adapter.after_rdma_receive(result)

    assert result.publishable is False


def test_trtllm_identity_is_internal_to_adapter():
    adapter = TrtllmAdapter(model_name="Qwen/Qwen2.5-7B")
    identity = adapter.build_identity()

    assert identity.model_name == "Qwen/Qwen2.5-7B"
    assert identity.backend_framework == p2p_pb2.BACKEND_FRAMEWORK_TRT_LLM
    assert identity.dtype == "unknown"


def test_trtllm_identity_uses_live_parameter_dtype():
    model = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(dtype=trtllm_adapter.torch.float16)])
    )
    adapter = TrtllmAdapter(model_name="Qwen/Qwen2.5-7B", model=model)
    identity = adapter.build_identity()

    assert identity.dtype == "float16"


def test_trtllm_load_context_uses_model_config_dtype_and_quantization(monkeypatch):
    monkeypatch.setattr(
        trtllm_adapter,
        "create_metadata_client",
        lambda **kwargs: SimpleNamespace(),
    )
    model = SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=SimpleNamespace(
                torch_dtype=trtllm_adapter.torch.float16,
                model_type="qwen2",
            ),
            quant_config=SimpleNamespace(quant_algo="nvfp4"),
        ),
        parameters=lambda: iter(()),
    )

    ctx = trtllm_adapter.build_trtllm_load_context(
        model_name="Qwen/Qwen2.5-7B",
        checkpoint_dir="/models/qwen",
        model=model,
    )

    assert ctx.model_config.dtype is trtllm_adapter.torch.float16
    assert ctx.model_config.quantization == "nvfp4"
    assert ctx.model_config.hf_text_config.model_type == "qwen2"
    assert ctx.identity.dtype == "float16"
    assert ctx.identity.quantization == "nvfp4"


def test_trtllm_worker_rank_is_shard_key_not_global_rank(monkeypatch):
    del monkeypatch
    mapping = SimpleNamespace(
        rank=5,
        tp_size=2,
        pp_size=2,
        cp_size=2,
        tp_rank=0,
        pp_rank=1,
        moe_ep_size=1,
    )

    adapter = TrtllmAdapter(model_name="Qwen/Qwen2.5-7B", mapping=mapping)
    identity = adapter.build_identity()

    assert adapter.get_global_rank() == 5
    assert adapter.get_worker_rank() == 2
    assert identity.tensor_parallel_size == 2
    assert identity.pipeline_parallel_size == 2


def test_trtllm_worker_rank_collapses_context_parallel_replicas(monkeypatch):
    # TRT-LLM CP ranks share the same model weights for a PP/TP shard, so they
    # must match to the same source worker.
    del monkeypatch
    cp_rank_0 = SimpleNamespace(
        rank=2,
        tp_size=2,
        pp_size=2,
        cp_size=2,
        tp_rank=1,
        pp_rank=0,
        cp_rank=0,
    )
    cp_rank_1 = SimpleNamespace(
        rank=3,
        tp_size=2,
        pp_size=2,
        cp_size=2,
        tp_rank=1,
        pp_rank=0,
        cp_rank=1,
    )

    assert (
        TrtllmAdapter(
            model_name="Qwen/Qwen2.5-7B",
            mapping=cp_rank_0,
        ).get_worker_rank()
        == 1
    )
    assert (
        TrtllmAdapter(
            model_name="Qwen/Qwen2.5-7B",
            mapping=cp_rank_1,
        ).get_worker_rank()
        == 1
    )


def test_checkpoint_loader_runs_shared_load_strategy_chain(
    monkeypatch,
    trtllm_loader_with_fake_trt,
):
    calls = {}

    class FakeClient:
        pass

    def fake_create_metadata_client(**kwargs):
        calls["client_kwargs"] = kwargs
        return FakeClient()

    def fake_chain_run(model_arg, ctx):
        calls["chain_model"] = model_arg
        calls["ctx"] = ctx
        ctx.selected_strategy = "rdma"
        return model_arg

    monkeypatch.setattr(
        trtllm_adapter,
        "create_metadata_client",
        fake_create_metadata_client,
    )
    monkeypatch.setattr(
        trtllm_loader_with_fake_trt.LoadStrategyChain,
        "run",
        fake_chain_run,
    )

    mapping = _fake_trtllm_mapping()
    model = object()
    loader = trtllm_loader_with_fake_trt.MXCheckpointLoader(
        mx_server_url="mx.example:8001",
        model_name="Qwen/Qwen2.5-7B",
    )

    result = loader.load_weights("/models/qwen", mapping=mapping, model=model)

    assert result == {}
    assert loader.p2p_succeeded is True
    assert calls["chain_model"] is model
    assert calls["ctx"].adapter.mapping is mapping
    assert calls["ctx"].adapter.model_name == "Qwen/Qwen2.5-7B"
    assert calls["ctx"].identity.model_name == "Qwen/Qwen2.5-7B"
    assert calls["client_kwargs"] == {
        "worker_rank": 0,
        "server_url": "mx.example:8001",
    }
    assert calls["ctx"].metadata_server_url == "mx.example:8001"


def test_checkpoint_loader_uses_modelexpress_load_format(trtllm_loader_with_fake_trt):
    loader = trtllm_loader_with_fake_trt.MXCheckpointLoader()

    assert loader.checkpoint_format == "modelexpress"


def test_checkpoint_loader_returns_default_strategy_weights(
    monkeypatch,
    trtllm_loader_with_fake_trt,
):
    fallback_weights = {"disk.weight": _FakeTensor()}

    monkeypatch.setattr(
        trtllm_adapter,
        "create_metadata_client",
        lambda **kwargs: SimpleNamespace(),
    )

    def fake_chain_run(model, ctx):
        ctx.selected_strategy = "default"
        return fallback_weights

    monkeypatch.setattr(
        trtllm_loader_with_fake_trt.LoadStrategyChain,
        "run",
        fake_chain_run,
    )

    loader = trtllm_loader_with_fake_trt.MXCheckpointLoader(
        mx_server_url="mx.example:8001",
    )

    result = loader.load_weights(
        "/models/qwen",
        mapping=_fake_trtllm_mapping(),
        model=object(),
    )

    assert result is fallback_weights
    assert loader.p2p_succeeded is False


def test_checkpoint_loader_default_strategy_none_is_not_p2p(
    monkeypatch,
    trtllm_loader_with_fake_trt,
):
    monkeypatch.setattr(
        trtllm_adapter,
        "create_metadata_client",
        lambda **kwargs: SimpleNamespace(),
    )

    def fake_chain_run(model, ctx):
        ctx.selected_strategy = "default"
        return None

    monkeypatch.setattr(
        trtllm_loader_with_fake_trt.LoadStrategyChain,
        "run",
        fake_chain_run,
    )

    loader = trtllm_loader_with_fake_trt.MXCheckpointLoader(
        mx_server_url="mx.example:8001",
    )

    result = loader.load_weights(
        "/models/qwen",
        mapping=_fake_trtllm_mapping(),
        model=object(),
    )

    assert result == {}
    assert loader.p2p_succeeded is False


def test_checkpoint_loader_preserves_native_consumable_weights(
    monkeypatch,
    trtllm_loader_with_fake_trt,
):
    fallback_weights = trtllm_loader_with_fake_trt.ConsumableWeightsDict(
        {"disk.weight": _FakeTensor()}
    )
    monkeypatch.setattr(
        trtllm_adapter,
        "create_metadata_client",
        lambda **kwargs: SimpleNamespace(),
    )

    def fake_chain_run(model, ctx):
        ctx.selected_strategy = "default"
        return fallback_weights

    monkeypatch.setattr(
        trtllm_loader_with_fake_trt.LoadStrategyChain,
        "run",
        fake_chain_run,
    )

    loader = trtllm_loader_with_fake_trt.MXCheckpointLoader(
        mx_server_url="mx.example:8001",
    )

    result = loader.load_weights(
        "/models/qwen",
        mapping=_fake_trtllm_mapping(),
        model=object(),
    )

    assert result is fallback_weights
    assert loader.p2p_succeeded is False


def test_checkpoint_loader_uses_hf_mapper_for_mx_fallback(
    trtllm_loader_with_fake_trt,
):
    loader = trtllm_loader_with_fake_trt.MXCheckpointLoader()
    model = object()
    config = SimpleNamespace(
        pretrained_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"])
    )

    mapper = loader.get_initialized_weight_mapper(model, config)

    assert trtllm_loader_with_fake_trt.AutoCheckpointMapper.calls == [
        ("HF", "Qwen3ForCausalLM")
    ]
    assert mapper.init_calls == [(model, config)]


def test_checkpoint_loader_model_kwarg_is_not_forwarded_to_disk_fallback(
    trtllm_loader_with_fake_trt,
):
    mapping = object()
    loader = trtllm_loader_with_fake_trt.MXCheckpointLoader()

    result = loader.load_weights(
        "/models/qwen",
        mapping=mapping,
        model=None,
        extra_option="kept",
    )

    assert list(result) == ["disk.weight"]
    assert loader.disk_loads == [
        ("/models/qwen", mapping, {"extra_option": "kept"}),
    ]


def test_model_less_load_does_not_clear_publish_context(
    monkeypatch,
    trtllm_loader_with_fake_trt,
):
    captured = {}

    monkeypatch.setattr(
        trtllm_adapter,
        "create_metadata_client",
        lambda **kwargs: SimpleNamespace(),
    )

    def fake_chain_run(model, ctx):
        captured["load_ctx"] = ctx
        ctx.selected_strategy = "rdma"
        return model

    monkeypatch.setattr(
        trtllm_loader_with_fake_trt.LoadStrategyChain,
        "run",
        fake_chain_run,
    )

    def fake_publish(result, ctx):
        captured["publish_result"] = result
        captured["publish_ctx"] = ctx

    import modelexpress.load_strategy as load_strategy

    monkeypatch.setattr(load_strategy, "publish_loaded_model", fake_publish)

    model = object()
    mapping = _fake_trtllm_mapping()
    loader = trtllm_loader_with_fake_trt.MXCheckpointLoader(
        mx_server_url="mx.example:8001",
        model_name="Qwen/Qwen2.5-7B",
    )

    loader.load_weights("/models/qwen", mapping=mapping, model=model)
    loader.load_weights("/models/draft", mapping=mapping)
    loader.post_load_publish(model, checkpoint_dir="/models/qwen")

    assert captured["publish_result"].model is model
    assert captured["publish_ctx"] is captured["load_ctx"]
    assert loader.disk_loads == [("/models/draft", mapping, {})]


def test_post_load_publish_reuses_load_context_worker_id(
    monkeypatch,
    trtllm_loader_with_fake_trt,
):
    captured = {}
    fallback_weights = {"disk.weight": _FakeTensor()}

    monkeypatch.setattr(
        trtllm_adapter,
        "create_metadata_client",
        lambda **kwargs: SimpleNamespace(),
    )

    def fake_chain_run(model, ctx):
        captured["load_ctx"] = ctx
        ctx.selected_strategy = "default"
        return fallback_weights

    monkeypatch.setattr(
        trtllm_loader_with_fake_trt.LoadStrategyChain,
        "run",
        fake_chain_run,
    )

    def fake_publish(result, ctx):
        captured["publish_result"] = result
        captured["publish_ctx"] = ctx

    import modelexpress.load_strategy as load_strategy

    monkeypatch.setattr(load_strategy, "publish_loaded_model", fake_publish)

    model = object()
    loader = trtllm_loader_with_fake_trt.MXCheckpointLoader(
        mx_server_url="mx.example:8001",
        model_name="Qwen/Qwen2.5-7B",
    )

    result = loader.load_weights(
        "/models/qwen",
        mapping=_fake_trtllm_mapping(),
        model=model,
    )
    loader.post_load_publish(model, checkpoint_dir="/models/qwen")

    assert result is fallback_weights
    assert captured["publish_result"].model is model
    assert captured["publish_ctx"] is captured["load_ctx"]
    assert captured["publish_ctx"].worker_id == captured["load_ctx"].worker_id


def test_post_load_publish_publishes_rdma_receiver(
    monkeypatch,
    trtllm_loader_with_fake_trt,
):
    captured = {}

    monkeypatch.setattr(
        trtllm_adapter,
        "create_metadata_client",
        lambda **kwargs: SimpleNamespace(),
    )

    def fake_chain_run(model, ctx):
        captured["load_ctx"] = ctx
        ctx.selected_strategy = "rdma"
        return model

    monkeypatch.setattr(
        trtllm_loader_with_fake_trt.LoadStrategyChain,
        "run",
        fake_chain_run,
    )

    def fake_publish(result, ctx):
        captured["publish_result"] = result
        captured["publish_ctx"] = ctx

    import modelexpress.load_strategy as load_strategy

    monkeypatch.setattr(load_strategy, "publish_loaded_model", fake_publish)

    model = object()
    loader = trtllm_loader_with_fake_trt.MXCheckpointLoader(
        mx_server_url="mx.example:8001",
        model_name="Qwen/Qwen2.5-7B",
    )

    result = loader.load_weights(
        "/models/qwen",
        mapping=_fake_trtllm_mapping(),
        model=model,
    )
    loader.post_load_publish(
        model,
        checkpoint_dir="/models/qwen",
        weights_preloaded=True,
    )

    assert result == {}
    assert loader.p2p_succeeded is True
    assert captured["publish_result"].model is model
    assert captured["publish_ctx"] is captured["load_ctx"]
    assert captured["publish_ctx"].worker_id == captured["load_ctx"].worker_id


def test_publish_as_source_reuses_matching_load_context(
    monkeypatch,
    trtllm_loader_with_fake_trt,
):
    captured = {}
    fallback_weights = {"disk.weight": _FakeTensor()}

    monkeypatch.setattr(
        trtllm_adapter,
        "create_metadata_client",
        lambda **kwargs: SimpleNamespace(),
    )

    def fake_chain_run(model, ctx):
        captured["load_ctx"] = ctx
        ctx.selected_strategy = "default"
        return fallback_weights

    monkeypatch.setattr(
        trtllm_loader_with_fake_trt.LoadStrategyChain,
        "run",
        fake_chain_run,
    )

    def fake_publish(result, ctx):
        captured["publish_result"] = result
        captured["publish_ctx"] = ctx

    import modelexpress.load_strategy as load_strategy

    monkeypatch.setattr(load_strategy, "publish_loaded_model", fake_publish)

    model = object()
    loader = trtllm_loader_with_fake_trt.MXCheckpointLoader(
        mx_server_url="mx.example:8001",
        model_name="Qwen/Qwen2.5-7B",
    )
    build_calls = []
    original_build_context = loader._build_load_context

    def counting_build_context(**kwargs):
        build_calls.append(kwargs)
        return original_build_context(**kwargs)

    loader._build_load_context = counting_build_context

    loader.load_weights(
        "/models/qwen",
        mapping=_fake_trtllm_mapping(),
        model=model,
    )
    loader.publish_as_source(model, checkpoint_dir="/models/qwen")

    assert captured["publish_result"].model is model
    assert captured["publish_ctx"] is captured["load_ctx"]
    assert len(build_calls) == 1


def test_checkpoint_loader_timeout_waits_then_uses_chain_default(
    monkeypatch,
    trtllm_loader_with_fake_trt,
):
    calls = {}

    monkeypatch.setattr(
        trtllm_adapter,
        "create_metadata_client",
        lambda **kwargs: SimpleNamespace(),
    )

    def fake_chain_run(model, ctx):
        calls["chain_ctx"] = ctx
        ctx.selected_strategy = "default"
        return ctx.adapter.load_via_native(LoadResult(value=model, model=model)).value

    monkeypatch.setattr(
        trtllm_loader_with_fake_trt.LoadStrategyChain,
        "run",
        fake_chain_run,
    )

    loader = trtllm_loader_with_fake_trt.MXCheckpointLoader(
        mx_server_url="mx.example:8001",
        query_timeout_s=0,
    )

    result = loader.load_weights(
        "/models/qwen",
        mapping=_fake_trtllm_mapping(),
        model=object(),
    )

    assert list(result) == ["disk.weight"]
    assert calls["chain_ctx"].worker_rank == 0
    assert calls["chain_ctx"].load_config.source_query_timeout_s == 0
    assert loader.p2p_succeeded is False


def test_checkpoint_loader_falls_back_when_chain_fails(
    monkeypatch,
    trtllm_loader_with_fake_trt,
):
    monkeypatch.setattr(
        trtllm_adapter,
        "create_metadata_client",
        lambda **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        trtllm_loader_with_fake_trt.LoadStrategyChain,
        "run",
        lambda model, ctx: (_ for _ in ()).throw(RuntimeError("transfer failed")),
    )

    loader = trtllm_loader_with_fake_trt.MXCheckpointLoader(
        mx_server_url="mx.example:8001",
    )

    result = loader.load_weights(
        "/models/qwen",
        mapping=_fake_trtllm_mapping(),
        model=object(),
    )

    assert list(result) == ["disk.weight"]
    assert loader.p2p_succeeded is False


def test_checkpoint_loader_publish_without_load_context_skips(
    caplog,
    monkeypatch,
    trtllm_loader_with_fake_trt,
):
    published = []

    def fake_publish(model, ctx):
        published.append((model, ctx))

    monkeypatch.setattr("modelexpress.load_strategy.publish_loaded_model", fake_publish)

    model = object()
    loader = trtllm_loader_with_fake_trt.MXCheckpointLoader(
        mx_server_url="mx.example:8001",
        model_name="Qwen/Qwen2.5-7B",
    )

    with caplog.at_level(trtllm_loader_with_fake_trt.logging.WARNING):
        loader._publish_as_source(model, checkpoint_dir="/unused/path", mapping=object())

    assert published == []
    assert "matching load context" in caplog.text


def test_publish_loaded_model_uses_shared_strategy_helpers(monkeypatch):
    captured = {}

    nixl_manager = object()
    ctx = SimpleNamespace(nixl_manager=None)
    model = SimpleNamespace()

    def fake_register_tensors(result, ctx_arg):
        captured["register_result"] = result
        captured["register_ctx"] = ctx_arg
        ctx_arg.nixl_manager = nixl_manager

    def fake_publish_metadata(ctx_arg):
        captured["publish_ctx"] = ctx_arg

    import modelexpress.load_strategy.base as load_strategy_base
    from modelexpress.load_strategy import publish_loaded_model

    monkeypatch.setattr(load_strategy_base, "register_tensors", fake_register_tensors)
    monkeypatch.setattr(load_strategy_base, "publish_metadata", fake_publish_metadata)

    publish_loaded_model(LoadResult(value=model, model=model), ctx)

    assert captured["register_result"].model is model
    assert captured["register_ctx"] is ctx
    assert captured["publish_ctx"] is ctx
    assert ctx.nixl_manager is nixl_manager
    assert not hasattr(model, "_mx_nixl_managers")
    assert model._mx_load_context is ctx


def test_publish_loaded_model_skips_non_publishable_result(monkeypatch):
    captured = {}
    ctx = SimpleNamespace()
    model = SimpleNamespace()
    result = LoadResult(value=model, model=model, publishable=False)

    def fake_register_tensors(result, ctx_arg):
        captured["register"] = (result, ctx_arg)

    def fake_publish_metadata(ctx_arg):
        captured["publish"] = ctx_arg

    import modelexpress.load_strategy.base as load_strategy_base
    from modelexpress.load_strategy import publish_loaded_model

    monkeypatch.setattr(load_strategy_base, "register_tensors", fake_register_tensors)
    monkeypatch.setattr(load_strategy_base, "publish_metadata", fake_publish_metadata)

    publish_loaded_model(result, ctx)

    assert captured == {}
    assert not hasattr(model, "_mx_load_context")


def test_rank_log_scope_flushes_fsyncs_and_mirrors_to_stderr(
    tmp_path, monkeypatch, capsys
):
    fsync_calls = []
    monkeypatch.setenv("MX_TRANSFER_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(trtllm.os, "fsync", lambda fd: fsync_calls.append(fd))

    with trtllm._rank_log_scope(3):
        trtllm.logging.getLogger("modelexpress").info("transfer metric")
    with trtllm._rank_log_scope(3):
        trtllm.logging.getLogger("modelexpress").info("publish metric")

    assert fsync_calls
    rank_log_text = (tmp_path / "rank3.log").read_text()
    stderr_text = capsys.readouterr().err
    assert "[ModelExpress] [I] transfer metric" in rank_log_text
    assert "[ModelExpress] [I] publish metric" in rank_log_text
    assert "[ModelExpress] [I] transfer metric" in stderr_text
    assert "[ModelExpress] [I] publish metric" in stderr_text
