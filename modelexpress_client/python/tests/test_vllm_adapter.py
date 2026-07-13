# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the vLLM engine adapter."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import pytest

from modelexpress.types import TensorDescriptor
from modelexpress.types import ManifestMismatchError
from modelexpress.load_strategy.context import LoadResult
from modelexpress.engines.vllm.adapter import (
    VllmAdapter,
    _apply_source_manifest_tensor_structure,
    _get_vllm_device_id,
    _get_vllm_worker_rank,
    _refresh_vllm_attention_runtime_tensors,
    build_vllm_load_context,
)
from modelexpress.metadata.publish import build_source_identity
from modelexpress.tensor_utils import _manifest_tensor_for_module_leaf
from modelexpress.quantization_providers.humming import HummingManifestProvider


def _vllm_config(*, rank: int, tp_size: int, pp_size: int):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            rank=rank,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
        )
    )


def test_worker_rank_uses_torch_distributed_global_rank():
    config = _vllm_config(rank=2, tp_size=4, pp_size=2)
    device = torch.device("cuda", 0)

    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_rank", return_value=6,
    ):
        assert _get_vllm_worker_rank(config, device) == 6


def test_worker_rank_distinguishes_dp_replicas():
    config = _vllm_config(rank=0, tp_size=4, pp_size=2)
    device = torch.device("cuda", 0)

    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_rank", return_value=5,
    ):
        dp0_rank = _get_vllm_worker_rank(config, device)

    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_rank", return_value=13,
    ):
        dp1_rank = _get_vllm_worker_rank(config, device)

    assert dp0_rank == 5
    assert dp1_rank == 13


def test_worker_rank_falls_back_to_parallel_config_rank_pre_init():
    # Pre-init / bare-cuda path: torch.distributed not initialised AND device
    # has no index. Falls back to parallel_config.rank so workers in the same
    # DP still get distinct keys.
    config = _vllm_config(rank=3, tp_size=4, pp_size=2)
    bare_device = torch.device("cuda")

    with patch("torch.distributed.is_initialized", return_value=False):
        assert _get_vllm_worker_rank(config, bare_device) == 3


def test_vllm_device_id_uses_current_platform_device(monkeypatch):
    fake_platforms = SimpleNamespace(
        current_platform=SimpleNamespace(
            current_device=lambda: 2,
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)

    assert _get_vllm_device_id(torch.device("cuda")) == 2


def test_vllm_is_cuda_alike_uses_current_platform(monkeypatch):
    fake_platforms = SimpleNamespace(
        current_platform=SimpleNamespace(
            is_cuda_alike=lambda: True,
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())

    assert adapter.is_cuda_alike() is True


def test_prepare_rdma_target_preserves_extra_config_for_dummy_loader(monkeypatch):
    captured = {}

    class DummyModelLoader:
        def __init__(self, load_config):
            captured["load_config"] = load_config

        def load_weights(self, model, model_config):
            captured["model"] = model
            captured["model_config"] = model_config

    fake_dummy_loader = SimpleNamespace(DummyModelLoader=DummyModelLoader)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.dummy_loader",
        fake_dummy_loader,
    )

    vllm_config = _context_config(load_device="cpu")
    vllm_config.load_config.model_loader_extra_config = {"concurrency": 4}
    model_config = _model_config()
    adapter = VllmAdapter(vllm_config, model_config)
    model = MagicMock()

    result = adapter.prepare_rdma_target(LoadResult(value=model, model=model))

    assert result.model is model
    assert captured["model"] is model
    assert captured["model_config"] is model_config
    dummy_config = captured["load_config"]
    assert dummy_config.load_format == "dummy"
    assert dummy_config.model_loader_extra_config == {"concurrency": 4}
    assert vllm_config.load_config.model_loader_extra_config == {"concurrency": 4}


def test_prepare_rdma_target_preserves_quantization_extra_config(monkeypatch):
    captured = {}

    class DummyModelLoader:
        def __init__(self, load_config):
            captured["load_config"] = load_config

        def load_weights(self, model, model_config):
            pass

    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.dummy_loader",
        SimpleNamespace(DummyModelLoader=DummyModelLoader),
    )

    extra_config = {
        "quantization": "humming",
        "humming": {"group_size": 128},
    }
    vllm_config = _context_config(load_device="cpu")
    vllm_config.load_config.model_loader_extra_config = extra_config

    adapter = VllmAdapter(vllm_config, _model_config())
    model = MagicMock()

    adapter.prepare_rdma_target(LoadResult(value=model, model=model))

    dummy_config = captured["load_config"]
    assert dummy_config.load_format == "dummy"
    assert dummy_config.model_loader_extra_config == extra_config


def test_load_via_native_captures_humming_runtime_during_weight_load(monkeypatch):
    humming_pkg = SimpleNamespace()
    humming_layer = SimpleNamespace()

    class FakeHummingLayerMethod:
        @classmethod
        def may_set_param(cls, layer, name, tensor):
            setattr(layer, name, nn.Parameter(tensor, requires_grad=False))

    humming_layer.HummingLayerMethod = FakeHummingLayerMethod
    monkeypatch.setitem(sys.modules, "humming", humming_pkg)
    monkeypatch.setitem(sys.modules, "humming.layer", humming_layer)

    class DefaultModelLoader:
        def __init__(self, load_config):
            self.load_config = load_config

        def load_weights(self, model, model_config):
            FakeHummingLayerMethod.may_set_param(
                model, "weight", torch.empty(8, dtype=torch.int32),
            )

    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.default_loader",
        SimpleNamespace(DefaultModelLoader=DefaultModelLoader),
    )

    vllm_config = _context_config(load_device="cpu")
    model_config = _model_config()
    model_config.quantization = "humming"
    adapter = VllmAdapter(vllm_config, model_config)
    model = nn.Module()

    result = adapter.load_via_native(LoadResult(value=model, model=model))

    assert result.model is model
    assert model._mx_rdma_tensors["weight"].data_ptr() == model.weight.data_ptr()
    assert model.weight.mx_runtime_role == "humming_packed_weight"


def test_apply_weight_iter_captures_humming_runtime(monkeypatch):
    humming_pkg = SimpleNamespace()
    humming_layer = SimpleNamespace()

    class FakeHummingLayerMethod:
        @classmethod
        def may_set_param(cls, layer, name, tensor):
            setattr(layer, name, nn.Parameter(tensor, requires_grad=False))

    humming_layer.HummingLayerMethod = FakeHummingLayerMethod
    monkeypatch.setitem(sys.modules, "humming", humming_pkg)
    monkeypatch.setitem(sys.modules, "humming.layer", humming_layer)

    class FakeModel(nn.Module):
        def load_weights(self, weights_iter):
            list(weights_iter)
            FakeHummingLayerMethod.may_set_param(
                self, "weight", torch.empty(8, dtype=torch.int32),
            )

    vllm_config = _context_config(load_device="cpu")
    model_config = _model_config()
    model_config.quantization = "humming"
    adapter = VllmAdapter(vllm_config, model_config)
    model = FakeModel()

    result = adapter.apply_weight_iter(
        LoadResult(value=model, model=model),
        iter([("weight", torch.empty(8, dtype=torch.int32))]),
    )

    assert result.model is model
    assert model._mx_rdma_tensors["weight"].data_ptr() == model.weight.data_ptr()
    assert model.weight.mx_runtime_role == "humming_packed_weight"


def test_build_source_identity_includes_quantization_extra_config():
    extra_config = {
        "quantization": "humming",
        "humming": {"group_size": 128},
    }
    vllm_config = _context_config(load_device="cpu")
    vllm_config.load_config.model_loader_extra_config = extra_config
    model_config = _model_config()
    model_config.quantization = "humming"

    identity = build_source_identity(vllm_config, model_config)

    assert identity.quantization == "humming"
    assert identity.extra_parameters["mx_manifest_schema_version"] == "2"
    assert identity.extra_parameters["mx_tensor_discovery_version"] == "2"
    assert (
        identity.extra_parameters["vllm_model_loader_extra_config"]
        == '{"humming":{"group_size":128},"quantization":"humming"}'
    )


def test_source_manifest_prepare_rebuilds_parameter_structure():
    model = nn.Module()
    model.proj = nn.Linear(4, 4, bias=False)
    original_weight = model.proj.weight
    original_nbytes = model.proj.weight.numel() * model.proj.weight.element_size()
    desc = TensorDescriptor(
        name="proj.weight",
        addr=0,
        size=8,
        device_id=0,
        dtype="uint8",
        shape=[8],
        stride=[1],
        storage_offset=0,
        storage_nbytes=8,
        layout_kind="contiguous",
        original_shape=[4, 4],
        original_dtype="torch.float32",
        original_nbytes=original_nbytes,
    )

    replaced = _apply_source_manifest_tensor_structure(
        model, [desc], torch.device("cpu"), provider=HummingManifestProvider(),
    )

    assert replaced == 1
    assert model.proj.weight is not original_weight
    assert model.proj.weight.shape == torch.Size([8])
    assert model.proj.weight.dtype == torch.uint8
    assert model.proj.weight.requires_grad is False


def test_source_manifest_prepare_overrides_stale_humming_runtime_mapping():
    model = nn.Module()
    model.block = nn.Module()
    model.block.weight = nn.Parameter(
        torch.empty((72, 1536), dtype=torch.int32),
        requires_grad=False,
    )
    stale_packed = model.block.weight.data
    model.block._mx_rdma_tensors = {"weight": stale_packed}
    model.block.humming_metas = {"": object()}
    desc = TensorDescriptor(
        name="block.weight",
        addr=0,
        size=24,
        device_id=0,
        dtype="torch.bfloat16",
        shape=[3, 4],
        stride=[4, 1],
        storage_offset=0,
        storage_nbytes=24,
        layout_kind="contiguous",
        original_nbytes=stale_packed.numel() * stale_packed.element_size(),
        runtime_role="weight",
        replace_policy="structural_replace",
    )

    replaced = _apply_source_manifest_tensor_structure(
        model, [desc], torch.device("cpu"), provider=HummingManifestProvider(),
    )
    tensor, runtime_role, replace_policy = _manifest_tensor_for_module_leaf(
        model.block,
        "weight",
        model.block.weight.data,
        "humming",
    )

    assert replaced == 1
    assert tensor.data_ptr() == model.block.weight.data.data_ptr()
    assert tensor.data_ptr() != stale_packed.data_ptr()
    assert runtime_role == "weight"
    assert replace_policy == "structural_replace"
    assert "weight" not in model.block._mx_rdma_tensors


def test_source_manifest_prepare_aligns_unquantized_quant_method(monkeypatch):
    class FakeHummingMethod:
        pass

    FakeHummingMethod.__module__ = "vllm.model_executor.layers.quantization.humming"

    class UnquantizedLinearMethod:
        pass

    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.linear",
        SimpleNamespace(UnquantizedLinearMethod=UnquantizedLinearMethod),
    )

    model = nn.Module()
    model.block = nn.Module()
    model.block.quant_method = FakeHummingMethod()
    model.block.weight = nn.Parameter(
        torch.empty((72, 1024), dtype=torch.int32),
        requires_grad=False,
    )
    original_nbytes = model.block.weight.numel() * model.block.weight.element_size()
    desc = TensorDescriptor(
        name="block.weight",
        addr=0,
        size=995328,
        device_id=0,
        dtype="torch.bfloat16",
        shape=[432, 1152],
        stride=[1152, 1],
        storage_offset=0,
        storage_nbytes=995328,
        layout_kind="contiguous",
        original_nbytes=original_nbytes,
        quant_method="vllm.model_executor.layers.linear.UnquantizedLinearMethod",
        runtime_role="weight",
        replace_policy="structural_replace",
    )

    replaced = _apply_source_manifest_tensor_structure(
        model, [desc], torch.device("cpu"),
    )

    assert replaced == 1
    assert isinstance(model.block.quant_method, UnquantizedLinearMethod)
    assert model.block.weight.shape == torch.Size([432, 1152])
    assert model.block.weight.dtype == torch.bfloat16


def test_source_manifest_prepare_rebuilds_buffer_structure():
    model = nn.Module()
    model.block = nn.Module()
    model.block.register_buffer("packed", torch.empty(4, dtype=torch.float32))
    original_buffer = model.block.packed
    original_nbytes = model.block.packed.numel() * model.block.packed.element_size()
    desc = TensorDescriptor(
        name="block.packed",
        addr=0,
        size=4,
        device_id=0,
        dtype="uint8",
        shape=[4],
        stride=[1],
        storage_offset=0,
        storage_nbytes=4,
        layout_kind="contiguous",
        original_nbytes=original_nbytes,
    )

    replaced = _apply_source_manifest_tensor_structure(
        model, [desc], torch.device("cpu"),
    )

    assert replaced == 1
    assert model.block.packed is not original_buffer
    assert model.block.packed.shape == torch.Size([4])
    assert model.block.packed.dtype == torch.uint8


def test_source_manifest_prepare_rejects_original_size_mismatch():
    model = nn.Module()
    model.proj = nn.Linear(4, 4, bias=False)
    desc = TensorDescriptor(
        name="proj.weight",
        addr=0,
        size=8,
        device_id=0,
        dtype="uint8",
        shape=[8],
        stride=[1],
        storage_offset=0,
        storage_nbytes=8,
        layout_kind="contiguous",
        original_nbytes=123,
    )

    with pytest.raises(ManifestMismatchError, match="original_nbytes"):
        _apply_source_manifest_tensor_structure(model, [desc], torch.device("cpu"))


def test_prepare_rdma_target_from_manifest_rebuilds_non_humming_structure():
    vllm_config = _context_config(load_device="cpu")
    model_config = _model_config()
    model_config.quantization = None
    adapter = VllmAdapter(vllm_config, model_config)
    model = nn.Module()
    model.proj = nn.Linear(4, 4, bias=False)
    original_weight = model.proj.weight
    desc = TensorDescriptor(
        name="proj.weight",
        addr=0,
        size=8,
        device_id=0,
        dtype="uint8",
        shape=[8],
        stride=[1],
        storage_offset=0,
        storage_nbytes=8,
        layout_kind="contiguous",
        original_nbytes=original_weight.numel() * original_weight.element_size(),
    )

    result = adapter.prepare_rdma_target_from_manifest(
        LoadResult(value=model, model=model), [desc],
    )

    assert result.model is model
    assert model.proj.weight is not original_weight
    assert model.proj.weight.shape == torch.Size([8])
    assert model.proj.weight.dtype == torch.uint8


def test_prepare_rdma_target_from_manifest_keeps_matching_tensor():
    vllm_config = _context_config(load_device="cpu")
    model_config = _model_config()
    model_config.quantization = None
    adapter = VllmAdapter(vllm_config, model_config)
    model = nn.Module()
    model.proj = nn.Linear(4, 4, bias=False)
    original_weight = model.proj.weight
    desc = TensorDescriptor(
        name="proj.weight",
        addr=0,
        size=original_weight.numel() * original_weight.element_size(),
        device_id=0,
        dtype="torch.float32",
        shape=[4, 4],
        stride=[4, 1],
        storage_offset=0,
        storage_nbytes=original_weight.numel() * original_weight.element_size(),
        layout_kind="contiguous",
        original_nbytes=original_weight.numel() * original_weight.element_size(),
    )

    result = adapter.prepare_rdma_target_from_manifest(
        LoadResult(value=model, model=model), [desc],
    )

    assert result.model is model
    assert model.proj.weight is original_weight


def test_prepare_rdma_target_from_manifest_rejects_humming_packed_rebuild():
    vllm_config = _context_config(load_device="cpu")
    model_config = _model_config()
    model_config.quantization = "humming"
    adapter = VllmAdapter(vllm_config, model_config)
    model = nn.Module()
    model.block = nn.Module()
    model.block.weight = nn.Parameter(
        torch.empty((72, 1024), dtype=torch.int32),
        requires_grad=False,
    )
    original_weight = model.block.weight
    desc = TensorDescriptor(
        name="block.weight",
        addr=0,
        size=995328,
        device_id=0,
        dtype="torch.bfloat16",
        shape=[432, 1152],
        stride=[1152, 1],
        storage_offset=0,
        storage_nbytes=995328,
        layout_kind="contiguous",
        tensor_kind="parameter",
        owner_module="block",
        owner_class="FakeHummingLinear",
        quant_method="vllm.model_executor.layers.quantization.humming.HummingMethod",
        runtime_role="humming_packed_weight",
        replace_policy="no_structural_replace",
    )

    with pytest.raises(ManifestMismatchError, match="no_structural_replace"):
        adapter.prepare_rdma_target_from_manifest(
            LoadResult(value=model, model=model), [desc],
        )

    assert model.block.weight is original_weight
    assert model.block.weight.shape == torch.Size([72, 1024])
    assert model.block.weight.dtype == torch.int32


def test_prepare_rdma_target_from_manifest_keeps_matching_humming_packed_tensor():
    class FakeHummingMethod:
        pass

    FakeHummingMethod.__module__ = "vllm.model_executor.layers.quantization.humming"

    vllm_config = _context_config(load_device="cpu")
    model_config = _model_config()
    model_config.quantization = "humming"
    adapter = VllmAdapter(vllm_config, model_config)
    model = nn.Module()
    model.block = nn.Module()
    model.block.quant_method = FakeHummingMethod()
    model.block.weight = nn.Parameter(
        torch.empty((72, 1024), dtype=torch.int32),
        requires_grad=False,
    )
    original_weight = model.block.weight
    original_quant_method = model.block.quant_method
    desc = TensorDescriptor(
        name="block.weight",
        addr=0,
        size=original_weight.numel() * original_weight.element_size(),
        device_id=0,
        dtype="torch.int32",
        shape=[72, 1024],
        stride=[1024, 1],
        storage_offset=0,
        storage_nbytes=original_weight.numel() * original_weight.element_size(),
        layout_kind="contiguous",
        tensor_kind="parameter",
        owner_module="block",
        owner_class="FakeHummingLinear",
        quant_method="vllm.model_executor.layers.linear.UnquantizedLinearMethod",
        runtime_role="weight",
        replace_policy="structural_replace",
    )

    result = adapter.prepare_rdma_target_from_manifest(
        LoadResult(value=model, model=model), [desc],
    )

    assert result.model is model
    assert model.block.weight is original_weight
    assert model.block.quant_method is original_quant_method


def test_after_rdma_receive_refreshes_attention_runtime_tensors(monkeypatch):
    entered = []

    class device_loading_context:
        def __init__(self, module, target_device):
            self.module = module
            self.target_device = target_device

        def __enter__(self):
            entered.append((self.module, self.target_device))
            return self.module

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.utils",
        SimpleNamespace(device_loading_context=device_loading_context),
    )

    class MLAAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Parameter(torch.tensor([1.0]))
            self.W_UV = torch.tensor([0.0])
            self.calls = 0

        def process_weights_after_loading(self, dtype):
            self.calls += 1
            self.W_UV = self.base.detach().to(dtype=dtype) + 1

    class FakeQuantMethod:
        def __init__(self):
            self.calls = 0

        def process_weights_after_loading(self, module):
            self.calls += 1

    model = nn.Module()
    model.mla = MLAAttention()
    model.linear = nn.Module()
    model.linear.quant_method = FakeQuantMethod()
    model.mla.base.data.fill_(41.0)

    vllm_config = _context_config(load_device="cpu")
    adapter = VllmAdapter(vllm_config, _model_config())

    result = adapter.after_rdma_receive(LoadResult(value=model, model=model))

    assert result.model is model
    assert model.mla.calls == 1
    assert model.mla.W_UV.item() == 42.0
    assert model.linear.quant_method.calls == 0
    assert entered == [(model.mla, torch.device("cpu"))]


def test_refresh_attention_runtime_tensors_matches_vllm_class_names():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def process_weights_after_loading(self, dtype):
            self.calls += 1

    class MMEncoderAttention(Attention):
        pass

    class NotAttention(Attention):
        pass

    model = nn.Module()
    model.attn = Attention()
    model.mm = MMEncoderAttention()
    model.other = NotAttention()

    refreshed = _refresh_vllm_attention_runtime_tensors(
        model,
        _model_config(),
        torch.device("cpu"),
    )

    assert refreshed == 2
    assert model.attn.calls == 1
    assert model.mm.calls == 1
    assert model.other.calls == 0


def test_build_vllm_load_context_uses_current_platform_for_bare_cuda(monkeypatch):
    _stub_vllm_current_device(monkeypatch, current_device=2)
    _stub_metadata_client(monkeypatch)
    vllm_config = _context_config(load_device=None)

    ctx = build_vllm_load_context(vllm_config, _model_config())

    assert ctx.target_device == torch.device("cuda")
    assert ctx.target_device.index is None
    assert ctx.device_id == 2


def test_build_vllm_load_context_keeps_explicit_cuda_index(monkeypatch):
    _stub_vllm_current_device(monkeypatch, current_device=2)
    _stub_metadata_client(monkeypatch)
    vllm_config = _context_config(load_device="cuda:3")

    ctx = build_vllm_load_context(vllm_config, _model_config())

    assert ctx.target_device == torch.device("cuda:3")
    assert ctx.target_device.index == 3
    assert ctx.device_id == ctx.target_device.index


def _stub_vllm_current_device(monkeypatch, *, current_device: int) -> None:
    fake_platforms = SimpleNamespace(
        current_platform=SimpleNamespace(
            current_device=lambda: current_device,
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)


def _stub_metadata_client(monkeypatch) -> None:
    monkeypatch.setattr(
        "modelexpress.engines.vllm.adapter.create_metadata_client",
        lambda worker_rank: object(),
    )


def _context_config(*, load_device):
    return SimpleNamespace(
        device_config=SimpleNamespace(device="cuda"),
        load_config=SimpleNamespace(device=load_device),
        parallel_config=SimpleNamespace(
            rank=0,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        ),
    )


def _model_config():
    return SimpleNamespace(
        dtype=torch.bfloat16,
        model="test-model",
        quantization=None,
        revision=None,
    )
