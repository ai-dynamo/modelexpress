# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tensor_utils: hidden tensor adoption, capture_tensor_attrs, checksums."""

import sys
import types

import torch
import torch.nn as nn
import pytest

from modelexpress.tensor_utils import (
    SOURCE_MANIFEST_TENSOR_NAMES_ATTR,
    _find_hidden_cuda_tensors,
    adopt_hidden_tensors,
    capture_humming_runtime_tensors_from_model,
    capture_humming_runtime_tensors,
    capture_tensor_attrs,
    collect_module_tensors,
    safe_checksum,
    storage_view,
    tensor_descriptor_layout,
)
from modelexpress.quantization_providers.humming import HummingManifestProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class QuantConfig:
    """Simulates a quant config with hidden tensors (like FusedMoEQuantConfig)."""

    def __init__(self, device="cpu"):
        self.a1_gscale = torch.randn(8, device=device)
        self.a2_gscale = torch.randn(8, device=device)
        self.some_int = 42
        self.some_string = "hello"


class NestedObj:
    """Object with tensors nested in dicts and lists."""

    def __init__(self, device="cpu"):
        self.scales = {"w1": torch.randn(4, device=device), "w2": torch.randn(4, device=device)}
        self.buffers = [torch.randn(2, device=device)]


class FakeQuant:
    """Simulates a quant method object with a config."""

    def __init__(self, device="cpu"):
        self.config = QuantConfig(device)


class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


class ModuleWithQuant(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.quant_method = FakeQuant(device="cpu")


class ModuleWithNested(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.nested = NestedObj(device="cpu")


class FakeCudaTensor:
    is_cuda = True
    dtype = torch.float32
    shape = torch.Size([1])
    data = None

    def __init__(self, ptr):
        self._ptr = ptr
        self.data = self

    def numel(self):
        return 1

    def element_size(self):
        return 4

    def data_ptr(self):
        return self._ptr

    def is_contiguous(self):
        return True


# ---------------------------------------------------------------------------
# _find_hidden_cuda_tensors
# ---------------------------------------------------------------------------


def test_tensor_descriptor_layout_includes_runtime_metadata():
    tensor = torch.empty(4)
    tensor.mx_tensor_kind = "parameter"
    tensor.mx_owner_module = "block"
    tensor.mx_owner_class = "FakeLinear"
    tensor.mx_quant_method = "humming.HummingMethod"
    tensor.mx_runtime_role = "humming_packed_weight"
    tensor.mx_replace_policy = "no_structural_replace"

    layout = tensor_descriptor_layout(tensor)

    assert layout["tensor_kind"] == "parameter"
    assert layout["owner_module"] == "block"
    assert layout["owner_class"] == "FakeLinear"
    assert layout["quant_method"] == "humming.HummingMethod"
    assert layout["runtime_role"] == "humming_packed_weight"
    assert layout["replace_policy"] == "no_structural_replace"


def test_humming_may_set_param_capture_records_runtime_tensor(monkeypatch):
    humming_pkg = types.ModuleType("humming")
    humming_layer = types.ModuleType("humming.layer")

    class FakeHummingLayerMethod:
        @classmethod
        def may_set_param(cls, layer, name, value):
            setattr(layer, name, value)

    humming_layer.HummingLayerMethod = FakeHummingLayerMethod
    monkeypatch.setitem(sys.modules, "humming", humming_pkg)
    monkeypatch.setitem(sys.modules, "humming.layer", humming_layer)

    layer = nn.Module()
    packed = torch.empty(8, dtype=torch.int32)

    with capture_humming_runtime_tensors(enabled=True):
        FakeHummingLayerMethod.may_set_param(layer, "weight", packed)

    assert layer._mx_rdma_tensors["weight"] is packed
    assert packed.mx_runtime_role == "humming_packed_weight"
    assert packed.mx_replace_policy == "no_structural_replace"


def test_vllm_humming_postload_capture_records_runtime_weight(monkeypatch):
    module_name = "vllm.model_executor.layers.quantization.humming"
    vllm_humming = types.ModuleType(module_name)

    class HummingLinearMethod:
        def process_weights_after_loading(self, layer):
            layer.weight = torch.empty(8, dtype=torch.int32)

    vllm_humming.HummingLinearMethod = HummingLinearMethod
    monkeypatch.setitem(sys.modules, module_name, vllm_humming)

    method = HummingLinearMethod()
    layer = nn.Module()

    with capture_humming_runtime_tensors(enabled=True):
        method.process_weights_after_loading(layer)

    assert layer._mx_rdma_tensors["weight"] is layer.weight
    assert layer.weight.mx_runtime_role == "humming_packed_weight"
    assert layer.weight.mx_replace_policy == "no_structural_replace"


def test_vllm_humming_postload_capture_ignores_dense_weight(monkeypatch):
    module_name = "vllm.model_executor.layers.quantization.humming"
    vllm_humming = types.ModuleType(module_name)

    class HummingLinearMethod:
        def process_weights_after_loading(self, layer):
            layer.weight = nn.Parameter(torch.empty(8, dtype=torch.bfloat16))

    vllm_humming.HummingLinearMethod = HummingLinearMethod
    monkeypatch.setitem(sys.modules, module_name, vllm_humming)

    method = HummingLinearMethod()
    layer = nn.Module()

    with capture_humming_runtime_tensors(enabled=True):
        method.process_weights_after_loading(layer)

    assert not hasattr(layer, "_mx_rdma_tensors")


def test_humming_model_capture_records_unique_direct_packed_weight():
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.empty(16, 16, dtype=torch.bfloat16))
    layer.register_buffer("runtime_weight", torch.empty(8, dtype=torch.int32))
    model = nn.Module()
    model.layer = layer

    recorded = capture_humming_runtime_tensors_from_model(model)

    assert recorded == 1
    assert layer._mx_rdma_tensors["weight"] is layer.runtime_weight


def test_humming_model_capture_does_not_treat_locks_as_packed_weight():
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.empty(16, 16, dtype=torch.bfloat16))
    layer.register_buffer("locks", torch.empty(1024, dtype=torch.int32))
    model = nn.Module()
    model.layer = layer

    recorded = capture_humming_runtime_tensors_from_model(model)

    assert recorded == 0
    assert not hasattr(layer, "_mx_rdma_tensors")


def test_humming_locks_are_skipped_from_manifest_collection():
    provider = HummingManifestProvider()

    assert provider.skip_manifest_tensor("layer.locks", "locks", "buffer")
    assert not provider.skip_manifest_tensor("layer.weight", "weight", "parameter")


def test_humming_model_capture_prefers_humming_meta_weight_name():
    class Meta:
        weight_name = "runtime_weight"

    layer = nn.Module()
    layer.weight = nn.Parameter(torch.empty(16, 16, dtype=torch.bfloat16))
    layer.runtime_weight = nn.Parameter(torch.empty(8, dtype=torch.int32), requires_grad=False)
    layer.humming_metas = {"": Meta()}
    model = nn.Module()
    model.layer = layer

    recorded = capture_humming_runtime_tensors_from_model(model)

    assert recorded == 1
    assert layer._mx_rdma_tensors["runtime_weight"] is layer.runtime_weight
    assert layer._mx_rdma_tensors["weight"] is layer.runtime_weight


def test_humming_model_capture_records_moe_meta_weights_without_locks():
    class Meta:
        def __init__(self, weight_name):
            self.weight_name = weight_name

    layer = nn.Module()
    layer.w13_weight = nn.Parameter(torch.empty(2, 3, 4, dtype=torch.int32), requires_grad=False)
    layer.w2_weight = nn.Parameter(torch.empty(2, 4, 3, dtype=torch.int32), requires_grad=False)
    layer.register_buffer("locks", torch.empty(1024, dtype=torch.int32))
    layer.humming_metas = {
        "w13": Meta("w13_weight"),
        "w2": Meta("w2_weight"),
    }
    model = nn.Module()
    model.layer = layer

    recorded = capture_humming_runtime_tensors_from_model(model)

    assert recorded == 1
    assert layer._mx_rdma_tensors["w13_weight"] is layer.w13_weight
    assert layer._mx_rdma_tensors["w2_weight"] is layer.w2_weight
    assert "locks" not in layer._mx_rdma_tensors
    assert "weight" not in layer._mx_rdma_tensors


def test_humming_model_capture_rejects_ambiguous_direct_packed_weights():
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.empty(16, 16, dtype=torch.bfloat16))
    layer.register_buffer("runtime_weight_a", torch.empty(8, dtype=torch.int32))
    layer.register_buffer("runtime_weight_b", torch.empty(8, dtype=torch.int32))
    model = nn.Module()
    model.layer = layer

    recorded = capture_humming_runtime_tensors_from_model(model)

    assert recorded == 0
    assert not hasattr(layer, "_mx_rdma_tensors")


def test_humming_runtime_mapping_weight_is_preferred():
    from modelexpress.tensor_utils import _manifest_tensor_for_module_leaf

    module = nn.Module()
    module.weight = nn.Parameter(torch.randn(538, 1152, dtype=torch.bfloat16))
    packed = torch.empty(1024, dtype=torch.int32)
    module._mx_rdma_tensors = {"weight": packed}

    tensor, runtime_role, replace_policy = _manifest_tensor_for_module_leaf(
        module,
        "weight",
        module.weight.data,
        "humming",
    )

    assert tensor.data_ptr() == packed.data_ptr()
    assert runtime_role == "humming_packed_weight"
    assert replace_policy == "no_structural_replace"


def test_global_humming_unquantized_weight_uses_structural_replace():
    from modelexpress.tensor_utils import _manifest_tensor_for_module_leaf

    module = nn.Module()
    module.weight = nn.Parameter(torch.randn(538, 1152, dtype=torch.bfloat16))

    tensor, runtime_role, replace_policy = _manifest_tensor_for_module_leaf(
        module,
        "weight",
        module.weight.data,
        "humming",
    )

    assert tensor.data_ptr() == module.weight.data.data_ptr()
    assert runtime_role == "weight"
    assert replace_policy == "structural_replace"


def test_humming_dense_weight_rejects_mismatch_when_mapping_missing():
    from modelexpress.tensor_utils import _manifest_tensor_for_module_leaf

    class FakeHummingMethod:
        pass

    FakeHummingMethod.__module__ = "vllm.model_executor.layers.quantization.humming"

    module = nn.Module()
    module.quant_method = FakeHummingMethod()
    module.weight = nn.Parameter(torch.randn(538, 1152, dtype=torch.bfloat16))

    tensor, runtime_role, replace_policy = _manifest_tensor_for_module_leaf(
        module,
        "weight",
        module.weight.data,
        "humming",
    )

    assert tensor.data_ptr() == module.weight.data.data_ptr()
    assert runtime_role == "humming_dense_weight"
    assert replace_policy == "reject_if_mismatch"


def test_collect_module_tensors_filters_to_source_manifest_names(monkeypatch):
    model = nn.Module()
    setattr(model, SOURCE_MANIFEST_TENSOR_NAMES_ATTR, {"keep.weight"})

    monkeypatch.setattr(
        "modelexpress.tensor_utils.iter_module_tensors",
        lambda module: [
            ("keep.weight", FakeCudaTensor(1), "parameter"),
            ("target.only", FakeCudaTensor(2), "buffer"),
        ],
    )

    tensors = collect_module_tensors(model)

    assert list(tensors) == ["keep.weight"]


class TestFindHiddenCudaTensors:
    def test_finds_tensor_on_plain_object(self):
        config = QuantConfig(device="cpu")
        # CPU tensors are not CUDA, so shouldn't be found
        results = _find_hidden_cuda_tensors(config, visited=set())
        assert len(results) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_finds_cuda_tensor_on_plain_object(self):
        config = QuantConfig(device="cuda")
        results = _find_hidden_cuda_tensors(config, visited=set())
        assert len(results) == 2  # a1_gscale, a2_gscale

    def test_finds_tensors_in_nested_dicts_and_lists(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        nested = NestedObj(device="cuda")
        results = _find_hidden_cuda_tensors(nested, visited=set())
        assert len(results) == 3  # w1, w2, buffers[0]

    def test_skips_non_tensor_attrs(self):
        config = QuantConfig(device="cpu")
        results = _find_hidden_cuda_tensors(config, visited=set())
        # CPU tensors are not CUDA; some_int and some_string are not tensors.
        # Nothing should be found.
        assert len(results) == 0

    def test_handles_circular_references(self):
        class Circular:
            pass

        obj = Circular()
        obj.self_ref = obj
        results = _find_hidden_cuda_tensors(obj, visited=set())
        assert len(results) == 0

    def test_respects_depth_limit(self):
        class Deep:
            pass

        # Build a chain deeper than the limit
        root = Deep()
        current = root
        for _ in range(25):
            child = Deep()
            current.child = child
            current = child
        if torch.cuda.is_available():
            current.tensor = torch.randn(4, device="cuda")

        results = _find_hidden_cuda_tensors(root, visited=set())
        # Depth 20 limit should prevent finding the deep tensor
        assert len(results) == 0


# ---------------------------------------------------------------------------
# adopt_hidden_tensors
# ---------------------------------------------------------------------------


class TestAdoptHiddenTensors:
    def test_no_hidden_tensors(self):
        model = SimpleModule()
        count = adopt_hidden_tensors(model)
        assert count == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_adopts_cuda_tensors_from_quant_method(self):
        module = ModuleWithQuant()
        module.quant_method = FakeQuant(device="cuda")
        count = adopt_hidden_tensors(module)
        assert count == 2  # a1_gscale, a2_gscale

        # Verify they're now in named_buffers
        buffer_names = {name for name, _ in module.named_buffers()}
        assert any("a1_gscale" in name for name in buffer_names)
        assert any("a2_gscale" in name for name in buffer_names)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_skips_already_registered_tensors(self):
        module = nn.Module()
        tensor = torch.randn(4, device="cuda")
        module.register_buffer("existing", tensor)

        class Holder:
            pass

        holder = Holder()
        holder.ref = tensor  # same tensor, already registered
        module.holder = holder

        count = adopt_hidden_tensors(module)
        assert count == 0  # should skip since data_ptr already registered

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_adopts_nested_tensors(self):
        module = ModuleWithNested()
        module.nested = NestedObj(device="cuda")
        count = adopt_hidden_tensors(module)
        assert count == 3  # w1, w2, buffers[0]

    def test_cpu_tensors_ignored(self):
        module = ModuleWithQuant()  # CPU tensors by default
        count = adopt_hidden_tensors(module)
        assert count == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_collect_uses_runtime_packed_weight_when_global_humming(self):
        module = nn.Module()
        module.weight = nn.Parameter(
            torch.randn(538, 1152, dtype=torch.bfloat16, device="cuda")
        )
        packed = torch.empty(1024, dtype=torch.int32, device="cuda")
        module._mx_rdma_tensors = {"weight": packed}

        tensors = collect_module_tensors(module, quantization="humming")

        assert tensors["weight"] is packed
        assert tensors["weight"].shape == torch.Size([1024])
        assert tensors["weight"].dtype == torch.int32
        assert tensors["weight"].mx_runtime_role == "humming_packed_weight"
        assert tensors["weight"].mx_replace_policy == "no_structural_replace"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_buffer_name_collisions_disambiguated(self):
        module = nn.Module()

        class A:
            def __init__(self):
                self.scale = torch.randn(4, device="cuda")

        a = A()
        b_dict = {"scale": torch.randn(4, device="cuda")}
        # Both attributes normalize to `_mx_<attr>_scale`; with "__dot__" the
        # attr prefix differs, but we also want collision handling if two
        # hidden tensors under one attr share a sanitized suffix.
        module.a = a
        module.b = b_dict
        # Second hidden tensor on `a` whose path collides after sanitization:
        # "scale.x" and "scale[x]" both normalize similarly.
        a.__dict__["inner"] = {"k": torch.randn(4, device="cuda")}

        count = adopt_hidden_tensors(module)
        assert count == 3
        buffer_ptrs = {buf.data_ptr() for _, buf in module.named_buffers()}
        assert len(buffer_ptrs) == 3  # no overwrite, all tensors survived


# ---------------------------------------------------------------------------
# capture_tensor_attrs
# ---------------------------------------------------------------------------


class TestCaptureTensorAttrs:
    def test_promotes_bare_cuda_tensor_to_buffer(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        module = nn.Module()
        tensor = torch.randn(4, device="cuda")
        with capture_tensor_attrs():
            module.my_tensor = tensor

        assert "my_tensor" in dict(module.named_buffers())

    def test_does_not_promote_parameter(self):
        module = nn.Module()
        param = nn.Parameter(torch.randn(4))
        with capture_tensor_attrs():
            module.my_param = param

        assert "my_param" in dict(module.named_parameters())
        assert "my_param" not in dict(module.named_buffers())

    def test_does_not_promote_cpu_tensor(self):
        module = nn.Module()
        with capture_tensor_attrs():
            module.cpu_tensor = torch.randn(4)

        assert "cpu_tensor" not in dict(module.named_buffers())

    def test_restores_setattr_after_exit(self):
        original = nn.Module.__setattr__
        with capture_tensor_attrs():
            assert nn.Module.__setattr__ is not original
        assert nn.Module.__setattr__ is original


# ---------------------------------------------------------------------------
# safe_checksum
# ---------------------------------------------------------------------------


class TestSafeChecksum:
    def test_deterministic(self):
        t = torch.ones(10)
        assert safe_checksum(t) == safe_checksum(t)

    def test_different_tensors_different_checksums(self):
        t1 = torch.ones(10)
        t2 = torch.zeros(10)
        assert safe_checksum(t1) != safe_checksum(t2)

    def test_scalar_tensor(self):
        t = torch.tensor(3.14)
        result = safe_checksum(t)
        assert not result.startswith("err:")

    def test_returns_8_hex_chars(self):
        t = torch.randn(100)
        result = safe_checksum(t)
        assert len(result) == 8
        int(result, 16)  # should be valid hex

    def test_permutation_sensitive(self):
        t1 = torch.tensor([1, 255], dtype=torch.uint8)
        t2 = torch.tensor([255, 1], dtype=torch.uint8)
        assert safe_checksum(t1) != safe_checksum(t2)

    def test_compensating_byte_delta_detected(self):
        t1 = torch.tensor([10, 20, 30], dtype=torch.uint8)
        t2 = torch.tensor([11, 19, 30], dtype=torch.uint8)
        assert safe_checksum(t1) != safe_checksum(t2)


# ---------------------------------------------------------------------------
# storage_view
# ---------------------------------------------------------------------------


class TestStorageView:
    def test_returns_contiguous_uint8(self):
        t = torch.randn(4, 4)
        sv = storage_view(t)
        assert sv.dtype == torch.uint8
        assert sv.is_contiguous()

    def test_covers_full_storage(self):
        t = torch.randn(4, 4)  # 16 floats * 4 bytes = 64 bytes
        sv = storage_view(t)
        assert sv.numel() == t.numel() * t.element_size()

    def test_non_contiguous_tensor_gets_full_storage(self):
        base = torch.randn(4, 4)
        view = base.T  # non-contiguous
        sv = storage_view(view)
        assert sv.numel() == base.numel() * base.element_size()


# ---------------------------------------------------------------------------
# collect_module_tensors
# ---------------------------------------------------------------------------


class TestCollectModuleTensors:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_collects_parameters(self):
        model = nn.Linear(4, 4, device="cuda")
        tensors = collect_module_tensors(model)
        assert "weight" in tensors
        assert "bias" in tensors

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_deduplicates_tied_weights(self):
        model = nn.Module()
        shared = nn.Parameter(torch.randn(4, 4, device="cuda"))
        model.register_parameter("a", shared)
        model.register_parameter("b", shared)
        tensors = collect_module_tensors(model)
        # Only one should be collected (same data_ptr)
        assert len(tensors) == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_non_contiguous_registered_as_storage(self):
        model = nn.Module()
        base = torch.randn(4, 4, device="cuda")
        view = base.T  # non-contiguous
        model.register_buffer("nc_view", view, persistent=False)
        tensors = collect_module_tensors(model)
        assert "nc_view.__storage" in tensors
