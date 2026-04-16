# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tensor_utils: hidden tensor adoption, capture_tensor_attrs, checksums."""

import torch
import torch.nn as nn
import pytest

from modelexpress.tensor_utils import (
    _find_hidden_cuda_tensors,
    adopt_hidden_tensors,
    capture_tensor_attrs,
    collect_module_tensors,
    safe_checksum,
    storage_view,
)


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


# ---------------------------------------------------------------------------
# _find_hidden_cuda_tensors
# ---------------------------------------------------------------------------


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
