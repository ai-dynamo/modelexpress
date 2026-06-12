# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tensor_utils: hidden tensor adoption, capture_tensor_attrs, checksums."""

import torch
import torch.nn as nn

from modelexpress.tensor_utils import (
    _find_hidden_accel_tensors,
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
# _find_hidden_accel_tensors
# ---------------------------------------------------------------------------


class TestFindHiddenAccelTensors:
    def test_default_cuda_backend_ignores_cpu_tensor(self):
        config = QuantConfig(device="cpu")
        # Default CUDA backend: CPU tensors must be ignored.
        results = _find_hidden_accel_tensors(config, visited=set())
        assert len(results) == 0

    def test_finds_backend_tensor_on_plain_object(self, mock_accelerator_backend_cls):
        backend = mock_accelerator_backend_cls(torch_device_type="cpu")
        config = QuantConfig(device="cpu")
        results = _find_hidden_accel_tensors(config, visited=set(), accelerator_backend=backend)
        assert len(results) == 2  # a1_gscale, a2_gscale

    def test_finds_tensors_in_nested_dicts_and_lists(self, mock_accelerator_backend_cls):
        backend = mock_accelerator_backend_cls(torch_device_type="cpu")
        nested = NestedObj(device="cpu")
        results = _find_hidden_accel_tensors(nested, visited=set(), accelerator_backend=backend)
        assert len(results) == 3  # w1, w2, buffers[0]

    def test_skips_non_tensor_attrs(self):
        config = QuantConfig(device="cpu")
        # Default CUDA backend: CPU tensors must be ignored.
        results = _find_hidden_accel_tensors(config, visited=set())
        # CPU tensors are not CUDA by default; ints and strings are not tensors.
        # Nothing should be found.
        assert len(results) == 0

    def test_handles_circular_references(self):
        class Circular:
            pass

        obj = Circular()
        obj.self_ref = obj
        # Default CUDA backend: this no-op CPU object graph must stay ignored.
        results = _find_hidden_accel_tensors(obj, visited=set())
        assert len(results) == 0

    def test_respects_depth_limit(self, mock_accelerator_backend_cls):
        backend = mock_accelerator_backend_cls(torch_device_type="cpu")

        class Deep:
            pass

        # Build a chain deeper than the limit
        root = Deep()
        current = root
        for _ in range(25):
            child = Deep()
            current.child = child
            current = child
        current.tensor = torch.randn(4)

        results = _find_hidden_accel_tensors(root, visited=set(), accelerator_backend=backend)
        # Depth 20 limit should prevent finding the deep tensor
        assert len(results) == 0


# ---------------------------------------------------------------------------
# adopt_hidden_tensors
# ---------------------------------------------------------------------------


class TestAdoptHiddenTensors:
    def test_no_hidden_tensors(self):
        model = SimpleModule()
        # Default CUDA backend: CPU tensors must be ignored.
        count = adopt_hidden_tensors(model)
        assert count == 0

    def test_adopts_backend_tensors_from_quant_method(self, mock_accelerator_backend_cls):
        backend = mock_accelerator_backend_cls(torch_device_type="cpu")
        module = ModuleWithQuant()
        count = adopt_hidden_tensors(module, backend)
        assert count == 2  # a1_gscale, a2_gscale

        # Verify they're now in named_buffers
        buffer_names = {name for name, _ in module.named_buffers()}
        assert any("a1_gscale" in name for name in buffer_names)
        assert any("a2_gscale" in name for name in buffer_names)

    def test_skips_already_registered_tensors(self, mock_accelerator_backend_cls):
        backend = mock_accelerator_backend_cls(torch_device_type="cpu")
        module = nn.Module()
        tensor = torch.randn(4)
        module.register_buffer("existing", tensor)

        class Holder:
            pass

        holder = Holder()
        holder.ref = tensor  # same tensor, already registered
        module.holder = holder

        # The CPU mock marks holder.ref as accelerator-owned; zero means dedup skipped it.
        count = adopt_hidden_tensors(module, backend)
        assert count == 0

    def test_adopts_nested_tensors(self, mock_accelerator_backend_cls):
        backend = mock_accelerator_backend_cls(torch_device_type="cpu")
        module = ModuleWithNested()
        count = adopt_hidden_tensors(module, backend)
        assert count == 3  # w1, w2, buffers[0]

    def test_cpu_tensors_ignored(self):
        module = ModuleWithQuant()  # CPU tensors by default
        # Default CUDA backend: CPU tensors must be ignored.
        count = adopt_hidden_tensors(module)
        assert count == 0

    def test_buffer_name_collisions_disambiguated(self, mock_accelerator_backend_cls):
        backend = mock_accelerator_backend_cls(torch_device_type="cpu")
        module = nn.Module()

        class A:
            def __init__(self):
                self.scale = torch.randn(4)

        a = A()
        b_dict = {"scale": torch.randn(4)}
        # Both attributes normalize to `_mx_<attr>_scale`; with "__dot__" the
        # attr prefix differs, but we also want collision handling if two
        # hidden tensors under one attr share a sanitized suffix.
        module.a = a
        module.b = b_dict
        # Second hidden tensor on `a` whose path collides after sanitization:
        # "scale.x" and "scale[x]" both normalize similarly.
        a.__dict__["inner"] = {"k": torch.randn(4)}

        count = adopt_hidden_tensors(module, backend)
        assert count == 3
        buffer_ptrs = {buf.data_ptr() for _, buf in module.named_buffers()}
        assert len(buffer_ptrs) == 3  # no overwrite, all tensors survived


# ---------------------------------------------------------------------------
# capture_tensor_attrs
# ---------------------------------------------------------------------------


class TestCaptureTensorAttrs:
    def test_promotes_bare_backend_tensor_to_buffer(self, mock_accelerator_backend_cls):
        backend = mock_accelerator_backend_cls(torch_device_type="cpu")
        module = nn.Module()
        tensor = torch.randn(4)
        with capture_tensor_attrs(backend):
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
    def test_collects_backend_parameters(self, mock_accelerator_backend_cls):
        backend = mock_accelerator_backend_cls(torch_device_type="cpu")
        model = nn.Linear(4, 4)
        tensors = collect_module_tensors(model, backend)
        assert "weight" in tensors
        assert "bias" in tensors

    def test_deduplicates_tied_weights(self, mock_accelerator_backend_cls):
        backend = mock_accelerator_backend_cls(torch_device_type="cpu")
        model = nn.Module()
        shared = nn.Parameter(torch.randn(4, 4))
        model.register_parameter("a", shared)
        model.register_parameter("b", shared)
        tensors = collect_module_tensors(model, backend)
        # Only one should be collected (same data_ptr)
        assert len(tensors) == 1

    def test_non_contiguous_registered_as_storage(self, mock_accelerator_backend_cls):
        backend = mock_accelerator_backend_cls(torch_device_type="cpu")
        model = nn.Module()
        base = torch.randn(4, 4)
        view = base.T  # non-contiguous
        model.register_buffer("nc_view", view, persistent=False)
        tensors = collect_module_tensors(model, backend)
        assert "nc_view.__storage" in tensors
