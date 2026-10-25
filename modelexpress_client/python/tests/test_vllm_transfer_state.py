# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime state must stay local even when another catalog name aliases it."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modelexpress.engines.vllm.adapter import VllmAdapter
from modelexpress.load_strategy.context import LoadResult
from modelexpress.tensor_utils import collect_module_tensors


def _model(*, registered=False, subclass=False, module_name=None):
    cls = type(
        "FlashInferMLAImpl",
        (nn.Module,) if registered else (object,),
        {"__module__": module_name or "vllm.v1.attention.backends.mla.flashinfer_mla"},
    )
    if subclass:
        cls = type("ApplicationImpl", (cls,), {"__module__": "application"})
    impl = cls()
    counter = torch.arange(16, dtype=torch.uint8)
    if registered:
        impl.register_buffer("_multi_ctas_kv_counter_buffer", counter, persistent=False)
        impl.register_buffer("scale_recip", torch.tensor([0.5]))
    else:
        impl._multi_ctas_kv_counter_buffer = counter
        impl.scale_recip = torch.tensor([0.5])
    model = nn.Module()
    model.weight = nn.Parameter(torch.ones(8))
    model.impl = impl
    return model, counter


def _discover(model, backend):
    adapter = VllmAdapter.__new__(VllmAdapter)
    adapter.accelerator_backend = backend
    adapter.vllm_config = SimpleNamespace(speculative_config=None)
    adapter.model_config = SimpleNamespace()
    return adapter.discover_tensors(LoadResult(value=model, model=model))


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("subclass", [False, True])
def test_counter_and_aliases_stay_local(registered, subclass, mock_accelerator_backend_cls):
    backend = mock_accelerator_backend_cls(torch_device_type="cpu")
    model, counter = _model(registered=registered, subclass=subclass)
    before = counter.clone()
    model.register_buffer("counter_alias", counter)
    model.register_buffer("counter_slice", counter[4:8])
    model.register_buffer("counter_transpose", counter.view(4, 4).T)
    for _ in range(2):
        catalog = _discover(model, backend)
        assert len(catalog) == 2
        assert catalog["weight"].data_ptr() == model.weight.data_ptr()
        scale = next(t for n, t in catalog.items() if "scale_recip" in n)
        assert scale.item() == 0.5
        assert torch.equal(counter, before)


def test_same_field_on_unrecognized_type_is_not_silently_dropped(mock_accelerator_backend_cls):
    backend = mock_accelerator_backend_cls(torch_device_type="cpu")
    model, counter = _model(module_name="application.quantization")
    catalog = _discover(model, backend)
    assert any(t.data_ptr() == counter.data_ptr() for t in catalog.values())
    assert len(catalog) == 3


def test_no_instance_counter_is_a_noop(mock_accelerator_backend_cls):
    backend = mock_accelerator_backend_cls(torch_device_type="cpu")
    model, _ = _model()
    del model.impl._multi_ctas_kv_counter_buffer
    catalog = _discover(model, backend)
    assert len(catalog) == 2


def test_arbitrary_registered_buffers_are_retained(mock_accelerator_backend_cls):
    backend = mock_accelerator_backend_cls(torch_device_type="cpu")
    model, _ = _model()
    model.register_buffer("runtime_counter", torch.ones(3))
    catalog = _discover(model, backend)
    assert "runtime_counter" in catalog


@pytest.mark.parametrize("kind", ["parameter", "partial_buffer", "widened_storage"])
def test_conflicting_transfer_is_rejected(kind, mock_accelerator_backend_cls):
    backend = mock_accelerator_backend_cls(torch_device_type="cpu")
    model = nn.Module()
    allocation = torch.arange(32, dtype=torch.float32)
    counter = allocation[16:]
    if kind == "parameter":
        model.weight = nn.Parameter(counter)
    elif kind == "partial_buffer":
        model.register_buffer("derived", allocation[8:24])
    else:
        model.weight = nn.Parameter(allocation[:16].view(4, 4).T)
    before = allocation.clone()
    with pytest.raises(ValueError, match="overlaps local runtime state"):
        collect_module_tensors(model, backend, preserve_local=[counter])
    assert torch.equal(allocation, before)


def test_disjoint_contiguous_weight_sharing_storage_is_retained(mock_accelerator_backend_cls):
    backend = mock_accelerator_backend_cls(torch_device_type="cpu")
    model = nn.Module()
    allocation = torch.arange(32, dtype=torch.float32)
    model.weight = nn.Parameter(allocation[:16])
    model.register_buffer("local", allocation[16:])
    catalog = collect_module_tensors(model, backend, preserve_local=[model.local])
    assert list(catalog) == ["weight"]
    assert torch.equal(catalog["weight"], allocation[:16])


def test_noncontiguous_local_definition_is_rejected(mock_accelerator_backend_cls):
    backend = mock_accelerator_backend_cls(torch_device_type="cpu")
    with pytest.raises(ValueError, match="must be contiguous"):
        collect_module_tensors(nn.Module(), backend, preserve_local=[torch.ones(4, 4).T])


def test_empty_local_tensor_does_not_hide_empty_weights(mock_accelerator_backend_cls):
    backend = mock_accelerator_backend_cls(torch_device_type="cpu")
    model = nn.Module()
    model.weight = nn.Parameter(torch.empty(0))
    catalog = collect_module_tensors(model, backend, preserve_local=[torch.empty(0)])
    assert list(catalog) == ["weight"]
