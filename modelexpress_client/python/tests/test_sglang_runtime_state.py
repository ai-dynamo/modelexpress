# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from modelexpress.engines.sglang.runtime_state import (
    finalize_runtime_state,
    prepare_runtime_tensors,
)
from modelexpress.types import ManifestMismatchError, TensorDescriptor


def _model():
    model = torch.nn.Module()
    model.proj = torch.nn.Module()
    layer = model.proj
    layer.weight = torch.nn.Parameter(torch.zeros(256, 256), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(torch.full((8, 8), -1.0), requires_grad=False)
    layer.block_fp8_mxfp8_ready = False

    def allocate(layer, *, scale_u8):
        layer.weight_scale_inv_swizzled = torch.nn.Parameter(
            torch.zeros(scale_u8.numel(), dtype=torch.uint8),
            requires_grad=False,
        )

    layer.quant_method = SimpleNamespace(
        block_fp8_as_mxfp8=True,
        weight_block_size=[32, 32],
        mxfp8_dense_backend=SimpleNamespace(
            is_flashinfer_cutedsl=lambda: True,
            is_flashinfer_cutlass=lambda: False,
        ),
        _process_mxfp8_linear_weight_scale=Mock(side_effect=allocate),
    )
    return model


def _manifest(derived=True):
    tensors = [
        TensorDescriptor(
            name="proj.weight", addr=1, size=256 * 256 * 4, device_id=0, dtype="torch.float32"
        )
    ]
    if derived:
        tensors.append(
            TensorDescriptor(
                name="proj.weight_scale_inv_swizzled",
                addr=2,
                size=2048,
                device_id=0,
                dtype="torch.uint8",
            )
        )
    return tensors


@pytest.fixture
def encode(monkeypatch):
    import sys

    encoder = Mock(return_value=torch.zeros(256, 8, dtype=torch.uint8))
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.quantization.fp8_utils",
        SimpleNamespace(block_fp8_scale_to_mxfp8_e8m0=encoder),
    )
    return encoder


def test_manifest_allocates_only_derived_storage_without_priming():
    model = _model()
    original = model.proj.weight_scale_inv.clone()
    weight_ptr = model.proj.weight.data_ptr()
    prepare_runtime_tensors(model, _manifest())
    assert torch.equal(original, model.proj.weight_scale_inv)
    assert model.proj.weight.data_ptr() == weight_ptr
    assert not model.proj.block_fp8_mxfp8_ready
    assert model.proj.weight_scale_inv_swizzled.numel() == 2048


def test_dispatch_validated_after_transfer_without_reallocating(encode):
    model = _model()
    prepare_runtime_tensors(model, _manifest())
    buf = model.proj.weight_scale_inv_swizzled
    buf.fill_(127)
    finalize_runtime_state(model)
    assert model.proj.block_fp8_mxfp8_ready
    assert model.proj.weight_scale_inv_swizzled is buf
    assert torch.all(buf == 127)
    model.proj.quant_method._process_mxfp8_linear_weight_scale.assert_called_once()
    encode.assert_called_once()


def test_ineligible_donor_does_not_allocate_or_enable_mxfp8(encode):
    model = _model()
    prepare_runtime_tensors(model, _manifest(derived=False))
    encode.side_effect = ValueError("non power of two")
    finalize_runtime_state(model)
    assert not model.proj.block_fp8_mxfp8_ready
    assert not hasattr(model.proj, "weight_scale_inv_swizzled")
    model.proj.quant_method._process_mxfp8_linear_weight_scale.assert_not_called()


@pytest.mark.parametrize("derived", [False, True])
def test_received_scales_must_agree_with_manifest(encode, derived):
    model = _model()
    prepare_runtime_tensors(model, _manifest(derived))
    if derived:
        encode.side_effect = ValueError("invalid scale")
    with pytest.raises(ManifestMismatchError, match="disagree"):
        finalize_runtime_state(model)


@pytest.mark.parametrize("field,value", [("dtype", "torch.float32"), ("dtype", ""), ("size", 1)])
def test_malformed_derived_descriptor_rejected(field, value):
    model = _model()
    source = _manifest()
    setattr(source[-1], field, value)
    with pytest.raises(ManifestMismatchError, match="descriptor"):
        prepare_runtime_tensors(model, source)


def test_duplicate_manifest_rejected():
    source = _manifest()
    with pytest.raises(ManifestMismatchError, match="Duplicate"):
        prepare_runtime_tensors(_model(), source + source)


def test_unsupported_reshuffling_backend_rejected():
    model = _model()
    model.proj.quant_method.mxfp8_dense_backend.is_flashinfer_cutedsl = lambda: False
    with pytest.raises(ManifestMismatchError, match="Unsupported"):
        prepare_runtime_tensors(model, _manifest())


def test_plain_weight_reader_cannot_accept_swizzled_layout():
    model = _model()
    model.proj.keep_plain_weight_layout = True
    with pytest.raises(ManifestMismatchError, match="Incompatible"):
        prepare_runtime_tensors(model, _manifest())


def test_shared_draft_layer_is_not_prepared():
    model = _model()
    prepare_runtime_tensors(model, [])
    assert not hasattr(model.proj, "_mx_block_fp8_manifest_prepared")


def test_kv_host_mirrors_refresh_without_renormalizing_device_values():
    model = torch.nn.Module()
    model.k_scale = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
    model.v_scale = torch.nn.Parameter(torch.tensor(4.0), requires_grad=False)
    model.k_scale_float = model.v_scale_float = 1.0
    ptrs = [model.k_scale.data_ptr(), model.v_scale.data_ptr()]
    finalize_runtime_state(model)
    assert (model.k_scale_float, model.v_scale_float) == (2.0, 4.0)
    assert (model.k_scale.item(), model.v_scale.item()) == (2.0, 4.0)
    assert ptrs == [model.k_scale.data_ptr(), model.v_scale.data_ptr()]


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_kv_host_scale_rejected(value):
    model = torch.nn.Module()
    model.k_scale_float = 1.0
    model.k_scale = torch.tensor(value)
    with pytest.raises(ManifestMismatchError, match="KV scale"):
        finalize_runtime_state(model)


def test_unquantized_kv_scales_are_absent():
    model = torch.nn.Module()
    model.k_scale = model.v_scale = None
    model.k_scale_float = model.v_scale_float = None
    finalize_runtime_state(model)
    assert model.k_scale_float is None
    assert model.v_scale_float is None


@pytest.mark.parametrize("scale", [1.0, torch.ones(2)])
def test_malformed_kv_scale_rejected(scale):
    model = torch.nn.Module()
    model.k_scale_float = None
    model.k_scale = scale
    with pytest.raises(ManifestMismatchError, match="KV scale"):
        finalize_runtime_state(model)


@torch.no_grad()
def test_transfer_engine_published_manifest_receives_and_finalizes_mxfp8(monkeypatch, encode):
    from modelexpress import p2p_pb2
    from modelexpress.engines.sglang.loader import MxModelLoader
    from modelexpress.metadata.payload import worker_tensor_descriptors

    donor = _model()
    prepare_runtime_tensors(donor, _manifest())
    tensors = dict(donor.named_parameters())
    publisher = Mock()
    monkeypatch.setattr("modelexpress.engines.sglang.loader.PublisherThread", publisher)
    ctx = SimpleNamespace(
        tensors=tensors,
        device_id=0,
        worker_rank=0,
        global_rank=0,
        worker_id="donor",
        identity=p2p_pb2.SourceIdentity(model_name="test"),
        accelerator_backend=SimpleNamespace(name="cuda"),
        mx_client=Mock(),
    )
    loader = MxModelLoader(SimpleNamespace(modelexpress_transport="transfer_engine"))
    weight_info = {
        name: (tensor.data_ptr(), tensor.numel(), tensor.element_size())
        for name, tensor in tensors.items()
    }
    assert loader._publish_transfer_engine_source(
        ctx=ctx, session_id="donor-session", weight_info=weight_info
    )
    publisher.call_args.kwargs["publish_fn"]()
    worker = ctx.mx_client.publish_metadata.call_args.args[1]
    manifest = list(worker_tensor_descriptors(worker))
    assert {t.name: t.dtype for t in manifest} == {
        name: str(tensor.dtype) for name, tensor in tensors.items()
    }
    receiver = _model()
    prepare_runtime_tensors(receiver, manifest)
    assert (
        receiver.proj.weight_scale_inv_swizzled.shape == donor.proj.weight_scale_inv_swizzled.shape
    )
    assert receiver.proj.weight_scale_inv_swizzled.dtype == torch.uint8

    targets = dict(receiver.named_parameters())
    for tensor in tensors.values():
        tensor.fill_(1)
    source_by_addr = {tensor.data_ptr(): tensor for tensor in tensors.values()}
    target_by_addr = {tensor.data_ptr(): tensor for tensor in targets.values()}

    def copy_tensors(session_id, target_ptrs, source_ptrs, sizes):
        assert session_id == "donor-session"
        for target_ptr, source_ptr, size in zip(target_ptrs, source_ptrs, sizes, strict=True):
            source = source_by_addr[source_ptr]
            assert size == source.numel() * source.element_size()
            target_by_addr[target_ptr].copy_(source)
        return 0

    transfer_engine = Mock()
    transfer_engine.batch_transfer_sync_read.side_effect = copy_tensors
    loader._receive_via_transfer_engine(targets, transfer_engine, worker, ctx)
    finalize_runtime_state(receiver)
    assert receiver.proj.block_fp8_mxfp8_ready
    for name, tensor in tensors.items():
        torch.testing.assert_close(targets[name], tensor)
    transfer_engine.batch_transfer_sync_read.assert_called_once()
    encode.assert_called_once()
