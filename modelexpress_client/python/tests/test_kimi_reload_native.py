# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regressions against pinned native vLLM implementations, without a full model."""

import hashlib
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

import pytest
import torch
from torch import nn

try:
    if version("vllm") != "0.19.0":
        pytest.skip("requires pinned vLLM 0.19.0", allow_module_level=True)
except PackageNotFoundError:
    pytest.skip("requires pinned vLLM 0.19.0", allow_module_level=True)

from modelexpress_rl.inference.engines.vllm.installer import _VllmInstaller
from modelexpress_rl.inference.engines.vllm.partial_checkpoint import (
    StagedBf16Checkpoint,
    _Copy,
    _stage_mla,
)


@pytest.mark.parametrize("fail_load", [False, True])
def test_native_kimi_generated_buffer_survives_repeated_reload(monkeypatch, fail_load):
    import vllm.config
    from vllm.model_executor.model_loader.reload import layerwise, meta
    from vllm.model_executor.models.kimi_k25_vit import (
        Learnable2DInterpPosEmbDivided_fixed,
    )

    monkeypatch.setattr(vllm.config, "set_current_vllm_config", lambda _: nullcontext())
    model = Learnable2DInterpPosEmbDivided_fixed(2, 2, 4, 1152).to(torch.bfloat16)
    layerwise.record_metadata_for_reloading(model)
    original = model.time_weight
    expected = original.clone()
    assert hashlib.sha256(expected.view(torch.uint8).numpy().tobytes()).hexdigest() == (
        "8d5ac7a7e8e62b89754fd27cfae6228724f949a82b4c18a0be2157a534626ee0"
    )
    metadata = layerwise.LAYERWISE_INFO[model].restore_metadata
    materialize = meta.materialize_meta_tensor
    generated_materializations = []

    def poison_uninitialized_buffer(tensor):
        result = materialize(tensor)
        if tensor.shape == original.shape:
            generated_materializations.append(tensor)
            result.fill_(9)
        return result

    monkeypatch.setattr(meta, "materialize_meta_tensor", poison_uninitialized_buffer)
    installer = _VllmInstaller(
        model=model,
        vllm_config=None,
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        device=torch.device("cpu"),
    )
    weight_address = model.weight.data_ptr()

    def load():
        model.weight.weight_loader(
            model.weight, torch.full_like(model.weight, 3, device="cpu")
        )
        if fail_load:
            raise RuntimeError("failed loader")

    for _ in range(2):
        if fail_load:
            with pytest.raises(RuntimeError, match="failed loader"):
                installer._reload(load)
        else:
            installer._reload(load)
            assert torch.equal(model.weight, torch.full_like(model.weight, 3))
            assert model.weight.data_ptr() == weight_address
        assert generated_materializations == []
        assert model.time_weight is original
        assert torch.equal(model.time_weight, expected)
        assert "time_weight" not in model.state_dict()
        assert layerwise.LAYERWISE_INFO[model].restore_metadata is metadata


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("after_full_reload", [False, True])
def test_native_mla_staging_preserves_host_ranges(
    monkeypatch, device, after_full_reload
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required to verify mixed CPU/CUDA scalar placement")
    import vllm.config
    from vllm.model_executor.layers.attention.attention import set_default_quant_scales
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.model_loader.reload import layerwise

    monkeypatch.setattr(vllm.config, "set_current_vllm_config", lambda _: nullcontext())
    projection = nn.Linear(2, 4, bias=False, dtype=torch.bfloat16, device=device)
    projection.quant_method = None
    attention = MLAAttention.__new__(MLAAttention)
    nn.Module.__init__(attention)
    attention.kv_b_proj = projection
    attention.quant_config = None
    attention.num_heads = attention.kv_lora_rank = 2
    attention.qk_nope_head_dim = attention.v_head_dim = 1
    attention.is_aiter_triton_fp4_bmm_enabled = False
    attention.is_aiter_triton_fp8_bmm_enabled = False
    attention.calculate_kv_scales = False
    set_default_quant_scales(attention, register_buffer=True)
    attention.to(device)
    attention.process_weights_after_loading(torch.bfloat16)
    model = nn.Module()
    model.attention = attention
    layerwise.record_metadata_for_reloading(model)
    if after_full_reload:
        installer = _VllmInstaller(
            model=model,
            vllm_config=None,
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            device=torch.device(device),
        )

        def load():
            projection.weight.weight_loader(
                projection.weight, torch.full_like(projection.weight, 2, device=device)
            )

        installer._reload(load)
    originals = {
        name: getattr(attention, name)
        for name in ("q_range", "k_range", "v_range", "_q_scale")
    }
    values = {name: tensor.clone() for name, tensor in originals.items()}
    derived = [attention.W_UV, attention.W_UK_T]
    shadow = nn.Linear(2, 4, bias=False, dtype=torch.bfloat16, device=device)
    shadow.quant_method = None
    with torch.no_grad():
        shadow.weight.fill_(3)
    with torch.device(device):
        copies = _stage_mla(attention, shadow, MLAAttention)
        copies.append(_Copy.stage(projection, "weight", shadow.weight))
    StagedBf16Checkpoint(tuple(copies)).install()
    assert attention.W_UV is derived[0] and attention.W_UK_T is derived[1]
    assert torch.equal(attention.W_UV, torch.full_like(attention.W_UV, 3))
    assert torch.equal(attention.W_UK_T, torch.full_like(attention.W_UK_T, 3))
    for name, original in originals.items():
        assert getattr(attention, name) is original
        assert torch.equal(original, values[name])
