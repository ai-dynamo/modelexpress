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
    StagedVllmCheckpoint,
    _contract,
    _Copy,
    _projection_aliases,
    _resolve_bindings,
    _stage_mla,
    _stage_weight,
    _validate_aliases,
    prepare_partial_checkpoint,
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
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_mla_staging_preserves_host_ranges(
    monkeypatch, device, after_full_reload, dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required to verify mixed CPU/CUDA scalar placement")
    import vllm.config
    from vllm.model_executor.layers.attention.attention import set_default_quant_scales
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.model_loader.reload import layerwise

    monkeypatch.setattr(vllm.config, "set_current_vllm_config", lambda _: nullcontext())
    projection = nn.Linear(2, 4, bias=False, dtype=dtype, device=device)
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
    attention.process_weights_after_loading(dtype)
    model = nn.Module()
    model.attention = attention
    layerwise.record_metadata_for_reloading(model)
    if after_full_reload:
        installer = _VllmInstaller(
            model=model,
            vllm_config=None,
            model_config=SimpleNamespace(dtype=dtype),
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
    shadow = nn.Linear(2, 4, bias=False, dtype=dtype, device=device)
    shadow.quant_method = None
    with torch.no_grad():
        shadow.weight.fill_(3)
    with torch.device(device):
        copies = _stage_mla(attention, shadow, MLAAttention)
        copies.append(_Copy.stage(projection, "weight", shadow.weight))
    *_, outer_type, wrapper_type = _contract()
    outer = outer_type.__new__(outer_type)
    nn.Module.__init__(outer)
    wrapper = wrapper_type.__new__(wrapper_type)
    nn.Module.__init__(wrapper)
    outer.kv_b_proj = wrapper.kv_b_proj = projection
    wrapper.mla_attn = attention
    outer.mla_attn = wrapper
    del model.attention
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    model.language_model.model.layers = nn.ModuleList([nn.Module()])
    model.language_model.model.layers[0].self_attn = outer
    paths = _projection_aliases(
        model,
        "language_model.model.layers.0.self_attn",
        projection,
        attention,
        outer_type,
        wrapper_type,
    )
    _validate_aliases(model, copies, paths)
    StagedVllmCheckpoint(tuple(copies)).install()
    assert attention.W_UV is derived[0] and attention.W_UK_T is derived[1]
    assert torch.equal(attention.W_UV, torch.full_like(attention.W_UV, 3))
    assert torch.equal(attention.W_UK_T, torch.full_like(attention.W_UK_T, 3))
    for name, original in originals.items():
        assert getattr(attention, name) is original
        assert torch.equal(original, values[name])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_non_kimi_mapped_projection(dtype, device, tmp_path):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for complete checkpoint preparation")
    from vllm.model_executor.layers.attention.attention import set_default_quant_scales

    contract = _contract()
    _, _, linear, method, mla, outer_type, wrapper_type = contract
    model = nn.Module()
    model.blocks = nn.ModuleList()
    mapping = {}
    for index in range(2):
        projection = linear.__new__(linear)
        nn.Module.__init__(projection)
        projection.weight = nn.Parameter(
            torch.zeros(4, 2, dtype=dtype, device=device), requires_grad=False
        )
        projection.weight.output_dim = 0
        projection.quant_method = method()
        projection.tp_rank, projection.tp_size = 1, 2
        attention = mla.__new__(mla)
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
        attention.process_weights_after_loading(dtype)
        outer = outer_type.__new__(outer_type)
        nn.Module.__init__(outer)
        wrapper = wrapper_type.__new__(wrapper_type)
        nn.Module.__init__(wrapper)
        outer.kv_b_proj = wrapper.kv_b_proj = projection
        wrapper.mla_attn = attention
        outer.mla_attn = wrapper
        model.blocks.append(outer)
        mapping[f"checkpoint.block.{index}.projection"] = f"blocks.{index}.kv_b_proj"
    bindings = _resolve_bindings(model, mapping, *contract)
    source = torch.arange(16, dtype=dtype).view(8, 2)
    copies, aliases = [], {}
    for binding in bindings.values():
        shadow = _stage_weight(binding.layer, source, linear.weight_loader)
        copies.extend(_stage_mla(binding.attention, shadow, mla))
        copies.append(_Copy.stage(binding.layer, "weight", shadow.weight))
        aliases.update(binding.aliases)
    _validate_aliases(model, copies, aliases)
    if device == "cuda":
        from modelexpress_rl.inference.checkpoint_selection import CheckpointChanges
        from modelexpress_rl.inference.receiver import PreparedCheckpoint
        from safetensors.torch import save_file

        save_file(
            {name: source.clone() for name in mapping}, tmp_path / "model.safetensors"
        )
        config = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=True, dtype=dtype),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=1, enable_expert_parallel=False
            ),
            cache_config=SimpleNamespace(cpu_offload_gb=0, cache_dtype="auto"),
        )
        staged = prepare_partial_checkpoint(
            model,
            config,
            PreparedCheckpoint(
                "target",
                tmp_path,
                {},
                CheckpointChanges("base", "target", frozenset(mapping)),
            ),
            serving_version="base",
            tensor_mapping=mapping,
        )
        assert staged is not None
        staged.install()
    else:
        StagedVllmCheckpoint(tuple(copies)).install()
    for block in model.blocks:
        assert torch.equal(block.kv_b_proj.weight, source[4:].to(device))
        key, value = source[4:].T.view(2, 2, 2).split([1, 1], dim=-1)
        assert torch.equal(
            block.mla_attn.mla_attn.W_UV, value.transpose(0, 1).to(device)
        )
        assert torch.equal(
            block.mla_attn.mla_attn.W_UK_T, key.permute(1, 2, 0).to(device)
        )


def test_default_mapping_covers_all_audited_layers_and_declines_unknown_model():
    from modelexpress_rl.inference.engines.vllm.checkpoint_bindings import (
        default_checkpoint_mapping,
    )
    from modelexpress_rl.inference.engines.vllm.partial_checkpoint import _Unsupported
    from vllm.model_executor.models.kimi_k25 import KimiK25ForConditionalGeneration

    *_, outer_type, _ = _contract()
    model = KimiK25ForConditionalGeneration.__new__(KimiK25ForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    model.language_model.model.layers = nn.ModuleList()
    for _ in range(3):
        block = nn.Module()
        block.self_attn = outer_type.__new__(outer_type)
        nn.Module.__init__(block.self_attn)
        model.language_model.model.layers.append(block)
    mapping = default_checkpoint_mapping(model)
    assert len(mapping) == 4
    assert (
        mapping["language_model.model.layers.2.self_attn.kv_b_proj.weight"]
        == "language_model.model.layers.2.self_attn.kv_b_proj"
    )
    with pytest.raises(_Unsupported, match="explicit checkpoint tensor mapping"):
        default_checkpoint_mapping(nn.Module())
