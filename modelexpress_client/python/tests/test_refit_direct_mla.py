# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from modelexpress.refit.reshard.types import IncompleteRefit
from modelexpress_rl.inference.engines.vllm.direct_mla import _prepare_mla_refresh


def fixture(materialized):
    source = torch.arange(2 * 8 * 7, dtype=torch.bfloat16).reshape(16, 7)
    key = source.view(2, 8, 7)[:, :3, :]
    value = source.view(2, 8, 7)[:, 3:, :].transpose(1, 2)
    module = SimpleNamespace(
        quant_config=None,
        dcp_q_replicate=False,
        is_aiter_triton_fp4_bmm_enabled=False,
        is_aiter_triton_fp8_bmm_enabled=False,
        num_heads=2,
        qk_nope_head_dim=3,
        v_head_dim=5,
        kv_lora_rank=7,
        kv_b_proj=SimpleNamespace(weight=source),
        W_UK_T=key.clone() if materialized else key,
        W_UV=value.clone() if materialized else value,
        _q_scale_float=7.0,
        _k_scale_float=7.0,
        _v_scale_float=7.0,
        _prob_scale_float=7.0,
    )
    for name in (
        "_q_scale",
        "_k_scale",
        "_v_scale",
        "_prob_scale",
        "_k_scale_cpu",
        "_v_scale_cpu",
    ):
        setattr(module, name, torch.tensor(7.0))
    return module


@pytest.mark.parametrize("materialized", [False, True])
def test_changing_weights_reset_scales_without_rebinding(materialized):
    module = fixture(materialized)
    plan = _prepare_mla_refresh(module)
    original = {
        name: value
        for name, value in vars(module).items()
        if isinstance(value, torch.Tensor)
    }
    for version in (1, 2, 3):
        module.kv_b_proj.weight.add_(version)
        for name in original:
            if name.startswith("_"):
                original[name].fill_(7)
        module._k_scale_float = 7.0
        metrics = plan.refresh()
        for name, tensor in original.items():
            assert getattr(module, name) is tensor
            if name.startswith("_"):
                assert tensor.item() == 1.0
        assert module._k_scale_float == 1.0
        expected = module.kv_b_proj.weight.reshape(2, 8, 7)
        assert torch.equal(module.W_UK_T, expected[:, :3, :])
        assert torch.equal(module.W_UV, expected[:, 3:, :].transpose(1, 2))
        assert metrics["mla_derived_copies"] == 2 * materialized
        assert metrics["mla_derived_shared_views"] == 2 * (not materialized)


@pytest.mark.parametrize("fp4,fp8", [(None, None), (None, False), (False, None)])
def test_nvidia_absent_rocm_capabilities_use_unquantized_refresh(fp4, fp8):
    module = fixture(True)
    module.is_aiter_triton_fp4_bmm_enabled = fp4
    module.is_aiter_triton_fp8_bmm_enabled = fp8
    plan = _prepare_mla_refresh(module)
    module.kv_b_proj.weight.add_(4)
    assert plan.refresh()["mla_derived_copies"] == 2
    expected = module.kv_b_proj.weight.reshape(2, 8, 7)
    assert torch.equal(module.W_UK_T, expected[:, :3, :])
    assert torch.equal(module.W_UV, expected[:, 3:, :].transpose(1, 2))


@pytest.mark.parametrize("value", [True, 0, "", [], torch.tensor(False)])
@pytest.mark.parametrize(
    "mode", ["is_aiter_triton_fp4_bmm_enabled", "is_aiter_triton_fp8_bmm_enabled"]
)
def test_unsupported_capability_values_still_reject_without_writes(mode, value):
    module = fixture(True)
    setattr(module, mode, value)
    before = module.W_UV.clone()
    with pytest.raises(IncompleteRefit, match="alternate MLA representation"):
        _prepare_mla_refresh(module)
    assert torch.equal(module.W_UV, before)
    assert module._q_scale.item() == 7.0


@pytest.mark.parametrize(
    "change",
    [
        "quant",
        "dcp",
        "aiter",
        "shape",
        "scale_shape",
        "float_type",
        "overlap",
        "internal_overlap",
        "scale_alias",
    ],
)
def test_unsupported_refresh_rejects_before_mutation(change):
    module = fixture(True)
    if change == "quant":
        module.quant_config = object()
    elif change == "dcp":
        module.dcp_q_replicate = True
    elif change == "aiter":
        module.is_aiter_triton_fp8_bmm_enabled = True
    elif change == "shape":
        module.W_UV = torch.full((2, 7, 4), 7, dtype=torch.bfloat16)
    elif change == "scale_shape":
        module._v_scale = torch.ones(1)
    elif change == "float_type":
        module._k_scale_float = 1
    elif change == "internal_overlap":
        module.W_UV = torch.full((1,), 7, dtype=torch.bfloat16).as_strided(
            (2, 7, 5), (0, 0, 0)
        )
    elif change == "scale_alias":
        module._v_scale_cpu = module._k_scale_cpu
    else:
        module.W_UK_T = module.kv_b_proj.weight.flatten()[:42].view(2, 3, 7)
    before = module.W_UV.clone()
    with pytest.raises(IncompleteRefit, match="unsupported direct MLA"):
        _prepare_mla_refresh(module)
    assert torch.equal(module.W_UV, before)
    assert module._q_scale.item() == 7.0


@pytest.mark.parametrize("change", ["rebind", "dimensions", "scale_rebind"])
def test_stale_refresh_plan_rejects_before_any_write(change):
    module = fixture(True)
    plan = _prepare_mla_refresh(module)
    if change == "rebind":
        module.W_UV = module.W_UV.clone()
    elif change == "dimensions":
        module.num_heads = 1
    else:
        module._k_scale_cpu = module._k_scale_cpu.clone()
    module.kv_b_proj.weight.add_(4)
    before = module.W_UV.clone()
    with pytest.raises(IncompleteRefit):
        plan.refresh()
    assert torch.equal(module.W_UV, before)
    assert module._q_scale.item() == 7.0
