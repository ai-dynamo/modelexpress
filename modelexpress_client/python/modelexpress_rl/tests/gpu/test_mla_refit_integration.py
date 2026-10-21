# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-vLLM test; checkpoints must differ in MLA KV projection weights."""

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.gpu
@pytest.mark.slow
def test_quantized_mla_checkpoint_refit():
    if not all(os.environ.get(key) for key in ("MX_TEST_MLA_BASE", "MX_TEST_MLA_UPDATED")):
        pytest.skip("set MX_TEST_MLA_BASE and MX_TEST_MLA_UPDATED to local checkpoints")

    # A subprocess avoids the parent test suite's vLLM module mocks.
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent('''
            import os
            import torch
            from vllm import LLM
            from vllm.model_executor.layers.quantization.utils.quant_utils import (
                get_and_maybe_dequant_weights,
            )
            from modelexpress_rl.inference.engines.vllm.installer import _VllmInstaller

            assert torch.cuda.is_available(), "a CUDA GPU is required"
            llm = LLM(
                model=os.environ["MX_TEST_MLA_BASE"],
                enforce_eager=True,
                max_model_len=256,
                max_num_seqs=1,
                skip_tokenizer_init=True,
            )
            config = llm.llm_engine.vllm_config
            assert config.quant_config is not None, "use a quantized MLA checkpoint"

            def refit(model):
                layers = [
                    layer for layer in model.modules()
                    if hasattr(layer, "W_UV") and hasattr(layer, "W_UK_T")
                ]
                assert layers, "checkpoint must expose MLA W_UV and W_UK_T"
                originals = [
                    (layer, {
                        name: (getattr(layer, name), getattr(layer, name).data_ptr(),
                               getattr(layer, name).clone())
                        for name in ("W_UV", "W_UK_T")
                    })
                    for layer in layers
                ]
                installer = _VllmInstaller(
                    model=model,
                    vllm_config=config,
                    model_config=config.model_config,
                    device=next(model.parameters()).device,
                )
                installer.install_checkpoint(os.environ["MX_TEST_MLA_UPDATED"])
                changed = {"W_UV": False, "W_UK_T": False}
                for layer, tensors in originals:
                    weight = get_and_maybe_dequant_weights(
                        layer.kv_b_proj, out_dtype=config.model_config.dtype,
                    ).T.reshape(
                        layer.kv_lora_rank, layer.num_heads,
                        layer.qk_nope_head_dim + layer.v_head_dim,
                    )
                    uk, uv = weight.split(
                        [layer.qk_nope_head_dim, layer.v_head_dim], dim=-1,
                    )
                    expected = {"W_UV": uv.transpose(0, 1), "W_UK_T": uk.permute(1, 2, 0)}
                    for name, (original, pointer, before) in tensors.items():
                        actual = getattr(layer, name)
                        assert actual is original
                        assert actual.data_ptr() == pointer
                        torch.testing.assert_close(actual, expected[name])
                        changed[name] |= not torch.equal(actual, before)
                assert all(changed.values()), "updated checkpoint must change both MLA tensors"
                return True

            assert all(llm.apply_model(refit))
        ''')],
        text=True,
        capture_output=True,
        timeout=1800,
    )
    assert result.returncode == 0, result.stdout + result.stderr
