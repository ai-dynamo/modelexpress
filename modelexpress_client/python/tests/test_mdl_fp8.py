# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from modelexpress.engines.vllm.mdl import MdlLoader


F8 = torch.float8_e4m3fn


def _parameter(shape, *, dtype=torch.float32):
    return torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)


class _LoaderlessModel(torch.nn.Module):
    def load_weights(self, *, weights):
        del weights
        raise AttributeError("processed FP8 parameters are not re-entrant")


def _load(model, monkeypatch, weights):
    monkeypatch.setenv("MX_LOAD_MODE", "direct")
    loader = MdlLoader(model)
    loader.load_weights(weights)
    return loader


def test_loaderless_fp8_direct_weight_and_scale_alias(monkeypatch):
    model = _LoaderlessModel()
    model.proj = torch.nn.Module()
    model.proj.weight = _parameter((2, 2), dtype=F8)
    model.proj.weight_scale = _parameter((2, 1))

    weight = torch.tensor([[1, 2], [3, 4]], dtype=F8)
    scale = torch.tensor([[0.25], [0.5]])
    loader = _load(
        model,
        monkeypatch,
        [("proj.weight", weight), ("proj.weight_scale_inv", scale)],
    )

    assert torch.equal(model.proj.weight.float(), weight.float())
    assert torch.equal(model.proj.weight_scale, scale)
    assert set(loader._direct) == {"proj.weight", "proj.weight_scale_inv"}


def test_loaderless_fp8_fused_qkv_gate_up_and_activation_scales(monkeypatch):
    model = _LoaderlessModel()
    model.attn = torch.nn.Module()
    model.attn.qkv_proj = torch.nn.Module()
    model.attn.qkv_proj.weight = _parameter((4, 2), dtype=F8)
    model.attn.qkv_proj.weight_scale = _parameter((4, 1))
    model.attn.qkv_proj.input_scale = _parameter((3,))
    model.mlp = torch.nn.Module()
    model.mlp.gate_up_proj = torch.nn.Module()
    model.mlp.gate_up_proj.weight = _parameter((4, 2), dtype=F8)
    model.mlp.gate_up_proj.weight_scale = _parameter((4, 1))
    model.mlp.gate_up_proj.input_scale = _parameter((2,))

    weights = []
    expected_qkv = []
    expected_qkv_scales = []
    for index, (name, rows) in enumerate((("q_proj", 2), ("k_proj", 1), ("v_proj", 1))):
        value = torch.full((rows, 2), index + 1, dtype=F8)
        scale = torch.full((rows, 1), index + 0.25)
        weights.extend(
            [
                (f"attn.{name}.weight", value),
                (f"attn.{name}.weight_scale_inv", scale),
                (f"attn.{name}.input_scale", torch.tensor([index + 0.5])),
            ]
        )
        expected_qkv.append(value)
        expected_qkv_scales.append(scale)

    expected_gate_up = []
    expected_gate_up_scales = []
    for index, name in enumerate(("gate_proj", "up_proj")):
        value = torch.full((2, 2), index + 5, dtype=F8)
        scale = torch.full((2, 1), index + 1.25)
        weights.extend(
            [
                (f"mlp.{name}.weight", value),
                (f"mlp.{name}.weight_scale_inv", scale),
                (f"mlp.{name}.activation_scale", torch.tensor([index + 2.5])),
            ]
        )
        expected_gate_up.append(value)
        expected_gate_up_scales.append(scale)

    loader = _load(model, monkeypatch, weights)

    assert torch.equal(
        model.attn.qkv_proj.weight.float(), torch.cat(expected_qkv).float()
    )
    assert torch.equal(model.attn.qkv_proj.weight_scale, torch.cat(expected_qkv_scales))
    assert torch.equal(model.attn.qkv_proj.input_scale, torch.tensor([0.5, 1.5, 2.5]))
    assert torch.equal(
        model.mlp.gate_up_proj.weight.float(), torch.cat(expected_gate_up).float()
    )
    assert torch.equal(
        model.mlp.gate_up_proj.weight_scale, torch.cat(expected_gate_up_scales)
    )
    assert torch.equal(model.mlp.gate_up_proj.input_scale, torch.tensor([2.5, 3.5]))
    assert len(loader._fused) == len(weights)


def test_loaderless_fp8_expert_weight_scale_and_input_scale_layouts(monkeypatch):
    model = _LoaderlessModel()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    mlp = torch.nn.Module()
    model.model.layers[0].mlp = mlp
    mlp.experts = torch.nn.Module()
    mlp.experts.w13_weight = _parameter((1, 4, 2), dtype=F8)
    mlp.experts.w2_weight = _parameter((1, 2, 2), dtype=F8)
    mlp.experts.w13_weight_scale = _parameter((1, 4, 1))
    mlp.experts.w2_weight_scale = _parameter((1, 2, 1))
    mlp.experts.w13_input_scale = _parameter((1, 2))
    mlp.experts.w2_input_scale = _parameter((1, 1))

    def expert_mapping():
        prefix = "model.layers.0.mlp."
        return [
            (prefix + "experts.w13_", prefix + "experts.0.gate_proj.", 0, "w1"),
            (prefix + "experts.w2_", prefix + "experts.0.down_proj.", 0, "w2"),
            (prefix + "experts.w13_", prefix + "experts.0.up_proj.", 0, "w3"),
        ]

    model.get_expert_mapping = expert_mapping
    gate = torch.full((2, 2), 3, dtype=F8)
    up = torch.full((2, 2), 7, dtype=F8)
    down = torch.full((2, 2), 9, dtype=F8)
    weights = [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", gate),
        ("model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv", torch.full((2, 1), 0.3)),
        ("model.layers.0.mlp.experts.0.gate_proj.input_scale", torch.tensor([1.0])),
        ("model.layers.0.mlp.experts.0.up_proj.weight", up),
        ("model.layers.0.mlp.experts.0.up_proj.weight_scale_inv", torch.full((2, 1), 0.7)),
        ("model.layers.0.mlp.experts.0.up_proj.input_scale", torch.tensor([2.0])),
        ("model.layers.0.mlp.experts.0.down_proj.weight", down),
        ("model.layers.0.mlp.experts.0.down_proj.weight_scale_inv", torch.full((2, 1), 0.9)),
        ("model.layers.0.mlp.experts.0.down_proj.input_scale", torch.tensor([3.0])),
    ]

    loader = _load(model, monkeypatch, weights)

    assert torch.equal(mlp.experts.w13_weight[0, :2].float(), gate.float())
    assert torch.equal(mlp.experts.w13_weight[0, 2:].float(), up.float())
    assert torch.equal(mlp.experts.w2_weight[0].float(), down.float())
    assert torch.equal(
        mlp.experts.w13_weight_scale[0, :, 0],
        torch.tensor([0.3, 0.3, 0.7, 0.7]),
    )
    assert torch.equal(mlp.experts.w13_input_scale[0], torch.tensor([1.0, 2.0]))
    assert torch.equal(mlp.experts.w2_input_scale[0], torch.tensor([3.0]))
    assert len(loader._expert) == len(weights)


class _RecordingModel(torch.nn.Module):
    """FP8 model whose stock load_weights records calls and then fails.

    Mirrors the real regime where vLLM's processed FP8 params make
    ``load_weights`` non-re-entrant; at TP>1 that call can hang rather than
    raise, so MDL must avoid it entirely.
    """

    def __init__(self):
        super().__init__()
        self.load_weights_calls = 0

    def load_weights(self, *, weights):
        del weights
        self.load_weights_calls += 1
        raise AttributeError("processed FP8 parameters are not re-entrant")


def _tp_load(model, monkeypatch, weights, *, tp_size, tp_rank=0, env=None):
    monkeypatch.setenv("MX_LOAD_MODE", "direct")
    if env:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
    loader = MdlLoader(model)
    loader._tp_size = tp_size
    loader._tp_rank = tp_rank
    loader.load_weights(weights)
    return loader


def test_fp8_tp2_loaderless_opt_in_skips_stock_call(monkeypatch):
    """The diagnostic opt-in skips the non-re-entrant stock FP8 cold load."""
    model = _RecordingModel()
    model.proj = torch.nn.Module()
    # Local (already TP-sharded) destination slots for tp_rank 0 of a tp_size=2
    # column-parallel projection: global (4, 2) weight -> local (2, 2), and a
    # block-aligned scale global (2, 1) -> local (1, 1).
    model.proj.weight = _parameter((2, 2), dtype=F8)
    model.proj.weight_scale = _parameter((1, 1))

    global_weight = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=F8)
    global_scale = torch.tensor([[0.25], [0.5]])
    loader = _tp_load(
        model,
        monkeypatch,
        [
            ("proj.weight", global_weight),
            ("proj.weight_scale_inv", global_scale),
        ],
        tp_size=2,
        tp_rank=0,
        env={"MX_FP8_LOADERLESS": "1"},
    )

    assert model.load_weights_calls == 0
    assert loader._loaderless is True
    assert torch.equal(model.proj.weight.float(), global_weight[:2].float())
    assert torch.equal(model.proj.weight_scale, global_scale[:1])


def test_fp8_tp2_loaderless_is_not_forced_by_default(monkeypatch):
    """TP2 stays on the legacy path until scale-layout resharding is correct."""
    model = _RecordingModel()
    model.proj = torch.nn.Module()
    model.proj.weight = _parameter((2, 2), dtype=F8)
    model.proj.weight_scale = _parameter((1, 1))

    global_weight = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=F8)
    global_scale = torch.tensor([[0.25], [0.5]])
    loader = _tp_load(
        model,
        monkeypatch,
        [
            ("proj.weight", global_weight),
            ("proj.weight_scale_inv", global_scale),
        ],
        tp_size=2,
        tp_rank=0,
    )

    # Guard disabled: stock load_weights is attempted (and fails), then the
    # exception path switches to loaderless — still correct, just not proactive.
    assert model.load_weights_calls == 1
    assert loader._loaderless is True
    assert torch.equal(model.proj.weight.float(), global_weight[:2].float())


def test_fp8_tp2_expert_gate_up_weight_and_scale_shards(monkeypatch):
    def build_model():
        model = _LoaderlessModel()
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        mlp = torch.nn.Module()
        model.model.layers[0].mlp = mlp
        mlp.experts = torch.nn.Module()
        # TP2 collision: each global gate/up tensor has the same shape as the
        # local fused W13 destination before it is split into gate/up halves.
        mlp.experts.w13_weight = _parameter((1, 4, 2), dtype=F8)
        mlp.experts.w2_weight = _parameter((1, 2, 2), dtype=F8)
        mlp.experts.w13_weight_scale = _parameter((1, 2, 1))
        mlp.experts.w2_weight_scale = _parameter((1, 1, 1))

        def expert_mapping():
            prefix = "model.layers.0.mlp."
            return [
                (
                    prefix + "experts.w13_",
                    prefix + "experts.0.gate_proj.",
                    0,
                    "w1",
                ),
                (
                    prefix + "experts.w2_",
                    prefix + "experts.0.down_proj.",
                    0,
                    "w2",
                ),
                (
                    prefix + "experts.w13_",
                    prefix + "experts.0.up_proj.",
                    0,
                    "w3",
                ),
            ]

        model.get_expert_mapping = expert_mapping
        return model, mlp

    gate = torch.tensor(
        [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=F8
    )
    up = torch.tensor(
        [[11, 12], [13, 14], [15, 16], [17, 18]], dtype=F8
    )
    down = torch.tensor([[21, 22, 23, 24], [25, 26, 27, 28]], dtype=F8)
    gate_scale = torch.tensor([[0.1], [0.2]])
    up_scale = torch.tensor([[0.3], [0.4]])
    down_scale = torch.tensor([[0.5, 0.6]])
    weights = [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", gate),
        (
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
            gate_scale,
        ),
        ("model.layers.0.mlp.experts.0.up_proj.weight", up),
        (
            "model.layers.0.mlp.experts.0.up_proj.weight_scale_inv",
            up_scale,
        ),
        ("model.layers.0.mlp.experts.0.down_proj.weight", down),
        (
            "model.layers.0.mlp.experts.0.down_proj.weight_scale_inv",
            down_scale,
        ),
    ]

    for rank in (0, 1):
        model, mlp = build_model()
        _tp_load(model, monkeypatch, weights, tp_size=2, tp_rank=rank)
        rows = slice(rank * 2, (rank + 1) * 2)
        cols = slice(rank * 2, (rank + 1) * 2)
        assert torch.equal(
            mlp.experts.w13_weight[0, :2].float(), gate[rows].float()
        )
        assert torch.equal(
            mlp.experts.w13_weight[0, 2:].float(), up[rows].float()
        )
        assert torch.equal(
            mlp.experts.w13_weight_scale[0, 0], gate_scale[rank]
        )
        assert torch.equal(
            mlp.experts.w13_weight_scale[0, 1], up_scale[rank]
        )
        assert torch.equal(
            mlp.experts.w2_weight[0].float(), down[:, cols].float()
        )
        assert torch.equal(
            mlp.experts.w2_weight_scale[0], down_scale[:, rank : rank + 1]
        )


def test_loaderless_fp8_coverage_fails_closed(monkeypatch):
    model = _LoaderlessModel()
    model.proj = torch.nn.Module()
    model.proj.weight = _parameter((2, 2), dtype=F8)
    model.proj.weight_scale = _parameter((2, 1))

    with pytest.raises(RuntimeError, match="uncovered local FP8/scale parameters"):
        _load(model, monkeypatch, [("proj.weight", torch.ones((2, 2), dtype=F8))])
