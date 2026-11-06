# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from modelexpress_rl.refit.reshard.types import IncompleteRefit
from modelexpress_rl.inference.engines.vllm import inplace as inplace_module
from modelexpress_rl.inference.engines.vllm import installer as installer_module
from modelexpress_rl.inference.engines.vllm import sparse_experts
from modelexpress_rl.inference.engines.vllm.surgical import (
    checkpoint_module,
    module_groups,
)
from modelexpress_rl.inference.staging import write_staging
from tests.refit_fixtures import _fake_modules, _save
from torch import nn

ORACLE = "vllm.model_executor.layers.fused_moe.oracle.fp8"
EXPERTS, INTER, HIDDEN, TP, RANK = 3, 4, 6, 2, 1
PREFIX = "model.layers.0.mlp.experts"


def _checkpoint_tensors(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)

    def weight(rows, cols):
        return torch.randint(
            0, 256, (rows, cols), generator=generator, dtype=torch.int32
        ).to(torch.uint8)

    def scale(rows, cols):
        return torch.rand((rows, cols), generator=generator) + 0.5

    tensors = {}
    for expert in range(EXPERTS):
        name = f"{PREFIX}.{expert}"
        tensors[f"{name}.gate_proj.weight"] = weight(INTER, HIDDEN)
        tensors[f"{name}.up_proj.weight"] = weight(INTER, HIDDEN)
        tensors[f"{name}.down_proj.weight"] = weight(HIDDEN, INTER)
        tensors[f"{name}.gate_proj.weight_scale_inv"] = scale(INTER // 2, HIDDEN // 2)
        tensors[f"{name}.up_proj.weight_scale_inv"] = scale(INTER // 2, HIDDEN // 2)
        tensors[f"{name}.down_proj.weight_scale_inv"] = scale(HIDDEN // 2, INTER // 2)
    # A near-zero scale of an unused expert, which processing clamps.
    tensors[f"{PREFIX}.2.down_proj.weight_scale_inv"][0, 1] = 1e-23
    return tensors


def _convert(w13, w2, s13, s2):
    """Fake per-expert kernel layout: gate/up swap and transposes, clamped scales."""
    half = w13.shape[1] // 2
    w13 = torch.cat([w13[:, half:], w13[:, :half]], 1).transpose(1, 2).contiguous()
    shalf = s13.shape[1] // 2
    s13 = torch.cat([s13[:, shalf:], s13[:, :shalf]], 1)
    return (
        w13,
        w2.transpose(1, 2).contiguous(),
        s13.clamp(min=1e-10),
        s2.clamp(min=1e-10),
    )


def _narrow(tensor, dim):
    size = tensor.shape[dim] // TP
    return tensor.narrow(dim, RANK * size, size)


def _live(tensors):
    def get(expert, proj, kind):
        return tensors[f"{PREFIX}.{expert}.{proj}.{kind}"]

    def gate_up(kind):
        return torch.stack(
            [
                torch.cat(
                    [
                        _narrow(get(e, "gate_proj", kind), 0),
                        _narrow(get(e, "up_proj", kind), 0),
                    ]
                )
                for e in range(EXPERTS)
            ]
        )

    def down(kind):
        return torch.stack(
            [_narrow(get(e, "down_proj", kind), 1) for e in range(EXPERTS)]
        )

    return _convert(
        gate_up("weight"),
        down("weight"),
        gate_up("weight_scale_inv"),
        down("weight_scale_inv"),
    )


class RoutedExperts(nn.Module):
    def __init__(self, tensors, backend="FLASHINFER_TRTLLM", kernel_copy=True):
        super().__init__()
        self.moe_config = SimpleNamespace(
            tp_rank=RANK,
            is_act_and_mul=True,
            moe_parallel_config=SimpleNamespace(tp_size=TP, ep_size=1),
        )
        w13, w2, s13, s2 = _live(tensors)
        self.w13_weight = nn.Parameter(w13, requires_grad=False)
        self.w2_weight = nn.Parameter(w2, requires_grad=False)
        self.w13_weight_scale_inv = nn.Parameter(s13, requires_grad=False)
        self.w2_weight_scale_inv = nn.Parameter(s2, requires_grad=False)
        # After a reload, the kernel config can hold copies of the layer's scales.
        scales = (
            (s13.clone(), s2.clone())
            if kernel_copy
            else (self.w13_weight_scale_inv, self.w2_weight_scale_inv)
        )
        self.quant_method = SimpleNamespace(
            fp8_backend=SimpleNamespace(name=backend),
            block_quant=True,
            weight_scale_name="weight_scale_inv",
            moe_quant_config=SimpleNamespace(w1_scale=scales[0], w2_scale=scales[1]),
        )

    def _map_global_expert_id_to_local_expert_id(self, expert_id):
        return expert_id

    def _load_model_weight_or_group_weight_scale(
        self, shard_dim, expert_data, shard_id, loaded_weight, tp_rank
    ):
        assert tp_rank == RANK
        loaded = _narrow(loaded_weight, shard_dim)
        if shard_id == "w2":
            expert_data.copy_(loaded)
            return
        half = expert_data.shape[shard_dim] // 2
        expert_data.narrow(shard_dim, 0 if shard_id == "w1" else half, half).copy_(
            loaded
        )


def _model(tensors, **kwargs):
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].mlp = nn.Module()
    model.model.layers[0].mlp.experts = RoutedExperts(tensors, **kwargs)
    return model


@pytest.fixture
def fake_oracle(monkeypatch):
    def convert(*, fp8_backend, layer, w13, w2, w13_scale, w2_scale, **_):
        assert fp8_backend.name == "FLASHINFER_TRTLLM"
        return _convert(w13, w2, w13_scale, w2_scale)

    _fake_modules(monkeypatch, {ORACLE: {"convert_to_fp8_moe_kernel_format": convert}})
    monkeypatch.setattr(inplace_module, "_routed_experts_class", lambda: RoutedExperts)


def _stage(path, base, target):
    tensors = {}
    for name in sorted(target):
        old = base[name].contiguous().view(-1).view(torch.uint8).numpy()
        new = target[name].contiguous().view(-1).view(torch.uint8).numpy()
        positions = np.flatnonzero(old ^ new).astype(np.uint32)
        if positions.size:
            tensors[name] = (old.size, positions, (old ^ new)[positions])
    path.mkdir(parents=True, exist_ok=True)
    write_staging(path, base_version="base", tensors=tensors)
    return path


def _target(base, changes):
    target = {name: tensor.clone() for name, tensor in base.items()}
    for name, edit in changes.items():
        edit(target[name])
    return target


def _assert_live(model, tensors):
    experts = model.model.layers[0].mlp.experts
    expected = _live(tensors)
    for live, value in zip(sparse_experts.live_tensors(experts), expected, strict=True):
        assert torch.equal(live.view(torch.uint8), value.contiguous().view(torch.uint8))
    for copy, value in zip(
        (
            experts.quant_method.moe_quant_config.w1_scale,
            experts.quant_method.moe_quant_config.w2_scale,
        ),
        expected[2:],
        strict=True,
    ):
        assert torch.equal(copy, value)


def _xor_bits(tensor, mask):
    tensor.view(torch.uint8)[mask] ^= 0x5A


def _changes():
    def set_scale(value):
        def edit(tensor):
            tensor[0, 1] = value

        return edit

    return {
        # Rows of both ranks of one gate and one up projection, and columns of one down.
        f"{PREFIX}.0.gate_proj.weight": lambda t: _xor_bits(
            t, (slice(None), slice(0, 3))
        ),
        f"{PREFIX}.1.up_proj.weight": lambda t: _xor_bits(t, (slice(1, 4), 2)),
        f"{PREFIX}.2.down_proj.weight": lambda t: _xor_bits(t, (4, slice(None))),
        f"{PREFIX}.1.gate_proj.weight_scale_inv": set_scale(0.125),
        # The clamped near-zero scale gets a real value, and a real one goes near zero.
        f"{PREFIX}.2.down_proj.weight_scale_inv": set_scale(0.75),
        f"{PREFIX}.0.down_proj.weight_scale_inv": set_scale(1e-30),
    }


@pytest.mark.parametrize("files", ["target", "base"])
def test_sparse_install_reproduces_the_target_layout_bit_for_bit(
    fake_oracle, tmp_path, files
):
    base = _checkpoint_tensors(5)
    target = _target(base, _changes())
    model = _model(base)
    staging = _stage(tmp_path / "staging", base, target)
    # Files of the target, or of the base with the target's delta staged on them.
    checkpoint = _save(tmp_path / "checkpoint", target if files == "target" else base)
    overlay = () if files == "target" else (staging,)
    installer = inplace_module.InPlaceInstaller(
        model, device=torch.device("cpu"), generic_reload=lambda load: None
    )
    pointers = [
        t.data_ptr()
        for t in sparse_experts.live_tensors(model.model.layers[0].mlp.experts)
    ]

    metrics = installer.install(checkpoint, staging, overlay=overlay)

    _assert_live(model, target)
    assert [
        t.data_ptr()
        for t in sparse_experts.live_tensors(model.model.layers[0].mlp.experts)
    ] == pointers
    assert metrics["perf/mx_receive_inplace_sparse_expert_layers"] == 1
    assert metrics["perf/mx_receive_inplace_sparse_scale_tensors"] == 3
    assert metrics["perf/mx_receive_inplace_sparse_expert_bytes"] > 0


def test_layout_maps_owned_elements_and_finds_the_scale_clamp(fake_oracle):
    base = _checkpoint_tensors(7)
    experts = _model(base).model.layers[0].mlp.experts
    shapes = {
        (proj, kind): tuple(base[f"{PREFIX}.0.{proj}_proj.{kind}"].shape)
        for proj in ("gate", "up", "down")
        for kind in ("weight", "weight_scale_inv")
    }

    layout = sparse_experts.derive_layout(experts, shapes, torch.device("cpu"))

    assert layout.floor == pytest.approx(1e-10)
    # Only scale writes read the owned elements' list.
    assert set(layout.sources) == {
        (proj, sparse_experts.SCALE) for proj in ("gate", "up", "down")
    }
    for (proj, kind), mapping in layout.maps.items():
        source = torch.nonzero(mapping >= 0, as_tuple=True)[0]
        if kind == sparse_experts.SCALE:
            assert torch.equal(layout.sources[(proj, kind)], source)
        target = sparse_experts._target(proj, kind)
        live = sparse_experts.live_tensors(experts)[target]
        checkpoint = base[f"{PREFIX}.1.{proj}_proj.{kind}"].reshape(-1)
        # Every owned element sits where the map says, and the rank owns half.
        assert source.numel() == checkpoint.numel() // TP
        placed = live[1].reshape(-1)[mapping[source]]
        expected = checkpoint[source]
        assert torch.equal(
            placed,
            expected.clamp(min=1e-10) if kind == "weight_scale_inv" else expected,
        )


def test_changes_owned_by_another_rank_write_nothing(fake_oracle, tmp_path):
    base = _checkpoint_tensors(9)
    # Rows 0 and 1 of gate belong to rank 0.
    target = _target(
        base,
        {
            f"{PREFIX}.0.gate_proj.weight": lambda t: _xor_bits(
                t, (slice(0, 2), slice(None))
            )
        },
    )
    model = _model(base)
    staging = _stage(tmp_path / "staging", base, target)
    installer = inplace_module.InPlaceInstaller(
        model, device=torch.device("cpu"), generic_reload=lambda load: None
    )

    metrics = installer.install(_save(tmp_path / "checkpoint", target), staging)

    assert metrics["perf/mx_receive_inplace_sparse_expert_bytes"] == 0
    _assert_live(model, target)


@pytest.mark.parametrize(
    ("backend", "scales", "unsupported"),
    [
        ("DEEPGEMM", False, True),
        ("FLASHINFER_TRTLLM", True, True),
        ("FLASHINFER_TRTLLM", False, False),
    ],
)
def test_arranged_installs_refuse_fp8_processing_that_changes_values(
    backend, scales, unsupported
):
    method = SimpleNamespace(fp8_backend=SimpleNamespace(name=backend))
    kinds = {"weight", "weight_scale_inv"} if scales else {"weight"}

    reason = inplace_module.fp8_moe_unsupported(
        SimpleNamespace(quant_method=method), kinds
    )

    assert (reason is not None) == unsupported


@pytest.mark.parametrize(
    ("cause", "reason"),
    [
        ("padded", "tensor-parallel shards are padded"),
        ("derivation", "OutOfMemoryError: CUDA out of memory"),
    ],
    ids=["padded", "derivation"],
)
def test_layers_sparse_writes_cannot_take_go_to_the_guarded_arranged_path(
    fake_oracle, monkeypatch, tmp_path, caplog, cause, reason
):
    base = _checkpoint_tensors(11)
    target = _target(
        base, {f"{PREFIX}.0.gate_proj.weight_scale_inv": lambda t: t.mul_(2)}
    )
    model = _model(base)

    def out_of_memory(*_args):
        raise torch.OutOfMemoryError("CUDA out of memory")

    if cause == "padded":
        model.model.layers[0].mlp.experts.moe_config.tp_shard_with_padding = True
    else:
        # A layout that fails to derive for any reason only rules the layer out.
        monkeypatch.setattr(sparse_experts, "derive_layout", out_of_memory)
    staging = _stage(tmp_path / "staging", base, target)
    agreed = []
    peers = inplace_module.Peers(
        0,
        1,
        lambda tensor: tensor,
        lambda flags: agreed.append(flags.tolist()) or flags,
    )
    installer = inplace_module.InPlaceInstaller(
        model, device=torch.device("cpu"), generic_reload=lambda load: None, peers=peers
    )

    with (
        caplog.at_level(logging.INFO),
        pytest.raises(inplace_module.InPlaceUnsupported, match="changes values"),
    ):
        installer.install(_save(tmp_path / "checkpoint", target), staging)

    assert reason in caplog.text
    # The rank still joins its peers in agreeing on the sparse layers.
    assert agreed == [[0]]
    _assert_live(model, base)


def test_ineligible_reasons_cover_backend_and_layout():
    base = _checkpoint_tensors(13)
    assert sparse_experts.ineligible(RoutedExperts(base)) is None
    assert "DEEPGEMM" in sparse_experts.ineligible(
        RoutedExperts(base, backend="DEEPGEMM")
    )
    experts = RoutedExperts(base)
    experts.quant_method.moe_quant_config.w1_scale = torch.zeros(1)
    assert "scales differ" in sparse_experts.ineligible(experts)


class Fp8LinearMethod:
    def __init__(self, kernel):
        self.fp8_linear = type(kernel, (), {})()


class _ValueModel(nn.Module):
    """One layer with an FP8 linear, an FP8 indexer wk, and a norm, recording loads."""

    def __init__(self, kernel):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = nn.Module()
        layer.self_attn.q_b_proj = nn.Linear(2, 2, bias=False)
        layer.self_attn.q_b_proj.quant_method = Fp8LinearMethod(kernel)
        layer.self_attn.indexer = nn.Module()
        layer.self_attn.indexer.wk_weights_proj = nn.Linear(2, 3, bias=False)
        layer.input_layernorm = nn.Module()
        layer.input_layernorm.weight = nn.Parameter(torch.ones(2), requires_grad=False)
        self.loads = []

    def load_weights(self, weights):
        self.loads.append(sorted(name for name, _tensor in weights))
        return set()


def _value_checkpoint():
    base = {
        "model.layers.0.self_attn.q_b_proj.weight": torch.ones(2, 2),
        "model.layers.0.self_attn.indexer.wk.weight": torch.ones(1, 2).to(
            torch.float8_e4m3fn
        ),
        "model.layers.0.self_attn.indexer.wk.weight_scale_inv": torch.ones(1, 1),
        "model.layers.0.self_attn.indexer.weights_proj.weight": torch.ones(2, 2),
        "model.layers.0.input_layernorm.weight": torch.ones(2),
    }
    target = {name: tensor.clone() for name, tensor in base.items()}
    target["model.layers.0.self_attn.q_b_proj.weight"][0, 0] = 2.0
    target["model.layers.0.self_attn.indexer.wk.weight_scale_inv"][0, 0] = 2.0
    target["model.layers.0.input_layernorm.weight"][1] = 3.0
    return base, target


@pytest.mark.parametrize(
    "kernel", ["DeepGemmFp8BlockScaledMMKernel", "CutlassFp8BlockScaledMMKernel"]
)
def test_modules_whose_processing_changes_values_reload_from_target_values(
    monkeypatch, tmp_path, kernel
):
    monkeypatch.setattr(inplace_module, "_routed_experts_class", lambda: None)
    base, target = _value_checkpoint()
    model = _ValueModel(kernel)
    staging = _stage(tmp_path / "staging", base, target)
    checkpoint = _save(tmp_path / "checkpoint", target)
    reloaded = []
    installer = inplace_module.InPlaceInstaller(
        model, device=torch.device("cpu"), generic_reload=lambda load: load(None)
    )

    installer.install(
        checkpoint, staging, reload_values=lambda names: reloaded.append(names) or {}
    )

    expected = {"model.layers.0.self_attn.indexer.wk.weight_scale_inv"}
    if kernel.startswith("DeepGemm"):
        expected.add("model.layers.0.self_attn.q_b_proj.weight")
    assert reloaded == [expected]
    # The XOR reload feeds neither those modules nor the indexer's fused weights_proj.
    fed = {name for batch in model.loads for name in batch}
    assert "model.layers.0.input_layernorm.weight" in fed
    assert not {name for name in fed if ".indexer." in name}
    assert ("model.layers.0.self_attn.q_b_proj.weight" in fed) == kernel.startswith(
        "Cutlass"
    )

    with pytest.raises(inplace_module.InPlaceUnsupported, match="changes the values"):
        installer.install(checkpoint, staging)


def test_indexer_wk_groups_with_weights_proj():
    names = [
        "model.layers.0.self_attn.indexer.wk.weight",
        "model.layers.0.self_attn.indexer.wk.weight_scale_inv",
        "model.layers.0.self_attn.indexer.weights_proj.weight",
        "model.layers.0.self_attn.indexer.wq_b.weight",
    ]

    assert (
        checkpoint_module(names[0], {})
        == "model.layers.0.self_attn.indexer.wk_weights_proj"
    )
    assert module_groups(names[1:2], names, {}) == set(names[:3])


def test_reload_window_discards_stale_and_rejects_stranded_indexer_pairs(caplog):
    model = nn.Module()
    model._pending_indexer_wk_fp8 = {
        "model.layers.0.self_attn.indexer": {"weight": torch.zeros(1)}
    }

    with caplog.at_level(logging.WARNING):
        installer_module._clear_pending_fp8_indexer_wk(model)

    assert model._pending_indexer_wk_fp8 == {} and "model.layers.0" in caplog.text
    installer_module._require_no_pending_fp8_indexer_wk(model)
    model._pending_indexer_wk_fp8["model.layers.1.self_attn.indexer"] = {
        "scale": torch.ones(1)
    }
    with pytest.raises(IncompleteRefit, match="model.layers.1"):
        installer_module._require_no_pending_fp8_indexer_wk(model)


class _IndexerBufferingModel:
    """Pairs FP8 indexer wk tensors like vLLM; ``drop_after`` strands one pair."""

    def __init__(self, drop_after=None):
        self._pending_indexer_wk_fp8 = {}
        self.written = {}
        self.drop_after = drop_after

    def load_weights(self, weights):
        for name, tensor in weights:
            if ".indexer.wk." not in name:
                continue
            prefix = name.rsplit(".wk.", 1)[0]
            entry = self._pending_indexer_wk_fp8.setdefault(prefix, {})
            entry["scale" if "weight_scale" in name else "weight"] = tensor
            if name == self.drop_after:
                self.drop_after = None
                self._pending_indexer_wk_fp8.clear()
                continue
            if "weight" in entry and "scale" in entry:
                pair = self._pending_indexer_wk_fp8.pop(prefix)
                self.written[prefix] = (pair["weight"].tolist(), pair["scale"].tolist())
        return set()


def _indexer_stream(buffer):
    # Every yielded tensor is a view into one reused read buffer.
    for layer in (0, 1):
        prefix = f"model.layers.{layer}.self_attn.indexer"
        for name, value in (
            (f"{prefix}.wk.weight", layer + 1),
            (f"{prefix}.weights_proj.weight", 0),
            (f"{prefix}.wk.weight_scale_inv", layer + 10),
        ):
            buffer.fill_(value)
            yield name, buffer[:]


def test_streamed_loads_rewrite_indexer_pairs_from_owned_copies():
    from modelexpress.engines.vllm.adapter import VllmAdapter
    from modelexpress.load_strategy.context import LoadResult

    model = _IndexerBufferingModel(
        drop_after="model.layers.1.self_attn.indexer.wk.weight"
    )
    adapter = object.__new__(VllmAdapter)

    adapter.apply_weight_iter(
        LoadResult(value=model, model=model),
        _indexer_stream(torch.zeros(2, dtype=torch.float8_e4m3fn)),
    )

    assert model.written == {
        "model.layers.0.self_attn.indexer": ([1, 1], [10, 10]),
        "model.layers.1.self_attn.indexer": ([2, 2], [11, 11]),
    }
    assert model._pending_indexer_wk_fp8 == {}


def test_streamed_loads_reject_an_indexer_weight_without_its_scale():
    from modelexpress.engines.vllm.adapter import VllmAdapter
    from modelexpress.load_strategy.context import LoadResult

    def stream():
        for name, tensor in _indexer_stream(torch.zeros(2, dtype=torch.float8_e4m3fn)):
            if name != "model.layers.1.self_attn.indexer.wk.weight_scale_inv":
                yield name, tensor

    adapter = object.__new__(VllmAdapter)
    model = _IndexerBufferingModel()
    with pytest.raises(RuntimeError, match="model.layers.1.self_attn.indexer"):
        adapter.apply_weight_iter(LoadResult(value=model, model=model), stream())
