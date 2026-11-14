# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from modelexpress_rl.inference.engines.vllm import installer as module
from modelexpress_rl.inference.plan import PreparedStreamingTensors, SourceReplayBatch
from modelexpress_rl.inference.trainer_plan import TrainerTransferPlan
from modelexpress_rl.refit.reshard.transfer_plan import LogicalTransferPlan
from modelexpress_rl.refit.reshard.types import (
    CaptureResult,
    IncompleteRefit,
    RecordedCopy,
)

LAYERWISE = "vllm.model_executor.model_loader.reload.layerwise"
MANIFEST = [("weight", torch.float32, (2, 2)), ("experts", torch.float32, (3, 2))]


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))
        self.loaded = []

    def load_weights(self, weights):
        self.loaded.extend(weights)


class _DefaultModelLoader:
    def __init__(self, load_config):
        assert load_config.load_format == "safetensors"
        self.local_expert_ids = None

    def _init_ep_weight_filter(self, _model_config):
        self.local_expert_ids = {1}


@pytest.fixture
def installer(monkeypatch):
    modules = {
        name: ModuleType(name)
        for name in (
            "vllm",
            "vllm.config",
            "vllm.model_executor",
            "vllm.model_executor.layers",
            "vllm.model_executor.layers.attention",
            "vllm.model_executor.layers.quantization",
            "vllm.model_executor.layers.quantization.base_config",
            "vllm.model_executor.model_loader",
            "vllm.model_executor.model_loader.default_loader",
            "vllm.model_executor.model_loader.ep_weight_filter",
            "vllm.model_executor.model_loader.weight_utils",
            "vllm.model_executor.model_loader.reload",
            "vllm.model_executor.model_loader.reload.meta",
            LAYERWISE,
        )
    }
    modules["vllm.config"].set_current_vllm_config = lambda _config: nullcontext()
    modules["vllm.model_executor.layers.quantization.base_config"].QuantizeMethodBase = type(
        "QuantizeMethodBase", (), {}
    )
    modules["vllm.model_executor.layers.attention"].is_deferred_attention_layer = lambda _layer: False
    modules["vllm.model_executor.model_loader.weight_utils"].default_weight_loader = None
    modules["vllm.model_executor.model_loader.default_loader"].DefaultModelLoader = _DefaultModelLoader
    modules["vllm.model_executor.model_loader.ep_weight_filter"].should_skip_weight = (
        lambda name, local_expert_ids: ".experts." in name and int(name.split(".")[-2]) not in local_expert_ids
    )
    modules["vllm.model_executor.model_loader.reload.meta"].materialize_layer = lambda _layer, _info: None
    layerwise = modules[LAYERWISE]
    layerwise.LAYERWISE_INFO = {}
    layerwise.initialize_layerwise_reload = lambda _model: None
    layerwise.finalize_layerwise_reload = lambda _model, _config: None
    layerwise._copy_and_restore_kernel_tensors = lambda _layer, _info: None
    layerwise._get_original_loader = lambda _parameter: None
    layerwise._place_kernel_tensors = lambda _layer, _info: None
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    model = _Model()
    return module._VllmInstaller(
        model=model,
        vllm_config=SimpleNamespace(quant_config=None, load_config=SimpleNamespace(load_format="modelexpress")),
        model_config=SimpleNamespace(),
        device=torch.device("cpu"),
        convert_native_to_hf=lambda sources: {f"hf.{name}": tensor for name, tensor in sources.items()},
        install_through_loader=True,
    )


def _staged(installer, arena):
    capture = installer._loader_replay[1][0]
    writes = tuple(
        (index, write, _read(arena[write.src_name], write.op_chain)) for index, write in enumerate(capture.copies)
    )
    return SourceReplayBatch(writes, capture.replay_id)


def _read(source, op_chain):
    for name, args, kwargs in op_chain:
        source = getattr(source, name)(*args, **dict(kwargs))
    return source.contiguous()


def _unstack_experts(sources):
    return {
        f"mlp.experts.{index}.weight": expert.t()
        for name, stacked in sources.items()
        for index, expert in enumerate(stacked.unbind(0))
    }


def test_loader_install_stages_every_source_whole_without_tracing(installer, monkeypatch):
    def trace(*_args, **_kwargs):
        raise AssertionError("installing through vLLM's loader traces nothing")

    monkeypatch.setattr(module, "capture_weights", trace)

    capture, layout = installer.capture(MANIFEST)

    assert capture.loader_replay and capture.source_replay
    assert capture.copies == [
        RecordedCopy("weight", (), "weight", 0, (2, 2), (2, 1), torch.float32),
        RecordedCopy("experts", (), "experts", 0, (3, 2), (2, 1), torch.float32),
    ]
    assert layout == {"weight": ((2, 2), torch.float32), "experts": ((3, 2), torch.float32)}
    assert installer.capture(MANIFEST)[0].replay_id == capture.replay_id


def test_loader_install_never_stages_another_ranks_experts(installer):
    manifest = [
        ("mlp.experts.0.weight", torch.bfloat16, (2, 2)),
        ("mlp.experts.1.weight", torch.bfloat16, (2, 2)),
        ("norm.weight", torch.float32, (2,)),
    ]

    capture, layout = installer.capture(manifest)

    assert [write.src_name for write in capture.copies] == ["mlp.experts.1.weight", "norm.weight"]
    assert layout == {"mlp.experts.1.weight": ((2, 2), torch.bfloat16), "norm.weight": ((2,), torch.float32)}


def test_loader_install_reads_only_owned_rows_of_stacked_experts(installer):
    installer._convert_native_to_hf = _unstack_experts
    stacked = torch.arange(12.0).view(3, 2, 2)

    capture, _ = installer.capture([("experts", torch.float32, (3, 2, 2))])
    installer.install_tensors(_staged(installer, {"experts": stacked}))

    assert [(write.op_chain, write.dest_shape) for write in capture.copies] == [((("narrow", (0, 1, 1), ()),), (1, 2, 2))]
    assert [(name, tensor.tolist()) for name, tensor in installer._model.loaded] == [
        ("mlp.experts.1.weight", stacked[1].t().tolist())
    ]


@pytest.mark.parametrize(("fallback", "accepted"), [((), True), (("experts",), False)])
def test_loader_plan_needs_every_source_staged_but_no_parameter_coverage(fallback, accepted):
    capture = CaptureResult(
        copies=[RecordedCopy("experts", (), "experts", 0, (3, 2), (2, 1), torch.float32)],
        source_replay=True,
        loader_replay=True,
    )
    plan = LogicalTransferPlan(source_replay=True, replay_id=capture.replay_id, fallback=fallback)

    if accepted:
        TrainerTransferPlan.validate_capture(capture, {"weight": ((2, 2), torch.float32)}, plan)
    else:
        with pytest.raises(IncompleteRefit, match="could not stage 1 sources"):
            TrainerTransferPlan.validate_capture(capture, {"weight": ((2, 2), torch.float32)}, plan)


def test_loader_install_streams_every_batch_through_one_load_weights_call(installer):
    capture, _ = installer.capture(MANIFEST)
    arena = torch.arange(10, dtype=torch.float32)
    weight, experts = capture.copies
    batches = [
        SourceReplayBatch(((0, weight, arena[:4].view(2, 2)),), capture.replay_id),
        SourceReplayBatch(((1, experts, arena[4:].view(3, 2)),), capture.replay_id),
    ]
    prepared = PreparedStreamingTensors(
        _batch_factory=lambda: iter(batches), replay_id=capture.replay_id, transfer_metrics={}
    )
    calls = []
    installer._model.load_weights = lambda weights: calls.append(list(weights))

    installer.install_streaming(prepared)
    arena.zero_()

    # vLLM may keep any weight until its layer is complete, so none may view the reused arena.
    assert [[(name, tensor.tolist()) for name, tensor in call] for call in calls] == [
        [
            ("hf.weight", [[0.0, 1.0], [2.0, 3.0]]),
            ("hf.experts", [[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]),
        ]
    ]


def test_loader_install_never_hands_vllm_another_ranks_experts(installer):
    installer._convert_native_to_hf = lambda sources: {
        **{f"mlp.experts.{index}.weight": expert for index, expert in enumerate(sources.get("experts", ()))},
        **({"weight": sources["weight"]} if "weight" in sources else {}),
    }
    installer.capture(MANIFEST)
    sources = {"weight": torch.zeros(2, 2), "experts": torch.arange(6.0).view(3, 2)}

    installer.install_tensors(_staged(installer, sources))

    assert [(name, tensor.tolist()) for name, tensor in installer._model.loaded] == [
        ("mlp.experts.1.weight", [2.0, 3.0]),
        ("weight", [[0.0, 0.0], [0.0, 0.0]]),
    ]


def test_loader_install_runs_only_vllms_own_reload(installer, monkeypatch):
    # ModelExpress's own reload rebinds aliases and graph-bound tensors; vLLM's reload
    # already restores every kernel tensor in place, as NCCL refits rely on.
    installer.capture(MANIFEST)
    events = []
    layerwise = sys.modules[LAYERWISE]
    monkeypatch.setattr(layerwise, "initialize_layerwise_reload", lambda _model: events.append("initialize"))
    monkeypatch.setattr(layerwise, "finalize_layerwise_reload", lambda _model, _config: events.append("finalize"))
    monkeypatch.setattr(installer, "_reload", lambda _load: events.append("modelexpress reload"))
    installer._model.load_weights = lambda weights: events.append(("load", sorted(name for name, _ in weights)))
    sources = {"weight": torch.zeros(2, 2), "experts": torch.zeros(3, 2)}

    installer.install_tensors(_staged(installer, sources))

    assert events == ["initialize", ("load", ["hf.experts", "hf.weight"]), "finalize"]
