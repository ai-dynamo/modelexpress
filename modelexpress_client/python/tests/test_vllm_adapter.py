# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the vLLM engine adapter."""

import json
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from modelexpress.engines.vllm.adapter import (
    DraftShardSelection,
    VllmAdapter,
    _SAFETENSORS_INDEX_NAME,
    _get_vllm_device_id,
    _get_vllm_worker_rank,
    _read_safetensors_index,
    _select_draft_weight_files,
    build_vllm_load_context,
)
from modelexpress.load_strategy.context import LoadResult


def _vllm_config(*, rank: int, tp_size: int, pp_size: int):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            rank=rank,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
        )
    )


@pytest.fixture(autouse=True)
def _stub_accelerator_backend_selection(monkeypatch, mock_accelerator_backend_cls):
    monkeypatch.setattr(
        "modelexpress.engines.vllm.adapter.accelerator_backend_for",
        lambda device: mock_accelerator_backend_cls(),
    )


def test_worker_rank_uses_torch_distributed_global_rank():
    config = _vllm_config(rank=2, tp_size=4, pp_size=2)
    device = torch.device("cuda", 0)

    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_rank", return_value=6,
    ):
        assert _get_vllm_worker_rank(config, device) == 6


def test_worker_rank_distinguishes_dp_replicas():
    config = _vllm_config(rank=0, tp_size=4, pp_size=2)
    device = torch.device("cuda", 0)

    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_rank", return_value=5,
    ):
        dp0_rank = _get_vllm_worker_rank(config, device)

    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_rank", return_value=13,
    ):
        dp1_rank = _get_vllm_worker_rank(config, device)

    assert dp0_rank == 5
    assert dp1_rank == 13


def test_worker_rank_falls_back_to_parallel_config_rank_pre_init():
    # Pre-init / bare-cuda path: torch.distributed not initialised AND device
    # has no index. Falls back to parallel_config.rank so workers in the same
    # DP still get distinct keys.
    config = _vllm_config(rank=3, tp_size=4, pp_size=2)
    bare_device = torch.device("cuda")

    with patch("torch.distributed.is_initialized", return_value=False):
        assert _get_vllm_worker_rank(config, bare_device) == 3


def test_vllm_device_id_uses_current_platform_device(monkeypatch):
    fake_platforms = SimpleNamespace(
        current_platform=SimpleNamespace(
            current_device=lambda: 2,
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)

    assert _get_vllm_device_id(torch.device("cuda")) == 2


def test_vllm_is_cuda_alike_uses_current_platform(
    monkeypatch,
    mock_accelerator_backend_cls,
):
    fake_platforms = SimpleNamespace(
        current_platform=SimpleNamespace(
            is_cuda_alike=lambda: True,
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)
    monkeypatch.setattr(
        "modelexpress.engines.vllm.adapter.accelerator_backend_for",
        lambda device: mock_accelerator_backend_cls(),
    )
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())

    assert adapter.is_cuda_alike() is True


def test_vllm_adapter_discovery_uses_backend_predicate(
    monkeypatch,
    mock_accelerator_backend_cls,
):
    backend = mock_accelerator_backend_cls(torch_device_type="cpu")
    monkeypatch.setattr(
        "modelexpress.engines.vllm.adapter.accelerator_backend_for",
        lambda device: backend,
    )
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())
    model = nn.Module()
    model.weight = nn.Parameter(torch.randn(4, 3))

    tensors = adapter.discover_tensors(SimpleNamespace(model=model))

    assert list(tensors) == ["weight"]


def test_build_vllm_load_context_uses_current_platform_for_bare_cuda(monkeypatch):
    _stub_vllm_current_device(monkeypatch, current_device=2)
    _stub_metadata_client(monkeypatch)
    vllm_config = _context_config(load_device=None)

    ctx = build_vllm_load_context(vllm_config, _model_config())

    assert ctx.target_device == torch.device("cuda")
    assert ctx.target_device.index is None
    assert ctx.device_id == 2


def test_build_vllm_load_context_keeps_explicit_cuda_index(monkeypatch):
    _stub_vllm_current_device(monkeypatch, current_device=2)
    _stub_metadata_client(monkeypatch)
    vllm_config = _context_config(load_device="cuda:3")

    ctx = build_vllm_load_context(vllm_config, _model_config())

    assert ctx.target_device == torch.device("cuda:3")
    assert ctx.target_device.index == 3
    assert ctx.device_id == ctx.target_device.index


def test_build_vllm_load_context_uses_node_rank_as_node_rank(monkeypatch):
    _stub_metadata_client(monkeypatch)
    vllm_config = _context_config(load_device="cuda:0")
    vllm_config.parallel_config.node_rank = 1

    ctx = build_vllm_load_context(vllm_config, _model_config())

    assert ctx.node_rank == 1


def test_before_rdma_receive_runs_layout_finalizers(monkeypatch):
    events = []

    def process_weights_after_loading(model, model_config, target_device):
        events.append(("process", target_device))

    _stub_vllm_process_weights_after_loading(monkeypatch, process_weights_after_loading)
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())
    model = _TopLevelModel(events)

    result = adapter.before_rdma_receive(LoadResult(value=model, model=model))

    assert result.model is model
    assert events == [
        ("finalize", "model", "finalize_mega_moe_weights"),
        ("process", torch.device("cpu")),
    ]


def test_after_rdma_receive_runs_derived_weight_finalizers(monkeypatch):
    events = []
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())
    model = _TopLevelModel(events)

    result = adapter.after_rdma_receive(LoadResult(value=model, model=model))

    assert result.model is model
    assert events == [
        ("finalize", "model", "finalize_mhc_broadcast_weights"),
    ]


def test_after_rdma_receive_refreshes_host_attention_scale_mirrors(
    mock_accelerator_backend_cls,
):
    """RDMA accelerator scales replace every stale host mirror in place."""
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())
    adapter.accelerator_backend = mock_accelerator_backend_cls(
        torch_device_type="cpu"
    )
    model = nn.Module()
    model.attn = _AttentionWithStaleHostScales()
    pointers = {
        name: tensor.data_ptr()
        for name, tensor in model.attn.named_buffers(recurse=False)
    }

    result = adapter.after_rdma_receive(LoadResult(value=model, model=model))

    assert result.model is model
    assert model.attn._q_scale_float == pytest.approx(0.25)
    # Match compressed-tensors' host conversion for per-head scales.
    assert model.attn._k_scale_float == pytest.approx(0.5)
    assert model.attn._v_scale_float == pytest.approx(0.75)
    # vLLM does not derive a host mirror from _prob_scale.
    assert model.attn._prob_scale_float == pytest.approx(1.0)
    assert model.attn._k_scale_cpu.item() == pytest.approx(0.5)
    assert model.attn._v_scale_cpu.item() == pytest.approx(0.75)
    assert model.attn.impl.bmm1_scale is None
    assert model.attn.impl.bmm2_scale is None
    assert model.attn.impl.o_sf_scale is None
    assert {
        name: tensor.data_ptr()
        for name, tensor in model.attn.named_buffers(recurse=False)
    } == pointers


def test_after_rdma_receive_refreshes_scales_after_model_finalizer(
    mock_accelerator_backend_cls,
):
    """The mirror refresh observes scale changes made by the finalizer."""
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())
    adapter.accelerator_backend = mock_accelerator_backend_cls(
        torch_device_type="cpu"
    )
    model = _AttentionScaleFinalizerModel()

    adapter.after_rdma_receive(LoadResult(value=model, model=model))

    assert model.finalized is True
    assert model.attn._q_scale.item() == pytest.approx(0.625)
    assert model.attn._q_scale_float == pytest.approx(0.625)


def test_after_rdma_receive_fails_when_fp8_scales_are_unrecognized(
    mock_accelerator_backend_cls,
):
    """An FP8 target cannot silently complete without refreshing any scales."""
    config = _context_config(load_device="cpu")
    config.cache_config.cache_dtype = "fp8"
    adapter = VllmAdapter(config, _model_config())
    adapter.accelerator_backend = mock_accelerator_backend_cls(
        torch_device_type="cpu"
    )
    model = nn.Module()

    with pytest.raises(RuntimeError, match="no attention module was refreshed"):
        adapter.after_rdma_receive(LoadResult(value=model, model=model))


def test_after_rdma_receive_fails_on_incomplete_scale_contract(
    mock_accelerator_backend_cls,
):
    """A partial private q/k/v scale contract fails closed."""
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())
    adapter.accelerator_backend = mock_accelerator_backend_cls(
        torch_device_type="cpu"
    )
    model = nn.Module()
    model.attn = _AttentionWithStaleHostScales()
    del model.attn._v_scale_float

    with pytest.raises(RuntimeError, match="Incomplete.*_v_scale_float"):
        adapter.after_rdma_receive(LoadResult(value=model, model=model))


def test_after_rdma_receive_rejects_invalid_python_host_scalar(
    mock_accelerator_backend_cls,
):
    """A renamed or type-changed Python scale fails the private contract."""
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())
    adapter.accelerator_backend = mock_accelerator_backend_cls(
        torch_device_type="cpu"
    )
    model = nn.Module()
    model.attn = _AttentionWithStaleHostScales()
    model.attn._q_scale_float = torch.tensor(1.0)

    with pytest.raises(RuntimeError, match="_q_scale_float must be a float"):
        adapter.after_rdma_receive(LoadResult(value=model, model=model))


def test_after_rdma_receive_rejects_invalid_cpu_scale(
    mock_accelerator_backend_cls,
):
    """Host tensor mirrors must preserve vLLM's singleton CPU contract."""
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())
    adapter.accelerator_backend = mock_accelerator_backend_cls(
        torch_device_type="cpu"
    )
    model = nn.Module()
    model.attn = _AttentionWithStaleHostScales()
    model.attn._k_scale_cpu = torch.ones(2)

    with pytest.raises(RuntimeError, match="_k_scale_cpu must be"):
        adapter.after_rdma_receive(LoadResult(value=model, model=model))


def test_after_rdma_receive_allows_missing_backend_specific_cpu_scales(
    mock_accelerator_backend_cls,
):
    """Backends without CPU scale mirrors still refresh Python host state."""
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())
    adapter.accelerator_backend = mock_accelerator_backend_cls(
        torch_device_type="cpu"
    )
    model = nn.Module()
    model.attn = _AttentionWithStaleHostScales()
    del model.attn._k_scale_cpu
    del model.attn._v_scale_cpu

    adapter.after_rdma_receive(LoadResult(value=model, model=model))

    assert model.attn._q_scale_float == pytest.approx(0.25)
    assert model.attn._k_scale_float == pytest.approx(0.5)
    assert model.attn._v_scale_float == pytest.approx(0.75)


def test_after_rdma_receive_allows_mla_flashinfer_cache_contract(
    mock_accelerator_backend_cls,
):
    """FlashInfer MLA legitimately has bmm caches without an output cache."""
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())
    adapter.accelerator_backend = mock_accelerator_backend_cls(
        torch_device_type="cpu"
    )
    model = nn.Module()
    model.attn = _AttentionWithStaleHostScales()
    del model.attn.impl.o_sf_scale

    adapter.after_rdma_receive(LoadResult(value=model, model=model))

    assert model.attn._q_scale_float == pytest.approx(0.25)
    assert model.attn._k_scale_float == pytest.approx(0.5)
    assert model.attn._v_scale_float == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("scale_name", "invalid_value"),
    (
        ("_q_scale", torch.tensor([])),
        ("_k_scale", torch.tensor([float("nan")])),
        ("_v_scale", torch.tensor(0.0)),
    ),
)
def test_after_rdma_receive_rejects_invalid_accelerator_scales(
    mock_accelerator_backend_cls,
    scale_name,
    invalid_value,
):
    """Accelerator scales must be nonempty, finite, and positive."""
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())
    adapter.accelerator_backend = mock_accelerator_backend_cls(
        torch_device_type="cpu"
    )
    model = nn.Module()
    model.attn = _AttentionWithStaleHostScales()
    setattr(model.attn, scale_name, invalid_value)

    with pytest.raises(RuntimeError, match="Invalid vLLM accelerator attention scale"):
        adapter.after_rdma_receive(LoadResult(value=model, model=model))


@pytest.mark.parametrize(
    ("owner_name", "cache_name"),
    (
        ("module", "_o_scale_float"),
        ("impl", "bmm1_scale"),
        ("impl", "bmm2_scale"),
        ("impl", "o_sf_scale"),
    ),
)
def test_after_rdma_receive_rejects_prewarmed_attention_scale_cache(
    mock_accelerator_backend_cls,
    owner_name,
    cache_name,
):
    """The compatibility shim only supports vLLM's cold-load lifecycle."""
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())
    adapter.accelerator_backend = mock_accelerator_backend_cls(
        torch_device_type="cpu"
    )
    model = nn.Module()
    model.attn = _AttentionWithStaleHostScales()
    owner = model.attn if owner_name == "module" else model.attn.impl
    setattr(owner, cache_name, 99.0)

    with pytest.raises(RuntimeError, match="requires a cold-load state"):
        adapter.after_rdma_receive(LoadResult(value=model, model=model))


def test_rdma_lifecycle_discovers_prepared_tensors_and_finalizes_received_weights(
    monkeypatch,
    mock_accelerator_backend_cls,
):
    """RDMA discovers the pre-finalized layout before applying source weights."""
    events = []

    def process_weights_after_loading(model, model_config, target_device):
        events.append(("process", target_device))

    _stub_vllm_process_weights_after_loading(monkeypatch, process_weights_after_loading)
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())
    adapter.accelerator_backend = mock_accelerator_backend_cls(torch_device_type="cpu")
    model = _RdmaLifecycleTopLevelModel(events)
    result = LoadResult(value=model, model=model)

    result = adapter.before_rdma_receive(result)
    tensors = adapter.discover_tensors(result)
    events.append(("discover", sorted(tensors)))

    # Simulate RDMA applying the source tensor into the region registered above.
    tensors["model.hc_attn_fn"].fill_(7)
    events.append(("rdma_receive", 7))
    result = adapter.after_rdma_receive(result)

    assert result.model is model
    assert events == [
        ("finalize", "model", "finalize_mega_moe_weights"),
        ("process", torch.device("cpu")),
        ("discover", ["model.hc_attn_fn"]),
        ("rdma_receive", 7),
        ("finalize", "model", "finalize_mhc_broadcast_weights", 7),
    ]


def test_finalize_model_specific_weights_requires_explicit_names_and_model():
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())

    with pytest.raises(TypeError, match="finalizer_names"):
        adapter._finalize_model_specific_weights(LoadResult(value=object()))

    with pytest.raises(RuntimeError, match="RDMA post-load processing"):
        adapter._finalize_model_specific_weights(LoadResult(value=object()), ())


def test_after_weight_iter_load_does_not_rerun_model_specific_finalizers(monkeypatch):
    events = []

    def process_weights_after_loading(model, model_config, target_device):
        events.append(("process", target_device))

    _stub_vllm_process_weights_after_loading(monkeypatch, process_weights_after_loading)
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())
    model = _TopLevelModel(events)

    result = adapter.after_weight_iter_load(LoadResult(value=model, model=model))

    assert result.model is model
    assert events == [("process", torch.device("cpu"))]


def _stub_vllm_current_device(monkeypatch, *, current_device: int) -> None:
    fake_platforms = SimpleNamespace(
        current_platform=SimpleNamespace(
            current_device=lambda: current_device,
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)


def _stub_vllm_process_weights_after_loading(monkeypatch, process_fn) -> None:
    packages = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
    ]
    for name in packages:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    utils = ModuleType("vllm.model_executor.model_loader.utils")
    utils.process_weights_after_loading = process_fn
    monkeypatch.setitem(sys.modules, utils.__name__, utils)


def _stub_metadata_client(monkeypatch) -> None:
    monkeypatch.setattr(
        "modelexpress.engines.vllm.adapter.create_metadata_client",
        lambda worker_rank: object(),
    )


def _context_config(*, load_device):
    return SimpleNamespace(
        device_config=SimpleNamespace(device="cuda"),
        load_config=SimpleNamespace(device=load_device),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            rank=0,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        ),
    )


def _model_config():
    return SimpleNamespace(
        dtype=torch.bfloat16,
        model="test-model",
        quantization=None,
        revision=None,
    )


class _TopLevelModel(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.model = _MegaMoeModel(events)
        self.standalone = _StandaloneFinalizer(events)


class _MegaMoeModel(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.layer = _MegaMoeLayer(events)

    def finalize_mega_moe_weights(self) -> None:
        self.events.append(("finalize", "model", "finalize_mega_moe_weights"))

    def finalize_mhc_broadcast_weights(self) -> None:
        self.events.append(
            ("finalize", "model", "finalize_mhc_broadcast_weights")
        )


class _MegaMoeLayer(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def finalize_mega_moe_weights(self) -> None:
        self.events.append(("finalize", "layer", "finalize_mega_moe_weights"))


class _RdmaLifecycleTopLevelModel(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.model = _RdmaLifecycleModel(events)


class _RdmaLifecycleModel(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.register_buffer("hc_attn_fn", torch.zeros(1))

    def finalize_mega_moe_weights(self) -> None:
        self.events.append(("finalize", "model", "finalize_mega_moe_weights"))

    def finalize_mhc_broadcast_weights(self) -> None:
        self.events.append(
            (
                "finalize",
                "model",
                "finalize_mhc_broadcast_weights",
                int(self.hc_attn_fn.item()),
            )
        )


class _AttentionWithStaleHostScales(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_q_scale", torch.tensor(0.25))
        self.register_buffer("_k_scale", torch.tensor([0.1, 0.5]))
        self.register_buffer("_v_scale", torch.tensor(0.75))
        self.register_buffer("_prob_scale", torch.tensor(0.125))
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0
        self._prob_scale_float = 1.0
        self._k_scale_cpu = torch.tensor(1.0)
        self._v_scale_cpu = torch.tensor(1.0)
        self._o_scale_float = None
        self.kv_cache_dtype = "fp8"
        self.impl = SimpleNamespace(
            bmm1_scale=None,
            bmm2_scale=None,
            o_sf_scale=None,
        )


class _AttentionScaleFinalizerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.finalized = False
        self.attn = _AttentionWithStaleHostScales()

    def finalize_mhc_broadcast_weights(self) -> None:
        self.attn._q_scale.fill_(0.625)
        self.finalized = True


class _StandaloneFinalizer(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def finalize_weights(self) -> None:
        self.events.append(("finalize", "standalone", "finalize_weights"))

    def finalize_weight(self) -> None:
        self.events.append(("finalize", "standalone", "finalize_weight"))

    def post_load_weights(self) -> None:
        self.events.append(("finalize", "standalone", "post_load_weights"))

    def finalize_cache(self) -> None:
        self.events.append(("finalize", "standalone", "finalize_cache"))

    def finalize_cache_weights(self) -> None:
        self.events.append(("finalize", "standalone", "finalize_cache_weights"))

    def finalize_requires_arg_weights(self, context) -> None:
        self.events.append(
            ("finalize", "standalone", "finalize_requires_arg_weights", context)
        )


class TestDraftWeightFileSelection:
    """A draft load streams only its own shards, and falls back to the full
    set when the checkpoint has no draft head."""

    def _write_index(self, tmp_path, weight_map):
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )

    def test_selects_only_mtp_shard(self, tmp_path):
        self._write_index(
            tmp_path,
            {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
                "mtp.fc.weight": "model-mtp.safetensors",
            },
        )
        files = [
            os.path.join(str(tmp_path), name)
            for name in (
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model-mtp.safetensors",
            )
        ]
        assert _select_draft_weight_files(str(tmp_path), files) == (
            DraftShardSelection.SELECTED,
            [os.path.join(str(tmp_path), "model-mtp.safetensors")],
        )

    def test_selects_mtp_shard_for_hf_repo_id(self, tmp_path):
        # model_uri is an HF repo id that only _prepare_weights can resolve;
        # the index has to be found next to the shards it returned.
        self._write_index(
            tmp_path,
            {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
                "mtp.fc.weight": "model-mtp.safetensors",
            },
        )
        files = [
            os.path.join(str(tmp_path), name)
            for name in (
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model-mtp.safetensors",
            )
        ]
        assert _select_draft_weight_files("Qwen/Qwen3.5-27B", files) == (
            DraftShardSelection.SELECTED,
            [os.path.join(str(tmp_path), "model-mtp.safetensors")],
        )

    def test_falls_back_without_draft_head(self, tmp_path):
        self._write_index(
            tmp_path, {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"}
        )
        files = [os.path.join(str(tmp_path), "model-00001-of-00001.safetensors")]
        assert _select_draft_weight_files(str(tmp_path), files) == (
            DraftShardSelection.NO_DRAFT_WEIGHTS,
            [],
        )

    def test_corrupt_index_reports_unresolved(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text(
            "{not json", encoding="utf-8"
        )
        files = [os.path.join(str(tmp_path), "model-00001-of-00001.safetensors")]
        assert _select_draft_weight_files("Qwen/Qwen3.5-27B", files) == (
            DraftShardSelection.UNRESOLVED,
            [],
        )

    def test_unreadable_index_reports_unresolved(self, tmp_path):
        files = [os.path.join(str(tmp_path), "model-00001-of-00001.safetensors")]
        with patch(
            "modelexpress.engines.vllm.adapter._read_safetensors_index",
            return_value=None,
        ):
            assert _select_draft_weight_files("Qwen/Qwen3.5-27B", files) == (
                DraftShardSelection.UNRESOLVED,
                [],
            )


def _stub_runai(monkeypatch, available: dict[str, str]) -> list:
    """Install a fake runai_model_streamer whose pull_files mirrors runai's
    real semantics: allow_pattern is fnmatched against the full object key, so
    an unanchored bare filename matches nothing. Returns the list of
    allow_pattern values it was called with.

    ``available`` maps object basenames (relative to model_uri) to file content.
    """
    import fnmatch

    calls: list = []

    def pull_files(model_uri, dest, allow_pattern=None):
        calls.append(allow_pattern)
        for key, content in available.items():
            full_key = f"{model_uri.rstrip('/')}/{key}"
            if any(fnmatch.fnmatch(full_key, pat) for pat in (allow_pattern or ["*"])):
                with open(os.path.join(dest, key), "w", encoding="utf-8") as handle:
                    handle.write(content)

    module = ModuleType("runai_model_streamer")
    module.pull_files = pull_files
    monkeypatch.setitem(sys.modules, "runai_model_streamer", module)
    return calls


class TestReadSafetensorsIndexObjectStore:
    """Reading the index from an object store depends on runai's glob matching
    the full object key; the bare filename this once used matched nothing."""

    def test_reads_index_via_anchored_glob(self, monkeypatch):
        index = {"weight_map": {"mtp.fc.weight": "model-mtp.safetensors"}}
        calls = _stub_runai(
            monkeypatch,
            {
                _SAFETENSORS_INDEX_NAME: json.dumps(index),
                "model-00001-of-00001.safetensors": "weights",
            },
        )
        # Reverting to a bare, unanchored pattern makes the fake fnmatch miss,
        # so this returns None and the assertion fails, as it should.
        assert _read_safetensors_index("s3://bucket/model") == index
        assert calls == [[f"*{_SAFETENSORS_INDEX_NAME}"]]

    def test_returns_none_and_warns_when_index_absent(self, monkeypatch, caplog):
        _stub_runai(monkeypatch, {"model-00001-of-00001.safetensors": "weights"})
        with caplog.at_level("WARNING", logger="modelexpress.engines.vllm.adapter"):
            assert _read_safetensors_index("s3://bucket/model") is None
        assert any("not found under" in rec.message for rec in caplog.records)

    def test_selects_mtp_shard_from_object_store(self, monkeypatch):
        index = {
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
                "mtp.fc.weight": "model-mtp.safetensors",
                "mtp.layers.0.input_layernorm.weight": "model-mtp.safetensors",
            }
        }
        _stub_runai(monkeypatch, {_SAFETENSORS_INDEX_NAME: json.dumps(index)})
        files = [
            f"s3://bucket/model/{name}"
            for name in (
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model-mtp.safetensors",
            )
        ]
        assert _select_draft_weight_files("s3://bucket/model", files) == (
            DraftShardSelection.SELECTED,
            ["s3://bucket/model/model-mtp.safetensors"],
        )
