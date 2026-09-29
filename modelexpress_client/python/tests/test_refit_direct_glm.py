# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM plumbing and focused MoE guards; synthetic fixtures are not engine admission."""

import sys
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest
import torch
from modelexpress_rl.inference.engines.vllm import direct_glm as glm
from modelexpress_rl.inference.engines.vllm.direct_copy import (
    _DirectDestination,
    _geometry,
)
from modelexpress_rl.inference.engines.vllm.installer import _VllmInstaller
from modelexpress_rl.inference.plan import PreparedStreamingTensors


@pytest.mark.parametrize("registered", ["reviewed", "replaced"])
def test_profile_checks_selected_registry_class(monkeypatch, registered):
    reviewed = type("ReviewedGlm", (), {})
    replacement = type("NewGlm", (), {})
    selected = reviewed if registered == "reviewed" else replacement
    fake_module = SimpleNamespace(ReviewedGlm=reviewed)
    registry_module = ModuleType("vllm.model_executor.models")
    registry_module.ModelRegistry = SimpleNamespace(
        resolve_model_cls=lambda architecture, config: (selected, architecture)
    )
    monkeypatch.setitem(sys.modules, registry_module.__name__, registry_module)
    monkeypatch.setattr(glm, "SOURCE_HASHES", {})
    monkeypatch.setattr(glm, "MODULE_COUNTS", {"reviewed.ReviewedGlm": 1})
    monkeypatch.setattr(glm, "MODEL_CLASS", "reviewed.ReviewedGlm")
    monkeypatch.setattr(glm.importlib, "import_module", lambda name: fake_module)
    glm._profile_classes.cache_clear()
    try:
        if registered == "reviewed":
            assert glm._profile_classes() == {reviewed: "reviewed.ReviewedGlm"}
        else:
            with pytest.raises(ValueError, match="registry selects an unreviewed"):
                glm._profile_classes()
    finally:
        glm._profile_classes.cache_clear()


def fixture(monkeypatch):
    model = torch.nn.Linear(2, 2)
    events = []
    destinations = tuple(
        _DirectDestination(n, p, _geometry(p, destination=True))
        for n, p in model.named_parameters()
    )
    refresh = SimpleNamespace(refresh=lambda: events.append("refresh"))
    signature = tuple((d.name, d.geometry) for d in destinations)
    monkeypatch.setattr(
        glm, "_inspect", lambda m, c: (destinations, (refresh,), signature)
    )
    config = SimpleNamespace()
    values = {n: torch.full_like(p, 7) for n, p in model.named_parameters()}

    def batches():
        try:
            for name, value in values.items():
                events.append("read:" + name)
                yield {name: value}
        finally:
            events.append("close")

    source = PreparedStreamingTensors(batches, frozenset(values), {})
    arguments = {
        "version_id": "v:1",
        "source": source,
        "batch_names": tuple(frozenset([n]) for n in values),
        "parameter_layout": {
            n: (tuple(p.shape), p.dtype) for n, p in model.named_parameters()
        },
    }
    return model, config, arguments, values, events


def test_selected_glm_path_copies_then_refreshes_without_reload(monkeypatch):
    model, config, arguments, values, events = fixture(monkeypatch)
    monkeypatch.setenv("MX_REFIT_GLM_DIRECT", "1")
    installer = _VllmInstaller(
        model=model, vllm_config=config, model_config=None, device=torch.device("cpu")
    )
    monkeypatch.setattr(
        installer, "install_streaming", lambda p: pytest.fail("generic reload entered")
    )
    addresses = {n: (id(p), p.data_ptr()) for n, p in model.named_parameters()}
    for version in range(3):
        for value in values.values():
            value.add_(1)
        events.clear()
        selected = installer.prepare_streaming_artifact(
            version=SimpleNamespace(version_id=f"v:{version}"),
            staging_device="cuda",
            staging_buffers=1,
            **{k: v for k, v in arguments.items() if k != "version_id"},
        )
        assert not events
        metrics = installer.install(selected)
        assert events == ["read:weight", "read:bias", "close", "refresh"]
        assert metrics["glm_direct_install"] == 1
        assert metrics["retention_batch_scans"] == 0
        assert selected.ownership.iterator is None
        assert {
            n: (id(p), p.data_ptr()) for n, p in model.named_parameters()
        } == addresses
        assert all(torch.equal(p, values[n]) for n, p in model.named_parameters())


@pytest.mark.parametrize("defect", ["dtype", "shape", "missing", "duplicate", "extra"])
def test_glm_layout_rejected_before_read(monkeypatch, defect):
    model, config, arguments, _, events = fixture(monkeypatch)
    if defect == "dtype":
        arguments["parameter_layout"]["weight"] = ((2, 2), torch.float16)
    elif defect == "shape":
        arguments["parameter_layout"]["weight"] = ((4,), torch.float32)
    elif defect == "missing":
        arguments["batch_names"] = arguments["batch_names"][:-1]
    elif defect == "duplicate":
        arguments["batch_names"] += (arguments["batch_names"][0],)
    else:
        arguments["batch_names"] += (frozenset(["unknown"]),)
    with pytest.raises(ValueError):
        glm.prepare_glm_direct(model, config, **arguments)
    assert not events


def test_glm_binding_change_rejected_before_read(monkeypatch):
    model, config, arguments, _, events = fixture(monkeypatch)
    selected = glm.prepare_glm_direct(model, config, **arguments)
    monkeypatch.setattr(glm, "_inspect", lambda m, c: ((), (), ("changed",)))
    with pytest.raises(ValueError, match="bindings changed"):
        glm.install_glm_direct(selected, model=model)
    assert not events


def test_glm_refresh_failure_closes_proven_source(monkeypatch):
    model, config, arguments, _, events = fixture(monkeypatch)
    selected = glm.prepare_glm_direct(model, config, **arguments)

    def failure():
        raise RuntimeError("refresh failed")

    selected.plan.mla[0].refresh = failure
    with pytest.raises(RuntimeError, match="refresh failed"):
        glm.install_glm_direct(selected, model=model)
    assert selected.ownership.iterator is None
    assert not selected.ownership.release_blocked
    assert "glm_direct_install" not in selected.metrics
    assert events[-1] == "close"


def test_glm_refresh_drain_failure_retains_ownership(monkeypatch):
    model, config, arguments, _, events = fixture(monkeypatch)
    selected = glm.prepare_glm_direct(model, config, **arguments)
    calls = 0

    def drain(plan):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("refresh drain failed")

    monkeypatch.setattr(glm, "_drain", drain)
    with pytest.raises(RuntimeError, match="refresh drain failed"):
        glm.install_glm_direct(selected, model=model)
    assert calls == 2
    assert "refresh" in events
    assert selected.ownership.drain_failed
    assert selected.ownership.release_blocked
    assert selected.ownership.iterator is not None
    assert "glm_direct_install" not in selected.metrics


def moe_fixture(monkeypatch):
    prefix = "vllm.model_executor.layers.fused_moe."

    def helper(relative):
        module_name, symbol = (prefix + relative).rsplit(".", 1)
        if module_name not in sys.modules:
            monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
        cls = type(symbol, (), {})
        monkeypatch.setattr(sys.modules[module_name], symbol, cls, raising=False)
        return cls()

    backend_module = ModuleType(prefix + "oracle.unquantized")
    backend_module.UnquantizedMoeBackend = Enum("UnquantizedMoeBackend", ["TRITON"])
    monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)
    method = helper("unquantized_fused_moe_method.UnquantizedFusedMoEMethod")
    method.unquantized_backend = backend_module.UnquantizedMoeBackend.TRITON
    method.moe = SimpleNamespace(
        has_bias=False,
        is_lora_enabled=False,
        moe_parallel_config=SimpleNamespace(enable_eplb=False),
    )
    quant = helper("config.FusedMoEQuantConfig")
    descriptor = helper("config.FusedMoEQuantDesc")
    for key in ("dtype", "scale", "zp", "bias", "alpha_or_gscale"):
        setattr(descriptor, key, None)
    for name in ("_a1", "_a2", "_w1", "_w2"):
        setattr(quant, name, descriptor)
    method.moe_quant_config = quant
    kernel = helper("modular_kernel.FusedMoEKernel")
    impl = helper("modular_kernel.FusedMoEKernelModularImpl")
    experts = helper("experts.triton_moe.TritonExperts")
    experts.quant_config = quant
    experts._lora_context = None
    experts.quantization_emulation = False
    prepare = helper("prepare_finalize.no_dp_ep.MoEPrepareAndFinalizeNoDPEPModular")
    impl.fused_experts = experts
    impl.prepare_finalize = prepare
    kernel.impl = impl
    method.moe_kernel = kernel
    return method, {
        "kernel": kernel,
        "impl": impl,
        "experts": experts,
        "prepare_finalize": prepare,
    }


def test_modular_moe_admits_absent_replay_state(monkeypatch):
    method, _ = moe_fixture(monkeypatch)
    glm._check_moe(method)


@pytest.mark.parametrize("owner", ["kernel", "impl", "experts", "prepare_finalize"])
@pytest.mark.parametrize(
    "field", ["routing_replay_capture_fn", "_routing_replay_buffer"]
)
@pytest.mark.parametrize("value", [None, object()])
def test_modular_moe_rejects_added_replay_state(monkeypatch, owner, field, value):
    method, owners = moe_fixture(monkeypatch)
    setattr(owners[owner], field, value)
    with pytest.raises(ValueError, match="unexpected modular MoE routing replay state"):
        glm._check_moe(method)


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        ("attention_config", "hisparse_config", object(), "HiSparse"),
        ("parallel_config", "enable_elastic_ep", True, "elastic"),
        ("parallel_config", "elastic_ep_max_dp_size", 2, "elastic"),
        ("kernel_config", "sparse_indexer_topk_backend", "torch", "indexer backend"),
    ],
)
def test_vllm030_rejects_unreviewed_configuration(section, field, value, message):
    config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=None),
        parallel_config=SimpleNamespace(
            enable_elastic_ep=False, elastic_ep_max_dp_size=1
        ),
        kernel_config=SimpleNamespace(sparse_indexer_topk_backend="auto"),
    )
    glm._check_vllm030_configuration(config)
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match=message):
        glm._check_vllm030_configuration(config)


def sparse_attention_fixture(monkeypatch):
    name = "vllm.v1.attention.backends.mla.index_group"
    group_module = ModuleType(name)
    group_class = type("SparseMLAIndexGroup", (), {})
    group_module.SparseMLAIndexGroup = group_class
    monkeypatch.setitem(sys.modules, name, group_module)
    group = group_class()
    group.logical_topk_indices = torch.empty((2, 4), dtype=torch.int32)
    group.has_indexer = True
    group.num_layers = 1
    module = type("DeepseekV32Attention", (), {})()
    module.hisparse_cache = None
    module.topk_indices_buffer = group.logical_topk_indices
    module.indexer = SimpleNamespace(
        topk_indices_buffer=group.logical_topk_indices,
        indexer_op=SimpleNamespace(topk_indices_buffer=group.logical_topk_indices),
    )
    module.impl = SimpleNamespace(
        index_group=group,
        index_group_index=0,
        topk_indices_buffer=group.logical_topk_indices,
    )
    return module, group


@pytest.mark.parametrize(
    "defect",
    [
        "hisparse",
        "group_type",
        "layer_index",
        "no_indexer",
        "multi_layer",
        "attention_buffer",
        "indexer_buffer",
        "indexer_op_buffer",
        "backend_buffer",
    ],
)
def test_vllm030_rejects_unreviewed_sparse_attention(monkeypatch, defect):
    module, group = sparse_attention_fixture(monkeypatch)
    glm._check_vllm030_module(module)
    if defect == "hisparse":
        module.hisparse_cache = object()
    elif defect == "group_type":
        module.impl.index_group = type("HiSparseMLAIndexGroup", (type(group),), {})()
    elif defect == "layer_index":
        module.impl.index_group_index = 1
    elif defect == "no_indexer":
        group.has_indexer = False
    elif defect == "multi_layer":
        group.num_layers = 2
    elif defect == "attention_buffer":
        module.topk_indices_buffer = group.logical_topk_indices.clone()
    elif defect == "indexer_buffer":
        module.indexer.topk_indices_buffer = group.logical_topk_indices.clone()
    elif defect == "indexer_op_buffer":
        module.indexer.indexer_op.topk_indices_buffer = (
            group.logical_topk_indices.clone()
        )
    else:
        module.impl.topk_indices_buffer = group.logical_topk_indices.clone()
    with pytest.raises(ValueError):
        glm._check_vllm030_module(module)


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_blocks", torch.zeros(1, dtype=torch.int32)),
        ("candidate_block_size", 1),
        ("candidate_write", True),
        ("compress_ratio", 2),
        ("use_fp4_cache", True),
        ("skip_k_cache_insert", True),
        ("topk_backend", "torch"),
    ],
)
def test_vllm030_rejects_unreviewed_sparse_indexer(field, value):
    module = type("SparseAttnIndexer", (), {})()
    module.candidate_blocks = None
    module.candidate_block_size = 0
    module.candidate_write = False
    module.compress_ratio = 1
    module.use_fp4_cache = False
    module.skip_k_cache_insert = False
    module.topk_backend = "auto"
    glm._check_vllm030_module(module)
    setattr(module, field, value)
    with pytest.raises(ValueError, match="alternate sparse indexer"):
        glm._check_vllm030_module(module)


@pytest.mark.parametrize("name", ["VocabParallelEmbedding", "ParallelLMHead"])
def test_vllm030_rejects_embedding_group_override(name):
    module = type(name, (), {})()
    module.parallel_group = None
    glm._check_vllm030_module(module)
    module.parallel_group = object()
    with pytest.raises(ValueError, match="embedding parallel group"):
        glm._check_vllm030_module(module)


def test_vllm030_rejects_elastic_moe_capacity():
    module = type("UnquantizedFusedMoEMethod", (), {})()
    module.moe = SimpleNamespace(elastic_ep_max_dp_size=1)
    glm._check_vllm030_module(module)
    module.moe.elastic_ep_max_dp_size = 2
    with pytest.raises(ValueError, match="elastic MoE"):
        glm._check_vllm030_module(module)


@pytest.mark.parametrize(
    "version,message",
    [
        ("0.29.0", "GLM module census differs"),
        ("0.30.0", "alternate sparse indexer representation"),
    ],
)
def test_version_specific_admission_rejects_before_read(monkeypatch, version, message):
    model = torch.nn.Module()
    indexer_type = type("SparseAttnIndexer", (torch.nn.Module,), {})
    model.indexer = indexer_type()
    model.indexer.candidate_blocks = torch.zeros(1, dtype=torch.int32)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="glm_moe_dsa"),
            dtype=torch.bfloat16,
            enforce_eager=True,
        ),
        quant_config=None,
        lora_config=None,
        speculative_config=None,
        attention_config=SimpleNamespace(hisparse_config=None),
        kernel_config=SimpleNamespace(sparse_indexer_topk_backend="auto"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=32,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_eplb=False,
            enable_dbo=False,
            use_sequence_parallel_moe=False,
            enable_expert_parallel=True,
            enable_elastic_ep=False,
            elastic_ep_max_dp_size=1,
        ),
    )
    monkeypatch.setattr(glm, "VLLM_VERSION", version)
    monkeypatch.setattr(
        glm,
        "_profile_classes",
        lambda: {type(model): glm.MODEL_CLASS, indexer_type: "test.SparseAttnIndexer"},
    )
    opened = []

    def batches():
        opened.append(True)
        yield {}

    with pytest.raises(ValueError, match=message):
        glm.prepare_glm_direct(
            model,
            config,
            version_id="test-version",
            source=PreparedStreamingTensors(batches, frozenset(), {}),
            batch_names=(),
            parameter_layout={},
        )
    assert not opened
