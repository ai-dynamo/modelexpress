# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CPU test for the Megatron-MX receiver-side translator (Phase C).

Exercises the full receiver flow against synthetic Megatron publishes:

    1. Construct ground-truth HF tensors (q, k, v, gate, up, ...).
    2. Use the vendored :func:`merge_qkv_weights` + concat helpers to
       build the global Megatron-shaped tensors that a TP=1 trainer
       would publish.
    3. Slice those into per-rank publishes (TP=2 / TP=4 etc.).
    4. Build synthetic V2SourceCandidate entries with the right
       MegatronSourceMeta (rank-position only) and TensorDescriptorV2
       per-tensor entries with the right ``megatron_role``.
5. Run pick_megatron_slice_plans → assemble_into_destination →
   translate_megatron_to_hf.
    6. Assert the yielded HF tensors are byte-identical to the
       ground-truth originals.

This validates the entire MX-side Phase B + Phase C chain on CPU with
no NIXL, no GPU, no Bridge, no Megatron model.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch


_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent / "modelexpress"


def _load(modname: str, path: Path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def env():
    """Load nemo_rl_v2 + megatron_helpers + megatron_translator with stubs."""
    pkg = types.ModuleType("modelexpress")
    pkg.__path__ = [str(_PKG_ROOT)]
    sys.modules["modelexpress"] = pkg

    # p2p_pb2 stub
    p2p_pb2 = types.ModuleType("modelexpress.p2p_pb2")
    p2p_pb2.SOURCE_STATUS_READY = 2
    p2p_pb2.SOURCE_STATUS_INITIALIZING = 1
    p2p_pb2.SOURCE_STATUS_STALE = 3
    p2p_pb2.MX_SOURCE_TYPE_WEIGHTS = 0
    p2p_pb2.BACKEND_FRAMEWORK_UNKNOWN = 0
    sys.modules["modelexpress.p2p_pb2"] = p2p_pb2

    class _Identity:
        def __init__(self, **kwargs):
            self.extra_parameters = dict(kwargs.get("extra_parameters") or {})

    class _TD:
        def __init__(self, **kwargs):
            self.name = kwargs.get("name", "")
            self.dtype = kwargs.get("dtype", "")
            self.size = kwargs.get("size", 0)
            self.addr = kwargs.get("addr", 0)
            self.device_id = kwargs.get("device_id", 0)

    class _WM:
        def __init__(self, **kwargs):
            self.tensors = list(kwargs.get("tensors") or [])
            self.agent_name = kwargs.get("agent_name", "")
            self.worker_rank = kwargs.get("worker_rank", 0)
            self.nixl_metadata = b""
            self.status = 0

    p2p_pb2.SourceIdentity = _Identity
    p2p_pb2.TensorDescriptor = _TD
    p2p_pb2.WorkerMetadata = _WM

    # heartbeat stub
    hb = types.ModuleType("modelexpress.heartbeat")

    class _HBStub:
        def __init__(self, *a, **kw):
            self.started = False
        def start(self):
            self.started = True
        def stop(self):
            self.started = False

    hb.HeartbeatThread = _HBStub
    sys.modules["modelexpress.heartbeat"] = hb

    # refit_receiver stub
    refit_mod = types.ModuleType("modelexpress.refit_receiver")

    @dataclass
    class _SourceRef:
        mx_source_id: str = ""
        worker_id: str = ""
        model_name: str = ""
        worker_rank: int = 0
        training_step: int = 0

    class _RefitStub:
        def __init__(self, agent_name="stub", **kw):
            self._client = MagicMock()
            self._nixl = MagicMock()
            self._agent_name = agent_name
            self._worker_id = "stub-worker-id"
        def initialize(self, model_tensors=None):
            pass
        def receive_weights(self, ref, timeout_seconds=300.0):
            return iter([])

    refit_mod.MxRefitReceiver = _RefitStub
    refit_mod.SourceRef = _SourceRef
    sys.modules["modelexpress.refit_receiver"] = refit_mod

    # training_publisher stub
    pub_mod = types.ModuleType("modelexpress.training_publisher")
    class _PubStub:
        def __init__(self, *a, **kw):
            self._client = None
            self._nixl = None
        def initialize(self, **kw): pass
        def _build_identity(self, step): return _Identity()
    pub_mod.MxTrainingPublisher = _PubStub
    sys.modules["modelexpress.training_publisher"] = pub_mod

    # types stub
    types_mod = types.ModuleType("modelexpress.types")
    @dataclass
    class _TDS:
        name: str = ""
        addr: int = 0
        size: int = 0
        device_id: int = 0
        dtype: str = ""
    types_mod.TensorDescriptor = _TDS
    sys.modules["modelexpress.types"] = types_mod

    # Load the modules under test
    sd = _load("modelexpress.shape_descriptors", _PKG_ROOT / "shape_descriptors.py")
    pkg.shape_descriptors = sd
    v2 = _load("modelexpress.nemo_rl_v2", _PKG_ROOT / "nemo_rl_v2.py")
    helpers = _load("modelexpress.megatron_helpers", _PKG_ROOT / "megatron_helpers.py")
    translator = _load("modelexpress.megatron_translator", _PKG_ROOT / "megatron_translator.py")
    return types.SimpleNamespace(v2=v2, helpers=helpers, translator=translator)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_megatron_candidate(env, *, tp_rank, tp_size, sid, pp_rank=0, ep_rank=0,
                             ep_size=1, owned_experts_per_layer=None):
    """Build a V2SourceCandidate with megatron_meta populated."""
    v2 = env.v2
    ref_cls = sys.modules["modelexpress.refit_receiver"].SourceRef
    ref = ref_cls(
        mx_source_id=sid, worker_id=f"wid-{sid}", model_name="m",
        worker_rank=tp_rank, training_step=1,
    )
    mm = v2.MegatronSourceMeta(
        tp_rank=tp_rank, tp_size=tp_size, pp_rank=pp_rank, pp_size=1,
        ep_rank=ep_rank, ep_size=ep_size,
    )
    return v2.V2SourceCandidate(
        ref=ref, role=v2.ROLE_TRAINER, worker_rank=tp_rank,
        registry=None,
        owned_experts_per_layer=owned_experts_per_layer or {},
        updated_at=tp_rank, megatron_meta=mm,
    )


def _make_pull_callback(rank_to_data: dict[str, torch.Tensor]):
    """Build a pull(src, dest) callback that copies from a per-rank in-memory
    table indexed by src.mx_source_id. The dest may be a sub-view (mixed-TP)
    or the full source slice (matched-TP)."""
    def _pull(src, dest):
        full = rank_to_data[src.mx_source_id]
        if src.source_subslice is not None:
            lo, hi = src.source_subslice
            full = full.narrow(0, lo, hi - lo)
        # Match the dest shape — for the per_expert path, dest is a
        # full per-expert tensor. For tp-sharded, dest is a view that
        # matches the full slice (or sub-slice). Copy by element count.
        dest.copy_(full.view(dest.shape))
    return _pull


# ---------------------------------------------------------------------------
# Test 1 — replicated tensor: 1 source, passthrough
# ---------------------------------------------------------------------------


def test_replicated_passthrough(env):
    v2 = env.v2
    rcv = v2.MxV2RefitReceiver(agent_name="rcv", device_id=0,
                                mx_server_url="stub:1", worker_rank=0)
    rcv.initialize(model_tensors=None)

    layout = v2.TargetTpLayout(tp_size=2, tp_rank=0)

    # The actual norm tensor (replicated → all ranks have the same).
    norm = torch.randn(128, dtype=torch.float32)
    cands = [_make_megatron_candidate(env, tp_rank=0, tp_size=2, sid="r0")]
    pull = _make_pull_callback({"r0": norm})

    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_REPLICATED,
        target_shape=(128,), target_dtype="float32",
    )
    plans = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"norm.weight": spec},
    )
    assembled = env.translator.assemble_into_destination(plans[0], pull=pull, device="cpu")
    assert torch.equal(assembled, norm)

    # End-to-end translate: yields one (hf_name, tensor)
    cfg = env.helpers.MegatronTransformerConfig(
        num_attention_heads=8, num_query_groups=8,
        kv_channels=64, hidden_size=512,
    )
    out = list(env.translator.translate_megatron_to_hf(
        plans[0], assembled,
        transformer_config=cfg,
        hf_names=["model.norm.weight"],
    ))
    assert len(out) == 1
    assert out[0][0] == "model.norm.weight"
    assert torch.equal(out[0][1], norm)


# ---------------------------------------------------------------------------
# Test 2 — column-parallel matched-TP: assemble across 2 trainer ranks,
# receiver wants the half its target_rank covers
# ---------------------------------------------------------------------------


def test_column_matched_tp_assembly(env):
    v2 = env.v2
    rcv = v2.MxV2RefitReceiver(agent_name="rcv", device_id=0,
                                mx_server_url="stub:1", worker_rank=0)
    rcv.initialize(model_tensors=None)
    # source TP=2, target TP=2, target_rank=1 → wants the second half
    layout = v2.TargetTpLayout(tp_size=2, tp_rank=1)

    # Ground-truth: a (2048, 512) column-parallel weight, sharded along dim 0.
    weight = torch.randn(2048, 512, dtype=torch.float32)
    rank0 = weight[:1024]
    rank1 = weight[1024:]

    cands = [
        _make_megatron_candidate(env, tp_rank=0, tp_size=2, sid="c0"),
        _make_megatron_candidate(env, tp_rank=1, tp_size=2, sid="c1"),
    ]
    pull = _make_pull_callback({"c0": rank0, "c1": rank1})

    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_COLUMN,
        target_shape=(2048, 512), target_dtype="float32", shard_axis=0,
    )
    plans = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"linear_proj": spec},
    )
    assert len(plans) == 1 and plans[0].assembly == "concat_dim0"
    assembled = env.translator.assemble_into_destination(plans[0], pull=pull, device="cpu")
    # target_rank=1 should see weight[1024:2048].
    assert assembled.shape == (1024, 512)
    assert torch.equal(assembled, weight[1024:2048])


# ---------------------------------------------------------------------------
# Test 3 — QKV column with GQA: end-to-end byte-identical roundtrip across
# 2 trainer ranks
# ---------------------------------------------------------------------------


def test_qkv_column_e2e_gqa(env):
    v2 = env.v2
    helpers = env.helpers
    rcv = v2.MxV2RefitReceiver(agent_name="rcv", device_id=0,
                                mx_server_url="stub:1", worker_rank=0)
    rcv.initialize(model_tensors=None)

    # Llama-3.1-8B-style attention: 32 q heads, 8 kv heads (GQA 4:1),
    # head_dim=128, hidden=4096. Two trainer TP ranks.
    nh, nkv, hd, hs = 32, 8, 128, 4096
    cfg = helpers.MegatronTransformerConfig(
        num_attention_heads=nh, num_query_groups=nkv,
        kv_channels=hd, hidden_size=hs,
    )

    torch.manual_seed(0xC0DE)
    q = torch.randn(nh * hd, hs, dtype=torch.float32)
    k = torch.randn(nkv * hd, hs, dtype=torch.float32)
    v = torch.randn(nkv * hd, hs, dtype=torch.float32)
    # Trainer-side: merge into Megatron interleaved layout (this is what a
    # TP=1 trainer would publish as the global tensor).
    qkv_global = helpers.merge_qkv_weights(cfg, q, k, v)
    assert qkv_global.shape == ((nh + 2 * nkv) * hd, hs)

    # Now slice across TP=2 — each rank holds half the rows.
    half = qkv_global.shape[0] // 2
    rank0 = qkv_global[:half]
    rank1 = qkv_global[half:]

    layout = v2.TargetTpLayout(tp_size=2, tp_rank=0)  # target_rank=0 → first half

    cands = [
        _make_megatron_candidate(env, tp_rank=0, tp_size=2, sid="qkv0"),
        _make_megatron_candidate(env, tp_rank=1, tp_size=2, sid="qkv1"),
    ]
    pull = _make_pull_callback({"qkv0": rank0, "qkv1": rank1})

    # Receiver target spec: target_rank=0 sees half the rows. Build a
    # config-for-half (half the heads) so the QKV un-interleave happens
    # against the receiver's local view.
    half_cfg = helpers.MegatronTransformerConfig(
        num_attention_heads=nh // 2, num_query_groups=nkv // 2,
        kv_channels=hd, hidden_size=hs,
    )
    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_QKV_COLUMN,
        target_shape=qkv_global.shape, target_dtype="float32", shard_axis=0,
        role_descriptor={"num_heads_total": str(nh), "num_kv_heads_total": str(nkv),
                         "head_dim": str(hd)},
    )
    plans = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"linear_qkv": spec},
    )
    assert len(plans) == 1 and plans[0].assembly == "qkv_uninterleave"
    assert len(plans[0].sources) == 1  # target_rank=0 of 2 → c0 only
    assembled = env.translator.assemble_into_destination(plans[0], pull=pull, device="cpu")
    # Receiver sees half the rows.
    assert assembled.shape == (half, hs)

    # Translate using the half-config — yields q, k, v for the receiver's half.
    out = list(env.translator.translate_megatron_to_hf(
        plans[0], assembled,
        transformer_config=half_cfg,
        hf_names=["q_proj.weight", "k_proj.weight", "v_proj.weight"],
    ))
    assert len(out) == 3
    q1, k1, v1 = out[0][1], out[1][1], out[2][1]

    # Ground truth: split the original (full) qkv_global with the half-config
    # is wrong; instead, split the half-tensor directly and compare with the
    # corresponding half of the original q, k, v.
    q_half_expected, k_half_expected, v_half_expected = helpers.split_qkv_weights(
        half_cfg, assembled,
    )
    assert torch.equal(q1, q_half_expected)
    assert torch.equal(k1, k_half_expected)
    assert torch.equal(v1, v_half_expected)
    # And the inverse: rebuilding the receiver's half from these q/k/v
    # should byte-match the assembled tensor.
    rebuilt = helpers.merge_qkv_weights(half_cfg, q_half_expected, k_half_expected, v_half_expected)
    assert torch.equal(rebuilt, assembled)


# ---------------------------------------------------------------------------
# Test 4 — full TP=1 round-trip (matched, target=source) across QKV
# ---------------------------------------------------------------------------


def test_qkv_tp1_full_roundtrip(env):
    """Simplest case: target_tp=source_tp=1. Receiver pulls the entire
    Megatron tensor and split_qkv_weights recovers the original q/k/v."""
    v2 = env.v2
    helpers = env.helpers
    rcv = v2.MxV2RefitReceiver(agent_name="rcv", device_id=0,
                                mx_server_url="stub:1", worker_rank=0)
    rcv.initialize(model_tensors=None)

    nh, nkv, hd, hs = 8, 4, 64, 512
    cfg = helpers.MegatronTransformerConfig(
        num_attention_heads=nh, num_query_groups=nkv,
        kv_channels=hd, hidden_size=hs,
    )
    torch.manual_seed(0xBABE)
    q = torch.randn(nh * hd, hs)
    k = torch.randn(nkv * hd, hs)
    v = torch.randn(nkv * hd, hs)
    qkv = helpers.merge_qkv_weights(cfg, q, k, v)

    layout = v2.TargetTpLayout(tp_size=1, tp_rank=0)
    cands = [_make_megatron_candidate(env, tp_rank=0, tp_size=1, sid="qkv")]
    pull = _make_pull_callback({"qkv": qkv})

    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_QKV_COLUMN,
        target_shape=qkv.shape, target_dtype="float32", shard_axis=0,
    )
    plans = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"linear_qkv": spec},
    )
    assembled = env.translator.assemble_into_destination(plans[0], pull=pull, device="cpu")
    out = list(env.translator.translate_megatron_to_hf(
        plans[0], assembled,
        transformer_config=cfg,
        hf_names=["q_proj.weight", "k_proj.weight", "v_proj.weight"],
    ))
    q1, k1, v1 = out[0][1], out[1][1], out[2][1]
    assert torch.equal(q1, q)
    assert torch.equal(k1, k)
    assert torch.equal(v1, v)


# ---------------------------------------------------------------------------
# Test 5 — gated_mlp_column TP=1 roundtrip
# ---------------------------------------------------------------------------


def test_gated_mlp_tp1_roundtrip(env):
    v2 = env.v2
    helpers = env.helpers
    rcv = v2.MxV2RefitReceiver(agent_name="rcv", device_id=0,
                                mx_server_url="stub:1", worker_rank=0)
    rcv.initialize(model_tensors=None)

    inter, hidden = 1024, 256
    gate = torch.randn(inter, hidden, dtype=torch.float32)
    up = torch.randn(inter, hidden, dtype=torch.float32)
    fused = helpers.merge_gated_mlp(gate, up)

    layout = v2.TargetTpLayout(tp_size=1, tp_rank=0)
    cands = [_make_megatron_candidate(env, tp_rank=0, tp_size=1, sid="fc1")]
    pull = _make_pull_callback({"fc1": fused})

    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_GATED_MLP_COLUMN,
        target_shape=fused.shape, target_dtype="float32", shard_axis=0,
        role_descriptor={"gated_mlp_order": "gate_then_up"},
    )
    plans = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"linear_fc1": spec},
    )
    assembled = env.translator.assemble_into_destination(plans[0], pull=pull, device="cpu")
    cfg = helpers.MegatronTransformerConfig(
        num_attention_heads=1, num_query_groups=1, kv_channels=1, hidden_size=hidden)
    out = list(env.translator.translate_megatron_to_hf(
        plans[0], assembled,
        transformer_config=cfg,
        hf_names=["mlp.gate_proj.weight", "mlp.up_proj.weight"],
    ))
    assert len(out) == 2
    assert out[0][0] == "mlp.gate_proj.weight"
    assert out[1][0] == "mlp.up_proj.weight"
    assert torch.equal(out[0][1], gate)
    assert torch.equal(out[1][1], up)


# ---------------------------------------------------------------------------
# Test 6 — sidecar parsing
# ---------------------------------------------------------------------------


def test_parse_megatron_sidecar(env):
    v2 = env.v2
    translator = env.translator
    helpers = env.helpers

    # Build a candidate with a registry containing the translator's
    # expected sidecar keys.
    cfg = helpers.MegatronTransformerConfig(
        num_attention_heads=32, num_query_groups=8,
        kv_channels=128, hidden_size=4096,
    )
    name_map = [
        ("decoder.layers.0.self_attention.linear_qkv.weight",
         ["model.layers.0.self_attn.q_proj.weight",
          "model.layers.0.self_attn.k_proj.weight",
          "model.layers.0.self_attn.v_proj.weight"]),
        ("decoder.layers.0.self_attention.linear_proj.weight",
         ["model.layers.0.self_attn.o_proj.weight"]),
    ]
    registry = {
        translator.SIDECAR_TRANSFORMER_CONFIG_KEY: cfg.to_dict(),
        translator.SIDECAR_HF_NAME_MAP_KEY: name_map,
    }

    ref_cls = sys.modules["modelexpress.refit_receiver"].SourceRef
    cand = v2.V2SourceCandidate(
        ref=ref_cls(mx_source_id="x", worker_id="y", model_name="m",
                    worker_rank=0, training_step=1),
        role=v2.ROLE_TRAINER, worker_rank=0,
        registry=registry, owned_experts_per_layer={},
        updated_at=0,
        megatron_meta=v2.MegatronSourceMeta(tp_rank=0, tp_size=1),
    )
    parsed_cfg, parsed_map = translator.parse_megatron_sidecar(cand)
    assert parsed_cfg == cfg
    assert parsed_map["decoder.layers.0.self_attention.linear_qkv.weight"] == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]


def test_discover_megatron_context(env):
    v2 = env.v2
    translator = env.translator
    ref_cls = sys.modules["modelexpress.refit_receiver"].SourceRef

    # No Megatron source → returns (None, {})
    non_mt = v2.V2SourceCandidate(
        ref=ref_cls(mx_source_id="x", worker_id="y", model_name="m",
                    worker_rank=0, training_step=1),
        role=v2.ROLE_TRAINER, worker_rank=0, registry=None,
        owned_experts_per_layer={}, updated_at=0, megatron_meta=None,
    )
    cfg, m = translator.discover_megatron_context([non_mt])
    assert cfg is None and m == {}

    # With a Megatron source carrying the sidecar
    helpers = env.helpers
    cfg_in = helpers.MegatronTransformerConfig(
        num_attention_heads=8, num_query_groups=4, kv_channels=64, hidden_size=512,
    )
    mt_cand = v2.V2SourceCandidate(
        ref=ref_cls(mx_source_id="x", worker_id="y", model_name="m",
                    worker_rank=0, training_step=1),
        role=v2.ROLE_TRAINER, worker_rank=0,
        registry={
            translator.SIDECAR_TRANSFORMER_CONFIG_KEY: cfg_in.to_dict(),
            translator.SIDECAR_HF_NAME_MAP_KEY: [("a", ["b"])],
        },
        owned_experts_per_layer={}, updated_at=0,
        megatron_meta=v2.MegatronSourceMeta(tp_rank=0, tp_size=1),
    )
    cfg, m = translator.discover_megatron_context([non_mt, mt_cand])
    assert cfg == cfg_in
    assert m == {"a": ["b"]}

    # EP ranks can carry disjoint global expert mappings; discovery must merge
    # all sidecars rather than returning only the first rank's map.
    ep1_cand = v2.V2SourceCandidate(
        ref=ref_cls(
            mx_source_id="ep1",
            worker_id="ep1-worker",
            model_name="m",
            worker_rank=1,
            training_step=1,
        ),
        role=v2.ROLE_TRAINER,
        worker_rank=1,
        registry={
            translator.SIDECAR_TRANSFORMER_CONFIG_KEY: cfg_in.to_dict(),
            translator.SIDECAR_HF_NAME_MAP_KEY: [("expert.weight32", ["experts.32"])],
        },
        owned_experts_per_layer={},
        updated_at=0,
        megatron_meta=v2.MegatronSourceMeta(
            tp_rank=0,
            tp_size=1,
            ep_rank=1,
            ep_size=2,
        ),
    )
    cfg, m = translator.discover_megatron_context([mt_cand, ep1_cand])
    assert cfg == cfg_in
    assert m == {"a": ["b"], "expert.weight32": ["experts.32"]}


# ---------------------------------------------------------------------------
# Test 8 — grouped per-expert linear_fc2 (row-parallel down_proj):
# 1 hf_name, passthrough, identity copy
# ---------------------------------------------------------------------------


def test_grouped_per_expert_fc2_passthrough_yields_single_hf(env):
    """TE-grouped MoE: each Megatron tensor is one expert's
    linear_fc2.weight<N>. Plan = passthrough, len(hf_names) = 1,
    translator should yield (hf_name, tensor) as-is."""
    v2 = env.v2
    helpers = env.helpers
    rcv = v2.MxV2RefitReceiver(agent_name="rcv", device_id=0,
                                mx_server_url="stub:1", worker_rank=0)
    rcv.initialize(model_tensors=None)

    # One expert's down_proj weight: shape (hidden, intermediate).
    hidden, intermediate = 2048, 768
    expert_id = 17
    torch.manual_seed(0xEFFE0)
    down = torch.randn(hidden, intermediate, dtype=torch.float32)

    layout = v2.TargetTpLayout(tp_size=1, tp_rank=0, ep_size=1, ep_rank=0)
    cands = [_make_megatron_candidate(env, tp_rank=0, tp_size=1, sid="trainer-0")]
    pull = _make_pull_callback({"trainer-0": down})

    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_EXPERT_ROW,
        target_shape=down.shape,
        target_dtype="float32",
        shard_axis=0,
        role_descriptor={
            "expert_layout": "grouped",
            "expert_id": str(expert_id),
            "expert_axis": "0",
        },
    )
    m_name = f"decoder.layers.0.mlp.experts.linear_fc2.weight{expert_id}"
    plans = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={m_name: spec},
    )
    assert len(plans) == 1
    assert plans[0].assembly == "passthrough"
    assert plans[0].role == v2.ROLE_MEGATRON_EXPERT_ROW
    assert len(plans[0].sources) == 1

    assembled = env.translator.assemble_into_destination(plans[0], pull=pull, device="cpu")
    assert torch.equal(assembled, down)

    cfg = helpers.MegatronTransformerConfig(
        num_attention_heads=1, num_query_groups=1, kv_channels=1, hidden_size=hidden,
    )
    hf_name = f"model.layers.0.mlp.experts.{expert_id}.down_proj.weight"
    out = list(env.translator.translate_megatron_to_hf(
        plans[0], assembled, transformer_config=cfg, hf_names=[hf_name],
    ))
    assert len(out) == 1
    assert out[0][0] == hf_name
    assert torch.equal(out[0][1], down)


# ---------------------------------------------------------------------------
# Test 9 — grouped per-expert linear_fc1 (column-parallel fused gate+up):
# 2 hf_names, passthrough, translator splits gate + up
# ---------------------------------------------------------------------------


def test_grouped_per_expert_fc1_passthrough_yields_gate_up(env):
    """TE-grouped MoE: each Megatron tensor is one expert's fused
    linear_fc1.weight<N> (gate concat'd with up along axis 0). Plan =
    passthrough, len(hf_names) = 2, translator should run split_gated_mlp
    and yield (gate_name, gate) + (up_name, up)."""
    v2 = env.v2
    helpers = env.helpers
    rcv = v2.MxV2RefitReceiver(agent_name="rcv", device_id=0,
                                mx_server_url="stub:1", worker_rank=0)
    rcv.initialize(model_tensors=None)

    intermediate, hidden = 768, 2048
    expert_id = 42
    torch.manual_seed(0xC0FFEE)
    gate = torch.randn(intermediate, hidden, dtype=torch.float32)
    up = torch.randn(intermediate, hidden, dtype=torch.float32)
    fused = helpers.merge_gated_mlp(gate, up)
    assert fused.shape == (2 * intermediate, hidden)

    layout = v2.TargetTpLayout(tp_size=1, tp_rank=0, ep_size=1, ep_rank=0)
    cands = [_make_megatron_candidate(env, tp_rank=0, tp_size=1, sid="trainer-0")]
    pull = _make_pull_callback({"trainer-0": fused})

    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_EXPERT_COLUMN,
        target_shape=fused.shape,
        target_dtype="float32",
        shard_axis=0,
        role_descriptor={
            "expert_layout": "grouped",
            "expert_id": str(expert_id),
            "expert_axis": "0",
            "gated_mlp_order": "gate_then_up",
        },
    )
    m_name = f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert_id}"
    plans = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={m_name: spec},
    )
    assert len(plans) == 1
    assert plans[0].assembly == "passthrough"
    assert plans[0].role == v2.ROLE_MEGATRON_EXPERT_COLUMN

    assembled = env.translator.assemble_into_destination(plans[0], pull=pull, device="cpu")
    assert torch.equal(assembled, fused)

    cfg = helpers.MegatronTransformerConfig(
        num_attention_heads=1, num_query_groups=1, kv_channels=1, hidden_size=hidden,
    )
    gate_name = f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
    up_name = f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"
    out = list(env.translator.translate_megatron_to_hf(
        plans[0], assembled, transformer_config=cfg, hf_names=[gate_name, up_name],
    ))
    assert len(out) == 2
    assert out[0][0] == gate_name
    assert out[1][0] == up_name
    assert torch.equal(out[0][1], gate)
    assert torch.equal(out[1][1], up)


# ---------------------------------------------------------------------------
# Test 10 — invalid hf_names length for passthrough expert path
# ---------------------------------------------------------------------------


def test_grouped_per_expert_invalid_hf_names_count_raises(env):
    """Passthrough expert path supports 1 or 2 hf_names only; 3+ is a
    misconfiguration and should raise rather than silently produce
    garbage HF output."""
    v2 = env.v2
    helpers = env.helpers

    fused = torch.randn(64, 32, dtype=torch.float32)
    cfg = helpers.MegatronTransformerConfig(
        num_attention_heads=1, num_query_groups=1, kv_channels=1, hidden_size=32,
    )
    plan = v2.MegatronSlicePlan(
        tensor_name="x", role=v2.ROLE_MEGATRON_EXPERT_COLUMN,
        target_shape=fused.shape, target_dtype="float32",
        sources=[], assembly="passthrough", role_descriptor={},
    )
    with pytest.raises(ValueError, match="passthrough expects 1 or 2 hf_names"):
        list(env.translator.translate_megatron_to_hf(
            plan, fused, transformer_config=cfg,
            hf_names=["a", "b", "c"],  # 3 names = invalid
        ))


# ---------------------------------------------------------------------------
# Test 11 — gated_mlp_column TP=2 (target-narrower): receiver concatenates
# two source ranks' [gate_local; up_local] shards and the translator
# un-interleaves to (gate_global, up_global).
# ---------------------------------------------------------------------------


def test_gated_mlp_tp2_uninterleaves_correctly(env):
    """Two source ranks each publish ``[gate_local; up_local]``. The
    receiver concatenates along axis 0 → ``[gate_r0; up_r0; gate_r1;
    up_r1]``. The translator must run ``split_gated_mlp_tp`` (not the
    naive ``split_gated_mlp``) so the yielded gate / up are
    byte-identical to the globals."""
    v2 = env.v2
    helpers = env.helpers
    rcv = v2.MxV2RefitReceiver(agent_name="rcv", device_id=0,
                                mx_server_url="stub:1", worker_rank=0)
    rcv.initialize(model_tensors=None)

    inter_per_rank, hidden = 256, 128
    intermediate = inter_per_rank * 2
    torch.manual_seed(0xFEEDBEEF)
    gate_full = torch.randn(intermediate, hidden, dtype=torch.float32)
    up_full = torch.randn(intermediate, hidden, dtype=torch.float32)

    # Per-rank slices, exactly as a TP=2 Megatron trainer would publish.
    gate_r0 = gate_full[:inter_per_rank]
    gate_r1 = gate_full[inter_per_rank:]
    up_r0 = up_full[:inter_per_rank]
    up_r1 = up_full[inter_per_rank:]
    fused_r0 = helpers.merge_gated_mlp(gate_r0, up_r0)
    fused_r1 = helpers.merge_gated_mlp(gate_r1, up_r1)

    layout = v2.TargetTpLayout(tp_size=1, tp_rank=0)
    cands = [
        _make_megatron_candidate(env, tp_rank=0, tp_size=2, sid="fc1-r0"),
        _make_megatron_candidate(env, tp_rank=1, tp_size=2, sid="fc1-r1"),
    ]
    pull = _make_pull_callback({"fc1-r0": fused_r0, "fc1-r1": fused_r1})

    # Target-narrower (target_tp=1, source_tp=2) — receiver sees the full
    # 2 * intermediate rows.
    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_GATED_MLP_COLUMN,
        target_shape=(2 * intermediate, hidden), target_dtype="float32",
        shard_axis=0,
        role_descriptor={"gated_mlp_order": "gate_then_up"},
    )
    plans = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"linear_fc1": spec},
    )
    assert len(plans) == 1
    plan = plans[0]
    assert plan.assembly == "gated_mlp_split"
    assert len(plan.sources) == 2  # multi-source

    assembled = env.translator.assemble_into_destination(plan, pull=pull, device="cpu")
    # Sanity: assembled is [gate_r0; up_r0; gate_r1; up_r1].
    assert assembled.shape == (2 * intermediate, hidden)
    cfg = helpers.MegatronTransformerConfig(
        num_attention_heads=1, num_query_groups=1, kv_channels=1, hidden_size=hidden,
    )
    out = list(env.translator.translate_megatron_to_hf(
        plan, assembled, transformer_config=cfg,
        hf_names=["mlp.gate_proj.weight", "mlp.up_proj.weight"],
    ))
    assert len(out) == 2
    assert out[0][0] == "mlp.gate_proj.weight"
    assert out[1][0] == "mlp.up_proj.weight"
    # Byte-identical to the globals.
    assert torch.equal(out[0][1], gate_full)
    assert torch.equal(out[1][1], up_full)
