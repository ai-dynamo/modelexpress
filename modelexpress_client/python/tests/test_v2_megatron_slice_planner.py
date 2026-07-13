# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Megatron slice planner.

Phase B of the Megatron-Core MX path. See
``temp/NemoRL_Megatron_MX_Design.md`` §5 for the role × layout matrix.

Each test builds a synthetic candidate set as if a Megatron trainer had
published native shards with the appropriate ``megatron_role`` extras,
then asks ``MxV2RefitReceiver.pick_megatron_slice_plans`` to produce a
plan and asserts on its sources / target_local_range / role_descriptor.
The tests use the same module-stubbing pattern as
``test_v2_source_picker.py`` so they run without NIXL or a live MX server.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent / "modelexpress"


def _load(modname: str, path: Path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def v2():
    """Load nemo_rl_v2 with NIXL / gRPC / heartbeat stubbed out."""
    pkg = types.ModuleType("modelexpress")
    pkg.__path__ = [str(_PKG_ROOT)]  # type: ignore[attr-defined]
    sys.modules["modelexpress"] = pkg

    p2p_pb2 = types.ModuleType("modelexpress.p2p_pb2")
    p2p_pb2.SOURCE_STATUS_READY = 2
    p2p_pb2.SOURCE_STATUS_INITIALIZING = 1
    p2p_pb2.SOURCE_STATUS_STALE = 3
    p2p_pb2.MX_SOURCE_TYPE_WEIGHTS = 0
    p2p_pb2.BACKEND_FRAMEWORK_UNKNOWN = 0
    sys.modules["modelexpress.p2p_pb2"] = p2p_pb2

    class _SourceIdentity:
        def __init__(self, **kwargs):
            self.model_name = kwargs.get("model_name", "")
            self.extra_parameters = dict(kwargs.get("extra_parameters") or {})

    class _TensorDescriptor:
        def __init__(self, **kwargs):
            self.name = kwargs.get("name", "")
            self.addr = kwargs.get("addr", 0)
            self.size = kwargs.get("size", 0)
            self.device_id = kwargs.get("device_id", 0)
            self.dtype = kwargs.get("dtype", "")

    class _WorkerMetadata:
        def __init__(self, **kwargs):
            self.worker_rank = kwargs.get("worker_rank", 0)
            self.nixl_metadata = kwargs.get("nixl_metadata", b"")
            self.tensors = list(kwargs.get("tensors") or [])
            self.status = kwargs.get("status", 0)
            self.agent_name = kwargs.get("agent_name", "")

    p2p_pb2.SourceIdentity = _SourceIdentity  # type: ignore[attr-defined]
    p2p_pb2.WorkerMetadata = _WorkerMetadata  # type: ignore[attr-defined]
    p2p_pb2.TensorDescriptor = _TensorDescriptor  # type: ignore[attr-defined]

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

    pub_mod = types.ModuleType("modelexpress.training_publisher")

    class _PubStub:
        def __init__(self, *a, **kw):
            self._client = None
            self._nixl = None

        def initialize(self, **kw):
            pass

        def publish_weights(self, *a, **kw):
            return "stub-sid"

        def mark_ready(self, **kw):
            return True

        def shutdown(self):
            pass

        def _build_identity(self, step):
            return p2p_pb2.SourceIdentity()

    pub_mod.MxTrainingPublisher = _PubStub
    sys.modules["modelexpress.training_publisher"] = pub_mod

    types_mod = types.ModuleType("modelexpress.types")

    @dataclass
    class _TD:
        name: str = ""
        addr: int = 0
        size: int = 0
        device_id: int = 0
        dtype: str = ""

    types_mod.TensorDescriptor = _TD
    sys.modules["modelexpress.types"] = types_mod

    sd = _load("modelexpress.shape_descriptors", _PKG_ROOT / "shape_descriptors.py")
    pkg.shape_descriptors = sd  # type: ignore[attr-defined]
    v2 = _load("modelexpress.nemo_rl_v2", _PKG_ROOT / "nemo_rl_v2.py")
    return v2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_megatron_candidate(
    v2,
    *,
    tp_rank: int,
    tp_size: int,
    pp_rank: int = 0,
    pp_size: int = 1,
    ep_rank: int = 0,
    ep_size: int = 1,
    updated_at: int = 0,
    sid: str | None = None,
    owned_experts_per_layer: dict | None = None,
    role: str | None = None,  # accepted+ignored for test-readability only
):
    """Build a V2SourceCandidate with megatron_meta populated.

    ``role`` is unused (just a label for test readability — production
    Megatron sources don't carry a per-source role).
    """
    ref_cls = v2.SourceRef if hasattr(v2, "SourceRef") else \
        sys.modules["modelexpress.refit_receiver"].SourceRef
    sid = sid or f"sid-{role or 'mt'}-tp{tp_rank}"
    ref = ref_cls(
        mx_source_id=sid,
        worker_id=f"wid-{sid}",
        model_name="m",
        worker_rank=tp_rank * 1,  # not load-bearing for these tests
        training_step=1,
    )
    mm = v2.MegatronSourceMeta(
        tp_rank=tp_rank, tp_size=tp_size,
        pp_rank=pp_rank, pp_size=pp_size,
        ep_rank=ep_rank, ep_size=ep_size,
    )
    return v2.V2SourceCandidate(
        ref=ref,
        role=v2.ROLE_TRAINER,
        worker_rank=tp_rank,
        registry=None,
        owned_experts_per_layer=owned_experts_per_layer or {},
        updated_at=updated_at,
        megatron_meta=mm,
    )


def _make_receiver(v2, *, target_tp=2, target_rank=0, ep_size=1, ep_rank=0):
    rcv = v2.MxV2RefitReceiver(
        agent_name="rcv",
        device_id=0,
        mx_server_url="mx-stub:8001",
        worker_rank=0,
    )
    rcv.initialize(model_tensors=None)
    layout = v2.TargetTpLayout(
        tp_size=target_tp, tp_rank=target_rank, ep_size=ep_size, ep_rank=ep_rank,
    )
    return rcv, layout


# ---------------------------------------------------------------------------
# Role: replicated
# ---------------------------------------------------------------------------


def test_replicated_picks_single_freshest_source(v2):
    rcv, layout = _make_receiver(v2)
    cands = [
        _make_megatron_candidate(v2, role=v2.ROLE_MEGATRON_REPLICATED,
                                 tp_rank=0, tp_size=2, sid="r-old", updated_at=10),
        _make_megatron_candidate(v2, role=v2.ROLE_MEGATRON_REPLICATED,
                                 tp_rank=0, tp_size=2, sid="r-new", updated_at=20),
    ]
    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_REPLICATED,
        target_shape=(128,),
        target_dtype="bfloat16",
    )
    plans = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"norm.weight": spec},
    )
    assert len(plans) == 1
    plan = plans[0]
    assert plan.role == v2.ROLE_MEGATRON_REPLICATED
    assert plan.assembly == "passthrough"
    assert len(plan.sources) == 1
    assert plan.sources[0].mx_source_id == "r-new"  # freshest wins
    assert plan.sources[0].target_local_range == (0, 128)


# ---------------------------------------------------------------------------
# Role: column / row — matched-TP
# ---------------------------------------------------------------------------


def test_column_matched_tp_two_sources(v2):
    """source_tp = target_tp = 2: each receiver pulls one source's full slice."""
    rcv, layout = _make_receiver(v2, target_tp=2, target_rank=1)
    cands = [
        _make_megatron_candidate(v2, role=v2.ROLE_MEGATRON_COLUMN, tp_rank=0, tp_size=2, sid="c0"),
        _make_megatron_candidate(v2, role=v2.ROLE_MEGATRON_COLUMN, tp_rank=1, tp_size=2, sid="c1"),
    ]
    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_COLUMN,
        target_shape=(2048, 1024),  # global
        target_dtype="bfloat16",
        shard_axis=0,
    )
    plans = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"linear.weight": spec},
    )
    plan = plans[0]
    assert plan.assembly == "concat_dim0"
    # target_rank=1 => receiver wants [1024:2048]; one source covers it (sid=c1).
    assert len(plan.sources) == 1
    src = plan.sources[0]
    assert src.mx_source_id == "c1"
    assert src.target_local_range == (0, 1024)  # offset within the receiver's window
    assert src.source_subslice is None
    # receiver-side target_shape is the per-rank window
    assert plan.target_shape == (1024, 1024)


def test_row_matched_tp(v2):
    """row-parallel = concat along dim 1, otherwise same shape as column."""
    rcv, layout = _make_receiver(v2, target_tp=2, target_rank=0)
    cands = [
        _make_megatron_candidate(v2, role=v2.ROLE_MEGATRON_ROW, tp_rank=0, tp_size=2, sid="r0"),
        _make_megatron_candidate(v2, role=v2.ROLE_MEGATRON_ROW, tp_rank=1, tp_size=2, sid="r1"),
    ]
    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_ROW,
        target_shape=(1024, 2048),
        target_dtype="bfloat16",
        shard_axis=1,
    )
    plan = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"o_proj": spec},
    )[0]
    assert plan.assembly == "concat_dim1"
    assert len(plan.sources) == 1
    assert plan.sources[0].mx_source_id == "r0"  # target_rank=0 → first slice
    assert plan.target_shape == (1024, 1024)


# ---------------------------------------------------------------------------
# Role: column — mixed-TP, target wider than source
# ---------------------------------------------------------------------------


def test_column_mixed_tp_target_wider(v2):
    """source_tp=2, target_tp=4: target_rank=0 wants [0:512] which is a sub-range
    of source rank 0's [0:1024] slice. source_subslice should be set."""
    rcv, layout = _make_receiver(v2, target_tp=4, target_rank=0)
    cands = [
        _make_megatron_candidate(v2, role=v2.ROLE_MEGATRON_COLUMN, tp_rank=0, tp_size=2, sid="c0"),
        _make_megatron_candidate(v2, role=v2.ROLE_MEGATRON_COLUMN, tp_rank=1, tp_size=2, sid="c1"),
    ]
    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_COLUMN,
        target_shape=(2048, 1024),
        target_dtype="bfloat16",
        shard_axis=0,
    )
    plan = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"col": spec},
    )[0]
    # target_rank=0 of 4 => [0:512]; source c0 covers [0:1024]; partial pull.
    assert len(plan.sources) == 1
    src = plan.sources[0]
    assert src.mx_source_id == "c0"
    assert src.target_local_range == (0, 512)
    assert src.source_subslice == (0, 512)
    assert plan.target_shape == (512, 1024)


def test_column_mixed_tp_target_narrower(v2):
    """source_tp=4, target_tp=2: target_rank=0 wants [0:1024] which spans source
    ranks 0 and 1 (each [0:512] and [512:1024])."""
    rcv, layout = _make_receiver(v2, target_tp=2, target_rank=0)
    cands = [
        _make_megatron_candidate(v2, role=v2.ROLE_MEGATRON_COLUMN, tp_rank=i, tp_size=4, sid=f"c{i}")
        for i in range(4)
    ]
    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_COLUMN,
        target_shape=(2048, 1024),
        target_dtype="bfloat16",
        shard_axis=0,
    )
    plan = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"col": spec},
    )[0]
    # Two sources contribute (c0, c1), each pulls its full slice (no subslice).
    assert len(plan.sources) == 2
    assert {s.mx_source_id for s in plan.sources} == {"c0", "c1"}
    by_sid = {s.mx_source_id: s for s in plan.sources}
    assert by_sid["c0"].target_local_range == (0, 512)
    assert by_sid["c1"].target_local_range == (512, 1024)
    assert all(s.source_subslice is None for s in plan.sources)
    assert plan.target_shape == (1024, 1024)


# ---------------------------------------------------------------------------
# Role: qkv_column — head counts aggregated correctly
# ---------------------------------------------------------------------------


def test_qkv_column_assembles_with_receiver_spec(v2):
    """QKV per-tensor descriptor is receiver-owned (typically derived from
    the HF model config). The planner forwards it verbatim into
    role_descriptor so the receiver-side translator can consume it.
    """
    rcv, layout = _make_receiver(v2, target_tp=2, target_rank=0)
    cands = [
        _make_megatron_candidate(v2, tp_rank=0, tp_size=2, sid="q0"),
        _make_megatron_candidate(v2, tp_rank=1, tp_size=2, sid="q1"),
    ]
    # Each source rank publishes (16+2*4)*128 = 3072 rows; global has 2*3072 = 6144 rows.
    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_QKV_COLUMN,
        target_shape=(6144, 4096),
        target_dtype="bfloat16",
        shard_axis=0,
        # Receiver supplies the descriptor directly (it knows num_heads
        # from the HF model config). Per-tensor role extras live on the
        # source's shape_registry entries; the receiver can also choose
        # to consume those instead.
        role_descriptor={
            "num_heads_total": "32",
            "num_kv_heads_total": "8",
            "head_dim": "128",
            "qkv_interleave": "by_head",
        },
    )
    plan = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"qkv": spec},
    )[0]
    assert plan.assembly == "qkv_uninterleave"
    assert len(plan.sources) == 1  # target_rank=0 of 2 covered by source rank 0
    rd = plan.role_descriptor
    # Receiver's spec descriptor is forwarded verbatim.
    assert rd["num_heads_total"] == "32"
    assert rd["num_kv_heads_total"] == "8"
    assert rd["head_dim"] == "128"
    assert rd["qkv_interleave"] == "by_head"


# ---------------------------------------------------------------------------
# Role: gated_mlp_column
# ---------------------------------------------------------------------------


def test_gated_mlp_assembly_and_descriptor(v2):
    rcv, layout = _make_receiver(v2, target_tp=2, target_rank=0)
    cands = [
        _make_megatron_candidate(v2, tp_rank=0, tp_size=2, sid="g0"),
        _make_megatron_candidate(v2, tp_rank=1, tp_size=2, sid="g1"),
    ]
    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_GATED_MLP_COLUMN,
        target_shape=(8192, 4096),
        target_dtype="bfloat16",
        shard_axis=0,
        role_descriptor={"gated_mlp_order": "gate_then_up"},
    )
    plan = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"fc1": spec},
    )[0]
    assert plan.assembly == "gated_mlp_split"
    assert plan.role_descriptor.get("gated_mlp_order") == "gate_then_up"


# ---------------------------------------------------------------------------
# Role: vocab_parallel
# ---------------------------------------------------------------------------


def test_vocab_parallel_concat_dim0(v2):
    rcv, layout = _make_receiver(v2, target_tp=2, target_rank=1)
    cands = [
        _make_megatron_candidate(v2, role=v2.ROLE_MEGATRON_VOCAB_PARALLEL,
                                 tp_rank=0, tp_size=2, sid="v0"),
        _make_megatron_candidate(v2, role=v2.ROLE_MEGATRON_VOCAB_PARALLEL,
                                 tp_rank=1, tp_size=2, sid="v1"),
    ]
    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_VOCAB_PARALLEL,
        target_shape=(151936, 4096),
        target_dtype="bfloat16",
        shard_axis=0,
    )
    plan = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"embed": spec},
    )[0]
    assert plan.assembly == "concat_dim0"
    assert len(plan.sources) == 1
    assert plan.sources[0].mx_source_id == "v1"  # target_rank=1 → second half


# ---------------------------------------------------------------------------
# Role: expert_column / expert_row — per-expert
# ---------------------------------------------------------------------------


def test_expert_column_per_expert_routing(v2):
    """EP=4 trainer publishes 32 experts (8 per rank); receiver wants experts
    [0, 7, 16, 31] (one from each EP rank). Plan should pick the right
    EP-rank source per expert."""
    rcv, layout = _make_receiver(v2, target_tp=1, target_rank=0, ep_size=8, ep_rank=0)
    cands = [
        _make_megatron_candidate(
            v2, role=v2.ROLE_MEGATRON_EXPERT_COLUMN,
            tp_rank=0, tp_size=1,
            ep_rank=ep_rank, ep_size=4,
            sid=f"e{ep_rank}",
            owned_experts_per_layer={3: set(range(ep_rank * 8, (ep_rank + 1) * 8))},
        )
        for ep_rank in range(4)
    ]
    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_EXPERT_COLUMN,
        target_shape=(2048, 4096),  # one expert's shape
        target_dtype="bfloat16",
        role_descriptor={
            "local_expert_ids": "0,7,16,31",
            "layer_id": "3",
        },
    )
    plan = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"experts.gate": spec},
    )[0]
    assert plan.assembly == "per_expert"
    assert len(plan.sources) == 4
    by_expert = {s.target_local_range[0]: s for s in plan.sources}
    assert by_expert[0].mx_source_id == "e0"     # expert 0 owned by ep_rank 0
    assert by_expert[7].mx_source_id == "e0"     # expert 7 also ep_rank 0
    assert by_expert[16].mx_source_id == "e2"    # expert 16 owned by ep_rank 2
    assert by_expert[31].mx_source_id == "e3"    # expert 31 owned by ep_rank 3
    for src in plan.sources:
        assert src.role_extras.get("expert_id") in {"0", "7", "16", "31"}


# ---------------------------------------------------------------------------
# Role: expert_column / expert_row — TE-grouped per-expert layout
# (each Megatron tensor IS one expert's full weight, e.g. linear_fc1.weight0)
# ---------------------------------------------------------------------------


def test_grouped_expert_column_picks_passthrough_plan(v2):
    """TE-grouped layout: each Megatron tensor is ONE expert's fused
    gate+up. Plan should be passthrough with a single source — not
    per_expert with N sources."""
    rcv, layout = _make_receiver(v2, target_tp=1, target_rank=0)
    # One trainer source publishes this expert's weight (EP=1).
    cands = [
        _make_megatron_candidate(
            v2, role=v2.ROLE_MEGATRON_EXPERT_COLUMN,
            tp_rank=0, tp_size=1, ep_rank=0, ep_size=1,
            sid="trainer-0",
        ),
    ]
    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_EXPERT_COLUMN,
        target_shape=(19456, 2048),  # fused gate+up shape for one expert (2 * intermediate, hidden)
        target_dtype="bfloat16",
        role_descriptor={
            "expert_layout": "grouped",
            "expert_id": "17",
            "expert_axis": "0",
        },
    )
    plan = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"decoder.layers.0.mlp.experts.linear_fc1.weight17": spec},
    )[0]
    assert plan.assembly == "passthrough"
    assert len(plan.sources) == 1
    assert plan.sources[0].mx_source_id == "trainer-0"
    assert plan.sources[0].target_local_range == (0, 19456)
    # Source covers the full target tensor (no sub-slicing in v0).
    assert plan.sources[0].source_subslice is None
    # role_extras forwarded so the receiver knows this is grouped.
    assert plan.sources[0].role_extras.get("expert_layout") == "grouped"
    assert plan.sources[0].role_extras.get("expert_id") == "17"


def test_grouped_expert_row_picks_passthrough_plan(v2):
    """Row-parallel variant: linear_fc2.weight17 — one expert's
    down_proj. Same passthrough shape, just a different role."""
    rcv, layout = _make_receiver(v2, target_tp=1, target_rank=0)
    cands = [
        _make_megatron_candidate(
            v2, role=v2.ROLE_MEGATRON_EXPERT_ROW,
            tp_rank=0, tp_size=1, ep_rank=0, ep_size=1,
            sid="trainer-0",
        ),
    ]
    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_EXPERT_ROW,
        target_shape=(2048, 9728),  # down_proj shape (hidden, intermediate)
        target_dtype="bfloat16",
        role_descriptor={
            "expert_layout": "grouped",
            "expert_id": "17",
            "expert_axis": "0",
        },
    )
    plan = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"decoder.layers.0.mlp.experts.linear_fc2.weight17": spec},
    )[0]
    assert plan.assembly == "passthrough"
    assert plan.role == v2.ROLE_MEGATRON_EXPERT_ROW
    assert len(plan.sources) == 1
    assert plan.sources[0].target_local_range == (0, 2048)


def test_legacy_expert_layout_still_uses_per_expert_assembly(v2):
    """Sanity: when expert_layout is unset or == "leading_axis", the
    planner falls back to the legacy _plan_per_expert path (multi-source,
    per_expert assembly)."""
    rcv, layout = _make_receiver(v2, target_tp=1, target_rank=0, ep_size=4, ep_rank=0)
    cands = [
        _make_megatron_candidate(
            v2, role=v2.ROLE_MEGATRON_EXPERT_COLUMN,
            tp_rank=0, tp_size=1, ep_rank=ep_rank, ep_size=4,
            sid=f"e{ep_rank}",
            owned_experts_per_layer={5: set(range(ep_rank * 2, (ep_rank + 1) * 2))},
        )
        for ep_rank in range(4)
    ]
    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_EXPERT_COLUMN,
        target_shape=(1024, 2048),
        target_dtype="bfloat16",
        role_descriptor={
            # expert_layout intentionally omitted — defaults to leading_axis
            "local_expert_ids": "0,3,5,7",
            "layer_id": "5",
        },
    )
    plan = rcv.pick_megatron_slice_plans(
        cands, target_tp_layout=layout,
        target_tensor_specs={"experts.gate": spec},
    )[0]
    assert plan.assembly == "per_expert"  # NOT passthrough
    # The legacy path picks one source per expert id.
    assert len(plan.sources) == 4


# ---------------------------------------------------------------------------
# Backwards compat: non-Megatron candidate set
# ---------------------------------------------------------------------------


def test_non_megatron_candidates_yield_empty_source_lists(v2):
    """If no candidate has megatron_meta, planner returns plans with empty
    sources — caller should detect and fall back to pick_best_source."""
    rcv, layout = _make_receiver(v2)

    @dataclass
    class _Ref:
        mx_source_id: str = "x"
        worker_id: str = "y"
        model_name: str = "m"
        worker_rank: int = 0
        training_step: int = 1

    cand = v2.V2SourceCandidate(
        ref=_Ref(),
        role=v2.ROLE_TRAINER,
        worker_rank=0,
        registry=None,
        owned_experts_per_layer={},
        updated_at=0,
        megatron_meta=None,
    )
    spec = v2.MegatronTensorSpec(
        role=v2.ROLE_MEGATRON_COLUMN,
        target_shape=(1024, 512),
        target_dtype="bfloat16",
    )
    plans = rcv.pick_megatron_slice_plans(
        [cand], target_tp_layout=layout,
        target_tensor_specs={"col": spec},
    )
    assert len(plans) == 1
    assert plans[0].sources == []


# ---------------------------------------------------------------------------
# Source extraction: discover_v2_sources should populate megatron_meta
# ---------------------------------------------------------------------------


def test_extract_megatron_meta_publisher_kind_marker(v2):
    """Source-level Megatron metadata is rank-position only — no role.

    Detection is via ``publisher_kind == "megatron"`` OR presence of
    ``tp_rank`` + ``tp_size``. Per-tensor role lives in the registry,
    not in source extras.
    """
    extra = {
        "mx_v2": "1",
        "publisher_kind": "megatron",
        "tp_rank": "2",
        "tp_size": "4",
        "pp_rank": "1",
        "pp_size": "2",
        "ep_rank": "0",
        "ep_size": "1",
    }
    mm = v2._extract_megatron_meta(extra)
    assert mm is not None
    assert (mm.tp_rank, mm.tp_size) == (2, 4)
    assert (mm.pp_rank, mm.pp_size) == (1, 2)
    assert (mm.ep_rank, mm.ep_size) == (0, 1)


def test_extract_megatron_meta_tp_keys_alone_trigger_detection(v2):
    """Even without publisher_kind, presence of tp_rank + tp_size signals
    a Megatron-shaped source."""
    extra = {"mx_v2": "1", "tp_rank": "0", "tp_size": "2"}
    mm = v2._extract_megatron_meta(extra)
    assert mm is not None
    assert mm.tp_size == 2


def test_extract_megatron_meta_returns_none_for_non_megatron(v2):
    extra = {"mx_v2": "1", "role": "trainer", "worker_rank": "0"}  # neither marker
    assert v2._extract_megatron_meta(extra) is None
