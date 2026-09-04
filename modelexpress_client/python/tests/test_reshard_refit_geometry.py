# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Geometry-capture mechanism test - no engine needed.

A tiny model with column-parallel, row-parallel, fused (qkv-style), unsharded,
and one UNSUPPORTED-op weight_loader. Exercises: contiguous/strided/fused-offset
op-chain capture, full copies, and per-source unsupported attribution (an
unsupported op is identified without producing an incorrect copy). Runs in any
torch env: pytest tests/test_reshard_refit_geometry.py
"""

import re

import pytest
import torch

from modelexpress.refit.reshard.geometry import (
    LazyWeight,
    build_lazy_weights,
    capture_geometry,
    capture_weights,
    convert_source_weights,
)
from modelexpress.refit.reshard.slice_plan import Shard, plan_pull
from modelexpress.refit.reshard.types import UnsupportedReshard

TP_RANK = 0

_QKV = {"q": (0, 4), "k": (4, 2), "v": (6, 2)}  # (dest row offset, rows) per shard


class ToyModel(torch.nn.Module):
    """Dest params at their SHARDED shapes (what a real engine holds); loaders
    narrow the full (lazy) source down to the shard, mirroring Column/Row/QKV."""

    def __init__(self, with_bad: bool = False):
        super().__init__()
        self.col = torch.nn.Parameter(
            torch.empty(4, 4)
        )  # ColumnParallel: full out=8 -> [0:4] contiguous
        self.col.weight_loader = self._col_loader
        self.row = torch.nn.Parameter(
            torch.empty(4, 4)
        )  # RowParallel: full in=8 -> [:,0:4] STRIDED
        self.row.weight_loader = self._row_loader
        self.qkv = torch.nn.Parameter(
            torch.empty(8, 4)
        )  # fused q(4)+k(2)+v(2), per-shard dest offsets
        self.qkv.weight_loader = self._qkv_loader
        self.norm = torch.nn.Parameter(torch.empty(4))  # unsharded: full copy
        self.norm.weight_loader = lambda param, loaded: param.data.copy_(loaded)
        if with_bad:
            self.bad = torch.nn.Parameter(
                torch.empty(4, 4)
            )  # loader uses an unsupported op
            self.bad.weight_loader = self._bad_loader

    def _col_loader(self, param, loaded):
        loaded = loaded.narrow(0, TP_RANK * param.shape[0], param.shape[0])
        param.data.copy_(loaded)

    def _row_loader(self, param, loaded):
        loaded = loaded.narrow(1, TP_RANK * param.shape[1], param.shape[1])
        param.data.copy_(loaded)

    def _qkv_loader(self, param, loaded, shard_id):
        off, size = _QKV[shard_id]
        param.data.narrow(0, off, size).copy_(loaded)

    def _bad_loader(self, param, loaded):
        # Arithmetic is not a pure view/slice op -> UnsupportedReshard.
        param.data.copy_(loaded * 2)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, loaded in weights:
            if name in _QKV:
                self.qkv.weight_loader(self.qkv, loaded, name)
            else:
                params[name].weight_loader(params[name], loaded)


def _manifest(with_bad: bool = False):
    f32 = torch.float32
    m = [
        ("col", f32, [8, 4]),
        ("row", f32, [4, 8]),
        ("q", f32, [4, 4]),
        ("k", f32, [2, 4]),
        ("v", f32, [2, 4]),
        ("norm", f32, [4]),
    ]
    if with_bad:
        m.append(("bad", f32, [4, 4]))
    return m


def test_capture_op_chains_and_dest_offsets():
    with torch.device("meta"):
        model = ToyModel()
    result = capture_geometry(model, _manifest())
    by_src = {c.src_name: c for c in result.copies}

    # ColumnParallel: contiguous row block, rank 0 -> narrow(dim0, 0, 4).
    assert by_src["col"].op_chain == (("narrow", (0, 0, 4), ()),)
    assert by_src["col"].dest_shape == (4, 4) and by_src["col"].dest_offset == 0

    # RowParallel: column slice -> narrow(dim1, 0, 4).
    assert by_src["row"].op_chain == (("narrow", (1, 0, 4), ()),)
    assert by_src["row"].dest_shape == (4, 4)

    # Fused qkv: each source full-copied into its own dest offset (row*cols).
    assert by_src["q"].op_chain == () and by_src["q"].dest_offset == 0
    assert by_src["k"].dest_offset == 16 and by_src["v"].dest_offset == 24
    assert all(by_src[s].param_name == "qkv" for s in ("q", "k", "v"))

    # Unsharded norm: full copy, empty chain.
    assert by_src["norm"].op_chain == () and by_src["norm"].dest_shape == (4,)

    assert result.unsupported == [] and result.unattributed == 0


def test_unsupported_op_falls_back_per_source():
    """An unsupported-op loader knocks out ONLY that source; others still captured."""
    with torch.device("meta"):
        model = ToyModel(with_bad=True)
    result = capture_geometry(model, _manifest(with_bad=True))
    by_src = {c.src_name: c for c in result.copies}

    assert result.unsupported == ["bad"]  # only the bad source
    # Everything else still captured normally (no whole-bake abort).
    assert "bad" not in by_src
    assert by_src["col"].op_chain == (("narrow", (0, 0, 4), ()),)
    assert by_src["q"].dest_offset == 0 and by_src["norm"].op_chain == ()


def test_unsupported_source_records_the_op_that_defeated_capture():
    """The count alone cannot distinguish an unexpressible fused layout from a
    loader that merely touched one op outside the allowlist, so keep the cause."""
    with torch.device("meta"):
        model = ToyModel(with_bad=True)
    result = capture_geometry(model, _manifest(with_bad=True))

    assert set(result.unsupported_reasons) == {"bad"}
    reason = result.unsupported_reasons["bad"]
    assert "unsupported op" in reason
    assert "aten.mul" in reason
    # The offending source and its op-chain stay in the message, which is what
    # makes a single failure actionable without a re-run.
    assert "'bad'" in reason


def test_summarize_unsupported_groups_one_cause_across_many_sources():
    """Thousands of sources failing for one reason must read as one cause. Each
    message embeds its own source name, so grouping has to ignore that tail."""
    from modelexpress.refit.reshard.types import summarize_unsupported

    reasons = {
        f"model.layers.0.mlp.experts.{i}.gate_proj.weight": (
            f"unsupported op aten.index_copy_ on lazy "
            f"'model.layers.0.mlp.experts.{i}.gate_proj.weight' (chain=());"
        )
        for i in range(128)
    }
    reasons["odd"] = "unsupported op aten.mul on lazy 'odd' (chain=());"

    assert summarize_unsupported(reasons) == [
        ("unsupported op aten.index_copy_", 128),
        ("unsupported op aten.mul", 1),
    ]


def test_summarize_unsupported_accepts_none_for_all_causes():
    from modelexpress.refit.reshard.types import summarize_unsupported

    reasons = {
        f"source-{index}": f"cause-{index} on lazy 'source-{index}'"
        for index in range(4)
    }

    assert summarize_unsupported(reasons, limit=None) == [
        (f"cause-{index}", 1) for index in range(4)
    ]


def test_capture_feeds_slice_plan():
    """Compose capture -> slice-plan: real captured copies drive plan_pull. The
    row-parallel source (full [4,8], need cols [0:4]) is strided -> 4 runs
    covering exactly the needed 16 elements, landing contiguously in the dest."""
    from modelexpress.refit.reshard.slice_plan import Shard, plan_pull

    with torch.device("meta"):
        model = ToyModel()
    result = capture_geometry(model, _manifest())
    row = next(c for c in result.copies if c.src_name == "row")

    shard = Shard(shard_offset=(0, 0), shape=(4, 8), session="s0", addr=0, elsize=4)
    segs = plan_pull(
        row, global_shape=(4, 8), src_dtype=torch.float32, elsize=4, shards=[shard]
    )

    assert len(segs) == 4
    assert sum(s.nbytes for s in segs) == 16 * 4  # exactly the needed slice, no waste
    assert sorted(s.dst_byte for s in segs) == [0, 16, 32, 48]  # contiguous dest rows


def _plain_default_loader(param, loaded):
    param.data.copy_(loaded)


class DefaultLoaderModel(torch.nn.Module):
    """A param with NO custom ``weight_loader``, loaded via the framework's
    default loader looked up with ``getattr(param, "weight_loader", default)`` -
    exactly how vLLM loads norm/RMSNorm weights. Regression for the bug where
    such params were dropped as 'unattributed' (never pulled) unless
    ``default_weight_loader`` is passed to ``capture_geometry``."""

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(
            torch.empty(4)
        )  # deliberately no weight_loader attr

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, loaded in weights:
            p = params[name]
            weight_loader = getattr(p, "weight_loader", _plain_default_loader)
            weight_loader(p, loaded)


def test_default_loader_param_needs_default_weight_loader():
    manifest = [("norm", torch.float32, [4])]

    # Without default_weight_loader: the param has no weight_loader, so it is not
    # stamped; its copy_ fires with no attribution -> unattributed -> NOT captured
    # (this is the norm-weights-never-pulled bug the step-0 verify caught).
    with torch.device("meta"):
        model = DefaultLoaderModel()
    missed = capture_geometry(model, manifest)
    assert missed.copies == []
    assert missed.unattributed == 1

    # With default_weight_loader: the param is stamped and the copy is attributed.
    with torch.device("meta"):
        model = DefaultLoaderModel()
    got = capture_geometry(model, manifest, default_weight_loader=_plain_default_loader)
    assert [c.src_name for c in got.copies] == ["norm"]
    assert got.copies[0].param_name == "norm"
    assert got.copies[0].op_chain == () and got.copies[0].dest_shape == (4,)
    assert got.unattributed == 0


def test_capture_weights_records_the_converted_source_name():
    """Pre-built weights whose keys were renamed (native -> loader name): the
    recorded copy names the ORIGINAL source (the lazy's _name), not the key."""
    f32 = torch.float32
    with torch.device("meta"):
        model = ToyModel()
    lazies = build_lazy_weights(
        [("native.col", f32, [8, 4]), ("native.norm", f32, [4])]
    )
    converted = {"col": lazies["native.col"], "norm": lazies["native.norm"]}

    by_src = {c.src_name: c for c in capture_weights(model, converted).copies}

    assert by_src["native.col"].op_chain == (("narrow", (0, 0, 4), ()),)
    assert by_src["native.norm"].op_chain == ()
    assert "col" not in by_src  # keyed by source, not the loader name


def test_capture_weights_requires_a_shared_recorder():
    """Weights from separate build_lazy_weights calls don't share a recorder."""
    f32 = torch.float32
    with torch.device("meta"):
        model = ToyModel()
    a = build_lazy_weights([("col", f32, [8, 4])])
    b = build_lazy_weights([("norm", f32, [4])])

    with pytest.raises(ValueError, match="share one recorder"):
        capture_weights(model, {"col": a["col"], "norm": b["norm"]})


_CONVERT_MANIFEST = [
    ("model.router.gate.weight", torch.bfloat16, (2, 4)),
    ("model.experts.w13", torch.bfloat16, (3, 2, 4)),
]


def test_convert_source_weights_identity_when_none():
    weights = convert_source_weights(None, _CONVERT_MANIFEST)

    assert set(weights) == {"model.router.gate.weight", "model.experts.w13"}
    for name, lazy in weights.items():
        assert isinstance(lazy, LazyWeight)
        assert lazy._name == name and lazy._ops == ()


def test_convert_source_weights_traces_a_rename():
    def convert(sd):
        return {"model.mlp.gate.weight": sd["model.router.gate.weight"]}  # rename

    weights = convert_source_weights(convert, _CONVERT_MANIFEST)

    (name,) = weights  # only the renamed source survives; the other is dropped
    assert name == "model.mlp.gate.weight"
    renamed = weights[name]
    assert renamed._name == "model.router.gate.weight" and renamed._ops == ()


def test_convert_source_weights_shares_one_recorder():
    # capture_weights relies on this: every derived lazy shares the source recorder.
    weights = convert_source_weights(
        lambda sd: {"a": sd["model.router.gate.weight"], "b": sd["model.experts.w13"]},
        _CONVERT_MANIFEST,
    )
    assert len({id(w._recorder) for w in weights.values()}) == 1


def test_convert_source_weights_rejects_a_synthesized_tensor():
    def convert(sd):
        return {
            "a": sd["model.router.gate.weight"],
            "synth": torch.zeros(2, dtype=torch.bfloat16),
        }

    with pytest.raises(UnsupportedReshard, match="synthesized"):
        convert_source_weights(convert, _CONVERT_MANIFEST)


def test_convert_source_weights_rejects_a_dtype_cast():
    def convert(sd):
        return {"a": sd["model.router.gate.weight"].to(torch.float32)}

    with pytest.raises(UnsupportedReshard):
        convert_source_weights(convert, _CONVERT_MANIFEST)


_E, _I, _H = 3, 2, 4  # experts, intermediate, hidden


class ToyMoEModel(torch.nn.Module):
    """Router + a vLLM-style STACKED fused expert param.

    Per-expert HF weights land in ``w13[expert, half]`` (half 0 = gate, 1 = up),
    mirroring how vLLM stacks gate/up experts into one param; the router is a
    direct copy. Exercises the native->HF conversion all the way into the fused
    destination slots.
    """

    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Parameter(torch.empty(_E, _H))
        self.gate.weight_loader = lambda p, loaded: p.data.copy_(loaded)
        self.w13 = torch.nn.Parameter(torch.empty(_E, 2, _I, _H))
        self.w13.weight_loader = self._expert_loader

    def _expert_loader(self, param, loaded, expert_id, half):
        param.data[expert_id, half].copy_(loaded)

    def load_weights(self, weights):
        for name, loaded in weights:
            if name == "gate.weight":
                self.gate.weight_loader(self.gate, loaded)
                continue
            m = re.fullmatch(r"experts\.(\d+)\.(gate|up)_proj\.weight", name)
            expert_id, half = int(m.group(1)), (0 if m.group(2) == "gate" else 1)
            self.w13.weight_loader(self.w13, loaded, expert_id, half)


def _per_expert_rename_convert(sd):
    """Rename per-expert native projections into the HF names vLLM's loader
    consumes. The trainer here publishes each expert separately, so this is pure
    renaming - the stacked-source case that needs an unstack (select) is covered
    where the view ops are added."""
    out = {"gate.weight": sd["router.gate"]}  # rename
    for e in range(_E):
        out[f"experts.{e}.gate_proj.weight"] = sd[f"experts.{e}.w1"]
        out[f"experts.{e}.up_proj.weight"] = sd[f"experts.{e}.w3"]
    return out


def test_moe_convert_capture_plan_reconstructs_end_to_end():
    """Full sequence for an MoE trainer: build lazies -> native->HF convert ->
    vLLM loader capture -> plan_pull -> reconstruct, and assert every byte lands
    in the right fused slot."""
    f32 = torch.float32
    router = torch.arange(_E * _H, dtype=f32).reshape(_E, _H)
    w1 = {
        e: (torch.arange(_I * _H, dtype=f32) + 100 + e).reshape(_I, _H)
        for e in range(_E)
    }
    w3 = {
        e: (torch.arange(_I * _H, dtype=f32) + 200 + e).reshape(_I, _H)
        for e in range(_E)
    }
    src_buf = {"router.gate": router.reshape(-1)}
    src_shape = {"router.gate": (_E, _H)}
    for e in range(_E):
        src_buf[f"experts.{e}.w1"] = w1[e].reshape(-1)
        src_buf[f"experts.{e}.w3"] = w3[e].reshape(-1)
        src_shape[f"experts.{e}.w1"] = (_I, _H)
        src_shape[f"experts.{e}.w3"] = (_I, _H)
    manifest = [(n, f32, s) for n, s in src_shape.items()]

    with torch.device("meta"):
        model = ToyMoEModel()
    hf = convert_source_weights(_per_expert_rename_convert, manifest)
    copies = capture_weights(model, hf).copies

    dst = {
        "gate": torch.zeros(_E, _H).reshape(-1),
        "w13": torch.zeros(_E, 2, _I, _H).reshape(-1),
    }
    for copy in copies:
        shape = src_shape[copy.src_name]
        shard = Shard(
            shard_offset=(0,) * len(shape), shape=shape, session="s", addr=0, elsize=4
        )
        segs = plan_pull(
            copy, global_shape=shape, src_dtype=f32, elsize=4, shards=[shard]
        )
        source, dest = src_buf[copy.src_name], dst[copy.param_name]
        for seg in segs:
            s0, d0, n = seg.src_addr // 4, seg.dst_byte // 4, seg.nbytes // 4
            dest[d0 : d0 + n] = source[s0 : s0 + n]

    gate_out = dst["gate"].reshape(_E, _H)
    w13_out = dst["w13"].reshape(_E, 2, _I, _H)
    assert torch.equal(gate_out, router)
    for e in range(_E):
        assert torch.equal(w13_out[e, 0], w1[e])  # gate_proj -> half 0
        assert torch.equal(w13_out[e, 1], w3[e])  # up_proj -> half 1


def test_capture_weights_composes_a_select_prefix_with_the_loader_op():
    """A select in the conversion prefix survives into the recorded op-chain,
    ahead of the loader's own ops."""
    f32 = torch.float32
    with torch.device("meta"):
        model = ToyModel()
    lazies = build_lazy_weights([("native.stack", f32, [2, 4])])
    converted = {"norm": lazies["native.stack"].select(0, 1)}  # unstack row 1 -> [4]

    (copy,) = capture_weights(model, converted).copies

    assert copy.src_name == "native.stack"
    assert copy.op_chain == (("select", (0, 1), ()),)  # prefix; copy_ is the sink
    assert copy.dest_shape == (4,)


def test_convert_source_weights_traces_expert_unstack():
    def convert(sd):
        stacked = sd["model.experts.w13"]
        return {
            f"model.experts.{e}.gate_proj.weight": stacked.select(0, e)
            for e in range(stacked.shape[0])
        }

    weights = convert_source_weights(convert, _CONVERT_MANIFEST)

    expert = weights["model.experts.2.gate_proj.weight"]
    assert expert._name == "model.experts.w13"
    assert expert._ops == (("select", (0, 2), ()),)
    assert tuple(expert.shape) == (2, 4)


def test_stacked_moe_convert_capture_plan_reconstructs_end_to_end():
    """Stacked-source variant of the MoE e2e: the trainer publishes experts
    stacked and the conversion unstacks them with select. Each per-expert slab is
    a zero-copy sub-range that lands in the right fused destination slot."""
    f32 = torch.float32
    router = torch.arange(_E * _H, dtype=f32).reshape(_E, _H)
    w1 = (torch.arange(_E * _I * _H, dtype=f32) + 100).reshape(
        _E, _I, _H
    )  # stacked gate
    w3 = (torch.arange(_E * _I * _H, dtype=f32) + 200).reshape(_E, _I, _H)  # stacked up
    src_buf = {
        "router.gate": router.reshape(-1),
        "experts.w1": w1.reshape(-1),
        "experts.w3": w3.reshape(-1),
    }
    src_shape = {
        "router.gate": (_E, _H),
        "experts.w1": (_E, _I, _H),
        "experts.w3": (_E, _I, _H),
    }
    manifest = [(n, f32, s) for n, s in src_shape.items()]

    def convert(sd):
        out = {"gate.weight": sd["router.gate"]}
        for e in range(_E):
            out[f"experts.{e}.gate_proj.weight"] = sd["experts.w1"].select(0, e)
            out[f"experts.{e}.up_proj.weight"] = sd["experts.w3"].select(0, e)
        return out

    with torch.device("meta"):
        model = ToyMoEModel()
    hf = convert_source_weights(convert, manifest)
    copies = capture_weights(model, hf).copies

    dst = {
        "gate": torch.zeros(_E, _H).reshape(-1),
        "w13": torch.zeros(_E, 2, _I, _H).reshape(-1),
    }
    for copy in copies:
        shape = src_shape[copy.src_name]
        shard = Shard(
            shard_offset=(0,) * len(shape), shape=shape, session="s", addr=0, elsize=4
        )
        segs = plan_pull(
            copy, global_shape=shape, src_dtype=f32, elsize=4, shards=[shard]
        )
        source, dest = src_buf[copy.src_name], dst[copy.param_name]
        for seg in segs:
            s0, d0, n = seg.src_addr // 4, seg.dst_byte // 4, seg.nbytes // 4
            dest[d0 : d0 + n] = source[s0 : s0 + n]

    gate_out = dst["gate"].reshape(_E, _H)
    w13_out = dst["w13"].reshape(_E, 2, _I, _H)
    assert torch.equal(gate_out, router)
    for e in range(_E):
        assert torch.equal(w13_out[e, 0], w1[e])
        assert torch.equal(w13_out[e, 1], w3[e])


if __name__ == "__main__":
    test_capture_op_chains_and_dest_offsets()
    test_unsupported_op_falls_back_per_source()
    test_capture_feeds_slice_plan()
    test_default_loader_param_needs_default_weight_loader()
    print("OK: geometry capture + unsupported attribution + slice-plan compose")
