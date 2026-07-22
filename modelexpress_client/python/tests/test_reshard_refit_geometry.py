# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Geometry-capture mechanism test - no engine needed.

A tiny model with column-parallel, row-parallel, fused (qkv-style), unsharded,
and one UNSUPPORTED-op weight_loader. Exercises: contiguous/strided/fused-offset
op-chain capture, full copies, and the per-source graceful fallback (an
unsupported op knocks out only that source, not the whole bake). Runs in any
torch env: pytest tests/test_reshard_refit_geometry.py
"""

import torch

from modelexpress.refit.reshard.geometry import capture_geometry

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
        # Arithmetic is not a pure view/slice op -> UnsupportedReshard -> fallback.
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


if __name__ == "__main__":
    test_capture_op_chains_and_dest_offsets()
    test_unsupported_op_falls_back_per_source()
    test_capture_feeds_slice_plan()
    test_default_loader_param_needs_default_weight_loader()
    print("OK: geometry capture + per-source fallback + slice-plan compose")
