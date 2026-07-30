# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Transfer plan + pull end-to-end CORRECTNESS test.

The strongest check in the pipeline: reconstruct inference params purely from
sharded sources via capture -> plan -> pull, and assert they equal the ground
truth the engine's own loader produces from the full tensors. Uses the
in-memory reference transport (real ctypes byte moves over CPU addresses), so a
plan that reconstructs correctly here reconstructs correctly on the wire.

Exercises the same layouts as the geometry test: column block (contiguous),
row-parallel column slice (strided, multi-run), fused qkv (per-shard offsets),
and full copy - plus the unsupported-op fallback path.

Run: pytest tests/test_reshard_refit_transfer.py
"""

import torch

from modelexpress.refit.reshard.geometry import capture_geometry
from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard.transfer_plan import (
    SourceInfo,
    TransferPlan,
    exact_descriptors,
    execute_transfer,
    plan_threshold_curve,
    plan_transfer,
    session_distribution,
)
from modelexpress.refit.reshard.transport import InMemoryReferenceTransport

# Reuse the ToyModel + manifest from the geometry test (same package, same dir).
from tests.test_reshard_refit_geometry import ToyModel, _manifest

EL = 4  # float32


def _full_sources():
    """Distinct-valued full source tensors, one per manifest entry."""
    shapes = {name: shape for name, _dtype, shape in _manifest()}
    srcs = {}
    base = 0.0
    for name, shape in shapes.items():
        n = 1
        for s in shape:
            n *= s
        srcs[name] = (
            (base + torch.arange(n, dtype=torch.float32)).reshape(shape).contiguous()
        )
        base += n  # keep value ranges disjoint across tensors
    return srcs


def test_reshard_reconstructs_ground_truth():
    srcs = _full_sources()

    # Ground truth: the engine's own loader run on the FULL tensors.
    truth_model = ToyModel()
    truth_model.load_weights(list(srcs.items()))
    truth = {name: p.detach().clone() for name, p in truth_model.named_parameters()}

    # Reconstruct target: zero it, then fill only via the reshard pull.
    recon_model = ToyModel()
    for p in recon_model.parameters():
        torch.nn.init.zeros_(p)
    recon_params = dict(recon_model.named_parameters())

    # Capture geometry on a disposable meta twin (no storage touched).
    with torch.device("meta"):
        meta_model = ToyModel()
    capture = capture_geometry(meta_model, _manifest())

    # Publish each full source as a single contiguous shard covering the whole
    # tensor (shard buffer IS the source tensor; addr = its data_ptr()).
    sources = {}
    for name, tensor in srcs.items():
        shard = Shard(
            shard_offset=(0,) * tensor.dim(),
            shape=tuple(tensor.shape),
            session=name,
            addr=tensor.data_ptr(),
            elsize=EL,
        )
        sources[name] = SourceInfo(
            global_shape=tuple(tensor.shape),
            dtype=torch.float32,
            elsize=EL,
            shards=[shard],
        )

    plan = plan_transfer(capture, sources)
    assert plan.fallback == []
    assert plan.bytes_planned() > 0

    stats = execute_transfer(
        plan,
        resolve_param_ptr=lambda name: recon_params[name].data_ptr(),
        transport=InMemoryReferenceTransport(),
    )
    assert stats["bytes"] == plan.bytes_planned()

    # Every param reconstructed bit-for-bit from shards alone.
    for name in truth:
        assert torch.equal(recon_params[name], truth[name]), f"mismatch for {name}"

    # Keep source tensors alive until after the memmoves above.
    assert all(t.data_ptr() for t in srcs.values())


def test_strided_source_reconstructs_exactly():
    """Focus the row-parallel case: a strided column-slice must land correctly
    across its multiple runs (which dim-0-only shard schemes can't serve)."""
    srcs = _full_sources()

    truth_model = ToyModel()
    truth_model.load_weights(list(srcs.items()))
    truth_row = dict(truth_model.named_parameters())["row"].detach().clone()

    recon_model = ToyModel()
    for p in recon_model.parameters():
        torch.nn.init.zeros_(p)
    recon_row = dict(recon_model.named_parameters())["row"]

    with torch.device("meta"):
        meta_model = ToyModel()
    capture = capture_geometry(meta_model, _manifest())

    row_src = srcs["row"]
    shard = Shard(
        shard_offset=(0, 0),
        shape=tuple(row_src.shape),
        session="row",
        addr=row_src.data_ptr(),
        elsize=EL,
    )
    sources = {"row": SourceInfo(tuple(row_src.shape), torch.float32, EL, [shard])}

    # Only the 'row' copy is planned here (others have no source -> fallback).
    plan = plan_transfer(capture, sources)
    execute_transfer(
        plan,
        resolve_param_ptr=lambda name: dict(recon_model.named_parameters())[
            name
        ].data_ptr(),
        transport=InMemoryReferenceTransport(),
    )
    assert torch.equal(recon_row, truth_row)
    assert row_src.data_ptr()  # keep alive


def _whole_tensor_sources(srcs):
    """Publish each full source as one contiguous shard covering the tensor."""
    sources = {}
    for name, tensor in srcs.items():
        shard = Shard(
            shard_offset=(0,) * tensor.dim(),
            shape=tuple(tensor.shape),
            session=name,
            addr=tensor.data_ptr(),
            elsize=EL,
        )
        sources[name] = SourceInfo(
            global_shape=tuple(tensor.shape),
            dtype=torch.float32,
            elsize=EL,
            shards=[shard],
        )
    return sources


def test_threshold_curve_trades_descriptors_for_bytes():
    """The full-pull threshold trades wire bytes against NIXL descriptors, and
    the curve has to expose both sides for a cost model to replace the cliff."""
    srcs = _full_sources()
    with torch.device("meta"):
        meta_model = ToyModel()
    capture = capture_geometry(meta_model, _manifest())
    sources = _whole_tensor_sources(srcs)

    thresholds = [1, 2, 1024]
    curve = plan_threshold_curve(capture, sources, thresholds)

    # Order is preserved, so a logged curve reads as requested.
    assert [row["threshold"] for row in curve] == thresholds

    by_threshold = {row["threshold"]: row for row in curve}
    tight, loose = by_threshold[1], by_threshold[1024]

    # A tight threshold promotes strided copies to full pulls: fewer descriptors
    # on the wire, but redundant bytes. A loose one keeps the exact plan.
    assert tight["full_pulls"] > loose["full_pulls"]
    assert tight["descriptors"] <= loose["descriptors"]
    assert tight["extra_wire_bytes"] > loose["extra_wire_bytes"]
    assert loose["extra_wire_bytes"] == 0
    assert loose["full_pulls"] == 0

    # The useful payload is invariant across the sweep - only redundancy moves.
    # This is what makes two sweep points comparable at all.
    assert len({row["useful_bytes"] for row in curve}) == 1

    # Promotion must not silently drop work.
    assert all(row["fallback"] == 0 for row in curve)

    assert all(t.data_ptr() for t in srcs.values())  # keep alive


def test_threshold_curve_does_not_disturb_the_real_plan():
    srcs = _full_sources()
    with torch.device("meta"):
        meta_model = ToyModel()
    capture = capture_geometry(meta_model, _manifest())
    sources = _whole_tensor_sources(srcs)

    before = plan_transfer(capture, sources)
    plan_threshold_curve(capture, sources, [1, 4, 64, 4096])
    after = plan_transfer(capture, sources)

    assert before.descriptor_count() == after.descriptor_count()
    assert before.bytes_planned() == after.bytes_planned()
    assert before.exact_bytes == after.exact_bytes
    assert len(before.full_pulls) == len(after.full_pulls)

    assert all(t.data_ptr() for t in srcs.values())  # keep alive


def test_session_distribution_accounts_every_planned_byte():
    srcs = _full_sources()
    with torch.device("meta"):
        meta_model = ToyModel()
    capture = capture_geometry(meta_model, _manifest())
    sources = _whole_tensor_sources(srcs)
    plan = plan_transfer(capture, sources)

    dist = session_distribution(plan)

    # Every byte and descriptor the plan will read is attributed to some session.
    assert sum(v["bytes"] for v in dist.values()) == plan.bytes_planned()
    assert sum(v["descriptors"] for v in dist.values()) == plan.descriptor_count()
    # Sessions here are per-tensor, so a multi-source plan must span several.
    assert len(dist) > 1

    assert all(t.data_ptr() for t in srcs.values())  # keep alive


def test_session_distribution_is_empty_for_an_empty_plan():
    assert session_distribution(TransferPlan()) == {}


def test_exact_descriptors_match_what_execute_transfer_reads():
    """The fused wire path builds descriptors via exact_descriptors() instead of
    letting execute_transfer read them, so the two must agree exactly."""
    srcs = _full_sources()
    with torch.device("meta"):
        meta_model = ToyModel()
    capture = capture_geometry(meta_model, _manifest())
    sources = _whole_tensor_sources(srcs)
    plan = plan_transfer(capture, sources)

    model = ToyModel()
    params = dict(model.named_parameters())
    resolve = lambda name: params[name].data_ptr()  # noqa: E731

    built = exact_descriptors(plan, resolve)

    recorded = []

    class _Recorder:
        def read(self, descriptors):
            recorded.extend(descriptors)

    stats = execute_transfer(plan, resolve_param_ptr=resolve, transport=_Recorder())

    assert built == recorded
    assert stats["segments"] == len(built)
    assert stats["bytes"] == sum(d.nbytes for d in built)

    assert all(t.data_ptr() for t in srcs.values())  # keep alive


def test_unsupported_source_routes_to_fallback():
    srcs = _full_sources()
    srcs["bad"] = torch.arange(16, dtype=torch.float32).reshape(4, 4).contiguous()

    with torch.device("meta"):
        meta_model = ToyModel(with_bad=True)
    capture = capture_geometry(meta_model, _manifest(with_bad=True))

    sources = {}
    for name, tensor in srcs.items():
        shard = Shard(
            (0,) * tensor.dim(), tuple(tensor.shape), name, tensor.data_ptr(), EL
        )
        sources[name] = SourceInfo(tuple(tensor.shape), torch.float32, EL, [shard])

    plan = plan_transfer(capture, sources)
    assert "bad" in plan.fallback
    # The good sources are still planned, not dropped.
    planned_params = {seg.param_name for seg in plan.segments}
    assert {"col", "row", "qkv", "norm"} <= planned_params
    assert all(t.data_ptr() for t in srcs.values())


if __name__ == "__main__":
    test_reshard_reconstructs_ground_truth()
    test_strided_source_reconstructs_exactly()
    test_unsupported_source_routes_to_fallback()
    print("OK: reshard reconstructs ground truth + strided + fallback")


# ------------------------------------------------- force_full_pull (gate coverage)
#
# `verify_full_pulls` can only digest a source whose whole shard lands in a staging
# buffer, because an exact-fetch segment is scattered straight into a live param and
# digesting it would mean digesting the destination layout instead of the source
# shard. On Topology B that left 6192 of 18867 sources checked - the row-parallel
# tensors that happened to be strided at generator TP2 - and said nothing about the
# other 12675, which include every norm, the gate/up projections, and all attention
# except o_proj.
#
# force_full_pull buys gate coverage with wire volume: a correctness run can afford
# that and a timing run cannot, which is why it defaults off and why one of the tests
# below pins the default.
#
# Read a pass under this flag narrowly. It proves the publisher's bytes for every
# source arrive intact. It does NOT verify the exact-fetch path, because forcing full
# pulls replaces the segment planning rather than checking it.
def _every_source_exact():
    """A case where nothing would be promoted, so promotion is unambiguous.

    The threshold is passed explicitly and loosely rather than left at the default:
    the point of the baseline is that `full_pulls` is empty for reasons the test
    controls, not because the toy model happens to sit under the default cliff.
    """
    srcs = _full_sources()
    with torch.device("meta"):
        meta_model = ToyModel()
    capture = capture_geometry(meta_model, _manifest())
    return capture, _whole_tensor_sources(srcs), srcs


LOOSE = 1024  # high enough that no copy in the toy model is ever promoted


def test_force_full_pull_promotes_what_the_threshold_would_not():
    capture, sources, srcs = _every_source_exact()

    plain = plan_transfer(capture, sources, max_segments_per_copy=LOOSE)
    assert plain.full_pulls == [], "baseline must leave everything on the exact path"
    assert plain.segments, "baseline must actually fetch something"

    forced = plan_transfer(
        capture, sources, max_segments_per_copy=LOOSE, force_full_pull=True
    )
    assert forced.full_pulls, "forced run promoted nothing"
    assert forced.segments == [], "nothing may remain on the exact path"

    assert all(t.data_ptr() for t in srcs.values())  # keep alive


def test_force_full_pull_never_loses_a_source():
    """Promotion must not drop work: gate coverage is worthless if it narrows the
    refit. This is the invariant Bug 8 violated by a different route."""
    capture, sources, srcs = _every_source_exact()

    def covered(plan):
        """Destination params the plan will fill, by whichever route.

        Keyed on the destination rather than the source: the question is whether
        the engine ends up with every parameter written, which is what Bug 8
        broke, and a promoted source serves its copies through
        ``FullPullSource.copies`` instead of through ``segments``.
        """
        return {segment.param_name for segment in plan.segments} | {
            copy.param_name for fp in plan.full_pulls for copy in fp.copies
        }

    plain = plan_transfer(capture, sources, max_segments_per_copy=LOOSE)
    forced = plan_transfer(
        capture, sources, max_segments_per_copy=LOOSE, force_full_pull=True
    )
    assert covered(forced) == covered(plain)
    assert forced.fallback == []
    assert all(t.data_ptr() for t in srcs.values())


def test_force_full_pull_costs_wire_and_that_is_the_trade():
    """Pinned so nobody enables this on a timing run without noticing."""
    capture, sources, srcs = _every_source_exact()
    plain = plan_transfer(capture, sources, max_segments_per_copy=LOOSE)
    forced = plan_transfer(
        capture, sources, max_segments_per_copy=LOOSE, force_full_pull=True
    )
    assert forced.bytes_planned() >= plain.bytes_planned()
    assert forced.extra_wire_bytes() >= plain.extra_wire_bytes()
    assert all(t.data_ptr() for t in srcs.values())


def test_force_full_pull_defaults_off(monkeypatch):
    """Every published benchmark row depends on this default staying cheap."""
    monkeypatch.delenv("MX_RESHARD_FORCE_FULL_PULL", raising=False)
    capture, sources, srcs = _every_source_exact()
    plan = plan_transfer(capture, sources, max_segments_per_copy=LOOSE)
    assert plan.full_pulls == []
    assert all(t.data_ptr() for t in srcs.values())


def test_force_full_pull_reads_the_environment(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_FORCE_FULL_PULL", "1")
    capture, sources, srcs = _every_source_exact()
    plan = plan_transfer(capture, sources, max_segments_per_copy=LOOSE)
    assert plan.full_pulls, "env flag did not take effect"
    assert all(t.data_ptr() for t in srcs.values())


def test_an_explicit_argument_beats_the_environment(monkeypatch):
    """So a timing harness can hard-disable it regardless of ambient env."""
    monkeypatch.setenv("MX_RESHARD_FORCE_FULL_PULL", "1")
    capture, sources, srcs = _every_source_exact()
    plan = plan_transfer(
        capture, sources, max_segments_per_copy=LOOSE, force_full_pull=False
    )
    assert plan.full_pulls == []
    assert all(t.data_ptr() for t in srcs.values())


def _single_shard_sources(srcs):
    """Each full source published as one contiguous whole-tensor shard."""
    sources = {}
    for name, tensor in srcs.items():
        sources[name] = SourceInfo(
            global_shape=tuple(tensor.shape),
            dtype=torch.float32,
            elsize=EL,
            shards=[
                Shard(
                    shard_offset=(0,) * tensor.dim(),
                    shape=tuple(tensor.shape),
                    session=name,
                    addr=tensor.data_ptr(),
                    elsize=EL,
                )
            ],
        )
    return sources


def test_plan_records_which_sources_feed_each_destination():
    """The plan is the only place this mapping exists whole: by the time planning
    ends, ``segments`` is a flat list of byte runs and the copy that produced each
    one is gone. Without it the receiver cannot pair a destination digest with the
    publisher claims behind it, which is the only way to audit one run on its own."""
    srcs = _full_sources()
    with torch.device("meta"):
        meta_model = ToyModel()
    capture = capture_geometry(meta_model, _manifest())

    plan = plan_transfer(capture, _single_shard_sources(srcs))

    assert plan.dest_sources, "no destination -> source mapping was recorded"
    # Every param the plan will write is named, and every source it names is real.
    for param, names in plan.dest_sources.items():
        assert names == sorted(names), f"{param} sources are not in stable order"
        for name in names:
            assert name in srcs
    assert all(t.data_ptr() for t in srcs.values())


def test_plan_records_the_mapping_for_forced_full_pulls_too():
    """The reference arm routes everything through the full-pull path. If the
    mapping were only built for exact segments, that arm would emit no source
    digests and its audit would silently degrade to no evidence."""
    srcs = _full_sources()
    with torch.device("meta"):
        meta_model = ToyModel()
    capture = capture_geometry(meta_model, _manifest())
    sources = _single_shard_sources(srcs)

    exact = plan_transfer(capture, sources, force_full_pull=False)
    forced = plan_transfer(capture, sources, force_full_pull=True)

    assert forced.dest_sources == exact.dest_sources
    assert all(t.data_ptr() for t in srcs.values())


def test_plan_omits_fallback_sources_from_the_mapping():
    """Fallbacks are not refit, so they have no publisher claim to pair against and
    must not appear as covered."""
    srcs = _full_sources()
    with torch.device("meta"):
        meta_model = ToyModel()
    capture = capture_geometry(meta_model, _manifest())

    sources = _single_shard_sources(srcs)
    dropped = sorted(sources)[0]
    del sources[dropped]

    plan = plan_transfer(capture, sources)

    assert dropped in plan.fallback
    for names in plan.dest_sources.values():
        assert dropped not in names
    assert all(t.data_ptr() for t in srcs.values())


def test_exact_replay_implies_force_full_pull(monkeypatch):
    """The replay gate reads the exact plan back out of the staging buffer, so an
    unstaged source cannot be replayed. Two independent switches would let a run
    look healthy while reporting `checked: 0`."""
    monkeypatch.delenv("MX_RESHARD_FORCE_FULL_PULL", raising=False)
    monkeypatch.setenv("MX_RESHARD_EXACT_REPLAY", "1")
    srcs = _full_sources()
    with torch.device("meta"):
        meta_model = ToyModel()
    capture = capture_geometry(meta_model, _manifest())

    plan = plan_transfer(capture, _single_shard_sources(srcs), max_segments_per_copy=LOOSE)

    assert plan.forced_full_pull is True
    assert plan.full_pulls, "nothing was staged, so nothing could be replayed"
    assert all(t.data_ptr() for t in srcs.values())


def test_exact_replay_does_not_override_an_explicit_argument(monkeypatch):
    """A caller asking for the exact plan outright still gets it; only the
    environment-derived default is widened."""
    monkeypatch.setenv("MX_RESHARD_EXACT_REPLAY", "1")
    srcs = _full_sources()
    with torch.device("meta"):
        meta_model = ToyModel()
    capture = capture_geometry(meta_model, _manifest())

    plan = plan_transfer(
        capture,
        _single_shard_sources(srcs),
        force_full_pull=False,
        max_segments_per_copy=LOOSE,
    )

    assert plan.forced_full_pull is False
    assert plan.full_pulls == []
    assert all(t.data_ptr() for t in srcs.values())
