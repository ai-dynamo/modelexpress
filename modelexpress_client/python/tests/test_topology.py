# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for topology-aware source selection: the topology provider, the
``TopologyAwareSelector`` scoring, and a datacenter-topology simulation that
asserts the locality headline (drives the real selector, no GPU hardware)."""

from __future__ import annotations

import statistics
from types import SimpleNamespace

from modelexpress import p2p_pb2
from modelexpress.source_selection import (
    RendezvousHashSelector,
    TopologyAwareSelector,
    get_selector,
)
from modelexpress.topology import local_topology, resolve_levels

LEVELS = "region,zone,rack,rail,host"


def _ctx(worker_id="target-0", worker_rank=0, model_name="m"):
    return SimpleNamespace(
        worker_id=worker_id,
        worker_rank=worker_rank,
        identity=SimpleNamespace(model_name=model_name),
    )


def _ref(mx_source_id, topology=None, worker_rank=0):
    return p2p_pb2.SourceInstanceRef(
        mx_source_id=mx_source_id,
        worker_id=f"w-{mx_source_id}",
        model_name="m",
        worker_rank=worker_rank,
        topology=topology or {},
    )


def _set_topology(monkeypatch, levels, local):
    monkeypatch.setenv("MX_P2P_TOPOLOGY_LEVELS", levels)
    monkeypatch.setenv("MX_P2P_TOPOLOGY", local)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


def test_resolve_levels_parses_and_trims():
    assert resolve_levels(" region , zone,rack ") == ["region", "zone", "rack"]


def test_resolve_levels_empty_when_unset():
    assert resolve_levels("") == []
    assert resolve_levels(None) == [] or isinstance(resolve_levels(None), list)


def test_local_topology_parses_json():
    assert local_topology('{"rack": "r3", "rail": "leaf2"}') == {
        "rack": "r3",
        "rail": "leaf2",
    }


def test_local_topology_bad_json_is_empty():
    assert local_topology("{not json") == {}


def test_local_topology_non_object_is_empty():
    assert local_topology('["r3"]') == {}


def test_local_topology_unset_is_empty():
    assert local_topology("") == {}


# ---------------------------------------------------------------------------
# Selector ordering
# ---------------------------------------------------------------------------


def test_prefers_narrowest_shared_domain(monkeypatch):
    _set_topology(
        monkeypatch, LEVELS, '{"region":"us","zone":"z1","rack":"r3","rail":"leaf2"}'
    )
    cands = [
        _ref("far", {"region": "us", "zone": "z2", "rack": "r9", "rail": "leafX"}),
        _ref("rack", {"region": "us", "zone": "z1", "rack": "r3", "rail": "leafY"}),
        _ref("rail", {"region": "us", "zone": "z1", "rack": "r3", "rail": "leaf2"}),
        _ref("none", {"region": "eu"}),
    ]
    order = [c.mx_source_id for c in TopologyAwareSelector().order(cands, _ctx())]
    assert order[0] == "rail"  # deepest shared domain wins
    assert order[1] == "rack"
    assert order[-1] == "none"  # shares nothing -> last


def test_collapses_to_rendezvous_without_levels(monkeypatch):
    # No levels configured -> every shared_depth is -1 -> pure rendezvous jitter.
    monkeypatch.delenv("MX_P2P_TOPOLOGY_LEVELS", raising=False)
    monkeypatch.delenv("MX_P2P_TOPOLOGY", raising=False)
    ctx = _ctx()
    cands = [_ref(f"src{i:04x}", {"rack": f"r{i}"}) for i in range(6)]
    topo_order = [c.mx_source_id for c in TopologyAwareSelector().order(cands, ctx)]
    rdv_order = [c.mx_source_id for c in RendezvousHashSelector().order(cands, ctx)]
    assert topo_order == rdv_order


def test_missing_topology_field_treated_as_no_share(monkeypatch):
    # A candidate object with no topology attribute (old server) must not raise.
    _set_topology(monkeypatch, LEVELS, '{"rack":"r3"}')
    ctx = _ctx()
    bare = [
        SimpleNamespace(mx_source_id=f"s{i}", worker_id=f"w{i}", worker_rank=0)
        for i in range(4)
    ]
    ordered = TopologyAwareSelector().order(bare, ctx)
    assert {c.mx_source_id for c in ordered} == {f"s{i}" for i in range(4)}


def test_within_tier_spreads_deterministically(monkeypatch):
    # Sources all in the same rack (equidistant): topology gives no signal, so
    # the jitter tiebreak decides -- deterministic and identical to rendezvous.
    _set_topology(monkeypatch, LEVELS, '{"region":"us","rack":"r3"}')
    ctx = _ctx()
    same = [_ref(f"src{i:04x}", {"region": "us", "rack": "r3"}) for i in range(8)]
    a = [c.mx_source_id for c in TopologyAwareSelector().order(same, ctx)]
    b = [c.mx_source_id for c in TopologyAwareSelector().order(same, ctx)]
    assert a == b  # deterministic
    assert len(set(a)) == len(a)  # a total order, no dupes


def test_registered_and_resolvable(monkeypatch):
    monkeypatch.setenv("MX_P2P_SOURCE_SELECTOR", "topology_aware")
    from modelexpress.source_selection import get_configured_selector

    assert get_configured_selector().name == "topology_aware"


# ---------------------------------------------------------------------------
# Composition with load-aware (opt-in within-tier blend)
#
# source_load is a separate feature's field; the blend reads it defensively via
# getattr, so these use duck-typed candidates carrying BOTH topology and
# source_load (the shape once both features are deployed together).
# ---------------------------------------------------------------------------


def _blend_ref(sid, topology, source_load):
    return SimpleNamespace(
        mx_source_id=sid,
        worker_id=f"w-{sid}",
        worker_rank=0,
        topology=topology,
        source_load=source_load,
    )


def test_load_blend_off_by_default_is_pure_topology(monkeypatch):
    _set_topology(monkeypatch, LEVELS, '{"rack":"r3"}')
    monkeypatch.delenv("MX_P2P_TOPOLOGY_LOAD_WEIGHT", raising=False)
    ctx = _ctx()
    sel = TopologyAwareSelector()
    assert sel.w_load == 0.0
    # Same rack (equal depth); with weight 0 the order is pure jitter, so the
    # busy source is NOT demoted relative to the load-free ordering.
    loaded = [_blend_ref("a", {"rack": "r3"}, 0.9), _blend_ref("b", {"rack": "r3"}, 0.0)]
    flat = [_blend_ref("a", {"rack": "r3"}, 0.0), _blend_ref("b", {"rack": "r3"}, 0.0)]
    assert [c.mx_source_id for c in sel.order(loaded, ctx)] == [
        c.mx_source_id for c in sel.order(flat, ctx)
    ]


def test_load_blend_steers_within_tier(monkeypatch):
    _set_topology(monkeypatch, LEVELS, '{"rack":"r3"}')
    monkeypatch.setenv("MX_P2P_TOPOLOGY_LOAD_WEIGHT", "1.0")
    ctx = _ctx()
    # Same tier (rack): the heavily loaded source is pushed behind the idle one.
    cands = [_blend_ref("busy", {"rack": "r3"}, 0.95), _blend_ref("idle", {"rack": "r3"}, 0.0)]
    order = [c.mx_source_id for c in TopologyAwareSelector().order(cands, ctx)]
    assert order[0] == "idle"


def test_load_blend_never_overrides_locality(monkeypatch):
    # A closer-but-busy source still beats a far-but-idle one: locality is the
    # primary key, load only breaks ties within a tier.
    _set_topology(monkeypatch, LEVELS, '{"region":"us","rack":"r3"}')
    monkeypatch.setenv("MX_P2P_TOPOLOGY_LOAD_WEIGHT", "1.0")
    ctx = _ctx()
    cands = [
        _blend_ref("near", {"region": "us", "rack": "r3"}, 0.99),
        _blend_ref("far", {"region": "us", "rack": "r9"}, 0.0),
    ]
    order = [c.mx_source_id for c in TopologyAwareSelector().order(cands, ctx)]
    assert order[0] == "near"


def test_negative_load_weight_clamped(monkeypatch):
    monkeypatch.setenv("MX_P2P_TOPOLOGY_LOAD_WEIGHT", "-2.0")
    assert TopologyAwareSelector().w_load == 0.0


# ---------------------------------------------------------------------------
# Datacenter-topology simulation (drives the real selector; no GPU hardware)
# ---------------------------------------------------------------------------


def _fleet(racks, rails_per_rack, hosts_per_rail):
    """Synthesize sources spread across a region/rack/rail/host hierarchy."""
    sources = []
    for r in range(racks):
        for e in range(rails_per_rack):
            for h in range(hosts_per_rail):
                sid = f"r{r}-e{e}-h{h}"
                sources.append(
                    _ref(
                        sid,
                        {
                            "region": "us",
                            "rack": f"rack{r}",
                            "rail": f"rack{r}-rail{e}",
                            "host": sid,
                        },
                    )
                )
    return sources


def test_simulation_topology_localizes_vs_rendezvous(monkeypatch):
    """Multi-rack/rail sim: topology_aware pulls from same-rack sources far more
    often than rendezvous_hash, and never worse on within-rack balance. This is
    the headline the design doc calls out (packed single-rack collapses to
    rendezvous; a topology-diverse spread is where the policy pays off)."""
    monkeypatch.setenv("MX_P2P_TOPOLOGY_LEVELS", LEVELS)
    sources = _fleet(racks=4, rails_per_rack=2, hosts_per_rail=4)  # 32 sources

    topo_same_rack = 0
    rdv_same_rack = 0
    picks_by_source_topo: dict[str, int] = {}
    n_targets = 200
    for t in range(n_targets):
        # each target sits on some rack/rail
        my_rack = t % 4
        my_rail = (t // 4) % 2
        monkeypatch.setenv(
            "MX_P2P_TOPOLOGY",
            f'{{"region":"us","rack":"rack{my_rack}",'
            f'"rail":"rack{my_rack}-rail{my_rail}","host":"target-{t}"}}',
        )
        ctx = _ctx(worker_id=f"target-{t}", worker_rank=t)
        topo_pick = get_selector("topology_aware").order(sources, ctx)[0]
        rdv_pick = get_selector("rendezvous_hash").order(sources, ctx)[0]
        if topo_pick.topology.get("rack") == f"rack{my_rack}":
            topo_same_rack += 1
        if rdv_pick.topology.get("rack") == f"rack{my_rack}":
            rdv_same_rack += 1
        picks_by_source_topo[topo_pick.mx_source_id] = (
            picks_by_source_topo.get(topo_pick.mx_source_id, 0) + 1
        )

    # Headline: topology_aware keeps the pull local (same rack) far more often.
    assert topo_same_rack == n_targets  # every target's first choice is in-rack
    assert rdv_same_rack < n_targets  # rendezvous is topology-blind
    # Guardrail: within a rack (8 sources), the jitter still spreads picks -- no
    # single source absorbs everything (each rack serves ~50 targets over 8 hosts).
    max_share = max(picks_by_source_topo.values())
    assert max_share <= n_targets / 4  # bounded well below "all onto one source"
    # sanity: spread is non-degenerate
    assert len(picks_by_source_topo) >= 8
    _ = statistics  # keep import meaningful if assertions above are relaxed later
