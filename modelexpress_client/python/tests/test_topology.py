# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for topology-aware source selection: the topology provider, the
``TopologyAwareSelector`` scoring, and a datacenter-topology simulation that
asserts the locality headline (drives the real selector, no GPU hardware)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from modelexpress import p2p_pb2
from modelexpress.source_selection import (
    RendezvousHashSelector,
    TopologyAwareSelector,
    get_selector,
)
from modelexpress.topology import local_topology, resolve_levels

LEVELS = "region,zone,block,rack,host"  # Grove ClusterTopology domains


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


@pytest.fixture(autouse=True)
def _no_dynamo_topology_dir(monkeypatch, tmp_path):
    # Neutralize the Dynamo topology-dir fallback so tests are hermetic; a test
    # that wants it points DYN_TOPOLOGY_MOUNT_PATH at a dir it populates.
    monkeypatch.setenv("DYN_TOPOLOGY_MOUNT_PATH", str(tmp_path / "absent"))


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


def test_resolve_levels_parses_and_trims():
    assert resolve_levels(" region , zone,rack ") == ["region", "zone", "rack"]


def test_resolve_levels_defaults_to_grove_order_when_unset(monkeypatch):
    from modelexpress.topology import GROVE_TOPOLOGY_DOMAINS

    monkeypatch.delenv("MX_P2P_TOPOLOGY_LEVELS", raising=False)
    assert resolve_levels() == list(GROVE_TOPOLOGY_DOMAINS)
    assert resolve_levels("") == list(GROVE_TOPOLOGY_DOMAINS)


def test_reads_dynamo_topology_dir(monkeypatch, tmp_path):
    # No MX_P2P_TOPOLOGY -> read the Dynamo operator's projected dir (one file
    # per domain, contents = value), the same source Dynamo's KV transfer reads.
    d = tmp_path / "topo"
    d.mkdir()
    (d / "block").write_text("b1\n")
    (d / "rack").write_text("r3")
    (d / "host").write_text("node7\n")
    (d / ".hidden").write_text("ignore")
    monkeypatch.setenv("DYN_TOPOLOGY_MOUNT_PATH", str(d))
    monkeypatch.delenv("MX_P2P_TOPOLOGY", raising=False)
    assert local_topology() == {"block": "b1", "rack": "r3", "host": "node7"}


def test_grove_domains_match_cluster_topology_enum():
    from modelexpress.topology import GROVE_TOPOLOGY_DOMAINS

    # Must match Grove ClusterTopology (clustertopologies.grove.io) spec.levels
    # domain enum exactly, or this node's map misaligns with the fleet.
    assert set(GROVE_TOPOLOGY_DOMAINS) == {
        "region",
        "zone",
        "datacenter",
        "block",
        "rack",
        "host",
        "numa",
    }


def test_resolve_levels_warns_on_non_grove_domain(caplog):
    import logging

    from modelexpress import topology as topo

    topo._warned_unknown.clear()
    with caplog.at_level(logging.WARNING, logger="modelexpress.topology"):
        levels = topo.resolve_levels("rack,rail")  # "rail" is not a Grove domain
    assert levels == ["rack", "rail"]  # non-Grove domain still used, not dropped
    assert any("rail" in rec.getMessage() for rec in caplog.records)


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
        monkeypatch, LEVELS, '{"region":"us","zone":"z1","block":"b1","rack":"r3"}'
    )
    cands = [
        _ref("far", {"region": "us", "zone": "z2", "block": "b9", "rack": "r9"}),
        _ref("block", {"region": "us", "zone": "z1", "block": "b1", "rack": "r9"}),
        _ref("rack", {"region": "us", "zone": "z1", "block": "b1", "rack": "r3"}),
        _ref("none", {"region": "eu"}),
    ]
    order = [c.mx_source_id for c in TopologyAwareSelector().order(cands, _ctx())]
    assert order[0] == "rack"  # deepest shared domain wins
    assert order[1] == "block"
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


def test_non_finite_load_weight_rejected(monkeypatch):
    # inf/nan would poison score() (inf * 0.0 -> nan) and break ordering; envs
    # rejects non-finite floats, so the weight falls back to 0.0.
    for bad in ("inf", "-inf", "nan"):
        monkeypatch.setenv("MX_P2P_TOPOLOGY_LOAD_WEIGHT", bad)
        assert TopologyAwareSelector().w_load == 0.0


def test_load_weight_falls_back_to_load_aware_env(monkeypatch):
    # One knob: when MX_P2P_TOPOLOGY_LOAD_WEIGHT is unset but the load_aware
    # weight is present (both features deployed), topology_aware uses it; an
    # explicit topology weight still overrides the shared knob.
    from modelexpress import envs as envs_mod

    monkeypatch.delenv("MX_P2P_TOPOLOGY_LOAD_WEIGHT", raising=False)
    monkeypatch.setattr(envs_mod, "MX_P2P_LOAD_WEIGHT", 0.8, raising=False)
    assert TopologyAwareSelector().w_load == 0.8
    monkeypatch.setenv("MX_P2P_TOPOLOGY_LOAD_WEIGHT", "0.0")
    assert TopologyAwareSelector().w_load == 0.0


# ---------------------------------------------------------------------------
# Datacenter-topology simulation (drives the real selector; no GPU hardware)
# ---------------------------------------------------------------------------


def _fleet(blocks, racks_per_block, hosts_per_rack):
    """Synthesize sources across a region/block/rack/host Grove hierarchy."""
    sources = []
    for b in range(blocks):
        for r in range(racks_per_block):
            for h in range(hosts_per_rack):
                sid = f"b{b}-r{r}-h{h}"
                sources.append(
                    _ref(
                        sid,
                        {
                            "region": "us",
                            "block": f"block{b}",
                            "rack": f"block{b}-rack{r}",
                            "host": sid,
                        },
                    )
                )
    return sources


def test_simulation_topology_localizes_vs_rendezvous(monkeypatch):
    """Multi-block/rack sim: topology_aware pulls from same-rack sources far more
    often than rendezvous_hash, and never worse on within-rack balance. This is
    the headline the design doc calls out (packed single-rack collapses to
    rendezvous; a topology-diverse spread is where the policy pays off)."""
    monkeypatch.setenv("MX_P2P_TOPOLOGY_LEVELS", LEVELS)
    sources = _fleet(blocks=4, racks_per_block=2, hosts_per_rack=4)  # 32 sources

    topo_same_rack = 0
    rdv_same_rack = 0
    # rack_id -> {source_id -> pick count}, to check balance *within* each rack.
    picks_by_rack: dict[str, dict[str, int]] = {}
    n_targets = 200
    for t in range(n_targets):
        # each target sits on some block/rack
        my_block = t % 4
        my_rack = (t // 4) % 2
        my_rack_id = f"block{my_block}-rack{my_rack}"
        monkeypatch.setenv(
            "MX_P2P_TOPOLOGY",
            f'{{"region":"us","block":"block{my_block}",'
            f'"rack":"{my_rack_id}","host":"target-{t}"}}',
        )
        ctx = _ctx(worker_id=f"target-{t}", worker_rank=t)
        topo_pick = get_selector("topology_aware").order(sources, ctx)[0]
        rdv_pick = get_selector("rendezvous_hash").order(sources, ctx)[0]
        if topo_pick.topology.get("rack") == my_rack_id:
            topo_same_rack += 1
        if rdv_pick.topology.get("rack") == my_rack_id:
            rdv_same_rack += 1
        rack = picks_by_rack.setdefault(my_rack_id, {})
        rack[topo_pick.mx_source_id] = rack.get(topo_pick.mx_source_id, 0) + 1

    # Headline: topology_aware keeps the pull local (same rack) far more often.
    assert topo_same_rack == n_targets  # every target's first choice is in-rack
    assert rdv_same_rack < n_targets  # rendezvous is topology-blind
    # Guardrail: within EACH rack the jitter tiebreak still spreads picks across
    # that rack's hosts -- no single source absorbs its whole rack. Every rack
    # (8 racks x ~25 targets each over 4 hosts) must use >= 2 distinct hosts, and
    # no host may take more than 60% of its rack's targets.
    assert len(picks_by_rack) == 8  # all racks served
    for rack_id, per_source in picks_by_rack.items():
        total = sum(per_source.values())
        assert len(per_source) >= 2, (rack_id, per_source)
        assert max(per_source.values()) <= 0.6 * total, (rack_id, per_source)
