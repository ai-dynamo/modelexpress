# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end robustness tests for load-aware source selection.

Runs the real client stack -- ``MxClient`` over a live gRPC channel into an
in-process ``P2pService`` server, through ``RdmaStrategy._find_source_instances``
and the registered selectors -- against a server that mimics the Rust
coordinator's load tracker (every ``GetMetadata`` counts as a selection;
``ListSources`` reports the count as ``active_transfers``).

Covers the robustness scenarios the policy must survive in production:
load steering over the wire, sequential fan-out spreading, old servers that
never populate the field, server restarts (fresh counters), extreme field
values, and rank filtering on the strategy path.
"""

from __future__ import annotations

import math
from collections import Counter
from concurrent import futures
from types import SimpleNamespace

import grpc
import pytest

from modelexpress import p2p_pb2, p2p_pb2_grpc
from modelexpress.client import MxClient
from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
from modelexpress.source_selection import (
    LoadAwareSelector,
    RendezvousHashSelector,
)

MODEL = "e2e-model"


class _LoadTrackingService(p2p_pb2_grpc.P2pServiceServicer):
    """In-process stand-in for the coordinator's selection-tracking behavior.

    Mirrors the Rust server: each found ``GetMetadata`` records a selection for
    that ``(mx_source_id, worker_id)``; ``ListSources`` surfaces the count as
    ``active_transfers`` (or always 0 when ``track_load=False``, emulating an
    older server without the tracker).
    """

    def __init__(self, workers: list[tuple[str, int]], track_load: bool = True):
        # workers: (worker_id, worker_rank) under one mx_source_id.
        self.mx_source_id = "e2esrcaaaabbbbcc"
        self.workers = workers
        self.track_load = track_load
        self.selections: Counter[str] = Counter()

    def ListSources(self, request, context):  # noqa: N802 (grpc method name)
        return p2p_pb2.ListSourcesResponse(
            instances=[
                p2p_pb2.SourceInstanceRef(
                    mx_source_id=self.mx_source_id,
                    worker_id=worker_id,
                    model_name=MODEL,
                    worker_rank=rank,
                    active_transfers=(
                        self.selections[worker_id] if self.track_load else 0
                    ),
                )
                for worker_id, rank in self.workers
            ]
        )

    def GetMetadata(self, request, context):  # noqa: N802
        self.selections[request.worker_id] += 1
        return p2p_pb2.GetMetadataResponse(
            found=True,
            worker=p2p_pb2.WorkerMetadata(worker_rank=0),
            mx_source_id=request.mx_source_id,
            worker_id=request.worker_id,
        )


def _serve(servicer: _LoadTrackingService):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    p2p_pb2_grpc.add_P2pServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, f"127.0.0.1:{port}"


@pytest.fixture
def four_source_server():
    servicer = _LoadTrackingService([(f"w{i}", 0) for i in range(4)])
    server, addr = _serve(servicer)
    yield servicer, addr
    server.stop(grace=None)


def _ctx(addr: str, worker_id: str = "target-0", worker_rank: int = 0):
    """Duck-typed LoadContext wired to a real MxClient over the live channel."""
    return SimpleNamespace(
        mx_client=MxClient(server_url=addr),
        worker_rank=worker_rank,
        global_rank=0,
        worker_id=worker_id,
        identity=p2p_pb2.SourceIdentity(model_name=MODEL),
    )


def _pick_order(addr: str, selector, worker_id: str) -> list[str]:
    """One selection round over the real wire: list -> rank -> preference list."""
    ctx = _ctx(addr, worker_id=worker_id)
    try:
        resp = ctx.mx_client.list_sources(
            identity=ctx.identity, status_filter=p2p_pb2.SOURCE_STATUS_READY
        )
        ordered = selector.order(list(resp.instances), ctx)
        return [c.worker_id for c in ordered]
    finally:
        ctx.mx_client.close()


def test_e2e_load_signal_steers_selection(four_source_server):
    servicer, addr = four_source_server
    # Establish the rendezvous favorite for this target over the live wire.
    favorite = _pick_order(addr, RendezvousHashSelector(), "target-0")[0]
    # Pile selections onto the favorite (as concurrent targets would).
    client = MxClient(server_url=addr)
    for _ in range(5):
        client.get_metadata(mx_source_id=servicer.mx_source_id, worker_id=favorite)
    client.close()

    # rendezvous_hash ignores the live load; load_aware demotes the busy source.
    assert _pick_order(addr, RendezvousHashSelector(), "target-0")[0] == favorite
    la_order = _pick_order(addr, LoadAwareSelector(), "target-0")
    assert la_order[0] != favorite
    assert la_order[-1] == favorite


def _run_fanout(servicer, addr: str, selector_factory, n_targets: int) -> Counter:
    servicer.selections.clear()
    picks: Counter[str] = Counter()
    client = MxClient(server_url=addr)
    try:
        for t in range(n_targets):
            chosen = _pick_order(addr, selector_factory(), f"target-{t}")[0]
            picks[chosen] += 1
            # The selection registers server-side, as RdmaStrategy's metadata
            # fetch does on the real coordinator.
            client.get_metadata(mx_source_id=servicer.mx_source_id, worker_id=chosen)
    finally:
        client.close()
    return picks


def test_e2e_sequential_fanout_spreads_load(four_source_server):
    servicer, addr = four_source_server
    n_targets, n_sources = 12, 4
    ideal = math.ceil(n_targets / n_sources)

    # Default weight is a hash/load blend: a strong hash preference may
    # override a small relative-load gap, so the spread is bounded, not exact.
    picks = _run_fanout(servicer, addr, LoadAwareSelector, n_targets)
    assert max(picks.values()) <= ideal + 1
    assert len(picks) == n_sources

    # A strong weight makes the load penalty dominate the unit-hash range,
    # giving strict least-loaded behavior: perfectly even assignment.
    picks = _run_fanout(
        servicer, addr, lambda: LoadAwareSelector(w_load=4.0), n_targets
    )
    assert max(picks.values()) == ideal
    assert len(picks) == n_sources


def test_e2e_old_server_without_load_field():
    # A server that never populates active_transfers (pre-upgrade coordinator):
    # load_aware must produce exactly the rendezvous_hash ordering.
    servicer = _LoadTrackingService([(f"w{i}", 0) for i in range(6)], track_load=False)
    server, addr = _serve(servicer)
    try:
        client = MxClient(server_url=addr)
        for _ in range(4):  # selections happen but are never surfaced
            client.get_metadata(mx_source_id=servicer.mx_source_id, worker_id="w0")
        client.close()
        la = _pick_order(addr, LoadAwareSelector(), "target-0")
        rh = _pick_order(addr, RendezvousHashSelector(), "target-0")
        assert la == rh
    finally:
        server.stop(grace=None)


def test_e2e_server_restart_resets_signal():
    # Selections against server #1 skew load_aware; a fresh server (empty
    # tracker, as after a crash/restart) must yield the rendezvous ordering
    # again, and steering must resume once new selections accumulate.
    workers = [(f"w{i}", 0) for i in range(4)]
    servicer1 = _LoadTrackingService(workers)
    server1, addr1 = _serve(servicer1)
    try:
        favorite = _pick_order(addr1, RendezvousHashSelector(), "target-0")[0]
        client = MxClient(server_url=addr1)
        for _ in range(5):
            client.get_metadata(mx_source_id=servicer1.mx_source_id, worker_id=favorite)
        client.close()
        assert _pick_order(addr1, LoadAwareSelector(), "target-0")[0] != favorite
    finally:
        server1.stop(grace=None)

    servicer2 = _LoadTrackingService(workers)  # fresh counters = restart
    server2, addr2 = _serve(servicer2)
    try:
        la = _pick_order(addr2, LoadAwareSelector(), "target-0")
        rh = _pick_order(addr2, RendezvousHashSelector(), "target-0")
        assert la == rh, "restarted server (zero load) must collapse to rendezvous"
        assert la[0] == favorite
        # Steering resumes as the fresh window repopulates.
        client = MxClient(server_url=addr2)
        for _ in range(5):
            client.get_metadata(mx_source_id=servicer2.mx_source_id, worker_id=favorite)
        client.close()
        assert _pick_order(addr2, LoadAwareSelector(), "target-0")[0] != favorite
    finally:
        server2.stop(grace=None)


def test_e2e_extreme_load_values(four_source_server):
    servicer, addr = four_source_server
    favorite = _pick_order(addr, RendezvousHashSelector(), "target-0")[0]
    # u32::MAX-scale counts must not break scoring or ordering.
    servicer.selections[favorite] = 2**32 - 1
    la_order = _pick_order(addr, LoadAwareSelector(), "target-0")
    assert set(la_order) == {f"w{i}" for i in range(4)}
    assert la_order[-1] == favorite


def test_e2e_strategy_path_rank_filtering_with_load(four_source_server, monkeypatch):
    # Full RdmaStrategy._find_source_instances over the wire: mixed ranks are
    # filtered to the target's rank before load-aware ranking.
    servicer, addr = four_source_server
    servicer.workers = [("w0", 0), ("w1", 1), ("w2", 0), ("w3", 1)]
    monkeypatch.setenv("MX_P2P_SOURCE_SELECTOR", "load_aware")

    ctx = _ctx(addr, worker_id="target-0", worker_rank=0)
    try:
        ordered = RdmaStrategy()._find_source_instances(ctx)
    finally:
        ctx.mx_client.close()
    assert {c.worker_id for c in ordered} == {"w0", "w2"}
    assert all(c.worker_rank == 0 for c in ordered)
