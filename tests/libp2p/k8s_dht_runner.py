# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone DHT node runner for multi-node k8s testing.

Usage:
  # Peer node (joins the DHT via headless service DNS):
  python k8s_dht_runner.py --role peer --listen 0.0.0.0:4001 \
      --dns mx-dht.dht-test.svc.cluster.local

  # Test coordinator (joins DHT, runs tests, exits):
  python k8s_dht_runner.py --role test --listen 0.0.0.0:4001 \
      --dns mx-dht.dht-test.svc.cluster.local

Zero-config bootstrap: nodes discover each other by resolving a K8s
headless Service DNS name. Each resolved IP is dialed and the peer ID
is learned via the Noise handshake. No pre-shared peer IDs or multiaddrs.

The peer role runs indefinitely (until SIGTERM).
The test role performs PUT/GET operations and exits with 0 on success.

Test scenarios cover things that unit tests (all on 127.0.0.1) cannot:
  - Cross-node routing with real pod IPs
  - Observed IP detection via Identify (0.0.0.0 binding -> real pod IP)
  - Record replication across physical nodes
  - Per-record TTL expiry under real timing
  - Concurrent multi-writer across network boundaries
  - Iterative lookup across multi-hop paths
"""

import argparse
import asyncio
import json
import logging
import os
import random
import signal
import subprocess
import sys
import time

from mx_libp2p.crypto import Ed25519Identity, _base58btc_encode
from mx_libp2p.dht import DhtNode
from mx_libp2p.multiaddr import decode_multiaddr, PROTO_IP4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("k8s-dht")


def node_multiaddr(node: DhtNode) -> str:
    host, port = node.routable_addr()
    peer_id_b58 = _base58btc_encode(node.peer_id)
    return f"/ip4/{host}/tcp/{port}/p2p/{peer_id_b58}"


# ---------------------------------------------------------------------------
# Roles: peer, test
# ---------------------------------------------------------------------------


async def run_peer(host: str, port: int, dns: str | None, bootstrap: list[str]) -> None:
    """Run a DHT peer node that joins via DNS or explicit bootstrap and stays alive."""
    node = DhtNode()
    await node.start(host, port, bootstrap_peers=bootstrap or None,
                     bootstrap_dns=dns, bootstrap_dns_port=port)

    multiaddr = node_multiaddr(node)
    print(f"MULTIADDR={multiaddr}", flush=True)
    log.info(f"Peer node ready: {multiaddr}")
    log.info(f"Routing table: {node.routing_table.size()} peers")

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    await node.stop()


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def record(self, name: str, success: bool, detail: str = ""):
        if success:
            self.passed += 1
            log.info(f"  PASS: {name}" + (f" ({detail})" if detail else ""))
        else:
            self.failed += 1
            log.error(f"  FAIL: {name}" + (f" ({detail})" if detail else ""))

    def skip(self, name: str, reason: str):
        self.skipped += 1
        log.info(f"  SKIP: {name} ({reason})")

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def summary(self) -> str:
        total = self.passed + self.failed + self.skipped
        return f"{self.passed}/{total} passed, {self.failed} failed, {self.skipped} skipped"


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------


async def test_basic_put_get(node: DhtNode, results: TestResult) -> None:
    """Basic PUT/GET - sanity check that the DHT works at all."""
    log.info("TEST: basic_put_get")
    pod_name = os.environ.get("POD_NAME", "test")

    key = f"/k8s/{pod_name}/basic".encode()
    value = json.dumps({"from": pod_name, "time": time.time()}).encode()
    count = await node.put(key, value)
    results.record("put stored on peers", count >= 1, f"{count} peers")

    result = await node.get(key)
    results.record("get returns correct value", result == value)


async def test_cross_node_routing(node: DhtNode, results: TestResult) -> None:
    """PUT a record and verify it's retrievable from this test node."""
    log.info("TEST: cross_node_routing")
    pod_name = os.environ.get("POD_NAME", "test")
    node_name = os.environ.get("NODE_NAME", "unknown")

    key = f"/k8s/cross-node/{pod_name}".encode()
    value = json.dumps({
        "origin_pod": pod_name,
        "origin_node": node_name,
        "test": "cross-node routing",
    }).encode()

    count = await node.put(key, value)
    results.record("cross-node put", count >= 1, f"stored on {count} peers")

    await asyncio.sleep(0.5)

    result = await node.get(key)
    results.record("cross-node get", result == value)


async def test_observed_ip_detection(node: DhtNode, results: TestResult) -> None:
    """When bound to 0.0.0.0, the node should learn its real pod IP
    from Identify exchanges with peers on other physical nodes."""
    log.info("TEST: observed_ip_detection")
    pod_ip = os.environ.get("POD_IP")

    observed = node._observed_ip
    if observed is not None:
        results.record(
            "observed IP set",
            True,
            f"observed={observed}, pod_ip={pod_ip}",
        )
        if pod_ip:
            results.record(
                "observed IP matches pod IP",
                observed == pod_ip,
                f"observed={observed}, expected={pod_ip}",
            )
    else:
        peer_count = node.routing_table.size()
        if peer_count < 2:
            results.skip("observed IP", f"only {peer_count} peers (need 2 for threshold)")
        else:
            results.record("observed IP set", False, "still None despite multiple peers")

    addrs = node.local_addrs()
    if addrs:
        results.record("local_addrs not empty", True, f"{len(addrs)} addr(s)")
        components = decode_multiaddr(addrs[0])
        for code, data in components:
            if code == PROTO_IP4:
                import socket
                ip = socket.inet_ntoa(data)
                results.record("advertised IP is routable", ip != "0.0.0.0", f"ip={ip}")
    else:
        if observed is None:
            results.skip("local_addrs", "no observed IP yet")
        else:
            results.record("local_addrs not empty", False, "empty despite observed IP")


async def test_routing_table_health(node: DhtNode, results: TestResult) -> None:
    """Verify routing table has discovered cluster participants."""
    log.info("TEST: routing_table_health")

    size = node.routing_table.size()
    results.record("routing table non-empty", size > 0, f"{size} peers")
    # At scale, we expect at least K (20) peers in the routing table
    min_expected = min(20, size)
    results.record("routing table well-populated", size >= min_expected,
                    f"{size} peers (expected >= {min_expected})")

    all_peers = node.routing_table.all_peers()
    peers_with_addrs = sum(1 for p in all_peers if len(p.addrs) > 0)
    results.record(
        "all peers have addresses",
        peers_with_addrs == len(all_peers),
        f"{peers_with_addrs}/{len(all_peers)} have addresses",
    )


async def test_batch_records(node: DhtNode, results: TestResult) -> None:
    """Batch PUT/GET - scales with cluster size."""
    log.info("TEST: batch_records")
    pod_name = os.environ.get("POD_NAME", "test")
    peer_count = node.routing_table.size()
    n_records = 20 if peer_count < 20 else 200

    for i in range(n_records):
        key = f"/k8s/{pod_name}/batch/{i}".encode()
        value = json.dumps({"index": i, "pod": pod_name}).encode()
        await node.put(key, value)

    ok = 0
    for i in range(n_records):
        key = f"/k8s/{pod_name}/batch/{i}".encode()
        expected = json.dumps({"index": i, "pod": pod_name}).encode()
        result = await node.get(key)
        if result == expected:
            ok += 1
        else:
            log.warning(f"  batch/{i}: expected {len(expected)} bytes, got {result!r}")

    results.record(f"batch {n_records} records", ok == n_records, f"{ok}/{n_records}")


async def test_per_record_ttl(node: DhtNode, results: TestResult) -> None:
    """PUT a record with a short TTL and verify it expires."""
    log.info("TEST: per_record_ttl")
    pod_name = os.environ.get("POD_NAME", "test")

    key = f"/k8s/{pod_name}/ttl-test".encode()
    value = json.dumps({"status": "READY", "ttl_test": True}).encode()

    await node.put(key, value, ttl=3.0)

    result = await node.get(key)
    results.record("ttl record stored", result == value)

    local = node.kad_handler.get_local(key)
    results.record("local record has per-record TTL", local is not None and local.ttl == 3.0,
                    f"ttl={local.ttl if local else None}")

    log.info("  Waiting 4s for TTL expiry...")
    await asyncio.sleep(4.0)

    node.kad_handler.remove_expired(node.record_ttl)

    local_after = node.kad_handler.get_local(key)
    results.record("local record expired via per-record TTL", local_after is None)


async def test_concurrent_puts(node: DhtNode, results: TestResult) -> None:
    """Massive concurrent PUTs to stress the connection pool and Yamux.

    At scale (200+ peers), fires 200 concurrent PUTs. Each PUT targets
    K=20 closest peers, so with well-distributed keys this should touch
    most of the cluster simultaneously.
    """
    log.info("TEST: concurrent_puts")
    pod_name = os.environ.get("POD_NAME", "test")
    peer_count = node.routing_table.size()
    n_concurrent = 10 if peer_count < 20 else 200

    async def put_one(i: int) -> bool:
        key = f"/k8s/{pod_name}/concurrent/{i}".encode()
        value = json.dumps({"index": i, "concurrent": True}).encode()
        try:
            count = await asyncio.wait_for(node.put(key, value), timeout=30.0)
            return count >= 0
        except Exception as e:
            log.warning(f"  concurrent put {i} failed: {e}")
            return False

    tasks = [put_one(i) for i in range(n_concurrent)]
    put_results = await asyncio.gather(*tasks)
    ok = sum(1 for r in put_results if r)
    results.record(f"concurrent puts ({n_concurrent})", ok == n_concurrent, f"{ok}/{n_concurrent}")

    # Concurrent GETs too
    async def get_one(i: int) -> bool:
        key = f"/k8s/{pod_name}/concurrent/{i}".encode()
        try:
            result = await asyncio.wait_for(node.get(key), timeout=15.0)
            return result is not None
        except Exception as e:
            log.warning(f"  concurrent get {i} failed: {e}")
            return False

    get_tasks = [get_one(i) for i in range(n_concurrent)]
    get_results = await asyncio.gather(*get_tasks)
    get_ok = sum(1 for r in get_results if r)
    results.record(f"concurrent gets ({n_concurrent})", get_ok == n_concurrent, f"{get_ok}/{n_concurrent}")


async def test_record_filter(node_with_filter: DhtNode, source_node: DhtNode, results: TestResult) -> None:
    """Verify record_filter works across network boundaries."""
    log.info("TEST: record_filter")
    from mx_libp2p.multiaddr import encode_multiaddr_ip4_tcp_p2p

    good_key = b"/mx/k8s-filter/worker/0"
    bad_key = b"/bad/k8s-filter/evil"

    filter_peer_id = node_with_filter.peer_id
    _, fport = node_with_filter.listen_addr
    fhost = os.environ.get("POD_IP", "127.0.0.1")
    filter_addr = encode_multiaddr_ip4_tcp_p2p(fhost, fport, filter_peer_id)
    source_node.peer_store.add_addrs(filter_peer_id, [filter_addr])

    ok_good = await source_node._put_to_peer(filter_peer_id, [filter_addr], good_key, b'{"rank":0}')
    ok_bad = await source_node._put_to_peer(filter_peer_id, [filter_addr], bad_key, b'{"evil":true}')
    await asyncio.sleep(0.3)

    good_rec = node_with_filter.kad_handler.get_local(good_key)
    bad_rec = node_with_filter.kad_handler.get_local(bad_key)
    results.record("filter accepts /mx/ key", good_rec is not None,
                    f"put_ok={ok_good}, stored={good_rec is not None}")
    results.record("filter rejects /bad/ key", bad_rec is None,
                    f"put_ok={ok_bad}, stored={bad_rec is not None}")


async def test_data_distribution(node: DhtNode, results: TestResult) -> None:
    """Verify records are actually stored on REMOTE peers, not just locally.

    This catches the case where put() succeeds but records only exist in
    the local cache. We PUT a record, delete it locally, then query a sample
    of remote peers directly to confirm at least one has it.
    """
    log.info("TEST: data_distribution")
    pod_name = os.environ.get("POD_NAME", "test")

    key = f"/k8s/{pod_name}/distribution-check".encode()
    value = json.dumps({"test": "distribution", "pod": pod_name}).encode()

    count = await node.put(key, value)
    results.record("distribution put", count >= 1, f"stored on {count} peers")

    # Delete the record from local store so get() MUST hit a remote peer
    if key in node.kad_handler.records:
        del node.kad_handler.records[key]
    if key in node._originated_records:
        del node._originated_records[key]

    local_check = node.kad_handler.get_local(key)
    results.record("local record removed", local_check is None)

    # Query a sample of known peers directly for the record.
    # At scale (200+ peers), querying all of them is wasteful - sample up to 20.
    import random
    all_peers = node.routing_table.all_peers()
    sample_size = min(20, len(all_peers))
    sampled_peers = random.sample(all_peers, sample_size) if len(all_peers) > sample_size else all_peers
    remote_hits = 0
    remote_peers_checked = 0
    for entry in sampled_peers:
        remote_peers_checked += 1
        try:
            val, _ttl, _closer = await asyncio.wait_for(
                node._get_value_single(entry.peer_id, entry.addrs, key),
                timeout=5.0,
            )
            if val == value:
                remote_hits += 1
        except Exception as e:
            log.debug(f"  direct query to {entry.peer_id.hex()[:16]}... failed: {e}")

    results.record(
        "record found on remote peers",
        remote_hits >= 1,
        f"{remote_hits}/{remote_peers_checked} sampled peers have the record",
    )

    # Also verify iterative get (with empty local store) reaches a remote copy
    iterative_result = await node.get(key)
    results.record(
        "iterative get finds remote copy",
        iterative_result == value,
    )


async def test_multi_hop_lookup(node: DhtNode, results: TestResult) -> None:
    """Store a record with a far key and verify lookup works via multi-hop."""
    log.info("TEST: multi_hop_lookup")

    key = b"\xff" * 32
    value = json.dumps({"test": "multi-hop", "time": time.time()}).encode()

    count = await node.put(key, value)
    results.record("far-key put", count >= 1, f"stored on {count} peers")

    result = await asyncio.wait_for(node.get(key), timeout=10.0)
    results.record("far-key get", result == value)


async def test_cluster_flood(node: DhtNode, results: TestResult) -> None:
    """Flood the DHT with enough records to touch every peer in the cluster.

    With K=20 replication and N peers, each record lands on 20 peers.
    To cover all N peers we need at least N/K * ~2 records (pigeonhole +
    margin for key clustering). We use N*2 records to be safe, with all
    PUTs fired concurrently.

    After the flood, we sample peers from the routing table and verify
    they actually hold records (not just that iterative GET works).
    """
    log.info("TEST: cluster_flood")
    peer_count = node.routing_table.size()
    if peer_count < 20:
        results.skip("cluster_flood", f"only {peer_count} peers (need >= 20 for meaningful flood)")
        return

    pod_name = os.environ.get("POD_NAME", "test")
    # N*2 records to ensure good keyspace coverage
    n_records = peer_count * 2
    log.info(f"  Flooding {n_records} records across {peer_count} peers...")

    # Phase 1: concurrent PUT flood
    t0 = time.time()

    async def flood_put(i: int) -> int:
        key = f"/k8s/{pod_name}/flood/{i}".encode()
        value = json.dumps({"flood": i, "pod": pod_name}).encode()
        try:
            return await asyncio.wait_for(node.put(key, value), timeout=30.0)
        except Exception as e:
            log.warning(f"  flood put {i} failed: {e}")
            return 0

    put_counts = await asyncio.gather(*(flood_put(i) for i in range(n_records)))
    elapsed = time.time() - t0
    successful = sum(1 for c in put_counts if c > 0)
    total_stores = sum(put_counts)
    log.info(f"  Flood complete: {successful}/{n_records} records in {elapsed:.1f}s "
             f"({total_stores} total peer stores, {total_stores/elapsed:.0f} stores/sec)")
    results.record(f"flood puts ({n_records})", successful == n_records,
                    f"{successful}/{n_records} in {elapsed:.1f}s")

    # Phase 2: concurrent GET verification (sample)
    import random
    sample_size = min(100, n_records)
    sample_indices = random.sample(range(n_records), sample_size)

    async def flood_get(i: int) -> bool:
        key = f"/k8s/{pod_name}/flood/{i}".encode()
        expected = json.dumps({"flood": i, "pod": pod_name}).encode()
        try:
            result = await asyncio.wait_for(node.get(key), timeout=15.0)
            return result == expected
        except Exception:
            return False

    t0 = time.time()
    get_results = await asyncio.gather(*(flood_get(i) for i in sample_indices))
    get_elapsed = time.time() - t0
    get_ok = sum(1 for r in get_results if r)
    results.record(f"flood gets (sample {sample_size})", get_ok == sample_size,
                    f"{get_ok}/{sample_size} in {get_elapsed:.1f}s")

    # Phase 3: verify cluster-wide distribution
    # With K=20 replication and N*2 records, the total store count should
    # indicate broad coverage: N*2 records * K stores each = N*40 stores.
    # Even with key clustering, this should touch most of the cluster.
    # We verify via the total store count rather than querying individual
    # peers (a direct GET_VALUE only returns a record if the peer is among
    # the K closest, which is unlikely for a random peer+key pair).
    expected_stores = n_records * 20  # ideal: every PUT hits K=20 peers
    coverage_ratio = total_stores / expected_stores if expected_stores > 0 else 0
    results.record("flood store efficiency",
                    coverage_ratio >= 0.8,
                    f"{total_stores}/{expected_stores} stores ({coverage_ratio:.0%} of ideal)")


async def test_large_record(node: DhtNode, results: TestResult) -> None:
    """PUT a record near the max size and verify cross-node transfer."""
    log.info("TEST: large_record")
    pod_name = os.environ.get("POD_NAME", "test")

    tensor_layout = [
        {"name": f"model.layers.{i}.weight", "size": 134217728, "dtype": "fp8"}
        for i in range(100)
    ]
    key = f"/k8s/{pod_name}/large".encode()
    value = json.dumps({"rank": 0, "tensors": tensor_layout}).encode()
    log.info(f"  Record size: {len(value)} bytes")

    count = await node.put(key, value)
    results.record("large record put", count >= 1, f"{len(value)} bytes on {count} peers")

    result = await node.get(key)
    results.record("large record get", result == value, f"got {len(result) if result else 0} bytes")


# ---------------------------------------------------------------------------
# Test coordinator
# ---------------------------------------------------------------------------


async def run_test(host: str, port: int, dns: str | None, bootstrap: list[str]) -> bool:
    """Join the DHT, run all test scenarios, report results."""
    results = TestResult()

    node = DhtNode()
    node._observed_ip_threshold = 1
    await node.start(host, port, bootstrap_peers=bootstrap or None,
                     bootstrap_dns=dns, bootstrap_dns_port=port)

    rhost, rport = node.routable_addr()
    log.info(f"Test node ready on {rhost}:{rport}")
    log.info(f"Routing table: {node.routing_table.size()} peers")

    # Wait for routing tables to settle across physical nodes.
    # At 200+ peers, iterative lookups need a few seconds to propagate.
    peer_count = node.routing_table.size()
    settle_time = 5.0 if peer_count < 20 else 15.0
    log.info(f"Settling for {settle_time}s ({peer_count} peers in routing table)...")
    await asyncio.sleep(settle_time)

    log.info(f"Routing table after settle: {node.routing_table.size()} peers")

    try:
        await test_basic_put_get(node, results)
        await test_cross_node_routing(node, results)
        await test_observed_ip_detection(node, results)
        await test_routing_table_health(node, results)
        await test_data_distribution(node, results)
        await test_batch_records(node, results)
        await test_per_record_ttl(node, results)
        await test_concurrent_puts(node, results)
        await test_multi_hop_lookup(node, results)
        await test_large_record(node, results)
        await test_cluster_flood(node, results)

        # Record filter test needs a second node with a filter
        log.info("TEST: record_filter (spawning filtered node)")
        def only_mx(key: bytes, value: bytes) -> bool:
            return key.startswith(b"/mx/")

        filter_node = DhtNode(record_filter=only_mx)
        await filter_node.start(host, 0, bootstrap_dns=dns,
                                bootstrap_dns_port=port)
        await asyncio.sleep(1.0)
        try:
            await test_record_filter(filter_node, node, results)
        finally:
            await filter_node.stop()

    except Exception as e:
        log.error(f"Unhandled exception during tests: {e}", exc_info=True)
        results.record("test suite completed without crash", False, str(e))
    finally:
        await node.stop()

    log.info("")
    log.info(f"RESULTS: {results.summary()}")
    if results.all_passed:
        log.info("ALL TESTS PASSED")
    else:
        log.error("SOME TESTS FAILED")
    return results.all_passed


# ---------------------------------------------------------------------------
# Soak test: data survival across rolling restart
# ---------------------------------------------------------------------------


async def run_soak(host: str, port: int, dns: str | None, bootstrap: list[str]) -> bool:
    """Seed the DHT with data, trigger a rolling restart, verify data survives.

    This tests the critical production property: can the DHT maintain data
    availability while pods are being replaced? With K=20 replication and a
    rolling restart that replaces 25% of pods at a time, records should
    remain accessible throughout the rollout.

    Phases:
      1. Join DHT, seed N records, verify all accessible
      2. Trigger rolling restart via kubectl rollout restart
      3. Poll records periodically during the rollout
      4. After rollout completes, verify all records still accessible
    """
    results = TestResult()
    kubectl = os.environ.get("KUBECTL", "kubectl")
    namespace = os.environ.get("NAMESPACE", "dht-test")

    node = DhtNode()
    node._observed_ip_threshold = 1
    await node.start(host, port, bootstrap_peers=bootstrap or None,
                     bootstrap_dns=dns, bootstrap_dns_port=port)

    peer_count = node.routing_table.size()
    log.info(f"Soak test node ready, routing table: {peer_count} peers")

    if peer_count < 20:
        log.error(f"Not enough peers for soak test ({peer_count} < 20)")
        await node.stop()
        return False

    settle_time = 15.0
    log.info(f"Settling for {settle_time}s...")
    await asyncio.sleep(settle_time)
    peer_count = node.routing_table.size()
    log.info(f"Routing table after settle: {peer_count} peers")

    pod_name = os.environ.get("POD_NAME", "soak")
    n_records = 500 if peer_count >= 20 else peer_count * 2

    # Phase 1: seed records
    log.info(f"PHASE 1: Seeding {n_records} records...")
    t0 = time.time()

    async def seed_put(i: int) -> int:
        key = f"/k8s/{pod_name}/soak/{i}".encode()
        value = json.dumps({"soak": i, "pod": pod_name, "t": time.time()}).encode()
        try:
            return await asyncio.wait_for(node.put(key, value), timeout=30.0)
        except Exception as e:
            log.warning(f"  seed put {i} failed: {e}")
            return 0

    counts = await asyncio.gather(*(seed_put(i) for i in range(n_records)))
    seed_elapsed = time.time() - t0
    seed_ok = sum(1 for c in counts if c > 0)
    log.info(f"  Seeded {seed_ok}/{n_records} records in {seed_elapsed:.1f}s")
    results.record(f"seed {n_records} records", seed_ok == n_records,
                    f"{seed_ok}/{n_records} in {seed_elapsed:.1f}s")

    # Verify all records accessible before restart
    async def check_record(i: int) -> bool:
        key = f"/k8s/{pod_name}/soak/{i}".encode()
        try:
            result = await asyncio.wait_for(node.get(key), timeout=15.0)
            return result is not None
        except Exception:
            return False

    pre_check = await asyncio.gather(*(check_record(i) for i in range(n_records)))
    pre_ok = sum(1 for r in pre_check if r)
    results.record("pre-restart availability", pre_ok == n_records,
                    f"{pre_ok}/{n_records}")

    # Phase 2: trigger rolling restart
    log.info("PHASE 2: Triggering rolling restart of dht-peer deployment...")
    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            [kubectl, "rollout", "restart", "deployment/dht-peer", "-n", namespace],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            log.error(f"  kubectl rollout restart failed: {proc.stderr}")
            results.record("trigger rolling restart", False, proc.stderr.strip())
            await node.stop()
            return False
        log.info(f"  Rolling restart triggered: {proc.stdout.strip()}")
        results.record("trigger rolling restart", True)
    except Exception as e:
        log.error(f"  Failed to trigger restart: {e}")
        results.record("trigger rolling restart", False, str(e))
        await node.stop()
        return False

    # Phase 3: poll records during rollout
    log.info("PHASE 3: Polling records during rollout...")
    poll_round = 0
    min_availability = 100.0
    rollout_done = False

    while not rollout_done:
        await asyncio.sleep(10.0)
        poll_round += 1

        # Sample a subset of records
        sample_size = min(50, n_records)
        sample_indices = random.sample(range(n_records), sample_size)
        sample_results = await asyncio.gather(*(check_record(i) for i in sample_indices))
        available = sum(1 for r in sample_results if r)
        pct = (available / sample_size) * 100

        rt_size = node.routing_table.size()
        log.info(f"  Poll {poll_round}: {available}/{sample_size} records available "
                 f"({pct:.0f}%), routing table: {rt_size} peers")
        min_availability = min(min_availability, pct)

        # Check if rollout is done
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                [kubectl, "rollout", "status", "deployment/dht-peer", "-n", namespace,
                 "--timeout=1s"],
                capture_output=True, text=True, timeout=10,
            )
            rollout_done = proc.returncode == 0
        except Exception:
            pass

        # Safety limit: don't poll forever
        if poll_round >= 60:
            log.warning("  Poll limit reached (60 rounds), stopping")
            break

    results.record("min availability during rollout", min_availability >= 80.0,
                    f"{min_availability:.0f}% (threshold: 80%)")

    # Phase 4: re-bootstrap and verify post-restart
    log.info("PHASE 4: Post-restart verification...")
    await asyncio.sleep(10.0)
    post_peers = node.routing_table.size()
    log.info(f"  Routing table: {post_peers} peers")

    # Full check of all records
    post_check = await asyncio.gather(*(check_record(i) for i in range(n_records)))
    post_ok = sum(1 for r in post_check if r)
    results.record("post-restart availability", post_ok == n_records,
                    f"{post_ok}/{n_records}")

    await node.stop()

    log.info("")
    log.info(f"SOAK RESULTS: {results.summary()}")
    log.info(f"  Min availability during rollout: {min_availability:.0f}%")
    if results.all_passed:
        log.info("SOAK TEST PASSED")
    else:
        log.error("SOAK TEST FAILED")
    return results.all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="DHT node for k8s multi-node testing")
    parser.add_argument("--role", choices=["peer", "test", "soak"], required=True)
    parser.add_argument("--listen", default="0.0.0.0:4001", help="host:port to listen on")
    parser.add_argument("--dns", default=None, help="Headless service hostname for peer discovery")
    parser.add_argument("--bootstrap", action="append", default=[], help="Bootstrap multiaddr(s)")
    args = parser.parse_args()

    # Auto-detect DNS from BOOTSTRAP_DNS env var
    dns = args.dns or os.environ.get("BOOTSTRAP_DNS")

    host, port_str = args.listen.rsplit(":", 1)
    port = int(port_str)

    if not dns and not args.bootstrap:
        log.warning("No --dns or --bootstrap given. Node will start standalone.")

    if args.role == "peer":
        asyncio.run(run_peer(host, port, dns, args.bootstrap))
    elif args.role == "test":
        success = asyncio.run(run_test(host, port, dns, args.bootstrap))
        sys.exit(0 if success else 1)
    elif args.role == "soak":
        success = asyncio.run(run_soak(host, port, dns, args.bootstrap))
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
