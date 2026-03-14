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
import signal
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
    """Verify routing table has discovered all cluster participants."""
    log.info("TEST: routing_table_health")

    size = node.routing_table.size()
    results.record("routing table non-empty", size > 0, f"{size} peers")
    results.record("knows multiple peers", size >= 2, f"{size} peers (expected >= 2)")

    all_peers = node.routing_table.all_peers()
    peers_with_addrs = sum(1 for p in all_peers if len(p.addrs) > 0)
    results.record(
        "all peers have addresses",
        peers_with_addrs == len(all_peers),
        f"{peers_with_addrs}/{len(all_peers)} have addresses",
    )


async def test_batch_records(node: DhtNode, results: TestResult) -> None:
    """Batch PUT/GET of 20 records."""
    log.info("TEST: batch_records")
    pod_name = os.environ.get("POD_NAME", "test")
    n_records = 20

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
    """Multiple concurrent PUTs to stress the connection pool and Yamux."""
    log.info("TEST: concurrent_puts")
    pod_name = os.environ.get("POD_NAME", "test")
    n_concurrent = 10

    async def put_one(i: int) -> bool:
        key = f"/k8s/{pod_name}/concurrent/{i}".encode()
        value = json.dumps({"index": i, "concurrent": True}).encode()
        try:
            count = await asyncio.wait_for(node.put(key, value), timeout=15.0)
            return count >= 0
        except Exception as e:
            log.warning(f"  concurrent put {i} failed: {e}")
            return False

    tasks = [put_one(i) for i in range(n_concurrent)]
    put_results = await asyncio.gather(*tasks)
    ok = sum(1 for r in put_results if r)
    results.record(f"concurrent puts ({n_concurrent})", ok == n_concurrent, f"{ok}/{n_concurrent}")

    get_ok = 0
    for i in range(n_concurrent):
        key = f"/k8s/{pod_name}/concurrent/{i}".encode()
        result = await node.get(key)
        if result is not None:
            get_ok += 1
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
    the local cache. We PUT a record, delete it locally, then query each
    remote peer directly to confirm at least one has it.
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

    # Query each known peer directly for the record
    all_peers = node.routing_table.all_peers()
    remote_hits = 0
    remote_peers_checked = 0
    for entry in all_peers:
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
        f"{remote_hits}/{remote_peers_checked} peers have the record",
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

    # Wait for routing tables to settle across physical nodes
    await asyncio.sleep(3.0)

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
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="DHT node for k8s multi-node testing")
    parser.add_argument("--role", choices=["peer", "test"], required=True)
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


if __name__ == "__main__":
    main()
