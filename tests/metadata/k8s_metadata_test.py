# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K8s integration test for the DHT metadata backend.

Tests DhtMetadataClient end-to-end in a real K8s cluster:
  - Publish/get worker metadata through the DHT
  - Cross-pod discovery (pod A publishes, pod B reads)
  - Status updates
  - Source listing
  - Incremental publishing (add workers one at a time)
  - Directory merge correctness

Roles:
  peer      - DHT backbone node, runs indefinitely
  publisher - Publishes known metadata and stays alive for readers
  test      - Runs all test scenarios and exits 0/1
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time

from mx_libp2p.dht import DhtNode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("k8s-metadata")


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
# Async helpers using the new source_id-based key schema
# ---------------------------------------------------------------------------


from modelexpress.dht_client import (
    _compute_mx_source_id,
    _worker_to_json,
    _json_to_worker_metadata,
    _worker_key,
    _worker_directory_key,
    _instances_key,
    _attrs_key,
    _sources_key,
    _FakeGetMetadataResponse,
    _FakeListSourcesResponse,
    _FakeSourceInstanceRef,
)


async def _async_publish(node, identity, worker, worker_id, ttl=300):
    """Publish one worker's metadata using the new source_id-based schema."""
    from modelexpress import p2p_pb2

    source_id = _compute_mx_source_id(identity)
    now = int(time.time() * 1000)
    rank = worker.worker_rank

    try:
        # Store worker record
        record = _worker_to_json(worker)
        await node.put(
            _worker_key(source_id, worker_id, rank),
            json.dumps(record).encode(),
            ttl=ttl,
        )

        # Merge rank into worker_id's directory
        raw = await node.get(_worker_directory_key(source_id, worker_id))
        existing_ranks = set(json.loads(raw).get("ranks", [])) if raw else set()
        existing_ranks.add(rank)
        directory = {"ranks": sorted(existing_ranks), "updated_at": now}
        await node.put(
            _worker_directory_key(source_id, worker_id),
            json.dumps(directory).encode(),
            ttl=ttl,
        )

        # Merge worker_id into instances directory
        raw = await node.get(_instances_key(source_id))
        worker_ids = set(json.loads(raw).get("worker_ids", [])) if raw else set()
        worker_ids.add(worker_id)
        instances = {"worker_ids": sorted(worker_ids), "updated_at": now}
        await node.put(
            _instances_key(source_id),
            json.dumps(instances).encode(),
            ttl=ttl,
        )

        # Store source attributes
        attrs = {"model_name": identity.model_name, "updated_at": now}
        await node.put(
            _attrs_key(source_id),
            json.dumps(attrs).encode(),
            ttl=ttl,
        )

        # Add to global sources list
        raw = await node.get(_sources_key())
        source_ids = set(json.loads(raw).get("source_ids", [])) if raw else set()
        source_ids.add(source_id)
        sources_list = {"source_ids": sorted(source_ids), "updated_at": now}
        await node.put(
            _sources_key(),
            json.dumps(sources_list).encode(),
            ttl=ttl,
        )

        return source_id
    except Exception as e:
        log.error(f"Publish failed: {e}")
        return None


async def _async_get(node, source_id, worker_id):
    """Get metadata for a specific source_id + worker_id."""
    try:
        raw = await node.get(_worker_directory_key(source_id, worker_id))
        if raw is None:
            return _FakeGetMetadataResponse(found=False)

        directory = json.loads(raw)
        ranks = directory.get("ranks", [])
        workers = []
        for rank in ranks:
            raw = await node.get(_worker_key(source_id, worker_id, rank))
            if raw:
                workers.append(_json_to_worker_metadata(json.loads(raw)))

        if not workers:
            return _FakeGetMetadataResponse(found=False)

        return _FakeGetMetadataResponse(
            found=True, worker=workers[0],
            mx_source_id=source_id, worker_id=worker_id,
        )
    except Exception as e:
        log.error(f"Get failed: {e}")
        return _FakeGetMetadataResponse(found=False)


async def _async_update_status(node, source_id, worker_id, worker_rank, status, ttl=300):
    """Update worker status in the DHT."""
    try:
        key = _worker_key(source_id, worker_id, worker_rank)
        raw = await node.get(key)
        if raw is None:
            return False
        record = json.loads(raw)
        record["status"] = int(status)
        record["updated_at"] = int(time.time() * 1000)
        await node.put(key, json.dumps(record).encode(), ttl=ttl)
        return True
    except Exception as e:
        log.error(f"Update status failed: {e}")
        return False


async def _async_list_sources(node):
    """List all source_ids and their model names."""
    try:
        raw = await node.get(_sources_key())
        if raw is None:
            return []
        data = json.loads(raw)
        results = []
        for sid in data.get("source_ids", []):
            attrs_raw = await node.get(_attrs_key(sid))
            model_name = ""
            if attrs_raw:
                model_name = json.loads(attrs_raw).get("model_name", "")
            results.append((sid, model_name))
        return results
    except Exception as e:
        log.error(f"List sources failed: {e}")
        return []


async def _async_list_instances(node, source_id):
    """List all worker_ids for a source."""
    try:
        raw = await node.get(_instances_key(source_id))
        if raw is None:
            return []
        return json.loads(raw).get("worker_ids", [])
    except Exception as e:
        log.error(f"List instances failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


async def run_peer(host: str, port: int, dns: str | None) -> None:
    """Run a DHT peer node that joins via DNS and stays alive."""
    node = DhtNode()
    await node.start(host, port, bootstrap_dns=dns, bootstrap_dns_port=port)
    log.info(f"Peer node ready, routing table: {node.routing_table.size()} peers")

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    await node.stop()


PUBLISHER_MODEL = "k8s-test-model"
PUBLISHER_WORKER_ID = "publisher-instance-1"


async def run_publisher(host: str, port: int, dns: str | None) -> None:
    """Publish known metadata and stay alive for readers to discover."""
    from modelexpress import p2p_pb2

    node = DhtNode()
    node._observed_ip_threshold = 1
    await node.start(host, port, bootstrap_dns=dns, bootstrap_dns_port=port)
    log.info(f"Publisher ready, routing table: {node.routing_table.size()} peers")

    await asyncio.sleep(5.0)  # Settle

    identity = p2p_pb2.SourceIdentity(model_name=PUBLISHER_MODEL)
    source_id = _compute_mx_source_id(identity)

    # Publish 2 workers for the well-known model
    workers = [
        p2p_pb2.WorkerMetadata(
            worker_rank=0,
            metadata_endpoint="10.0.0.1:5555",
            agent_name="publisher-agent-0",
            worker_grpc_endpoint="10.0.0.1:5556",
            status=p2p_pb2.SOURCE_STATUS_READY,
            updated_at=int(time.time() * 1000),
        ),
        p2p_pb2.WorkerMetadata(
            worker_rank=1,
            metadata_endpoint="10.0.0.2:5555",
            agent_name="publisher-agent-1",
            worker_grpc_endpoint="10.0.0.2:5556",
            status=p2p_pb2.SOURCE_STATUS_READY,
            updated_at=int(time.time() * 1000),
        ),
    ]

    # Each worker gets published separately (matching the MxClient interface)
    for w in workers:
        sid = await _async_publish(node, identity, w, PUBLISHER_WORKER_ID)
        if sid:
            log.info(f"Published worker rank {w.worker_rank} (source_id={sid})")
        else:
            log.error(f"Failed to publish worker rank {w.worker_rank}")

    # Signal readiness via a well-known key
    await node.put(b"/mx/_test_signal/publisher-ready",
                   json.dumps({
                       "ready": True,
                       "model": PUBLISHER_MODEL,
                       "source_id": source_id,
                       "worker_id": PUBLISHER_WORKER_ID,
                   }).encode(),
                   ttl=600)
    log.info("Publisher signal key set, staying alive...")

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    await node.stop()


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------


async def test_publish_and_get(node: DhtNode, results: TestResult) -> None:
    """Publish metadata for a model and read it back."""
    log.info("TEST: publish_and_get")
    from modelexpress import p2p_pb2

    identity = p2p_pb2.SourceIdentity(model_name="test-publish-get")
    worker_id = "test-wid-1"

    w0 = p2p_pb2.WorkerMetadata(
        worker_rank=0,
        metadata_endpoint="10.1.0.1:5555",
        agent_name="test-agent-0",
        worker_grpc_endpoint="10.1.0.1:5556",
        status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
    )
    w1 = p2p_pb2.WorkerMetadata(
        worker_rank=1,
        metadata_endpoint="10.1.0.2:5555",
        agent_name="test-agent-1",
        worker_grpc_endpoint="10.1.0.2:5556",
        status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
    )

    sid0 = await _async_publish(node, identity, w0, worker_id)
    sid1 = await _async_publish(node, identity, w1, worker_id)
    results.record("publish metadata", sid0 is not None)
    results.record("consistent source_id", sid0 == sid1)

    await asyncio.sleep(1.0)

    # Get rank 0
    resp = await _async_get(node, sid0, worker_id)
    results.record("get finds worker", resp.found)

    if resp.found and resp.worker:
        results.record("worker rank", resp.worker.worker_rank == 0)
        results.record("worker endpoint", resp.worker.metadata_endpoint == "10.1.0.1:5555")
        results.record("worker agent_name", resp.worker.agent_name == "test-agent-0")
        results.record("worker grpc_endpoint", resp.worker.worker_grpc_endpoint == "10.1.0.1:5556")

    # Check directory has both ranks
    raw = await node.get(_worker_directory_key(sid0, worker_id))
    if raw:
        directory = json.loads(raw)
        results.record("directory has both ranks",
                       sorted(directory.get("ranks", [])) == [0, 1],
                       str(directory.get("ranks")))


async def test_update_status(node: DhtNode, results: TestResult) -> None:
    """Publish, update status, verify the status change persists."""
    log.info("TEST: update_status")
    from modelexpress import p2p_pb2

    identity = p2p_pb2.SourceIdentity(model_name="test-update-status")
    worker_id = "status-wid-1"

    w = p2p_pb2.WorkerMetadata(
        worker_rank=0,
        metadata_endpoint="10.2.0.1:5555",
        agent_name="status-agent-0",
        status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
    )

    source_id = await _async_publish(node, identity, w, worker_id)
    await asyncio.sleep(0.5)

    ok = await _async_update_status(
        node, source_id, worker_id, 0, p2p_pb2.SOURCE_STATUS_READY)
    results.record("update status call", ok)

    await asyncio.sleep(0.5)

    resp = await _async_get(node, source_id, worker_id)
    if resp.found and resp.worker:
        results.record("status updated",
                       resp.worker.status == p2p_pb2.SOURCE_STATUS_READY,
                       f"got status={resp.worker.status}")
    else:
        results.record("status updated", False, "worker not found after update")


async def test_list_sources(node: DhtNode, results: TestResult) -> None:
    """Publish multiple sources and verify they appear in the sources list."""
    log.info("TEST: list_sources")
    from modelexpress import p2p_pb2

    for model_name in ["list-model-a", "list-model-b"]:
        identity = p2p_pb2.SourceIdentity(model_name=model_name)
        w = p2p_pb2.WorkerMetadata(worker_rank=0, metadata_endpoint="x:0",
                                    agent_name="a")
        await _async_publish(node, identity, w, "list-wid")

    await asyncio.sleep(0.5)

    sources = await _async_list_sources(node)
    model_names = [m for _, m in sources]
    results.record("list-model-a in list", "list-model-a" in model_names, str(model_names))
    results.record("list-model-b in list", "list-model-b" in model_names, str(model_names))


async def test_multiple_worker_ids(node: DhtNode, results: TestResult) -> None:
    """Publish from two worker_ids and verify both appear in instances."""
    log.info("TEST: multiple_worker_ids")
    from modelexpress import p2p_pb2

    identity = p2p_pb2.SourceIdentity(model_name="test-multi-worker")

    w0 = p2p_pb2.WorkerMetadata(
        worker_rank=0, metadata_endpoint="10.4.0.1:5555", agent_name="agent-a")
    w1 = p2p_pb2.WorkerMetadata(
        worker_rank=0, metadata_endpoint="10.4.0.2:5555", agent_name="agent-b")

    source_id = await _async_publish(node, identity, w0, "wid-a")
    await _async_publish(node, identity, w1, "wid-b")
    await asyncio.sleep(0.5)

    instances = await _async_list_instances(node, source_id)
    results.record("two worker_ids", len(instances) == 2, str(instances))
    results.record("wid-a in instances", "wid-a" in instances)
    results.record("wid-b in instances", "wid-b" in instances)

    # Each worker_id should have its own rank 0
    resp_a = await _async_get(node, source_id, "wid-a")
    resp_b = await _async_get(node, source_id, "wid-b")
    results.record("wid-a found", resp_a.found)
    results.record("wid-b found", resp_b.found)
    if resp_a.found and resp_a.worker:
        results.record("wid-a endpoint", resp_a.worker.metadata_endpoint == "10.4.0.1:5555")
    if resp_b.found and resp_b.worker:
        results.record("wid-b endpoint", resp_b.worker.metadata_endpoint == "10.4.0.2:5555")


async def test_cross_pod_discovery(node: DhtNode, results: TestResult) -> None:
    """Read metadata published by the publisher pod (cross-pod, different DHT node)."""
    log.info("TEST: cross_pod_discovery")

    # Wait for publisher signal
    log.info("  Waiting for publisher signal...")
    signal_data = None
    for attempt in range(30):
        raw = await node.get(b"/mx/_test_signal/publisher-ready")
        if raw:
            signal_data = json.loads(raw)
            break
        await asyncio.sleep(1.0)

    if not signal_data:
        results.record("publisher signal received", False, "timed out after 30s")
        return

    results.record("publisher signal received", True)
    source_id = signal_data["source_id"]
    worker_id = signal_data["worker_id"]
    model_name = signal_data["model"]

    resp = await _async_get(node, source_id, worker_id)
    results.record("cross-pod: worker found", resp.found)

    if resp.found and resp.worker:
        # The directory should have ranks [0, 1] for this worker_id
        raw = await node.get(_worker_directory_key(source_id, worker_id))
        if raw:
            directory = json.loads(raw)
            ranks = sorted(directory.get("ranks", []))
            results.record("cross-pod: 2 ranks", ranks == [0, 1], str(ranks))

        # Read both ranks individually
        for rank in [0, 1]:
            raw = await node.get(_worker_key(source_id, worker_id, rank))
            if raw:
                w = _json_to_worker_metadata(json.loads(raw))
                results.record(f"cross-pod: rank {rank} readable", True)
                if rank == 0:
                    results.record("cross-pod: w0 endpoint",
                                  w.metadata_endpoint == "10.0.0.1:5555",
                                  w.metadata_endpoint)
                    results.record("cross-pod: w0 agent_name",
                                  w.agent_name == "publisher-agent-0",
                                  w.agent_name)
                    results.record("cross-pod: w0 tensors", len(w.tensors) == 2,
                                  f"got {len(w.tensors)}")
                    results.record("cross-pod: w0 status ready",
                                  w.status == 2)  # SOURCE_STATUS_READY
                elif rank == 1:
                    results.record("cross-pod: w1 endpoint",
                                  w.metadata_endpoint == "10.0.0.2:5555",
                                  w.metadata_endpoint)
                    results.record("cross-pod: w1 tensors", len(w.tensors) == 1,
                                  f"got {len(w.tensors)}")
            else:
                results.record(f"cross-pod: rank {rank} readable", False, "got None")

    # Verify source appears in global list
    sources = await _async_list_sources(node)
    source_ids = [sid for sid, _ in sources]
    results.record("cross-pod: source in list", source_id in source_ids, str(sources))


async def test_json_schema_correctness(node: DhtNode, results: TestResult) -> None:
    """Verify the raw JSON stored in DHT matches expected schema."""
    log.info("TEST: json_schema_correctness")
    from modelexpress import p2p_pb2

    identity = p2p_pb2.SourceIdentity(model_name=PUBLISHER_MODEL)
    source_id = _compute_mx_source_id(identity)

    # Read the raw bytes from DHT and verify JSON structure
    raw = await node.get(_worker_key(source_id, PUBLISHER_WORKER_ID, 0))
    if raw is None:
        results.record("raw worker record readable", False, "got None")
        return

    results.record("raw worker record readable", True)
    record = json.loads(raw)

    # Verify expected fields exist
    expected_fields = ["worker_rank", "metadata_endpoint", "agent_name",
                       "tensors", "status", "updated_at"]
    for field in expected_fields:
        results.record(f"schema: '{field}' present", field in record,
                      f"keys={list(record.keys())}")

    # Verify tensor sub-structure
    if record.get("tensors"):
        t = record["tensors"][0]
        tensor_fields = ["name", "addr", "size", "device_id", "dtype"]
        for field in tensor_fields:
            results.record(f"schema: tensor '{field}' present", field in t)

    # Verify worker directory
    raw_dir = await node.get(_worker_directory_key(source_id, PUBLISHER_WORKER_ID))
    if raw_dir:
        dir_record = json.loads(raw_dir)
        results.record("schema: directory has 'ranks'", "ranks" in dir_record)
        results.record("schema: directory ranks correct",
                      sorted(dir_record.get("ranks", [])) == [0, 1],
                      str(dir_record.get("ranks")))

    # Verify instances directory
    raw_inst = await node.get(_instances_key(source_id))
    if raw_inst:
        inst_record = json.loads(raw_inst)
        results.record("schema: instances has 'worker_ids'", "worker_ids" in inst_record)
        results.record("schema: publisher worker_id present",
                      PUBLISHER_WORKER_ID in inst_record.get("worker_ids", []))

    # Verify sources list
    raw_sources = await node.get(_sources_key())
    if raw_sources:
        sources_record = json.loads(raw_sources)
        results.record("schema: sources has 'source_ids'", "source_ids" in sources_record)


# ---------------------------------------------------------------------------
# Test coordinator
# ---------------------------------------------------------------------------


async def run_test(host: str, port: int, dns: str | None) -> bool:
    """Join the DHT, run all metadata test scenarios, report results."""
    results = TestResult()

    node = DhtNode()
    node._observed_ip_threshold = 1
    await node.start(host, port, bootstrap_dns=dns, bootstrap_dns_port=port)

    log.info(f"Test node ready, routing table: {node.routing_table.size()} peers")

    # Settle to let routing tables propagate
    await asyncio.sleep(5.0)
    log.info(f"Routing table after settle: {node.routing_table.size()} peers")

    try:
        # Self-contained tests (publish + get within same node)
        await test_publish_and_get(node, results)
        await test_update_status(node, results)
        await test_list_sources(node, results)
        await test_multiple_worker_ids(node, results)

        # Cross-pod tests (read metadata published by the publisher pod)
        await test_cross_pod_discovery(node, results)

        # Schema verification (checks raw JSON in DHT)
        await test_json_schema_correctness(node, results)

    except Exception as e:
        log.error(f"Unhandled exception: {e}", exc_info=True)
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
    parser = argparse.ArgumentParser(description="DHT metadata backend K8s test")
    parser.add_argument("--role", choices=["peer", "publisher", "test"], required=True)
    parser.add_argument("--listen", default="0.0.0.0:4001", help="host:port to listen on")
    args = parser.parse_args()

    dns = os.environ.get("BOOTSTRAP_DNS")
    host, port_str = args.listen.rsplit(":", 1)
    port = int(port_str)

    if args.role == "peer":
        asyncio.run(run_peer(host, port, dns))
    elif args.role == "publisher":
        asyncio.run(run_publisher(host, port, dns))
    elif args.role == "test":
        success = asyncio.run(run_test(host, port, dns))
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
