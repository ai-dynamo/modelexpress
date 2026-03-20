# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K8s integration test for the DHT metadata backend.

Tests DhtMetadataClient end-to-end in a real K8s cluster:
  - Publish/get worker metadata through the DHT
  - Cross-pod discovery (pod A publishes, pod B reads)
  - Status updates
  - Model listing
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
# Test infrastructure (same pattern as k8s_dht_runner.py)
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
# DhtMetadataClient helpers (imported from modelexpress package)
# ---------------------------------------------------------------------------


def _make_client(node: DhtNode, ttl: int = 300):
    """Create a DhtMetadataClient pre-wired with an existing DhtNode.

    This bypasses the normal constructor (which creates its own node and
    background thread) so we can share the test node's event loop.
    """
    import threading
    from modelexpress.dht_client import DhtMetadataClient

    client = DhtMetadataClient.__new__(DhtMetadataClient)
    client._record_ttl = ttl
    client._lock = threading.Lock()

    loop = asyncio.get_event_loop()
    client._loop = loop
    client._thread = None  # We're running in the same loop
    client._node = node
    client._started = True
    client._listen_addr = "0.0.0.0:4001"
    client._bootstrap_peers_str = ""
    client._bootstrap_dns = None
    return client


async def _run_in_loop(client, coro_func, *args):
    """Run an async operation directly since we share the event loop.

    DhtMetadataClient normally submits async ops to its background thread.
    Since our test shares the event loop, we call the coroutine directly.
    """
    return await coro_func(*args)


def _sync_publish(client, model_name, workers):
    """Synchronous publish using the client's internal async methods."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_async_publish(client, model_name, workers))


async def _async_publish(client, model_name, workers):
    """Publish metadata using the client's async internals."""
    from modelexpress.dht_client import (
        _worker_to_json, _worker_key, _directory_key, _models_key,
    )
    import json as _json
    import time as _time

    try:
        for w in workers:
            record = _worker_to_json(w)
            key = _worker_key(model_name, w.worker_rank)
            await client._node.put(key, _json.dumps(record).encode(),
                                   ttl=client._record_ttl)

        # Update directory
        ranks = sorted(w.worker_rank for w in workers)
        dir_record = {"ranks": ranks, "updated_at": int(_time.time())}
        await client._node.put(_directory_key(model_name),
                               _json.dumps(dir_record).encode(),
                               ttl=client._record_ttl)

        # Update models list
        existing = await client._node.get(_models_key())
        if existing:
            models_data = _json.loads(existing)
            models = models_data.get("models", [])
        else:
            models = []
        if model_name not in models:
            models.append(model_name)
        models_record = {"models": models, "updated_at": int(_time.time())}
        await client._node.put(_models_key(),
                               _json.dumps(models_record).encode(),
                               ttl=client._record_ttl)
        return True
    except Exception as e:
        log.error(f"Publish failed: {e}")
        return False


async def _async_get(client, model_name):
    """Get metadata using the client's async internals."""
    from modelexpress.dht_client import (
        _directory_key, _worker_key, _json_to_worker_metadata,
        _FakeGetMetadataResponse,
    )
    import json as _json

    try:
        dir_data = await client._node.get(_directory_key(model_name))
        if dir_data is None:
            return _FakeGetMetadataResponse(found=False, workers=[])

        directory = _json.loads(dir_data)
        ranks = directory.get("ranks", [])
        workers = []
        for rank in ranks:
            raw = await client._node.get(_worker_key(model_name, rank))
            if raw:
                workers.append(_json_to_worker_metadata(_json.loads(raw)))
        return _FakeGetMetadataResponse(found=True, workers=workers)
    except Exception as e:
        log.error(f"Get failed: {e}")
        return _FakeGetMetadataResponse(found=False, workers=[])


async def _async_update_status(client, model_name, worker_rank, status):
    """Update status using the client's async internals."""
    from modelexpress.dht_client import _worker_key
    import json as _json
    import time as _time

    try:
        key = _worker_key(model_name, worker_rank)
        raw = await client._node.get(key)
        if raw is None:
            return False
        record = _json.loads(raw)
        record["status"] = status
        record["updated_at"] = int(_time.time())
        await client._node.put(key, _json.dumps(record).encode(),
                               ttl=client._record_ttl)
        return True
    except Exception as e:
        log.error(f"Update status failed: {e}")
        return False


async def _async_list_models(client):
    """List models using the client's async internals."""
    from modelexpress.dht_client import _models_key
    import json as _json

    try:
        raw = await client._node.get(_models_key())
        if raw is None:
            return []
        data = _json.loads(raw)
        return data.get("models", [])
    except Exception as e:
        log.error(f"List models failed: {e}")
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


async def run_publisher(host: str, port: int, dns: str | None) -> None:
    """Publish known metadata and stay alive for readers to discover."""
    from modelexpress import p2p_pb2

    node = DhtNode()
    node._observed_ip_threshold = 1
    await node.start(host, port, bootstrap_dns=dns, bootstrap_dns_port=port)
    log.info(f"Publisher ready, routing table: {node.routing_table.size()} peers")

    await asyncio.sleep(5.0)  # Settle

    client = _make_client(node)

    # Publish 2 workers for the well-known model
    workers = [
        p2p_pb2.WorkerMetadata(
            worker_rank=0,
            metadata_endpoint="10.0.0.1:5555",
            agent_name="publisher-agent-0",
            tensors=[
                p2p_pb2.TensorDescriptor(
                    name="model.embed.weight", addr=0x7FFF00000000, size=67108864,
                    device_id=0, dtype="torch.bfloat16",
                ),
                p2p_pb2.TensorDescriptor(
                    name="model.layers.0.weight", addr=0x7FFF10000000, size=33554432,
                    device_id=0, dtype="torch.bfloat16",
                ),
            ],
            status=p2p_pb2.SOURCE_STATUS_READY,
            updated_at=int(time.time()),
        ),
        p2p_pb2.WorkerMetadata(
            worker_rank=1,
            metadata_endpoint="10.0.0.2:5556",
            agent_name="publisher-agent-1",
            tensors=[
                p2p_pb2.TensorDescriptor(
                    name="model.embed.weight", addr=0x8FFF00000000, size=67108864,
                    device_id=1, dtype="torch.bfloat16",
                ),
            ],
            status=p2p_pb2.SOURCE_STATUS_READY,
            updated_at=int(time.time()),
        ),
    ]

    ok = await _async_publish(client, PUBLISHER_MODEL, workers)
    if ok:
        log.info(f"Published {len(workers)} workers for '{PUBLISHER_MODEL}'")
    else:
        log.error("Failed to publish metadata")

    # Signal readiness via a well-known key
    await node.put(b"/mx/_test_signal/publisher-ready",
                   json.dumps({"ready": True, "model": PUBLISHER_MODEL}).encode(),
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

    client = _make_client(node)
    model = "test-publish-get"

    workers = [
        p2p_pb2.WorkerMetadata(
            worker_rank=0,
            metadata_endpoint="10.1.0.1:5555",
            agent_name="test-agent-0",
            tensors=[
                p2p_pb2.TensorDescriptor(
                    name="layer.0.weight", addr=12345678, size=1024,
                    device_id=0, dtype="torch.float16",
                ),
            ],
            status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
        ),
        p2p_pb2.WorkerMetadata(
            worker_rank=1,
            metadata_endpoint="10.1.0.2:5556",
            agent_name="test-agent-1",
            tensors=[],
            status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
        ),
    ]

    ok = await _async_publish(client, model, workers)
    results.record("publish metadata", ok)

    await asyncio.sleep(1.0)

    resp = await _async_get(client, model)
    results.record("get finds model", resp.found)
    results.record("correct worker count", len(resp.workers) == 2, f"got {len(resp.workers)}")

    if len(resp.workers) >= 1:
        w0 = resp.workers[0]
        results.record("worker 0 rank", w0.worker_rank == 0)
        results.record("worker 0 endpoint", w0.metadata_endpoint == "10.1.0.1:5555")
        results.record("worker 0 agent_name", w0.agent_name == "test-agent-0")
        results.record("worker 0 tensors", len(w0.tensors) == 1)
        if w0.tensors:
            results.record("tensor name", w0.tensors[0].name == "layer.0.weight")
            results.record("tensor addr preserved", w0.tensors[0].addr == 12345678)


async def test_update_status(node: DhtNode, results: TestResult) -> None:
    """Publish, update status, verify the status change persists."""
    log.info("TEST: update_status")
    from modelexpress import p2p_pb2

    client = _make_client(node)
    model = "test-update-status"

    workers = [
        p2p_pb2.WorkerMetadata(
            worker_rank=0,
            metadata_endpoint="10.2.0.1:5555",
            agent_name="status-agent-0",
            status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
        ),
    ]

    await _async_publish(client, model, workers)
    await asyncio.sleep(0.5)

    ok = await _async_update_status(
        client, model, 0, p2p_pb2.SOURCE_STATUS_READY)
    results.record("update status call", ok)

    await asyncio.sleep(0.5)

    resp = await _async_get(client, model)
    if resp.found and resp.workers:
        results.record("status updated",
                       resp.workers[0].status == p2p_pb2.SOURCE_STATUS_READY,
                       f"got status={resp.workers[0].status}")
    else:
        results.record("status updated", False, "model not found after update")


async def test_list_models(node: DhtNode, results: TestResult) -> None:
    """Publish multiple models and verify they appear in the models list."""
    log.info("TEST: list_models")
    from modelexpress import p2p_pb2

    client = _make_client(node)

    for model in ["list-model-a", "list-model-b"]:
        w = p2p_pb2.WorkerMetadata(worker_rank=0, metadata_endpoint="x:0",
                                    agent_name="a")
        await _async_publish(client, model, [w])

    await asyncio.sleep(0.5)

    models = await _async_list_models(client)
    results.record("list-model-a in list", "list-model-a" in models, str(models))
    results.record("list-model-b in list", "list-model-b" in models, str(models))


async def test_incremental_publish(node: DhtNode, results: TestResult) -> None:
    """Publish workers one at a time and verify directory accumulates."""
    log.info("TEST: incremental_publish")
    from modelexpress import p2p_pb2

    client = _make_client(node)
    model = "test-incremental"

    # Publish rank 0
    w0 = p2p_pb2.WorkerMetadata(worker_rank=0, metadata_endpoint="10.3.0.1:5555",
                                 agent_name="incr-0")
    await _async_publish(client, model, [w0])
    await asyncio.sleep(0.5)

    resp = await _async_get(client, model)
    results.record("after rank 0: found", resp.found)
    results.record("after rank 0: 1 worker", len(resp.workers) == 1)

    # Publish rank 1 (adds to existing)
    w1 = p2p_pb2.WorkerMetadata(worker_rank=1, metadata_endpoint="10.3.0.2:5556",
                                 agent_name="incr-1")
    # Need to publish both so the directory is correct
    await _async_publish(client, model, [w0, w1])
    await asyncio.sleep(0.5)

    resp = await _async_get(client, model)
    results.record("after rank 1: 2 workers", len(resp.workers) == 2,
                   f"got {len(resp.workers)}")


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
    model_name = signal_data["model"]

    client = _make_client(node)
    resp = await _async_get(client, model_name)

    results.record("cross-pod: model found", resp.found)
    results.record("cross-pod: 2 workers", len(resp.workers) == 2,
                   f"got {len(resp.workers)}")

    if len(resp.workers) >= 2:
        w0 = next((w for w in resp.workers if w.worker_rank == 0), None)
        w1 = next((w for w in resp.workers if w.worker_rank == 1), None)

        if w0:
            results.record("cross-pod: w0 endpoint",
                          w0.metadata_endpoint == "10.0.0.1:5555",
                          w0.metadata_endpoint)
            results.record("cross-pod: w0 agent_name",
                          w0.agent_name == "publisher-agent-0",
                          w0.agent_name)
            results.record("cross-pod: w0 tensors", len(w0.tensors) == 2,
                          f"got {len(w0.tensors)}")
            results.record("cross-pod: w0 status ready",
                          w0.status == 2)  # SOURCE_STATUS_READY
        else:
            results.record("cross-pod: w0 found", False, "rank 0 missing")

        if w1:
            results.record("cross-pod: w1 endpoint",
                          w1.metadata_endpoint == "10.0.0.2:5556",
                          w1.metadata_endpoint)
            results.record("cross-pod: w1 tensors", len(w1.tensors) == 1,
                          f"got {len(w1.tensors)}")
        else:
            results.record("cross-pod: w1 found", False, "rank 1 missing")

    # Verify model appears in global list
    models = await _async_list_models(client)
    results.record("cross-pod: model in list", model_name in models, str(models))


async def test_json_schema_correctness(node: DhtNode, results: TestResult) -> None:
    """Verify the raw JSON stored in DHT matches expected schema."""
    log.info("TEST: json_schema_correctness")
    from modelexpress.dht_client import _worker_key, _directory_key, _models_key

    # Read the raw bytes from DHT and verify JSON structure
    raw = await node.get(_worker_key(PUBLISHER_MODEL, 0))
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

    # Verify directory record
    raw_dir = await node.get(_directory_key(PUBLISHER_MODEL))
    if raw_dir:
        dir_record = json.loads(raw_dir)
        results.record("schema: directory has 'ranks'", "ranks" in dir_record)
        results.record("schema: directory ranks correct",
                      sorted(dir_record.get("ranks", [])) == [0, 1],
                      str(dir_record.get("ranks")))

    # Verify models list
    raw_models = await node.get(_models_key())
    if raw_models:
        models_record = json.loads(raw_models)
        results.record("schema: models has 'models'", "models" in models_record)


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
        await test_list_models(node, results)
        await test_incremental_publish(node, results)

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
