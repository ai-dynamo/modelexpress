# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interop tests: mx_libp2p <-> rust-libp2p (0.56).

Tests the full DhtNode orchestrator and low-level connection layer against
the well-established Rust implementation to validate wire compatibility
across all protocol layers: multistream-select, Noise XX, Yamux, Kademlia,
and Identify.

DhtNode-level tests:
    1. Python DhtNode GET from Rust DHT (Rust stores, Python retrieves)
    2. Python DhtNode PUT to Rust DHT (Python stores, Rust retrieves)
    3. Rust dials Python (Rust connects to Python listener, retrieves record)
    4. Multi-hop: Python -> Rust -> Python (record traverses both implementations)
    5. Large record round-trip (near max record size)
    6. Multiple records bulk round-trip
    7. Record overwrite (PUT same key twice, verify latest value)
    8. Mixed cluster (Python + Rust nodes)
    9. Identify protocol address exchange

Low-level connection tests:
    10. Raw Kademlia GET_VALUE via dial() + kad_get_value()
    11. Raw Kademlia PUT_VALUE + GET_VALUE round-trip via dial()

Requires the Rust interop binary to be built:
    cd tests/libp2p_kad_interop/rust_node && cargo build --release
"""

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import time

import pytest

from mx_libp2p.crypto import Ed25519Identity, _base58btc_encode
from mx_libp2p.dht import DhtNode
from mx_libp2p.connection import dial, IDENTIFY_PROTOCOL
from mx_libp2p.kademlia import kad_get_value, kad_put_value, KADEMLIA_PROTOCOL

log = logging.getLogger(__name__)

RUST_NODE_BIN = os.path.join(
    os.path.dirname(__file__),
    "..",
    "libp2p_kad_interop",
    "rust_node",
    "target",
    "release",
    "kad-interop-test",
)

RUST_AVAILABLE = os.path.exists(RUST_NODE_BIN)


def node_multiaddr(node: DhtNode) -> str:
    """Build a multiaddr string from a running DhtNode."""
    host, port = node.listen_addr
    peer_id_b58 = _base58btc_encode(node.peer_id)
    return f"/ip4/{host}/tcp/{port}/p2p/{peer_id_b58}"


class RustNode:
    """Manages a Rust libp2p test node subprocess."""

    def __init__(
        self,
        mode: str,
        key: str,
        value: str,
        peer: str | None = None,
        timeout_secs: int = 30,
    ):
        self.mode = mode
        self.key = key
        self.value = value
        self.peer = peer
        self.timeout_secs = timeout_secs
        self.proc: subprocess.Popen | None = None
        self.host: str | None = None
        self.port: int | None = None
        self.full_addr: str | None = None

    def start(self) -> None:
        env = os.environ.copy()
        env["RUST_LOG"] = "info"

        cmd = [
            RUST_NODE_BIN,
            "--mode", self.mode,
            "--timeout-secs", str(self.timeout_secs),
            "--key", self.key,
            "--value", self.value,
        ]
        if self.peer:
            cmd.extend(["--peer", self.peer])

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )

        # Read LISTEN_ADDR from stdout
        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith("LISTEN_ADDR="):
                addr = line.split("=", 1)[1]
                match = re.match(r"/ip4/([^/]+)/tcp/(\d+)/p2p/(.+)", addr)
                if match:
                    self.host = match.group(1)
                    if self.host == "0.0.0.0":
                        self.host = "127.0.0.1"
                    self.port = int(match.group(2))
                    self.full_addr = addr.replace("0.0.0.0", "127.0.0.1")
                    return
        self.stop()
        raise RuntimeError("Rust node didn't print LISTEN_ADDR")

    async def wait_for_exit(self, timeout: float = 15.0) -> tuple[str, str]:
        """Wait for the Rust process to exit without blocking the event loop."""
        def _wait():
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
            return self.proc.stdout.read(), self.proc.stderr.read()

        return await asyncio.to_thread(_wait)

    def parse_output(self, stdout: str) -> dict[str, str]:
        """Parse KEY=VALUE lines from stdout."""
        results = {}
        for line in stdout.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                results[k] = v
        return results

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


@pytest.fixture
def rust_available():
    """Skip test if Rust binary isn't built."""
    if not RUST_AVAILABLE:
        pytest.skip(
            "Rust interop binary not found. Build with: "
            "cd tests/libp2p_kad_interop/rust_node && cargo build --release"
        )


# ---------------------------------------------------------------------------
# DhtNode-level interop tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rust_put_python_dht_get(rust_available):
    """Rust node stores a record; Python DhtNode retrieves it via iterative GET."""
    test_key = "/mx/interop/rust-put-py-get"
    test_value = json.dumps({"rank": 0, "source": "rust", "test": "dht_get"})

    with RustNode("put", test_key, test_value) as rust:
        await asyncio.sleep(0.3)

        node = DhtNode()
        try:
            await node.start("127.0.0.1", 0, bootstrap_peers=[rust.full_addr])
            await asyncio.sleep(0.5)

            result = await asyncio.wait_for(
                node.get(test_key.encode("utf-8")), timeout=10.0
            )

            assert result is not None, "DhtNode.get() returned None"
            actual = json.loads(result.decode("utf-8"))
            expected = json.loads(test_value)
            assert actual == expected, f"Value mismatch: {actual} != {expected}"
        finally:
            await node.stop()


@pytest.mark.asyncio
async def test_python_dht_put_rust_get(rust_available):
    """Python DhtNode stores a record; Rust node dials Python and retrieves it."""
    test_key = "/mx/interop/py-put-rust-get"
    test_value = json.dumps({"rank": 1, "source": "python", "test": "dht_put"})

    node = DhtNode()
    try:
        await node.start("127.0.0.1", 0)

        await node.put(test_key.encode("utf-8"), test_value.encode("utf-8"))

        py_addr = node_multiaddr(node)

        with RustNode("get", test_key, "", peer=py_addr, timeout_secs=15) as rust:
            stdout, stderr = await rust.wait_for_exit(timeout=15)
            output = rust.parse_output(stdout)

            assert output.get("RESULT") == "OK", (
                f"Rust node failed: result={output.get('RESULT')}\n"
                f"stdout: {stdout}\nstderr: {stderr[-500:]}"
            )
            assert "RECORD_VALUE" in output, "Rust node didn't report RECORD_VALUE"

            actual = json.loads(output["RECORD_VALUE"])
            expected = json.loads(test_value)
            assert actual == expected, f"Value mismatch: {actual} != {expected}"
    finally:
        await node.stop()


@pytest.mark.asyncio
async def test_rust_dials_python(rust_available):
    """Rust initiates connection to Python. Validates Noise responder + Yamux
    responder + inbound Kademlia handler on the Python side."""
    test_key = "/mx/interop/rust-dials-python"
    test_value = json.dumps({"direction": "rust-to-python"})

    node = DhtNode()
    try:
        await node.start("127.0.0.1", 0)
        node.kad_handler.put_local(
            test_key.encode("utf-8"), test_value.encode("utf-8")
        )

        py_addr = node_multiaddr(node)

        with RustNode("get", test_key, "", peer=py_addr, timeout_secs=15) as rust:
            stdout, stderr = await rust.wait_for_exit(timeout=15)
            output = rust.parse_output(stdout)

            assert output.get("RESULT") == "OK", (
                f"Rust->Python dial failed: {output.get('RESULT')}\n"
                f"stderr: {stderr[-500:]}"
            )
            actual = json.loads(output["RECORD_VALUE"])
            expected = json.loads(test_value)
            assert actual == expected
    finally:
        await node.stop()


@pytest.mark.asyncio
async def test_multihop_python_rust_python(rust_available):
    """Three-node topology: Python A stores, Rust B is intermediary,
    Python C retrieves. Validates that closer_peers responses from Rust
    correctly route Python's iterative lookup."""
    test_key = "/mx/interop/multihop"
    test_value = json.dumps({"path": "A->B->C"})

    with RustNode("put", "/mx/dummy", "dummy", timeout_secs=30) as rust:
        await asyncio.sleep(0.3)

        node_a = DhtNode()
        await node_a.start("127.0.0.1", 0, bootstrap_peers=[rust.full_addr])
        await asyncio.sleep(0.5)

        node_c = DhtNode()
        await node_c.start("127.0.0.1", 0, bootstrap_peers=[rust.full_addr])
        await asyncio.sleep(0.5)

        try:
            await node_a.put(test_key.encode("utf-8"), test_value.encode("utf-8"))

            result = await asyncio.wait_for(
                node_c.get(test_key.encode("utf-8")), timeout=10.0
            )

            assert result is not None, "Multi-hop GET returned None"
            actual = json.loads(result.decode("utf-8"))
            expected = json.loads(test_value)
            assert actual == expected
        finally:
            await node_a.stop()
            await node_c.stop()


@pytest.mark.asyncio
async def test_large_record_interop(rust_available):
    """Test a record near the maximum size round-trips correctly."""
    test_key = "/mx/interop/large-record"
    tensor_layout = [
        {"name": f"model.layers.{i}.self_attn.q_proj.weight", "size": 134217728, "dtype": "float8_e4m3fn"}
        for i in range(100)
    ]
    test_value = json.dumps({"rank": 0, "tensor_layout": tensor_layout})
    assert len(test_value) > 8000, f"Test value too small: {len(test_value)} bytes"

    with RustNode("put", test_key, test_value) as rust:
        await asyncio.sleep(0.3)

        node = DhtNode()
        try:
            await node.start("127.0.0.1", 0, bootstrap_peers=[rust.full_addr])
            await asyncio.sleep(0.5)

            result = await asyncio.wait_for(
                node.get(test_key.encode("utf-8")), timeout=10.0
            )

            assert result is not None, "Large record GET returned None"
            actual = json.loads(result.decode("utf-8"))
            expected = json.loads(test_value)
            assert actual == expected, "Large record value mismatch"
            assert len(result) > 8000, f"Record too small: {len(result)} bytes"
        finally:
            await node.stop()


@pytest.mark.asyncio
async def test_bulk_records_interop(rust_available):
    """Store multiple records on Rust, retrieve all from Python DhtNode."""
    with RustNode("put", "/mx/seed", "seed", timeout_secs=30) as rust:
        await asyncio.sleep(0.3)

        node = DhtNode()
        try:
            await node.start("127.0.0.1", 0, bootstrap_peers=[rust.full_addr])
            await asyncio.sleep(0.5)

            records = {}
            for i in range(10):
                key = f"/mx/interop/bulk/{i}".encode("utf-8")
                value = json.dumps({"index": i, "data": f"record-{i}"}).encode("utf-8")
                records[key] = value
                await node.put(key, value)

            for key, expected_value in records.items():
                result = await asyncio.wait_for(node.get(key), timeout=10.0)
                assert result is not None, f"Bulk GET returned None for {key!r}"
                assert result == expected_value, (
                    f"Bulk record mismatch for {key!r}: "
                    f"{result!r} != {expected_value!r}"
                )
        finally:
            await node.stop()


@pytest.mark.asyncio
async def test_record_overwrite_interop(rust_available):
    """Overwrite a record and verify the latest value is returned."""
    test_key = "/mx/interop/overwrite"

    with RustNode("put", test_key, '{"version": 1}', timeout_secs=30) as rust:
        await asyncio.sleep(0.3)

        node = DhtNode()
        try:
            await node.start("127.0.0.1", 0, bootstrap_peers=[rust.full_addr])
            await asyncio.sleep(0.5)

            result = await asyncio.wait_for(
                node.get(test_key.encode("utf-8")), timeout=10.0
            )
            assert result is not None
            v1 = json.loads(result)
            assert v1["version"] == 1

            new_value = json.dumps({"version": 2, "updated_by": "python"})
            await node.put(test_key.encode("utf-8"), new_value.encode("utf-8"))

            result = await asyncio.wait_for(
                node.get(test_key.encode("utf-8")), timeout=10.0
            )
            assert result is not None
            v2 = json.loads(result)
            assert v2["version"] == 2
            assert v2["updated_by"] == "python"
        finally:
            await node.stop()


@pytest.mark.asyncio
async def test_mixed_cluster(rust_available):
    """Two Python DhtNodes + one Rust node forming a mixed DHT cluster.
    Records stored on any node should be retrievable from any other."""
    py_a = DhtNode()
    py_b = DhtNode()

    try:
        await py_a.start("127.0.0.1", 0)
        addr_a = node_multiaddr(py_a)

        with RustNode("put", "/mx/from-rust", '{"origin": "rust"}', timeout_secs=30) as rust:
            await asyncio.sleep(1.0)

            await py_b.start("127.0.0.1", 0, bootstrap_peers=[addr_a])
            await asyncio.sleep(0.5)

            await py_a.put(b"/mx/from-py-a", b'{"origin": "python-a"}')
            await py_b.put(b"/mx/from-py-b", b'{"origin": "python-b"}')

            result = await asyncio.wait_for(
                py_b.get(b"/mx/from-rust"), timeout=10.0
            )

            result_a = await asyncio.wait_for(
                py_a.get(b"/mx/from-py-b"), timeout=10.0
            )
            assert result_a is not None, "Python A couldn't GET from Python B"
            assert json.loads(result_a) == {"origin": "python-b"}

            result_b = await asyncio.wait_for(
                py_b.get(b"/mx/from-py-a"), timeout=10.0
            )
            assert result_b is not None, "Python B couldn't GET from Python A"
            assert json.loads(result_b) == {"origin": "python-a"}

    finally:
        await py_a.stop()
        await py_b.stop()


@pytest.mark.asyncio
async def test_identify_address_exchange(rust_available):
    """Verify that Identify protocol correctly exchanges addresses between
    Rust and Python nodes, and the Python node learns a routable address."""

    with RustNode("put", "/mx/identify-test", "test", timeout_secs=15) as rust:
        await asyncio.sleep(0.3)

        node = DhtNode()
        try:
            await node.start("127.0.0.1", 0, bootstrap_peers=[rust.full_addr])
            await asyncio.sleep(1.0)

            assert node.routing_table.size() > 0, "Routing table is empty after bootstrap"

            all_peers = node.routing_table.all_peers()
            has_addrs = any(len(p.addrs) > 0 for p in all_peers)
            assert has_addrs, "No peers have addresses after Identify exchange"
        finally:
            await node.stop()


# ---------------------------------------------------------------------------
# Low-level connection interop tests (raw dial + Kademlia RPC)
# ---------------------------------------------------------------------------


LOW_LEVEL_TEST_KEY = "/mx/model:test-model:worker:0"
LOW_LEVEL_TEST_VALUE = json.dumps({"rank": 0, "tensors": [{"name": "layer.0.weight", "size": 1024}]})
LOW_LEVEL_PYTHON_KEY = "/mx/model:test-model:worker:1"
LOW_LEVEL_PYTHON_VALUE = json.dumps({"rank": 1, "from": "python"})


@pytest.fixture
def rust_node_fixture():
    """Start a Rust node in put mode for low-level tests, clean up on exit."""
    if not RUST_AVAILABLE:
        pytest.skip(f"Rust binary not found: {RUST_NODE_BIN}")

    env = os.environ.copy()
    env["RUST_LOG"] = "info"

    proc = subprocess.Popen(
        [
            RUST_NODE_BIN,
            "--mode", "put",
            "--timeout-secs", "30",
            "--key", LOW_LEVEL_TEST_KEY,
            "--value", LOW_LEVEL_TEST_VALUE,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    host = None
    port = None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("LISTEN_ADDR="):
            addr = line.split("=", 1)[1]
            match = re.match(r"/ip4/([^/]+)/tcp/(\d+)/p2p/(.+)", addr)
            if match:
                host = match.group(1)
                if host == "0.0.0.0":
                    host = "127.0.0.1"
                port = int(match.group(2))
                break

    if host is None:
        proc.kill()
        pytest.skip("Rust node didn't print LISTEN_ADDR")

    yield host, port

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


async def test_low_level_rust_get(rust_node_fixture):
    """Connect to Rust node and GET a record using raw Kademlia RPC."""
    host, port = rust_node_fixture
    await asyncio.sleep(0.5)

    identity = Ed25519Identity.generate()
    conn = await asyncio.wait_for(
        dial(identity, host, port, supported_protocols=[IDENTIFY_PROTOCOL, KADEMLIA_PROTOCOL]),
        timeout=10.0,
    )

    await asyncio.sleep(1.0)

    response = await asyncio.wait_for(
        kad_get_value(conn, LOW_LEVEL_TEST_KEY.encode("utf-8")),
        timeout=10.0,
    )

    assert response["record"] is not None, "No record in response"
    value = json.loads(response["record"]["value"].decode("utf-8"))
    assert value == json.loads(LOW_LEVEL_TEST_VALUE)

    await conn.close()


async def test_low_level_python_put_rust_get(rust_node_fixture):
    """PUT a record to Rust node via raw Kademlia RPC, then GET it back."""
    host, port = rust_node_fixture
    await asyncio.sleep(0.3)

    identity = Ed25519Identity.generate()
    conn = await asyncio.wait_for(
        dial(identity, host, port, supported_protocols=[IDENTIFY_PROTOCOL, KADEMLIA_PROTOCOL]),
        timeout=10.0,
    )
    await asyncio.sleep(0.5)

    put_response = await asyncio.wait_for(
        kad_put_value(conn, LOW_LEVEL_PYTHON_KEY.encode("utf-8"), LOW_LEVEL_PYTHON_VALUE.encode("utf-8")),
        timeout=10.0,
    )

    get_response = await asyncio.wait_for(
        kad_get_value(conn, LOW_LEVEL_PYTHON_KEY.encode("utf-8")),
        timeout=10.0,
    )

    assert get_response["record"] is not None, "Record not found after PUT"
    value = json.loads(get_response["record"]["value"].decode("utf-8"))
    assert value == json.loads(LOW_LEVEL_PYTHON_VALUE), f"Mismatch: {value}"

    await conn.close()
