#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional interop test: both GET and PUT between Python and Rust.

Test 1: Rust PUT -> Python GET (validated above)
Test 2: Python PUT -> Rust GET
"""

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
from mx_libp2p.crypto import Ed25519Identity
from mx_libp2p.connection import dial, IDENTIFY_PROTOCOL
from mx_libp2p.kademlia import kad_get_value, kad_put_value, KADEMLIA_PROTOCOL

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
log = logging.getLogger("test")

RUST_NODE_BIN = os.path.join(
    os.path.dirname(__file__),
    "..",
    "libp2p_kad_interop",
    "rust_node",
    "target",
    "release",
    "kad-interop-test",
)

TEST_KEY = "/mx/model:test-model:worker:0"
TEST_VALUE_RUST = json.dumps({"rank": 0, "from": "rust"})
TEST_VALUE_PYTHON = json.dumps({"rank": 1, "from": "python"})
PYTHON_KEY = "/mx/model:test-model:worker:1"


def start_rust_node(mode: str, key: str, value: str, peer: str | None = None, timeout_secs: int = 30) -> subprocess.Popen:
    env = os.environ.copy()
    env["RUST_LOG"] = "info"

    cmd = [RUST_NODE_BIN, "--mode", mode, "--timeout-secs", str(timeout_secs), "--key", key, "--value", value]
    if peer:
        cmd.extend(["--peer", peer])

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )


def read_rust_addr(proc: subprocess.Popen) -> tuple[str, int, str]:
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("LISTEN_ADDR="):
            addr = line.split("=", 1)[1]
            match = re.match(r"/ip4/([^/]+)/tcp/(\d+)/p2p/(.+)", addr)
            if match:
                host = match.group(1)
                if host == "0.0.0.0":
                    host = "127.0.0.1"
                return host, int(match.group(2)), addr
    raise RuntimeError("No LISTEN_ADDR from Rust node")


async def test_rust_put_python_get():
    """Test 1: Rust stores a record, Python retrieves it."""
    log.info("--- Test 1: Rust PUT -> Python GET ---")

    rust_proc = start_rust_node("put", TEST_KEY, TEST_VALUE_RUST)
    try:
        host, port, _ = read_rust_addr(rust_proc)
        await asyncio.sleep(0.3)

        identity = Ed25519Identity.generate()
        conn = await asyncio.wait_for(
            dial(identity, host, port, supported_protocols=[IDENTIFY_PROTOCOL, KADEMLIA_PROTOCOL]),
            timeout=10.0,
        )
        await asyncio.sleep(0.5)

        response = await asyncio.wait_for(
            kad_get_value(conn, TEST_KEY.encode("utf-8")),
            timeout=10.0,
        )

        assert response["record"] is not None, "No record in response"
        value = json.loads(response["record"]["value"].decode("utf-8"))
        assert value == json.loads(TEST_VALUE_RUST), f"Mismatch: {value}"
        log.info("PASS: Rust PUT -> Python GET")

        await conn.close()
    finally:
        rust_proc.send_signal(signal.SIGTERM)
        rust_proc.wait(timeout=5)


async def test_python_put_rust_get():
    """Test 2: Python stores a record on Rust node, Rust node can serve it."""
    log.info("--- Test 2: Python PUT -> Rust GET ---")

    # Start Rust in "put" mode (so it's listening and has an existing record)
    rust_proc = start_rust_node("put", TEST_KEY, TEST_VALUE_RUST)
    try:
        host, port, full_addr = read_rust_addr(rust_proc)
        await asyncio.sleep(0.3)

        identity = Ed25519Identity.generate()
        conn = await asyncio.wait_for(
            dial(identity, host, port, supported_protocols=[IDENTIFY_PROTOCOL, KADEMLIA_PROTOCOL]),
            timeout=10.0,
        )
        await asyncio.sleep(0.5)

        # PUT a new record via Python
        log.info(f"Putting record with key: {PYTHON_KEY}")
        put_response = await asyncio.wait_for(
            kad_put_value(conn, PYTHON_KEY.encode("utf-8"), TEST_VALUE_PYTHON.encode("utf-8")),
            timeout=10.0,
        )
        log.info(f"PUT response type: {put_response['type']}")

        # GET it back to verify it was stored
        log.info("Getting record back...")
        get_response = await asyncio.wait_for(
            kad_get_value(conn, PYTHON_KEY.encode("utf-8")),
            timeout=10.0,
        )

        assert get_response["record"] is not None, "Record not found after PUT"
        value = json.loads(get_response["record"]["value"].decode("utf-8"))
        assert value == json.loads(TEST_VALUE_PYTHON), f"Mismatch: {value}"
        log.info("PASS: Python PUT -> readback GET")

        await conn.close()
    finally:
        rust_proc.send_signal(signal.SIGTERM)
        rust_proc.wait(timeout=5)


async def main():
    log.info("=== Bidirectional Interop Test: Pure Python <-> rust-libp2p ===")
    log.info("")

    await test_rust_put_python_get()
    log.info("")
    await test_python_put_rust_get()

    log.info("")
    log.info("=== ALL TESTS PASSED ===")
    print("\nRESULT=OK")


if __name__ == "__main__":
    asyncio.run(main())
