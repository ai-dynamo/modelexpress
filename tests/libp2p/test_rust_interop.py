# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-level interop test: connect pure-Python to a rust-libp2p node.

Validates the full protocol stack (multistream-select + Noise XX + Yamux +
Kademlia GET) at the connection level, without DhtNode orchestration.

Requires the Rust interop binary:
    cd tests/libp2p_kad_interop/rust_node && cargo build --release
"""

import asyncio
import json
import logging
import os
import re
import signal
import subprocess

import pytest

from mx_libp2p.crypto import Ed25519Identity
from mx_libp2p.connection import dial, IDENTIFY_PROTOCOL
from mx_libp2p.kademlia import kad_get_value, KADEMLIA_PROTOCOL

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
TEST_VALUE = json.dumps({"rank": 0, "tensors": [{"name": "layer.0.weight", "size": 1024}]})


def start_rust_node() -> tuple[subprocess.Popen, str, int]:
    """Start the Rust node in put mode and return (process, host, port)."""
    env = os.environ.copy()
    env["RUST_LOG"] = "info"

    proc = subprocess.Popen(
        [
            RUST_NODE_BIN,
            "--mode", "put",
            "--timeout-secs", "30",
            "--key", TEST_KEY,
            "--value", TEST_VALUE,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    for line in proc.stdout:
        line = line.strip()
        if line.startswith("LISTEN_ADDR="):
            addr = line.split("=", 1)[1]
            match = re.match(r"/ip4/([^/]+)/tcp/(\d+)/p2p/(.+)", addr)
            if match:
                host = match.group(1)
                if host == "0.0.0.0":
                    host = "127.0.0.1"
                return proc, host, int(match.group(2))

    proc.kill()
    raise RuntimeError("Rust node didn't print LISTEN_ADDR")


@pytest.fixture
def rust_node():
    """Start a Rust node for the test, clean up on exit."""
    if not os.path.exists(RUST_NODE_BIN):
        pytest.skip(f"Rust binary not found: {RUST_NODE_BIN}")

    proc, host, port = start_rust_node()
    yield host, port
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


async def test_low_level_rust_get(rust_node):
    """Connect to Rust node and GET a record using raw Kademlia RPC."""
    host, port = rust_node
    await asyncio.sleep(0.5)

    identity = Ed25519Identity.generate()
    conn = await asyncio.wait_for(
        dial(identity, host, port, supported_protocols=[IDENTIFY_PROTOCOL, KADEMLIA_PROTOCOL]),
        timeout=10.0,
    )

    await asyncio.sleep(1.0)

    response = await asyncio.wait_for(
        kad_get_value(conn, TEST_KEY.encode("utf-8")),
        timeout=10.0,
    )

    assert response["record"] is not None, "No record in response"
    value = json.loads(response["record"]["value"].decode("utf-8"))
    assert value == json.loads(TEST_VALUE)

    await conn.close()
