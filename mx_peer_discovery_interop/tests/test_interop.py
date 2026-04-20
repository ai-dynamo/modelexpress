# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-language mDNS interop tests.

Spawns the Rust test-peer binary as a subprocess, runs a Python
MdnsDiscovery in-process, and verifies each half discovers the other on
the default ``_mx-peer._tcp.local.`` service type.

Prerequisites: the Rust binary must be built first:

    cargo build -p mx-peer-discovery --bin mx-peer-discovery-test-peer

Tests auto-skip if the binary isn't found, so running the Python side
alone isn't a failure - it's just a non-result.

These tests exercise real mDNS multicast traffic on the host. In
sandboxed CI environments without multicast routing, they'll time out
rather than pass; mark as network-dependent if wiring into CI.
"""

import asyncio
import subprocess
from pathlib import Path

import pytest

from mx_peer_discovery.mdns import Config, MdnsDiscovery

# Walk from this test file up to the workspace root where Cargo's target/
# lives. test file -> tests/ -> mx_peer_discovery_interop/ -> workspace.
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

# Match the [[bin]] name in mx_peer_discovery/Cargo.toml.
RUST_BINARY_NAME = "mx-peer-discovery-test-peer"


def _find_rust_binary() -> Path | None:
    for profile in ("debug", "release"):
        candidate = WORKSPACE_ROOT / "target" / profile / RUST_BINARY_NAME
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


async def _drain_stdout_for_instance(
    proc: asyncio.subprocess.Process,
    wanted_instance: str,
    deadline: float,
) -> bool:
    """Read the Rust peer's stdout until it reports the wanted instance or the deadline hits.

    Returns True on match, False on timeout / EOF.
    """
    assert proc.stdout is not None
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            return False
        try:
            raw = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
        except asyncio.TimeoutError:
            return False
        if not raw:
            return False
        line = raw.decode(errors="replace").strip()
        # Format emitted by src/bin/test_peer.rs:
        #   DISCOVERED\t<instance>\t<port>\t<addrs>\t<txt>
        if line.startswith("DISCOVERED\t"):
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1] == wanted_instance:
                return True


@pytest.mark.asyncio
async def test_python_rust_cross_discovery():
    rust_binary = _find_rust_binary()
    if rust_binary is None:
        pytest.skip(
            "Rust test peer binary not found. Build it first with:\n"
            "  cargo build -p mx-peer-discovery --bin mx-peer-discovery-test-peer"
        )

    python_instance = "python-interop-peer"
    rust_instance = "rust-interop-peer"

    # Track what the Python side has seen.
    saw_rust = asyncio.Event()

    def on_resolved(instance: str, _port: int, _addrs: list[str], _txt: dict[str, str]) -> None:
        if instance == rust_instance:
            saw_rust.set()

    py_config = Config(
        hostname="py-interop.local.",
        ip="127.0.0.1",
        port=54001,
        txt={"lang": "python"},
        on_resolved=on_resolved,
        instance_name=python_instance,
    )
    py_discovery = MdnsDiscovery(py_config)
    await py_discovery.start()

    rust_proc = await asyncio.create_subprocess_exec(
        str(rust_binary),
        "--instance-name", rust_instance,
        "--port", "54002",
        "--ip", "127.0.0.1",
        "--hostname", "rust-interop.local.",
        "--duration-secs", "30",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Budget: zeroconf and mdns-sd both announce on register and re-announce
    # at their own cadences; 20s is generous. On healthy LANs discovery is
    # usually under 2s. We double the budget when the host network stack is
    # slow to join the multicast group.
    deadline = asyncio.get_event_loop().time() + 20.0

    try:
        # Race: read Rust stdout for "saw python", wait for Python callback
        # to fire for "saw rust".
        rust_saw_python_task = asyncio.create_task(
            _drain_stdout_for_instance(rust_proc, python_instance, deadline),
        )
        py_saw_rust_task = asyncio.create_task(saw_rust.wait())

        # Wait for both with a combined deadline.
        _, pending = await asyncio.wait(
            [rust_saw_python_task, py_saw_rust_task],
            timeout=20.0,
            return_when=asyncio.ALL_COMPLETED,
        )

        rust_saw_python = (
            rust_saw_python_task.done() and rust_saw_python_task.result() is True
        )
        py_saw_rust_value = saw_rust.is_set()

        # Cancel anything still pending before asserting.
        for task in pending:
            task.cancel()

        assert py_saw_rust_value, (
            f"Python peer never saw Rust peer ({rust_instance}) within 20s. "
            "mDNS multicast may not be routable on this host."
        )
        assert rust_saw_python, (
            f"Rust peer never saw Python peer ({python_instance}) within 20s. "
            "mDNS multicast may not be routable on this host."
        )
    finally:
        # Tear down both halves cleanly.
        rust_proc.terminate()
        try:
            await asyncio.wait_for(rust_proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            rust_proc.kill()
            await rust_proc.wait()
        await py_discovery.stop()
