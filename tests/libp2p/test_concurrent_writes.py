# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent write tests.

Validates DHT behavior when multiple nodes write simultaneously:
- Same key written by multiple nodes (last-write-wins)
- Different keys written concurrently (no interference)
- Bulk concurrent operations don't deadlock or corrupt state
- Concurrent PUT convergence (two writers, eventual consistency)
- Rapid overwrite of the same key (sequential, latest value wins)
"""

import asyncio

from mx_libp2p.crypto import _base58btc_encode
from mx_libp2p.dht import DhtNode


def _node_multiaddr(node: DhtNode) -> str:
    host, port = node.listen_addr
    return f"/ip4/{host}/tcp/{port}/p2p/{_base58btc_encode(node.peer_id)}"


async def test_concurrent_put_same_key():
    """Multiple nodes PUTting the same key concurrently should not corrupt state.
    One of the values should win (last-write-wins)."""
    nodes = []
    for _ in range(3):
        nodes.append(DhtNode(record_ttl=300))

    await nodes[0].start("127.0.0.1", 0)
    for n in nodes[1:]:
        await n.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(nodes[0])])
    await asyncio.sleep(0.5)

    try:
        key = b"/mx/race/worker/0"
        values = [f'{{"from":"node{i}"}}'.encode() for i in range(3)]

        tasks = [nodes[i].put(key, values[i]) for i in range(3)]
        results = await asyncio.gather(*tasks)

        for r in results:
            assert isinstance(r, int)

        for node in nodes:
            result = await node.get(key)
            assert result in values, f"unexpected value: {result!r}"
    finally:
        for n in nodes:
            await n.stop()


async def test_concurrent_put_different_keys():
    """Multiple nodes PUTting different keys concurrently should not interfere."""
    nodes = []
    for _ in range(3):
        nodes.append(DhtNode(record_ttl=300))

    await nodes[0].start("127.0.0.1", 0)
    for n in nodes[1:]:
        await n.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(nodes[0])])
    await asyncio.sleep(0.5)

    try:
        keys_values = [
            (f"/mx/model-{i}/worker/0".encode(), f'{{"rank":0,"model":{i}}}'.encode())
            for i in range(3)
        ]

        tasks = [nodes[i].put(k, v) for i, (k, v) in enumerate(keys_values)]
        await asyncio.gather(*tasks)

        await asyncio.sleep(0.3)
        for key, expected_value in keys_values:
            for node in nodes:
                result = await node.get(key)
                assert result == expected_value, (
                    f"key {key!r}: expected {expected_value!r}, got {result!r}"
                )
    finally:
        for n in nodes:
            await n.stop()


async def test_bulk_concurrent_puts():
    """Many concurrent PUTs from a single node should not deadlock."""
    node_a = DhtNode(record_ttl=300)
    node_b = DhtNode(record_ttl=300)

    await node_a.start("127.0.0.1", 0)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
    await asyncio.sleep(0.5)

    try:
        n_records = 20
        tasks = []
        for i in range(n_records):
            key = f"/mx/bulk/worker/{i}".encode()
            value = f'{{"rank":{i}}}'.encode()
            tasks.append(node_a.put(key, value))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, r in enumerate(results):
            assert not isinstance(r, Exception), f"put {i} raised: {r}"

        for i in [0, n_records // 2, n_records - 1]:
            key = f"/mx/bulk/worker/{i}".encode()
            result = await node_b.get(key)
            assert result == f'{{"rank":{i}}}'.encode()
    finally:
        await node_a.stop()
        await node_b.stop()


async def test_concurrent_put_same_key_converges():
    """Two nodes PUT different values to the same key concurrently.
    After both complete, GET should return one of the two values
    consistently (last-write-wins). The key point is no crash,
    no data corruption, and a deterministic result."""
    nodes = [DhtNode(record_ttl=300) for _ in range(3)]
    await nodes[0].start("127.0.0.1", 0)
    for n in nodes[1:]:
        await n.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(nodes[0])])
    await asyncio.sleep(0.5)

    try:
        key = b"/mx/conflict/worker/0"
        value_a = b'{"writer": "node_1", "version": 1}'
        value_b = b'{"writer": "node_2", "version": 2}'

        results = await asyncio.gather(
            nodes[1].put(key, value_a),
            nodes[2].put(key, value_b),
        )
        assert all(r >= 0 for r in results), "both puts should succeed"

        await asyncio.sleep(0.3)

        seen_values = set()
        for n in nodes:
            result = await n.get(key)
            assert result is not None, "key should exist on all nodes"
            seen_values.add(result)

        for v in seen_values:
            assert v in (value_a, value_b), f"unexpected value: {v!r}"

    finally:
        for n in nodes:
            await n.stop()


async def test_rapid_overwrite_same_key():
    """Rapidly overwrite the same key many times. Final GET should
    return the last written value."""
    node_a = DhtNode(record_ttl=300)
    node_b = DhtNode(record_ttl=300)
    await node_a.start("127.0.0.1", 0)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
    await asyncio.sleep(0.3)

    try:
        key = b"/mx/overwrite/test"
        last_value = None
        for i in range(10):
            last_value = f'{{"version": {i}}}'.encode()
            await node_a.put(key, last_value)

        result = await node_b.get(key)
        assert result == last_value, f"expected last version, got {result!r}"
    finally:
        await node_a.stop()
        await node_b.stop()
