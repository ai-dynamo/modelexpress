# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SLURM hostlist expansion and bootstrap tests.

Covers:
- Simple comma-separated hostlists
- Range expansion with and without zero-padding
- Bracket lists (commas inside brackets)
- Mixed ranges and lists in brackets
- Multiple bracket groups (cartesian product)
- Single hostnames and empty input
- Top-level comma splitting that respects brackets
- DhtNode SLURM bootstrap via loopback
"""

import asyncio
import logging

import pytest

from mx_libp2p.dht import DhtNode
from mx_libp2p.slurm import expand_hostlist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# -- Unit tests: expand_hostlist (no network) ---------------------------------


def test_simple_list():
    assert expand_hostlist("host1,host2") == ["host1", "host2"]


def test_range():
    assert expand_hostlist("node[01-04]") == ["node01", "node02", "node03", "node04"]


def test_range_no_padding():
    assert expand_hostlist("gpu[1-3]") == ["gpu1", "gpu2", "gpu3"]


def test_bracket_list():
    assert expand_hostlist("gpu[1,3,5]") == ["gpu1", "gpu3", "gpu5"]


def test_mixed_range_list():
    assert expand_hostlist("n[1-3,5,8-9]") == ["n1", "n2", "n3", "n5", "n8", "n9"]


def test_multiple_brackets():
    result = expand_hostlist("rack[1-2]-node[01-02]")
    assert result == [
        "rack1-node01", "rack1-node02",
        "rack2-node01", "rack2-node02",
    ]


def test_no_brackets():
    assert expand_hostlist("singlehost") == ["singlehost"]


def test_empty():
    assert expand_hostlist("") == []


def test_comma_with_brackets():
    result = expand_hostlist("a[1-2],b[3-4]")
    assert result == ["a1", "a2", "b3", "b4"]


def test_zero_padding_preserved():
    result = expand_hostlist("n[08-12]")
    assert result == ["n08", "n09", "n10", "n11", "n12"]


# -- Integration test: DhtNode SLURM bootstrap via loopback ------------------


@pytest.mark.asyncio
async def test_dht_slurm_bootstrap():
    """Two DhtNodes where one bootstraps from the other via SLURM hostlist pointing at localhost."""
    node_a = DhtNode()
    node_b = DhtNode()

    try:
        # Start node A as the "seed" peer
        await node_a.start("127.0.0.1", 0, enable_mdns=False)
        assert node_a.listen_addr is not None
        _, port_a = node_a.listen_addr

        # Start node B using SLURM hostlist bootstrap pointing at A.
        # Bind on 0.0.0.0 so the self-filter (which checks listen addr)
        # doesn't discard 127.0.0.1 as "our own IP".
        await node_b.start(
            "0.0.0.0", 0,
            bootstrap_slurm="localhost",
            bootstrap_dns_port=port_a,
            enable_mdns=False,
        )

        # Wait for routing tables to populate
        for _ in range(20):
            if node_a.routing_table.size() > 0 and node_b.routing_table.size() > 0:
                break
            await asyncio.sleep(0.2)

        assert node_b.routing_table.size() > 0, "Node B did not discover node A via SLURM bootstrap"

        # Verify PUT/GET works
        key = b"/test/slurm-key"
        value = b"discovered-via-slurm"
        stored = await node_b.put(key, value)
        assert stored > 0, "PUT failed"

        retrieved = await node_a.get(key)
        assert retrieved == value, f"GET returned {retrieved!r}, expected {value!r}"

    finally:
        await node_b.stop()
        await node_a.stop()
