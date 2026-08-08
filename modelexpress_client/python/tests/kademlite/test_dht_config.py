# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DhtNode constructor configuration of k and alpha.

Verifies that the per-instance replication factor and lookup parallelism
are honored by the routing table, exposed via accessors, and validated.
"""

import pytest

from kademlite.dht import DhtNode
from kademlite.routing import ALPHA, K


def test_default_k_and_alpha():
    node = DhtNode()
    assert node.k == K
    assert node.alpha == ALPHA
    assert node.routing_table.k == K


def test_custom_k_and_alpha():
    node = DhtNode(k=8, alpha=5)
    assert node.k == 8
    assert node.alpha == 5
    assert node.routing_table.k == 8


def test_k_zero_rejected():
    with pytest.raises(ValueError):
        DhtNode(k=0)


def test_k_negative_rejected():
    with pytest.raises(ValueError):
        DhtNode(k=-1)


def test_alpha_zero_rejected():
    with pytest.raises(ValueError):
        DhtNode(alpha=0)


def test_alpha_negative_rejected():
    with pytest.raises(ValueError):
        DhtNode(alpha=-1)


def test_k_bool_rejected():
    """bool is a subclass of int in Python; ``DhtNode(k=True)`` would
    silently behave like ``k=1``. Reject explicitly with TypeError."""
    with pytest.raises(TypeError):
        DhtNode(k=True)
    with pytest.raises(TypeError):
        DhtNode(k=False)


def test_k_float_rejected():
    """Floats slip through ``k <= 0`` checks and would only fail later
    when used as a count. Reject at construction."""
    with pytest.raises(TypeError):
        DhtNode(k=1.5)


def test_alpha_bool_rejected():
    with pytest.raises(TypeError):
        DhtNode(alpha=True)


def test_alpha_float_rejected():
    with pytest.raises(TypeError):
        DhtNode(alpha=2.0)


def test_two_nodes_independent_k():
    """Two DhtNode instances in the same process can hold different k values."""
    a = DhtNode(k=8)
    b = DhtNode(k=40)
    assert a.k == 8
    assert b.k == 40
    assert a.routing_table.k == 8
    assert b.routing_table.k == 40


def test_kad_handler_uses_node_k_for_inbound_responses():
    """Inbound FIND_NODE / GET_VALUE responses honor the node's configured k.

    Regression: prior to this fix, KadHandler._closest_peers_encoded
    hardcoded count=20 regardless of DhtNode(k=...).
    """
    a = DhtNode(k=8)
    b = DhtNode(k=40)
    assert a.kad_handler._k == 8
    assert b.kad_handler._k == 40
