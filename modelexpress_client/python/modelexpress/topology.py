# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Datacenter topology signal for topology-aware source selection.

A worker reports its RDMA-fabric location as a ``{level: domain_value}`` map
(e.g. ``{"rack": "r3", "rail": "leaf2", "host": "node7"}``), published once at
registration and surfaced on ``SourceInstanceRef.topology``. The
``topology_aware`` selector ranks candidates by the narrowest domain the target
and source share (see ``source_selection.TopologyAwareSelector``).

Both the level ORDER (broad -> narrow) and the per-node values come from the
same node labels Dynamo/Grove already use; MX consumes them through the
environment so it stays runtime-agnostic and needs no Kubernetes API access from
the worker itself:

- ``MX_P2P_TOPOLOGY_LEVELS``: comma-separated level names, broad -> narrow,
  matching the cluster's Dynamo ``ClusterTopology`` ``spec.levels``, e.g.
  ``"region,zone,datacenter,block,rack,rail,host"``. The exact RDMA ordering
  (whether rail sits above rack) is fabric-dependent, so it is configured here
  rather than fixed in code.
- ``MX_P2P_TOPOLOGY``: a JSON object of ``{level: value}`` for THIS node, e.g.
  ``'{"rack":"r3","rail":"leaf2","host":"node7"}'``. The deploying operator
  populates it from the node's topology labels. Missing or unparseable yields
  ``{}``, so topology-aware selection degrades to rendezvous ordering rather
  than failing.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger("modelexpress.topology")


def resolve_levels(raw: Optional[str] = None) -> list[str]:
    """Ordered topology levels (broad -> narrow) from ``MX_P2P_TOPOLOGY_LEVELS``.

    Returns ``[]`` when unset, which makes ``topology_aware`` collapse to
    rendezvous ordering (no level is ever shared).
    """
    if raw is None:
        from . import envs

        raw = envs.MX_P2P_TOPOLOGY_LEVELS
    if not raw:
        return []
    return [lvl.strip() for lvl in raw.split(",") if lvl.strip()]


def local_topology(raw: Optional[str] = None) -> dict[str, str]:
    """This node's ``{level: value}`` map from ``MX_P2P_TOPOLOGY`` (JSON).

    Best-effort: an unset or unparseable value yields ``{}`` (this node then
    shares no domain with any source, i.e. rendezvous ordering).
    """
    if raw is None:
        from . import envs

        raw = envs.MX_P2P_TOPOLOGY
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception as e:
        logger.warning("Invalid MX_P2P_TOPOLOGY (%r): %s", raw, e)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("MX_P2P_TOPOLOGY is not a JSON object: %r", raw)
        return {}
    return {str(k): str(v) for k, v in parsed.items() if v is not None}
