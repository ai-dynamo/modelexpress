# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Datacenter topology signal for topology-aware source selection.

A worker reports its RDMA-fabric location as a ``{domain: value}`` map (e.g.
``{"block": "b1", "rack": "r3", "host": "node7"}``), published once at
registration and surfaced on ``SourceInstanceRef.topology``. The
``topology_aware`` selector ranks candidates by the narrowest domain the target
and source share (see ``source_selection.TopologyAwareSelector``).

The representation matches **Grove's** ``ClusterTopology`` CRD
(``clustertopologies.grove.io``), which is the source-of-truth topology hierarchy
Dynamo/Grove expose to workloads. Its ``spec.levels`` is an ordered list (broad
-> narrow) of ``{domain, key}`` entries, where ``domain`` is a platform-agnostic
level from the fixed set below and ``key`` is the node-label key carrying that
domain's value for a node. MX therefore keys its map on the Grove ``domain`` (so
the metadata lines up across the fleet) and takes the value from the node's label
for that domain's ``key``. Both come through the environment, so MX stays
runtime-agnostic and needs no Kubernetes API access from the worker:

- ``MX_P2P_TOPOLOGY_LEVELS``: comma-separated Grove domains, broad -> narrow,
  matching the cluster's ``ClusterTopology`` ``spec.levels`` order, e.g.
  ``"region,zone,datacenter,block,rack,host"``.
- ``MX_P2P_TOPOLOGY``: a JSON object of ``{domain: value}`` for THIS node, e.g.
  ``'{"block":"b1","rack":"r3","host":"node7"}'``. The deploying operator
  populates it from the node's labels using the ``ClusterTopology`` domain->key
  mapping. Missing or unparseable yields ``{}``, so topology-aware selection
  degrades to rendezvous ordering rather than failing.

NVLink is not a level here: MX P2P moves weights between *different* replicas
(different nodes, RDMA transport), so the relevant domains are the inter-node
ones (``rack``/``block`` and above); ``host``/``numa`` matter only for
co-located pairs, which use the NVLink backend NIXL selects automatically.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger("modelexpress.topology")

# Grove ClusterTopology (clustertopologies.grove.io, v1alpha1) domain enum,
# broadest -> narrowest. Levels outside this set still work (a ClusterTopology
# binding may extend it) but are flagged, since a typo would silently misalign
# this node's map with the rest of the fleet.
GROVE_TOPOLOGY_DOMAINS = (
    "region",
    "zone",
    "datacenter",
    "block",
    "rack",
    "host",
    "numa",
)
_warned_unknown: set[str] = set()


def resolve_levels(raw: Optional[str] = None) -> list[str]:
    """Ordered topology levels (broad -> narrow) from ``MX_P2P_TOPOLOGY_LEVELS``.

    Unset defaults to Grove's canonical domain order, so ``topology_aware`` works
    out-of-the-box on a Dynamo-managed cluster (where the per-node values arrive
    via the injected topology, see ``local_topology``). Ordering still collapses
    to rendezvous whenever a node has no topology values.
    """
    if raw is None:
        from . import envs

        raw = envs.MX_P2P_TOPOLOGY_LEVELS
    if not raw:
        return list(GROVE_TOPOLOGY_DOMAINS)
    levels = [lvl.strip() for lvl in raw.split(",") if lvl.strip()]
    for lvl in levels:
        if lvl not in GROVE_TOPOLOGY_DOMAINS and lvl not in _warned_unknown:
            _warned_unknown.add(lvl)
            logger.warning(
                "MX_P2P_TOPOLOGY_LEVELS contains %r, not a Grove ClusterTopology "
                "domain %s; topology_aware still works but this node's map may "
                "not align with the rest of the fleet.",
                lvl,
                GROVE_TOPOLOGY_DOMAINS,
            )
    return levels


# Dynamo's operator projects each scheduled worker's node topology into a
# directory (one file per Grove domain, contents = this node's value for it) --
# the same source Dynamo's own topology-aware KV transfer reads. Consuming it
# means MX reports exactly the topology Dynamo already resolved for the node, so
# the metadata lines up on real datacenter hardware with no extra wiring.
_DYNAMO_TOPOLOGY_DIR_ENV = "DYN_TOPOLOGY_MOUNT_PATH"
_DYNAMO_TOPOLOGY_DIR_DEFAULT = "/etc/dynamo/topology"


def _read_dynamo_topology_dir() -> dict[str, str]:
    import os

    path = os.environ.get(_DYNAMO_TOPOLOGY_DIR_ENV, _DYNAMO_TOPOLOGY_DIR_DEFAULT)
    out: dict[str, str] = {}
    try:
        names = os.listdir(path)
    except Exception:
        return {}
    for name in names:
        if name.startswith("."):
            continue
        fp = os.path.join(path, name)
        try:
            if not os.path.isfile(fp):
                continue
            with open(fp) as f:
                value = f.read().strip()
        except Exception:
            continue
        if value:
            out[name] = value
    return out


def local_topology(raw: Optional[str] = None) -> dict[str, str]:
    """This node's ``{domain: value}`` topology map.

    Resolution order: the explicit ``MX_P2P_TOPOLOGY`` JSON override, else the
    Dynamo operator's projected topology directory (``DYN_TOPOLOGY_MOUNT_PATH``,
    default ``/etc/dynamo/topology`` -- one file per Grove domain). Best-effort:
    unset/unparseable/absent yields ``{}`` (this node then shares no domain with
    any source, i.e. rendezvous ordering).
    """
    if raw is None:
        from . import envs

        raw = envs.MX_P2P_TOPOLOGY
    if not raw:
        return _read_dynamo_topology_dir()
    try:
        parsed = json.loads(raw)
    except Exception as e:
        logger.warning("Invalid MX_P2P_TOPOLOGY (%r): %s", raw, e)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("MX_P2P_TOPOLOGY is not a JSON object: %r", raw)
        return {}
    return {str(k): str(v) for k, v in parsed.items() if v is not None}
