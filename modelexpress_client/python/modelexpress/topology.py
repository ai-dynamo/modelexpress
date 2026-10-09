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
import os
import time
from dataclasses import dataclass
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
    except json.JSONDecodeError as e:
        logger.warning("Invalid MX_P2P_TOPOLOGY (%r): %s", raw, e)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("MX_P2P_TOPOLOGY is not a JSON object: %r", raw)
        return {}
    return {str(k): str(v) for k, v in parsed.items() if v is not None}


TOPOLOGY_ENFORCEMENTS = ("required", "preferred")
_DEFAULT_ENFORCEMENT = "required"
DOMAIN_WAIT_TIMEOUT_S = 30.0
_DOMAIN_WAIT_POLL_S = 1.0
# Topology sources whose wait already expired in this process. Load and publish
# both resolve the domain; without this a missing file would cost the timeout
# twice. Later calls still read once, so a late-arriving file is picked up.
_expired_waits: set[tuple[str, Optional[str], str]] = set()


@dataclass(frozen=True)
class TopologyPolicy:
    """Transfer-domain constraint between a target and its P2P sources.

    Mirrors Dynamo's KV-transfer policy (``DYN_KV_TRANSFER_DOMAIN`` /
    ``DYN_KV_TRANSFER_ENFORCEMENT``): ``required`` makes sources outside this
    node's ``domain`` value ineligible, ``preferred`` only moves same-domain
    sources ahead of the rest. ``local_value`` is None when this node's value
    could not be resolved.
    """

    domain: str
    enforcement: str
    local_value: Optional[str]

    @property
    def required(self) -> bool:
        return self.enforcement == "required"


def wait_for_domain(
    domain: str,
    timeout: Optional[float] = None,
    poll_interval: float = _DOMAIN_WAIT_POLL_S,
) -> dict[str, str]:
    """``local_topology()``, polled until it carries ``domain`` or ``timeout``.

    The Dynamo operator copies the node label onto the pod only after the pod
    is scheduled, and the kubelet refreshes the Downward API volume later
    still, so the domain file can be absent for the first seconds of startup.
    """
    key = (
        domain,
        os.environ.get("MX_P2P_TOPOLOGY"),
        os.environ.get(_DYNAMO_TOPOLOGY_DIR_ENV, _DYNAMO_TOPOLOGY_DIR_DEFAULT),
    )
    if timeout is None:
        timeout = DOMAIN_WAIT_TIMEOUT_S
    if key in _expired_waits:
        timeout = 0.0
    deadline = time.monotonic() + timeout
    topology = local_topology()
    while domain not in topology and time.monotonic() < deadline:
        time.sleep(min(poll_interval, max(deadline - time.monotonic(), 0.0)))
        topology = local_topology()
    if domain not in topology:
        _expired_waits.add(key)
    return topology


def resolve_policy(timeout: Optional[float] = None) -> Optional[TopologyPolicy]:
    """The configured transfer-domain policy, or None when none is configured."""
    from . import envs

    if timeout is None:
        timeout = DOMAIN_WAIT_TIMEOUT_S

    domain = (envs.MX_P2P_TOPOLOGY_DOMAIN or "").strip()
    if not domain:
        return None
    enforcement = (envs.MX_P2P_TOPOLOGY_ENFORCEMENT or _DEFAULT_ENFORCEMENT).strip().lower()
    if enforcement not in TOPOLOGY_ENFORCEMENTS:
        # Fail closed: a typo must not silently re-enable cross-domain transfers.
        logger.warning(
            "MX_P2P_TOPOLOGY_ENFORCEMENT=%r is not one of %s; using %r",
            envs.MX_P2P_TOPOLOGY_ENFORCEMENT,
            TOPOLOGY_ENFORCEMENTS,
            _DEFAULT_ENFORCEMENT,
        )
        enforcement = _DEFAULT_ENFORCEMENT
    # Under preferred an unknown value only keeps the selector's order, which is
    # not worth delaying the load for.
    if enforcement != "required":
        timeout = 0.0
    start = time.monotonic()
    local_value = wait_for_domain(domain, timeout=timeout).get(domain)
    if local_value is None:
        logger.warning(
            "MX_P2P_TOPOLOGY_DOMAIN=%r but this node reports no value for it "
            "(waited %.1fs; check MX_P2P_TOPOLOGY or %s)",
            domain,
            time.monotonic() - start,
            os.environ.get(_DYNAMO_TOPOLOGY_DIR_ENV, _DYNAMO_TOPOLOGY_DIR_DEFAULT),
        )
    return TopologyPolicy(domain=domain, enforcement=enforcement, local_value=local_value)


def blocks_selection(policy: Optional[TopologyPolicy]) -> bool:
    """Whether P2P selection must stop: required, but this node's value is unknown.

    Under ``required`` no source can be proven same-domain, so callers fall back
    without listing sources.
    """
    return policy is not None and policy.required and policy.local_value is None


def in_domain(candidate, policy: TopologyPolicy) -> bool:
    """Whether ``candidate`` published this node's value for the policy domain."""
    raw = getattr(candidate, "topology", None)
    return (
        policy.local_value is not None
        and bool(raw)
        and raw.get(policy.domain) == policy.local_value
    )


def count_in_domain(candidates: list, policy: TopologyPolicy) -> int:
    """Same-domain candidates, for the ``topology_matched`` funnel stage.

    Counted separately from ``apply_policy``'s output because ``preferred``
    keeps every candidate.
    """
    return sum(1 for c in candidates if in_domain(c, policy))


def warn_if_no_domain_match(
    candidates: list, policy: Optional[TopologyPolicy], worker: object
) -> None:
    """Explain an empty selection when ``required`` dropped every compatible source.

    Otherwise the only signal is ``topology_matched=0``. The usual cause during
    a rollout is sources whose client predates published topology metadata.
    """
    if policy is None or not policy.required or not candidates:
        return
    if any(in_domain(c, policy) for c in candidates):
        return
    missing = sum(
        1 for c in candidates if not (getattr(c, "topology", None) or {}).get(policy.domain)
    )
    logger.warning(
        "[Worker %s] No P2P source shares %s=%r: %d compatible source(s), %d "
        "without published %r (client predates topology metadata, or no topology "
        "source), %d in another %s. Upgrade sources before targets require the "
        "domain, or use MX_P2P_TOPOLOGY_ENFORCEMENT=preferred during the rollout.",
        worker,
        policy.domain,
        policy.local_value,
        len(candidates),
        missing,
        policy.domain,
        len(candidates) - missing,
        policy.domain,
    )


def apply_policy(candidates: list, policy: Optional[TopologyPolicy]) -> list:
    """Filter (required) or stably partition (preferred) ordered candidates.

    Order within each group is preserved, so this composes with any selector.
    Under ``required`` a source that published no value for the domain is
    ineligible, and so is every source when this node's own value is unknown.
    """
    if policy is None:
        return list(candidates)
    matched = [c for c in candidates if in_domain(c, policy)]
    if policy.required:
        return matched
    return matched + [c for c in candidates if not in_domain(c, policy)]
