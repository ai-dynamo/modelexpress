# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DNS and SLURM-hostlist substrate probes.

Resolves hostnames to IP addresses via ``getaddrinfo``. Two entry points:

- :func:`resolve_hostname`: single hostname (e.g. a Kubernetes headless
  Service name) -> list of IPs.
- :func:`resolve_hostlist`: SLURM compact hostlist notation -> list of IPs
  (expanded through :func:`mx_peer_discovery.slurm.expand_hostlist`, each
  name resolved, results merged and deduplicated).

Both return ordered, deduplicated IPs. Callers are responsible for dialing,
connection setup, and filtering their own IPs via :func:`filter_own_ips`.
"""

import asyncio
import logging
import socket

from mx_peer_discovery.slurm import expand_hostlist

log = logging.getLogger(__name__)


async def resolve_hostname(hostname: str, port: int) -> list[str]:
    """Resolve a hostname to its TCP-reachable IPv4/IPv6 addresses.

    :param hostname: DNS name to resolve. Typically a Kubernetes headless
        Service, a Consul service entry, or any name that resolves to
        multiple A/AAAA records.
    :param port: Port to probe; only used to hint ``getaddrinfo``'s
        service-type filter to TCP.
    :returns: Ordered list of unique IP address strings. Empty on DNS
        failure (logged at WARNING and swallowed).
    """
    try:
        loop = asyncio.get_event_loop()
        infos = await loop.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        log.warning(f"DNS resolution failed for {hostname}: {e}")
        return []
    return list(dict.fromkeys(info[4][0] for info in infos))


async def resolve_hostlist(hostlist: str, port: int) -> list[str]:
    """Expand a SLURM hostlist and resolve each name to IPs.

    :param hostlist: SLURM compact hostlist notation (see
        :func:`mx_peer_discovery.slurm.expand_hostlist` for supported
        syntax).
    :param port: Port to probe; see :func:`resolve_hostname`.
    :returns: Ordered list of unique IP address strings from all
        successfully-resolved hostnames. Unresolvable hostnames are
        logged at DEBUG and skipped. Empty if the hostlist expands to
        no names or no hostnames resolve.
    """
    hostnames = expand_hostlist(hostlist)
    if not hostnames:
        log.warning(f"hostlist expansion produced no hosts: {hostlist!r}")
        return []

    loop = asyncio.get_event_loop()
    ip_set: dict[str, None] = {}
    for hostname in hostnames:
        try:
            infos = await loop.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
            for info in infos:
                ip_set.setdefault(info[4][0], None)
        except socket.gaierror as e:
            log.debug(f"hostlist: failed to resolve {hostname}: {e}")
    return list(ip_set)


def filter_own_ips(ips: list[str], own_ips: set[str]) -> list[str]:
    """Remove IPs matching our own from the list, preserving order.

    :param ips: Candidate peer IPs (typically from
        :func:`resolve_hostname` or :func:`resolve_hostlist`).
    :param own_ips: IPs the caller considers its own. Typically bind
        address plus any observed external IPs. The substrate layer
        does not introspect interfaces: the caller decides what "own"
        means.
    :returns: ``ips`` with any entry appearing in ``own_ips`` removed,
        order preserved.
    """
    return [ip for ip in ips if ip not in own_ips]
