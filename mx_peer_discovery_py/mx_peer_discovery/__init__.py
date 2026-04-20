# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Peer discovery substrate probes for ModelExpress.

This package provides substrate-agnostic peer discovery mechanisms used to
bootstrap higher-level peer networks. It has no libp2p dependencies.

Currently exposes:
- :mod:`mx_peer_discovery.mdns`: multicast DNS service discovery (RFC 6762/6763).
- :mod:`mx_peer_discovery.slurm`: SLURM compact hostlist expansion.
- :mod:`mx_peer_discovery.dns`: hostname resolution via ``getaddrinfo``.
"""

from mx_peer_discovery.dns import (
    filter_own_ips,
    resolve_hostlist,
    resolve_hostname,
)
from mx_peer_discovery.mdns import (
    DEFAULT_SERVICE_NAME,
    MdnsDiscovery,
)
from mx_peer_discovery.slurm import expand_hostlist

__all__ = [
    "DEFAULT_SERVICE_NAME",
    "MdnsDiscovery",
    "expand_hostlist",
    "filter_own_ips",
    "resolve_hostlist",
    "resolve_hostname",
]
