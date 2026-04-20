# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Peer discovery substrate probes for ModelExpress.

This package provides substrate-agnostic peer discovery mechanisms used to
bootstrap higher-level peer networks.

Currently exposes:
- :mod:`mx_peer_discovery.mdns`: multicast DNS service discovery (RFC 6762/6763).
- :mod:`mx_peer_discovery.slurm`: SLURM compact hostlist expansion.
- :mod:`mx_peer_discovery.dns`: hostname resolution via ``getaddrinfo``.
- :mod:`mx_peer_discovery.static`: explicit ``host:port`` peer lists.
"""

from mx_peer_discovery.dns import (
    filter_own_ips,
    resolve_hostlist,
    resolve_hostname,
)
from mx_peer_discovery.mdns import (
    DEFAULT_SERVICE_TYPE,
    Config,
    MdnsDiscovery,
)
from mx_peer_discovery.slurm import expand_hostlist
from mx_peer_discovery.static import (
    DEFAULT_ENV_VAR,
    endpoints_from_env,
    parse_endpoints,
)

__all__ = [
    "Config",
    "DEFAULT_ENV_VAR",
    "DEFAULT_SERVICE_TYPE",
    "MdnsDiscovery",
    "endpoints_from_env",
    "expand_hostlist",
    "filter_own_ips",
    "parse_endpoints",
    "resolve_hostlist",
    "resolve_hostname",
]
