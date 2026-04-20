# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static peer-list substrate.

For deployments without any discovery substrate (no K8s API, no SLURM, no
mDNS reachability), peers can be listed explicitly via configuration.

Two entry points:

- :func:`parse_endpoints`: parse a ``host:port,host:port`` string into
  ``(host, port)`` tuples.
- :func:`endpoints_from_env`: read an environment variable (default
  ``MX_PEER_ENDPOINTS``) and parse it.

Input format: comma-separated ``host:port`` entries. IPv6 addresses must
be bracketed: ``[::1]:4001``. Whitespace around entries is ignored.
Malformed entries are logged at WARNING and skipped (the rest still
parse), so a single typo doesn't blank the whole list.
"""

import logging
import os

log = logging.getLogger(__name__)

DEFAULT_ENV_VAR = "MX_PEER_ENDPOINTS"


def parse_endpoints(value: str) -> list[tuple[str, int]]:
    """Parse a comma-separated endpoint string into ``(host, port)`` tuples.

    :param value: Comma-separated endpoint list. Example:
        ``"10.0.0.1:4001,10.0.0.2:4001,[::1]:4001"``.
    :returns: List of ``(host, port)`` tuples in input order. Empty on
        empty input. Malformed entries are skipped with a WARNING.
    """
    if not value or not value.strip():
        return []
    result: list[tuple[str, int]] = []
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parsed = _parse_one_endpoint(entry)
        if parsed is not None:
            result.append(parsed)
    return result


def endpoints_from_env(env_var: str = DEFAULT_ENV_VAR) -> list[tuple[str, int]]:
    """Read an environment variable and parse it as an endpoint list.

    :param env_var: Environment variable name. Default
        :data:`DEFAULT_ENV_VAR` (``MX_PEER_ENDPOINTS``).
    :returns: List of ``(host, port)`` tuples. Empty if the variable is
        unset or empty.
    """
    return parse_endpoints(os.environ.get(env_var, ""))


def _parse_one_endpoint(entry: str) -> tuple[str, int] | None:
    """Parse a single ``host:port`` entry. IPv6 must be bracketed.

    Returns ``None`` on malformed input (with a WARNING log line).
    """
    if entry.startswith("["):
        close_idx = entry.find("]")
        if close_idx == -1:
            log.warning(f"malformed IPv6 endpoint (missing ']'): {entry!r}")
            return None
        host = entry[1:close_idx]
        rest = entry[close_idx + 1 :]
        if not rest.startswith(":"):
            log.warning(f"missing ':port' after bracketed host: {entry!r}")
            return None
        port_str = rest[1:]
    else:
        if ":" not in entry:
            log.warning(f"endpoint missing ':port': {entry!r}")
            return None
        host, _, port_str = entry.rpartition(":")
        if not host:
            log.warning(f"endpoint missing host: {entry!r}")
            return None

    if not host:
        log.warning(f"endpoint missing host: {entry!r}")
        return None

    try:
        port = int(port_str)
    except ValueError:
        log.warning(f"invalid port in endpoint: {entry!r}")
        return None
    if not 1 <= port <= 65535:
        log.warning(f"port out of range in endpoint: {entry!r}")
        return None
    return (host, port)
