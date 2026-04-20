# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Metadata client factory.

Picks the metadata client implementation based on ``MX_METADATA_BACKEND``.
Callers construct a client via :func:`create_metadata_client` rather than
instantiating a specific class, so deployments can switch between the
server-as-coordinator mode (``MxClient`` talking to modelexpress-server)
and the peer-direct mode (``PeerDirectMetadataClient`` using
``mx_peer_discovery`` substrates) without code changes.

Supported ``MX_METADATA_BACKEND`` values:

- ``peer-direct``: New peer-first mode. No central server. Uses
  ``PeerDirectMetadataClient`` with the substrate named by
  ``MX_PEER_DISCOVERY_SUBSTRATE`` (default ``mdns``, also ``static``).
- ``redis``, ``redis-only``, ``kubernetes``, ``k8s``, ``crd``,
  ``kubernetes-only``: Legacy server-as-coordinator mode. These are
  server-side backend selectors; the client talks to the central
  modelexpress-server via ``MxClient`` regardless of which backend the
  server uses.
- unset / empty / any other value: ``MxClient`` default.

The peer-direct mode is the forward direction; legacy server-backed modes
remain available for deployments that haven't migrated.
"""

from __future__ import annotations

import logging
import os

from .client import MxClient
from .peer_direct_client import PeerDirectMetadataClient

logger = logging.getLogger("modelexpress.client_factory")

_PEER_DIRECT_BACKEND = "peer-direct"
_LEGACY_SERVER_BACKENDS = frozenset({
    "redis",
    "redis-only",
    "kubernetes",
    "k8s",
    "crd",
    "kubernetes-only",
})


def create_metadata_client():
    """Construct the metadata client selected by ``MX_METADATA_BACKEND``.

    Returns an object duck-typed to the ``MxClient`` interface (publish_metadata,
    list_sources, get_metadata, update_status, close), so callers do not
    need to branch on the concrete type.
    """
    backend = os.environ.get("MX_METADATA_BACKEND", "").strip().lower()

    if backend == _PEER_DIRECT_BACKEND:
        substrate = os.environ.get("MX_PEER_DISCOVERY_SUBSTRATE", "mdns").strip().lower()
        logger.info(
            f"Creating PeerDirectMetadataClient (substrate={substrate})"
        )
        return PeerDirectMetadataClient(substrate=substrate)

    if backend and backend not in _LEGACY_SERVER_BACKENDS:
        logger.warning(
            f"Unknown MX_METADATA_BACKEND={backend!r}; falling back to MxClient. "
            f"Known values: peer-direct, {', '.join(sorted(_LEGACY_SERVER_BACKENDS))}."
        )

    logger.info(f"Creating MxClient (backend={backend or 'default'})")
    return MxClient()
