# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Metadata client factory.

Selects the metadata client implementation from the
``MX_METADATA_BACKEND`` env var. The returned object duck-types
:class:`MxClient`, so callers (loaders, strategies) never need to
branch on backend.

Supported values:

- ``""`` / ``"server"`` / ``"redis"`` / ``"kubernetes"`` / ``"k8s"`` /
  ``"crd"`` - central ModelExpress server (default). Returns
  :class:`MxClient`.
- ``"k8s-service"`` / ``"service"`` - K8s Service-routed peer-direct.
  Returns :class:`K8sServiceMetadataClient`.

Additional backends (peer-direct with mDNS, libp2p DHT, etc.) may
slot in here over time without touching call sites.
"""

from __future__ import annotations

import logging
import os
from typing import Union

from .client import MxClient
from .k8s_service_client import K8sServiceMetadataClient

logger = logging.getLogger("modelexpress.client_factory")

MetadataClient = Union[MxClient, K8sServiceMetadataClient]

_CENTRAL_BACKEND_ALIASES = frozenset({
    "", "server", "redis", "kubernetes", "k8s", "crd",
})
_K8S_SERVICE_ALIASES = frozenset({"k8s-service", "service"})


def create_metadata_client(
    worker_rank: int | None = None,
) -> MetadataClient:
    """Create the metadata client dictated by ``MX_METADATA_BACKEND``.

    ``worker_rank`` is only consumed by backends that resolve a rank-
    specific endpoint (currently :class:`K8sServiceMetadataClient`);
    others ignore it.
    """
    backend = os.environ.get("MX_METADATA_BACKEND", "").lower().strip()
    if backend in _CENTRAL_BACKEND_ALIASES:
        logger.debug("create_metadata_client: central MxClient (backend=%r)", backend)
        return MxClient()
    if backend in _K8S_SERVICE_ALIASES:
        logger.info(
            "create_metadata_client: K8sServiceMetadataClient "
            "(backend=%r, worker_rank=%s)",
            backend, worker_rank,
        )
        return K8sServiceMetadataClient(worker_rank=worker_rank)
    raise ValueError(
        f"Unknown MX_METADATA_BACKEND value {backend!r}. "
        f"Supported: {sorted(_CENTRAL_BACKEND_ALIASES | _K8S_SERVICE_ALIASES)}"
    )
