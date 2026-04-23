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
- ``"k8s-service"`` / ``"service"`` - K8s-Service-routed decentralized
  backend. Returns :class:`MxK8sServiceClient`.
"""

from __future__ import annotations

import logging
import os

from .client import MxClient, MxClientBase
from .k8s_service_client import MxK8sServiceClient

logger = logging.getLogger("modelexpress.client_factory")

_CENTRAL_BACKEND_ALIASES = frozenset({
    "", "server", "redis", "kubernetes", "k8s", "crd",
})
_K8S_SERVICE_ALIASES = frozenset({"k8s-service", "service"})


def create_metadata_client(
    worker_rank: int | None = None,
) -> MxClientBase:
    """Create the metadata client dictated by ``MX_METADATA_BACKEND``.

    ``worker_rank`` is only consumed by backends that resolve a rank-
    specific endpoint (currently :class:`MxK8sServiceClient`);
    others ignore it.
    """
    backend = os.environ.get("MX_METADATA_BACKEND", "").lower().strip()
    if backend in _CENTRAL_BACKEND_ALIASES:
        logger.debug("create_metadata_client: central MxClient (backend=%r)", backend)
        return MxClient()
    if backend in _K8S_SERVICE_ALIASES:
        logger.info(
            "create_metadata_client: MxK8sServiceClient "
            "(backend=%r, worker_rank=%s)",
            backend, worker_rank,
        )
        return MxK8sServiceClient(worker_rank=worker_rank)
    raise ValueError(
        f"Unknown MX_METADATA_BACKEND value {backend!r}. "
        f"Supported: {sorted(_CENTRAL_BACKEND_ALIASES | _K8S_SERVICE_ALIASES)}"
    )
