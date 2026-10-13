# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Partition source discovery by locality.

``ListSources`` is keyed by ``mx_source_id``, which hashes every field of
``SourceIdentity`` including ``extra_parameters``. Folding a deployer-chosen
locality value into the identity therefore makes the broker return only the
peers that share it, without any server or wire-format change. Deployers use
it where a worker can reach only some of the peers the broker knows about, for
example a Kubernetes namespace whose network policy allows pod-to-pod traffic
only inside the namespace: every unreachable candidate otherwise costs a full
manifest-fetch timeout and a model re-initialization before the next one is
tried.

The value is opaque to ModelExpress. ``MX_SOURCE_DOMAIN`` unset leaves every
identity, and every ``mx_source_id``, exactly as before.
"""

from __future__ import annotations

import logging

from .. import envs, p2p_pb2

logger = logging.getLogger("modelexpress.metadata.source_domain")

SOURCE_DOMAIN_KEY = "source_domain"

_announced: set[str] = set()


def source_domain() -> str:
    """The configured locality, or ``""`` when discovery is not partitioned."""
    return envs.MX_SOURCE_DOMAIN


def apply_source_domain(identity: p2p_pb2.SourceIdentity) -> p2p_pb2.SourceIdentity:
    """Fold ``MX_SOURCE_DOMAIN`` into ``identity.extra_parameters`` in place.

    Returns the same identity for call-site convenience. A no-op when the
    variable is unset or blank.
    """
    domain = source_domain()
    if not domain:
        return identity
    identity.extra_parameters[SOURCE_DOMAIN_KEY] = domain
    if domain not in _announced:
        _announced.add(domain)
        logger.info(
            "Source discovery partitioned by MX_SOURCE_DOMAIN=%r "
            "(extra_parameters[%r]); peers outside this domain are not offered",
            domain,
            SOURCE_DOMAIN_KEY,
        )
    return identity
