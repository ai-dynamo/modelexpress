# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""kademlite: minimal, asyncio-native libp2p implementation.

Supports exactly one protocol stack: TCP + Noise XX (Ed25519) + Yamux + Kademlia.
No pluggable transports, no relay, no pubsub. Purpose-built for DHT metadata exchange.
"""

from importlib.metadata import PackageNotFoundError, version

from .dht import DhtNode  # noqa: F401
from .dht_bootstrap import NoKnownPeersError  # noqa: F401

try:
    __version__ = version("kademlite")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0+unknown"

__all__ = ["DhtNode", "NoKnownPeersError"]
