# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mx_libp2p: minimal, asyncio-native libp2p implementation for ModelExpress.

Supports exactly one protocol stack: TCP + Noise XX (Ed25519) + Yamux + Kademlia.
No pluggable transports, no relay, no pubsub. Purpose-built for DHT metadata exchange.
"""

from .dht import DhtNode  # noqa: F401

__all__ = ["DhtNode"]
