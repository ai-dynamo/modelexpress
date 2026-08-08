# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone DHT bootstrap seed.

Runs a participation-only kademlite node that joins the Kademlia mesh,
helps route, and serves as a stable bootstrap target for ``dht``-backend
workers. It publishes no records and consumes none; it exists purely so a
joining worker always has a known, long-lived peer to dial.

Run one seed, or a small quorum behind a headless Service, and point worker
``MX_DHT_BOOTSTRAP_DNS`` at that Service. A multi-replica quorum should
cross-bootstrap to itself (set ``MX_DHT_BOOTSTRAP_DNS`` on the seeds too,
pointing at their own Service) so the quorum converges among itself before
workers arrive - that pre-converged frontier is what keeps a large,
simultaneous worker cold-start from stalling.

Invoke with ``python -m modelexpress.dht_seed``.

Environment:
  MX_DHT_LISTEN          listen address, ``host:port`` (default "0.0.0.0:4001").
  MX_DHT_BOOTSTRAP_DNS   optional DNS name the seed bootstraps from, typically
                         the seed quorum's own headless Service for cross-seed
                         convergence. Unset means singleton-seed mode.
  MX_DHT_BOOTSTRAP_PORT  port at which MX_DHT_BOOTSTRAP_DNS peers are dialed
                         (default "4001").
"""

from __future__ import annotations

import asyncio
import logging
import os

from kademlite import DhtNode

logger = logging.getLogger("modelexpress.dht_seed")

_DEFAULT_LISTEN = "0.0.0.0:4001"
_DEFAULT_BOOTSTRAP_PORT = 4001


def _parse_listen(listen: str) -> tuple[str, int]:
    """Split a ``host:port`` listen string, defaulting the host to 0.0.0.0."""
    if ":" in listen:
        host, port_str = listen.rsplit(":", 1)
        return host or "0.0.0.0", int(port_str)
    return "0.0.0.0", int(listen)


async def _run() -> None:
    host, port = _parse_listen(os.environ.get("MX_DHT_LISTEN", _DEFAULT_LISTEN))

    bootstrap_dns = os.environ.get("MX_DHT_BOOTSTRAP_DNS", "").strip() or None
    env_port = os.environ.get("MX_DHT_BOOTSTRAP_PORT", "").strip()
    bootstrap_port = int(env_port) if env_port else _DEFAULT_BOOTSTRAP_PORT

    node = DhtNode()
    # With bootstrap_dns set (multi-replica quorum), the seed dials its own
    # headless Service - which resolves to every seed IP - and runs one
    # convergence pass so the quorum meshes before workers arrive. A singleton
    # seed (bootstrap_dns unset) simply listens and waits for workers to dial.
    await node.start(
        host,
        port,
        bootstrap_dns=bootstrap_dns,
        bootstrap_dns_port=bootstrap_port,
        enable_mdns=False,
        wait_until_routable=bool(bootstrap_dns),
    )
    logger.info(
        "dht seed listening on %s:%d (peer_id=%s, bootstrap_dns=%s)",
        host,
        port,
        node.peer_id.hex(),
        bootstrap_dns or "(none)",
    )

    try:
        await asyncio.Future()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await node.stop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(_run())


if __name__ == "__main__":
    main()
