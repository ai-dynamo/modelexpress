# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python side of the libp2p Kademlia DHT interop test.

Modes:
  --mode put  : Start a node, put a record, print the multiaddr, wait.
  --mode get  : Connect to a peer, get the record, verify, exit.
"""

import argparse
import asyncio
import logging
import secrets

from libp2p import new_host
from libp2p.crypto.ed25519 import create_new_key_pair
from libp2p.kad_dht.kad_dht import DHTMode, KadDHT
from libp2p.peer.peerinfo import info_from_p2p_addr
from multiaddr import Multiaddr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


async def run_put(args):
    """Start a node, store a record, and wait for peers to query it."""
    key_pair = create_new_key_pair(secrets.token_bytes(32))
    host = new_host(key_pair=key_pair)
    dht = KadDHT(host, DHTMode.SERVER)

    async with host.run(listen_addrs=[Multiaddr("/ip4/0.0.0.0/tcp/0")]):
        addrs = host.get_addrs()
        peer_id = host.get_id()
        full_addr = f"{addrs[0]}/p2p/{peer_id}"
        log.info("Listening on: %s", full_addr)
        print(f"LISTEN_ADDR={full_addr}", flush=True)

        # Store the value
        key = args.key
        value = args.value.encode()
        await dht.put_value(key, value)
        log.info("Stored record: key=%s, value=%s", key, args.value)

        # Wait for peers to query
        await asyncio.sleep(args.timeout_secs)
        log.info("Put node shutting down.")


async def run_get(args):
    """Connect to a peer, retrieve a record, and verify it."""
    key_pair = create_new_key_pair(secrets.token_bytes(32))
    host = new_host(key_pair=key_pair)
    dht = KadDHT(host, DHTMode.CLIENT)

    async with host.run(listen_addrs=[Multiaddr("/ip4/0.0.0.0/tcp/0")]):
        # Connect to the peer
        peer_addr = Multiaddr(args.peer)
        peer_info = info_from_p2p_addr(peer_addr)
        host.get_peerstore().add_addrs(peer_info.peer_id, peer_info.addrs, 300)
        await host.connect(peer_info)
        log.info("Connected to: %s", peer_info.peer_id)

        # Small delay to let routing settle
        await asyncio.sleep(1)

        # Retrieve the value
        try:
            value = await asyncio.wait_for(
                dht.get_value(args.key),
                timeout=args.timeout_secs,
            )
            decoded = value.decode() if isinstance(value, bytes) else str(value)
            log.info("Got record: key=%s, value=%s", args.key, decoded)
            print(f"RECORD_VALUE={decoded}", flush=True)
            print("RESULT=OK", flush=True)
        except TimeoutError:
            log.error("Timed out waiting for record")
            print("RESULT=TIMEOUT", flush=True)
            raise SystemExit(1)
        except Exception as e:
            log.error("Failed to get record: %s", e)
            print(f"RESULT=FAIL:{e}", flush=True)
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="py-libp2p Kademlia interop test")
    parser.add_argument("--mode", required=True, choices=["put", "get"])
    parser.add_argument(
        "--key", default="mx:model:test-model:worker:0"
    )
    parser.add_argument(
        "--value",
        default='{"rank":0,"tensors":[{"name":"layer.0.weight","size":1024}]}',
    )
    parser.add_argument("--peer", help="Peer multiaddr (required for get mode)")
    parser.add_argument("--timeout-secs", type=int, default=30)
    args = parser.parse_args()

    if args.mode == "put":
        asyncio.run(run_put(args))
    elif args.mode == "get":
        if not args.peer:
            parser.error("--peer is required in get mode")
        asyncio.run(run_get(args))


if __name__ == "__main__":
    main()
