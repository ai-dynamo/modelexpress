# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python side of the libp2p Kademlia DHT interop test.

Modes:
  --mode put  : Start a node, put a record, print the multiaddr, wait.
  --mode get  : Connect to a peer, get the record, verify, exit.

Uses trio (not asyncio) since py-libp2p is trio-native.
"""

import argparse
import logging
import secrets
import sys

import trio
from multiaddr import Multiaddr

from libp2p import new_host
from libp2p.crypto.secp256k1 import create_new_key_pair
from libp2p.kad_dht.kad_dht import DHTMode, KadDHT
from libp2p.records.validator import Validator
from libp2p.tools.async_service import background_trio_service
from libp2p.tools.utils import info_from_p2p_addr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class MxValidator(Validator):
    """Accept any non-empty value."""

    def validate(self, key: str, value: bytes) -> None:
        if not value:
            raise ValueError("Value cannot be empty")

    def select(self, key: str, values: list[bytes]) -> int:
        return 0


async def run_put(args):
    """Start a node, store a record, and wait for peers to query it."""
    key_pair = create_new_key_pair(secrets.token_bytes(32))
    host = new_host(key_pair=key_pair)

    async with host.run(listen_addrs=[Multiaddr(f"/ip4/0.0.0.0/tcp/0")]):
        async with trio.open_nursery() as nursery:
            nursery.start_soon(host.get_peerstore().start_cleanup_task, 60)

            peer_id = host.get_id()
            addrs = host.get_addrs()
            full_addr = f"{addrs[0]}/p2p/{peer_id.to_string()}"
            log.info("Listening on: %s", full_addr)
            print(f"LISTEN_ADDR={full_addr}", flush=True)

            dht = KadDHT(host, DHTMode.SERVER)
            dht.register_validator("mx", MxValidator())

            async with background_trio_service(dht):
                log.info("DHT service started in SERVER mode")

                # Store the value using a namespaced key
                key = args.key
                value = args.value.encode()
                await dht.put_value(key, value)
                log.info("Stored record: key=%s, value=%s", key, args.value)

                # Wait for peers to query
                await trio.sleep(args.timeout_secs)
                log.info("Put node shutting down.")
                nursery.cancel_scope.cancel()


async def run_get(args):
    """Connect to a peer, retrieve a record, and verify it."""
    key_pair = create_new_key_pair(secrets.token_bytes(32))
    host = new_host(key_pair=key_pair)

    async with host.run(listen_addrs=[Multiaddr(f"/ip4/0.0.0.0/tcp/0")]):
        async with trio.open_nursery() as nursery:
            nursery.start_soon(host.get_peerstore().start_cleanup_task, 60)

            # Connect to the peer
            peer_addr = Multiaddr(args.peer)
            peer_info = info_from_p2p_addr(peer_addr)
            host.get_peerstore().add_addrs(peer_info.peer_id, peer_info.addrs, 3600)
            await host.connect(peer_info)
            log.info("Connected to: %s", peer_info.peer_id)

            dht = KadDHT(host, DHTMode.CLIENT)
            dht.register_validator("mx", MxValidator())

            # Add connected peers to the routing table
            for pid in host.get_peerstore().peer_ids():
                await dht.routing_table.add_peer(pid)

            async with background_trio_service(dht):
                log.info("DHT service started in CLIENT mode")

                # Small delay to let routing settle
                await trio.sleep(1)

                # Retrieve the value
                key = args.key
                log.info("Looking up key: %s", key)

                try:
                    with trio.fail_after(args.timeout_secs):
                        value = await dht.get_value(key)
                    if value:
                        decoded = value.decode() if isinstance(value, bytes) else str(value)
                        log.info("Got record: key=%s, value=%s", key, decoded)
                        print(f"RECORD_VALUE={decoded}", flush=True)
                        print("RESULT=OK", flush=True)
                    else:
                        log.error("get_value returned None")
                        print("RESULT=FAIL:None", flush=True)
                        sys.exit(1)
                except trio.TooSlow:
                    log.error("Timed out waiting for record")
                    print("RESULT=TIMEOUT", flush=True)
                    sys.exit(1)
                except Exception as e:
                    log.error("Failed to get record: %s", e)
                    print(f"RESULT=FAIL:{e}", flush=True)
                    sys.exit(1)

            nursery.cancel_scope.cancel()


def main():
    parser = argparse.ArgumentParser(description="py-libp2p Kademlia interop test")
    parser.add_argument("--mode", required=True, choices=["put", "get"])
    parser.add_argument("--key", default="/mx/model:test-model:worker:0")
    parser.add_argument(
        "--value",
        default='{"rank":0,"tensors":[{"name":"layer.0.weight","size":1024}]}',
    )
    parser.add_argument("--peer", help="Peer multiaddr (required for get mode)")
    parser.add_argument("--timeout-secs", type=int, default=30)
    args = parser.parse_args()

    if args.mode == "put":
        trio.run(run_put, args)
    elif args.mode == "get":
        if not args.peer:
            parser.error("--peer is required in get mode")
        trio.run(run_get, args)


if __name__ == "__main__":
    main()
