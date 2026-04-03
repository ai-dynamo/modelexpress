# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identify protocol (/ipfs/id/1.0.0).

Reference: https://github.com/libp2p/specs/blob/master/identify/README.md

After a connection is established, libp2p nodes exchange Identify messages to
learn about each other's capabilities, addresses, and public keys.
"""

from .crypto import Ed25519Identity
from .proto.identify_pb2 import Identify as IdentifyProto


def encode_identify_msg(
    identity: Ed25519Identity,
    listen_addrs: list[bytes],
    observed_addr: bytes,
    protocols: list[str],
    protocol_version: str = "ipfs/0.1.0",
    agent_version: str = "mx-libp2p/0.1.0",
) -> bytes:
    """Encode an Identify protobuf message."""
    msg = IdentifyProto()
    msg.publicKey = identity.public_key_proto
    for addr in listen_addrs:
        msg.listenAddrs.append(addr)
    for proto in protocols:
        msg.protocols.append(proto)
    msg.observedAddr = observed_addr
    msg.protocolVersion = protocol_version
    msg.agentVersion = agent_version
    return msg.SerializeToString()


def decode_identify_msg(data: bytes) -> dict:
    """Decode an Identify protobuf message.

    Returns a dict with keys: public_key, listen_addrs, observed_addr,
    protocols, protocol_version, agent_version.
    """
    msg = IdentifyProto()
    msg.ParseFromString(data)
    return {
        "public_key": msg.publicKey or None,
        "listen_addrs": list(msg.listenAddrs),
        "protocols": list(msg.protocols),
        "observed_addr": msg.observedAddr or None,
        "protocol_version": msg.protocolVersion or None,
        "agent_version": msg.agentVersion or None,
    }
