# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cryptographic identity: Ed25519 key pairs, peer IDs, and protobuf encoding.

Peer IDs in libp2p are derived from the public key:
  - Serialize the public key as a protobuf (crypto.proto)
  - If the serialized key is <= 42 bytes, the peer ID is the multihash identity(key_bytes)
  - Otherwise, the peer ID is sha256(key_bytes)

Ed25519 public keys are 32 bytes, so serialized they're ~36 bytes -> identity multihash.

Reference: https://github.com/libp2p/specs/blob/master/peer-ids/peer-ids.md
"""

import hashlib

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)

from .proto.crypto_pb2 import PublicKey as PublicKeyProto

# Ed25519 key type constant (matches crypto.proto KeyType enum)
_KEY_TYPE_ED25519 = 1


def _encode_uvarint(value: int) -> bytes:
    """Encode an unsigned integer as a varint (LEB128).

    Used for multistream-select and multiaddr framing.
    """
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def _serialize_pubkey_proto(key_type: int, key_data: bytes) -> bytes:
    """Serialize a public key as a protobuf PublicKey message."""
    msg = PublicKeyProto()
    msg.Type = key_type
    msg.Data = key_data
    return msg.SerializeToString()


def parse_public_key_proto(data: bytes) -> tuple[int, bytes]:
    """Parse a protobuf PublicKey message. Returns (key_type, key_data)."""
    msg = PublicKeyProto()
    msg.ParseFromString(data)
    if not msg.Data:
        raise ValueError("incomplete PublicKey protobuf")
    return msg.Type, msg.Data


def _peer_id_from_pubkey_proto(pubkey_proto: bytes) -> bytes:
    """Derive a libp2p peer ID from a serialized PublicKey protobuf.

    For keys <= 42 bytes serialized: identity multihash (0x00, length, data)
    For larger keys: sha256 multihash (0x12, 0x20, sha256(data))
    """
    if len(pubkey_proto) <= 42:
        return b"\x00" + _encode_uvarint(len(pubkey_proto)) + pubkey_proto
    else:
        return b"\x12\x20" + hashlib.sha256(pubkey_proto).digest()


class Ed25519Identity:
    """An Ed25519 key pair used as a libp2p node identity."""

    def __init__(self, private_key: Ed25519PrivateKey):
        self._private_key = private_key
        self._public_key = private_key.public_key()
        self._public_key_bytes = self._public_key.public_bytes(
            Encoding.Raw, PublicFormat.Raw
        )
        # Cache the protobuf-encoded public key and peer ID
        self._pubkey_proto = _serialize_pubkey_proto(_KEY_TYPE_ED25519, self._public_key_bytes)
        self._peer_id = _peer_id_from_pubkey_proto(self._pubkey_proto)

    @classmethod
    def generate(cls) -> "Ed25519Identity":
        """Generate a new random Ed25519 identity."""
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_seed(cls, seed: bytes) -> "Ed25519Identity":
        """Create an identity from a 32-byte seed (deterministic, for testing)."""
        return cls(Ed25519PrivateKey.from_private_bytes(seed))

    @property
    def private_key(self) -> Ed25519PrivateKey:
        return self._private_key

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._public_key

    @property
    def public_key_bytes(self) -> bytes:
        """Raw 32-byte Ed25519 public key."""
        return self._public_key_bytes

    @property
    def public_key_proto(self) -> bytes:
        """Protobuf-encoded public key (crypto.proto PublicKey message)."""
        return self._pubkey_proto

    @property
    def peer_id(self) -> bytes:
        """Raw peer ID bytes (multihash)."""
        return self._peer_id

    @property
    def peer_id_b58(self) -> str:
        """Base58btc-encoded peer ID string."""
        return _base58btc_encode(self._peer_id)

    def sign(self, data: bytes) -> bytes:
        """Sign data with the Ed25519 private key."""
        return self._private_key.sign(data)


def peer_id_from_ed25519_public_key(pub_key_bytes: bytes) -> bytes:
    """Derive a libp2p peer ID from raw Ed25519 public key bytes."""
    pubkey_proto = _serialize_pubkey_proto(_KEY_TYPE_ED25519, pub_key_bytes)
    return _peer_id_from_pubkey_proto(pubkey_proto)


def verify_ed25519_signature(
    public_key_bytes: bytes, data: bytes, signature: bytes
) -> bool:
    """Verify an Ed25519 signature."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature

    pub = Ed25519PublicKey.from_public_bytes(public_key_bytes)
    try:
        pub.verify(signature, data)
        return True
    except InvalidSignature:
        return False


# Base58btc encoding (used for peer ID display)
_B58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _base58btc_encode(data: bytes) -> str:
    """Encode bytes as base58btc (Bitcoin-style)."""
    n = int.from_bytes(data, "big")
    result = bytearray()
    while n > 0:
        n, r = divmod(n, 58)
        result.append(_B58_ALPHABET[r])
    # Handle leading zeros
    for byte in data:
        if byte == 0:
            result.append(_B58_ALPHABET[0])
        else:
            break
    return bytes(reversed(result)).decode("ascii")


def _base58btc_decode(s: str) -> bytes:
    """Decode a base58btc string to bytes."""
    n = 0
    for char in s.encode("ascii"):
        idx = _B58_ALPHABET.find(char)
        if idx < 0:
            raise ValueError(f"invalid base58 character: {chr(char)!r}")
        n = n * 58 + idx
    result = []
    while n > 0:
        n, r = divmod(n, 256)
        result.append(r)
    # Handle leading '1' chars (zero bytes)
    for char in s:
        if char == "1":
            result.append(0)
        else:
            break
    return bytes(reversed(result))
