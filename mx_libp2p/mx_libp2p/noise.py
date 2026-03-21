# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Noise XX handshake with libp2p-specific framing.

Reference:
  - https://github.com/libp2p/specs/blob/master/noise/README.md
  - Noise Protocol Framework: https://noiseprotocol.org/noise.html

libp2p uses Noise XX with:
  - Cipher: ChaChaPoly (or AESGCM)
  - DH: X25519
  - Hash: SHA-256
  - Pattern: XX (mutual authentication)

The XX pattern:
  -> e                     (initiator sends ephemeral public key)
  <- e, ee, s, es          (responder sends ephemeral + static, DH)
  -> s, se                 (initiator sends static, DH)

libp2p-specific additions:
  1. Each Noise message is length-prefixed with a 2-byte big-endian length
  2. The handshake payload (sent in messages 2 and 3) is a protobuf:
     message NoiseHandshakePayload {
       bytes identity_key = 1;    // protobuf-encoded libp2p public key
       bytes identity_sig = 2;    // signature over "noise-libp2p-static-key:" + static DH pubkey
     }
  3. After handshake, the transport uses Noise frames:
     2-byte big-endian length + encrypted payload

This implementation uses the `cryptography` library for X25519 DH and ChaCha20-Poly1305.
"""

import asyncio
import hmac
import hashlib
import struct

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from .crypto import (
    Ed25519Identity,
    parse_public_key_proto,
    peer_id_from_ed25519_public_key,
    verify_ed25519_signature,
)
from .proto.noise_pb2 import NoiseHandshakePayload

# Noise protocol constants
NOISE_PROTOCOL_NAME = b"Noise_XX_25519_ChaChaPoly_SHA256"
EMPTY = b""
SIGNATURE_PREFIX = b"noise-libp2p-static-key:"

# Max Noise message size (65535 bytes)
MAX_NOISE_MSG_SIZE = 65535


class CipherState:
    """Noise CipherState - encrypts/decrypts with a key and nonce counter."""

    # Maximum nonce value per the Noise spec (2^64 - 1)
    _MAX_NONCE = (1 << 64) - 1

    def __init__(self, key: bytes | None = None):
        self.k = key
        self.n = 0

    def has_key(self) -> bool:
        return self.k is not None

    def _nonce_bytes(self) -> bytes:
        """Encode nonce as 12 bytes: 4 zero bytes + 8-byte little-endian counter."""
        if self.n > self._MAX_NONCE:
            raise ValueError("Noise nonce overflow: session must be terminated")
        return b"\x00\x00\x00\x00" + struct.pack("<Q", self.n)

    def encrypt_with_ad(self, ad: bytes, plaintext: bytes) -> bytes:
        if self.k is None:
            return plaintext
        aead = ChaCha20Poly1305(self.k)
        ct = aead.encrypt(self._nonce_bytes(), plaintext, ad)
        self.n += 1
        return ct

    def decrypt_with_ad(self, ad: bytes, ciphertext: bytes) -> bytes:
        if self.k is None:
            return ciphertext
        aead = ChaCha20Poly1305(self.k)
        pt = aead.decrypt(self._nonce_bytes(), ciphertext, ad)
        self.n += 1
        return pt


class SymmetricState:
    """Noise SymmetricState - manages chaining key and handshake hash."""

    def __init__(self):
        # Initialize with protocol name (or hash if > 32 bytes)
        if len(NOISE_PROTOCOL_NAME) <= 32:
            self.h = NOISE_PROTOCOL_NAME + b"\x00" * (32 - len(NOISE_PROTOCOL_NAME))
        else:
            self.h = hashlib.sha256(NOISE_PROTOCOL_NAME).digest()
        self.ck = self.h  # chaining key starts as h
        self.cipher = CipherState()

    def mix_hash(self, data: bytes) -> None:
        self.h = hashlib.sha256(self.h + data).digest()

    def mix_key(self, input_key_material: bytes) -> None:
        # HKDF with ck as salt, input_key_material as IKM
        temp_key, output = self._hkdf2(self.ck, input_key_material)
        self.ck = temp_key
        # Truncate output to 32 bytes for the cipher key
        self.cipher = CipherState(output[:32])

    def encrypt_and_hash(self, plaintext: bytes) -> bytes:
        ct = self.cipher.encrypt_with_ad(self.h, plaintext)
        self.mix_hash(ct)
        return ct

    def decrypt_and_hash(self, ciphertext: bytes) -> bytes:
        pt = self.cipher.decrypt_with_ad(self.h, ciphertext)
        self.mix_hash(ciphertext)
        return pt

    def split(self) -> tuple[CipherState, CipherState]:
        """Split into two CipherStates for transport."""
        temp_key1, temp_key2 = self._hkdf2(self.ck, EMPTY)
        return CipherState(temp_key1[:32]), CipherState(temp_key2[:32])

    @staticmethod
    def _hkdf2(salt: bytes, ikm: bytes) -> tuple[bytes, bytes]:
        """HKDF-SHA256 extract + expand to produce two 32-byte outputs."""
        # Extract
        prk = hmac.new(salt, ikm, hashlib.sha256).digest()
        # Expand: T1 = HMAC(prk, 0x01)
        t1 = hmac.new(prk, b"\x01", hashlib.sha256).digest()
        # Expand: T2 = HMAC(prk, T1 || 0x02)
        t2 = hmac.new(prk, t1 + b"\x02", hashlib.sha256).digest()
        return t1, t2


def _make_handshake_payload(identity: Ed25519Identity, static_dh_pubkey: bytes) -> bytes:
    """Create the libp2p Noise handshake payload protobuf."""
    sig_data = SIGNATURE_PREFIX + static_dh_pubkey
    signature = identity.sign(sig_data)

    msg = NoiseHandshakePayload()
    msg.identity_key = identity.public_key_proto
    msg.identity_sig = signature
    return msg.SerializeToString()


def _parse_handshake_payload(data: bytes) -> tuple[bytes, bytes]:
    """Parse a NoiseHandshakePayload protobuf. Returns (identity_key_proto, identity_sig)."""
    msg = NoiseHandshakePayload()
    msg.ParseFromString(data)
    if not msg.identity_key or not msg.identity_sig:
        raise ValueError("incomplete NoiseHandshakePayload")
    return msg.identity_key, msg.identity_sig


def _verify_handshake_payload(
    payload_bytes: bytes, remote_static_dh_pubkey: bytes
) -> tuple[int, bytes]:
    """Verify the handshake payload signature and extract the remote's identity.

    Returns (key_type, public_key_bytes) of the remote peer.
    Raises ValueError if signature verification fails.
    """
    identity_key_proto, identity_sig = _parse_handshake_payload(payload_bytes)

    # Parse the libp2p public key
    key_type, pub_key_bytes = parse_public_key_proto(identity_key_proto)

    # Verify signature: sign("noise-libp2p-static-key:" + remote static DH key)
    sig_data = SIGNATURE_PREFIX + remote_static_dh_pubkey
    if not verify_ed25519_signature(pub_key_bytes, sig_data, identity_sig):
        raise ValueError("handshake payload signature verification failed")

    return key_type, pub_key_bytes


async def _read_noise_msg(reader: asyncio.StreamReader) -> bytes:
    """Read a length-prefixed Noise message (2-byte big-endian length + data)."""
    length_bytes = await reader.readexactly(2)
    length = struct.unpack(">H", length_bytes)[0]
    if length > MAX_NOISE_MSG_SIZE:
        raise ValueError(f"Noise message too large: {length}")
    return await reader.readexactly(length)


def _write_noise_msg(writer: asyncio.StreamWriter, data: bytes) -> None:
    """Write a length-prefixed Noise message."""
    writer.write(struct.pack(">H", len(data)) + data)


class NoiseTransport:
    """Encrypted transport after Noise handshake completion.

    Wraps an asyncio reader/writer with Noise encryption.
    Each frame is: 2-byte big-endian length + encrypted payload.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        send_cipher: CipherState,
        recv_cipher: CipherState,
        remote_peer_id: bytes,
    ):
        self.reader = reader
        self.writer = writer
        self.send_cipher = send_cipher
        self.recv_cipher = recv_cipher
        self.remote_peer_id = remote_peer_id
        self._read_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    async def read_msg(self) -> bytes:
        """Read and decrypt a Noise transport message."""
        async with self._read_lock:
            data = await _read_noise_msg(self.reader)
            return self.recv_cipher.decrypt_with_ad(EMPTY, data)

    async def write_msg(self, plaintext: bytes) -> None:
        """Encrypt and write a Noise transport message."""
        async with self._write_lock:
            ct = self.send_cipher.encrypt_with_ad(EMPTY, plaintext)
            _write_noise_msg(self.writer, ct)
            await self.writer.drain()

    def close(self) -> None:
        try:
            if self.writer.transport:
                self.writer.transport.abort()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        try:
            self.writer.close()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


async def handshake_initiator(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    identity: Ed25519Identity,
) -> NoiseTransport:
    """Perform the Noise XX handshake as the initiator (dialer).

    Returns a NoiseTransport for encrypted communication.
    """
    # Generate ephemeral X25519 key pair
    e_priv = X25519PrivateKey.generate()
    e_pub = e_priv.public_key().public_bytes_raw()

    # Generate static X25519 key pair (used for DH, bound to identity via signature)
    s_priv = X25519PrivateKey.generate()
    s_pub = s_priv.public_key().public_bytes_raw()

    state = SymmetricState()
    # Mix in the prologue (empty for libp2p)
    state.mix_hash(EMPTY)

    # -> e (Message 1: initiator sends ephemeral key)
    state.mix_hash(e_pub)
    # XX pattern msg1: just e + encrypted empty payload (no identity payload yet)
    msg1 = e_pub + state.encrypt_and_hash(EMPTY)
    _write_noise_msg(writer, msg1)
    await writer.drain()

    # <- e, ee, s, es (Message 2: responder's turn)
    msg2 = await _read_noise_msg(reader)
    offset = 0

    # Read responder's ephemeral key (32 bytes)
    re_pub_bytes = msg2[offset : offset + 32]
    offset += 32
    state.mix_hash(re_pub_bytes)
    re_pub = X25519PublicKey.from_public_bytes(re_pub_bytes)

    # ee: DH(e, re)
    ee_shared = e_priv.exchange(re_pub)
    state.mix_key(ee_shared)

    # s: responder's static key (encrypted)
    # Encrypted static key = 32 bytes key + 16 bytes poly1305 tag
    rs_encrypted = msg2[offset : offset + 48]
    offset += 48
    rs_pub_bytes = state.decrypt_and_hash(rs_encrypted)

    # es: DH(e, rs)
    rs_pub = X25519PublicKey.from_public_bytes(rs_pub_bytes)
    es_shared = e_priv.exchange(rs_pub)
    state.mix_key(es_shared)

    # Decrypt the handshake payload
    payload_encrypted = msg2[offset:]
    remote_payload_bytes = state.decrypt_and_hash(payload_encrypted)

    # Verify the remote's identity and signature
    _key_type, remote_pub_key_bytes = _verify_handshake_payload(
        remote_payload_bytes, rs_pub_bytes
    )

    # -> s, se (Message 3: initiator sends static key + payload)
    # Encrypt our static key
    s_encrypted = state.encrypt_and_hash(s_pub)

    # se: DH(s, re)
    se_shared = s_priv.exchange(re_pub)
    state.mix_key(se_shared)

    # Create and encrypt our handshake payload
    payload3 = _make_handshake_payload(identity, s_pub)
    payload3_encrypted = state.encrypt_and_hash(payload3)

    msg3 = s_encrypted + payload3_encrypted
    _write_noise_msg(writer, msg3)
    await writer.drain()

    # Split into transport cipher states
    c1, c2 = state.split()

    # Derive remote peer ID from their public key
    remote_peer_id = peer_id_from_ed25519_public_key(remote_pub_key_bytes)

    # Initiator sends with c1, receives with c2
    return NoiseTransport(reader, writer, c1, c2, remote_peer_id)


async def handshake_responder(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    identity: Ed25519Identity,
) -> NoiseTransport:
    """Perform the Noise XX handshake as the responder (listener).

    Returns a NoiseTransport for encrypted communication.
    """
    # Generate ephemeral and static X25519 key pairs
    e_priv = X25519PrivateKey.generate()
    e_pub = e_priv.public_key().public_bytes_raw()

    s_priv = X25519PrivateKey.generate()
    s_pub = s_priv.public_key().public_bytes_raw()

    state = SymmetricState()
    state.mix_hash(EMPTY)

    # <- e (Message 1: read initiator's ephemeral key)
    msg1 = await _read_noise_msg(reader)
    re_pub_bytes = msg1[:32]
    state.mix_hash(re_pub_bytes)
    re_pub = X25519PublicKey.from_public_bytes(re_pub_bytes)

    # Always decrypt payload (Noise framework always calls DecryptAndHash,
    # even on empty payload - the mix_hash side-effect must happen)
    state.decrypt_and_hash(msg1[32:])

    # -> e, ee, s, es (Message 2)
    state.mix_hash(e_pub)

    # ee: DH(e, re)
    ee_shared = e_priv.exchange(re_pub)
    state.mix_key(ee_shared)

    # s: encrypt our static key
    s_encrypted = state.encrypt_and_hash(s_pub)

    # es: DH(s_responder, e_initiator) - named from the Noise pattern perspective
    # where token names are always (initiator_key, responder_key)
    es_shared = s_priv.exchange(re_pub)
    state.mix_key(es_shared)

    # Create and encrypt our handshake payload
    payload2 = _make_handshake_payload(identity, s_pub)
    payload2_encrypted = state.encrypt_and_hash(payload2)

    msg2 = e_pub + s_encrypted + payload2_encrypted
    _write_noise_msg(writer, msg2)
    await writer.drain()

    # <- s, se (Message 3: read initiator's static key + payload)
    msg3 = await _read_noise_msg(reader)
    offset = 0

    # Decrypt initiator's static key (32 + 16 = 48 bytes)
    rs_encrypted = msg3[offset : offset + 48]
    offset += 48
    rs_pub_bytes = state.decrypt_and_hash(rs_encrypted)

    # se: DH(e_responder, s_initiator) - responder computes the se token
    rs_pub = X25519PublicKey.from_public_bytes(rs_pub_bytes)
    se_shared = e_priv.exchange(rs_pub)
    state.mix_key(se_shared)

    # Decrypt the handshake payload
    payload3_encrypted = msg3[offset:]
    remote_payload_bytes = state.decrypt_and_hash(payload3_encrypted)

    # Verify remote identity
    _key_type, remote_pub_key_bytes = _verify_handshake_payload(
        remote_payload_bytes, rs_pub_bytes
    )

    # Split into transport cipher states
    c1, c2 = state.split()

    # Derive remote peer ID from their public key
    remote_peer_id = peer_id_from_ed25519_public_key(remote_pub_key_bytes)

    # Responder sends with c2, receives with c1
    return NoiseTransport(reader, writer, c2, c1, remote_peer_id)
