# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact tensor-byte XOR delta codec."""

from __future__ import annotations

import numpy as np

from .base import CodecError

TENSOR_BYTE_XOR = "mx.tensor_bytes.xor.v1"


def _xor(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise CodecError(
            "tensor-byte XOR inputs must have the same byte length: "
            f"{len(left)} != {len(right)}"
        )
    left_array = np.frombuffer(left, dtype=np.uint8)
    right_array = np.frombuffer(right, dtype=np.uint8)
    return np.bitwise_xor(left_array, right_array).tobytes()


def encode_delta(delta_method: str, base: bytes, target: bytes) -> bytes:
    if delta_method != TENSOR_BYTE_XOR:
        raise CodecError(f"unsupported delta_method {delta_method!r}")
    return _xor(base, target)


def decode_delta(delta_method: str, base: bytes, delta: bytes) -> bytes:
    if delta_method != TENSOR_BYTE_XOR:
        raise CodecError(f"unsupported delta_method {delta_method!r}")
    return _xor(base, delta)
