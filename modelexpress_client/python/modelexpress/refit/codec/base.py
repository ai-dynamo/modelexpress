# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical checksums and bounded payload compression."""

from __future__ import annotations

import google_crc32c
import zstandard

NO_COMPRESSION = "none"
ZSTD_COMPRESSION = "zstd"


class CodecError(ValueError):
    """Encoded bytes do not satisfy the selected codec contract."""


def crc32c_hex(data: bytes) -> str:
    """Return the catalog's bare eight-character lowercase CRC32C."""
    return f"{google_crc32c.value(data):08x}"


def compress_payload(algorithm: str, data: bytes) -> bytes:
    if algorithm == NO_COMPRESSION:
        return data
    if algorithm == ZSTD_COMPRESSION:
        return zstandard.ZstdCompressor(
            level=3,
            threads=0,
            write_checksum=True,
            write_content_size=True,
        ).compress(data)
    raise CodecError(f"unsupported compression_algorithm {algorithm!r}")


def decompress_payload(algorithm: str, data: bytes, *, expected_size: int) -> bytes:
    if expected_size < 0:
        raise CodecError("expected decoded size must be non-negative")
    try:
        if algorithm == NO_COMPRESSION:
            decoded = data
        elif algorithm == ZSTD_COMPRESSION:
            parameters = zstandard.get_frame_parameters(data)
            if parameters.content_size != expected_size:
                raise CodecError(
                    f"decoded size declared by zstd frame does not match expected decoded size "
                    f"{expected_size}"
                )
            if not parameters.has_checksum or parameters.dict_id:
                raise CodecError(
                    "zstd payload does not use the canonical checksum/dictionary profile"
                )
            maximum_window = max(expected_size, 128 * 1024)
            if parameters.window_size > maximum_window:
                raise CodecError(
                    "zstd payload window exceeds the bounded canonical profile"
                )
            decoded = zstandard.ZstdDecompressor().decompress(
                data,
                max_output_size=expected_size,
                allow_extra_data=False,
            )
        else:
            raise CodecError(f"unsupported compression_algorithm {algorithm!r}")
    except zstandard.ZstdError as exc:
        raise CodecError(f"invalid zstd payload: {exc}") from exc
    if len(decoded) != expected_size:
        raise CodecError(
            f"decoded size {len(decoded)} does not match expected decoded size {expected_size}"
        )
    return decoded
