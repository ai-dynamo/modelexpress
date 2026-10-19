# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared safetensors header parser."""

import json
import struct

import pytest
import torch
from safetensors.torch import save as st_save

from modelexpress.safetensors_header import (
    HEADER_LENGTH_BYTES,
    header_length,
    parse_header,
    torch_dtype,
)


def _split(data: bytes) -> bytes:
    length = header_length(data, "test")
    return data[HEADER_LENGTH_BYTES : HEADER_LENGTH_BYTES + length]


def test_parses_real_file_and_skips_metadata():
    tensors = {
        "a": torch.zeros(3, 4),
        "b": torch.ones(5, dtype=torch.bfloat16),
        "c": torch.zeros(0, 2),
    }
    data = st_save(tensors, metadata={"format": "pt"})
    header = parse_header(_split(data), "test")
    assert header.data_start == HEADER_LENGTH_BYTES + header.header_len
    assert header.data_end == len(data)
    by_name = {entry.name: entry for entry in header.tensors}
    assert set(by_name) == {"a", "b", "c"}
    assert by_name["a"].shape == (3, 4) and by_name["a"].nbytes == 48
    assert by_name["b"].dtype == "BF16" and by_name["b"].nbytes == 10
    assert by_name["c"].nbytes == 0
    assert [e.begin for e in header.tensors] == sorted(e.begin for e in header.tensors)


def _raw(header: dict) -> bytes:
    return json.dumps(header).encode()


@pytest.mark.parametrize(
    ("header", "match"),
    [
        ({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}, "needs 8"),
        (
            {
                "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
            },
            "overlap",
        ),
        ({"w": {"dtype": "F32", "shape": [1], "data_offsets": [8, 4]}}, "Invalid data_offsets"),
        ({"w": {"dtype": "F4", "shape": [2], "data_offsets": [0, 1]}}, "Unsupported safetensors dtype 'F4' for tensor 'w'"),
        ({"w": {"shape": [2], "data_offsets": [0, 8]}}, "Malformed entry"),
    ],
)
def test_rejects_invalid_headers(header, match):
    with pytest.raises(ValueError, match=match):
        parse_header(_raw(header), "test")


def test_rejects_non_object_json():
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_header(b"[1, 2]", "test")


def test_rejects_bad_json():
    with pytest.raises(ValueError, match="Invalid safetensors header JSON"):
        parse_header(b"{not json", "test")


def test_header_length_bounds():
    with pytest.raises(ValueError, match="shorter than 8 bytes"):
        header_length(b"\x00" * 4, "test")
    with pytest.raises(ValueError, match="too large"):
        header_length(struct.pack("<Q", 1 << 40), "test")


@pytest.mark.parametrize(
    ("name", "attr"),
    [("U16", "uint16"), ("U32", "uint32"), ("U64", "uint64"), ("F8_E8M0", "float8_e8m0fnu")],
)
def test_optional_dtypes(name, attr):
    if not hasattr(torch, attr):
        pytest.skip(f"torch has no {attr}")
    assert torch_dtype(name, "t") is getattr(torch, attr)
