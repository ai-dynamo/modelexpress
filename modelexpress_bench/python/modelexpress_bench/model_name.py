# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoding for the synthetic `model_name` smuggle field.

Mirrors the Rust ``modelexpress_bench::model_name::BenchSpec``. Keeps the
encoding symmetric so a Python bench server and the Rust bench client (and
vice versa) interpret ``bench:<bytes>:<files>`` identically.
"""

from __future__ import annotations

from dataclasses import dataclass

BENCH_PREFIX = "bench:"


@dataclass(frozen=True)
class BenchSpec:
    bytes_per_file: int
    file_count: int

    @property
    def total_bytes(self) -> int:
        return self.bytes_per_file * self.file_count

    def encode(self) -> str:
        return f"{BENCH_PREFIX}{self.bytes_per_file}:{self.file_count}"

    @classmethod
    def parse(cls, name: str) -> "BenchSpec":
        if not name.startswith(BENCH_PREFIX):
            raise ValueError(f"model_name does not start with {BENCH_PREFIX!r}: {name!r}")
        rest = name[len(BENCH_PREFIX):]
        parts = rest.split(":")
        if len(parts) == 1:
            bytes_per_file = int(parts[0])
            file_count = 1
        elif len(parts) == 2:
            bytes_per_file, file_count = int(parts[0]), int(parts[1])
        else:
            raise ValueError(f"too many components in {name!r}")
        if file_count < 1:
            raise ValueError("file count must be >= 1")
        if bytes_per_file < 0:
            raise ValueError("bytes_per_file must be >= 0")
        return cls(bytes_per_file=bytes_per_file, file_count=file_count)
