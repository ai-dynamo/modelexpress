# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core type definitions for ModelExpress P2P metadata service."""

from dataclasses import dataclass


@dataclass
class TensorDescriptor:
    """Descriptor for a tensor in GPU memory."""
    name: str
    addr: int
    size: int
    device_id: int
    dtype: str


@dataclass
class WorkerMetadata:
    """Metadata for a single GPU worker."""
    worker_rank: int
    nixl_metadata: bytes
    tensors: list[TensorDescriptor]


@dataclass
class GetMetadataResponse:
    """Response from GetMetadata RPC."""
    found: bool
    workers: list[WorkerMetadata]
