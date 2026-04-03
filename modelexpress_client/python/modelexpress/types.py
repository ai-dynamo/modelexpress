# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core type definitions for ModelExpress P2P metadata service."""

from dataclasses import dataclass


class ManifestMismatchError(Exception):
    """Raised when source and target tensor manifests are incompatible.

    This is NOT a source-side failure - both sides are healthy, but their
    runtime environments produce different tensor structures. The transfer
    would silently produce incorrect inference results, so we abort and
    fall back to disk loading.
    """


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
    tensors: list[TensorDescriptor]
    nixl_metadata: bytes = b""
    transfer_engine_session_id: str = ""
    # P2P metadata exchange fields (opt-in via MX_P2P_METADATA=1)
    metadata_endpoint: str = ""
    agent_name: str = ""
    worker_grpc_endpoint: str = ""


@dataclass
class GetMetadataResponse:
    """Response from GetMetadata RPC."""
    found: bool
    workers: list[WorkerMetadata]
