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
    """Metadata for a single GPU worker.

    NIXL agent blobs are NOT stored here. Workers exchange them peer-to-peer
    via NIXL's native listen thread. Tensor manifests are served directly by
    the source worker's gRPC server at worker_grpc_endpoint.
    """
    worker_rank: int
    metadata_endpoint: str = ""
    agent_name: str = ""
    worker_grpc_endpoint: str = ""
    transfer_engine_session_id: str = ""


@dataclass
class GetMetadataResponse:
    """Response from GetMetadata RPC."""
    found: bool
    workers: list[WorkerMetadata]
