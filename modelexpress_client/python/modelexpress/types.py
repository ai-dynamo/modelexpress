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
    via NIXL's native listen thread. The metadata_endpoint field tells targets
    where to connect, and agent_name identifies the remote NIXL agent.
    """
    worker_rank: int
    tensors: list[TensorDescriptor]
    metadata_endpoint: str = ""
    agent_name: str = ""
    transfer_engine_session_id: str = ""


@dataclass
class GetMetadataResponse:
    """Response from GetMetadata RPC."""
    found: bool
    workers: list[WorkerMetadata]
