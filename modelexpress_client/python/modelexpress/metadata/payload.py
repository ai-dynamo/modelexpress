# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for WorkerMetadata source_payload migration."""

from __future__ import annotations

from collections.abc import Sequence

from .. import p2p_pb2


def tensor_source_metadata(
    tensors: Sequence[p2p_pb2.TensorDescriptor],
) -> p2p_pb2.TensorSourceMetadata:
    return p2p_pb2.TensorSourceMetadata(tensors=list(tensors))


def worker_tensor_descriptors(worker: p2p_pb2.WorkerMetadata):
    payload = worker.WhichOneof("source_payload")
    if payload == "tensor_source":
        return worker.tensor_source.tensors
    if payload == "artifact_source":
        return []
    return worker.tensors


def worker_tensor_count(worker: p2p_pb2.WorkerMetadata) -> int:
    return len(worker_tensor_descriptors(worker))


def accelerators_compatible(target: str, source: str) -> bool:
    """Return whether ``target`` and ``source`` accelerator families match.

    Empty values mean unknown and are accepted for backward compatibility with
    workers published before accelerator metadata existed. This is the single
    compatibility rule shared by RDMA tensor source selection and artifact
    source discovery, in both their pre-fetch and post-fetch checks.
    """
    if not target or not source:
        return True
    return target == source
