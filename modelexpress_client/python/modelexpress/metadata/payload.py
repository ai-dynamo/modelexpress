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


# Accelerator families whose weight bytes are transferable cross-vendor over
# NIXL. Weights are plain tensor bytes whose dtype and size are validated on the
# receive path before any transfer, so they carry across families. Generated
# artifacts (CUDA graphs, torch.compile, Triton, DeepGEMM, TileLang, CuTe,
# FlashInfer caches) are accelerator/arch-specific and must never cross.
HETEROGENEOUS_NIXL_WEIGHT_PAIRS = {
    frozenset(("cuda", "xpu")),
}

# Source types eligible for heterogeneous transfer. WEIGHTS only: LoRA has no
# live tensor publish/transfer path yet, so it stays strict same-family until
# one exists and is tested. Add MX_SOURCE_TYPE_LORA here once that lands.
HETEROGENEOUS_NIXL_SOURCE_TYPES = {
    p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
}


def accelerators_compatible(
    target: str,
    source: str,
    *,
    mx_source_type: int | None = None,
) -> bool:
    """Return whether ``target`` may pull from ``source`` given the source type.

    Empty values mean unknown and are accepted for backward compatibility with
    workers published before accelerator metadata existed. Exact family matches
    are always compatible. Cross-family transfer is allowed only for source
    types in ``HETEROGENEOUS_NIXL_SOURCE_TYPES`` and family pairs in
    ``HETEROGENEOUS_NIXL_WEIGHT_PAIRS``.

    ``mx_source_type`` defaults to ``None`` so the gate fails closed: a caller
    that does not explicitly pass a hetero-eligible source type only ever gets
    strict same-family compatibility. This is the single compatibility rule
    shared by RDMA tensor source selection and artifact source discovery, in
    both their pre-fetch and post-fetch checks.
    """
    if not target or not source:
        return True
    if target == source:
        return True
    if mx_source_type in HETEROGENEOUS_NIXL_SOURCE_TYPES:
        return frozenset((target, source)) in HETEROGENEOUS_NIXL_WEIGHT_PAIRS
    return False
