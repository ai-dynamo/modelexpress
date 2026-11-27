# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integrity of the GPU byte ranges exposed by a static weight publication."""

from __future__ import annotations

import logging
import struct
import time
import uuid
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING

from .. import envs, p2p_pb2
from ..adapter import StrategyFailed, StrategyRecoveryError

if TYPE_CHECKING:
    import torch

    from ..load_strategy.context import LoadContext
    from ..types import TensorDescriptor

logger = logging.getLogger(__name__)
FORMAT_VERSION = 1
MAIN_SCOPE = "main"


def integrity_enabled(ctx: LoadContext) -> bool:
    return (
        envs.MX_WEIGHT_INTEGRITY
        and getattr(ctx.mx_client, "REQUIRES_P2P_METADATA", False) is not True
        and not ctx.is_reload
        and ctx.desired_version_uid is None
        and ctx.accelerator_backend.name == "cuda"
        and envs.MX_LOAD_STRATEGY_CHAIN == "INFERENCE"
        and ctx.identity.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_WEIGHTS
        and ctx.identity.backend_framework == p2p_pb2.BACKEND_FRAMEWORK_SGLANG
    )


@contextmanager
def agree_load_phase(
    ctx: LoadContext, enabled: bool, phase: str, *, recovering: bool = False
):
    if not enabled:
        yield
        return

    error, status = None, None
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - every rank must report its phase outcome
        error = exc
        status = (
            str(exc),
            recovering or isinstance(exc, StrategyRecoveryError),
            not isinstance(exc, StrategyFailed) or exc.mutated,
        )
    # Use the same phase markers as the outer load strategy chain.
    from ..load_strategy import _gather_all_ranks

    states = _gather_all_ranks(phase, status, ctx)
    failures = [state for state in states if state is not None]
    if failures:
        message = f"{phase} failed in the load group: {failures[0][0]}"
        if any(fatal for _, fatal, _ in failures):
            raise StrategyRecoveryError(message) from error
        dirty = any(mutated for _, _, mutated in failures) or len(failures) != len(
            states
        )
        raise StrategyFailed(message, mutated=dirty) from error


def weight_scope(ctx: LoadContext) -> str:
    return ctx.draft_tensor_namespace if ctx.p2p_role == "draft" else MAIN_SCOPE


def manifest_digest(
    manifest: p2p_pb2.GetTensorManifestResponse,
    ctx: LoadContext,
    source_id: str,
    worker_id: str,
) -> p2p_pb2.WeightDigest | None:
    scope = weight_scope(ctx)
    matches = [entry for entry in manifest.weight_digests if entry.scope == scope]
    if not matches:
        if any(not entry.scope for entry in manifest.weight_digests):
            raise ValueError("weight digest has an empty scope")
        return None
    if len(matches) != 1:
        raise ValueError(f"duplicate weight digests for {scope!r}")
    if (
        manifest.mx_source_id != source_id
        or not worker_id
        or manifest.worker_id != worker_id
        or manifest.worker_rank != ctx.worker_rank
        or not manifest.metadata_endpoint
        or not manifest.agent_name
    ):
        raise ValueError("worker identity or endpoint differs from the weight manifest")
    digest = matches[0]
    if (
        digest.format_version != FORMAT_VERSION
        or len(digest.digest) != 32
        or not digest.generation
    ):
        raise ValueError(f"invalid or unsupported weight digest for {scope!r}")
    return digest


def registered_tensors(ctx: LoadContext) -> dict[str, torch.Tensor]:
    """Check the actual NIXL addresses and return this scope's nonempty tensors."""
    registered = {
        item.name: item
        for item in ctx.nixl_manager.tensor_descriptors
        if item.name in ctx.tensors
    }
    tensors = {name: tensor for name, tensor in ctx.tensors.items() if tensor.numel()}
    if not tensors:
        raise ValueError("weight scope has no transferable tensor bytes")
    for name, tensor in tensors.items():
        item = registered.get(name)
        size = tensor.numel() * tensor.element_size()
        address = tensor.data_ptr()
        storage = tensor.untyped_storage()
        offset = address - storage.data_ptr()
        if (
            item is None
            or not address
            or item.addr != address
            or item.size != size
            or item.dtype != str(tensor.dtype)
            or not tensor.is_contiguous()
            or offset < 0
            or offset + size > storage.nbytes()
        ):
            raise ValueError(f"tensor {name!r} no longer matches its NIXL registration")
    return tensors


def validate_transfer(
    source: Iterable[TensorDescriptor], target: Mapping[str, torch.Tensor]
) -> list[TensorDescriptor]:
    catalog = {}
    seen = set()
    for entry in source:
        if not entry.name or entry.name in seen:
            raise ValueError(
                f"invalid or duplicate name {entry.name!r} in source manifest"
            )
        seen.add(entry.name)
        if entry.size:
            catalog[entry.name] = entry
    if catalog.keys() != target.keys():
        missing = sorted(target.keys() - catalog.keys())[:3]
        extra = sorted(catalog.keys() - target.keys())[:3]
        raise ValueError(
            f"weight manifest range mismatch (missing={missing}, extra={extra})"
        )
    for name, tensor in target.items():
        entry = catalog[name]
        if (
            not entry.addr
            or entry.mem_type != p2p_pb2.TENSOR_MEMORY_TYPE_VRAM
            or entry.size != tensor.numel() * tensor.element_size()
            or entry.dtype != str(tensor.dtype)
        ):
            raise ValueError(f"incompatible source tensor {name!r}")
    return list(catalog.values())


def _encode_text(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack(">I", len(data)) + data


def _leaf_prefix(rank: int, scope: str, name: str, size: int) -> bytes:
    prefix = (
        struct.pack(">III", 0, FORMAT_VERSION, rank)
        + _encode_text(scope)
        + _encode_text(name)
        + struct.pack(">Q", size)
    )
    return prefix + bytes(-len(prefix) % 64)


def content_digest(tensors: Mapping[str, torch.Tensor], rank: int, scope: str) -> bytes:
    """Hash live weights on the GPU and return only the 32-byte digest."""
    import torch

    from ._weight_sha256 import BLOCK_BYTES, hash_leaves, hash_parents, round_constants

    if not tensors:
        raise ValueError("weight scope has no transferable tensor bytes")
    device = next(iter(tensors.values())).device
    if device.type != "cuda" or any(
        tensor.device != device for tensor in tensors.values()
    ):
        raise ValueError("weight digests require one CUDA device per scope")
    inputs = []
    prefix_words = []
    leaf_count = 0
    for name in sorted(tensors, key=lambda name: name.encode("utf-8")):
        raw = tensors[name].detach().view(-1).view(torch.uint8)
        size = raw.numel()
        prefix = _leaf_prefix(rank, scope, name, size)
        start = len(prefix_words)
        prefix_words.extend(struct.unpack(f">{len(prefix) // 4}i", prefix))
        inputs.append((raw, start, len(prefix_words), leaf_count))
        leaf_count += (size + BLOCK_BYTES - 1) // BLOCK_BYTES

    with torch.cuda.device(device):
        constants = round_constants(device)
        prefixes = torch.tensor(prefix_words, dtype=torch.int32, device=device)
        nodes = torch.empty((leaf_count, 8), dtype=torch.int32, device=device)
        for raw, start, end, leaf in inputs:
            hash_leaves(raw, prefixes[start:end], nodes, constants, leaf)
        while nodes.shape[0] > 1:
            parents = torch.empty(
                ((nodes.shape[0] + 1) // 2, 8), dtype=torch.int32, device=device
            )
            hash_parents(nodes, parents, constants)
            nodes = parents
        return struct.pack(">8i", *nodes[0].cpu().tolist())


def source_digest(ctx: LoadContext) -> p2p_pb2.WeightDigest:
    scope = weight_scope(ctx)
    start = time.perf_counter()
    tensors = registered_tensors(ctx)
    ctx.accelerator_backend.set_device(ctx.device_id)
    ctx.accelerator_backend.synchronize()
    verified = ctx.verified_weight_digest
    ctx.verified_weight_digest = None
    reused = verified is not None and verified.scope == scope
    digest = (
        verified.digest if reused else content_digest(tensors, ctx.worker_rank, scope)
    )
    summary = p2p_pb2.WeightDigest(
        scope=scope,
        format_version=FORMAT_VERSION,
        digest=digest,
        generation=uuid.uuid4().hex,
    )
    logger.info(
        "[TIMING] MX weight digest source: rank=%s scope=%s generation=%s compute_ms=%.3f reused=%s",
        ctx.worker_rank,
        scope,
        summary.generation,
        (time.perf_counter() - start) * 1000,
        reused,
    )
    return summary


def verify_digest(
    tensors: Mapping[str, torch.Tensor], rank: int, expected: p2p_pb2.WeightDigest
) -> None:
    start = time.perf_counter()
    digest = content_digest(tensors, rank, expected.scope)
    computed = time.perf_counter()
    matches = digest == expected.digest
    logger.info(
        "[TIMING] MX weight digest target: rank=%s scope=%s generation=%s compute_ms=%.3f compare_ms=%.3f result=%s",
        rank,
        expected.scope,
        expected.generation,
        (computed - start) * 1000,
        (time.perf_counter() - computed) * 1000,
        "match" if matches else "mismatch",
    )
    if not matches:
        raise ValueError(f"RDMA weight digest mismatch for {expected.scope!r}")
