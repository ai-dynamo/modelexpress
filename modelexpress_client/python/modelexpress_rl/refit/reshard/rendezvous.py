# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Encode trainer shard metadata and build source geometry for RL refit."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

from modelexpress.nixl_transfer import (
    NIXL_DRAM_MEM_TYPE,
    NIXL_MEM_TYPES,
    NIXL_VRAM_MEM_TYPE,
)
from modelexpress_rl.refit.reshard.slice_plan import Shard
from modelexpress_rl.refit.reshard.transfer_plan import SourceInfo

logger = logging.getLogger("modelexpress_rl.refit.reshard.rendezvous")

_SCHEMA = "mx.reshard.shard_table.v1"
_HOST_SCHEMA = "mx.reshard.shard_table.v2"


@dataclass
class PublishedShard:
    """One published shard of a source tensor: the sub-box it covers and where
    to READ it from (owning agent / device / base address).

    ``digest`` is a position-sensitive digest over the shard's bytes, from
    :func:`modelexpress_rl.refit.reshard.verify.tensor_digest`. ``None`` means the
    publisher did not compute one, which a checker must read as *unchecked* rather
    than as agreement.
    """

    agent_name: str
    device_id: int
    addr: int
    shard_offset: tuple
    shape: tuple
    digest: str | None = None
    memory_type: str = NIXL_VRAM_MEM_TYPE

    def __post_init__(self) -> None:
        if self.memory_type not in NIXL_MEM_TYPES:
            raise ValueError(f"unsupported source memory type {self.memory_type!r}")


@dataclass
class PublishedTensor:
    """A full source tensor as published: its full shape/dtype and the shards
    that cover it (one per owning rank)."""

    name: str
    dtype: str  # e.g. "torch.bfloat16"
    elsize: int
    full_shape: tuple
    shards: list  # list[PublishedShard]


def _encode_shard(shard) -> dict:
    """One shard as JSON.

    ``digest`` is omitted rather than written as null when the publisher has none, so
    a blob from a publisher that does not digest stays byte-identical to what a client
    without this field would have produced.
    """
    encoded = {
        "agent_name": shard.agent_name,
        "device_id": shard.device_id,
        "addr": shard.addr,
        "shard_offset": list(shard.shard_offset),
        "shape": list(shard.shape),
    }
    if shard.digest is not None:
        encoded["digest"] = shard.digest
    if shard.memory_type != NIXL_VRAM_MEM_TYPE:
        encoded["memory_type"] = shard.memory_type
    return encoded


def _schema_for(tensors: list) -> str:
    # Old readers must reject host addresses rather than issue VRAM reads.
    return (
        _HOST_SCHEMA
        if any(s.memory_type != NIXL_VRAM_MEM_TYPE for t in tensors for s in t.shards)
        else _SCHEMA
    )


def _encode_tensor_entries(tensors: list) -> list[dict]:
    """Encode published tensors, shared by the shard table and the blob wrapper.

    The wrapper used to build the shard table and immediately parse it back to
    reach these entries, paying a serialize and a parse of the whole table to
    get at a value it could construct directly.
    """
    return [
        {
            "name": tensor.name,
            "dtype": tensor.dtype,
            "elsize": tensor.elsize,
            "full_shape": list(tensor.full_shape),
            "shards": [_encode_shard(shard) for shard in tensor.shards],
        }
        for tensor in tensors
    ]


def encode_shard_table(tensors: list) -> bytes:
    """Serialize published tensors + shards to a JSON blob."""
    payload = {
        "schema": _schema_for(tensors),
        "tensors": _encode_tensor_entries(tensors),
    }
    return json.dumps(payload).encode("utf-8")


def decode_shard_table(blob: bytes) -> list:
    """Inverse of ``encode_shard_table``; returns ``list[PublishedTensor]``."""
    payload = json.loads(blob.decode("utf-8"))
    schema = payload.get("schema")
    if schema not in (_SCHEMA, _HOST_SCHEMA):
        raise ValueError(
            f"unexpected shard-table schema {schema!r} "
            f"(want one of {_SCHEMA!r}, {_HOST_SCHEMA!r})"
        )
    return decode_shard_entries(payload["tensors"])


def decode_shard_entries(entries: list) -> list:
    """Build ``PublishedTensor``s from already-parsed shard-table entries.

    Split out so a caller holding a parsed blob does not have to re-serialize the
    table just to hand it back to a decoder that immediately re-parses it. On
    Qwen3-30B-A3B that round-trip cost ~0.47 s per refit across 16 trainer
    sources, for nothing.
    """
    tensors = []
    for t in entries:
        shards = [
            PublishedShard(
                agent_name=s["agent_name"],
                device_id=int(s["device_id"]),
                addr=int(s["addr"]),
                shard_offset=tuple(s["shard_offset"]),
                shape=tuple(s["shape"]),
                digest=s.get("digest"),
                memory_type=s.get("memory_type", NIXL_VRAM_MEM_TYPE),
            )
            for s in t["shards"]
        ]
        tensors.append(
            PublishedTensor(
                name=t["name"],
                dtype=t["dtype"],
                elsize=int(t["elsize"]),
                full_shape=tuple(t["full_shape"]),
                shards=shards,
            )
        )
    return tensors


def _torch_dtype(label: str):
    import torch

    return getattr(torch, label.split(".")[-1])


def build_sources(tensors: list, *, session_to_memory: dict | None = None) -> tuple:
    """Turn decoded ``PublishedTensor``s into the planning inputs.

    Returns ``(sources, session_to_agent, session_to_device)`` where ``sources``
    is ``{src_name: SourceInfo}`` for ``plan_transfer`` and the two maps drive
    ``NixlReshardTransport``. Each shard's ``session`` is its owning agent name.
    Returned mappings and source rows are immutable and own their geometry.
    Host-aware callers must supply ``session_to_memory``, populated here with
    the memory type for each session, and forward it to the transport.
    """
    sources = {}
    session_to_agent = {}
    session_to_device = {}
    memory_types = {}
    for t in tensors:
        dtype = _torch_dtype(t.dtype)
        shards = []
        for s in t.shards:
            session = s.agent_name
            if s.memory_type == NIXL_DRAM_MEM_TYPE and session_to_memory is None:
                raise ValueError("host sources require a memory-type-aware receiver")
            if session in session_to_device and (
                session_to_device[session] != s.device_id
                or memory_types[session] != s.memory_type
            ):
                raise ValueError(f"conflicting device or memory type for {session!r}")
            memory_types[session] = s.memory_type
            shards.append(
                Shard(
                    shard_offset=tuple(s.shard_offset),
                    shape=tuple(s.shape),
                    session=session,
                    addr=s.addr,
                    elsize=t.elsize,
                    digest=s.digest,
                )
            )
            session_to_agent[session] = s.agent_name
            session_to_device[session] = s.device_id
        sources[t.name] = SourceInfo(
            global_shape=tuple(t.full_shape),
            dtype=dtype,
            elsize=t.elsize,
            shards=tuple(shards),
        )
    if session_to_memory is not None:
        session_to_memory.update(memory_types)
    return (
        MappingProxyType(sources),
        MappingProxyType(session_to_agent),
        MappingProxyType(session_to_device),
    )


def merge_shard_tables(tables: list) -> list:
    """Merge per-rank ``list[PublishedTensor]`` into one, concatenating shards
    for the same source across ranks (reshard fans in cross-rank). full_shape /
    dtype / elsize must agree across ranks for a given tensor name.

    Replica publishers advertise the same geometric shard through DP/EDP
    replication, so exactly one representative is retained per exact
    offset/shape. Retaining all of them is what the merge used to do, and it
    costs a read per replica: a replicated tensor under DP8 was pulled eight
    times into the same destination bytes, and each extra owner added a P2P
    handshake. It also defeats the full-pull optimization, whose dim-0
    partitioner fails closed on overlapping shards and falls back to the exact
    plan that carries the duplicates.

    Selecting among candidates assumes they really are replicas. They are
    byte-identical by construction when they come from DP/EDP replication, but a
    publisher emitting parallelism-local names makes two *different* tensors
    collide under one name, and then retaining the first installs bytes that
    belong to the other. Publishers must therefore use globally unique names for
    parallelism-local tensors.
    """
    merged: dict = {}
    # name -> geometry -> first shard, insertion-ordered so the retained geometry
    # sequence is deterministic.
    candidates: dict = {}
    for table in tables:
        for t in table:
            cur = merged.get(t.name)
            if cur is None:
                merged[t.name] = PublishedTensor(
                    t.name, t.dtype, t.elsize, t.full_shape, []
                )
                candidates[t.name] = {}
            elif cur.full_shape != t.full_shape or cur.dtype != t.dtype:
                raise ValueError(
                    f"tensor {t.name!r} published with inconsistent shape/dtype across ranks: "
                    f"{cur.full_shape}/{cur.dtype} vs {t.full_shape}/{t.dtype}"
                )
            per_geometry = candidates[t.name]
            for shard in t.shards:
                geometry = (tuple(shard.shard_offset), tuple(shard.shape))
                per_geometry.setdefault(geometry, shard)

    for name, tensor in merged.items():
        tensor.shards.extend(candidates[name].values())
    return list(merged.values())


# --- Rendezvous blob (rides in WorkerMetadata.nixl_metadata) -----------------
# Reshard owns both ends of its publish/discover, so it packs the NIXL agent
# metadata AND the shard table into one blob. TEMPORARY: replaced when the proto
# gains typed shard fields (then agent metadata rides nixl_metadata directly and
# shards ride typed descriptors).


def wrap_rendezvous_blob(
    agent_metadata: bytes,
    agent_name: str,
    metadata_endpoint: str,
    tensors: list,
) -> bytes:
    """Pack ``{agent_meta, agent_name, metadata_endpoint, shard_table}`` into one
    JSON blob. ``metadata_endpoint`` (``host:listen_port`` of the trainer's NIXL
    listen thread) is what the receiver's ``fetch_remote_and_wait`` connects to
    for the P2P memory-registration handshake (the central agent-metadata blob
    alone does not make the registrations resolvable for RDMA reads).

    Counts and byte totals are included in every blob.
    """
    payload = {
        "schema": _schema_for(tensors),
        "agent_name": agent_name,
        "metadata_endpoint": metadata_endpoint,
        "agent_meta_b64": base64.b64encode(agent_metadata).decode("ascii"),
        "tensors": _encode_tensor_entries(tensors),
        "tensor_count": len(tensors),
        "total_bytes": sum(
            math.prod(shard.shape) * tensor.elsize
            for tensor in tensors
            for shard in tensor.shards
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def structural_manifest_digest(blob: bytes) -> str:
    """Hash transfer structure while excluding version-specific content digests.

    Falls back to hashing the raw bytes for anything this cannot read as a
    rendezvous payload, and says so, because the fallback is not equivalent: a
    raw-byte digest moves with the per-shard content digests, so every version
    looks structurally different and plan reuse turns itself off. That is a
    safe direction to fail in -- a replan is correct, just slower -- but it is
    not one to fail in quietly, since the symptom is a warm refit that stays as
    expensive as a cold one for no visible reason.
    """
    try:
        payload = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        logger.warning(
            "rendezvous manifest is not a decodable JSON blob; falling back to a "
            "content-sensitive digest, which disables transfer-plan reuse"
        )
        return hashlib.sha256(blob).hexdigest()
    if not isinstance(payload, dict):
        logger.warning(
            "rendezvous manifest did not decode to an object; falling back to a "
            "content-sensitive digest, which disables transfer-plan reuse"
        )
        return hashlib.sha256(blob).hexdigest()
    for tensor in payload.get("tensors", ()):
        for shard in tensor.get("shards", ()):
            shard.pop("digest", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class RendezvousPayload(NamedTuple):
    """One trainer rank's unwrapped rendezvous blob.

    Named fields retain the existing positional tuple access.
    """

    agent_metadata: bytes
    agent_name: str
    metadata_endpoint: str
    tensors: list
    tensor_count: int
    total_bytes: int


def unwrap_rendezvous_blob(blob: bytes, *, stable: bool = False) -> RendezvousPayload:
    """Decode a blob; stable metadata also requires counts and excludes checksums."""
    payload = json.loads(blob.decode("utf-8"))
    if payload.get("schema") not in (_SCHEMA, _HOST_SCHEMA):
        raise ValueError(f"unexpected rendezvous blob schema {payload.get('schema')!r}")
    agent_metadata = base64.b64decode(payload["agent_meta_b64"])
    agent_name = payload["agent_name"]
    metadata_endpoint = payload.get("metadata_endpoint", "")
    tensors = decode_shard_entries(payload["tensors"])
    byte_count = sum(
        math.prod(shard.shape) * tensor.elsize
        for tensor in tensors for shard in tensor.shards
    )
    tensor_count = payload.get("tensor_count")
    if tensor_count is None:
        tensor_count = len(tensors)
    total_bytes = payload.get("total_bytes")
    if total_bytes is None:
        total_bytes = byte_count
    if stable:
        if (
            type(payload["tensor_count"]) is not int or tensor_count <= 0
            or type(payload["total_bytes"]) is not int or total_bytes <= 0
            or tensor_count != len(tensors) or total_bytes != byte_count
            or any("digest" in shard for tensor in payload["tensors"] for shard in tensor["shards"])
        ):
            raise ValueError("stable metadata contains version state or invalid counts")
        if len({tensor.name for tensor in tensors}) != len(tensors):
            raise ValueError("stable metadata contains duplicate tensor names")
        for tensor in tensors:
            identities = {
                (shard.agent_name, shard.addr, tuple(shard.shard_offset), tuple(shard.shape))
                for shard in tensor.shards
            }
            if len(identities) != len(tensor.shards):
                raise ValueError("stable metadata contains duplicate physical shards")
    return RendezvousPayload(
        agent_metadata,
        agent_name,
        metadata_endpoint,
        tensors,
        tensor_count,
        total_bytes,
    )
