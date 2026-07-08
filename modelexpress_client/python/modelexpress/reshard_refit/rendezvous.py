# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Shard-geometry rendezvous for the reshard weight broadcast.

TEMPORARY / NEXT STEP - add typed shard fields to the proto. This whole module
works around ``TensorDescriptor`` carrying only ``name/addr/size/device_id/dtype``
and no per-dim shard geometry. Until the proto has those fields, the trainer packs
the resharding side-table (per source tensor: full shape + each shard's per-dim
offset/shape + owning NIXL agent/device/base address) into a self-describing JSON
blob that rides alongside the NIXL agent metadata; the inference side decodes it
into the ``modelexpress.reshard_refit`` planning inputs (a ``SourceInfo`` per source +
the shard -> owning-agent/device maps). When the proto gains those fields, delete
the encode/decode here and build the same maps from typed descriptors -
``NixlReshardTransport`` and the slice-plan / pull core are untouched.

RENDEZVOUS IDENTITY: trainer and inference must compute the SAME
``SourceIdentity`` for a role (inference builds the ``role="trainer"`` identity to
DISCOVER it), so the identity may contain only fields both sides derive
identically. They differ in ``tp/pp/ep`` (FSDP vs vLLM tp) and framework, so we
cannot reuse ``build_source_identity`` wholesale; instead we derive the two shared
values faithfully - ``model_name`` (the single ``[model] name`` both configs
inherit) and ``mx_version`` (the ``modelexpress`` package version) - with a fixed
framework as the only other hash key. The served dtype is deliberately NOT in the
identity (the receiver builds it before discovering anything the trainer served);
the real dtype rides in the shard table (``PublishedTensor.dtype``, from the
publisher). See :meth:`_identity`.

Encode/decode are dependency-free; only ``build_sources`` touches torch, to map
the dtype label back to a ``torch.dtype`` for the dtype-match check (a raw RDMA
copy is byte-for-byte, so source and dest dtypes must agree).
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass

from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from modelexpress.reshard_refit.transfer_plan import SourceInfo
from modelexpress.reshard_refit.slice_plan import Shard

logger = logging.getLogger("modelexpress.reshard_refit.rendezvous")

_SCHEMA = "mx.reshard.shard_table.v1"


def _mx_version() -> str:
    """The ``modelexpress`` package version, folded into the SourceIdentity hash
    so trainer and inference on the same MX build resolve the same mx_source_id.
    Derived (not a literal) so it tracks the real build."""
    from importlib.metadata import version as pkg_version

    try:
        return pkg_version("modelexpress")
    except Exception:
        return "0.0.0"


@dataclass
class PublishedShard:
    """One published shard of a source tensor: the sub-box it covers and where
    to READ it from (owning agent / device / base address)."""

    agent_name: str
    device_id: int
    addr: int
    shard_offset: tuple
    shape: tuple


@dataclass
class PublishedTensor:
    """A full source tensor as published: its full shape/dtype and the shards
    that cover it (one per owning rank)."""

    name: str
    dtype: str  # e.g. "torch.bfloat16"
    elsize: int
    full_shape: tuple
    shards: list  # list[PublishedShard]


def encode_shard_table(tensors: list) -> bytes:
    """Serialize published tensors + shards to a JSON blob."""
    payload = {
        "schema": _SCHEMA,
        "tensors": [
            {
                "name": t.name,
                "dtype": t.dtype,
                "elsize": t.elsize,
                "full_shape": list(t.full_shape),
                "shards": [
                    {
                        "agent_name": s.agent_name,
                        "device_id": s.device_id,
                        "addr": s.addr,
                        "shard_offset": list(s.shard_offset),
                        "shape": list(s.shape),
                    }
                    for s in t.shards
                ],
            }
            for t in tensors
        ],
    }
    return json.dumps(payload).encode("utf-8")


def decode_shard_table(blob: bytes) -> list:
    """Inverse of ``encode_shard_table``; returns ``list[PublishedTensor]``."""
    payload = json.loads(blob.decode("utf-8"))
    schema = payload.get("schema")
    if schema != _SCHEMA:
        raise ValueError(f"unexpected shard-table schema {schema!r} (want {_SCHEMA!r})")
    tensors = []
    for t in payload["tensors"]:
        shards = [
            PublishedShard(
                agent_name=s["agent_name"],
                device_id=int(s["device_id"]),
                addr=int(s["addr"]),
                shard_offset=tuple(s["shard_offset"]),
                shape=tuple(s["shape"]),
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


def build_sources(tensors: list) -> tuple:
    """Turn decoded ``PublishedTensor``s into the planning inputs.

    Returns ``(sources, session_to_agent, session_to_device)`` where ``sources``
    is ``{src_name: SourceInfo}`` for ``plan_transfer`` and the two maps drive
    ``NixlReshardTransport``. Each shard's ``session`` is its owning agent name.
    """
    sources = {}
    session_to_agent = {}
    session_to_device = {}
    for t in tensors:
        dtype = _torch_dtype(t.dtype)
        shards = []
        for s in t.shards:
            session = s.agent_name
            shards.append(
                Shard(
                    shard_offset=s.shard_offset,
                    shape=s.shape,
                    session=session,
                    addr=s.addr,
                    elsize=t.elsize,
                )
            )
            session_to_agent[session] = s.agent_name
            session_to_device[session] = s.device_id
        sources[t.name] = SourceInfo(
            global_shape=t.full_shape,
            dtype=dtype,
            elsize=t.elsize,
            shards=shards,
        )
    return sources, session_to_agent, session_to_device


def merge_shard_tables(tables: list) -> list:
    """Merge per-rank ``list[PublishedTensor]`` into one, concatenating shards
    for the same source across ranks (reshard fans in cross-rank). full_shape /
    dtype / elsize must agree across ranks for a given tensor name."""
    merged: dict = {}
    for table in tables:
        for t in table:
            cur = merged.get(t.name)
            if cur is None:
                merged[t.name] = PublishedTensor(t.name, t.dtype, t.elsize, t.full_shape, list(t.shards))
                continue
            if cur.full_shape != t.full_shape or cur.dtype != t.dtype:
                raise ValueError(
                    f"tensor {t.name!r} published with inconsistent shape/dtype across ranks: "
                    f"{cur.full_shape}/{cur.dtype} vs {t.full_shape}/{t.dtype}"
                )
            cur.shards.extend(t.shards)
    return list(merged.values())


# --- Rendezvous blob (rides in WorkerMetadata.nixl_metadata) -----------------
# Reshard owns both ends of its publish/discover, so it packs the NIXL agent
# metadata AND the shard table into one blob. TEMPORARY: replaced when the proto
# gains typed shard fields (then agent metadata rides nixl_metadata directly and
# shards ride typed descriptors).


def wrap_rendezvous_blob(agent_metadata: bytes, agent_name: str, metadata_endpoint: str, tensors: list) -> bytes:
    """Pack ``{agent_meta, agent_name, metadata_endpoint, shard_table}`` into one
    JSON blob. ``metadata_endpoint`` (``host:listen_port`` of the trainer's NIXL
    listen thread) is what the receiver's ``fetch_remote_and_wait`` connects to
    for the P2P memory-registration handshake (the central agent-metadata blob
    alone does not make the registrations resolvable for RDMA reads)."""
    payload = {
        "schema": _SCHEMA,
        "agent_name": agent_name,
        "metadata_endpoint": metadata_endpoint,
        "agent_meta_b64": base64.b64encode(agent_metadata).decode("ascii"),
        "tensors": json.loads(encode_shard_table(tensors).decode("utf-8"))["tensors"],
    }
    return json.dumps(payload).encode("utf-8")


def unwrap_rendezvous_blob(blob: bytes) -> tuple:
    """Inverse of ``wrap_rendezvous_blob``; returns ``(agent_metadata, agent_name,
    metadata_endpoint, tensors)``."""
    payload = json.loads(blob.decode("utf-8"))
    if payload.get("schema") != _SCHEMA:
        raise ValueError(f"unexpected rendezvous blob schema {payload.get('schema')!r}")
    agent_metadata = base64.b64decode(payload["agent_meta_b64"])
    agent_name = payload["agent_name"]
    metadata_endpoint = payload.get("metadata_endpoint", "")
    tensors = decode_shard_table(json.dumps({"schema": _SCHEMA, "tensors": payload["tensors"]}).encode("utf-8"))
    return agent_metadata, agent_name, metadata_endpoint, tensors


class MxReshardRendezvous:
    """Thin rendezvous over ``MxClient`` for the reshard broadcast.

    Trainer ranks ``publish`` their (agent metadata + shard table) blob under a
    role-stamped identity; inference workers ``discover_trainers`` all trainer
    ranks and merge their shard tables. Delegates all gRPC to ``MxClient`` and
    distinguishes roles via ``SourceIdentity.extra_parameters['role']`` so they
    hash to different ``mx_source_id``s.
    """

    def __init__(
        self,
        client: MxClient,
        role: str,
        rank: int,
        model_name: str,
        worker_id: str = "",
    ) -> None:
        self.client = client
        self.role = role
        self.rank = rank
        # The served model name (the single ``[model] name`` both trainer and
        # inference inherit) - a shared identity field both sides derive equally.
        self.model_name = model_name
        self.worker_id = worker_id or str(uuid.uuid4())
        self._mx_source_id: str | None = None

    def _identity(self, role: str) -> "p2p_pb2.SourceIdentity":
        # Only fields BOTH sides derive identically (see module docstring): the
        # shared model_name + mx_version + a fixed framework, with the role in
        # extra_parameters. No dtype here - the receiver builds this identity to
        # DISCOVER the trainer (before it knows anything the trainer served), so
        # the served dtype can't be a hash input; it rides in the shard table
        # (``PublishedTensor.dtype``, from the publisher) instead.
        return p2p_pb2.SourceIdentity(
            mx_version=_mx_version(),
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
            model_name=self.model_name,
            backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
            extra_parameters={"role": role},
        )

    def publish(self, blob: bytes) -> str:
        """Publish this rank's rendezvous blob (agent meta + shard table)."""
        worker = p2p_pb2.WorkerMetadata(worker_rank=self.rank, nixl_metadata=blob)
        self._mx_source_id = self.client.publish_metadata(self._identity(self.role), worker, self.worker_id)
        return self._mx_source_id

    def discover_trainers(
        self,
        expected_trainers: int,
        timeout: float = 1200.0,
        poll_interval: float = 1.0,
    ) -> list:
        """Block until ``expected_trainers`` trainer ranks are visible, then
        fetch + unwrap each. Returns ``list[(agent_metadata, agent_name,
        metadata_endpoint, tensors)]``, one per trainer rank."""
        trainer_id = self._identity("trainer")
        deadline = time.monotonic() + timeout
        while True:
            resp = self.client.list_sources(trainer_id)
            instances = list(resp.instances)
            if len(instances) >= expected_trainers:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out after {timeout}s waiting for {expected_trainers} trainer ranks "
                    f"(saw {len(instances)})"
                )
            time.sleep(poll_interval)

        payloads = []
        for inst in instances:
            meta = self.client.get_metadata(inst.mx_source_id, inst.worker_id)
            if not meta.found:
                continue
            payloads.append(unwrap_rendezvous_blob(meta.worker.nixl_metadata))
        return payloads


def gather_sources(
    client: MxClient,
    expected_trainers: int,
    model_name: str,
    role: str = "inference",
    rank: int = 0,
    timeout: float = 1200.0,
) -> tuple:
    """One-call inference helper: discover all trainer ranks, merge their shard
    tables, and build the planning inputs (per-source ``SourceInfo`` + the
    shard -> owning-agent/device maps).

    Returns ``(sources, session_to_agent, session_to_device, agent_endpoints)``
    where ``agent_endpoints`` is ``{agent_name: metadata_endpoint}`` for the
    caller to ``fetch_remote_and_wait`` (P2P) before pulling."""
    rdv = MxReshardRendezvous(client, role=role, rank=rank, model_name=model_name)
    payloads = rdv.discover_trainers(expected_trainers, timeout=timeout)
    tables = [tensors for (_meta, _name, _ep, tensors) in payloads]
    agent_endpoints = {name: ep for (_meta, name, ep, _tensors) in payloads}
    merged = merge_shard_tables(tables)
    sources, session_to_agent, session_to_device = build_sources(merged)
    return sources, session_to_agent, session_to_device, agent_endpoints
