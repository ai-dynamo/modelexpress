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
into the ``modelexpress.refit.reshard`` planning inputs (a ``SourceInfo`` per source +
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
import os
import time
import uuid
from dataclasses import dataclass

from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard.transfer_plan import SourceInfo

logger = logging.getLogger("modelexpress.refit.reshard.rendezvous")

_SCHEMA = "mx.reshard.shard_table.v1"

# On by default: a same-name/same-geometry pair with different digests is always a
# defect, and the failure it causes otherwise is silent and undetectable downstream.
# Costs nothing when publishers omit digests.
_STRICT_DIGESTS = os.environ.get("MX_RESHARD_STRICT_DIGESTS", "1") not in ("0", "false", "False")


def _mx_version() -> str:
    """The ``modelexpress`` package version, folded into the SourceIdentity hash
    so trainer and inference on the same MX build resolve the same mx_source_id.
    Derived (not a literal) so it tracks the real build."""
    from importlib.metadata import PackageNotFoundError, version as pkg_version

    try:
        return pkg_version("modelexpress")
    except PackageNotFoundError:
        return "0.0.0"


@dataclass
class PublishedShard:
    """One published shard of a source tensor: the sub-box it covers and where
    to READ it from (owning agent / device / base address).

    ``digest`` is an optional position-sensitive digest of the shard's bytes (see
    ``verify.tensor_digest``), present only when the publisher ran with
    ``MX_RESHARD_VERIFY=1``. It lets a receiver prove the bytes it pulled are the
    bytes the trainer holds. Optional and defaulted so a mixed fleet still
    interoperates: an older publisher simply omits it and the receiver reports the
    shard as unchecked rather than failing."""

    agent_name: str
    device_id: int
    addr: int
    shard_offset: tuple
    shape: tuple
    digest: str | None = None


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
                        # Omitted when absent so the blob stays byte-identical to
                        # the pre-digest schema for publishers not verifying.
                        **({"digest": s.digest} if s.digest else {}),
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
                digest=s.get("digest"),
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
                    digest=s.digest,
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


def merge_shard_tables(
    tables: list, replica_offset: int = 0, strict_digests: bool = _STRICT_DIGESTS
) -> list:
    """Merge per-rank ``list[PublishedTensor]`` into one, concatenating shards
    for the same source across ranks (reshard fans in cross-rank). Replica
    publishers can advertise the same geometric shard through DP/EP replication;
    exactly one representative is retained for each exact offset/shape.
    full_shape / dtype / elsize must agree across ranks for a given tensor name.

    ``replica_offset`` selects *which* representative. With DP8 / EDP2 there are
    up to 8 byte-identical copies of a shard on distinct ranks and distinct NICs;
    the default 0 always takes the first publisher seen, which means every
    receiver in the fleet reads the same shard from the same rank while the other
    replicas serve nothing. Passing the receiver's global rank rotates the choice
    so receivers spread their reads over the available replicas. Correctness is
    unaffected either way: the candidates are replicas of the same weights and are
    byte-identical by construction.

    Geometry set and ordering are identical for every ``replica_offset``, so only
    the owning agent and address of each shard change - the resulting plan has the
    same shape, segment count and byte count.

    ``strict_digests`` raises when two publishers offer the same name and geometry
    with *different* digests. Such offers are not replicas: the name means two
    different tensors somewhere upstream, and picking either one installs bytes
    that belong to the other. That is exactly how Bug 8 hid - pipeline-local layer
    indices made both PP stages publish ``decoder.layers.0..23``, the tiebreak
    silently dropped one stage's half of the model, and every existing gate passed
    because bytes that are never requested are never checked. Only offers that
    carry a digest can be compared, so this is effective when publishers run with
    ``MX_RESHARD_VERIFY=1``.
    """
    merged: dict = {}
    # name -> geometry -> candidate shards, insertion-ordered so the retained
    # geometry sequence is independent of replica_offset.
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
                per_geometry.setdefault(geometry, []).append(shard)

    for name, tensor in merged.items():
        for geometry, offers in candidates[name].items():
            if strict_digests and len(offers) > 1:
                _assert_offers_are_replicas(name, geometry, offers)
            tensor.shards.append(offers[replica_offset % len(offers)])
    return list(merged.values())


def _assert_offers_are_replicas(name: str, geometry: tuple, offers: list) -> None:
    """Raise if competing offers for one name/geometry hold different bytes."""
    by_digest: dict = {}
    for shard in offers:
        if shard.digest:
            by_digest.setdefault(shard.digest, []).append(shard.agent_name)
    if len(by_digest) <= 1:
        return
    detail = ", ".join(
        f"{digest[:12]} from {sorted(set(agents))}" for digest, agents in by_digest.items()
    )
    offset, shape = geometry
    raise ValueError(
        f"tensor {name!r} shard offset={offset} shape={shape} was published by "
        f"multiple ranks with {len(by_digest)} distinct digests ({detail}). These "
        f"are not replicas, so the first-writer-wins tiebreak would install one "
        f"rank's bytes under the other's name. The usual cause is a publisher "
        f"emitting parallelism-local names - see Bug 8, pipeline-local layer "
        f"indices. Set MX_RESHARD_STRICT_DIGESTS=0 to downgrade this to the old "
        f"silent behaviour."
    )


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
    publisher_step: int | None = None,
) -> bytes:
    """Pack ``{agent_meta, agent_name, metadata_endpoint, shard_table}`` into one
    JSON blob. ``metadata_endpoint`` (``host:listen_port`` of the trainer's NIXL
    listen thread) is what the receiver's ``fetch_remote_and_wait`` connects to
    for the P2P memory-registration handshake (the central agent-metadata blob
    alone does not make the registrations resolvable for RDMA reads).

    ``publisher_step`` stamps the table with the training step whose weights it
    describes. A receiver otherwise has no way to tell a current table from one
    published a step ago, and it needs to: the shard table carries the per-shard
    digests a receiver verifies against, so a table one step behind makes correctly
    delivered bytes read as corruption. Inferring freshness instead - "did any digest
    change since prepare?" - works only when a table is wholly stale or wholly
    current, and breaks under *partial* propagation across many publishers, where it
    reports one lagging publisher's shard as a hard defect. The stamp turns that
    inference into an observation. Omitted rather than null when absent, so an older
    receiver reading a newer blob is unaffected.
    """
    payload = {
        "schema": _SCHEMA,
        "agent_name": agent_name,
        "metadata_endpoint": metadata_endpoint,
        "agent_meta_b64": base64.b64encode(agent_metadata).decode("ascii"),
        "tensors": json.loads(encode_shard_table(tensors).decode("utf-8"))["tensors"],
    }
    if publisher_step is not None:
        payload["publisher_step"] = int(publisher_step)
    return json.dumps(payload).encode("utf-8")


def unwrap_rendezvous_blob(blob: bytes) -> tuple:
    """Inverse of ``wrap_rendezvous_blob``; returns ``(agent_metadata, agent_name,
    metadata_endpoint, tensors)``.

    Arity is preserved for callers that predate the step stamp; use
    :func:`unwrap_rendezvous_blob_with_step` to read it.
    """
    return unwrap_rendezvous_blob_with_step(blob)[:4]


def unwrap_rendezvous_blob_with_step(blob: bytes) -> tuple:
    """As :func:`unwrap_rendezvous_blob`, plus the publisher's step stamp.

    Returns ``(agent_metadata, agent_name, metadata_endpoint, tensors,
    publisher_step)``, where ``publisher_step`` is ``None`` for a publisher that
    predates the stamp - which must be read as "unknown", never as step 0.
    """
    payload = json.loads(blob.decode("utf-8"))
    if payload.get("schema") != _SCHEMA:
        raise ValueError(f"unexpected rendezvous blob schema {payload.get('schema')!r}")
    agent_metadata = base64.b64decode(payload["agent_meta_b64"])
    agent_name = payload["agent_name"]
    metadata_endpoint = payload.get("metadata_endpoint", "")
    tensors = decode_shard_table(
        json.dumps({"schema": _SCHEMA, "tensors": payload["tensors"]}).encode("utf-8")
    )
    raw_step = payload.get("publisher_step")
    publisher_step = None if raw_step is None else int(raw_step)
    return agent_metadata, agent_name, metadata_endpoint, tensors, publisher_step


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
        # The blob is built only after CUDA buffers are registered with NIXL,
        # so this publication is immediately readable. Discovery is READY-only;
        # leaving proto3's default UNKNOWN status makes every real receiver wait
        # until timeout even though all transport metadata is present.
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=self.rank,
            nixl_metadata=blob,
            status=p2p_pb2.SOURCE_STATUS_READY,
        )
        # Hand the heartbeat the newest blob on every publish, before publishing.
        # See _start_rendezvous_heartbeat for why this assignment is the whole fix.
        self._hb_worker = worker
        self._mx_source_id = self.client.publish_metadata(
            self._identity(self.role), worker, self.worker_id
        )
        self._start_rendezvous_heartbeat()
        return self._mx_source_id

    def _start_rendezvous_heartbeat(self) -> None:
        """Re-publish periodically so the server reaper keeps this source READY.

        The rendezvous publish is one-shot, but the server marks a source STALE once
        its heartbeat lapses and ``discover_trainers`` filters on READY, so a receiver
        whose own init runs long would find zero trainers while every trainer is alive
        and registered.

        What the heartbeat re-publishes is ``self._hb_worker``, re-read on each beat,
        and that indirection is load-bearing. The first version captured the ``worker``
        argument in the closure, so the thread re-advertised the blob from the *first*
        publish for the lifetime of the process. Later publishes still wrote fresh
        blobs, and the heartbeat then overwrote them with the original - a race decided
        by whoever wrote last.

        The visible symptom was a refit that verified clean on one rank and reported a
        digest mismatch on another for the same tensor, because a shard table carries
        the publisher's digests: a receiver comparing correctly delivered bytes against
        a step-0 digest sees corruption that is not there. Freezing the table also pins
        the buffer addresses in it, so the same defect would advertise dead addresses if
        the registration ever moved. Re-reading the attribute costs nothing and removes
        both.
        """
        import logging as _logging
        import os as _os
        import threading as _threading

        period = float(_os.environ.get("MX_RESHARD_HEARTBEAT_S", "30"))
        if period <= 0 or getattr(self, "_hb_thread", None) is not None:
            return
        _log = _logging.getLogger(__name__)
        stop = _threading.Event()

        def _beat() -> None:
            while not stop.wait(period):
                worker = getattr(self, "_hb_worker", None)
                if worker is None:
                    continue
                try:
                    self.client.publish_metadata(
                        self._identity(self.role), worker, self.worker_id
                    )
                except Exception:
                    _log.debug("reshard rendezvous heartbeat failed", exc_info=True)

        thread = _threading.Thread(
            target=_beat, name=f"mx-reshard-hb-{self.rank}", daemon=True
        )
        self._hb_stop = stop
        self._hb_thread = thread
        thread.start()

    def stop_heartbeat(self, timeout: float = 0.0) -> None:
        """Stop this rendezvous' heartbeat thread, if it has one.

        Needed because the idempotence guard above is *per object*, and a caller
        that builds a new rendezvous per publish therefore accumulates threads -
        each one re-asserting the blob it was created with. That reopens exactly the
        defect ``_hb_worker`` was introduced to close: the server's table reverts to
        an older snapshot on a timer, and a receiver then checks correctly delivered
        bytes against a step-0 digest. Whoever replaces a rendezvous must retire the
        one it replaces.
        """
        stop = getattr(self, "_hb_stop", None)
        thread = getattr(self, "_hb_thread", None)
        if stop is not None:
            stop.set()
        if thread is not None and timeout > 0:
            thread.join(timeout)
        self._hb_stop = None
        self._hb_thread = None

    def discover_trainers(
        self,
        expected_trainers: int,
        timeout: float = 1200.0,
        poll_interval: float = 1.0,
    ) -> list:
        """Block until ``expected_trainers`` trainer ranks are visible, then
        fetch + unwrap each. Returns ``list[(agent_metadata, agent_name,
        metadata_endpoint, tensors)]``, one per trainer rank.

        Arity preserved for existing callers; see
        :meth:`discover_trainers_with_steps` for the publisher step stamps.
        """
        payloads = self.discover_trainers_with_steps(
            expected_trainers, timeout=timeout, poll_interval=poll_interval
        )
        return [p[:4] for p in payloads]

    def discover_trainers_with_steps(
        self,
        expected_trainers: int,
        timeout: float = 1200.0,
        poll_interval: float = 1.0,
    ) -> list:
        """As :meth:`discover_trainers`, with each publisher's step stamp appended.

        Returns ``list[(agent_metadata, agent_name, metadata_endpoint, tensors,
        publisher_step)]``. ``publisher_step`` is ``None`` for a publisher that does
        not stamp.
        """
        trainer_id = self._identity("trainer")
        deadline = time.monotonic() + timeout
        empty = 0
        while True:
            resp = self.client.list_sources(
                trainer_id,
                status_filter=p2p_pb2.SOURCE_STATUS_READY,
            )
            instances = list(resp.instances)
            payloads, empty = [], 0
            if len(instances) >= expected_trainers:
                for inst in instances:
                    meta = self.client.get_metadata(inst.mx_source_id, inst.worker_id)
                    if not meta.found:
                        continue
                    payload = unwrap_rendezvous_blob_with_step(
                        meta.worker.nixl_metadata
                    )
                    # A rank that advertises no tensors has registered nothing to
                    # read. Counting it toward the quorum lets the receiver stop
                    # waiting for the ranks that matter and then stall in the P2P
                    # handshake instead, so it does not count.
                    if not payload[3]:
                        empty += 1
                        continue
                    payloads.append(payload)
                if len(payloads) >= expected_trainers:
                    break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out after {timeout}s waiting for {expected_trainers} trainer ranks "
                    f"(saw {len(instances)} READY source(s), {len(payloads)} with a "
                    f"non-empty shard table, {empty} empty)"
                )
            time.sleep(poll_interval)

        logger.info(
            "[reshard] discovered %d trainer rank(s)%s: %s",
            len(payloads),
            f" ({empty} skipped as empty)" if empty else "",
            ", ".join(
                f"{name}@{endpoint}[{len(tensors)}]"
                + ("" if pstep is None else f"@step{pstep}")
                for (_meta, name, endpoint, tensors, pstep) in payloads
            ),
        )
        return payloads


def gather_sources(
    client: MxClient,
    expected_trainers: int,
    model_name: str,
    role: str = "inference",
    rank: int = 0,
    timeout: float = 1200.0,
    replica_offset: int = 0,
) -> tuple:
    """One-call inference helper: discover all trainer ranks, merge their shard
    tables, and build the planning inputs (per-source ``SourceInfo`` + the
    shard -> owning-agent/device maps).

    ``replica_offset`` is forwarded to :func:`merge_shard_tables` to choose which
    duplicate replica serves each shard; see there for why it matters.

    Returns ``(sources, session_to_agent, session_to_device, agent_endpoints)``
    where ``agent_endpoints`` is ``{agent_name: metadata_endpoint}`` for the
    caller to ``fetch_remote_and_wait`` (P2P) before pulling.

    Arity preserved for external callers; :func:`gather_sources_with_steps` adds the
    per-publisher step stamps."""
    return gather_sources_with_steps(
        client,
        expected_trainers,
        model_name,
        role=role,
        rank=rank,
        timeout=timeout,
        replica_offset=replica_offset,
    )[:4]


def gather_sources_with_steps(
    client: MxClient,
    expected_trainers: int,
    model_name: str,
    role: str = "inference",
    rank: int = 0,
    timeout: float = 1200.0,
    replica_offset: int = 0,
) -> tuple:
    """As :func:`gather_sources`, plus ``session_to_step``.

    ``session_to_step`` maps each session to the training step its publisher stamped
    on the table, or ``None`` where the publisher does not stamp. Per session rather
    than one value for the discovery because publishers propagate independently: under
    partial propagation some sessions are current and others a step behind, and a
    single flag cannot express that without either excusing a real defect or
    condemning a healthy shard.
    """
    rdv = MxReshardRendezvous(client, role=role, rank=rank, model_name=model_name)
    payloads = rdv.discover_trainers_with_steps(expected_trainers, timeout=timeout)
    tables = [tensors for (_meta, _name, _ep, tensors, _st) in payloads]
    agent_endpoints = {name: ep for (_meta, name, ep, _tensors, _st) in payloads}
    agent_to_step = {name: st for (_meta, name, _ep, _tensors, st) in payloads}
    merged = merge_shard_tables(tables, replica_offset=replica_offset)
    sources, session_to_agent, session_to_device = build_sources(merged)
    session_to_step = {
        session: agent_to_step.get(agent)
        for session, agent in session_to_agent.items()
    }
    return (
        sources,
        session_to_agent,
        session_to_device,
        agent_endpoints,
        session_to_step,
    )
