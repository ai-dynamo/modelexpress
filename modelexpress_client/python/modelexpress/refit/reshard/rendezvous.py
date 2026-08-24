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
import time
import uuid
from dataclasses import dataclass
from typing import NamedTuple

from modelexpress import envs, p2p_pb2
from modelexpress.client import MxClient
from modelexpress.metadata.publisher import PublisherThread
from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard.transfer_plan import SourceInfo

logger = logging.getLogger("modelexpress.refit.reshard.rendezvous")

_SCHEMA = "mx.reshard.shard_table.v1"


def _mx_version() -> str:
    """The ``modelexpress`` package version, folded into the SourceIdentity hash
    so trainer and inference on the same MX build resolve the same mx_source_id.
    Derived (not a literal) so it tracks the real build."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    try:
        return pkg_version("modelexpress")
    except PackageNotFoundError:
        return "0.0.0"


@dataclass
class PublishedShard:
    """One published shard of a source tensor: the sub-box it covers and where
    to READ it from (owning agent / device / base address).

    ``digest`` is a position-sensitive digest over the shard's bytes, from
    :func:`modelexpress.refit.reshard.verify.tensor_digest`. ``None`` means the
    publisher did not compute one, which a checker must read as *unchecked* rather
    than as agreement.
    """

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
    return encoded


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
                "shards": [_encode_shard(s) for s in t.shards],
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
    delivered bytes read as corruption. Inferring freshness instead - asking whether
    any digest changed since planning - holds only when a table is wholly stale or
    wholly current, and breaks under partial propagation across many publishers,
    where it reports one lagging publisher's shard as a hard defect. The stamp turns
    that inference into an observation.

    The key is omitted rather than written as null when absent, so a blob from an
    unstamped publisher is byte-identical to one an older client would have produced.
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


class RendezvousPayload(NamedTuple):
    """One trainer rank's unwrapped rendezvous blob.

    A plain tuple until now, which read as ``payload[3]`` at the point where the
    interesting question is asked - whether this rank published any tensors. Being
    a tuple subclass, it still indexes as before.

    ``publisher_step`` is ``None`` for a publisher that does not stamp, and that
    must be read as "unknown", never as step 0: a publisher read as step 0 looks
    permanently behind, and a consumer that excuses lagging publishers would then
    excuse all of its shards, going quiet instead of strict. It carries a default
    so that reading the stamp is a new field rather than a second unwrap function,
    which is also why unpacking must now name six values. ``tensor_count`` keeps
    the published table size available when ``with_tensors=False`` deliberately
    leaves ``tensors`` empty; ``None`` preserves direct and older callers.
    """

    agent_metadata: bytes
    agent_name: str
    metadata_endpoint: str
    tensors: list
    publisher_step: int | None = None
    tensor_count: int | None = None

    def entry_count(self) -> int:
        """How many shard-table entries this rank published.

        Distinct from ``len(self.tensors)`` because a payload unwrapped for a
        quorum check carries the count without paying to build the table. Asking
        "did this rank publish anything" must stay answerable on both.
        """
        return len(self.tensors) if self.tensor_count is None else self.tensor_count


def unwrap_rendezvous_blob(
    blob: bytes, *, with_tensors: bool = True
) -> RendezvousPayload:
    """Inverse of ``wrap_rendezvous_blob``.

    ``with_tensors=False`` decodes everything except the shard table, for callers
    that only need the version stamp. Building the table dominates this function
    (~97 ms vs ~9 ms per source on Qwen3-30B-A3B's 4922 entries), and a per-step
    quorum check re-does it for a table that never changes between steps.
    ``tensor_count`` is still populated, so an empty publisher stays detectable.
    """
    payload = json.loads(blob.decode("utf-8"))
    if payload.get("schema") != _SCHEMA:
        raise ValueError(f"unexpected rendezvous blob schema {payload.get('schema')!r}")
    agent_metadata = base64.b64decode(payload["agent_meta_b64"])
    agent_name = payload["agent_name"]
    metadata_endpoint = payload.get("metadata_endpoint", "")
    entries = payload["tensors"]
    tensors = decode_shard_entries(entries) if with_tensors else []
    raw_step = payload.get("publisher_step")
    publisher_step = None if raw_step is None else int(raw_step)
    return RendezvousPayload(
        agent_metadata,
        agent_name,
        metadata_endpoint,
        tensors,
        publisher_step,
        len(entries),
    )


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
        self._publisher: PublisherThread | None = None

    def _identity(self, role: str) -> p2p_pb2.SourceIdentity:
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
        """Publish a complete rendezvous blob as immediately discoverable.

        Callers build ``blob`` only after the NIXL agent and source buffers are
        registered, so publication is the readiness boundary for this minimal
        rendezvous API. Re-publishing refreshes the worker record for the same
        ``worker_id``.
        """
        heartbeat_period = envs.MX_HEARTBEAT_INTERVAL_SECS
        if heartbeat_period <= 0:
            raise ValueError(
                f"MX_HEARTBEAT_INTERVAL_SECS must be positive, got {heartbeat_period}"
            )
        if self._publisher is not None:
            # Stop the old status heartbeat before replacing the publication.
            # Stopping it afterwards would mark the replacement STALE when the
            # identity and worker ID resolve to the same source.
            self._publisher.stop()
            self._publisher = None
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=self.rank,
            nixl_metadata=blob,
            status=p2p_pb2.SOURCE_STATUS_READY,
        )
        self._mx_source_id = self.client.publish_metadata(
            self._identity(self.role), worker, self.worker_id
        )
        self._publisher = PublisherThread(
            mx_client=self.client,
            mx_source_id=self._mx_source_id,
            worker_id=self.worker_id,
            worker_rank=self.rank,
            interval_secs=heartbeat_period,
        )
        self._publisher.start()
        return self._mx_source_id

    def close(self) -> None:
        """Stop heartbeats and best-effort mark the published source STALE."""
        if self._publisher is not None:
            self._publisher.stop()
            self._publisher = None
        self._mx_source_id = None

    def _fetch_metadata(self, instances: list) -> list:
        """Fetch each instance's metadata, in order.

        Returns one entry per instance, ``None`` where the fetch raised. A single
        rank's transport error must not abort the sweep: the poll loop's job is to
        report how many ranks are actually readable, and one raised exception would
        instead surface as a discovery failure with no count at all.

        Serial on purpose, which is the opposite of what the shape of the problem
        suggests. Issuing these round-trips from a thread pool was tried and made
        it **worse**: the fetch went from 4.02 s to ~6.9 s median on 16 sources.
        The contention is not in this process. Every receiver rank runs this loop,
        so 16 ranks x 16 sources put 256 requests and ~358 MB of shard table
        in flight against one metadata server, and queueing there costs more than
        the serial round-trips saved. Making this concurrent requires reducing what
        the server has to serve first - see the note in ``discover_trainers``.
        """
        return [self._get_metadata_or_none(inst) for inst in instances]

    def _get_metadata_or_none(self, inst):
        try:
            return self.client.get_metadata(inst.mx_source_id, inst.worker_id)
        except Exception as exc:  # noqa: BLE001 - one bad rank must not end the sweep
            logger.warning(
                "[reshard] get_metadata failed for source %s worker %s: %s",
                inst.mx_source_id,
                inst.worker_id,
                exc,
            )
            return None

    def discover_trainers(
        self,
        expected_trainers: int,
        timeout: float = 1200.0,
        poll_interval: float = 1.0,
        with_tensors: bool = True,
    ) -> list:
        """Block until ``expected_trainers`` trainer ranks are visible **with a
        non-empty shard table**, then return them.

        Returns ``list[RendezvousPayload]``, one per trainer rank.

        ``with_tensors=False`` skips building the shard tables, for a caller that
        only needs each rank's version stamp. The quorum semantics are unchanged -
        emptiness is still judged on the published entry count - but a per-step
        check no longer rebuilds tables that are identical every step. Anything
        that reads shard geometry must leave this ``True``.

        A rank counts toward the quorum only once its published table names at
        least one tensor. A rank that advertises READY with nothing to read has
        registered no memory, so satisfying the quorum with it makes the receiver
        stop waiting for the ranks that do have bytes and then stall in the P2P
        handshake instead - a timeout attributed to the wrong component.

        Exactly ``expected_trainers`` ranks are returned. A stale source from an
        earlier run can still be READY, and every extra rank returned here becomes
        an extra peer to handshake and an extra set of shards competing to describe
        the same tensor names.
        """
        trainer_id = self._identity("trainer")
        deadline = time.monotonic() + timeout
        # Cost split, kept because this call dominates a MoE refit: measured at
        # 51% of a 10.25 s refit on Qwen3-30B-A3B, 2.7x the weight transfer it
        # precedes. On the cluster the fetch is ~4.0 s of that and the parse ~0.8 s.
        # The round-trips stay serial; see ``_fetch_metadata`` for the measurement
        # that ruled out a thread pool. ``fetch_s`` includes both ``list_sources``
        # and the metadata sweep.
        fetch_s = 0.0
        parse_s = 0.0
        polls = 0
        while True:
            list_t0 = time.perf_counter()
            resp = self.client.list_sources(
                trainer_id,
                status_filter=p2p_pb2.SOURCE_STATUS_READY,
            )
            instances = list(resp.instances)
            fetch_s += time.perf_counter() - list_t0
            payloads, empty = [], 0
            # Every visible READY source is inspected, whatever the count: with
            # fewer sources than expected the shard-table state is exactly what a
            # timeout here has to report, and reaching the quorum on instances
            # alone says nothing about whether any of them published bytes.
            #
            # The fetch is the dominant term (~4.0 s of ~4.8 s here) and it is
            # server-bound, not client-bound: see ``_fetch_metadata`` for the
            # measurement that ruled out concurrency. Cutting it further means
            # sending less, which needs ``publisher_step`` carried in the
            # ``list_sources`` instance record so the quorum costs one small call
            # instead of one full shard table per rank. That is a protocol change.
            fetch_t0 = time.perf_counter()
            metas = self._fetch_metadata(instances)
            fetch_s += time.perf_counter() - fetch_t0
            for meta in metas:
                if meta is None or not meta.found:
                    continue
                parse_t0 = time.perf_counter()
                payload = unwrap_rendezvous_blob(
                    meta.worker.nixl_metadata, with_tensors=with_tensors
                )
                parse_s += time.perf_counter() - parse_t0
                if payload.entry_count() == 0:
                    empty += 1
                    continue
                payloads.append(payload)
                if len(payloads) >= expected_trainers:
                    break
            polls += 1
            if len(payloads) >= expected_trainers:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out after {timeout}s waiting for {expected_trainers} "
                    f"trainer ranks (saw {len(instances)} READY source(s), "
                    f"{len(payloads)} with a non-empty shard table, {empty} empty)"
                )
            time.sleep(poll_interval)

        logger.warning(
            "MX_DISCOVER_COST %s",
            json.dumps(
                {
                    "schema": "mx-discover-cost-v1",
                    "rank": self.rank,
                    "sources": len(payloads),
                    # entry_count, not len(tensors): the quorum path does not build
                    # the tables, and this figure is what shows the cost tracks
                    # source count rather than bytes moved.
                    "tensors": sum(p.entry_count() for p in payloads),
                    "tables_built": with_tensors,
                    "grpc_fetch_s": round(fetch_s, 6),
                    "blob_parse_s": round(parse_s, 6),
                    "polls": polls,
                }
            ),
        )
        logger.info(
            "[reshard] discovered %d trainer rank(s)%s: %s",
            len(payloads),
            f" ({empty} skipped as empty)" if empty else "",
            ", ".join(
                f"{p.agent_name}@{p.metadata_endpoint}[{len(p.tensors)}]"
                for p in payloads
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
) -> tuple:
    """One-call inference helper: discover all trainer ranks, merge their shard
    tables, and build the planning inputs (per-source ``SourceInfo`` + the
    shard -> owning-agent/device maps).

    Returns ``(sources, session_to_agent, session_to_device, agent_endpoints)``
    where ``agent_endpoints`` is ``{agent_name: metadata_endpoint}`` for the
    caller to ``fetch_remote_and_wait`` (P2P) before pulling."""
    rdv = MxReshardRendezvous(client, role=role, rank=rank, model_name=model_name)
    payloads = rdv.discover_trainers(expected_trainers, timeout=timeout)
    tables = [p.tensors for p in payloads]
    agent_endpoints = {p.agent_name: p.metadata_endpoint for p in payloads}
    merged = merge_shard_tables(tables)
    sources, session_to_agent, session_to_device = build_sources(merged)
    return sources, session_to_agent, session_to_device, agent_endpoints
