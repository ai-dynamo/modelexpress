# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-stage: read a READY source's weights into caller-owned device memory.

After a level-2 sleep (``MxModelLoader.on_sleep``), a checkpointed and
restored vLLM worker gets its weights back from ``reload_model()`` only once
the process restore has finished, so the whole weight transfer sits after the
restore. Pre-stage lets a separate helper process do the same discovery and
the same one-sided RDMA READ *while* the engine process is being restored:

1. The helper allocates device memory for the weight regions the checkpointed
   process released. It knows their sizes, and for every tensor the offset
   inside a region, from a layout recorded at checkpoint time.
2. It finds the source rank with the normal MX discovery path
   (``ListSources`` -> ``GetMetadata`` -> the source worker's
   ``PrepareTensorRead``) and reads every mapped catalog entry into its memory
   with one batched READ (``NixlTransferManager.execute_read_batch``).
3. The restored process takes that memory over (for example a POSIX FD of a
   ``cuMemCreate`` allocation, mapped back at its original virtual
   addresses). That hand-over is engine-specific and not part of this module.

The module needs no engine or model object: the caller passes the
``SourceIdentity`` (for vLLM see
``engines.vllm.source_identity.build_source_identity_from_values``) and a
callable that maps catalog names to local addresses. ``PrestageReceiver``
wraps CUDA device memory.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch

from . import envs, p2p_pb2
from .metadata.payload import accelerators_compatible
from .nixl_transfer import NixlTransferManager
from .source_selection import get_configured_selector
from .topology import apply_policy, blocks_selection, resolve_policy, warn_if_no_domain_match
from .types import TensorDescriptor

logger = logging.getLogger("modelexpress.prestage")

# Same bound as RdmaStrategy's MAX_SOURCE_RETRIES.
DEFAULT_MAX_CANDIDATES = 3


class PrestageError(RuntimeError):
    """Discovery, layout or transfer failure of a pre-stage read."""


@dataclass
class PrestageSource:
    """One source worker, resolved far enough to read from it.

    ``tensor_read`` is the source's read lease when discovery took one; call
    ``release()`` once the READ has completed (or been abandoned).
    """

    mx_source_id: str
    worker_id: str
    worker_rank: int
    agent_name: str
    metadata_endpoint: str
    worker_grpc_endpoint: str
    accelerator: str
    tensors: list[TensorDescriptor]
    timings: dict[str, float] = field(default_factory=dict)
    tensor_read: Any = None
    _release_errors: list[str] = field(default_factory=list, repr=False)

    def release(self) -> list[str]:
        """Release the read lease, if any, and return the release errors.

        A failed or interrupted release is not retried (closing a lease also
        closes its channel); the source drops the lease when it expires. Its
        error stays in the result of every later call.
        """
        tensor_read, self.tensor_read = self.tensor_read, None
        if tensor_read is not None:
            try:
                tensor_read.close()
            except BaseException as exc:
                self._release_errors.append(f"release tensor read lease: {exc!r}"[:300])
                if not isinstance(exc, Exception):
                    raise
        return list(self._release_errors)


@dataclass
class ReadPlan:
    """Byte ranges for ``execute_read_batch`` plus the entries left out."""

    ranges: list[tuple[int, int, int, int]]
    matched: int
    matched_bytes: int
    unmatched: list[str]
    unmatched_bytes: int


def discover_source(
    client: Any,
    identity: p2p_pb2.SourceIdentity,
    worker_rank: int,
    *,
    receiver_id: str,
    accelerator: str = "cuda",
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    manifest_timeout_s: float = 10.0,
    tensor_read_lease: bool = True,
    exclude_worker_ids: Collection[str] = (),
) -> PrestageSource:
    """Resolve one READY P2P source worker for ``worker_rank``.

    Mirrors ``RdmaStrategy``: rank match, accelerator compatibility, the
    configured selector's order, the topology policy, at most
    ``max_candidates`` attempts, and the accelerator re-check on the fetched
    metadata. ``receiver_id`` is this receiver's stable ID; the hashing
    selectors key their order on it, as on a loader's ``worker_id``. Unlike
    ``RdmaStrategy`` this stops before the handshake: when ``connect()`` or
    ``read()`` fails, call again with that worker in ``exclude_worker_ids``.

    With ``tensor_read_lease`` (the default) the manifest comes from
    ``PrepareTensorRead``, which also takes a source read lease; the returned
    source holds it until ``release()``, and the leased manifest's rank and
    accelerator are checked again. The lease protects the read only from
    source changes that drain tensor reads first, and only until it expires.
    Without it the manifest comes from ``GetTensorManifest``, as in the default
    RDMA load strategy, and the caller must ensure the source keeps serving
    until the READ returns.

    Raises ``PrestageError`` when no candidate yields metadata with P2P
    endpoints and a tensor manifest.
    """
    from .metadata.worker_server import fetch_tensor_manifest, prepare_tensor_read

    topology_policy = resolve_policy()
    if blocks_selection(topology_policy):
        raise PrestageError(
            f"topology domain {topology_policy.domain!r} is required but "
            f"unknown for this node"
        )
    excluded = set(exclude_worker_ids)
    t0 = time.perf_counter()
    listed = client.list_sources(
        identity=identity, status_filter=p2p_pb2.SOURCE_STATUS_READY
    )
    t_list = time.perf_counter() - t0
    candidates = [
        inst
        for inst in listed.instances
        if inst.worker_rank == worker_rank
        and inst.worker_id not in excluded
        and accelerators_compatible(
            accelerator,
            inst.accelerator,
            mx_source_type=identity.mx_source_type,
            quantization=identity.quantization,
            dtype=identity.dtype,
        )
    ]
    ctx = SimpleNamespace(
        identity=identity,
        worker_rank=worker_rank,
        global_rank=worker_rank,
        worker_id=receiver_id,
    )
    ordered = apply_policy(get_configured_selector().order(candidates, ctx), topology_policy)
    warn_if_no_domain_match(candidates, topology_policy, receiver_id)
    if not ordered:
        raise PrestageError(
            f"no READY source for worker_rank {worker_rank} "
            f"({len(listed.instances)} READY instances for this identity, "
            f"{len(candidates)} rank and accelerator matched, "
            f"{len(excluded)} excluded)"
        )
    max_retries, retry_backoff = client.worker_rpc_retry_policy()
    errors: list[str] = []
    for inst in ordered[:max_candidates]:
        tensor_read = None
        try:
            t1 = time.perf_counter()
            resp = client.get_metadata(
                mx_source_id=inst.mx_source_id, worker_id=inst.worker_id
            )
            t_meta = time.perf_counter() - t1
            if not resp.found:
                errors.append(f"{inst.worker_id}: metadata not found")
                continue
            worker = resp.worker
            if not worker.worker_grpc_endpoint or not worker.metadata_endpoint:
                errors.append(f"{inst.worker_id}: no P2P endpoints")
                continue
            if not _accelerator_matches(accelerator, worker.accelerator, identity):
                errors.append(f"{inst.worker_id}: accelerator {worker.accelerator!r}")
                continue
            t2 = time.perf_counter()
            if tensor_read_lease:
                tensor_read, nbytes = prepare_tensor_read(
                    worker.worker_grpc_endpoint,
                    inst.mx_source_id,
                    worker_id=inst.worker_id,
                    timeout=manifest_timeout_s,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff,
                )
                manifest = tensor_read.manifest
                # A new channel to a Service endpoint may reach another process
                # than the one GetMetadata described.
                if manifest.worker_rank != worker_rank:
                    raise PrestageError(
                        f"leased manifest is for worker_rank {manifest.worker_rank}"
                    )
                if not _accelerator_matches(accelerator, manifest.accelerator, identity):
                    raise PrestageError(
                        f"leased manifest accelerator {manifest.accelerator!r}"
                    )
                protos = manifest.tensors
                agent_name = manifest.agent_name
                metadata_endpoint = manifest.metadata_endpoint
                source_accelerator = manifest.accelerator
            else:
                protos, nbytes = fetch_tensor_manifest(
                    worker.worker_grpc_endpoint,
                    inst.mx_source_id,
                    manifest_timeout_s,
                    worker_id=inst.worker_id,
                )
                agent_name = worker.agent_name
                metadata_endpoint = worker.metadata_endpoint
                source_accelerator = worker.accelerator
            t_man = time.perf_counter() - t2
            if not agent_name or not metadata_endpoint:
                raise PrestageError("manifest has no NIXL agent or endpoint")
            source = PrestageSource(
                mx_source_id=inst.mx_source_id,
                worker_id=inst.worker_id,
                worker_rank=worker_rank,
                agent_name=agent_name,
                metadata_endpoint=metadata_endpoint,
                worker_grpc_endpoint=worker.worker_grpc_endpoint,
                accelerator=source_accelerator,
                tensors=[
                    TensorDescriptor(
                        name=t.name,
                        addr=t.addr,
                        size=t.size,
                        device_id=t.device_id,
                        dtype=t.dtype,
                    )
                    for t in protos
                ],
                timings={
                    "list_sources_s": round(t_list, 4),
                    "get_metadata_s": round(t_meta, 4),
                    "tensor_manifest_s": round(t_man, 4),
                    "tensor_manifest_bytes": nbytes,
                    "candidates": len(ordered),
                },
                tensor_read=tensor_read,
            )
            tensor_read = None  # owned by the returned source
            return source
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            errors.append(f"{inst.worker_id}: {type(exc).__name__}: {exc}"[:300])
        finally:
            # Also on KeyboardInterrupt and friends, which propagate afterwards.
            if tensor_read is not None:
                error = _close_lease(tensor_read)
                if error is not None:
                    errors.append(f"{inst.worker_id}: {error}")
    raise PrestageError(
        f"no usable source for worker_rank {worker_rank}: {errors[:3]}"
    )


def _accelerator_matches(
    accelerator: str, source_accelerator: str, identity: p2p_pb2.SourceIdentity
) -> bool:
    """The authoritative (post-fetch) accelerator check ``RdmaStrategy`` uses."""
    return accelerators_compatible(
        accelerator,
        source_accelerator,
        mx_source_type=identity.mx_source_type,
        quantization=identity.quantization,
        dtype=identity.dtype,
        defer_unknown=False,
    )


def _close_lease(tensor_read: Any) -> str | None:
    """Release a lease and close its channel; returns the error, if any."""
    try:
        tensor_read.close()
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        return f"release tensor read lease: {exc!r}"[:300]
    return None


def plan_reads(
    tensors: list[TensorDescriptor],
    resolve: Callable[[TensorDescriptor], int | None],
) -> ReadPlan:
    """Map catalog entries to local addresses.

    ``resolve(descriptor)`` returns the local address that receives the
    entry's ``size`` bytes, or ``None`` when the caller does not stage that
    entry (for example a buffer the checkpoint kept). It must raise when a
    name maps but its size or dtype does not. Zero-size entries are skipped.

    Names are the source catalog's own: a speculative draft's entries carry
    the ``mx_draft::`` prefix and a ``<name>.__storage`` entry covers a whole
    underlying storage, so a recorded layout must keep those names as they
    are.
    """
    ranges: list[tuple[int, int, int, int]] = []
    unmatched: list[str] = []
    matched = matched_bytes = unmatched_bytes = 0
    for t in tensors:
        if t.size < 0:
            raise PrestageError(f"catalog entry {t.name!r} has a negative size")
        if t.size == 0:
            continue
        local = resolve(t)
        if local is None:
            unmatched.append(t.name)
            unmatched_bytes += int(t.size)
            continue
        ranges.append((int(t.addr), int(local), int(t.size), int(t.device_id)))
        matched += 1
        matched_bytes += int(t.size)
    return ReadPlan(ranges, matched, matched_bytes, unmatched, unmatched_bytes)


class PrestageReceiver:
    """A receive-only NIXL agent over caller-owned CUDA memory regions.

    ``listen_port`` defaults to 0 (an OS-assigned port): a P2P metadata fetch
    (``fetch_remote_metadata``) only queues the request, and the agent's
    communication thread, which NIXL starts together with its listener, sends
    it and processes the reply. Nothing needs to connect to this port.

    One receiver connects to one source. Usage::

        source = discover_source(client, identity, rank, receiver_id="helper-0")
        rcv = PrestageReceiver("helper-rank0", device_id=0)
        try:
            rcv.initialize()
            rcv.register([(base, size)])
            rcv.connect(source)
            rcv.read(plan_reads(source.tensors, resolve))
        finally:
            errors = rcv.close() + source.release()
        # hand the memory over only when errors == []
    """

    def __init__(
        self,
        agent_name: str,
        device_id: int = 0,
        listen_port: int = 0,
        accelerator_backend: Any = None,
    ):
        """Create the manager; nothing touches NIXL until ``initialize()``."""
        self.device_id = int(device_id)
        self.manager = NixlTransferManager(
            agent_name,
            self.device_id,
            listen_port=listen_port,
            accelerator_backend=accelerator_backend,
        )
        self.remote_agent: str | None = None
        self.timings: dict[str, float] = {}
        # Cleanup failures; reported by every close(), never dropped.
        self._cleanup_errors: list[str] = []

    def initialize(self) -> None:
        """Create the NIXL agent and its listener on ``device_id``."""
        t0 = time.perf_counter()
        self.manager.initialize()
        self.timings["agent_s"] = round(time.perf_counter() - t0, 4)

    def register(self, regions: list[tuple[int, int]]) -> None:
        """Register ``(address, size)`` regions of this receiver's CUDA device.

        One memory registration per region; a region may span several
        allocations of one contiguous virtual-address range when the transport
        allows it.
        """
        device = f"cuda:{self.device_id}"
        views: dict[str, torch.Tensor] = {}
        for i, (addr, size) in enumerate(regions):
            view = _raw_u8(addr, size, device)
            if (
                view.data_ptr() != int(addr)
                or view.numel() != int(size)
                or view.device.index != self.device_id
            ):
                raise PrestageError(
                    f"region {i}: the view does not alias the caller's memory"
                )
            views[f"region{i}"] = view
        t0 = time.perf_counter()
        self.manager.register_tensors(views, force_per_tensor=True)
        self.timings["register_s"] = round(time.perf_counter() - t0, 4)
        self.timings["regions"] = len(regions)

    def connect(self, source: PrestageSource, timeout_s: float | None = None) -> None:
        """Load the source agent's NIXL metadata over the P2P socket.

        ``timeout_s`` defaults to ``MX_NIXL_METADATA_TIMEOUT``.
        """
        if self.remote_agent is not None:
            raise PrestageError(f"already connected to {self.remote_agent}")
        if timeout_s is None:
            timeout_s = envs.MX_NIXL_METADATA_TIMEOUT
        host, port = source.metadata_endpoint.rsplit(":", 1)
        t0 = time.perf_counter()
        self.remote_agent = source.agent_name
        try:
            self.manager.fetch_remote_and_wait(
                source.agent_name, host, int(port), timeout_seconds=timeout_s
            )
        except BaseException:
            # Metadata of a fetch that failed part-way can land later, so the
            # peer is removed either way (best effort, see close()).
            self.manager.remove_remote_agent(source.agent_name)
            self.remote_agent = None
            raise
        self.timings["handshake_s"] = round(time.perf_counter() - t0, 4)

    def read(self, plan: ReadPlan, timeout_s: float = 300.0) -> tuple[int, int, float]:
        """One batched READ of every planned range.

        Returns ``(bytes, ranges, seconds)``; the duration includes descriptor
        preparation and the final device synchronization.
        """
        if self.remote_agent is None:
            raise PrestageError("connect() first")
        total, count, duration = self.manager.execute_read_batch(
            self.remote_agent, plan.ranges, timeout_seconds=timeout_s
        )
        self.timings["read_s"] = round(duration, 4)
        self.timings["read_bytes"] = int(total)
        self.timings["read_ranges"] = int(count)
        return total, count, duration

    def close(self) -> list[str]:
        """Disconnect, deregister and drop the agent.

        Returns every error from releasing the caller's memory (deregistration
        and agent shutdown) this receiver has seen; later calls report them
        again. An empty list means all registrations of the caller's memory
        were released, so the memory can be handed over.

        Disconnecting the source is best effort, as in ``RdmaStrategy``: in
        P2P the source never loads this receiver's metadata, so it cannot
        reach the caller's memory, and ``remove_remote_agent`` logs a refusal.
        """
        if self.remote_agent is not None:
            self.manager.remove_remote_agent(self.remote_agent)
            self.remote_agent = None
        self._cleanup_errors += self.manager.deregister_all()
        try:
            self.manager.shutdown()
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            self._cleanup_errors.append(f"shutdown: {exc!r}"[:300])
        return list(self._cleanup_errors)

    def __enter__(self) -> PrestageReceiver:
        """Return the receiver; ``close()`` runs on exit."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Close the receiver and log the cleanup errors.

        A caller that hands the memory over should call ``close()`` itself
        and check its result instead.
        """
        errors = self.close()
        if errors:
            logger.warning("PrestageReceiver cleanup errors: %s", errors)


def _raw_u8(ptr: int, nbytes: int, device: str) -> torch.Tensor:
    """uint8 tensor over raw CUDA memory (no copy, no ownership)."""

    class _CudaArray:
        def __init__(self, p: int, n: int):
            """Describe ``n`` bytes at device address ``p`` (CUDA Array Interface v3)."""
            self.__cuda_array_interface__ = {
                "shape": (int(n),),
                "typestr": "|u1",
                "data": (int(p), False),
                "version": 3,
            }

    return torch.as_tensor(_CudaArray(ptr, nbytes), device=device)
