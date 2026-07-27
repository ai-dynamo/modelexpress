# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Engine-agnostic receiver for the no-gather slice-resharding weight refit.

``ReshardReceiver`` owns everything an inference engine needs to pull a resharded
weight update over NIXL that is NOT engine-specific: build this rank's NIXL agent,
discover + P2P-handshake the trainer shards, capture the model's own load
geometry, build the pull plan, allocate + register the receive/staging buffers,
and per refit RDMA the needed slices in + cast dtype-mismatched sources.

The two engine-specific steps are abstract hooks a subclass implements:

  * :meth:`_capture` - run the engine's ``load_weights`` with zero-storage
    placeholders (on a meta twin for a quantized model) to record where each
    source lands, and report the load-time param layout to size the buffers.
  * :meth:`_install` - install the RDMA'd receive buffers into the live params
    (a plain copy for bf16, or re-quantize via the engine's post-load path).

So an sglang / trtllm receiver only implements those two hooks; discover, plan,
transport, buffers and the router dtype-cast are shared here.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque

import torch

from modelexpress.client import MxClient
from modelexpress.nixl_transfer import NixlTransferManager
from modelexpress.refit.reshard.cuda_pool import classic_cuda_alloc
from modelexpress.refit.reshard.rendezvous import gather_sources
from modelexpress.refit.reshard.transfer_plan import (
    exact_descriptors,
    execute_transfer,
    plan_threshold_curve,
    plan_transfer,
    session_distribution,
)
from modelexpress.refit.reshard.transport import (
    NixlReshardTransport,
    ReadDescriptor,
)
from modelexpress.refit.reshard.types import CaptureResult, UnsupportedReshard
from modelexpress.refit.reshard.verify import (
    FILL_SENTINEL,
    VERIFY,
    VERIFY_STRICT,
    fill_sentinel,
    verify_full_pulls,
)

logger = logging.getLogger("modelexpress.refit.reshard.receiver")

# Batch the per-view re-slice copies of full-pulled sources into one _foreach_copy_.
# Semantically identical to the copy_() loop; set to "0" to fall back.
_BATCH_INSTALL = os.environ.get("MX_RESHARD_BATCH_INSTALL", "1") == "1"
# Benchmarks need the stage split without turning on INFO for every dependency,
# so the record is emitted at WARNING. Set to "0" to silence.
_STAGE_RECORD = os.environ.get("MX_REFIT_STAGE_RECORD", "1") == "1"
# The exact / full-pull / convert reads target disjoint buffers and nothing reads
# those buffers until every phase completes, so they can be posted as one batch
# instead of drained in turn. Set to "0" for the phased path, which is also the
# only way to recover per-phase wire attribution.
_FUSED_WIRE = os.environ.get("MX_RESHARD_FUSED_WIRE", "1") == "1"
# Rotate which byte-identical DP/EDP replica serves each shard, per receiver rank,
# instead of every receiver reading from the first publisher discovered. Off by
# default until the per-session distribution measurement says it is needed.
_SPREAD_SOURCES = os.environ.get("MX_RESHARD_SPREAD_SOURCES", "0") == "1"
# Re-discover every step and diff against the cached plan's view. Diagnostic for
# the cached-plan TODO in update_weights; costs a full discovery round trip per
# step, so it is off unless something is being investigated.
_ADDR_RECHECK = os.environ.get("MX_RESHARD_ADDR_RECHECK", "0") == "1"
# Whole-handshake budget across every peer and every retry. A refit timeout is the
# wrong bound: it lets one unreachable peer consume the entire refit. It must still
# be generous, because a peer can be unreachable for minutes for a legitimate
# reason - registering tens of GB with the EFA provider blocks its listen thread -
# and the receiver's only correct response to that is to keep trying.
_HANDSHAKE_TIMEOUT_S = float(os.environ.get("MX_RESHARD_HANDSHAKE_TIMEOUT_S", "900"))
# Ceiling on a single dial. A reachable peer answers in well under a second, so a
# short attempt costs nothing when things are healthy and, when they are not, frees
# the budget to try a different peer instead of blocking on one.
_HANDSHAKE_ATTEMPT_S = float(os.environ.get("MX_RESHARD_HANDSHAKE_ATTEMPT_S", "20"))
# Pause after a full pass over the pending peers yields no progress, so a transient
# stall is waited out rather than hammered.
_HANDSHAKE_BACKOFF_S = float(os.environ.get("MX_RESHARD_HANDSHAKE_BACKOFF_S", "2"))


def handshake_with_peers(
    manager,
    agent_endpoints: dict,
    total_timeout: float,
    attempt_timeout: float | None = None,
) -> None:
    """Fetch every trainer's NIXL metadata, bounded, retried and logged per peer.

    Three properties, each earned from a failure mode observed on EFA:

    *Bounded overall*, not per peer against the refit timeout. A publisher whose
    process is gone still has its endpoint in the catalog (the reaper only marks it
    stale after a heartbeat lapse, and an abandoned run can keep heartbeating), so
    dialing it blocks. Charging a whole refit timeout to one dead peer hangs the
    refit long past the driver's own deadline.

    *Retried, and deferred rather than fatal on first failure.* A peer can be
    listening yet transiently unable to accept - its accept loop is a thread in a
    process that is busy publishing thousands of tensors, and a listen backlog that
    never drains silently drops SYNs. That is indistinguishable from a dead peer
    within a single dial, but not across several seconds, so a failed peer goes to
    the back of the queue and the next one is tried instead of aborting the refit.

    *Logged per peer.* Without it the last line in the log is "P2P-fetching remote
    metadata" and there is no way to tell which peer is at fault, or whether it
    stalled on the first or the last.
    """
    attempt_timeout = attempt_timeout or _HANDSHAKE_ATTEMPT_S
    pending = deque(agent_endpoints.items())
    total = len(pending)
    attempts: dict = {name: 0 for name in agent_endpoints}
    last_error: dict = {}
    deadline = time.monotonic() + total_timeout
    succeeded = 0
    # Consecutive failures with no success in between; one full pass over the
    # pending peers without progress means waiting is better than spinning.
    stalled = 0

    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            outstanding = ", ".join(
                f"{name}@{endpoint} ({attempts[name]} attempt(s), last: "
                f"{type(last_error.get(name)).__name__}: {last_error.get(name)})"
                for name, endpoint in pending
            )
            raise RuntimeError(
                f"[reshard] P2P handshake incomplete after {total_timeout:.0f}s: "
                f"{succeeded} of {total} peer(s) answered. Outstanding: "
                f"{outstanding}. These publishers are advertised in the MX catalog "
                f"but did not answer - either the process is gone while something "
                f"still heartbeats its source, or its NIXL listen thread is not "
                f"accepting."
            )

        agent_name, endpoint = pending.popleft()
        host, port_str = endpoint.rsplit(":", 1)
        this_timeout = max(1.0, min(attempt_timeout, remaining))
        attempts[agent_name] += 1
        logger.info(
            "[reshard] _prepare: handshake %d/%d %s at %s (attempt %d, timeout=%.0fs)",
            succeeded + 1,
            total,
            agent_name,
            endpoint,
            attempts[agent_name],
            this_timeout,
        )
        started = time.perf_counter()
        try:
            manager.fetch_remote_and_wait(
                agent_name, host, int(port_str), timeout_seconds=this_timeout
            )
        except Exception as exc:
            last_error[agent_name] = exc
            logger.warning(
                "[reshard] _prepare: handshake %s at %s failed after %.1fs on "
                "attempt %d (%s: %s); deferring, %d peer(s) still pending",
                agent_name,
                endpoint,
                time.perf_counter() - started,
                attempts[agent_name],
                type(exc).__name__,
                exc,
                len(pending) + 1,
            )
            pending.append((agent_name, endpoint))
            stalled += 1
            if stalled >= len(pending):
                time.sleep(min(_HANDSHAKE_BACKOFF_S, max(0.0, deadline - time.monotonic())))
                stalled = 0
            continue

        succeeded += 1
        stalled = 0
        last_error.pop(agent_name, None)
        logger.info(
            "[reshard] _prepare: handshake %d/%d %s ok in %.2fs (attempt %d)",
            succeeded,
            total,
            agent_name,
            time.perf_counter() - started,
            attempts[agent_name],
        )

    retried = {n: c for n, c in attempts.items() if c > 1}
    if retried:
        logger.warning(
            "[reshard] _prepare: handshake completed with retries: %s", retried
        )


def _replay_ops(tensor: torch.Tensor, op_chain: tuple) -> torch.Tensor:
    """Replay a captured loader view chain on a staged full-source tensor."""
    value = tensor
    for op_name, args, frozen_kwargs in op_chain:
        kwargs = dict(frozen_kwargs)
        if op_name == "__getitem__":
            value = value.__getitem__(*args)
        else:
            value = getattr(value, op_name)(*args, **kwargs)
    return value


class ReshardReceiver:
    """Pull-mode slice-resharding weight receiver (engine-agnostic).

    Lifecycle: construct once (builds the NIXL agent + metadata client), then
    call :meth:`update_weights` per weight update. The first call lazily discovers the
    trainer shards, captures geometry, and builds the plan + buffers (cached);
    every refit re-reads the same trainer buffer addresses (now holding the
    step's refreshed weights).
    """

    def __init__(
        self,
        *,
        model_name: str,
        mx_server: str,
        agent_name: str,
        local_rank: int,
        global_rank: int,
        num_trainer_sources: int,
        device: "torch.device",
        listen_port: int,
        timeout: float = 1200.0,
    ) -> None:
        """Build this rank's NIXL agent + metadata client.

        Args:
            model_name: the served model name (the shared ``[model] name`` both
                trainer and inference inherit) - the rendezvous identity key.
            mx_server: ``host:port`` of the modelexpress metadata server.
            agent_name: this rank's NIXL agent name.
            local_rank: device index (the NIXL device id).
            global_rank: rendezvous rank (``rank_offset + local_rank``).
            num_trainer_sources: number of trainer ranks publishing shards (all
                must be discovered before planning, since a slice can fan in
                across ranks).
            device: the torch device receive buffers are allocated on.
            listen_port: NIXL listen port for this rank's agent. The receiver
                needs a listen thread (MX's P2P metadata exchange is
                bidirectional); the caller owns port assignment so it can avoid
                colliding with a colocated trainer publisher (which listens on
                ``MX_METADATA_PORT + device_id``).
            timeout: rendezvous / per-pull timeout seconds.
        """
        self._device = device
        self._model_name = model_name
        self._num_trainer_sources = num_trainer_sources
        self._timeout = timeout
        self._global_rank = global_rank

        # TODO(transport-agnostic): the receiver is engine-agnostic but still
        # transport-bound to NIXL (this manager, NixlReshardTransport, and the
        # fetch_remote_and_wait P2P handshake in _prepare). Abstract these behind
        # a transport interface so non-NIXL backends can plug in.
        self._manager = NixlTransferManager(
            agent_name=agent_name, device_id=local_rank, listen_port=listen_port
        )
        self._manager.initialize()
        self._mx_client = MxClient(server_url=mx_server)

        self._plan = None  # built lazily on the first refit
        self._sources: dict = {}  # src_name -> SourceInfo, kept for the verify gate
        self._transport: NixlReshardTransport | None = None
        self._recv_buffers: dict[
            str, torch.Tensor
        ] = {}  # param_name -> receive buffer at load-time layout
        self._staging: dict[
            str, torch.Tensor
        ] = {}  # dtype-convert param -> bf16 staging (RDMA target)
        self._staging_ptr: dict[str, int] = {}
        self._full_staging: dict[str, torch.Tensor] = {}
        self._full_staging_ptr: dict[str, int] = {}
        self._param_ptr: dict[
            str, int
        ] = {}  # segment param_name -> receive-buffer data_ptr
        # One-time _prepare() costs, folded into the first refit's stage record so
        # the cold step is attributed instead of appearing as one opaque block.
        self._prepare_stages: dict[str, float] = {}

        logger.info(
            "[reshard] receiver init: agent=%s global_rank=%d trainer_sources=%d",
            agent_name,
            global_rank,
            num_trainer_sources,
        )

    # ------------------------------------------------------------- engine hooks
    def _capture(self, manifest: list) -> "tuple[CaptureResult, dict]":
        """Record where each published source lands in the engine's load-time
        param layout, without moving data.

        Returns ``(capture, param_layout)`` where ``param_layout`` is
        ``{param_name: (shape, dtype)}`` at the LOAD-TIME layout (bf16, pre-quant)
        - used to size the receive buffers. For a quantized model this is captured
        on a fresh meta twin (the live params are post-quantization); for a bf16
        model it may be the live model directly."""
        raise NotImplementedError

    def _install(self, recv_buffers: dict) -> None:
        """Install the RDMA'd receive buffers into the live params.

        For a bf16 model this is effectively making the buffers the live params;
        for a quantized model it re-runs the engine's post-load processing
        (quantize + derive) with the buffers as the load-time params. Must be
        CUDA-graph-safe (write into the graph-bound storage)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ prepare
    def _prepare(self, timeout: float) -> None:
        """One-time: discover trainer shards, connect their agents, capture load
        geometry, build the pull plan, and allocate + register buffers."""
        # _prepare runs once per receiver process and dominates the cold refit, so
        # every phase is timed and folded into the first stage record. Spans are
        # sequential and non-overlapping, so they may be summed.
        self._prepare_stages = {}
        logger.info(
            "[reshard] _prepare: discovering %d trainer source(s) (timeout=%.0fs)",
            self._num_trainer_sources,
            timeout,
        )
        # With MX_RESHARD_SPREAD_SOURCES, rotate which duplicate DP/EDP replica
        # serves each shard by this rank, so the fleet's reads spread over the
        # replicas instead of every receiver hitting the same publisher.
        replica_offset = self._global_rank if _SPREAD_SOURCES else 0
        _t = time.perf_counter()
        sources, session_to_agent, session_to_device, agent_endpoints = gather_sources(
            self._mx_client,
            expected_trainers=self._num_trainer_sources,
            model_name=self._model_name,
            role="inference",
            rank=self._global_rank,
            timeout=timeout,
            replica_offset=replica_offset,
        )
        # Includes waiting for trainers to publish, so this is partly the
        # trainer's readiness rather than a receiver-side cost.
        self._prepare_stages["prepare_discover_s"] = time.perf_counter() - _t
        # Retained for the verify gate, which needs each shard's published digest
        # and its box within the full tensor after the plan has been built.
        self._sources = sources
        logger.info(
            "[reshard] _prepare: discovered %d source(s), %d agent(s); P2P-fetching remote metadata",
            len(sources),
            len(agent_endpoints),
        )
        # P2P memory handshake (mirrors MX's vLLM RDMA path): fetch each trainer's
        # NIXL metadata (incl. its memory registrations) via its listen thread, so
        # prep_xfer_dlist can resolve the remote addresses. The central
        # add_remote_agent(blob) path does NOT convey the registrations.
        # Serial: each peer's metadata is fetched and waited for before the next
        # one starts, so this scales with the trainer count.
        #
        # Bounded per peer, NOT by the global refit timeout. A publisher whose
        # process is gone still has its endpoint in the catalog (the reaper only
        # marks it stale after a heartbeat lapse, and an abandoned run can keep
        # heartbeating), so dialing it blocks. Charging the full refit timeout to a
        # single dead peer hangs the whole refit long past the driver's own
        # deadline, and - with no per-agent logging - does so silently: the last
        # thing in the log is "P2P-fetching remote metadata" and there is no way to
        # tell which of the 16 peers is at fault, or whether it stalled on the
        # first or the last.
        _t = time.perf_counter()
        handshake_with_peers(
            self._manager,
            agent_endpoints,
            total_timeout=min(timeout, _HANDSHAKE_TIMEOUT_S),
        )
        self._prepare_stages["prepare_handshake_s"] = time.perf_counter() - _t

        manifest = [
            (name, src.dtype, tuple(src.global_shape)) for name, src in sources.items()
        ]
        logger.info(
            "[reshard] _prepare: capturing geometry over %d manifest entries",
            len(manifest),
        )
        _t = time.perf_counter()
        capture, param_layout = self._capture(manifest)
        self._prepare_stages["prepare_capture_s"] = time.perf_counter() - _t
        logger.info(
            "[reshard] _prepare: captured %d copies, %d unsupported",
            len(capture.copies),
            len(capture.unsupported),
        )

        # The plan encodes THIS discovery's topology: each trainer's registered
        # buffer addresses, per-source shard boundaries, and fan-in across ranks.
        # It is built once and reused every step (see the guard in
        # update_weights), which assumes the trainer set + their shard layout +
        # their buffer addresses are stable for the run.
        _t = time.perf_counter()
        plan = plan_transfer(capture, sources)
        self._prepare_stages["prepare_plan_s"] = time.perf_counter() - _t
        self._log_session_distribution(plan)
        # Timed separately so an opt-in sweep never inflates prepare_plan_s.
        _t = time.perf_counter()
        self._log_threshold_curve(capture, sources)
        sweep_s = time.perf_counter() - _t
        if sweep_s:
            self._prepare_stages["prepare_sweep_s"] = sweep_s
        if plan.fallback:
            # Fallback params are dropped from the RDMA plan and never pulled or
            # installed, so they would silently keep their initial (base-model)
            # weights for the entire run. Until the full-pull/loader path exists
            # (TODO), fail loudly rather than serve stale weights.
            raise UnsupportedReshard(
                f"[reshard] {len(plan.fallback)} source(s) need the unimplemented "
                f"full-pull path (unsupported reshard ops); refusing to serve stale "
                f"weights. Params: {plan.fallback[:10]}"
            )
        self._transport = NixlReshardTransport(
            self._manager, session_to_agent, session_to_device, timeout_seconds=timeout
        )
        self._plan = plan

        # dtype-mismatched sources (e.g. a bf16-served router for an fp32 dest):
        # one persistent bf16 STAGING buffer per convert param, registered as an
        # RDMA target (classic cudaMalloc so the HCA can RDMA into it); each refit
        # we RDMA into staging then cast staging -> the (load-time) receive buffer.
        # Allocation and registration are accumulated separately across all three
        # buffer groups: registration is already batched (one call per group, not
        # one per tensor), so separating them says whether that batching is
        # already sufficient or the cold cost lives elsewhere.
        alloc_s = 0.0
        register_s = 0.0

        self._staging = {}
        self._staging_ptr = {}
        if plan.converts:
            _t = time.perf_counter()
            with classic_cuda_alloc():
                self._staging = {
                    c.param_name: torch.empty(
                        c.dest_shape, dtype=c.src_dtype, device=self._device
                    )
                    for c in plan.converts
                }
            alloc_s += time.perf_counter() - _t
            _t = time.perf_counter()
            self._manager.register_tensors(
                {f"__stage__{n}": t for n, t in self._staging.items()}
            )
            register_s += time.perf_counter() - _t
            self._staging_ptr = {n: t.data_ptr() for n, t in self._staging.items()}

        # Descriptor-heavy strided copies pull each complete source into one
        # persistent contiguous staging tensor, then replay captured loader views
        # locally. Each source shard contributes one bounded descriptor.
        self._full_staging = {}
        self._full_staging_ptr = {}
        if plan.full_pulls:
            _t = time.perf_counter()
            with classic_cuda_alloc():
                self._full_staging = {
                    full_pull.src_name: torch.empty(
                        full_pull.global_shape,
                        dtype=full_pull.dtype,
                        device=self._device,
                    )
                    for full_pull in plan.full_pulls
                }
            alloc_s += time.perf_counter() - _t
            _t = time.perf_counter()
            self._manager.register_tensors(
                {
                    f"__full__{name}": tensor
                    for name, tensor in self._full_staging.items()
                }
            )
            register_s += time.perf_counter() - _t
            self._full_staging_ptr = {
                name: tensor.data_ptr() for name, tensor in self._full_staging.items()
            }

        # Receive buffers: one per captured param at its CAPTURED (load-time)
        # shape/dtype, classic cudaMalloc, registered once. The live params are
        # NOT RDMA targets; _install() writes the buffers into the live params.
        # Segment params (captured == served) are the RDMA targets - register them
        # + point _param_ptr at them. Convert params (router) are captured fp32 ->
        # their bf16 staging is the RDMA target and the refit casts into the buffer.
        all_params = sorted({c.param_name for c in capture.copies})
        seg_params = {seg.param_name for seg in plan.segments}
        self._recv_buffers = {}
        _t = time.perf_counter()
        with classic_cuda_alloc():
            for name in all_params:
                shape, dtype = param_layout[name]
                self._recv_buffers[name] = torch.empty(
                    tuple(shape), dtype=dtype, device=self._device
                )
        alloc_s += time.perf_counter() - _t
        self._param_ptr = {}
        if seg_params:
            _t = time.perf_counter()
            self._manager.register_tensors(
                {f"__recv__{n}": self._recv_buffers[n] for n in seg_params}
            )
            register_s += time.perf_counter() - _t
            for name in seg_params:
                self._param_ptr[name] = self._recv_buffers[name].data_ptr()

        self._prepare_stages["prepare_alloc_s"] = alloc_s
        self._prepare_stages["prepare_register_s"] = register_s

        logger.info(
            "[reshard] prepared: %d descriptor(s), %d full-pull source(s), "
            "%d convert(s), %.1f MB/pull, %d descriptor(s) saved, "
            "%.1f MB extra wire, %d unbounded source(s), %d fallback",
            plan.descriptor_count(),
            len(plan.full_pulls),
            len(plan.converts),
            plan.bytes_planned() / 1e6,
            plan.descriptor_savings(),
            plan.extra_wire_bytes() / 1e6,
            len(plan.unbounded_sources),
            len(plan.fallback),
        )

    def _log_session_distribution(self, plan) -> None:
        """Report how this receiver's reads spread over the publishing ranks.

        Emitted at WARNING as JSON so a benchmark harness can recover it without
        enabling INFO across every dependency. Cheap: one pass over an
        already-built plan, once per receiver process.
        """
        dist = session_distribution(plan)
        if not dist:
            return
        loads = sorted((v["bytes"] for v in dist.values()), reverse=True)
        total = sum(loads)
        record = {
            "sessions": len(dist),
            "total_bytes": total,
            "max_session_bytes": loads[0],
            "min_session_bytes": loads[-1],
            # 1.0 means perfectly even; N means one session carries everything.
            "imbalance": round(loads[0] / (total / len(loads)), 3),
            "per_session": {
                s: {"bytes": v["bytes"], "descriptors": v["descriptors"]}
                for s, v in sorted(dist.items())
            },
        }
        logger.warning(
            "[reshard] session-distribution %s", json.dumps(record, sort_keys=True)
        )

    def _log_threshold_curve(self, capture: CaptureResult, sources: dict) -> None:
        """Report the descriptor/byte tradeoff across full-pull thresholds.

        Off unless ``MX_RESHARD_PLAN_SWEEP`` names thresholds (e.g.
        ``64,256,1024,4096``), because each one costs an extra throwaway plan
        build. Emitted at WARNING so it survives Dynamo/vLLM log filtering, and
        as JSON so a benchmark run can parse the whole curve out of one cold
        start rather than one redeployment per threshold.
        """
        raw = os.environ.get("MX_RESHARD_PLAN_SWEEP", "").strip()
        if not raw:
            return
        try:
            thresholds = [int(t) for t in raw.split(",") if t.strip()]
        except ValueError:
            logger.warning(
                "[reshard] MX_RESHARD_PLAN_SWEEP=%r is not a comma-separated "
                "list of integers; skipping the sweep",
                raw,
            )
            return
        for row in plan_threshold_curve(capture, sources, thresholds):
            logger.warning("[reshard] plan-sweep %s", json.dumps(row, sort_keys=True))

    # ----------------------------------------------------------- update_weights
    @torch.no_grad()
    def _recheck_sources(self, step: int, timeout: float) -> None:
        """Re-discover and diff this step's publication against the cached one.

        Diagnostic for the cached-plan TODO below. ``_prepare`` runs once, so both
        the source *addresses* the wire reads and the source *digests* the verify
        gate compares against are frozen at the first refit. Those two staleness
        bugs look identical from the report - a mismatch either way - but they are
        opposites:

        * addresses changed -> the wire is reading the wrong memory and installing
          genuinely wrong weights. A correctness bug in the refit.
        * addresses stable, digests changed -> the wire read the right memory and
          delivered current weights, and the gate is comparing them against the
          first step's digests. A bug in the gate, not in the refit.

        So this counts both rather than assuming either.
        """
        replica_offset = self._global_rank if _SPREAD_SOURCES else 0
        _t = time.perf_counter()
        fresh, _agents, _devices, _endpoints = gather_sources(
            self._mx_client,
            expected_trainers=self._num_trainer_sources,
            model_name=self._model_name,
            role="inference",
            rank=self._global_rank,
            timeout=timeout,
            replica_offset=replica_offset,
        )
        # Keyed by (session, box), never by position. Discovery order is not
        # stable, and merge_shard_tables keeps the first offer of each geometry,
        # so two discoveries legitimately pin the same box to different - but
        # byte-identical - replicas. Comparing positionally therefore reports a
        # replica reshuffle as an address change: the first version of this probe
        # did exactly that and claimed up to 100% of addresses had moved, which
        # cannot be true or every shard would mismatch rather than 19% of them.
        compared = addr_changed = digest_changed = both = 0
        reselected = unmatched = 0
        for name, cached in self._sources.items():
            current = fresh.get(name)
            if current is None:
                unmatched += len(cached.shards)
                continue
            by_key = {
                (s.session, tuple(s.shard_offset)): s for s in current.shards
            }
            for old in cached.shards:
                new = by_key.get((old.session, tuple(old.shard_offset)))
                if new is None:
                    # That box is now served by a different rank; the pinned
                    # publisher is no longer offering it.
                    reselected += 1
                    continue
                compared += 1
                moved = old.addr != new.addr
                redigested = (
                    old.digest is not None
                    and new.digest is not None
                    and old.digest != new.digest
                )
                addr_changed += moved
                digest_changed += redigested
                both += moved and redigested
        logger.warning(
            "MX_REFIT_ADDR_RECHECK %s",
            json.dumps(
                {
                    "schema": "refit-addr-recheck-v2",
                    "step": step,
                    "compared": compared,
                    "addr_changed": addr_changed,
                    "digest_changed": digest_changed,
                    "addr_and_digest_changed": both,
                    "reselected_to_other_rank": reselected,
                    "unmatched_sources": unmatched,
                    "recheck_s": round(time.perf_counter() - _t, 3),
                }
            ),
        )

    def update_weights(self, step: int, *, timeout: float | None = None) -> dict:
        """RDMA-pull the needed slices into the receive buffers, cast the
        dtype-mismatched ones, then install into the live params."""
        timeout = timeout if timeout is not None else self._timeout
        # TODO(re-plan on topology change): the plan is built once and cached, so
        # a mid-run change in the trainer set - a trainer restart (new buffer
        # addresses), a reshard (new shard boundaries / fan-in), or scaling the
        # trainer count - is NOT picked up; every step re-reads the first
        # discovery's addresses. Adapt the plan when topology changes (e.g.
        # re-discover + rebuild if a version/epoch token or address set differs).
        # Stage spans are sequential and non-overlapping, so they may be summed.
        # Each GPU-work span ends with an explicit sync, otherwise the async
        # launches would be attributed to whichever stage happens to sync next.
        stages: dict[str, float] = {}

        if self._plan is None:
            self._prepare(timeout)
            # One-time setup, so it lands only in the cold step's record. Without
            # this the cold step attributes ~2% of its own duration.
            stages.update(self._prepare_stages)
        assert self._plan is not None and self._transport is not None

        if _ADDR_RECHECK:
            self._recheck_sources(step, timeout)

        # RDMA the sliced bf16 into the receive buffers (segments) and per-param
        # staging (dtype-convert / router). No live param is written by RDMA.
        #
        # The three read phases target disjoint destinations - exact segments land
        # in the receive buffers, full pulls in full staging, converts in convert
        # staging - and every reader of those buffers (reslice, dtype cast) runs
        # after all reads complete. So they carry no ordering dependency and can be
        # issued as one batch, which is what _FUSED_WIRE does. Phased mode drains
        # each in turn and is kept for the A/B and for per-phase attribution.
        if FILL_SENTINEL and self._full_staging:
            # Before any descriptor is issued, so anything still holding the
            # sentinel afterwards was not written by this step's wire.
            _t = time.perf_counter()
            fill_sentinel(self._full_staging)
            torch.cuda.synchronize(self._device)
            stages["sentinel_fill_s"] = time.perf_counter() - _t

        full_descriptors = [
            ReadDescriptor(
                session=segment.session,
                src_addr=segment.src_addr,
                dst_addr=(self._full_staging_ptr[full_pull.src_name] + segment.dst_byte),
                nbytes=segment.nbytes,
            )
            for full_pull in self._plan.full_pulls
            for segment in full_pull.segments
        ]
        convert_descriptors = [
            ReadDescriptor(
                session=segment.session,
                src_addr=segment.src_addr,
                dst_addr=self._staging_ptr[convert.param_name] + segment.dst_byte,
                nbytes=segment.nbytes,
            )
            for convert in self._plan.converts
            for segment in convert.segments
        ]

        if _FUSED_WIRE:
            descriptors = exact_descriptors(
                self._plan, lambda name: self._param_ptr[name]
            )
            stats = {
                "segments": len(descriptors),
                "bytes": sum(d.nbytes for d in descriptors),
                "fallback": list(self._plan.fallback),
            }
            _t = time.perf_counter()
            self._transport.read(descriptors + full_descriptors + convert_descriptors)
            stages["wire_fused_s"] = time.perf_counter() - _t
        else:
            _t = time.perf_counter()
            stats = execute_transfer(
                self._plan,
                resolve_param_ptr=lambda name: self._param_ptr[name],
                transport=self._transport,
            )
            stages["wire_exact_s"] = time.perf_counter() - _t
            if full_descriptors:
                _t = time.perf_counter()
                self._transport.read(full_descriptors)
                stages["wire_full_s"] = time.perf_counter() - _t
            if convert_descriptors:
                _t = time.perf_counter()
                self._transport.read(convert_descriptors)
                stages["wire_convert_s"] = time.perf_counter() - _t

        stats["segments"] += len(full_descriptors) + len(convert_descriptors)
        stats["bytes"] += sum(
            d.nbytes for d in full_descriptors + convert_descriptors
        )

        if self._plan.full_pulls:
            # Local re-slice of every full-pulled source. One copy_() per captured
            # view means thousands of individual kernel launches, whose Python and
            # launch overhead can rival the RDMA itself; _foreach_copy_ issues the
            # same copies as a single batched op.
            _t = time.perf_counter()
            destinations = []
            source_views = []
            for full_pull in self._plan.full_pulls:
                full_tensor = self._full_staging[full_pull.src_name]
                for copy in full_pull.copies:
                    source_view = _replay_ops(full_tensor, copy.op_chain)
                    receive_buffer = self._recv_buffers[copy.param_name]
                    destination = receive_buffer.as_strided(
                        copy.dest_shape,
                        copy.dest_stride,
                        receive_buffer.storage_offset() + copy.dest_offset,
                    )
                    if _BATCH_INSTALL:
                        destinations.append(destination)
                        source_views.append(source_view)
                    else:
                        destination.copy_(source_view)
            if _BATCH_INSTALL and destinations:
                torch._foreach_copy_(destinations, source_views)
            torch.cuda.synchronize(self._device)
            stages["reslice_s"] = time.perf_counter() - _t
            stages["reslice_copies"] = float(len(self._plan.full_pulls))

        if self._plan.converts:
            # Cast the served bf16 staging into the (fp32) receive buffer - a torch
            # op, so the RDMA never crosses dtypes. _install writes the buffer.
            _t = time.perf_counter()
            for convert in self._plan.converts:
                self._recv_buffers[convert.param_name].copy_(
                    self._staging[convert.param_name]
                )
            torch.cuda.synchronize(self._device)
            stages["convert_s"] = time.perf_counter() - _t

        # Before install, while the staging buffers still hold exactly what the
        # wire delivered. Installing first would leave a mismatch ambiguous between
        # a bad transfer and a bad install.
        verify_report = None
        if VERIFY and self._plan.full_pulls:
            _t = time.perf_counter()
            verify_report = verify_full_pulls(
                full_staging=self._full_staging, sources=self._sources
            )
            stages["verify_s"] = time.perf_counter() - _t

        _t = time.perf_counter()
        self._install(self._recv_buffers)
        torch.cuda.synchronize(self._device)
        stages["install_s"] = time.perf_counter() - _t

        if verify_report is not None:
            # WARNING so a benchmark harness captures it alongside the stage record.
            logger.warning(
                "MX_REFIT_VERIFY %s",
                json.dumps({"schema": "refit-verify-v1", "step": step, **verify_report}),
            )
            if verify_report["mismatches"]:
                raise RuntimeError(
                    f"[reshard] parameter verification FAILED at step {step}: "
                    f"{verify_report['mismatches']} of {verify_report['checked']} "
                    f"checked shard(s) differ from the publisher's digest. The bytes "
                    f"this rank pulled are not the bytes the trainer holds, so the "
                    f"engine would now generate from wrong weights. First: "
                    f"{verify_report['detail'][:3]}"
                )
            if not verify_report["checked"]:
                message = (
                    f"[reshard] MX_RESHARD_VERIFY is on but 0 of "
                    f"{verify_report['skipped_no_digest']} shard(s) carried a "
                    f"publisher digest, so this refit is UNVERIFIED. The publishers "
                    f"are older than this receiver: either they predate the digest "
                    f"entirely, or they were not started with MX_RESHARD_VERIFY=1. "
                    f"Zero mismatches here means no evidence, not a correctness pass."
                )
                if VERIFY_STRICT:
                    raise RuntimeError(message)
                logger.warning("%s", message)

        metrics = {
            "step": step,
            "bytes_received": stats["bytes"],
            "segments": stats["segments"],
            "converts": len(self._plan.converts),
            "full_pull_sources": len(self._plan.full_pulls),
            "exact_descriptors": self._plan.exact_descriptor_count,
            "descriptor_savings": self._plan.descriptor_savings(),
            "extra_wire_bytes": self._plan.extra_wire_bytes(),
            "unbounded_sources": len(self._plan.unbounded_sources),
            "fallback": len(stats["fallback"]),
        }
        logger.info(
            "[reshard] refit step=%d bytes=%.1fMB descriptors=%d "
            "(saved=%d, extra_wire=%.1fMB) full_pulls=%d converts=%d "
            "unbounded=%d fallback=%d",
            step,
            stats["bytes"] / 1e6,
            stats["segments"],
            self._plan.descriptor_savings(),
            self._plan.extra_wire_bytes() / 1e6,
            len(self._plan.full_pulls),
            len(self._plan.converts),
            len(self._plan.unbounded_sources),
            len(stats["fallback"]),
        )
        metrics.update({k: round(v, 6) for k, v in stages.items()})
        if _STAGE_RECORD:
            accounted = sum(
                v for k, v in stages.items() if k.endswith("_s")
            )
            record = {
                "schema": "refit-stage-v2",
                "step": step,
                "bytes": stats["bytes"],
                "segments": stats["segments"],
                "batch_install": _BATCH_INSTALL,
                "accounted_s": round(accounted, 6),
                **{k: round(v, 6) for k, v in stages.items()},
            }
            # WARNING so benchmark harnesses capture it without enabling INFO
            # across every dependency.
            logger.warning("MX_REFIT_STAGE %s", json.dumps(record))
        return metrics
