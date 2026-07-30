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
from modelexpress.refit.reshard.dest_digest_report import (
    RECORD_MARKER as _DEST_DIGEST_MARKER,
)
from modelexpress.refit.reshard.dest_digest_report import dest_digest_record
from modelexpress.refit.reshard.rendezvous import (
    gather_sources,
    gather_sources_with_steps,
)
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
    DEST_DIGEST,
    EXACT_REPLAY,
    FILL_SENTINEL,
    VERIFY,
    VERIFY_STRICT,
    compare_exact_replay,
    digest_destination,
    exact_replay_digests,
    fill_sentinel,
    source_expectation_digests,
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
# Per-rank fabric ceiling in Gbps. A refit that beats it did not transfer (Bug 10).
# Zero disables the check: only the operator knows the real limit for their fabric.
_MAX_GBPS = float(os.environ.get("MX_RESHARD_MAX_GBPS", "0") or 0)
# Rotate which byte-identical DP/EDP replica serves each shard, per receiver rank,
# instead of every receiver reading from the first publisher discovered. Off by
# default until the per-session distribution measurement says it is needed.
_SPREAD_SOURCES = os.environ.get("MX_RESHARD_SPREAD_SOURCES", "0") == "1"
# Re-discover every step and diff against the cached plan's view. Diagnostic for
# the cached-plan TODO in update_weights; costs a full discovery round trip per
# step, so it is off unless something is being investigated.
_ADDR_RECHECK = os.environ.get("MX_RESHARD_ADDR_RECHECK", "0") == "1"
# Verify against digests from a fresh discovery rather than the ones captured at
# prepare time. On by default because the prepare-time digests are simply wrong for
# any step past the first; the escape hatch exists to reproduce the old behaviour.
_VERIFY_FRESH_DIGESTS = (
    os.environ.get("MX_RESHARD_VERIFY_FRESH_DIGESTS", "1") not in ("0", "false", "False")
)
# Refuse a refit that does not cover the engine's parameter bytes. Off by default
# because partial and subset refit are intended features; benchmark harnesses must
# turn it on. It is the only gate that can see a param the loader never asked for:
# every other check compares bytes that arrived against the publisher's digest for
# the same name, so a tensor that is never requested is never checked.
_REQUIRE_FULL_COVERAGE = (
    os.environ.get("MX_RESHARD_REQUIRE_FULL_COVERAGE", "0") == "1"
)
# Not 1.0: a handful of engine params are legitimately not refit material (rotary
# inv_freq and similar non-float buffers that surface as params in some models),
# and failing a complete refit over a few kilobytes would make the gate unusable.
_COVERAGE_FLOOR = float(os.environ.get("MX_RESHARD_COVERAGE_FLOOR", "0.995"))
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
    def _prepare(self, timeout: float, step: int | None = None) -> None:
        """One-time: discover trainer shards, connect their agents, capture load
        geometry, build the pull plan, and allocate + register buffers.

        ``step`` only labels the publisher step stamps recorded here, so that the later
        fresh discovery in the same refit compares against the previous refit rather
        than against this one."""
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
        (
            sources,
            session_to_agent,
            session_to_device,
            agent_endpoints,
            session_to_step,
        ) = gather_sources_with_steps(
            self._mx_client,
            expected_trainers=self._num_trainer_sources,
            model_name=self._model_name,
            role="inference",
            rank=self._global_rank,
            timeout=timeout,
            replica_offset=replica_offset,
        )
        # Baseline for the staleness delta. Without one recorded here, the first
        # re-discovery has nothing to compare against - and that is step 2, the exact
        # step where the receiver was seen to read the previous step's table.
        self._note_publisher_steps(session_to_step, step=step)
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
        self._log_coverage(capture, param_layout, all_params, plan)

    def _log_coverage(self, capture, param_layout, all_params, plan) -> None:
        """Report what this rank asked the wire for, against what it will install.

        Emitted at WARNING as JSON. Everything here was already computed and
        already logged - at INFO, which no benchmark run has ever captured, so
        `useful_bytes_per_rank` has been *derived* on the analysis side rather
        than measured. That derivation is what left Topology B unreconcilable:
        it moves 330 GB across 16 ranks against a first-principles need of
        488 GB, and with no measured destination footprint there is no way to
        tell whether the model of the sharding is wrong or the refit is
        incomplete.

        `unsupported` is the number that answers it. A param the loader wants
        and the planner cannot serve is silently absent from the wire, so a
        non-zero count here is a coverage hole - the correctness gate cannot see
        it, because the gate verifies bytes that arrived and says nothing about
        bytes that never did.
        """
        # `param_layout` is the engine's COMPLETE parameter set; `all_params` is the
        # subset this refit will write. The ratio is the coverage nothing else
        # measures, and it needs no engine-specific hook.
        installed = set(all_params)
        dest_bytes = 0
        engine_bytes = 0
        missed: list[str] = []
        for name, (shape, dtype) in param_layout.items():
            count = 1
            for dim in shape:
                count *= int(dim)
            nbytes = count * torch.empty(0, dtype=dtype).element_size()
            engine_bytes += nbytes
            if name in installed:
                dest_bytes += nbytes
            else:
                missed.append(name)
        coverage = (dest_bytes / engine_bytes) if engine_bytes else 0.0
        unsupported = list(getattr(capture, "unsupported", []) or [])
        record = {
            "schema": "refit-coverage-v1",
            "rank": self._global_rank,
            "params_installed": len(all_params),
            "engine_params": len(param_layout),
            "dest_bytes": dest_bytes,
            "engine_bytes": engine_bytes,
            "coverage_pct": round(100.0 * coverage, 4),
            "params_never_written": len(missed),
            "params_never_written_sample": sorted(missed)[:10],
            "copies_captured": len(capture.copies),
            "unsupported": len(unsupported),
            "unsupported_sample": [str(u)[:120] for u in unsupported[:10]],
            "planned_wire_bytes": plan.bytes_planned(),
            "extra_wire_bytes": plan.extra_wire_bytes(),
            "descriptors": plan.descriptor_count(),
            "descriptor_savings": plan.descriptor_savings(),
            "full_pull_sources": len(plan.full_pulls),
            "unbounded_sources": len(plan.unbounded_sources),
            "converts": len(plan.converts),
            "fallback": len(plan.fallback),
        }
        logger.warning("MX_REFIT_COVERAGE %s", json.dumps(record))

        if _REQUIRE_FULL_COVERAGE and coverage < _COVERAGE_FLOOR:
            # Opt-in rather than always-on: partial and subset refit are intended
            # features, and for those a coverage below 1.0 is the point. What is
            # never acceptable is a *benchmark* row measuring an incomplete refit,
            # because its wire volume and timings are then the wrong magnitude and
            # get compared against complete ones. Topology B was published as
            # beating Topology A on every axis while refitting 51% of the model.
            raise RuntimeError(
                f"refit covers {100.0 * coverage:.2f}% of the engine's parameter "
                f"bytes ({dest_bytes} of {engine_bytes}); "
                f"{len(missed)} of {len(param_layout)} params would keep their "
                f"previous values, e.g. {sorted(missed)[:5]}. No digest gate can "
                f"detect this - bytes that are never requested are never checked. "
                f"Set MX_RESHARD_REQUIRE_FULL_COVERAGE=0 for an intentionally "
                f"partial refit."
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
    def _fresh_sources(self, timeout: float, step: int | None = None) -> dict:
        """Re-run discovery and return the current shard table.

        Uses the same ``replica_offset`` as ``_prepare`` so the fresh table pins the
        same replicas the plan did wherever the offers have not changed; otherwise a
        rotation would look like a difference.
        """
        fresh = gather_sources_with_steps(
            self._mx_client,
            expected_trainers=self._num_trainer_sources,
            model_name=self._model_name,
            role="inference",
            rank=self._global_rank,
            timeout=timeout,
            replica_offset=self._global_rank if _SPREAD_SOURCES else 0,
        )
        self._note_publisher_steps(fresh[4], step=step)
        return fresh[0]

    def _note_publisher_steps(self, session_to_step: dict, step: int | None = None):
        """Record which publishers advanced their step stamp since the *previous refit*.

        Kept as a delta rather than compared against the receiver's own step counter on
        purpose: the publisher's ``version`` and the receiver's refit ``step`` are
        separate counters, and assuming they agree would silently invert this check if
        they were ever offset. "Did this publisher's table move since the last refit?" is
        the question the staleness verdict actually needs, and it needs no such
        assumption - only that a publisher publishes once per refit.

        Keyed by refit rather than by call, because a refit discovers more than once: at
        prepare and again for the fresh table. Both see the same publication, so a
        per-call delta finds every stamp unchanged and pronounces every publisher stale.
        Measured on hardware in ``gate-stepstamp-v14``, which flagged all 16 publishers
        at step 1 - the step whose comparison is cleanest, since the weights are still
        the freshly loaded checkpoint and no trajectory has diverged. A real mismatch
        there would have been excused. Recording per step makes repeated discoveries
        within one refit idempotent.

        A publisher that does not stamp contributes ``None``, which is recorded as
        unknown and never treated as stale - an absent stamp is not evidence.
        """
        history = getattr(self, "_publisher_steps_by_step", None)
        if history is None:
            history = self._publisher_steps_by_step = {}
        current = {
            session: int(stamp)
            for session, stamp in session_to_step.items()
            if stamp is not None
        }
        key = -1 if step is None else int(step)
        previous = history.get(key - 1, {})
        stale = {
            session
            for session, stamp in current.items()
            if session in previous and stamp <= previous[session]
        }
        history[key] = current
        # Retained for ``stamps_seen``: whether any publisher stamps at all is a
        # different question from whether any is behind.
        self._publisher_steps = current
        # No previous refit means nothing can be shown to have lagged.
        self._stale_sessions = frozenset(stale) if previous else frozenset()

    @torch.no_grad()
    def _recheck_sources(self, step: int, timeout: float) -> dict:
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

        So this counts both rather than assuming either. On Topology B it returned
        addr_changed 0 across ~4.4M comparisons with digest_changed 1 on exactly the
        ranks that reported a mismatch, which settled Bug 9 as the second case.

        Returns the freshly discovered table so the verify gate can compare against
        current digests instead of prepare-time ones, rather than paying for a
        second discovery.
        """
        _t = time.perf_counter()
        fresh = self._fresh_sources(timeout, step=step)
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
        return fresh

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

        # Whether this call is the one that captured self._sources, which is the only
        # step on which that table describes current weights. Keyed on the plan rather
        # than on step == 1, because _prepare() runs on the first refit this receiver
        # serves and the caller's step numbering is not ours to assume.
        prepared_this_step = self._plan is None
        if self._plan is None:
            self._prepare(timeout, step=step)
            # One-time setup, so it lands only in the cold step's record. Without
            # this the cold step attributes ~2% of its own duration.
            stages.update(self._prepare_stages)
        assert self._plan is not None and self._transport is not None

        # Current digests for the verify gate. `self._sources` is frozen at
        # `_prepare()`, so on any step past the first it describes weights the
        # trainer has since updated, and comparing against it reports training as
        # corruption (Bug 9). Only paid when verification is on: it costs a
        # discovery (~0.8-1.6 s at 16 ranks), which a timing run must not absorb.
        fresh_sources = None
        if _ADDR_RECHECK:
            fresh_sources = self._recheck_sources(step, timeout)
        elif (VERIFY or DEST_DIGEST) and _VERIFY_FRESH_DIGESTS and step > 1:
            # DEST_DIGEST needs this for the same reason VERIFY does. Its source
            # fingerprint is only interpretable against current publisher claims; a
            # table frozen at _prepare() never changes, so every weight training
            # legitimately moved would read as the destination moving on its own -
            # the strongest finding the audit has, generated wholesale and wrongly.
            # Both gates are diagnostic and off by default, so the discovery cost is
            # acceptable here in a way it would not be on a timing run.
            fresh_sources = self._fresh_sources(timeout, step=step)

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
                full_staging=self._full_staging,
                sources=self._sources,
                fresh_sources=fresh_sources,
                step=step,
                stale_sessions=getattr(self, "_stale_sessions", None),
                stamps_seen=bool(getattr(self, "_publisher_steps", None)),
            )
            stages["verify_s"] = time.perf_counter() - _t

        # Also before install, and for the same reason, but covering the whole
        # destination rather than the full-pull sources: every fetch path has landed
        # in the receive buffers by now, so this is the only point where the
        # exact-segment path is observable at all.
        # The in-process differential. Placed with the other pre-install gates and
        # for the same reason: it reads the assembled destination, and installing
        # first would leave a difference ambiguous between transfer and install.
        #
        # Unlike the two-arm destination-digest comparison, this needs no second run
        # and makes no assumption that weights held still, because both
        # implementations consume the bytes this one refit received.
        replay_report = None
        if EXACT_REPLAY and self._plan.full_pulls:
            _t = time.perf_counter()
            replayed, replay_stats = exact_replay_digests(
                plan=self._plan,
                sources=self._sources,
                full_staging=self._full_staging,
                recv_buffers=self._recv_buffers,
            )
            replay_report = {
                **compare_exact_replay(
                    replayed=replayed,
                    received=digest_destination(self._recv_buffers),
                ),
                **replay_stats,
            }
            stages["exact_replay_s"] = time.perf_counter() - _t

        dest_digests = None
        source_digests = None
        source_digest_stats = None
        if DEST_DIGEST:
            _t = time.perf_counter()
            dest_digests = digest_destination(self._recv_buffers)
            # Paired with the destination digests in the same record and at the same
            # instant. Taken from the freshly discovered table when there is one, so
            # the claims describe the weights as of this step rather than as of
            # _prepare() - the frozen-expectation mistake Bug 9 was.
            source_digests, source_digest_stats = source_expectation_digests(
                dest_sources=self._plan.dest_sources,
                sources=self._sources,
                fresh_sources=fresh_sources,
                # self._sources is frozen at _prepare(), so it only describes the
                # current weights on the step that prepared it.
                expectation_is_current=(
                    prepared_this_step or fresh_sources is not None
                ),
            )
            stages["dest_digest_s"] = time.perf_counter() - _t

        _t = time.perf_counter()
        self._install(self._recv_buffers)
        torch.cuda.synchronize(self._device)
        stages["install_s"] = time.perf_counter() - _t

        if dest_digests is not None:
            # WARNING for the same reason as the verify record: the benchmark
            # harnesses capture at WARNING. One record per rank per step; the
            # comparison itself is offline, since a single run has nothing to
            # compare against.
            logger.warning(
                "%s%s",
                _DEST_DIGEST_MARKER,
                json.dumps(
                    dest_digest_record(
                        step=step,
                        rank=self._global_rank,
                        forced_full_pull=self._plan.forced_full_pull,
                        digests=dest_digests,
                        # Carried so the comparison can tell where the reference
                        # arm silently reverted to the exact path it is meant to
                        # be checking; see dest_digest_record.
                        unbounded_sources=self._plan.unbounded_sources,
                        fallback_sources=self._plan.fallback,
                        source_digests=source_digests,
                        source_digest_stats=source_digest_stats,
                    )
                ),
            )

        if replay_report is not None:
            logger.warning(
                "MX_REFIT_EXACT_REPLAY %s",
                json.dumps(
                    {
                        "schema": "refit-exact-replay-v1",
                        # Without the rank, 16 receivers' records for one step are
                        # indistinguishable from one receiver retrying that step,
                        # and any reader that de-duplicates retries - which it must,
                        # since a retried refit re-emits the record - would keep one
                        # and discard fifteen. Coverage then reads at a sixteenth of
                        # the truth, and a rank that replayed nothing disappears
                        # behind a rank that did.
                        "rank": self._global_rank,
                        "step": step,
                        **replay_report,
                    }
                ),
            )
            if replay_report["mismatches"]:
                raise RuntimeError(
                    f"[reshard] exact-segment replay FAILED at step {step}: "
                    f"{replay_report['mismatches']} of {replay_report['checked']} "
                    f"destination param(s) differ between the exact segment plan and "
                    f"the staged re-slice, over identical received bytes. Both paths "
                    f"read the same staging buffer here, so this is a segment "
                    f"offset/stride defect in plan_pull, not a transfer fault. "
                    f"First: {replay_report['detail'][:3]}"
                )
            if not replay_report["checked"]:
                message = (
                    f"[reshard] MX_RESHARD_EXACT_REPLAY is on but 0 destination "
                    f"params were comparable at step {step}, so the exact path is "
                    f"UNCHECKED. Zero mismatches here means no evidence, not a pass. "
                    f"The gate already forces full pulls, so staging is not the "
                    f"explanation: look instead for every source having taken the "
                    f"fallback path, or a plan with no exact segments to replay. "
                    f"stats={ {k: v for k, v in replay_report.items() if k != 'detail'} }"
                )
                if VERIFY_STRICT:
                    raise RuntimeError(message)
                logger.warning("%s", message)

        if verify_report is not None:
            # WARNING so a benchmark harness captures it alongside the stage record.
            logger.warning(
                "MX_REFIT_VERIFY %s",
                json.dumps(
                    {
                        "schema": "refit-verify-v2",
                        # Same reason the replay record carries one. Without it the
                        # records from N receiver ranks are indistinguishable, so a
                        # reader cannot attribute a mismatch to a rank and a reader
                        # that de-duplicates by step keeps one of N and reports a
                        # fraction of the coverage as if it were the whole run. That
                        # is survivable at the two ranks of the small rig and not at
                        # Topology B's sixteen.
                        "rank": self._global_rank,
                        "step": step,
                        **verify_report,
                    }
                ),
            )
            # Keyed on the attributable count rather than the raw one. With publisher
            # step stamps present the two differ: a shard whose publisher is provably a
            # step behind is excluded, while a mismatch from a current publisher in the
            # same report still aborts. Falls back to the raw count for a report that
            # predates the field, so an older receiver's behaviour is unchanged.
            attributable = verify_report.get("attributable_mismatches")
            if attributable is None:
                # A report from a build that predates the field. Reproduce the old
                # conjunction exactly rather than guessing.
                attributable = (
                    verify_report["mismatches"]
                    if verify_report.get("reference_is_current", True)
                    else 0
                )
            if attributable:
                raise RuntimeError(
                    f"[reshard] parameter verification FAILED at step {step}: "
                    f"{attributable} of {verify_report['checked']} "
                    f"checked shard(s) differ from the publisher's digest. The bytes "
                    f"this rank pulled are not the bytes the trainer holds, so the "
                    f"engine would now generate from wrong weights. First: "
                    f"{verify_report['detail'][:3]}"
                )
            if verify_report["mismatches"]:
                # Aborting here would be aborting on a reference we can prove is
                # older than the bytes. That is what killed two runs on 2026-07-30,
                # where every `want` turned out to be bit-for-bit the initial
                # checkpoint digest while `got` tracked training. Loud, and not
                # fatal: the shards are unverified, which is worth knowing and is
                # not the same as wrong.
                logger.warning(
                    "[reshard] parameter verification is UNATTRIBUTABLE at step "
                    "%s: %s",
                    step,
                    verify_report.get("unattributable_reason", ""),
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
                # Byte economics travel with the timings. These were INFO-only,
                # so every published row so far has carried an *estimated*
                # useful-bytes figure reconstructed after the fact. Wire minus
                # extra is the measured one.
                "extra_wire_bytes": self._plan.extra_wire_bytes(),
                "descriptor_savings": self._plan.descriptor_savings(),
                "exact_descriptors": self._plan.exact_descriptor_count,
                "full_pull_sources": len(self._plan.full_pulls),
                "unbounded_sources": len(self._plan.unbounded_sources),
                "converts": len(self._plan.converts),
                "fallback": len(stats["fallback"]),
                **{k: round(v, 6) for k, v in stages.items()},
            }
            # WARNING so benchmark harnesses capture it without enabling INFO
            # across every dependency.
            logger.warning("MX_REFIT_STAGE %s", json.dumps(record))
        self._check_throughput_ceiling(step, stats["bytes"], stages)
        return metrics

    def _check_throughput_ceiling(self, step: int, wire_bytes: int, stages: dict):
        """Refuse a wire rate the fabric cannot physically produce.

        An impossible rate is not a measurement: it is evidence the transport reported
        completions it did not earn, so it must abort rather than be recorded with a
        caveat. Off unless a ceiling is configured, because only the operator knows
        their fabric's real limit.

        Set the ceiling to a bound a rank genuinely cannot exceed - normally the whole
        node's aggregate - and not to an expected or derated rate. On 2026-07-27 this
        was configured at 381.6 Gbps, being one 400 Gb/s EFA adapter at 95.4%, and it
        aborted a healthy refit measured at 386.5 Gbps. The rank had two of the node's
        four adapters available to it, so the rate was legal and the guard was simply
        wrong. A check that aborts has to be calibrated so that tripping it is proof of
        a defect; anything tighter costs good runs.

        Note what this therefore does not do. It was written believing it would have
        caught Bug 10, the silent no-op reads that reported 40.61 GB in 0.84 s while
        delivering nothing. At a correct ceiling for that hardware it would not have:
        386.9 Gbps is comfortably legal across two 400 Gb/s adapters. Silent no-op
        reads are caught by the parameter-equality gate, not by arithmetic. This guard
        only catches a transport that returns so fast it beats the whole node.
        """
        if _MAX_GBPS <= 0 or wire_bytes <= 0:
            return
        wire_s = stages.get("wire_fused_s")
        if wire_s is None:
            wire_s = sum(
                stages.get(k, 0.0)
                for k in ("wire_exact_s", "wire_full_s", "wire_convert_s")
            )
        if not wire_s or wire_s <= 0:
            return
        implied_gbps = wire_bytes * 8 / wire_s / 1e9
        if implied_gbps <= _MAX_GBPS:
            return
        detail = {
            "schema": "refit-impossible-throughput-v1",
            "step": step,
            "wire_bytes": wire_bytes,
            "wire_s": round(wire_s, 6),
            "implied_gbps": round(implied_gbps, 1),
            "ceiling_gbps": _MAX_GBPS,
        }
        logger.warning("MX_REFIT_IMPOSSIBLE_THROUGHPUT %s", json.dumps(detail))
        raise RuntimeError(
            f"[reshard] step {step} moved {wire_bytes} bytes in {wire_s:.4f}s, an "
            f"implied {implied_gbps:.1f} Gbps against a per-rank ceiling of "
            f"{_MAX_GBPS:.1f} Gbps. The fabric cannot do this, so the transport "
            f"reported completions without delivering payload (see Bug 10: "
            f"libfabric device selection falling back to devices the pod does not "
            f"own). Treat this refit as failed, not fast."
        )
