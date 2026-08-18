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
import math
import time
from collections import deque

import torch

from modelexpress import envs
from modelexpress.client import MxClient
from modelexpress.nixl_transfer import NixlTransferManager
from modelexpress.refit.reshard.cuda_pool import classic_cuda_alloc
from modelexpress.refit.reshard.rendezvous import gather_sources
from modelexpress.refit.reshard.transfer_plan import (
    exact_descriptors,
    execute_transfer,
    plan_transfer,
)
from modelexpress.refit.reshard.transport import (
    NixlReshardTransport,
    ReadDescriptor,
)
from modelexpress.refit.reshard.types import (
    CaptureResult,
    IncompleteRefit,
    UnsupportedReshard,
    summarize_unsupported,
)

logger = logging.getLogger("modelexpress.refit.reshard.receiver")


def _max_gbps() -> float:
    """Per-rank fabric ceiling in Gbps; 0 disables the check.

    Read at call time so a harness can set it per run. Off unless configured
    because only the operator knows the real per-rank limit for their fabric.

    A non-finite ceiling disables the check rather than being compared against.
    ``float("nan")`` parses happily, and every comparison against NaN is False, so
    a fat-fingered value would slip past the "is it configured" test and then fail
    the "is the rate acceptable" test - turning the guard into a refit that always
    aborts. Infinity is accepted by the comparison but means the same as off.
    """
    ceiling = envs.MX_RESHARD_MAX_GBPS
    if not math.isfinite(ceiling):
        logger.warning(
            "MX_RESHARD_MAX_GBPS=%r is not a finite number; the throughput ceiling "
            "is disabled for this run",
            ceiling,
        )
        return 0.0
    return ceiling


def handshake_endpoints_for_plan(
    plan, session_to_agent: dict, agent_endpoints: dict
) -> dict:
    """Narrow ``agent_endpoints`` to the trainers ``plan`` reads from.

    Discovery has to collect every trainer's shard table, since which trainers own
    the bytes a receiver is missing is only known once the slice arithmetic is
    done. Handshaking with every one of them afterwards is a different matter: the
    handshake exists to resolve remote memory registrations for reads, so a
    trainer this rank never reads from costs a dial and buys nothing. Left
    unnarrowed that is one dial per receiver-trainer pair, which grows as the
    product of both sides.

    Fails closed when a trainer the plan reads from has no endpoint to dial.
    Skipping it instead would push the failure into ``prep_xfer_dlist``, which
    cannot say which peer it was missing.
    """
    needed = {
        session_to_agent[session]
        for session in plan.sessions()
        if session in session_to_agent
    }
    missing = sorted(needed - set(agent_endpoints))
    if missing:
        raise RuntimeError(
            f"[reshard] {len(missing)} trainer(s) in the transfer plan published no "
            f"metadata endpoint to handshake with: {missing[:10]}"
        )
    return {
        agent_name: endpoint
        for agent_name, endpoint in agent_endpoints.items()
        if agent_name in needed
    }


def handshake_with_peers(
    manager,
    agent_endpoints: dict,
    total_timeout: float,
    attempt_timeout: float | None = None,
) -> None:
    """Fetch every trainer's NIXL metadata, bounded, retried and logged per peer.

    Three properties, each earned from a failure mode observed on a live fabric:

    *Bounded overall*, not per peer against the refit timeout. A publisher whose
    process is gone still has its endpoint in the catalog - the reaper only marks
    it stale after a heartbeat lapse, and an abandoned run can keep heartbeating -
    so dialing it blocks. Charging a whole refit timeout to one dead peer hangs
    the refit long past the driver's own deadline.

    *Retried, and deferred rather than fatal on first failure.* A peer can be
    listening yet transiently unable to accept: its accept loop is a thread in a
    process that is busy publishing thousands of tensors, and a listen backlog
    that never drains silently drops connection attempts. That is
    indistinguishable from a dead peer within a single dial, but not across
    several seconds, so a failed peer goes to the back of the queue and the next
    one is tried instead of aborting the refit.

    *Logged per peer.* Without it the last line in the log reports that remote
    metadata is being fetched, and there is no way to tell which peer is at
    fault, or whether it stalled on the first dial or the last.
    """
    attempt_timeout = attempt_timeout or envs.MX_RESHARD_HANDSHAKE_ATTEMPT_S
    backoff = envs.MX_RESHARD_HANDSHAKE_BACKOFF_S
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
        # No floor: the deadline check above already guarantees remaining > 0, and
        # rounding a sub-second remainder up to a second overruns the very budget
        # this function exists to hold. A dial that gets 10 ms and fails is the
        # intended outcome there - the next pass reports the bounded error.
        this_timeout = min(attempt_timeout, remaining)
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
            # Parsed inside the retry so a malformed entry is reported as this
            # peer's failure, not raised past every other peer's handshake.
            host, port_str = endpoint.rsplit(":", 1)
            manager.fetch_remote_and_wait(
                agent_name, host, int(port_str), timeout_seconds=this_timeout
            )
        except Exception as exc:  # noqa: BLE001 - any dial failure is retryable
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
                time.sleep(min(backoff, max(0.0, deadline - time.monotonic())))
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

    retried = {name: count for name, count in attempts.items() if count > 1}
    if retried:
        logger.warning(
            "[reshard] _prepare: handshake completed with retries: %s", retried
        )


def _coverage_floor() -> float:
    """The fraction of engine parameter bytes a gated refit must install.

    What a *complete* refit scores is engine- and model-specific, because the
    denominator is whatever the engine enumerates as its load-time parameters and
    some of those are legitimately not refit material - rotary `inv_freq` and
    similar derived buffers. Hence a configurable floor rather than a hard 1.0.

    The default is deliberately loose. It is sized to catch a gross hole, of the
    order of a missing layer range, expert group or pipeline half, not to encode
    any one model's parameter accounting; the non-refit remainder is small in
    bytes even where it is many tensors by count. Tighten it per model once
    coverage records exist for that model rather than guessing upward here.

    A negative floor passes every refit, which silently disables the gate the
    caller just asked for, and a floor above 1.0 rejects every refit including a
    complete one. Both are rejected rather than honored.
    """
    floor = float(envs.MX_RESHARD_COVERAGE_FLOOR)
    if not 0.0 <= floor <= 1.0:
        raise ValueError(
            f"MX_RESHARD_COVERAGE_FLOOR must be a fraction in [0.0, 1.0], got {floor}"
        )
    return floor


def _fused_wire_enabled() -> bool:
    """Whether to issue the exact, full-pull and convert reads as one batch.

    Read at call time so an A/B can toggle it without re-importing. Set
    ``MX_RESHARD_FUSED_WIRE=0`` to drain each phase in turn.
    """
    return envs.MX_RESHARD_FUSED_WIRE


def _batch_install_enabled() -> bool:
    """Whether to re-slice full-pulled sources with one batched copy.

    Read at call time so an A/B can toggle it without re-importing. Set
    ``MX_RESHARD_BATCH_INSTALL=0`` to fall back to one ``copy_()`` per view.
    """
    return envs.MX_RESHARD_BATCH_INSTALL


def _cache_descriptors_enabled() -> bool:
    """Whether to reuse the read descriptor lists across steps.

    Read at call time so an A/B can toggle it without re-importing. Set
    ``MX_RESHARD_CACHE_DESCRIPTORS=0`` to rebuild them on every step.
    """
    return envs.MX_RESHARD_CACHE_DESCRIPTORS


def _destinations_are_disjoint(destinations: list[torch.Tensor]) -> bool:
    """Conservatively check whether destination views occupy separate storage.

    A strided view can contain holes, so its min/max storage span may overlap
    another view even when their logical elements do not. Treating that case as
    overlapping only loses batching; it cannot change copy order or bytes.
    """
    spans_by_storage: dict[int, list[tuple[int, int]]] = {}
    for tensor in destinations:
        if tensor.numel() == 0:
            continue
        min_element = int(tensor.storage_offset())
        max_element = min_element
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
            extent = (int(size) - 1) * int(stride)
            min_element += min(0, extent)
            max_element += max(0, extent)
        element_size = int(tensor.element_size())
        spans_by_storage.setdefault(
            int(tensor.untyped_storage().data_ptr()), []
        ).append(
            (
                min_element * element_size,
                (max_element + 1) * element_size,
            )
        )
    for spans in spans_by_storage.values():
        previous_end = -1
        for start, end in sorted(spans):
            if start < previous_end:
                return False
            previous_end = max(previous_end, end)
    return True


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
        device: torch.device,
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
        # Read descriptors for the cached plan, reused across steps. Tied to the
        # plan's lifetime: whatever rebuilds the plan must drop this too, or the
        # next refit would RDMA into the previous plan's addresses.
        self._cached_descriptors: tuple | None = None
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
        # One-time setup costs, measured in _prepare and reported with the first
        # refit's stage record. Kept separate from the per-step timings so a cold
        # first step is not read as a slow steady state.
        self._prepare_stages: dict[str, float] = {}

        logger.info(
            "[reshard] receiver init: agent=%s global_rank=%d trainer_sources=%d",
            agent_name,
            global_rank,
            num_trainer_sources,
        )

    # ------------------------------------------------------------- engine hooks
    def _capture(self, manifest: list) -> tuple[CaptureResult, dict]:
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
        """One-time: discover trainer shards, capture load geometry, build the pull
        plan, connect the trainers it reads from, and allocate + register buffers."""
        logger.info(
            "[reshard] _prepare: discovering %d trainer source(s) (timeout=%.0fs)",
            self._num_trainer_sources,
            timeout,
        )
        self._prepare_stages = {}
        _t = time.perf_counter()
        sources, session_to_agent, session_to_device, agent_endpoints = gather_sources(
            self._mx_client,
            expected_trainers=self._num_trainer_sources,
            model_name=self._model_name,
            role="inference",
            rank=self._global_rank,
            timeout=timeout,
        )
        self._prepare_stages["prepare_discover_s"] = time.perf_counter() - _t
        logger.info(
            "[reshard] _prepare: discovered %d source(s), %d agent(s)",
            len(sources),
            len(agent_endpoints),
        )
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
        all_params = sorted({c.param_name for c in capture.copies})
        # Before the fail-closed rejection below, not after: an unsupported plan is
        # the case where naming the hole matters most, and raising first would leave
        # the run with no record of what it was about to miss.
        self._log_coverage(capture, param_layout, all_params, plan)
        if plan.fallback:
            # Fallback params are dropped from the RDMA plan and never pulled or
            # installed, so they would silently keep their initial (base-model)
            # weights for the entire run. Until the full-pull/loader path exists
            # (TODO), fail loudly rather than serve stale weights.
            #
            # Carry the capture causes, not just the names. A rejection that
            # reports only which tensors are missing forces the reader back onto
            # the cluster to find out which op defeated capture.
            causes = summarize_unsupported(
                getattr(capture, "unsupported_reasons", {}) or {}
            )
            cause_text = (
                "; ".join(f"{count} x {cause}" for cause, count in causes)
                if causes
                else "cause not recorded at capture"
            )
            raise UnsupportedReshard(
                f"[reshard] {len(plan.fallback)} source(s) need the unimplemented "
                f"full-pull path (unsupported reshard ops); refusing to serve stale "
                f"weights. Causes: {cause_text}. Params: {plan.fallback[:10]}"
            )
        # P2P memory handshake (mirrors MX's vLLM RDMA path): fetch each trainer's
        # NIXL metadata (incl. its memory registrations) via its listen thread, so
        # prep_xfer_dlist can resolve the remote addresses. The central
        # add_remote_agent(blob) path does NOT convey the registrations.
        #
        # After planning, for two reasons. It only has to precede the first read,
        # and by here the plan says which trainers are actually read from, so peers
        # this rank never touches are not dialed. It also means an unsupported plan
        # is rejected above without spending the handshake budget first.
        # prepare_handshake_s covers the dial, and only the peers the plan reads
        # from. It is therefore not comparable with the same figure from a run
        # that dialed every discovered trainer before planning.
        _t = time.perf_counter()
        handshake_endpoints = handshake_endpoints_for_plan(
            plan, session_to_agent, agent_endpoints
        )
        logger.info(
            "[reshard] _prepare: P2P-fetching remote metadata from %d of %d agent(s)",
            len(handshake_endpoints),
            len(agent_endpoints),
        )
        handshake_with_peers(
            self._manager,
            handshake_endpoints,
            envs.MX_RESHARD_HANDSHAKE_TIMEOUT_S,
        )
        self._prepare_stages["prepare_handshake_s"] = time.perf_counter() - _t

        self._transport = NixlReshardTransport(
            self._manager, session_to_agent, session_to_device, timeout_seconds=timeout
        )
        self._plan = plan
        # New plan, so any descriptors cached for the old one are stale by
        # construction: they carry the previous plan's source addresses.
        self._cached_descriptors = None

        # dtype-mismatched sources (e.g. a bf16-served router for an fp32 dest):
        # one persistent bf16 STAGING buffer per convert param, registered as an
        # RDMA target (classic cudaMalloc so the HCA can RDMA into it); each refit
        # we RDMA into staging then cast staging -> the (load-time) receive buffer.
        # Allocation and registration happen in three places below (convert
        # staging, full-pull staging, receive buffers). They are accumulated into
        # one figure each rather than reported per buffer class, because the
        # actionable question is how much of a cold start is cudaMalloc versus HCA
        # registration.
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

    def _log_coverage(self, capture, param_layout, all_params, plan) -> None:
        """Report what this rank asked the wire for, against what it will install.

        Emitted at WARNING as JSON so a benchmark harness can recover it without
        turning on INFO across every dependency. That is the point: everything
        here was already computed, and already logged at INFO, which no
        benchmark run has captured. So `useful_bytes_per_rank` has been
        *derived* analysis-side from an assumed sharding rather than measured,
        and a derived number cannot distinguish a wrong model of the sharding
        from an incomplete refit.

        This is also the only check that can see a parameter the loader never
        asked for. Every other check compares arrived bytes against the
        publisher's digest for the same name, so bytes that are never requested
        are never checked. A refit covering half the model passes all of them.

        `unsupported` is the companion signal: a parameter the loader wants and
        the planner cannot serve is silently absent from the wire, so a non-zero
        count is a coverage hole by construction.
        """
        # `param_layout` is the engine's COMPLETE parameter set; `all_params` is
        # the subset this refit will write. The ratio is the coverage nothing
        # else measures, and it needs no engine-specific hook.
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
            # Grouped causes, so the harvested record explains an incomplete
            # refit without needing the run's console output alongside it.
            "unsupported_causes": [
                {"cause": cause[:200], "sources": count}
                for cause, count in summarize_unsupported(
                    getattr(capture, "unsupported_reasons", {}) or {}
                )
            ],
            "planned_wire_bytes": plan.bytes_planned(),
            "extra_wire_bytes": plan.extra_wire_bytes(),
            "descriptors": plan.descriptor_count(),
            "descriptor_savings": plan.descriptor_savings(),
            "full_pull_sources": len(plan.full_pulls),
            "unbounded_sources": len(plan.unbounded_sources),
            "converts": len(plan.converts),
            "fallback": len(plan.fallback),
        }
        # Severity follows the record's content, not the fact that a record exists.
        # This is emitted once per refit on a healthy run too, and a per-refit line
        # at WARNING teaches operators that MX warnings are routine, which costs
        # more than it buys the first time one is not.
        complete = not missed and not unsupported and not plan.fallback
        logger.log(
            logging.INFO if complete else logging.WARNING,
            "MX_REFIT_COVERAGE %s",
            json.dumps(record),
        )

        # Opt-in rather than always-on: partial and subset refit are intended
        # features, and for those a coverage below 1.0 is the point. What is never
        # acceptable is a *benchmark* row measuring an incomplete refit, because
        # its wire volume and timings are then the wrong magnitude and get
        # compared against complete ones.
        if envs.MX_RESHARD_REQUIRE_FULL_COVERAGE and coverage < _coverage_floor():
            raise IncompleteRefit(
                f"refit covers {100.0 * coverage:.2f}% of the engine's parameter "
                f"bytes ({dest_bytes} of {engine_bytes}); "
                f"{len(missed)} of {len(param_layout)} params would keep their "
                f"previous values, e.g. {sorted(missed)[:5]}. No digest gate can "
                f"detect this - bytes that are never requested are never checked. "
                f"Set MX_RESHARD_REQUIRE_FULL_COVERAGE=0 for an intentionally "
                f"partial refit."
            )

    # ----------------------------------------------------------- update_weights
    @torch.no_grad()
    def update_weights(self, step: int, *, timeout: float | None = None) -> dict:
        """RDMA-pull the needed slices into the receive buffers, cast the
        dtype-mismatched ones, then install into the live params."""
        timeout = timeout if timeout is not None else self._timeout
        stages: dict[str, float] = {}
        # TODO(re-plan on topology change): the plan is built once and cached, so
        # a mid-run change in the trainer set - a trainer restart (new buffer
        # addresses), a reshard (new shard boundaries / fan-in), or scaling the
        # trainer count - is NOT picked up; every step re-reads the first
        # discovery's addresses. Adapt the plan when topology changes (e.g.
        # re-discover + rebuild if a version/epoch token or address set differs).
        if self._plan is None:
            self._prepare(timeout)
            # Attributed to the step that paid for it, so the stage record for a
            # cold first refit accounts for its own setup instead of leaving it
            # unattributed.
            stages.update(self._prepare_stages)
        assert self._plan is not None and self._transport is not None

        # RDMA the sliced bf16 into the receive buffers (segments) and per-param
        # staging (dtype-convert / router). No live param is written by RDMA.
        #
        # The three read phases target disjoint destinations - exact segments land
        # in the receive buffers, full pulls in full staging, converts in convert
        # staging - and every reader of those buffers (the re-slice below, the
        # dtype cast) runs after all reads complete. So the phases carry no
        # ordering dependency and are issued as one batch by default. Phased mode
        # drains each in turn and is kept for the A/B.
        # Descriptor construction is timed and cached. Timed because it is real
        # per-step work that used to fall outside every stage, so it surfaced only
        # as unattributed time and pushed the record below the attribution floor a
        # breakdown has to clear to be worth reporting. Cached because a descriptor
        # is a (session, src_addr, dst_addr, nbytes) tuple derived from the plan and
        # the registered buffer addresses: the plan is built once and reused, and
        # the buffers are registered once, so every step was re-deriving an
        # identical list of hundreds of thousands of objects in Python.
        _t = time.perf_counter()
        cached = _cache_descriptors_enabled()
        fused = _fused_wire_enabled()
        # The fused flag is part of the key, not just the payload: the phased arm
        # does not build the exact descriptors, so a cache filled under one arm
        # cannot serve the other. An A/B that toggles it mid-process is exactly the
        # case this is for.
        reusable = (
            self._cached_descriptors is not None
            and self._cached_descriptors[0] == fused
        )
        if not cached or not reusable:
            full_descriptors = [
                ReadDescriptor(
                    session=segment.session,
                    src_addr=segment.src_addr,
                    dst_addr=(
                        self._full_staging_ptr[full_pull.src_name] + segment.dst_byte
                    ),
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
            # Only the fused path issues the exact segments as descriptors; the
            # phased path hands the plan to execute_transfer instead, so building
            # them here would be wasted work for that arm.
            exact = (
                exact_descriptors(self._plan, lambda name: self._param_ptr[name])
                if fused
                else None
            )
            # nbytes of the auxiliary descriptors, summed once for the same reason.
            aux_bytes = sum(
                descriptor.nbytes
                for descriptor in (*full_descriptors, *convert_descriptors)
            )
            if cached:
                self._cached_descriptors = (
                    fused,
                    full_descriptors,
                    convert_descriptors,
                    exact,
                    aux_bytes,
                )
        else:
            _, full_descriptors, convert_descriptors, exact, aux_bytes = (
                self._cached_descriptors
            )
        stages["descriptor_build_s"] = time.perf_counter() - _t

        if fused:
            assert exact is not None
            descriptors = exact
            stats = {
                "segments": len(descriptors),
                "bytes": sum(descriptor.nbytes for descriptor in descriptors),
                "fallback": list(self._plan.fallback),
            }
            _t = time.perf_counter()
            self._transport.read(descriptors + full_descriptors + convert_descriptors)
            stages["wire_fused_s"] = time.perf_counter() - _t
        else:
            # A plan can be all converts or all full pulls. Timing an exact phase
            # that moved nothing records a duration that is not a wire duration: it
            # inflates accounted_s, and because the phased implied rate divides the
            # bytes by the summed wire stages, it drags that rate down and makes the
            # throughput ceiling less likely to catch a run that deserves it.
            #
            # Gated on segments, not exact_descriptor_count. The latter is the
            # pre-bounding baseline that descriptor_savings() subtracts from, and it
            # counts every copy including the ones that became full pulls or
            # converts, so a bounded plan can carry a nonzero count with nothing
            # left to issue. exact_descriptors() reads segments and nothing else.
            if self._plan.segments:
                _t = time.perf_counter()
                stats = execute_transfer(
                    self._plan,
                    resolve_param_ptr=lambda name: self._param_ptr[name],
                    transport=self._transport,
                )
                stages["wire_exact_s"] = time.perf_counter() - _t
            else:
                stats = {
                    "segments": 0,
                    "bytes": 0,
                    "fallback": list(self._plan.fallback),
                }
            if full_descriptors:
                _t = time.perf_counter()
                self._transport.read(full_descriptors)
                stages["wire_full_s"] = time.perf_counter() - _t
            if convert_descriptors:
                _t = time.perf_counter()
                self._transport.read(convert_descriptors)
                stages["wire_convert_s"] = time.perf_counter() - _t

        stats["segments"] += len(full_descriptors) + len(convert_descriptors)
        stats["bytes"] += aux_bytes

        # Before anything reads the receive buffers, and well before _install
        # commits them to live parameters. An impossible rate means the transport
        # reported completions it did not earn, so the buffers cannot be trusted;
        # checking after the install would document the corruption rather than
        # prevent it. Everything the check needs is known by this point.
        self._check_throughput_ceiling(step, stats["bytes"], stages)

        # Local re-slice of every full-pulled source into its receive buffer, and
        # the dtype cast for every converted param. Both read staging written by
        # the reads above, so both must run after the wire completes.
        # Each GPU-side stage is followed by a synchronize so its duration is the
        # work itself rather than the point at which a later stage happens to
        # block. Without that, launch-bound stages read as free and whichever
        # stage syncs first absorbs the whole queue.
        if self._plan.full_pulls:
            # Local re-slice of every full-pulled source. One copy_() per captured
            # view means thousands of individual kernel launches, whose Python and
            # launch overhead can rival the RDMA itself; _foreach_copy_ issues the
            # same copies as a single batched op.
            _t = time.perf_counter()
            batched = _batch_install_enabled()
            destinations: list[torch.Tensor] = []
            source_views: list[torch.Tensor] = []
            copies_done = 0
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
                    if batched:
                        destinations.append(destination)
                        source_views.append(source_view)
                    else:
                        destination.copy_(source_view)
                        copies_done += 1
            reslice_copies = len(destinations) if batched else copies_done
            if batched and destinations:
                if _destinations_are_disjoint(destinations):
                    torch._foreach_copy_(destinations, source_views)
                else:
                    # ``_foreach_copy_`` does not define ordering for overlapping
                    # destinations. Preserve the captured loader order instead.
                    for destination, source_view in zip(
                        destinations, source_views, strict=True
                    ):
                        destination.copy_(source_view)
            torch.cuda.synchronize(self._device)
            stages["reslice_s"] = time.perf_counter() - _t
            # Views, not sources: the per-view launch count is what batching
            # removes, and the source count is already reported separately.
            stages["reslice_copies"] = float(reslice_copies)

        # Cast the served bf16 staging into the (fp32) receive buffer - a torch
        # op, so the RDMA never crosses dtypes. _install writes the buffer.
        if self._plan.converts:
            _t = time.perf_counter()
            for convert in self._plan.converts:
                self._recv_buffers[convert.param_name].copy_(
                    self._staging[convert.param_name]
                )
            torch.cuda.synchronize(self._device)
            stages["convert_s"] = time.perf_counter() - _t

        _t = time.perf_counter()
        self._install(self._recv_buffers)
        torch.cuda.synchronize(self._device)
        stages["install_s"] = time.perf_counter() - _t

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
        if envs.MX_REFIT_STAGE_RECORD:
            record = {
                "schema": "refit-stage-v2",
                # Every rank emits its own line, so without this a fleet-wide
                # capture cannot tell them apart - and the number that matters for
                # a refit is the slowest rank, not the average of an anonymous pile.
                "rank": self._global_rank,
                "step": step,
                "bytes": stats["bytes"],
                "segments": stats["segments"],
                # Sum of the measured stages. Compared against the caller's own
                # end-to-end figure it gives the unattributed remainder, which is
                # the number that says whether this record can be trusted as a
                # breakdown. The stages do not overlap, so summing them is valid
                # here; that stops being true if a stage is ever made concurrent.
                "accounted_s": round(
                    sum(v for k, v in stages.items() if k.endswith("_s")), 6
                ),
                # Byte economics travel with the timings. These were INFO-only, so
                # a harness that captured the timings still had to reconstruct the
                # useful-byte figure after the fact. Wire minus extra is measured.
                "extra_wire_bytes": self._plan.extra_wire_bytes(),
                "descriptor_savings": self._plan.descriptor_savings(),
                "exact_descriptors": self._plan.exact_descriptor_count,
                "full_pull_sources": len(self._plan.full_pulls),
                "unbounded_sources": len(self._plan.unbounded_sources),
                "converts": len(self._plan.converts),
                "fallback": len(stats["fallback"]),
                "fused_wire": _fused_wire_enabled(),
                # The install arm this record was measured under. Without it a
                # captured record cannot be attributed to a batched or per-view
                # re-slice, which is the whole point of an A/B.
                "batch_install": _batch_install_enabled(),
                **{k: round(v, 6) for k, v in stages.items()},
            }
            # WARNING so a benchmark harness captures it without turning on INFO
            # across every dependency.
            logger.warning("MX_REFIT_STAGE %s", json.dumps(record))
        return metrics

    def _check_throughput_ceiling(
        self, step: int, wire_bytes: int, stages: dict
    ) -> None:
        """Refuse a wire rate the fabric cannot physically produce.

        One run delivered nothing and reported the fastest refit yet measured:
        40.61 GB in 0.84 s, an implied 387 Gbps, on a pod holding two adapters
        worth about 191 Gbps. Coverage read 100%, fallback and unsupported were 0,
        and the addresses and digests were stable. The only signal that dissented
        was the parameter-equality gate, which timing runs switch off by design,
        so without this check that run becomes the best row in the matrix.

        An impossible rate is not a fast measurement, it is evidence the transport
        reported completions it did not earn, so this aborts rather than recording
        the number with a caveat. The impossible-throughput record below is emitted
        before the raise so the evidence survives the abort; the ordinary stage
        record is not reached, which is the point - the run produced no trustworthy
        timing to file.

        Called before the receive buffers are read or installed. Running it after
        the install would leave the guard describing weights it had already let
        through, which is the one outcome it exists to prevent.

        Off unless a ceiling is configured, because only the operator knows the
        real per-rank limit for their fabric.
        """
        ceiling = _max_gbps()
        if ceiling <= 0 or wire_bytes <= 0:
            return
        # Fused mode reports one wire span; phased mode reports up to three, and
        # they run in turn, so summing them is the phased equivalent.
        wire_s = stages.get("wire_fused_s")
        if wire_s is None:
            wire_s = sum(
                stages.get(key, 0.0)
                for key in ("wire_exact_s", "wire_full_s", "wire_convert_s")
            )
        if wire_s <= 0:
            return
        implied_gbps = wire_bytes * 8 / wire_s / 1e9
        if implied_gbps <= ceiling:
            return
        detail = {
            "schema": "refit-impossible-throughput-v1",
            # Which rank saw the impossible rate. One rank's adapters being
            # misselected is a different diagnosis from the whole pod's.
            "rank": self._global_rank,
            "step": step,
            "wire_bytes": wire_bytes,
            "wire_s": round(wire_s, 6),
            "implied_gbps": round(implied_gbps, 1),
            "ceiling_gbps": ceiling,
        }
        logger.warning("MX_REFIT_IMPOSSIBLE_THROUGHPUT %s", json.dumps(detail))
        raise RuntimeError(
            f"[reshard] step {step} moved {wire_bytes} bytes in {wire_s:.4f}s, an "
            f"implied {implied_gbps:.1f} Gbps against a per-rank ceiling of "
            f"{ceiling:.1f} Gbps. The fabric cannot deliver that, so the transport "
            f"reported completions without moving the payload - one known cause is "
            f"the fabric library selecting adapters this container does not own. "
            f"Treat this refit as failed, not fast."
        )
