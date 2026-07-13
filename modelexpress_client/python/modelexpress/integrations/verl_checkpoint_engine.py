# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""verl-side checkpoint engine that uses MX rank-to-rank publishing.

This is the verl-facing entry point for the no-allgather refit contract:

- :class:`VerlMxCheckpointEngine` is constructed once on each trainer worker
  (via the verl :class:`CheckpointEngine` plugin hook —
  ``checkpoint_engine.custom_backend_module`` config knob), optionally with
  a Ray actor handle to the matching inference replica for push-mode
  notifications.
- On each refit step, :meth:`publish_weights` is called from the trainer:
  it wraps the (FSDP) state dict in :class:`PlacementDescriptor` entries and
  publishes via :class:`RankLocalPublisher` — strictly local-shard bytes,
  no gather to rank 0.
- On the inference side, :class:`VerlMxRolloutLoader` is constructed once
  per inference replica and called from the rollout-engine init hook to
  pull rank-local shards via :class:`MxRefitReceiver` + the
  :class:`CoveragePlan` returned by :func:`plan_coverage`.

Earlier MX verl integrations published via ``tensor.full_tensor()`` — an
explicit FSDP allgather on the trainer side. This module is the
rank-to-rank upgrade path that drops the allgather and aligns the verl
data path with the contract used by the PrimeRL ``mx_v2`` broadcast and
the NemoRL DTensor publisher.

The migration story: existing verl deployments using the gather-based
integration continue to work unchanged. New deployments opt into the
rank-to-rank contract by registering this engine in place of the
gather-based one. Falls back to :meth:`MxRefitReceiver.receive_weights`
when the receiver lacks the per-segment fast path.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ..rank_local_publisher import (
    PlacementDescriptor,
    RankLocalPublisher,
)
from ..rl_reshard_planner import plan_coverage, summarize_plan
from ..rl_slice_descriptors import (
    CoveragePlan,
    PlanIncompleteError,
    QuantizationMetadataError,
    QuantizationScope,
    SliceOwnership,
    SliceRequest,
)

logger = logging.getLogger("modelexpress.integrations.verl")


# ---------------------------------------------------------------------------
# Trainer-side
# ---------------------------------------------------------------------------


@dataclass
class VerlPublishConfig:
    """Per-cycle config for :class:`VerlMxCheckpointEngine.publish_weights`.

    Args:
        model_name: identifier shared with the inference side (HF model id).
        step: current RL step number; used both for the v1 catalog entry
            and for receiver-side freshest-per-rank dedup.
        compile_target: kernel layout tag for the bytes we're publishing.
            ``"bf16_cast"`` for vanilla bf16 weights; ``"cutlass_fp8"`` for
            inference-ready FP8.
        compile_metadata: kernel-specific parameters carried alongside the
            tag. Defaults empty.
        same_rank_routing_hint: if True, hint to receivers that this rank
            prefers same-rank readers (for multi-NIC GB200 fabrics). Honored
            by :func:`plan_coverage` via worker_rank ordering.
        wait_for_inference_ack: if True, block in :meth:`publish_weights`
            until the inference replica posts an ack via :meth:`record_ack`.
            False for the typical async-RL case where the trainer keeps going.
        ack_timeout_s: how long to wait for ``record_ack`` when
            ``wait_for_inference_ack`` is True. Defaults to 120s (matches
            the conservative inference-side load budget).
    """

    model_name: str
    step: int
    compile_target: str = "bf16_cast"
    compile_metadata: dict[str, object] = field(default_factory=dict)
    same_rank_routing_hint: bool = True
    wait_for_inference_ack: bool = False
    ack_timeout_s: float = 120.0


class VerlMxCheckpointEngine:
    """verl trainer-side checkpoint engine using MX rank-to-rank publishing.

    Drop-in replacement for the gather-then-publish path used by earlier MX
    verl integrations. Instantiated once per FSDP worker (verl wires this
    up through its :class:`CheckpointEngine` plugin hook). On each refit,
    calls :meth:`publish_weights` with the current FSDP state dict — no
    allgather.

    Args:
        publisher: the v1 :class:`MxTrainingPublisher` for this rank.
            Caller is responsible for ``initialize()`` + lifecycle.
        worker_rank: this trainer worker's rank index. Used both for
            ``SliceOwnership.worker_rank`` and for the :class:`MxClient`
            same-rank routing hint.
        rollout_actor: optional Ray actor handle for the matching
            inference replica's rollout worker. Used by
            :meth:`publish_weights` to push the ``source_id`` to the
            inference side. ``None`` selects pull-only mode where the
            inference rollout polls the catalog itself.
    """

    def __init__(
        self,
        publisher: Any,  # MxTrainingPublisher
        *,
        worker_rank: int,
        rollout_actor: Any | None = None,
    ) -> None:
        self._publisher = publisher
        self._worker_rank = worker_rank
        self._rollout_actor = rollout_actor

        self._last_source_id: str | None = None
        self._last_ownerships: list[SliceOwnership] = []
        self._ack_cv = threading.Condition()
        self._last_ack_step: int = -1

    # ------------------------------------------------------------------
    # State-dict ingestion
    # ------------------------------------------------------------------

    def publish_weights(
        self,
        state_dict: dict[str, Any],
        config: VerlPublishConfig,
        *,
        placement_overrides: dict[str, PlacementDescriptor] | None = None,
        quantization_overrides: dict[str, QuantizationScope] | None = None,
    ) -> str:
        """Publish ``state_dict`` to MX without a global allgather.

        For each tensor:

        - If it's a ``torch.distributed.tensor.DTensor``, call
          :meth:`RankLocalPublisher.add_dtensor` — auto-derives placement
          from ``.placements`` + ``.device_mesh``, no gather.
        - Otherwise, the caller must provide an entry in
          ``placement_overrides`` (for Megatron-Core; verl Gen 2 today
          stays on FSDP-DTensor so this branch is rarely hit).

        Args:
            state_dict: ``{name: tensor}`` mapping for this rank's view of
                the model.
            config: per-cycle :class:`VerlPublishConfig`.
            placement_overrides: explicit placement for non-DTensor entries.
            quantization_overrides: per-tensor quant scope (e.g. mark
                ``weight_scale_inv`` as ``"global-required"`` so the
                planner rejects zero-copy on it).

        Returns:
            The ``mx_source_id`` assigned by the MX server.
        """
        rlp = RankLocalPublisher(
            self._publisher,
            model_name=config.model_name,
            worker_rank=self._worker_rank,
        )

        placement_overrides = placement_overrides or {}
        quantization_overrides = quantization_overrides or {}

        for name, tensor in state_dict.items():
            quant_scope = quantization_overrides.get(name, "absent")
            if name in placement_overrides:
                rlp.add_explicit_shard(
                    name,
                    tensor,
                    placement_overrides[name],
                    compile_target=config.compile_target,
                    compile_metadata=config.compile_metadata,
                    quantization_scope=quant_scope,
                )
                continue

            if _is_dtensor(tensor):
                rlp.add_dtensor(
                    name,
                    tensor,
                    compile_target=config.compile_target,
                    compile_metadata=config.compile_metadata,
                    quantization_scope=quant_scope,
                )
                continue

            # Plain torch.Tensor — treat as REPLICATE across the world.
            # This matches the historical v1 behavior for small replicated
            # buffers (layer norms, embeddings on TP-1).
            rlp.add_explicit_shard(
                name,
                tensor,
                PlacementDescriptor(
                    placement_kind="REPLICATE",
                    global_shape=tuple(tensor.shape),
                ),
                compile_target=config.compile_target,
                compile_metadata=config.compile_metadata,
                quantization_scope=quant_scope,
            )

        source_id = rlp.publish(step=config.step)
        ownerships = rlp.drain_slice_ownerships()
        self._last_source_id = source_id
        self._last_ownerships = ownerships

        # Push the source_id to the rollout replica if we have a handle.
        if self._rollout_actor is not None:
            try:
                self._rollout_actor.notify_new_source.remote(
                    source_id, config.step, ownerships
                )
            except AttributeError:
                # Backward-compat shim: older RolloutWorker had no
                # notify_new_source. Fall back to the polling path.
                logger.warning(
                    "rollout_actor lacks notify_new_source; receivers must poll"
                )

        if config.wait_for_inference_ack:
            self._wait_for_ack(config.step, timeout_s=config.ack_timeout_s)

        return source_id

    # ------------------------------------------------------------------
    # Ack handshake (optional, sync-RL path)
    # ------------------------------------------------------------------

    def record_ack(self, step: int) -> None:
        """Called by the rollout side once the new step is loaded.

        Used only when ``VerlPublishConfig.wait_for_inference_ack`` is True
        (sync RL with strict trainer-pauses). The async-RL path doesn't
        block on this.
        """
        with self._ack_cv:
            if step > self._last_ack_step:
                self._last_ack_step = step
                self._ack_cv.notify_all()

    def _wait_for_ack(self, step: int, *, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        with self._ack_cv:
            while self._last_ack_step < step:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"VerlMxCheckpointEngine: no ack for step={step} "
                        f"within {timeout_s:.1f}s (last_ack={self._last_ack_step})"
                    )
                self._ack_cv.wait(timeout=remaining)

    # ------------------------------------------------------------------
    # Introspection (used by tests + dashboards)
    # ------------------------------------------------------------------

    @property
    def last_source_id(self) -> str | None:
        return self._last_source_id

    @property
    def last_ownerships(self) -> tuple[SliceOwnership, ...]:
        return tuple(self._last_ownerships)


# ---------------------------------------------------------------------------
# Inference-side
# ---------------------------------------------------------------------------


class VerlMxRolloutLoader:
    """verl inference-side loader using MX rank-to-rank pull.

    Drops the gather-based ``vllm.collective_rpc("load_weights")`` step
    used by earlier MX verl integrations and replaces it with a
    planner-driven NIXL pull that grabs only the bytes this rank
    actually needs.

    One instance per rollout worker (vLLM TP rank). Construct with a
    pre-initialized :class:`MxRefitReceiver`, the receiver's TP layout
    (so we can build :class:`SliceRequest` entries), and an optional
    Ray actor handle to the matching trainer.

    Args:
        receiver: pre-initialized :class:`MxRefitReceiver` for this rank.
        receiver_rank: vLLM TP rank index for this worker.
        receiver_tp_size: vLLM TP world size.
        trainer_actor: optional Ray actor handle for the trainer
            checkpoint engine — used to send the ack on sync-RL.
        compile_target_filter: optional whitelist of source compile_target
            tags. If None, accept any.
    """

    def __init__(
        self,
        receiver: Any,  # MxRefitReceiver
        *,
        receiver_rank: int,
        receiver_tp_size: int,
        trainer_actor: Any | None = None,
        compile_target_filter: frozenset[str] | None = None,
    ) -> None:
        self._receiver = receiver
        self._receiver_rank = receiver_rank
        self._receiver_tp_size = receiver_tp_size
        self._trainer_actor = trainer_actor
        self._compile_target_filter = compile_target_filter

        # Latest source_id we've heard about; written by notify_new_source,
        # read by load_step.
        self._pending_source_id: str | None = None
        self._pending_step: int = -1
        self._pending_ownerships: list[SliceOwnership] = []
        self._pending_cv = threading.Condition()

    # ------------------------------------------------------------------
    # Notification from trainer
    # ------------------------------------------------------------------

    def notify_new_source(
        self,
        source_id: str,
        step: int,
        ownerships: Iterable[SliceOwnership],
    ) -> None:
        """Called by the trainer-side checkpoint engine when a new source
        is ready. Wakes any thread blocked in :meth:`load_step`.
        """
        with self._pending_cv:
            if step <= self._pending_step:
                logger.info(
                    "rollout_loader: skipping stale source step=%d (have %d)",
                    step,
                    self._pending_step,
                )
                return
            self._pending_source_id = source_id
            self._pending_step = step
            self._pending_ownerships = list(ownerships)
            self._pending_cv.notify_all()

    # ------------------------------------------------------------------
    # Main load path
    # ------------------------------------------------------------------

    def load_step(
        self,
        local_state_dict: dict[str, Any],
        *,
        timeout_s: float = 120.0,
        send_ack: bool = False,
    ) -> CoveragePlan:
        """Load the most recently published trainer step into ``local_state_dict``.

        Blocks until a source is available, then:

        1. Builds :class:`SliceRequest` entries from ``local_state_dict``
           (one per tensor; range = this rank's TP shard).
        2. Runs :func:`plan_coverage` against the published
           :class:`SliceOwnership` entries.
        3. Issues NIXL reads via :class:`MxRefitReceiver` for each
           :class:`SegmentPlan`.
        4. Optionally sends the ack back to the trainer.

        Args:
            local_state_dict: ``{name: tensor}`` for this rank's view of
                the model — bytes get written in-place into these buffers.
            timeout_s: how long to wait for a source to appear.
            send_ack: if True, call ``trainer_actor.record_ack.remote(step)``
                after the load completes (sync-RL).

        Returns:
            The :class:`CoveragePlan` that was executed. Callers (and
            tests) can inspect ``summarize_plan(...)`` for byte-counts
            and source-rank distribution.

        Raises:
            TimeoutError: if no source arrives within ``timeout_s``.
            PlanIncompleteError: if the published sources don't cover all
                requested ranges.
            QuantizationMetadataError: if a global-required quant tensor
                is in the request set; caller must use a non-zero-copy
                fallback for that tensor.
        """
        source_id, step, ownerships = self._wait_for_source(timeout_s)

        requests = list(
            self._build_requests(
                local_state_dict, compile_target_filter=self._compile_target_filter
            )
        )

        plan = plan_coverage(sources=ownerships, requests=requests)
        plan.raise_if_incomplete()

        summary = summarize_plan(plan)
        logger.info(
            "rollout_loader: rank=%d step=%d source_id=%s plan=%s",
            self._receiver_rank,
            step,
            source_id,
            summary,
        )

        self._execute_plan(plan, source_id=source_id)

        if send_ack and self._trainer_actor is not None:
            try:
                self._trainer_actor.record_ack.remote(step)
            except AttributeError:
                logger.warning(
                    "trainer_actor lacks record_ack; ack skipped"
                )

        return plan

    def _wait_for_source(
        self, timeout_s: float
    ) -> tuple[str, int, list[SliceOwnership]]:
        deadline = time.monotonic() + timeout_s
        with self._pending_cv:
            while self._pending_source_id is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"VerlMxRolloutLoader: no source within {timeout_s:.1f}s"
                    )
                self._pending_cv.wait(timeout=remaining)
            sid = self._pending_source_id
            step = self._pending_step
            ownerships = list(self._pending_ownerships)
            self._pending_source_id = None
            self._pending_ownerships = []
            return sid, step, ownerships

    def _build_requests(
        self,
        local_state_dict: dict[str, Any],
        *,
        compile_target_filter: frozenset[str] | None,
    ) -> Iterator[SliceRequest]:
        """Translate the local state-dict into one :class:`SliceRequest` per tensor.

        Assumes simple TP-only sharding on axis 0 for projection-style weights
        and REPLICATE on layer norms / embeddings. Real verl wiring will pass
        a richer layout descriptor (e.g. inferred from
        ``model.config.tp_plan``); this stub uses a conservative default that
        keeps the rank-to-rank semantics intact for the validated test cases.
        """
        for name, tensor in local_state_dict.items():
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype) if hasattr(tensor, "dtype") else "torch.bfloat16"
            # Heuristic: TP-shard projection weights on axis 0; replicate norms.
            shard_axis = 0 if _looks_tp_sharded(name) else None
            if shard_axis is None:
                global_range = (0, shape[0])
            else:
                axis_size = shape[shard_axis] * self._receiver_tp_size
                shard = axis_size // self._receiver_tp_size
                lo = self._receiver_rank * shard
                hi = (self._receiver_rank + 1) * shard if self._receiver_rank < self._receiver_tp_size - 1 else axis_size
                global_range = (lo, hi)

            yield SliceRequest(
                tensor_name=name,
                global_range=global_range,
                shard_axis=shard_axis,
                dtype=dtype,
                receiver_rank=self._receiver_rank,
                target_addr=int(tensor.data_ptr()) if hasattr(tensor, "data_ptr") else 0,
                target_offset=0,
                compile_target_filter=compile_target_filter,
            )

    def _execute_plan(self, plan: CoveragePlan, *, source_id: str) -> None:
        """Drive each :class:`SegmentPlan` through :class:`MxRefitReceiver`.

        The receiver exposes :meth:`receive_segment` (added by the verl
        prototype on top of v1; falls back to the slower
        :meth:`receive_weights` if the new entry point isn't present).
        """
        receive_segment = getattr(self._receiver, "receive_segment", None)
        if receive_segment is None:
            # v1-only fallback: pull every tensor via the existing
            # receive_weights path, ignoring the plan. This is the
            # gather-equivalent behavior the earlier MX verl integrations
            # used — keeps deployments without the v1.5 fast-path entry
            # point working unchanged.
            logger.warning(
                "MxRefitReceiver lacks receive_segment; falling back to "
                "v1 receive_weights (gather-equivalent)"
            )
            for _name, _tensor in self._receiver.receive_weights(source_id):
                pass
            return

        for seg in plan.segments:
            receive_segment(
                source_id=source_id,
                source_rank=seg.source.worker_rank,
                source_addr=seg.source.nixl_addr,
                source_offset=seg.source_range[0],
                target_addr=seg.request.target_addr + seg.request.target_offset,
                byte_count=seg.byte_count,
            )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _is_dtensor(t: Any) -> bool:
    """Cheap DTensor check without importing torch unconditionally."""
    cls_name = type(t).__name__
    if cls_name == "DTensor":
        return True
    # Walk MRO for the case where DTensor is a subclass of torch.Tensor.
    return any(c.__name__ == "DTensor" for c in type(t).__mro__)


def _looks_tp_sharded(name: str) -> bool:
    """Heuristic for "this tensor is row-sharded on axis 0 by TP".

    Mirrors the conventions used in HF transformers + vLLM model code:
    projection layers (``q_proj``, ``k_proj``, ``v_proj``, ``gate_proj``,
    ``up_proj``, ``down_proj``, ``o_proj``) and embeddings shard, while
    layer norms / biases stay replicated.

    This is intentionally simple — a real implementation reads the
    model's TP plan from HF config. Kept here so the prototype runs
    end-to-end without a HF dependency at planning time.
    """
    sharded_keywords = (
        ".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.",
        ".gate_proj.", ".up_proj.", ".down_proj.",
        ".embed_tokens.", ".lm_head.",
    )
    return any(kw in name for kw in sharded_keywords)
