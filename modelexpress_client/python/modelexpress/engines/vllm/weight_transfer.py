# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM ``WeightTransferEngine`` adapter for ModelExpress + NIXL.

This module is the **upstream-facing form** of all the Phase 2 / 3 / 4
work landed across this branch and the prime-rl follow-up PRs. It
wraps the v2 fat clients (:class:`MxV2RefitReceiver` +
:class:`MxV2TrainingPublisher`) behind vLLM's native ``WeightTransferEngine``
abstract base (introduced in the 2026-05-28 vLLM Native RL APIs blog),
so RL frameworks can pick it up via the standard four-phase lifecycle:

::

    from vllm import LLM
    from vllm.config import WeightTransferConfig
    import modelexpress.engines.vllm.weight_transfer  # noqa: F401 — registers "mx_nixl"

    llm = LLM(model="...", weight_transfer_config=WeightTransferConfig(backend="mx_nixl"))
    llm.init_weight_transfer_engine(WeightTransferInitRequest(
        init_info=MxInitInfo(
            mx_server_url="modelexpress-server:8001",
            model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
            worker_rank=0, agent_name="vllm-inference-r0", device_id=0,
        )
    ))

    # per training step:
    llm.start_weight_update()
    llm.update_weights(WeightTransferUpdateRequest(
        update_info=MxUpdateInfo(
            version=step,
            compile_target_filter={"cutlass_fp8"},
            target_tp_layout=None,  # matched-TP fast path; set for mixed-TP
        )
    ))
    llm.finish_weight_update()

What's wrapped from each phase:

* Phase 2 — heartbeat + freshest-per-rank dedup + same-rank-only filter
  in the discovery layer
* Phase 3a — ``compile_target`` + ``compile_metadata`` tagging on the
  publisher side (carried in v2 ``TensorDescriptorV2``)
* Phase 3b — ``compile_target_filter`` + ``required_compile_metadata``
  on the receiver side (refuses incompatible bytes before RDMA)
* Phase 4 — multi-source slice picker + stitching for mixed-TP /
  mixed-EP between trainer and inference

The engine handles two receive paths:

1. **Matched TP/EP** (the common case today): single-source same-rank
   pull via ``MxV2RefitReceiver.receive_from``. ``MxUpdateInfo.target_tp_layout``
   is ``None``.
2. **Mixed-TP** (e.g. trainer FSDP=4 ↔ inference TP=8): the multi-source
   plan is computed via ``discover_v2_sources_for_slice`` and stitched
   in ``receive_via_plan``. Caller sets ``MxUpdateInfo.target_tp_layout``.

Design rationale, comparison to vLLM's built-in NCCL / IPC backends, and
comparison to Anyscale's RDT (PR #43375) plugin are in
``pensieve/RL/PrimeRL/10_mx_weight_transfer_engine_design.md``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch
from torch import Tensor

from ...nemo_rl_v2 import (
    MxV2RefitReceiver,
    MxV2TrainingPublisher,
    TargetTPLayout,
    TrainerWorldLayout,
)
from ...shape_descriptors import COMPILE_TARGET_HF_RAW

logger = logging.getLogger("modelexpress.engines.vllm.weight_transfer")


# ----------------------------------------------------------------------------
# Init / Update / TrainerSend info dataclasses.
#
# We do NOT subclass vLLM's ``WeightTransferInitInfo`` / ``WeightTransferUpdateInfo``
# at module import time, because we want the adapter to be import-safe
# in environments where vLLM is not installed (tests, the publisher
# side, benchmark harnesses). We expose plain dataclasses and let the
# registration step (at the bottom of this file) do the subclass
# substitution against vLLM's bases if available.
# ----------------------------------------------------------------------------


@dataclass
class MxInitInfo:
    """Initialization data for the MX backend.

    Args:
        mx_server_url: gRPC URL of the ModelExpress server (e.g.
            ``"modelexpress-server.kavin.svc.cluster.local:8001"``).
        model_name: model identifier shared between trainer and inference.
            Receivers filter discovery by this exact string.
        worker_rank: this receiver's rank index. With ``MxV2RefitReceiver``'s
            same-rank-only default, the receiver pulls from the trainer
            rank with the same index.
        agent_name: NIXL agent name for this receiver. Conventionally
            ``f"vllm-inference-r{worker_rank}"``.
        device_id: CUDA device this receiver writes into.
        listen_port: optional NIXL listen port; ``None`` = auto-pick.
        publish_self_as_replica: if True, after a successful receive
            this engine calls ``publish_self_as_source`` so subsequent
            receivers can pull from this rank instead of the trainer
            (TensorHub-style pipeline replication / tree fan-out).
            Recommended ``True`` for elastic deployments.
    """

    mx_server_url: str
    model_name: str
    worker_rank: int
    agent_name: str
    device_id: int = 0
    listen_port: int | None = None
    publish_self_as_replica: bool = True


@dataclass
class MxUpdateInfo:
    """Per-refit update data.

    Args:
        version: monotonic step counter the trainer is at. Receiver pulls
            sources with ``training_step >= version``.
        target_tp_layout: receiver's local TP/EP slice descriptor (Phase 4).
            ``None`` (default) → matched-TP fast path: single-source
            same-rank pull via ``discover_v2_sources`` + ``receive_from``.
            Set to a ``TargetTPLayout`` when trainer and inference have
            different TP/EP layouts; engine then uses
            ``discover_v2_sources_for_slice`` + ``receive_via_plan``.
        compile_target_filter: optional whitelist of compile-target strings.
            ``None`` = accept anything (back-compat). Set to e.g.
            ``{"cutlass_fp8"}`` or ``{"cutlass_fp8", "hf_raw"}`` to refuse
            sources whose tensors are tagged with a target outside this set.
        required_compile_metadata: optional kv subset that every tensor's
            ``compile_metadata`` must agree with. Useful for pinning block
            sizes, scale layouts, kernel versions.
        timeout_seconds: cap on per-receive RDMA wait.
        same_rank_only: enforce same-rank trainer-to-inference peering
            (required on GCP GB200 multi-NIC fabrics where rdma-0..3
            are separate L3 subnets).
        dedup_freshest_per_rank: keep only the freshest published source
            per ``worker_rank`` (the bug class our Phase 2 PR codifies).
    """

    version: int
    target_tp_layout: TargetTPLayout | None = None
    compile_target_filter: set[str] | frozenset[str] | None = None
    required_compile_metadata: dict[str, Any] | None = None
    timeout_seconds: float = 300.0
    same_rank_only: bool = True
    dedup_freshest_per_rank: bool = True


@dataclass
class MxTrainerSendArgs:
    """Optional trainer-side args for :meth:`MxWeightTransferEngine.trainer_send_weights`.

    Trainers that drive sends through this engine pass these args at each
    publish to control how the bytes are tagged. The publisher itself is
    long-lived and should be reused across steps; only ``version`` and
    optionally ``compile_target`` / ``compile_metadata`` change per step.
    """

    publisher: MxV2TrainingPublisher       # long-lived, heartbeat-started
    version: int                            # the training step
    compile_target: str = COMPILE_TARGET_HF_RAW
    compile_metadata: dict[str, Any] | None = None
    # Per-tensor MoE expert metadata. Map tensor name → expert axis and
    # tuple of expert IDs this rank owns. Leave empty for non-expert tensors.
    expert_axis_map: dict[str, int] = field(default_factory=dict)
    owned_expert_ids: dict[str, tuple[int, ...]] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# The engine itself.
#
# We hold off on subclassing vLLM's ``WeightTransferEngine`` until the
# registration step (see bottom of file), so this module imports cleanly
# without vLLM. Tests can exercise the methods directly on this class.
# ----------------------------------------------------------------------------


class MxWeightTransferEngine:
    """ModelExpress + NIXL adapter for vLLM's WeightTransferEngine API.

    Receiver side wraps :class:`MxV2RefitReceiver` and exposes the four
    capabilities prime-rl currently bolts on by hand:

      - heartbeat-aware rendezvous (Phase 2)
      - ``compile_target`` / ``compile_metadata`` filtering (Phase 3a/3b)
      - multi-source slice picker for mixed-TP (Phase 4)
      - tree fan-out so newcomers pull from already-loaded peers, not
        the trainer (TensorHub pipeline pattern, opt-in via
        :attr:`MxInitInfo.publish_self_as_replica`)

    Trainer side wraps :class:`MxV2TrainingPublisher` via the optional
    :meth:`trainer_send_weights` classmethod. Trainers can drive
    publishes outside this engine — the method is a convenience for the
    case where the trainer wants the engine to own the publish.

    Args:
        init_info: optional pre-built init info, allowing the engine to
            be constructed and initialized in one go. If omitted, the
            caller must invoke :meth:`init_transfer_engine` before any
            :meth:`receive_weights` call.
    """

    # vLLM's WeightTransferEngine declares these class attributes
    # pointing at the request-info dataclasses. We populate them so the
    # factory can find our types post-registration.
    init_info_cls = MxInitInfo
    update_info_cls = MxUpdateInfo

    def __init__(self, init_info: MxInitInfo | None = None) -> None:
        self._receiver: MxV2RefitReceiver | None = None
        self._init_info: MxInitInfo | None = None
        if init_info is not None:
            self.init_transfer_engine(init_info)

    # ------------------------------------------------------------------
    # Receiver-side API (the WeightTransferEngine contract).
    # ------------------------------------------------------------------

    def init_transfer_engine(self, init_info: MxInitInfo) -> None:
        """Stand up the MX v2 receiver.

        NIXL register doesn't happen here — vLLM gives us the model's
        param buffers only after the engine is initialized, and the
        receiver's ``initialize()`` is what binds them. We defer to the
        first :meth:`receive_weights` call so the engine can be
        instantiated and configured before vLLM has built its workers.
        """
        self._init_info = init_info
        self._receiver = MxV2RefitReceiver(
            agent_name=init_info.agent_name,
            device_id=init_info.device_id,
            mx_server_url=init_info.mx_server_url,
            worker_rank=init_info.worker_rank,
            listen_port=init_info.listen_port,
        )
        logger.info(
            "MxWeightTransferEngine init: agent=%s worker_rank=%s server=%s",
            init_info.agent_name,
            init_info.worker_rank,
            init_info.mx_server_url,
        )

    def receive_weights(
        self,
        update_info: MxUpdateInfo,
        load_weights: Callable[[list[tuple[str, Tensor]]], None],
    ) -> None:
        """Pull weights via NIXL RDMA + feed them through vLLM's load_weights.

        Mode selection happens on ``update_info.target_tp_layout``:

          - ``None`` → matched-TP fast path. Calls ``discover_v2_sources``
            + ``receive_from`` (single source, same-rank). Applies the
            Phase 3 filters at discovery time so incompatible sources
            are rejected before any RDMA cycles are spent.
          - non-``None`` → mixed-TP / Phase-4 path. Calls
            ``discover_v2_sources_for_slice`` to build the
            ``SliceCoveragePlan``, then ``receive_via_plan`` to stitch
            the slice from N publisher ranks. Applies the same Phase 3
            filters at discovery time.

        Args:
            update_info: the per-step request descriptor. See
                :class:`MxUpdateInfo` for fields.
            load_weights: vLLM-provided callback. Each yielded tensor
                is fed in as a single-element list so vLLM's
                ``stacked_params_mapping`` can handle HF→fused name
                remapping per call (matching the convention used by
                NCCL / IPC / RDT backends).
        """
        if self._receiver is None or self._init_info is None:
            raise RuntimeError(
                "MxWeightTransferEngine.init_transfer_engine() must be called first"
            )

        # Lazy initialize: the v2 receiver needs to be initialize()'d
        # exactly once. We pass model_tensors=None because the current
        # v0 of this adapter uses the scratch-buffer path (it writes
        # into receiver-allocated buffers, then yields them for vLLM's
        # load_weights to consume — matching RDT's pattern).
        # ``None`` (not ``{}``) is the correct sentinel: NixlTransferManager
        # rejects empty descriptor lists, but ``MxRefitReceiver.initialize``
        # treats None as "skip register_memory; caller will register
        # per-receive". When the upstream API gets a register_destinations
        # hook (proposed extension, see design doc §5.1), this is where
        # we'd pre-register vLLM's named_parameters for zero-copy receive.
        if not self._receiver._initialized:
            self._receiver.initialize(model_tensors=None)

        # ----- Phase 4 path: mixed-TP / multi-source -----
        if update_info.target_tp_layout is not None:
            # Mixed-TP slice assembly inherently spans multiple trainer
            # ranks to cover one receiver-rank's window. The default
            # ``same_rank_only=True`` on update_info is correct for the
            # matched-TP fast path below (each receiver rank pulls from
            # the corresponding trainer rank), but it would prevent
            # discovery of the adjacent trainer ranks needed here. Only
            # honor the caller's same_rank_only if they set it
            # explicitly via the alternate ``mixed_tp_same_rank_only``
            # field (rare); otherwise force False for cross-rank lookup.
            mixed_tp_same_rank = getattr(
                update_info, "mixed_tp_same_rank_only", False,
            )
            plan = self._receiver.discover_v2_sources_for_slice(
                model_name=self._init_info.model_name,
                target_layout=update_info.target_tp_layout,
                min_version=update_info.version,
                same_rank_only=mixed_tp_same_rank,
                compile_target_filter=update_info.compile_target_filter,
                required_compile_metadata=update_info.required_compile_metadata,
            )
            if not plan.fully_covered:
                raise RuntimeError(
                    f"MxWeightTransferEngine: no covering source set for "
                    f"version={update_info.version}; missing={plan.missing}"
                )
            for name, tensor in self._receiver.receive_via_plan(
                plan, timeout_seconds=update_info.timeout_seconds
            ):
                load_weights([(name, tensor)])
        else:
            # ----- Fast path: matched-TP / single-source -----
            candidates = self._receiver.discover_v2_sources(
                model_name=self._init_info.model_name,
                min_version=update_info.version,
                same_rank_only=update_info.same_rank_only,
                compile_target_filter=update_info.compile_target_filter,
                required_compile_metadata=update_info.required_compile_metadata,
            )
            chosen = self._receiver.pick_best_source(candidates)
            if chosen is None:
                raise RuntimeError(
                    f"MxWeightTransferEngine: no source matches filters for "
                    f"version={update_info.version}; "
                    f"compile_target_filter={update_info.compile_target_filter}, "
                    f"required_compile_metadata={update_info.required_compile_metadata}"
                )
            # Scratch path: receiver allocates buffers matching the
            # publisher's layout, NIXL writes into them, we yield them
            # for the load_weights callback. This matches Anyscale's
            # RDT plugin pattern and works without pre-registered
            # model parameters — the common cold-start case for vLLM
            # and the only sensible mode for the benchmark harness.
            # Once vLLM exposes register_destinations, this can switch
            # to the zero-copy `receive_from` path (design doc §5.1).
            for name, tensor in self._receiver.receive_from_scratch(
                chosen, timeout_seconds=update_info.timeout_seconds
            ):
                load_weights([(name, tensor)])

        # Tree fan-out / pipeline replication: after a successful
        # receive, optionally publish this rank's buffers so subsequent
        # receivers (newcomers in an elastic deployment) can pull from
        # us instead of the trainer.
        #
        # Known limitation: this adapter's v0 uses ``receive_from_scratch``,
        # which writes into locally-allocated buffers that the receiver
        # does not track in ``_registered_buffers``. ``publish_self_as_source``
        # operates on ``_registered_buffers`` (set by
        # ``initialize(model_tensors=...)``), so calling it after a scratch
        # receive is a no-op — the receiver advertises nothing and later
        # workers silently fall through to the trainer instead of this
        # replica.
        #
        # Until the adapter is refactored to either (a) carry the scratch
        # buffers into ``_registered_buffers`` post-receive, or (b) switch
        # to the pre-registered ``receive_from`` fast path once vLLM
        # exposes ``register_destinations`` (design doc §5.1), tree fan-out
        # via this adapter is best skipped. Log loudly if it's requested
        # so the misconfiguration is visible instead of silent.
        if self._init_info.publish_self_as_replica:
            if not self._receiver._registered_buffers:
                logger.warning(
                    "MxWeightTransferEngine: publish_self_as_replica=True but "
                    "the receiver is in scratch-buffer mode "
                    "(model_tensors=None at initialize), so publish_self_as_source "
                    "would have nothing to advertise. Tree fan-out is a no-op for "
                    "this cycle. Disable publish_self_as_replica or pass "
                    "model_tensors at initialize."
                )
            else:
                try:
                    self._receiver.publish_self_as_source(
                        version=update_info.version,
                        model_name=self._init_info.model_name,
                    )
                except Exception as e:  # noqa: BLE001
                    # Pipeline replication is best-effort — it's an
                    # optimization for elastic deployments, not a
                    # correctness requirement.
                    logger.warning(
                        "MxWeightTransferEngine: publish_self_as_source failed: %s; "
                        "tree fan-out disabled for this cycle",
                        e,
                    )

    # ------------------------------------------------------------------
    # Optional trainer-side API.
    # ------------------------------------------------------------------

    @classmethod
    def trainer_send_weights(
        cls,
        iterator: Iterator[tuple[str, Tensor]],
        trainer_args: MxTrainerSendArgs,
    ) -> str:
        """Publish all tensors yielded by ``iterator`` as one v2 publish.

        The trainer typically calls this once per training step. The
        ``compile_target`` and ``compile_metadata`` on
        :class:`MxTrainerSendArgs` propagate into every tensor's
        :class:`TensorDescriptorV2`, which is the wire form that
        receivers filter on via :attr:`MxUpdateInfo.compile_target_filter`.

        Args:
            iterator: ``(name, tensor)`` pairs to publish. For DTensors,
                each tensor MUST be the rank-local shard
                (``.to_local()``) — never the gathered full tensor.
                That's the whole point of the v2 rank-to-rank design.
            trainer_args: see :class:`MxTrainerSendArgs`.

        Returns:
            the ``mx_source_id`` (16-hex hash) assigned by the server.
        """
        pub = trainer_args.publisher
        for name, tensor in iterator:
            is_expert = name in trainer_args.expert_axis_map
            pub.add_tensor(
                name=name,
                tensor=tensor,
                is_expert=is_expert,
                expert_axis=trainer_args.expert_axis_map.get(name, 0),
                owned_expert_ids=trainer_args.owned_expert_ids.get(name, ()),
                compile_target=trainer_args.compile_target,
                compile_metadata=trainer_args.compile_metadata,
            )
        return pub.publish(version=trainer_args.version)

    # ------------------------------------------------------------------
    # Metrics surface — what benchmarks + dashboards read.
    # ------------------------------------------------------------------

    @property
    def last_transfer_stats(self) -> "TransferStats | None":
        """The :class:`TransferStats` from the most recent
        :meth:`receive_weights` call. ``None`` before the first call.

        For the multi-source path, this reflects the LAST contributing
        source's stats; the full per-source history is on
        ``self.transfer_history``.
        """
        if self._receiver is None:
            return None
        return self._receiver._receiver.last_stats  # MxV2RefitReceiver → MxRefitReceiver

    @property
    def transfer_history(self) -> "list[TransferStats]":
        """Per-call :class:`TransferStats` history across the engine's
        lifetime. Each item corresponds to one underlying NIXL
        ``receive_from_source`` invocation."""
        if self._receiver is None:
            return []
        return self._receiver._receiver.history

    @property
    def last_discovery_seconds(self) -> float:
        """Wall time of the most recent ``discover_v2_sources`` call —
        the control-plane round-trip latency. Distinct from data-plane
        RDMA time."""
        if self._receiver is None:
            return 0.0
        return self._receiver._last_discovery_seconds


# ----------------------------------------------------------------------------
# Registration with vLLM's WeightTransferEngineFactory.
#
# We import vLLM lazily and try/except: if vLLM is not installed in this
# environment (publisher side, tests, harnesses), the module still loads
# and the class is usable directly — only the factory registration is
# skipped. The MX_WEIGHT_TRANSFER_AUTOREGISTER env var lets callers opt
# out of auto-registration even when vLLM IS installed (useful for
# environments where vLLM is present but the user wants a different
# backend name or doesn't want the side-effect).
# ----------------------------------------------------------------------------


def _register_with_vllm() -> bool:
    """Register ``MxWeightTransferEngine`` with vLLM's factory.

    Returns True on successful registration, False if vLLM isn't
    available or the user opted out via env var.
    """
    if os.environ.get("MX_WEIGHT_TRANSFER_AUTOREGISTER") == "0":
        return False

    try:
        from vllm.distributed.weight_transfer import WeightTransferEngineFactory
        from vllm.distributed.weight_transfer.base import (
            WeightTransferEngine as _VllmWeightTransferEngineBase,
        )
    except ImportError:
        logger.debug(
            "vLLM not available; MxWeightTransferEngine remains usable "
            "directly but is not registered with WeightTransferEngineFactory"
        )
        return False

    # Subclass to bind to vLLM's actual base. Without this, vLLM's
    # factory may reject our engine as not-a-WeightTransferEngine.
    # MxWeightTransferEngine must come FIRST in the MRO so its concrete
    # implementations (receive_weights, send_weights, etc.) resolve before
    # _VllmWeightTransferEngineBase's abstract definitions. With the base
    # first, the abstract methods would keep the subclass abstract and any
    # overlapping method names would resolve to the vLLM stub instead of
    # our implementation.
    class _MxEngineForVllm(MxWeightTransferEngine, _VllmWeightTransferEngineBase):
        pass

    try:
        WeightTransferEngineFactory.register_engine("mx_nixl", _MxEngineForVllm)
        logger.info("Registered MxWeightTransferEngine as backend='mx_nixl'")
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Failed to register MxWeightTransferEngine with vLLM: %s", e
        )
        return False


# Auto-register on import. Callers who don't want this can set
# MX_WEIGHT_TRANSFER_AUTOREGISTER=0 before importing this module.
_AUTOREGISTERED = _register_with_vllm()


__all__ = [
    "MxInitInfo",
    "MxUpdateInfo",
    "MxTrainerSendArgs",
    "MxWeightTransferEngine",
]
