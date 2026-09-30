# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete optional MILES protocol for NCCL M2N collective refit."""

from __future__ import annotations

import asyncio
import logging
import math
import operator
import os
from dataclasses import replace
from typing import Any
from uuid import uuid4

import grpc
import torch
import torch.distributed as dist
from modelexpress import auth

from ..rendezvous import CollectiveRendezvous
from ..types import MeshSpec, ParamPlan, Placement, ReshardPlan
from ._common import _dtype_label, _text
from ._common import _endpoint as _normalize_endpoint
from .miles import (
    CollectiveTopology,
    MilesPublisher,
    MilesTrainerSession,
)
from .wire import (
    CollectiveControl,
    encode_control,
    plan_to_wire,
    topology_to_wire,
)

logger = logging.getLogger("modelexpress_rl.collective.integrations.miles_protocol")

_DEFAULT_CONNECT_TIMEOUT_S = 10.0
_DEFAULT_PUBLISH_GROUPS = 1
_DEFAULT_ABI_VERSION = "miles-sglang-bf16-replicated-v1"


def _gloo_group():
    from miles.utils.distributed_utils import get_gloo_group

    return get_gloo_group()


def _arg(args: Any, name: str, default: Any = None) -> Any:
    value = getattr(args, name, None)
    return default if value is None else value


def _server_endpoint(args: Any) -> str:
    address = _arg(
        args,
        "modelexpress_server_address",
        os.environ.get("MX_SERVER_ADDRESS"),
    )
    if address is None:
        raise ValueError(
            "the ModelExpress M2N protocol needs the mx-server address: pass "
            "--modelexpress-server-address <host:port> or set MX_SERVER_ADDRESS"
        )
    return _normalize_endpoint(address)


def _publish_group_count(args: Any) -> int:
    raw = _arg(
        args,
        "modelexpress_m2n_publish_groups",
        os.environ.get("MX_MILES_PUBLISH_GROUPS"),
    )
    if raw is None:
        return _DEFAULT_PUBLISH_GROUPS

    def invalid() -> ValueError:
        return ValueError(
            f"modelexpress_m2n_publish_groups must be a positive integer, got {raw!r}"
        )

    if isinstance(raw, bool):
        raise invalid()
    if isinstance(raw, str):
        try:
            count = int(raw.strip(), 10)
        except ValueError:
            raise invalid() from None
    else:
        try:
            count = operator.index(raw)
        except TypeError:
            raise invalid() from None
    if count < 1:
        raise invalid()
    return count


def _connect_timeout_s(args: Any) -> float:
    raw = _arg(
        args,
        "modelexpress_m2n_connect_timeout_s",
        os.environ.get("MX_MILES_CONNECT_TIMEOUT_S"),
    )
    if raw is None:
        return _DEFAULT_CONNECT_TIMEOUT_S

    def invalid() -> ValueError:
        return ValueError(
            "modelexpress_m2n_connect_timeout_s must be a positive number of "
            f"seconds, got {raw!r}"
        )

    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        raise invalid() from None
    if isinstance(raw, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise invalid()
    return timeout


def _abi_version(args: Any) -> str:
    requested = _arg(
        args,
        "modelexpress_m2n_abi_version",
        os.environ.get("MX_MILES_ABI_VERSION"),
    )
    if requested is None:
        return _DEFAULT_ABI_VERSION
    return _text(requested, "modelexpress_m2n_abi_version")


def _entry_wire_bytes(entry: ParamPlan) -> int:
    if _dtype_label(entry.dtype) != "bfloat16":
        raise ValueError(f"{entry.name}: unsupported collective dtype {entry.dtype!r}")
    size = torch.bfloat16.itemsize
    for extent in entry.global_shape:
        size *= int(extent)
    return size


def _chunk_publish_groups(
    entries: list[ParamPlan],
    requested: int,
) -> tuple[tuple[str, ...], ...]:
    """Split canonical-order plan entries into byte-balanced publish groups.

    Grouping is purely a scheduling decision: both sides canonical-sort
    within their own groups, so any contiguous chunking of a
    canonical-ordered plan yields the same per-lane tensor sequence on the
    wire. ``requested`` is a cap, not a quota: the result never exceeds the
    entry count, and an entry larger than the byte target forms its own
    group unless the group-count cap has already been reached — once it
    binds, the remaining entries accumulate into the final group, which may
    exceed the target (sizes [2, 100, 2, 100] with ``requested=3`` yield
    ``([a], [b], [c, d])``). Every group is non-empty and the result covers
    every entry exactly once, in canonical plan order.
    """
    if requested < 1:
        raise ValueError(f"publish group count must be positive, got {requested}")
    if not entries:
        raise ValueError("the frozen plan has no bulk parameters")
    group_count = min(requested, len(entries))
    total = sum(_entry_wire_bytes(entry) for entry in entries)
    target = max(1, -(-total // group_count))
    groups: list[tuple[str, ...]] = []
    current: list[str] = []
    current_bytes = 0
    for entry in entries:
        entry_bytes = _entry_wire_bytes(entry)
        if (
            current
            and current_bytes + entry_bytes > target
            and len(groups) < group_count - 1
        ):
            groups.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(entry.name)
        current_bytes += entry_bytes
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _check_response(response: Any) -> Any:
    if isinstance(response, dict):
        success = response.get("success")
        message = response.get("message", "")
    else:
        success = getattr(response, "success", None)
        message = getattr(response, "message", "")
    # Fail closed on a control boundary: only an explicit success=True is a
    # success. A missing field or an unexpected shape is a broken receiver,
    # not a quiet pass.
    if success is not True:
        raise RuntimeError(
            str(message or "SGLang returned no success=true collective response")
        )
    return response


def _log_late_future_result(future: Any) -> None:
    try:
        future.result()
    except BaseException:
        logger.warning("a dropped MILES generator future failed late", exc_info=True)


def _retire_dropped_futures(futures: list) -> None:
    """Cancel or observe futures a failure arm will not await.

    A dropped future that settles later would race teardown, and its
    exception would never be retrieved. Cancel whatever has not started
    (``cancel()`` reports rather than raises) and register an observer for
    the rest so late failures reach the log.
    """
    for future in futures:
        if future.cancel():
            continue
        try:
            future.add_done_callback(_log_late_future_result)
        except BaseException:
            logger.warning(
                "observing a dropped MILES generator future failed",
                exc_info=True,
            )


def _await_endpoint_ready(channel: Any, *, endpoint: str, timeout_s: float) -> None:
    """Fail loudly at connect time when the mx-server is unreachable."""
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_s)
    except Exception as error:
        raise RuntimeError(
            f"cannot reach the ModelExpress server at {endpoint!r} within "
            f"{timeout_s:.1f}s; check modelexpress_server_address / "
            "MX_SERVER_ADDRESS and that the mx-server is running"
        ) from error


def _validate_args(args: Any) -> None:
    """Validate the ModelExpress M2N arguments before model load.

    Attached to ``build_protocol`` as the MILES ``validate_args`` hook so a
    missing or malformed mx-server address fails during argument validation
    instead of deep inside the first weight-update round.
    """
    try:
        _server_endpoint(args)
    except ValueError as error:
        raise ValueError(f"invalid modelexpress_server_address: {error}") from error
    run_id = _arg(args, "modelexpress_m2n_run_id", os.environ.get("MX_MILES_RUN_ID"))
    if run_id is not None:
        _text(run_id, "modelexpress_m2n_run_id")
    _publish_group_count(args)
    _connect_timeout_s(args)
    _abi_version(args)


class MilesCollectiveProtocolCore:
    """MILES bucket-stream bridge used by the lazy protocol factory.

    Maps the ModelExpress trainer client flow onto the MILES weight-transfer
    seam: ``begin_sync`` materializes and validates the base-weight stream,
    freezes the plan/topology contract, and arms the round. The first
    ``send_bucket`` prepares the trainer session and generator fan-out
    (inside the engine pause window) and opens the round; each bucket drains
    the publish groups its tensors complete, in canonical plan order.
    ``finalize`` finishes the trainer round and waits for the generator
    fan-out.
    """

    def __init__(self, args: Any) -> None:
        self.args = args
        self.rollout_engines = None
        self.is_sender: bool | None = None
        # miles' updater reads protocol.group_name for its progress display;
        # WeightTransferProtocol.__init__ sets "miles", so name this path.
        self.group_name = "modelexpress-m2n"
        self._parallel_state = None
        self._engine_gpu_counts: tuple[int, ...] = ()
        self._engine_gpu_offsets: tuple[int, ...] = ()
        self._tensors: dict[str, torch.Tensor] | None = None
        self._canonical_shapes: dict[str, tuple[int, ...]] | None = None
        self._topology: CollectiveTopology | None = None
        self._plan: ReshardPlan | None = None
        self._publish_groups: tuple[tuple[str, ...], ...] | None = None
        self._group_of: dict[str, int] = {}
        self._publish_group_count = _publish_group_count(args)
        self._connect_timeout_s = _connect_timeout_s(args)
        requested_run_id = _arg(
            args,
            "modelexpress_m2n_run_id",
            os.environ.get("MX_MILES_RUN_ID"),
        )
        self._run_id = (
            _text(requested_run_id, "modelexpress_m2n_run_id")
            if requested_run_id is not None
            else None
        )
        self._run_id_agreed = False
        self._session: MilesTrainerSession | None = None
        self._channel = None
        self._rendezvous = None
        self._closed = False
        self._close_pending = False
        self._round_version: str | None = None
        self._round_begun = False
        self._round_seen: set[str] = set()
        self._pending: list[set[str]] = []
        self._local_names: set[str] = set()
        self._next_group = 0
        self._round_futures: list = []

    def _validate_selector(self, selector: str) -> None:
        if selector not in ("all", "target"):
            raise ValueError("MILES NCCL M2N supports the base target model only")
        if (
            getattr(self.args, "sglang_speculative_algorithm", None)
            and selector != "target"
        ):
            raise ValueError(
                "MILES NCCL M2N supports speculation only with a frozen draft "
                "and target-only weight updates; disable trainer MTP layers."
            )

    def connect(
        self,
        rollout_engines,
        engine_gpu_counts,
        engine_gpu_offsets,
        parallel_state,
        placement,
        selector,
    ) -> None:
        if self._closed:
            raise RuntimeError(
                "the MILES NCCL M2N protocol is closed; reconnecting requires a "
                "new protocol instance"
            )
        if self._round_version is not None:
            raise RuntimeError(
                "cannot reconnect while the round for version "
                f"{self._round_version!r} is in flight"
            )
        self._validate_selector(selector)
        if getattr(placement, "gather_pp", True):
            raise ValueError(
                "MILES NCCL M2N requires PP-local HF tensors "
                "(WeightUpdatePlacement.gather_pp=False)"
            )
        if not getattr(placement, "gather_tp", True) or not getattr(
            placement, "gather_ep", True
        ):
            raise ValueError("MILES NCCL M2N requires TP and EP gathering")
        if engine_gpu_counts is None or engine_gpu_offsets is None:
            raise ValueError(
                "MILES NCCL M2N requires explicit engine GPU counts and offsets"
            )
        counts = tuple(int(value) for value in engine_gpu_counts)
        offsets = tuple(int(value) for value in engine_gpu_offsets)
        if len(counts) != len(rollout_engines) or len(offsets) != len(counts):
            raise ValueError("engine GPU topology must cover every rollout engine")
        if any(value <= 0 for value in counts) or any(value < 0 for value in offsets):
            raise ValueError(
                "engine GPU counts must be positive and offsets non-negative"
            )
        slots = [
            offset + local_rank
            for offset, count in zip(offsets, counts, strict=True)
            for local_rank in range(count)
        ]
        if len(slots) != len(set(slots)):
            raise ValueError("engine GPU ranges must not overlap")
        if any(
            getattr(getattr(parallel_state, name, None), "size", None) != 1
            for name in ("tp", "ep", "etp", "cp", "intra_dp", "indep_dp")
        ):
            raise ValueError(
                "the MILES NCCL M2N path requires exactly one source "
                "rank per PP partition (TP/EP/CP/DP must all be one)"
            )
        # miles re-calls connect() when the rollout engine set heals or the
        # trainer goes stale. A stale session must not survive into the next
        # round: the first send_bucket would skip preparation and fan
        # run_round out to engines that never received prepare. Tear the live
        # session down (best-effort; the old engines may already be broken)
        # so the next begin_sync re-prepares against the new engine set. The
        # frozen contract is re-validated at the next begin_sync, so a heal
        # that changes the engine GPU topology fails closed there instead of
        # silently publishing into a reshaped engine set.
        if (
            self._session is not None
            or self._rendezvous is not None
            or self._channel is not None
        ):
            logger.info(
                "MILES NCCL M2N reconnect: tearing down the previous session"
            )
            self._teardown_for_reconnect()
        self.rollout_engines = tuple(rollout_engines)
        self._engine_gpu_counts = counts
        self._engine_gpu_offsets = offsets
        self._parallel_state = parallel_state
        # Every trainer rank owns its PP partition's tensors and drives its
        # own lane, so every rank is a sender.
        self.is_sender = True

    def begin_sync(self, weight_version: int, iter_buckets) -> bool:
        """Materialize and validate the base-weight stream and arm the round.

        Each canonical tensor is validated and copied into its stable wire
        buffer as it arrives, so the materialized stream is never pinned whole;
        ``send_bucket`` only tracks arrival order against those buffers. The
        copies run on the caller's current stream; the session orders that
        producer stream before the lane streams at round begin, so every
        begin_sync for a session must run on the same ambient stream. Session
        preparation is deferred to the first ``send_bucket`` so the channel,
        rendezvous join, and NCCL bootstrap happen inside the engine pause
        window rather than before it.
        """
        canonical_shapes: dict[str, tuple[int, ...]] | None = None
        prepared_tensors: dict[str, torch.Tensor] | None = None
        local_exception: BaseException | None = None
        local_error = ""
        try:
            if self.rollout_engines is None:
                raise RuntimeError("connect must run before begin_sync")
            if self._closed:
                raise RuntimeError("the MILES NCCL M2N protocol is closed")
            if self._round_version is not None:
                raise RuntimeError(
                    f"the round for version {self._round_version!r} never finalized"
                )
            first_round = self._canonical_shapes is None
            frozen: dict[str, tuple[int, ...]]
            prepared: dict[str, torch.Tensor]
            if first_round:
                frozen = {}
                prepared = {}
            else:
                if self._tensors is None:
                    raise RuntimeError("wire buffers are unavailable")
                frozen = self._canonical_shapes or {}
                prepared = self._tensors
            seen: set[str] = set()
            shapes: dict[str, tuple[int, ...]] = {}
            for bucket in iter_buckets(materialize=True):
                for name, tensor in bucket:
                    if ":" in name:
                        raise ValueError("LoRA/adaptor tensors are not supported")
                    if name in seen:
                        raise ValueError(f"duplicate HF weight name {name!r}")
                    seen.add(name)
                    if tensor.ndim == 0:
                        raise ValueError(f"{name}: scalar weights are not supported")
                    if tensor.dtype is not torch.bfloat16:
                        raise ValueError(
                            f"{name}: only BF16 base weights are supported"
                        )
                    if not tensor.is_contiguous():
                        raise ValueError(
                            f"{name}: materialized HF weight is not contiguous"
                        )
                    shape = tuple(int(dim) for dim in tensor.shape)
                    if first_round:
                        # The contiguity check above already pins the memory
                        # format, so clone() preserves it.
                        wire = tensor.clone()
                        prepared[name] = wire
                    else:
                        # A mid-stream failure can leave earlier wire buffers
                        # already overwritten; that is safe because a failed
                        # begin_sync never arms a round, and the next begin_sync
                        # rewrites every buffer before use.
                        if frozen.get(name) != shape:
                            raise RuntimeError(
                                "MILES tensor names or canonical shapes changed"
                            )
                        prepared[name].copy_(tensor)
                    shapes[name] = shape
            if not shapes:
                raise ValueError("MILES produced no base-model tensors")
            if not first_round and shapes.keys() != frozen.keys():
                raise RuntimeError("MILES tensor names or canonical shapes changed")
            canonical_shapes = shapes
            prepared_tensors = prepared
        except BaseException as error:
            local_exception = error
            local_error = repr(error)
        trainer_world = dist.get_world_size()
        errors = [""] * trainer_world
        dist.all_gather_object(errors, local_error, group=_gloo_group())
        failures = [
            f"rank {rank}: {error}" for rank, error in enumerate(errors) if error
        ]
        if failures:
            if trainer_world == 1 and local_exception is not None:
                raise local_exception
            error = RuntimeError(
                "MILES NCCL M2N begin_sync preparation failed: "
                + "; ".join(failures[:4])
            )
            if local_exception is not None:
                raise error from local_exception
            raise error
        if canonical_shapes is None or prepared_tensors is None:
            raise RuntimeError("MILES NCCL M2N preparation produced no tensor state")
        if self._canonical_shapes is None:
            self._canonical_shapes = canonical_shapes
            self._tensors = prepared_tensors
        self._build_frozen_contract()
        self._arm_round(weight_version)
        return True

    def _arm_round(self, weight_version: int) -> None:
        # begin_sync already rejected an unfinalized round before arming.
        if self._plan is None or self._publish_groups is None:
            raise RuntimeError("the frozen plan is unavailable")
        rank = dist.get_rank()
        local_names = {
            entry.name for entry in self._plan.bulk if entry.partition_id == rank
        }
        self._local_names = local_names
        self._pending = [set(group) & local_names for group in self._publish_groups]
        self._round_seen = set()
        self._next_group = 0
        self._round_begun = False
        self._round_futures = []
        self._round_version = str(weight_version)

    def _disarm_round(self) -> None:
        self._round_version = None
        self._round_begun = False
        self._round_seen = set()
        self._pending = []
        self._local_names = set()
        self._next_group = 0
        self._round_futures = []

    def _build_frozen_contract(self) -> None:
        if self._tensors is None:
            raise RuntimeError("begin_sync has not materialized weights")
        trainer_world = dist.get_world_size()
        partition_count = int(self._parallel_state.pp.size)
        if trainer_world != partition_count:
            raise ValueError(
                "the trainer world must contain exactly one rank per PP partition"
            )
        # One rank source: dist.get_rank() is what _arm_round and
        # _prepare_sessions use for ownership, so the manifest must agree.
        # The pure-PP connect gate makes the two indices equal; if that gate
        # is ever relaxed, fail closed here rather than publish under a
        # partition id the rest of the round does not use.
        rank = dist.get_rank()
        pp_rank = int(self._parallel_state.pp.rank)
        if pp_rank != rank:
            raise ValueError(
                "the trainer's distributed rank must equal its PP partition "
                f"index: rank {rank} != pp.rank {pp_rank}"
            )
        if not self._run_id_agreed:
            requested_run_id = self._run_id
            # One collective settles both branches: every rank contributes
            # its requested run id (or None) plus a generated fallback, so
            # the no-config case needs no second gather.
            candidates: list[tuple[str | None, str] | None] = [None] * trainer_world
            dist.all_gather_object(
                candidates,
                (requested_run_id, uuid4().hex),
                group=_gloo_group(),
            )
            gathered = [candidate for candidate in candidates if candidate is not None]
            requested_ids = [requested for requested, _generated in gathered]
            configured_run_ids = {
                run_id for run_id in requested_ids if run_id is not None
            }
            if configured_run_ids:
                if len(configured_run_ids) != 1 or any(
                    run_id is None for run_id in requested_ids
                ):
                    raise ValueError(
                        "modelexpress_m2n_run_id must be set identically on "
                        "every trainer rank"
                    )
                agreed_run_id = next(iter(configured_run_ids))
            else:
                agreed_run_id = gathered[0][1]
            self._run_id = _text(agreed_run_id, "modelexpress_m2n_run_id")
            self._run_id_agreed = True
        if self._run_id is None:
            raise RuntimeError("MILES NCCL M2N run identity is unavailable")
        slot_prefix = f"{self._run_id}:"
        generator_slots = tuple(
            f"{slot_prefix}generator-{slot}"
            for offset, count in zip(
                self._engine_gpu_offsets,
                self._engine_gpu_counts,
                strict=True,
            )
            for slot in range(offset, offset + count)
        )
        topology = CollectiveTopology(
            # The rendezvous model identity is an adapter constant, not the
            # HF model id: miles' parser has no `model` argument, and the
            # deployed receivers form groups against this exact value.
            model_name="miles-model",
            trainer_slots=tuple(
                f"{slot_prefix}trainer-{rank}" for rank in range(trainer_world)
            ),
            generator_slots=generator_slots,
            source_partition_count=partition_count,
            m2n_abi_version=_abi_version(self.args),
        )
        if self._plan is not None:
            # Later rounds: the manifest all-gather and plan rebuild ran on
            # the first round, and their inputs are frozen — the wire buffers
            # and canonical shapes are pinned (begin_sync only copy_()s into
            # them after revalidating names and shapes), the trainer world is
            # static for the process group's lifetime, and the run id is
            # agreed. Re-gathering would return the round-one manifest, so
            # skip it; the skip decision itself is uniform because _plan is
            # set on every rank or on none (the post-gather validation is a
            # deterministic function of the identical gathered manifest). The
            # topology is still rebuilt and compared every round: a reconnect
            # that heals into a reshaped engine GPU topology must fail closed.
            if topology != self._topology:
                raise RuntimeError(
                    "MILES tensor names, shapes, or topology changed"
                )
            return
        src_mesh = MeshSpec((1,), rank_offset=0)
        dst_mesh = MeshSpec((len(generator_slots),), rank_offset=1)
        local_manifest = [
            (
                name,
                tuple(int(dim) for dim in tensor.shape),
                "bfloat16",
                rank,
            )
            for name, tensor in self._tensors.items()
        ]
        manifests: list[list[tuple[str, tuple[int, ...], str, int]] | None] = [
            None
        ] * trainer_world
        dist.all_gather_object(manifests, local_manifest, group=_gloo_group())
        ordered_manifest = [
            item for manifest in manifests if manifest is not None for item in manifest
        ]
        names = [name for name, _shape, _dtype, _owner in ordered_manifest]
        if len(names) != len(set(names)):
            raise ValueError("PP-local HF tensor names must be disjoint across ranks")
        owners = {owner for _name, _shape, _dtype, owner in ordered_manifest}
        if owners != set(range(partition_count)):
            raise ValueError(
                "every PP partition must contribute at least one canonical tensor"
            )
        entries = [
            ParamPlan(
                name=name,
                global_shape=shape,
                dtype=dtype,
                partition_id=owner,
                src_mesh=src_mesh,
                src_placements=(Placement.replicate(),),
                dst_mesh=dst_mesh,
                dst_placements=(Placement.replicate(),),
            )
            for name, shape, dtype, owner in ordered_manifest
        ]
        # Canonical order is the wire contract: both sides canonical-sort
        # within their own layer groups, so a canonical-ordered plan makes
        # every contiguous grouping produce the same per-lane tensor
        # sequence no matter where the group boundaries fall.
        entries.sort(key=lambda entry: entry.canonical())
        groups = _chunk_publish_groups(entries, self._publish_group_count)
        group_keys = {
            name: f"publish-group-{index}"
            for index, group in enumerate(groups)
            for name in group
        }
        plan = ReshardPlan(
            bulk=[
                replace(entry, group_key=group_keys[entry.name]) for entry in entries
            ],
            source_partition_count=partition_count,
        )
        self._topology = topology
        self._plan = plan
        self._publish_groups = groups
        self._group_of = {
            name: index for index, group in enumerate(groups) for name in group
        }

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        """Publish each frozen plan group as soon as this bucket completes it.

        The bucket tensors were already copied into the stable wire buffers
        during ``begin_sync``; this hook only tracks which canonical names
        arrived so each publish group goes out once its local entries are
        complete. Groups publish strictly in canonical plan order.
        """
        if self._round_version is None:
            raise RuntimeError("begin_sync must arm a round before send_bucket")
        if self._closed:
            raise RuntimeError("the MILES NCCL M2N protocol is closed")
        if not bucket:
            return
        version = self._round_version
        names = [name for name, _tensor in bucket]
        # A bucket that names unknown, foreign, or repeated tensors means the
        # trainer's bucket stream diverged from the frozen contract — a bug,
        # not a transient error. The generators are already paused mid-round,
        # so continuing a diverged round would risk publishing a wrong-plan
        # weight set; close the protocol like any other mid-round failure.
        try:
            unknown = [name for name in names if name not in self._group_of]
            if unknown:
                raise ValueError(
                    "MILES NCCL M2N bucket carries tensors outside the frozen plan: "
                    f"{unknown[:5]}"
                )
            foreign = [name for name in names if name not in self._local_names]
            if foreign:
                raise ValueError(
                    "MILES NCCL M2N bucket carries tensors owned by another PP "
                    f"partition: {foreign[:5]}"
                )
            repeated = [name for name in names if name in self._round_seen]
            if repeated:
                raise ValueError(
                    "MILES NCCL M2N bucket repeats tensors already seen in round "
                    f"{version}: {repeated[:5]}"
                )
        except BaseException as error:
            self._close_preserving(error)
            raise
        if not self._round_begun:
            if self._session is None:
                try:
                    self._prepare_sessions()
                except BaseException as error:
                    self._close_preserving(error)
                    raise
            self._begin_round(version)
        try:
            for name in names:
                self._round_seen.add(name)
                self._pending[self._group_of[name]].discard(name)
            self._drain_ready_groups(version)
        except BaseException as error:
            self._close_preserving(error)
            raise

    def _begin_round(self, version: str) -> None:
        session = self._session
        if session is None:
            raise RuntimeError("collective sessions were not prepared")
        rank = dist.get_rank()
        futures: list = []
        submission_error = ""
        if rank == 0:
            operation_id = (
                f"miles-{session.membership.group_id}-weight-version-{version}"
            )
            try:
                futures = self._generator_futures(
                    "run_round",
                    version=version,
                    operation_id=operation_id,
                )
            except BaseException as error:
                submission_error = repr(error)
        submission_errors = [""] * dist.get_world_size()
        dist.all_gather_object(
            submission_errors,
            submission_error,
            group=_gloo_group(),
        )
        submission_failures = [
            f"rank {index}: {error}"
            for index, error in enumerate(submission_errors)
            if error
        ]
        if submission_failures:
            primary = RuntimeError(
                "MILES NCCL M2N round generator submission failed: "
                + "; ".join(submission_failures[:4])
            )
            self._close_preserving(primary)
            raise primary
        self._round_futures = futures
        try:
            session.begin_round(version=version)
        except BaseException as error:
            self._close_preserving(error)
            raise
        self._round_begun = True

    def _drain_ready_groups(self, version: str) -> tuple[tuple[str, ...], ...]:
        if self._session is None or self._publish_groups is None:
            raise RuntimeError("collective sessions were not prepared")
        groups = self._publish_groups
        while self._next_group < len(groups) and not self._pending[self._next_group]:
            self._session.publish_group(
                version=version,
                layer_group_id=self._next_group,
            )
            self._next_group += 1
        return groups

    def after_base_weights(self) -> None:
        """No-op hook: publish groups drain incrementally in send_bucket."""

    def finalize(self, weight_version: int) -> None:
        """Finish the armed round and wait for the generator fan-out."""
        version = str(weight_version)
        if self._round_version is None:
            raise RuntimeError("begin_sync must arm a round before finalize")
        if version != self._round_version:
            raise RuntimeError(
                f"finalize names version {version!r}, but the round in flight "
                f"is {self._round_version!r}"
            )
        rank = dist.get_rank()
        trainer_world = dist.get_world_size()
        local_exception: BaseException | None = None
        local_error = ""
        try:
            if not self._round_begun:
                raise RuntimeError(
                    "MILES NCCL M2N received no weight buckets for the armed round"
                )
            if self._closed or self._session is None:
                raise RuntimeError("the MILES NCCL M2N protocol closed mid-round")
            groups = self._drain_ready_groups(version)
            if self._next_group != len(groups):
                missing = sorted(self._pending[self._next_group])
                raise RuntimeError(
                    "MILES NCCL M2N bucket stream ended before publish group "
                    f"{self._next_group} completed; missing {missing[:5]}"
                )
            self._session.finish_round(version=version)
        except BaseException as error:
            local_exception = error
            local_error = repr(error)
        errors = [""] * trainer_world
        dist.all_gather_object(errors, local_error, group=_gloo_group())
        futures_error = ""
        if rank == 0:
            if any(errors):
                _retire_dropped_futures(self._round_futures)
            else:
                try:
                    self._wait_generator_futures(self._round_futures)
                except BaseException as error:
                    futures_error = repr(error)
        status = [futures_error]
        dist.broadcast_object_list(status, src=0, group=_gloo_group())
        failures = [error for error in errors + status if error]
        self._disarm_round()
        if failures:
            primary = RuntimeError(
                "MILES NCCL M2N round failed: " + "; ".join(failures[:4])
            )
            self._close_preserving(primary)
            if local_exception is not None:
                raise primary from local_exception
            raise primary

    def after_engines_resumed(self) -> None:
        """No-op hook: the generator fan-out already settled in finalize."""

    async def _send_control(
        self,
        client: Any,
        control: CollectiveControl,
        *,
        plan_wire: dict | None = None,
        topology_wire: dict | None = None,
    ):
        response = await client.update_weights_from_distributed(
            names=[],
            dtypes=[],
            shapes=[],
            group_name=encode_control(
                control, plan_wire=plan_wire, topology_wire=topology_wire
            ),
            flush_cache=False,
            selector="target",
        )
        return _check_response(response)

    async def _send_controls(
        self,
        requests: list[tuple[Any, CollectiveControl]],
        *,
        plan_wire: dict | None = None,
        topology_wire: dict | None = None,
    ):
        # return_exceptions=True: every engine settles before the first error
        # propagates, so teardown never races an in-flight control RPC.
        results = await asyncio.gather(
            *(
                self._send_control(
                    client, control, plan_wire=plan_wire, topology_wire=topology_wire
                )
                for client, control in requests
            ),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        # Only the first error propagates; the rest would vanish silently.
        for error in errors[1:]:
            logger.warning("MILES NCCL M2N generator control failed: %r", error)
        if errors:
            raise errors[0]
        return results

    def _generator_futures(self, action: str, **kwargs):
        from miles.utils import async_utils

        if self.rollout_engines is None:
            raise RuntimeError("rollout engines are not connected")
        prepare = action == "prepare"
        # Every generator receives the same endpoint and the same encoded
        # plan/topology; resolve and encode them once per fan-out.
        endpoint = _server_endpoint(self.args) if prepare else None
        requests = []
        generator_slot_offset = 0
        for client, count in zip(
            self.rollout_engines,
            self._engine_gpu_counts,
            strict=True,
        ):
            control = CollectiveControl(
                action=action,
                plan=self._plan if prepare else None,
                topology=self._topology if prepare else None,
                generator_slot_offset=generator_slot_offset if prepare else None,
                endpoint=endpoint,
                **kwargs,
            )
            requests.append((client, control))
            generator_slot_offset += count
        if not requests:
            return []
        plan_wire = plan_to_wire(self._plan) if prepare else None
        topology_wire = topology_to_wire(self._topology) if prepare else None
        coroutine = self._send_controls(
            requests, plan_wire=plan_wire, topology_wire=topology_wire
        )
        try:
            return [async_utils.submit(coroutine)]
        except BaseException:
            coroutine.close()
            raise

    @staticmethod
    def _wait_generator_futures(futures) -> None:
        if not futures:
            return
        from miles.utils import async_utils

        async_utils.wait_futures(futures)

    def _prepare_sessions(self) -> None:
        if self._session is not None:
            return
        if (
            self._tensors is None
            or self._topology is None
            or self._plan is None
            or self._publish_groups is None
        ):
            raise RuntimeError("begin_sync must run before session preparation")
        rank = dist.get_rank()
        # Every local step is inside the fan-out: a failure on one rank must
        # reach the shared gather instead of stranding its peers there.
        local_error = ""
        try:
            local_tensors = dict(self._tensors)
            expected_local_names = {
                entry.name for entry in self._plan.bulk if entry.partition_id == rank
            }
            if set(local_tensors) != expected_local_names:
                missing = sorted(expected_local_names - set(local_tensors))
                unexpected = sorted(set(local_tensors) - expected_local_names)
                raise ValueError(
                    f"PP partition {rank} tensor ownership does not match the global "
                    f"plan (missing={missing[:5]}, unexpected={unexpected[:5]})"
                )
            if not local_tensors:
                raise ValueError(f"PP partition {rank} owns no collective parameters")
            endpoint = _server_endpoint(self.args)
            channel = auth.with_auth(grpc.insecure_channel(endpoint))
            try:
                _await_endpoint_ready(
                    channel,
                    endpoint=endpoint,
                    timeout_s=self._connect_timeout_s,
                )
            except BaseException:
                try:
                    channel.close()
                except BaseException:
                    logger.warning(
                        "closing the unreachable ModelExpress channel failed",
                        exc_info=True,
                    )
                raise
            self._channel = channel
            rendezvous = CollectiveRendezvous(channel)
            self._rendezvous = rendezvous
            publisher = MilesPublisher(
                plan=self._plan,
                source_partition=rank,
                tensors=local_tensors,
            )
            self._session = MilesTrainerSession.create(
                rendezvous=rendezvous,
                topology=self._topology,
                publisher=publisher,
                source_partition=rank,
                slot_id=self._topology.trainer_slots[rank],
                worker_id=f"miles-{self._topology.trainer_slots[rank]}-{uuid4().hex}",
                index_in_role=rank,
                layer_groups=self._publish_groups,
                device=next(iter(local_tensors.values())).device,
            )
        except BaseException as error:
            local_error = repr(error)
        setup_errors = [""] * dist.get_world_size()
        dist.all_gather_object(setup_errors, local_error, group=_gloo_group())
        setup_failures = [
            f"rank {index}: {error}"
            for index, error in enumerate(setup_errors)
            if error
        ]
        if setup_failures:
            primary = RuntimeError(
                "MILES NCCL M2N local session setup failed: "
                + "; ".join(setup_failures[:4])
            )
            self._close_preserving(primary)
            raise primary
        session = self._session
        if session is None:
            raise RuntimeError("collective session was not prepared")

        generator_futures = []
        submission_error = ""
        if rank == 0:
            try:
                generator_futures = self._generator_futures("prepare")
            except BaseException as error:
                submission_error = repr(error)
        submission_errors = [""] * dist.get_world_size()
        dist.all_gather_object(
            submission_errors,
            submission_error,
            group=_gloo_group(),
        )
        submission_failures = [
            f"rank {index}: {error}"
            for index, error in enumerate(submission_errors)
            if error
        ]
        if submission_failures:
            primary = RuntimeError(
                "MILES NCCL M2N generator preparation submission failed: "
                + "; ".join(submission_failures[:4])
            )
            self._close_preserving(primary)
            raise primary

        local_error = ""
        try:
            session.prepare()
        except BaseException as error:
            local_error = repr(error)
        if rank == 0:
            if local_error:
                _retire_dropped_futures(generator_futures)
            else:
                try:
                    self._wait_generator_futures(generator_futures)
                except BaseException as error:
                    local_error = repr(error)
        errors = [""] * dist.get_world_size()
        dist.all_gather_object(errors, local_error, group=_gloo_group())
        failures = [error for error in errors if error]
        if failures:
            primary = RuntimeError(
                "MILES NCCL M2N session preparation failed: " + "; ".join(failures[:4])
            )
            self._close_preserving(primary)
            raise primary

    def _close_preserving(self, primary: BaseException) -> None:
        """Best-effort close while `primary` propagates.

        close() itself issues a rank-0 generator RPC, which is the most
        likely thing to be broken when a round just failed; never let that
        secondary failure mask `primary`. A failed close leaves
        ``_close_pending`` set so a later close() retries the remaining
        cleanup; rounds stay rejected either way.
        """
        try:
            self.close()
        except BaseException as close_error:
            logger.warning(
                "MILES NCCL M2N close() during failure handling failed: %r",
                close_error,
            )
            add_note = getattr(primary, "add_note", None)
            if add_note is not None:
                add_note(f"close() during failure handling failed: {close_error!r}")

    def _teardown_for_reconnect(self) -> None:
        """Best-effort teardown of the live session before a reconnect.

        The old engine set may already be broken — that is why miles is
        reconnecting — so every step is logged, never raised, and every
        reference is dropped whether or not its close succeeded. The close
        control fan-out still targets the engines this protocol is bound to
        (the previous set); receivers treat a duplicate close as an
        idempotent no-op. The protocol stays open: the next begin_sync
        re-validates the frozen contract and the first send_bucket
        re-prepares against the new engine set.
        """
        if (
            self.rollout_engines is not None
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_rank() == 0
        ):
            try:
                self._close_generator_fanout()
            except BaseException:
                logger.warning(
                    "MILES NCCL M2N reconnect close fan-out failed", exc_info=True
                )
        session = self._session
        self._session = None
        if session is not None:
            try:
                session.close()
            except BaseException:
                logger.warning(
                    "MILES NCCL M2N reconnect session teardown failed",
                    exc_info=True,
                )
        if self._rendezvous is not None:
            try:
                self._rendezvous.close()
            except BaseException:
                logger.warning(
                    "MILES NCCL M2N reconnect rendezvous teardown failed",
                    exc_info=True,
                )
            self._rendezvous = None
        if self._channel is not None:
            try:
                self._channel.close()
            except BaseException:
                logger.warning(
                    "MILES NCCL M2N reconnect channel teardown failed",
                    exc_info=True,
                )
            self._channel = None
        self._close_pending = False

    def _close_generator_fanout(self) -> None:
        """Send the close control to every connected engine from rank 0.

        Round futures a failed round left un-awaited are retired first; their
        late results are logged, never raised. Receivers treat a duplicate
        close as an idempotent no-op, so a retry may safely re-send this.
        """
        _retire_dropped_futures(self._round_futures)
        self._round_futures = []
        futures = self._generator_futures("close")
        self._wait_generator_futures(futures)

    def close(self) -> None:
        """Tear down the generator fan-out, session, rendezvous, and channel.

        The first close attempt is terminal: later rounds are rejected even
        when teardown itself fails, and the protocol never reopens. Every
        resource is attempted even when an earlier one fails, and the first
        failure propagates after the rest settle.

        Retry semantics are per resource. A failed attempt keeps the
        resources it could not close: a later ``close()`` re-sends the rank-0
        generator close fan-out (receivers treat a duplicate close as an
        idempotent no-op) and re-runs exactly the closes that did not
        complete. The session's own teardown is one-shot — it runs once
        whether or not it raised — so a retry never re-runs it. When a call
        completes everything that remained, ``_close_pending`` clears and
        "teardown complete" is logged; a retry with nothing left to do
        returns silently, and so does a close with nothing to tear down.
        """
        if self._closed and not self._close_pending:
            return
        self._closed = True
        self._close_pending = True
        first_error: BaseException | None = None
        tore_down = False
        if (
            self.rollout_engines is not None
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_rank() == 0
        ):
            try:
                self._close_generator_fanout()
            except BaseException as error:
                first_error = error
            else:
                tore_down = True
        session = self._session
        self._session = None
        if session is not None:
            try:
                session.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            else:
                tore_down = True
        # The protocol created the rendezvous and owns the final close.
        # CollectiveRendezvous.close() is idempotent, so this also covers a
        # rendezvous the session's one-shot teardown already closed.
        if self._rendezvous is not None:
            try:
                self._rendezvous.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            else:
                self._rendezvous = None
                tore_down = True
        if self._channel is not None:
            try:
                self._channel.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            else:
                self._channel = None
                tore_down = True
        if first_error is not None:
            raise first_error
        self._close_pending = False
        if tore_down:
            rank = (
                dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            )
            logger.info("MILES NCCL M2N teardown complete trainer_rank=%d", rank)


def build_protocol(args: Any):
    """Build a MILES protocol without importing MILES at package import time."""
    from miles.backends.training_utils.weight_update.protocol import (
        WeightTransferProtocol,
    )

    class MilesModelExpressProtocol(
        MilesCollectiveProtocolCore, WeightTransferProtocol
    ):
        # The ABC defaults carry the seam contract: required_placement is
        # WeightUpdatePlacement(gather_pp=False) (miles protocol.py), and
        # supports_lora / use_weight_update_session inherit likewise.
        # connect() still rejects a gather_pp placement at runtime, so a
        # miles-side default change fails closed there.

        def __init__(self, protocol_args: Any) -> None:
            WeightTransferProtocol.__init__(self, protocol_args)
            MilesCollectiveProtocolCore.__init__(self, protocol_args)

    return MilesModelExpressProtocol(args)


build_protocol.validate_args = _validate_args

__all__ = ["MilesCollectiveProtocolCore", "build_protocol"]
