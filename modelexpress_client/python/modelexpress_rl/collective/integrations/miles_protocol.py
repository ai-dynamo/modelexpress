# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete optional MILES protocol for NCCL M2N collective refit."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import grpc
import torch
import torch.distributed as dist

from modelexpress import auth
from modelexpress import envs as mx_envs
from modelexpress.client import _get_server_url

from ..rendezvous import CollectiveRendezvous
from ..types import MeshSpec, ParamPlan, Placement, ReshardPlan
from ._common import (
    REPLICATED_DESTINATION_ABI,
    SHARDED_DESTINATION_ABI,
    _endpoint as _normalize_endpoint,
)
from .miles import (
    CollectiveTopology,
    MilesPublisher,
    MilesTrainerSession,
)
from .sglang_layout import SglangModelFacts, destination_shard_dim
from .sglang_receiver import RECEIVER_PATH
from .manifest import manifest_to_wire

logger = logging.getLogger("modelexpress_rl.collective.integrations.miles_protocol")

# Fixed: one publish group per round and the connect probe deadline.
_CONNECT_TIMEOUT_S = 10.0
_DST_LAYOUT_ENV = "MX_MILES_DST_LAYOUT"
_DST_LAYOUT_ABI = {
    "replicate": REPLICATED_DESTINATION_ABI,
    "sharded": SHARDED_DESTINATION_ABI,
}


@dataclass(frozen=True)
class TensorLayout:
    """One HF-canonical tensor's shape and the dim the trainer's TP ranks
    split evenly (None: every rank holds the identical whole tensor)."""

    global_shape: tuple[int, ...]
    shard_dim: int | None

    def __post_init__(self) -> None:
        shape = tuple(self.global_shape)
        if not shape or any(
            not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0
            for extent in shape
        ):
            raise ValueError(
                f"global_shape must be non-empty positive integers, got {shape!r}"
            )
        object.__setattr__(self, "global_shape", shape)
        dim = self.shard_dim
        if dim is not None and (
            not isinstance(dim, int)
            or isinstance(dim, bool)
            or not 0 <= dim < len(shape)
        ):
            raise ValueError(f"shard_dim must be None or a dim of {shape}, got {dim!r}")


def _dst_layout() -> str:
    layout = os.environ.get(_DST_LAYOUT_ENV, "replicate").strip() or "replicate"
    if layout not in _DST_LAYOUT_ABI:
        raise ValueError(
            f"{_DST_LAYOUT_ENV} must be one of {sorted(_DST_LAYOUT_ABI)}, "
            f"got {layout!r}"
        )
    return layout


def _gloo_group():
    from miles.utils.distributed_utils import get_gloo_group

    return get_gloo_group()


def _rank_and_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _all_gather(value: Any) -> list[Any]:
    """Every trainer rank's ``value``, in rank order, over the Gloo group."""
    _rank, world = _rank_and_world()
    if world == 1:
        return [value]
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, value, group=_gloo_group())
    return gathered


def _broadcast_from_rank_zero(value: Any) -> Any:
    _rank, world = _rank_and_world()
    if world == 1:
        return value
    box = [value]
    dist.broadcast_object_list(box, src=0, group=_gloo_group())
    return box[0]


def _gathered_failures(local_error: str) -> list[str]:
    """Fan one local error string out; return every rank's non-empty error."""
    return [
        f"rank {rank}: {error}"
        for rank, error in enumerate(_all_gather(local_error))
        if error
    ]


def _group_size_and_rank(parallel_state: Any, name: str) -> tuple[int, int]:
    group = getattr(parallel_state, name, None)
    size = getattr(group, "size", None)
    rank = getattr(group, "rank", None)
    if (
        not isinstance(size, int)
        or not isinstance(rank, int)
        or size < 1
        or not 0 <= rank < size
    ):
        raise ValueError(
            f"MILES parallel state {name!r} needs an integer size >= 1 and a "
            f"rank in [0, size), got size={size!r} rank={rank!r}"
        )
    return size, rank


def _model_facts(args: Any) -> SglangModelFacts:
    checkpoint = getattr(args, "hf_checkpoint", None)
    if not checkpoint:
        raise ValueError(
            f"{_DST_LAYOUT_ENV}=sharded reads the model's head counts and "
            "vocabulary from the HF config; pass --hf-checkpoint"
        )
    return SglangModelFacts.from_checkpoint(str(checkpoint))


def _server_endpoint() -> str:
    # The shared resolver's precedence (MODEL_EXPRESS_URL, then
    # MX_SERVER_ADDRESS), minus its localhost default: this path must be
    # configured explicitly.
    configured = mx_envs.MODEL_EXPRESS_URL
    if configured is None:
        configured = mx_envs.MX_SERVER_ADDRESS
    if not configured:
        raise ValueError(
            "the ModelExpress M2N protocol needs the mx-server address: set "
            "MODEL_EXPRESS_URL or MX_SERVER_ADDRESS=<host:port>"
        )
    # The scheme policy must see the configured value: _get_server_url strips
    # an http(s) prefix, which would hide a secure scheme _endpoint rejects.
    _normalize_endpoint(configured)
    return _normalize_endpoint(_get_server_url(configured))


def _server_host_port(endpoint: str) -> tuple[str, int]:
    """Split the mx-server ``host:port`` for ``master_address``/``master_port``."""
    host, separator, port = endpoint.rpartition(":")
    host = host.removeprefix("[").removesuffix("]")
    if not separator or not host or not port.isdecimal() or not 0 < int(port) < 65536:
        raise ValueError(
            f"the ModelExpress server address must be host:port to reach the "
            f"engine receivers, got {endpoint!r}"
        )
    return host, int(port)


def _int_tuple(values: Any, field: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a list of integers") from error


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
            f"{timeout_s:.1f}s; check MX_SERVER_ADDRESS and that the mx-server "
            "is running"
        ) from error


class MilesCollectiveProtocolCore:
    """MILES bucket-stream bridge used by the lazy protocol factory.

    Every trainer rank of a DP x TP trainer is a sender in one reshard lane;
    rank 0 alone drives the generator fan-out, and local failures fan out
    over the trainer's Gloo group. Round flow: see the integrations README.
    """

    def __init__(self, args: Any) -> None:
        self.args = args
        self.rollout_engines = None
        self.is_sender: bool | None = None
        # miles' updater reads protocol.group_name for its progress display;
        # WeightTransferProtocol.__init__ sets "miles", so name this path.
        self.group_name = "modelexpress-m2n"
        self._parallel_state = None
        self._selector = "target"
        self._engine_gpu_counts: tuple[int, ...] = ()
        self._engine_gpu_offsets: tuple[int, ...] = ()
        self._dp_size = 1
        self._tp_size = 1
        self._lane_rank = 0
        self._iterator_layouts: Any = None
        self._tensors: dict[str, torch.Tensor] | None = None
        self._local_shapes: dict[str, tuple[int, ...]] | None = None
        self._layouts: dict[str, TensorLayout] | None = None
        self._topology: CollectiveTopology | None = None
        self._plan: ReshardPlan | None = None
        self._publish_groups: tuple[tuple[str, ...], ...] | None = None
        self._group_of: dict[str, int] = {}
        # Resolve and check the endpoint now so a missing or unparseable
        # address fails when MILES builds the protocol, not inside the first
        # weight update.
        self._endpoint = _server_endpoint()
        _server_host_port(self._endpoint)
        # The engine-side receiver group, named at prepare.
        self._engine_group_name: str | None = None
        self._dst_layout = _dst_layout()
        # Agreed across trainer ranks at the first begin_sync: rank 0's fresh
        # hex id prefixes every slot, and hex can never contain the Redis
        # lane-record delimiters.
        self._run_id: str | None = None
        self._session: MilesTrainerSession | None = None
        self._channel = None
        self._rendezvous = None
        self._closed = False
        self._close_pending = False
        self._round_version: str | None = None
        self._round_operation_id: str | None = None
        self._round_begun = False
        self._round_seen: set[str] = set()
        self._pending: list[set[str]] = []
        self._next_group = 0
        self._round_futures: list = []

    def _validate_selector(self, selector: str) -> None:
        # SG-1 rejects every receiver round while a draft model exists; fail
        # here at connect instead of in the first round on every engine.
        if getattr(self.args, "sglang_speculative_algorithm", None):
            raise ValueError(
                "MILES NCCL M2N does not support speculative decoding: SGLang "
                "rejects external weight-update receiver rounds while a draft "
                "model exists; restart the engines without a speculative "
                "algorithm to use receiver refit."
            )
        if selector not in ("all", "target"):
            raise ValueError("MILES NCCL M2N supports the base target model only")

    def configure_model(self, iterator: Any) -> None:
        """Record the trainer's HF weight iterator for ``_tensor_layouts``.

        Nothing calls this at the pinned MILES revision, so the
        ``_tensor_layouts`` gate is inert; ``connect``'s gather rejection is
        the live guard against trainer-local sources.
        """
        self._iterator_layouts = getattr(iterator, "tensor_layouts", None)

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
        counts = _int_tuple(engine_gpu_counts, "engine_gpu_counts")
        offsets = _int_tuple(engine_gpu_offsets, "engine_gpu_offsets")
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
        if self._dst_layout == "sharded" and len(set(counts)) != 1:
            raise ValueError(
                f"{_DST_LAYOUT_ENV}=sharded requires every engine to have the "
                f"same TP size, got engine GPU counts {list(counts)}"
            )
        # The source mesh is (DP, TP): every other parallel dimension is one.
        if any(
            getattr(getattr(parallel_state, name, None), "size", None) != 1
            for name in ("pp", "ep", "etp", "cp", "indep_dp")
        ):
            raise ValueError(
                "the MILES NCCL M2N path supports a DP x TP trainer only "
                "(PP/EP/ETP/CP must be one, and DP must be intra-DP)"
            )
        dp_size, dp_rank = _group_size_and_rank(parallel_state, "intra_dp")
        tp_size, tp_rank = _group_size_and_rank(parallel_state, "tp")
        # Reconnect tears the stale session down: a skipped re-prepare would
        # fan run_round out to engines that never received prepare.
        if (
            self._session is not None
            or self._rendezvous is not None
            or self._channel is not None
        ):
            logger.info("MILES NCCL M2N reconnect: tearing down the previous session")
            self._teardown_for_reconnect()
        self.rollout_engines = tuple(rollout_engines)
        self._engine_gpu_counts = counts
        self._engine_gpu_offsets = offsets
        self._parallel_state = parallel_state
        self._selector = selector
        self._dp_size = dp_size
        self._tp_size = tp_size
        self._lane_rank = dp_rank * tp_size + tp_rank
        # Every trainer rank holds every tensor whole (TP is gathered) and
        # posts it in the one reshard lane, so every rank is a sender.
        self.is_sender = True

    def _tensor_layouts(
        self, local_shapes: dict[str, tuple[int, ...]]
    ) -> dict[str, TensorLayout]:
        """The layout of every tensor this rank's stream yielded, by HF name.

        ``tensor_layouts`` None is the gathered-stream contract: whole tensors
        on every rank. An iterator carrying ``tensor_layouts`` is asking for
        trainer-local shards, which this build does not support.
        """
        if self._iterator_layouts is not None:
            raise ValueError(
                "the MILES iterator's tensor_layouts describe trainer-local TP "
                "shards; MILES NCCL M2N needs a gathered stream (tensor_layouts "
                "absent or None) because trainer-local sources are not supported"
            )
        return {name: TensorLayout(shape, None) for name, shape in local_shapes.items()}

    def begin_sync(self, weight_version: int, iter_buckets) -> bool:
        """Materialize and validate the base-weight stream and arm the round.

        Each tensor is validated and copied into its stable wire buffer as it
        arrives; ``send_bucket`` only tracks arrival order. A local failure
        fans out over the trainer's Gloo group before the contract build's
        collectives so no peer strands inside one.
        """
        local_shapes: dict[str, tuple[int, ...]] | None = None
        prepared_tensors: dict[str, torch.Tensor] | None = None
        layouts: dict[str, TensorLayout] | None = None
        facts: SglangModelFacts | None = None
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
            first_round = self._local_shapes is None
            frozen: dict[str, tuple[int, ...]]
            prepared: dict[str, torch.Tensor]
            if first_round:
                frozen = {}
                prepared = {}
            else:
                if self._tensors is None:
                    raise RuntimeError("wire buffers are unavailable")
                frozen = self._local_shapes or {}
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
                        prepared[name] = tensor.clone()
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
            layouts = self._tensor_layouts(shapes)
            if not first_round and layouts != self._layouts:
                raise RuntimeError("MILES tensor layouts changed")
            if first_round and self._dst_layout == "sharded":
                facts = _model_facts(self.args)
            local_shapes = shapes
            prepared_tensors = prepared
        except BaseException as error:
            local_exception = error
            local_error = repr(error)
        failures = _gathered_failures(local_error)
        if failures:
            _rank, world = _rank_and_world()
            if world == 1 and local_exception is not None:
                raise local_exception
            error = RuntimeError(
                "MILES NCCL M2N begin_sync preparation failed: "
                + "; ".join(failures[:4])
            )
            if local_exception is not None:
                raise error from local_exception
            raise error
        if local_shapes is None or prepared_tensors is None or layouts is None:
            raise RuntimeError("MILES NCCL M2N preparation produced no tensor state")
        # The contract build reads the new layouts and shapes from arguments;
        # nothing commits until it returns successfully.
        self._build_frozen_contract(facts, layouts)
        if self._local_shapes is None:
            self._local_shapes = local_shapes
            self._tensors = prepared_tensors
            self._layouts = layouts
        self._arm_round(weight_version)
        return True

    def _arm_round(self, weight_version: int) -> None:
        if self._plan is None or self._publish_groups is None:
            raise RuntimeError("the frozen plan is unavailable")
        self._pending = [set(group) for group in self._publish_groups]
        self._round_seen = set()
        self._next_group = 0
        self._round_begun = False
        self._round_futures = []
        self._round_version = str(weight_version)

    def _disarm_round(self) -> None:
        self._round_operation_id = None
        self._round_version = None
        self._round_begun = False
        self._round_seen = set()
        self._pending = []
        self._next_group = 0
        self._round_futures = []

    def _build_frozen_contract(
        self, facts: SglangModelFacts | None, layouts: dict[str, TensorLayout]
    ) -> None:
        """Build or re-check the frozen topology/plan without committing state.

        Everything the collectives need (``layouts`` for the manifest) comes
        in as an argument; ``self._run_id`` runs through a local so a failed
        build leaves no protocol state behind.
        """
        _rank, trainer_world = _rank_and_world()
        if trainer_world != self._dp_size * self._tp_size:
            raise ValueError(
                f"the trainer world ({trainer_world}) must be exactly DP x TP "
                f"({self._dp_size} x {self._tp_size})"
            )
        run_id = self._run_id
        if run_id is None:
            # Every rank proposes a fresh id and rank 0's wins, so every rank
            # names the same slots without any configuration.
            run_id = _all_gather(uuid4().hex)[0]
        slot_prefix = f"{run_id}:"
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
            # HF model id: miles' parser has no `model` argument. It is part of
            # the frozen contract the receiver sees; keep it in step with the
            # deployed receiver build.
            model_name="miles-model",
            trainer_slots=tuple(
                f"{slot_prefix}trainer-{lane_rank}"
                for lane_rank in range(trainer_world)
            ),
            generator_slots=generator_slots,
            source_partition_count=1,
            m2n_abi_version=_DST_LAYOUT_ABI[self._dst_layout],
        )
        if self._plan is not None:
            # Later rounds rebuild only the topology and compare it: a
            # reconnect that heals into a reshaped engine GPU topology fails
            # closed. The skip is uniform because _plan is set on every rank
            # or on none.
            if topology != self._topology:
                raise RuntimeError("MILES tensor names, shapes, or topology changed")
            return
        local_manifest = sorted(
            (name, layout.global_shape, layout.shard_dim)
            for name, layout in layouts.items()
        )
        manifests = _all_gather((self._lane_rank, local_manifest))
        lane_ranks = sorted(lane_rank for lane_rank, _manifest in manifests)
        if lane_ranks != list(range(trainer_world)):
            raise ValueError(
                "trainer ranks must cover every (DP, TP) coordinate exactly once; "
                f"got lane ranks {lane_ranks}"
            )
        reference = manifests[0][1]
        diverged = [
            rank
            for rank, (_lane_rank, manifest) in enumerate(manifests)
            if manifest != reference
        ]
        if diverged:
            raise ValueError(
                "every trainer rank must report the same HF names, global shapes, "
                f"and shard dims; ranks {diverged[:5]} differ from rank 0"
            )
        src_mesh = (
            MeshSpec((self._tp_size,))
            if self._dp_size == 1
            else MeshSpec((self._dp_size, self._tp_size))
        )
        dst_mesh, engine_tp = self._destination_mesh(
            len(generator_slots), src_mesh.size
        )
        entries = []
        for name, global_shape, shard_dim in reference:
            inner = (
                Placement.replicate()
                if shard_dim is None
                else Placement.shard(shard_dim)
            )
            src_placements = (Placement.replicate(),) * (len(src_mesh.shape) - 1) + (
                inner,
            )
            dst_dim = None
            if self._dst_layout == "sharded":
                if facts is None:
                    raise RuntimeError("sharded destinations need the model facts")
                dst_dim = destination_shard_dim(name, global_shape, engine_tp, facts)
            dst_inner = (
                Placement.replicate() if dst_dim is None else Placement.shard(dst_dim)
            )
            dst_placements = (Placement.replicate(),) * (len(dst_mesh.shape) - 1) + (
                dst_inner,
            )
            entries.append(
                ParamPlan(
                    name=name,
                    global_shape=global_shape,
                    dtype="bfloat16",
                    partition_id=0,
                    src_mesh=src_mesh,
                    src_placements=src_placements,
                    dst_mesh=dst_mesh,
                    dst_placements=dst_placements,
                    group_key="publish-group-0",
                )
            )
        # Canonical order is the wire contract: the receiver canonical-sorts
        # too, so both sides post the same per-lane tensor sequence.
        entries.sort(key=lambda entry: entry.canonical())
        group = tuple(entry.name for entry in entries)
        # Commit only here: every check above passed.
        self._run_id = run_id
        self._topology = topology
        self._plan = ReshardPlan(bulk=entries, source_partition_count=1)
        self._publish_groups = (group,)
        self._group_of = dict.fromkeys(group, 0)

    def _destination_mesh(
        self, generator_count: int, trainer_count: int
    ) -> tuple[MeshSpec, int]:
        """The generator mesh and the TP size a sharded entry splits over.

        ``replicate``: a flat ``(generators,)`` mesh. ``sharded``: ``(TP,)``
        for one engine, ``(engines, TP)`` with a replicated engine axis for
        several, in slot order.
        """
        if self._dst_layout == "replicate":
            return MeshSpec((generator_count,), rank_offset=trainer_count), 1
        engine_tp = self._engine_gpu_counts[0]
        engines = len(self._engine_gpu_counts)
        if engines == 1:
            return MeshSpec((engine_tp,), rank_offset=trainer_count), engine_tp
        return (
            MeshSpec((engines, engine_tp), rank_offset=trainer_count),
            engine_tp,
        )

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
        # A bucket that names unknown or repeated tensors means the
        # trainer's bucket stream diverged from the frozen contract — a bug,
        # not a transient error. The generators are already paused mid-round,
        # so continuing a diverged round would risk publishing a wrong-plan
        # weight set; close the protocol like any other mid-round failure.
        local_exception: BaseException | None = None
        local_error = ""
        try:
            unknown = [name for name in names if name not in self._group_of]
            if unknown:
                raise ValueError(
                    "MILES NCCL M2N bucket carries tensors outside the frozen plan: "
                    f"{unknown[:5]}"
                )
            within_bucket = sorted({name for name in names if names.count(name) > 1})
            if within_bucket:
                raise ValueError(
                    "MILES NCCL M2N bucket repeats a tensor within one bucket: "
                    f"{within_bucket[:5]}"
                )
            repeated = [name for name in names if name in self._round_seen]
            if repeated:
                raise ValueError(
                    "MILES NCCL M2N bucket repeats tensors already seen in round "
                    f"{version}: {repeated[:5]}"
                )
        except BaseException as error:
            local_exception = error
            local_error = repr(error)
        # Every rank joins the vote even with no local divergence, so one
        # rank's diverged stream fails its peers here instead of stranding
        # them inside the round-setup collectives below.
        failures = _gathered_failures(local_error)
        if failures:
            _rank, world = _rank_and_world()
            if world == 1 and local_exception is not None:
                self._close_preserving(local_exception)
                raise local_exception
            error = RuntimeError(
                "MILES NCCL M2N bucket stream diverged: " + "; ".join(failures[:4])
            )
            self._close_preserving(error)
            if local_exception is not None:
                raise error from local_exception
            raise error
        if not self._round_begun:
            if self._session is None:
                try:
                    self._prepare_sessions()
                except BaseException as error:
                    self._close_preserving(error)
                    raise
            try:
                self._begin_round(version)
            except BaseException as error:
                self._close_preserving(error)
                raise
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
        rank, _world = _rank_and_world()
        futures: list = []
        submission_error = ""
        operation_id = None
        if rank == 0:
            try:
                operation_id = session.create_transfer(version=version)
                if not operation_id:
                    raise RuntimeError("MX did not create a transfer operation")
                # Set before the fan-out so the terminal close can still
                # report the created operation if the submission fails.
                self._round_operation_id = operation_id
                futures = self._generator_futures(
                    "run_round",
                    version=version,
                    operation_id=operation_id,
                )
            except BaseException as error:
                submission_error = repr(error)
        # Tracked before the gather so the caller's terminal close retires
        # them if anything below raises.
        self._round_futures = futures
        submission_failures = _gathered_failures(submission_error)
        if submission_failures:
            raise RuntimeError(
                "MILES NCCL M2N round generator submission failed: "
                + "; ".join(submission_failures[:4])
            )
        operation_id = _broadcast_from_rank_zero(operation_id)
        if not operation_id:
            raise RuntimeError("MX did not create a transfer operation")
        if rank != 0:
            # Rank 0 assigned above, from create_transfer.
            self._round_operation_id = operation_id
        session.begin_round(version=version)
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
        rank, _world = _rank_and_world()
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
            self._session.finish_round(
                version=version, operation_id=self._round_operation_id
            )
        except BaseException as error:
            local_exception = error
            local_error = repr(error)
        failures = _gathered_failures(local_error)
        futures_error = ""
        if rank == 0:
            if failures:
                _retire_dropped_futures(self._round_futures)
            else:
                try:
                    self._wait_generator_futures(self._round_futures)
                except BaseException as error:
                    futures_error = repr(error)
        futures_error = _broadcast_from_rank_zero(futures_error)
        if futures_error:
            failures.append(f"generators: {futures_error}")
        if failures:
            primary = RuntimeError(
                "MILES NCCL M2N round failed: " + "; ".join(failures[:4])
            )
            self._report_round_failure(primary)
            self._disarm_round()
            self._close_preserving(primary)
            if local_exception is not None:
                raise primary from local_exception
            raise primary
        self._disarm_round()

    def after_engines_resumed(self) -> None:
        """No-op hook: the generator fan-out already settled in finalize."""

    @staticmethod
    async def _call_engine(client: Any, endpoint: str, payload: dict[str, Any]):
        """POST one SGLang engine endpoint through the MILES API client.

        The public ``call_endpoint`` wins; clients without it (MILES at the
        pinned revisions) go through the private ``_make_request``, which has
        the same ``(endpoint, payload)`` shape.
        """
        # TODO: drop the _make_request fallback when MILES exports a public
        # call_endpoint, or typed init/update/destroy wrappers that accept the
        # receiver fields this protocol sends.
        call = getattr(client, "call_endpoint", None) or getattr(
            client, "_make_request", None
        )
        if call is None:
            raise RuntimeError(
                "the MILES SGLang API client has neither call_endpoint nor "
                "_make_request; cannot drive the SGLang external receiver"
            )
        return _check_response(await call(endpoint, payload))

    async def _destroy_on_engine(self, client: Any, payload: dict[str, Any]) -> None:
        import httpx

        # SG-1 forgets a receiver even when destroy() raises, and answers a
        # repeat destroy with HTTP 400 {"success": false, "message": "The
        # group to be destroyed does not exist."}, which MILES's client
        # raises as HTTPStatusError via raise_for_status. Group absence is
        # the end state close() wants, so only that exact answer is accepted;
        # every other status or body still fails the fan-out.
        try:
            await self._call_engine(client, "destroy_weights_update_group", payload)
        except httpx.HTTPStatusError as error:
            response = error.response
            if response.status_code != 400 or "does not exist" not in response.text:
                raise
        except RuntimeError as error:
            # A call_endpoint that returns the parsed success=false body
            # instead of raising surfaces it as _check_response's RuntimeError.
            if "does not exist" not in str(error):
                raise

    def _engine_payload(
        self,
        action: str,
        *,
        generator_slot_offset: int,
        manifest: dict[str, Any] | None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        group_name = self._engine_group_name
        if action == "prepare":
            host, port = _server_host_port(self._endpoint)
            # SG-1's InitWeightsUpdateGroupReqInput: receiver groups create no
            # torch process group; backend is unused on that path.
            return "init_weights_update_group", {
                "master_address": host,
                "master_port": port,
                "rank_offset": generator_slot_offset,
                "world_size": len(self._topology.generator_slots),
                "group_name": group_name,
                "receiver": RECEIVER_PATH,
                "receiver_init_payload": manifest,
            }
        if action == "run_round":
            version = str(kwargs["version"])
            # SG-1's UpdateWeightsFromDistributedReqInput carries an opaque
            # receiver_payload for external receiver groups; the weight lists
            # stay empty because the receiver writes into the live model.
            return "update_weights_from_distributed", {
                "names": [],
                "dtypes": [],
                "shapes": [],
                "group_name": group_name,
                "receiver_payload": {
                    "operation_id": str(kwargs["operation_id"]),
                    "version": version,
                },
                # flush_cache=False matches the incumbent MILES flows: the
                # session frame already flushed the engine cache at pause for
                # the flushing pause modes, and in-place pause preserves it
                # on purpose (MILES's own update calls default to False too).
                "flush_cache": False,
                # Echo the connect-time selector so the connect admission and
                # the round payload agree.
                "selector": self._selector,
            }
        if action == "close":
            return "destroy_weights_update_group", {"group_name": group_name}
        raise ValueError(f"unknown engine action {action!r}")

    def _public_engine_calls(self, action: str, **kwargs: Any) -> list:
        """One SGLang public-contract call per engine for ``action``."""
        if not self.rollout_engines:
            return []
        manifest = None
        if action == "prepare":
            if self._plan is None or self._topology is None:
                raise RuntimeError("begin_sync must run before engine preparation")
            # A fresh group per prepare: a reconnect's engines never saw the
            # old name, and the old engines were destroyed before it.
            self._engine_group_name = f"mx-m2n-{uuid4().hex}"
            manifest = manifest_to_wire(self._plan, self._topology)
        elif self._engine_group_name is None:
            if action == "close":
                # Nothing was prepared, so no engine holds a group.
                return []
            raise RuntimeError("the engine receiver group was not prepared")
        calls = []
        # rank_offset is each engine's dense start position in the flattened
        # generator slot sequence, not its physical engine_gpu_offset: the
        # receiver resolves generator_slots[rank_offset + tp_rank] while the
        # slot names carry the physical offsets. Incumbent MILES protocols
        # compute rank offsets from the same cursor.
        generator_slot_offset = 0
        for client, count in zip(
            self.rollout_engines, self._engine_gpu_counts, strict=True
        ):
            endpoint, payload = self._engine_payload(
                action,
                generator_slot_offset=generator_slot_offset,
                manifest=manifest,
                **kwargs,
            )
            if action == "close":
                calls.append(self._destroy_on_engine(client, payload))
            else:
                calls.append(self._call_engine(client, endpoint, payload))
            generator_slot_offset += count
        return calls

    def _generator_futures(self, action: str, **kwargs) -> list:
        from miles.utils import async_utils

        if self.rollout_engines is None:
            raise RuntimeError("rollout engines are not connected")
        futures = []
        for call in self._public_engine_calls(action, **kwargs):
            try:
                futures.append(async_utils.submit(call))
            except BaseException:
                call.close()
                _retire_dropped_futures(futures)
                raise
        return futures

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
        rank, _world = _rank_and_world()
        lane_rank = self._lane_rank
        # Every local step is inside the failure fan-out so one rank's error
        # reaches the shared gather; from there send_bucket closes the protocol.
        local_error = ""
        try:
            local_tensors = dict(self._tensors)
            expected_names = {entry.name for entry in self._plan.bulk}
            if set(local_tensors) != expected_names:
                missing = sorted(expected_names - set(local_tensors))
                unexpected = sorted(set(local_tensors) - expected_names)
                raise ValueError(
                    f"lane rank {lane_rank} tensors do not match the global plan "
                    f"(missing={missing[:5]}, unexpected={unexpected[:5]})"
                )
            endpoint = self._endpoint
            channel = auth.with_auth(grpc.insecure_channel(endpoint))
            try:
                _await_endpoint_ready(
                    channel,
                    endpoint=endpoint,
                    timeout_s=_CONNECT_TIMEOUT_S,
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
                source_partition=0,
                tensors=local_tensors,
                source_rank=lane_rank,
            )
            slot_id = self._topology.trainer_slots[lane_rank]
            self._session = MilesTrainerSession.create(
                rendezvous=rendezvous,
                topology=self._topology,
                publisher=publisher,
                source_partition=0,
                slot_id=slot_id,
                worker_id=f"miles-{slot_id}-{uuid4().hex}",
                index_in_role=lane_rank,
                layer_groups=self._publish_groups,
                device=next(iter(local_tensors.values())).device,
            )
        except BaseException as error:
            local_error = repr(error)
        setup_failures = _gathered_failures(local_error)
        if setup_failures:
            raise RuntimeError(
                "MILES NCCL M2N local session setup failed: "
                + "; ".join(setup_failures[:4])
            )
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
        submission_failures = _gathered_failures(submission_error)
        if submission_failures:
            raise RuntimeError(
                "MILES NCCL M2N generator preparation submission failed: "
                + "; ".join(submission_failures[:4])
            )

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
        failures = _gathered_failures(local_error)
        if failures:
            raise RuntimeError(
                "MILES NCCL M2N session preparation failed: " + "; ".join(failures[:4])
            )

    def _report_round_failure(self, primary: BaseException) -> None:
        if self._session is None or self._round_operation_id is None:
            return
        try:
            self._session.report_failure(self._round_operation_id, primary)
        except BaseException:
            logger.warning("reporting the trainer round failure failed", exc_info=True)

    def _close_preserving(self, primary: BaseException) -> None:
        """Best-effort close while ``primary`` propagates.

        close() re-contacts the engines — the likeliest breakage after a
        failed round — so its failure is logged, never masking ``primary``.
        """
        self._report_round_failure(primary)
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
        """Tear down the live session before a reconnect, best effort.

        The old engines may be why miles is reconnecting, so every step is
        logged rather than raised; the next round re-prepares under a fresh
        group name.
        """
        if self._drives_destroy_fan_out():
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

    def _drives_destroy_fan_out(self) -> bool:
        if self.rollout_engines is None:
            return False
        if not dist.is_available() or not dist.is_initialized():
            # Unreachable in a MILES run (the updater drives dist.barrier);
            # the test harnesses land here.
            logger.debug(
                "engine destroy fan-out skipped: torch.distributed is not initialized"
            )
            return False
        if dist.get_rank() != 0:
            logger.debug("engine destroy fan-out skipped: rank 0 drives the engines")
            return False
        return True

    def _close_generator_fanout(self) -> None:
        """Destroy the receiver group on every connected engine from rank 0.

        Retires unsettled round futures first; SG-1's destroy is not
        idempotent, so an already-gone group is accepted as done
        (``_destroy_on_engine``). The group name clears only once every
        engine answered.
        """
        _retire_dropped_futures(self._round_futures)
        self._round_futures = []
        futures = self._generator_futures("close")
        self._wait_generator_futures(futures)
        self._engine_group_name = None

    def close(self) -> None:
        """Tear down the generator fan-out, session, rendezvous, and channel.

        Terminal for rounds even when teardown fails, best effort across
        resources, retried per retained resource: a retry re-runs the closes
        that failed and can still make progress (rendezvous, channel, fan-out).
        The session's own close is one-shot, so a retry would no-op.
        """
        if self._closed and not self._close_pending:
            return
        self._closed = True
        self._close_pending = True
        first_error: BaseException | None = None
        tore_down = False
        if self._drives_destroy_fan_out():
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
        # The protocol owns the rendezvous' final close;
        # CollectiveRendezvous.close() is idempotent.
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


__all__ = ["MilesCollectiveProtocolCore", "TensorLayout", "build_protocol"]
