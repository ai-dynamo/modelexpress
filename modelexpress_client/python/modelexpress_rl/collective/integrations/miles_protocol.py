# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete optional MILES protocol for NCCL M2N collective refit."""

from __future__ import annotations

import logging
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
from ._common import _endpoint as _normalize_endpoint
from .miles import (
    CollectiveTopology,
    MilesPublisher,
    MilesTrainerSession,
)
from .sglang_receiver import RECEIVER_PATH
from .manifest import manifest_to_wire

logger = logging.getLogger("modelexpress_rl.collective.integrations.miles_protocol")

# Fixed for the basic path: one trainer rank, one publish group, and the
# receiver build's ABI identity. Change them only in step with the receiver.
_CONNECT_TIMEOUT_S = 10.0
_ABI_VERSION = "miles-sglang-bf16-replicated-v1"


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

    Maps the ModelExpress trainer client flow onto the MILES weight-transfer
    seam; the integration README carries the round lifecycle.
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
        self._tensors: dict[str, torch.Tensor] | None = None
        self._canonical_shapes: dict[str, tuple[int, ...]] | None = None
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
        # A fresh hex id per protocol: it prefixes every slot, and hex can
        # never contain the Redis lane-record delimiters.
        self._run_id = uuid4().hex
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
        # SG-1 rejects every receiver round while a speculative draft model
        # exists, whatever selector the round carries; detect the
        # misconfiguration here so connect fails fast with an actionable
        # error instead of the first round failing on every engine.
        if getattr(self.args, "sglang_speculative_algorithm", None):
            raise ValueError(
                "MILES NCCL M2N does not support speculative decoding: SGLang "
                "rejects external weight-update receiver rounds while a draft "
                "model exists; restart the engines without a speculative "
                "algorithm to use receiver refit."
            )
        if selector not in ("all", "target"):
            raise ValueError("MILES NCCL M2N supports the base target model only")

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
        # Every parallel dimension at one means exactly one trainer rank.
        if any(
            getattr(getattr(parallel_state, name, None), "size", None) != 1
            for name in ("pp", "tp", "ep", "etp", "cp", "intra_dp", "indep_dp")
        ):
            raise ValueError(
                "the MILES NCCL M2N path supports a single trainer rank "
                "(PP/TP/EP/CP/DP must all be one)"
            )
        # miles re-calls connect() when the rollout engine set heals; a stale
        # session must not survive (its engines never received prepare).
        # Reachable only before any round has failed: a failed round closes
        # the protocol terminally, and a mid-round reconnect is rejected
        # above.
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
        # The single trainer rank owns every tensor and drives the lane.
        self.is_sender = True

    def begin_sync(self, weight_version: int, iter_buckets) -> bool:
        """Materialize and validate the base-weight stream and arm the round.

        Round one freezes the plan/topology contract; later calls revalidate
        the stream against it and copy into its stable wire buffers.
        """
        if self.rollout_engines is None:
            raise RuntimeError("connect must run before begin_sync")
        if self._closed:
            raise RuntimeError("the MILES NCCL M2N protocol is closed")
        if self._round_version is not None:
            raise RuntimeError(
                f"the round for version {self._round_version!r} never finalized"
            )
        canonical = self._canonical_shapes
        first_round = canonical is None
        frozen: dict[str, tuple[int, ...]]
        prepared: dict[str, torch.Tensor]
        if first_round:
            frozen = {}
            prepared = {}
        else:
            if self._tensors is None or canonical is None:
                raise RuntimeError("wire buffers are unavailable")
            frozen = canonical
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
                    raise ValueError(f"{name}: only BF16 base weights are supported")
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
        if first_round:
            # Nothing is committed until the contract builds: a failed first
            # call leaves first_round true, so a retry revalidates the whole
            # stream instead of inheriting the failed attempt's shapes.
            self._build_frozen_contract(prepared)
            self._canonical_shapes = shapes
            self._tensors = prepared
        else:
            self._build_frozen_contract(None)
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

    def _build_frozen_contract(self, tensors: dict[str, torch.Tensor] | None) -> None:
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
            # HF model id: miles' parser has no `model` argument. It is part of
            # the frozen contract the receiver sees; keep it in step with the
            # deployed receiver build.
            model_name="miles-model",
            trainer_slots=(f"{slot_prefix}trainer-0",),
            generator_slots=generator_slots,
            source_partition_count=1,
            m2n_abi_version=_ABI_VERSION,
        )
        if self._plan is not None:
            # Later rounds reuse the round-one plan: the wire buffers and
            # canonical shapes are pinned (begin_sync only copy_()s into them
            # after revalidating names and shapes). The topology is still
            # rebuilt and compared every round: a reconnect that heals into a
            # reshaped engine GPU topology must fail closed.
            if topology != self._topology:
                raise RuntimeError("MILES tensor names, shapes, or topology changed")
            return
        if tensors is None:
            raise RuntimeError("begin_sync has not materialized weights")
        src_mesh = MeshSpec((1,), rank_offset=0)
        dst_mesh = MeshSpec((len(generator_slots),), rank_offset=1)
        entries = [
            ParamPlan(
                name=name,
                global_shape=tuple(int(dim) for dim in tensor.shape),
                dtype="bfloat16",
                partition_id=0,
                src_mesh=src_mesh,
                src_placements=(Placement.replicate(),),
                dst_mesh=dst_mesh,
                dst_placements=(Placement.replicate(),),
                group_key="publish-group-0",
            )
            for name, tensor in tensors.items()
        ]
        # Canonical order is the wire contract: the receiver canonical-sorts
        # too, so both sides post the same per-lane tensor sequence.
        entries.sort(key=lambda entry: entry.canonical())
        group = tuple(entry.name for entry in entries)
        self._topology = topology
        self._plan = ReshardPlan(bulk=entries, source_partition_count=1)
        self._publish_groups = (group,)
        self._group_of = dict.fromkeys(group, 0)

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
            self._close_preserving(error)
            raise
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
        operation_id = session.create_transfer(version=version)
        if not operation_id:
            raise RuntimeError("MX did not create a transfer operation")
        self._round_operation_id = operation_id
        # Tracked before begin_round so the caller's terminal close retires
        # them if anything below raises.
        self._round_futures = self._generator_futures(
            "run_round",
            version=version,
            operation_id=operation_id,
        )
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
        local_exception: BaseException | None = None
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
        futures_exception: BaseException | None = None
        if local_exception is not None:
            _retire_dropped_futures(self._round_futures)
        else:
            try:
                self._wait_generator_futures(self._round_futures)
            except BaseException as error:
                futures_exception = error
        failure = local_exception or futures_exception
        if failure is not None:
            primary = RuntimeError(f"MILES NCCL M2N round failed: {failure!r}")
            self._report_round_failure(primary)
            self._disarm_round()
            self._close_preserving(primary)
            raise primary from failure
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
        # Failures propagate to send_bucket, which closes the protocol and so
        # releases whatever of the channel/rendezvous/session was created.
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
            tensors=dict(self._tensors),
        )
        slot_id = self._topology.trainer_slots[0]
        session = MilesTrainerSession.create(
            rendezvous=rendezvous,
            topology=self._topology,
            publisher=publisher,
            source_partition=0,
            slot_id=slot_id,
            worker_id=f"miles-{slot_id}-{uuid4().hex}",
            index_in_role=0,
            layer_groups=self._publish_groups,
            device=next(iter(self._tensors.values())).device,
        )
        self._session = session

        generator_futures = self._generator_futures("prepare")
        try:
            session.prepare()
        except BaseException:
            _retire_dropped_futures(generator_futures)
            raise
        self._wait_generator_futures(generator_futures)

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


__all__ = ["MilesCollectiveProtocolCore", "build_protocol"]
