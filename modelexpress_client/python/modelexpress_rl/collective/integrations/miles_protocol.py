# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete optional MILES protocol for NCCL M2N collective refit."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any
from uuid import uuid4

import grpc
import torch
import torch.distributed as dist

from modelexpress import auth
from ... import refit_collective_pb2 as collective_pb
from .. import envs
from ..rendezvous import CollectiveRendezvous
from ..types import MeshSpec, ParamPlan, Placement, PlacementKind, ReshardPlan
from ._common import _endpoint as _normalize_endpoint
from ._common import _exact_tensor_sha256, _local_shape, _text
from .miles import (
    CollectiveTopology,
    MilesPublisher,
    MilesTrainerSession,
    MilesTransferCoordinator,
)
from .wire import (
    CollectiveControl,
    TensorDigestMap,
    encode_control,
    tensor_equality_receipt,
)

logger = logging.getLogger("modelexpress_rl.collective.integrations.miles_protocol")
_TERMINAL_STATES = frozenset(
    {
        collective_pb.COLLECTIVE_TRANSFER_STATE_COMPLETE,
        collective_pb.COLLECTIVE_TRANSFER_STATE_FAILED,
        collective_pb.COLLECTIVE_TRANSFER_STATE_ABORTED,
    }
)


def _gloo_group():
    from miles.utils.distributed_utils import get_gloo_group

    return get_gloo_group()


def _arg(args: Any, name: str, default: Any = None) -> Any:
    value = getattr(args, name, None)
    return default if value is None else value


def _endpoint(args: Any) -> str:
    return _normalize_endpoint(
        _arg(
            args,
            "modelexpress_server_address",
            os.environ.get("MX_SERVER_ADDRESS", "127.0.0.1:50051"),
        )
    )


def _check_response(response: Any) -> Any:
    if isinstance(response, dict):
        success = response.get("success")
        message = response.get("message", "")
    else:
        success = getattr(response, "success", None)
        message = getattr(response, "message", "")
    if success is False:
        raise RuntimeError(str(message or "SGLang rejected the collective command"))
    return response


class MilesCollectiveProtocolCore:
    """MILES whole-round bridge used by the lazy protocol factory."""

    owns_update_round = True
    supports_lora = False
    use_weight_update_session = True

    def __init__(self, args: Any) -> None:
        self.args = args
        self.rollout_engines = None
        self.group_name = "modelexpress-m2n"
        self.update_weight_metrics: dict[str, float] = {}
        self._placement = None
        self._parallel_state = None
        self._engine_gpu_counts: tuple[int, ...] = ()
        self._engine_gpu_offsets: tuple[int, ...] = ()
        self._tensors: dict[str, torch.Tensor] | None = None
        self._canonical_shapes: dict[str, tuple[int, ...]] | None = None
        self._topology: CollectiveTopology | None = None
        self._plan: ReshardPlan | None = None
        self._tensor_digests: TensorDigestMap | None = None
        self._verify_tensor_equality = envs.MX_MILES_VERIFY_TENSOR_EQUALITY
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
        self._coordinator: MilesTransferCoordinator | None = None
        self._channel = None
        self._rendezvous = None
        self._generators_prepared = False
        self._closed = False

    def connect(
        self,
        rollout_engines,
        engine_gpu_counts,
        engine_gpu_offsets,
        parallel_state,
        placement,
        selector,
    ) -> None:
        if selector not in ("all", "target"):
            raise ValueError("MILES NCCL M2N supports the base target model only")
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
            getattr(getattr(parallel_state, name), "size", 1) != 1
            for name in ("tp", "ep", "etp", "cp", "intra_dp", "indep_dp")
        ):
            raise ValueError(
                "the initial MILES NCCL M2N path requires exactly one source "
                "rank per PP partition (TP/EP/CP/DP must all be one)"
            )
        self.rollout_engines = tuple(rollout_engines)
        self._engine_gpu_counts = counts
        self._engine_gpu_offsets = offsets
        self._parallel_state = parallel_state
        self._placement = placement

    def begin_sync(self, weight_version: int, iter_buckets) -> bool:
        canonical_shapes: dict[str, tuple[int, ...]] | None = None
        prepared_tensors: dict[str, torch.Tensor] | None = None
        local_exception: BaseException | None = None
        local_error = ""
        try:
            if self.rollout_engines is None:
                raise RuntimeError("connect must run before begin_sync")
            if self._closed:
                raise RuntimeError("the MILES NCCL M2N protocol is closed")
            canonical: dict[str, torch.Tensor] = {}
            for bucket in iter_buckets(materialize=True):
                for name, tensor in bucket:
                    if ":" in name:
                        raise ValueError("LoRA/adaptor tensors are not supported")
                    if name in canonical:
                        raise ValueError(f"duplicate HF weight name {name!r}")
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
                    canonical[name] = tensor
            if not canonical:
                raise ValueError("MILES produced no base-model tensors")
            canonical_shapes = {
                name: tuple(int(dim) for dim in tensor.shape)
                for name, tensor in canonical.items()
            }
            if self._canonical_shapes is None:
                prepared_tensors = {}
                for name, tensor in canonical.items():
                    wire = torch.empty(
                        tuple(tensor.shape),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                    wire.copy_(tensor)
                    prepared_tensors[name] = wire
            else:
                if canonical_shapes != self._canonical_shapes:
                    raise RuntimeError("MILES tensor names or canonical shapes changed")
                if self._tensors is None:
                    raise RuntimeError("wire buffers are unavailable")
                prepared_tensors = self._tensors
                for name, tensor in canonical.items():
                    prepared_tensors[name].copy_(tensor)
        except BaseException as error:  # noqa: BLE001
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
            raise RuntimeError(
                "MILES NCCL M2N begin_sync preparation failed: "
                + "; ".join(failures[:4])
            )
        if canonical_shapes is None or prepared_tensors is None:
            raise RuntimeError("MILES NCCL M2N preparation produced no tensor state")
        if self._canonical_shapes is None:
            self._canonical_shapes = canonical_shapes
            self._tensors = prepared_tensors
        self._build_frozen_contract()
        if self._verify_tensor_equality:
            self._build_tensor_digests()
        else:
            self._tensor_digests = None
        return True

    def _build_frozen_contract(self) -> None:
        if self._tensors is None:
            raise RuntimeError("begin_sync has not materialized weights")
        trainer_world = dist.get_world_size()
        partition_count = int(self._parallel_state.pp.size)
        if trainer_world != partition_count:
            raise ValueError(
                "the trainer world must contain exactly one rank per PP partition"
            )
        if not self._run_id_agreed:
            requested_run_id = self._run_id
            run_id_candidates: list[str | None] = [None] * trainer_world
            dist.all_gather_object(
                run_id_candidates,
                requested_run_id,
                group=_gloo_group(),
            )
            configured_run_ids = {
                candidate for candidate in run_id_candidates if candidate is not None
            }
            if configured_run_ids:
                if len(configured_run_ids) != 1 or any(
                    candidate is None for candidate in run_id_candidates
                ):
                    raise ValueError(
                        "modelexpress_m2n_run_id must be set identically on "
                        "every trainer rank"
                    )
                agreed_run_id = next(iter(configured_run_ids))
            else:
                generated_run_ids: list[str | None] = [None] * trainer_world
                dist.all_gather_object(
                    generated_run_ids,
                    uuid4().hex,
                    group=_gloo_group(),
                )
                agreed_run_id = generated_run_ids[0]
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
            model_name=str(_arg(self.args, "model", "miles-model")),
            trainer_slots=tuple(
                f"{slot_prefix}trainer-{rank}" for rank in range(trainer_world)
            ),
            generator_slots=generator_slots,
            source_partition_count=partition_count,
            m2n_abi_version=str(
                _arg(
                    self.args,
                    "modelexpress_m2n_abi_version",
                    "miles-sglang-bf16-replicated-v1",
                )
            ),
        )
        src_mesh = MeshSpec((1,), rank_offset=0)
        dst_mesh = MeshSpec((len(generator_slots),), rank_offset=1)
        pp_rank = int(self._parallel_state.pp.rank)
        local_manifest = [
            (
                name,
                tuple(int(dim) for dim in tensor.shape),
                "bfloat16",
                pp_rank,
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
        plan = ReshardPlan(
            bulk=[
                ParamPlan(
                    name=name,
                    global_shape=shape,
                    dtype=dtype,
                    partition_id=owner,
                    src_mesh=src_mesh,
                    src_placements=(Placement.replicate(),),
                    dst_mesh=dst_mesh,
                    dst_placements=(Placement.replicate(),),
                    group_key=name,
                )
                for name, shape, dtype, owner in ordered_manifest
            ],
            source_partition_count=partition_count,
        )
        if self._plan is not None and (
            plan.bulk != self._plan.bulk or topology != self._topology
        ):
            raise RuntimeError("MILES tensor names, shapes, or topology changed")
        self._topology = topology
        self._plan = plan

    def _build_tensor_digests(self) -> None:
        if self._tensors is None or self._plan is None:
            raise RuntimeError("the frozen tensor plan is unavailable")
        rank = dist.get_rank()
        local_digests: list[tuple[str, str]] = []
        local_error = ""
        try:
            local_entries = [
                entry for entry in self._plan.bulk if entry.partition_id == rank
            ]
            for entry in local_entries:
                wire_tensor = self._tensors[entry.name]
                local_shape = _local_shape(
                    entry.global_shape,
                    entry.dst_mesh,
                    entry.dst_placements,
                )
                canonical = wire_tensor
                for placement in entry.dst_placements:
                    if placement.kind is not PlacementKind.SHARD:
                        continue
                    dim = placement.dim
                    if dim is None:
                        raise ValueError(
                            f"{entry.name}: shard placement has no tensor dimension"
                        )
                    canonical = canonical.narrow(dim, 0, local_shape[dim])
                if tuple(canonical.shape) != local_shape:
                    raise ValueError(
                        f"{entry.name}: canonical wire shape {tuple(canonical.shape)} "
                        f"does not match receiver shape {local_shape}"
                    )
                local_digests.append((entry.name, _exact_tensor_sha256(canonical)))
        except BaseException as error:  # noqa: BLE001
            local_error = repr(error)

        trainer_world = dist.get_world_size()
        errors = [""] * trainer_world
        dist.all_gather_object(errors, local_error, group=_gloo_group())
        failures = [
            f"rank {worker_rank}: {error}"
            for worker_rank, error in enumerate(errors)
            if error
        ]
        if failures:
            raise RuntimeError(
                "MILES NCCL M2N tensor digest preparation failed: "
                + "; ".join(failures[:4])
            )

        gathered: list[list[tuple[str, str]] | None] = [None] * trainer_world
        dist.all_gather_object(gathered, local_digests, group=_gloo_group())
        global_digests = [
            item for rank_digests in gathered if rank_digests for item in rank_digests
        ]
        expected_names = sorted(self._plan.parameter_names())
        digest_names = [name for name, _digest in global_digests]
        if len(digest_names) != len(set(digest_names)):
            raise ValueError("PP-local tensor digest names must be globally unique")
        if sorted(digest_names) != expected_names:
            missing = sorted(set(expected_names) - set(digest_names))
            extra = sorted(set(digest_names) - set(expected_names))
            raise ValueError(
                "PP-local tensor digests must exactly match the frozen plan "
                f"(missing={missing[:5]}, extra={extra[:5]})"
            )
        self._tensor_digests = tuple(sorted(global_digests))

    async def _send_control(self, client: Any, control: CollectiveControl):
        response = await client.update_weights_from_distributed(
            names=[],
            dtypes=[],
            shapes=[],
            group_name=encode_control(control),
            flush_cache=False,
            selector="target",
        )
        return _check_response(response)

    async def _send_controls(self, requests: list[tuple[Any, CollectiveControl]]):
        return await asyncio.gather(
            *(self._send_control(client, control) for client, control in requests)
        )

    def _generator_futures(self, action: str, **kwargs):
        from miles.utils import async_utils

        if self.rollout_engines is None:
            raise RuntimeError("rollout engines are not connected")
        requests = []
        generator_slot_offset = 0
        for client, count in zip(
            self.rollout_engines,
            self._engine_gpu_counts,
            strict=True,
        ):
            control = CollectiveControl(
                action=action,
                plan=self._plan if action == "prepare" else None,
                topology=self._topology if action == "prepare" else None,
                generator_slot_offset=(
                    generator_slot_offset if action == "prepare" else None
                ),
                endpoint=_endpoint(self.args) if action == "prepare" else None,
                **kwargs,
            )
            requests.append((client, control))
            generator_slot_offset += count
        if not requests:
            return []
        coroutine = self._send_controls(requests)
        try:
            return [async_utils.submit(coroutine)]
        except BaseException:  # noqa: BLE001
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
        if self._tensors is None or self._topology is None or self._plan is None:
            raise RuntimeError("begin_sync must run before session preparation")
        rank = dist.get_rank()
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
        endpoint = _endpoint(self.args)
        channel = auth.with_auth(grpc.insecure_channel(endpoint))
        rendezvous = CollectiveRendezvous(channel)
        publisher = MilesPublisher(
            plan=self._plan,
            source_partition=rank,
            tensors=local_tensors,
            aliases={name: name for name in local_tensors},
        )
        session = MilesTrainerSession.create(
            rendezvous=rendezvous,
            topology=self._topology,
            publisher=publisher,
            source_partition=rank,
            slot_id=self._topology.trainer_slots[rank],
            worker_id=(f"miles-{self._topology.trainer_slots[rank]}-{uuid4().hex}"),
            index_in_role=rank,
            layer_groups=tuple((name,) for name in self._plan.parameter_names()),
            device=next(iter(local_tensors.values())).device,
        )
        self._channel = channel
        self._rendezvous = rendezvous
        self._session = session
        self._coordinator = MilesTransferCoordinator(rendezvous, self._topology)

        generator_futures = []
        submission_error = ""
        if rank == 0:
            try:
                generator_futures = self._generator_futures("prepare")
            except BaseException as error:  # noqa: BLE001
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
            self.close()
            raise RuntimeError(
                "MILES NCCL M2N generator preparation submission failed: "
                + "; ".join(submission_failures[:4])
            )

        local_error = ""
        try:
            session.prepare()
        except BaseException as error:  # noqa: BLE001
            local_error = repr(error)
        if rank == 0 and not local_error:
            try:
                self._wait_generator_futures(generator_futures)
                self._generators_prepared = True
            except BaseException as error:  # noqa: BLE001
                if not local_error:
                    local_error = repr(error)
        errors = [""] * dist.get_world_size()
        dist.all_gather_object(errors, local_error, group=_gloo_group())
        failures = [error for error in errors if error]
        if failures:
            self.close()
            raise RuntimeError(
                "MILES NCCL M2N session preparation failed: " + "; ".join(failures[:4])
            )

    def execute_update_round(self, context: Any) -> None:
        if not context.sync_base or context.adapters:
            raise ValueError("MILES NCCL M2N supports base-model-only update rounds")
        self._prepare_sessions()
        if self._session is None or self._coordinator is None:
            raise RuntimeError("collective sessions were not prepared")
        version = str(context.weight_version)
        rank = dist.get_rank()
        trainer_world = dist.get_world_size()
        operation: list[tuple[str | None, str]] = [(None, "")]
        create_exception: BaseException | None = None
        if rank == 0:
            try:
                group_id = self._session.membership.group_id
                operation_id = self._coordinator.create(
                    version,
                    idempotency_key=f"miles-{group_id}-weight-version-{version}",
                ).operation_id
                operation[0] = (str(operation_id), "")
            except BaseException as error:  # noqa: BLE001
                create_exception = error
                operation[0] = (None, repr(error))
        dist.broadcast_object_list(operation, src=0, group=_gloo_group())
        operation_id, create_error = operation[0]
        if create_error:
            if trainer_world == 1 and create_exception is not None:
                raise create_exception
            try:
                self.close()
            except BaseException:  # noqa: BLE001
                self._closed = True
            raise RuntimeError(
                "MILES NCCL M2N round failed: "
                f"rank 0 coordinator create: {create_error}"
            )
        if operation_id is None:
            raise RuntimeError(
                "MILES NCCL M2N round failed: coordinator returned no operation ID"
            )
        if self._verify_tensor_equality and self._tensor_digests is None:
            raise RuntimeError("MILES NCCL M2N tensor digests are unavailable")
        if rank == 0 and self._tensor_digests is not None:
            receipt = tensor_equality_receipt(
                version=version,
                operation_id=operation_id,
                tensor_digests=self._tensor_digests,
            )
            logger.info(
                "MILES tensor equality receipt role=trainer version=%s "
                "operation_id=%s tensor_count=%d sha256=%s",
                version,
                operation_id,
                len(self._tensor_digests),
                receipt,
            )
        generator_futures = []
        submission_error = ""
        submission_exception: BaseException | None = None
        if rank == 0:
            try:
                generator_futures = self._generator_futures(
                    "run_round",
                    version=version,
                    operation_id=operation_id,
                    tensor_digests=self._tensor_digests,
                )
            except BaseException as error:  # noqa: BLE001
                submission_exception = error
                submission_error = repr(error)
        submission_errors = [""] * trainer_world
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
            terminal_errors: list[str] = []
            if rank == 0:
                try:
                    self._session.report_failure(
                        operation_id=operation_id,
                        error=submission_exception
                        or RuntimeError("; ".join(submission_failures)),
                    )
                except BaseException as error:  # noqa: BLE001
                    terminal_errors.append(repr(error))
                try:
                    self._coordinator.delete(operation_id)
                except BaseException as error:  # noqa: BLE001
                    terminal_errors.append(repr(error))
            status = ["; ".join(terminal_errors)]
            dist.broadcast_object_list(status, src=0, group=_gloo_group())
            self.close()
            failures = submission_failures + [error for error in status if error]
            raise RuntimeError(
                "MILES NCCL M2N round generator submission failed: "
                + "; ".join(failures[:4])
            )

        local_error = ""
        try:
            self._session.run_round(version=version, operation_id=operation_id)
        except BaseException as error:  # noqa: BLE001
            local_error = repr(error)
        errors = [""] * dist.get_world_size()
        dist.all_gather_object(errors, local_error, group=_gloo_group())

        coordinator_error: BaseException | None = None
        if rank == 0:
            try:
                if not any(errors):
                    self._wait_generator_futures(generator_futures)
                    self._wait_terminal(operation_id)
            except BaseException as error:  # noqa: BLE001
                coordinator_error = error
            finally:
                try:
                    self._coordinator.delete(operation_id)
                except BaseException as error:  # noqa: BLE001
                    if coordinator_error is None:
                        coordinator_error = error
        status = [repr(coordinator_error) if coordinator_error is not None else ""]
        dist.broadcast_object_list(status, src=0, group=_gloo_group())
        failures = [message for message in errors + status if message]
        if failures:
            self.close()
            raise RuntimeError(
                "MILES NCCL M2N round failed: " + "; ".join(failures[:4])
            )

    def _wait_terminal(self, operation_id: str) -> None:
        if self._coordinator is None:
            raise RuntimeError("transfer coordinator is unavailable")
        timeout_s = envs.MX_NCCL_REFIT_TRANSFER_TIMEOUT_S
        deadline = time.monotonic() + timeout_s
        while True:
            transfer = self._coordinator.get(operation_id)
            if transfer.state in _TERMINAL_STATES:
                if transfer.state != collective_pb.COLLECTIVE_TRANSFER_STATE_COMPLETE:
                    raise RuntimeError(
                        transfer.failure_message
                        or f"collective operation ended in state {transfer.state}"
                    )
                logger.info(
                    "MILES NCCL M2N operation terminal operation_id=%s "
                    "version=%s state=COMPLETE",
                    operation_id,
                    transfer.version_id,
                )
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"collective operation {operation_id} did not finish "
                    f"within {timeout_s:.0f}s"
                )
            time.sleep(0.1)

    def close(self) -> None:
        if self._closed:
            return
        first_error: BaseException | None = None
        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            try:
                futures = self._generator_futures("close")
                self._wait_generator_futures(futures)
            except BaseException as error:  # noqa: BLE001
                first_error = error
        session = self._session
        self._session = None
        try:
            if session is not None:
                session.close()
        except BaseException as error:  # noqa: BLE001
            if first_error is None:
                first_error = error
        finally:
            self._coordinator = None
            if session is None and self._rendezvous is not None:
                try:
                    self._rendezvous.close()
                except BaseException as error:  # noqa: BLE001
                    if first_error is None:
                        first_error = error
            self._rendezvous = None
            if self._channel is not None:
                try:
                    self._channel.close()
                except BaseException as error:  # noqa: BLE001
                    if first_error is None:
                        first_error = error
                finally:
                    self._channel = None
            self._generators_prepared = False
        if first_error is not None:
            raise first_error
        self._closed = True
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        logger.info("MILES NCCL M2N clean shutdown complete trainer_rank=%d", rank)

    def pop_metrics(self) -> dict[str, float]:
        metrics, self.update_weight_metrics = self.update_weight_metrics, {}
        return metrics


def build_protocol(args: Any):
    """Build a MILES protocol without importing MILES at package import time."""
    from miles.backends.training_utils.weight_update.hf_weight_iterator import (
        WeightUpdatePlacement,
    )
    from miles.backends.training_utils.weight_update.protocol import (
        WeightTransferProtocol,
    )

    class MilesModelExpressProtocol(
        MilesCollectiveProtocolCore, WeightTransferProtocol
    ):
        required_placement = WeightUpdatePlacement(gather_pp=False)
        requires_exact_placement = True

        def __init__(self, protocol_args: Any) -> None:
            WeightTransferProtocol.__init__(self, protocol_args)
            MilesCollectiveProtocolCore.__init__(self, protocol_args)

    return MilesModelExpressProtocol(args)


__all__ = ["MilesCollectiveProtocolCore", "build_protocol"]
