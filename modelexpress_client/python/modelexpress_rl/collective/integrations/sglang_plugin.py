# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang general plugin for worker-local MILES NCCL M2N participation."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Lock
from typing import Any
from uuid import uuid4

import grpc

from modelexpress import auth

from .. import envs
from ..rendezvous import CollectiveRendezvous
from ._common import _endpoint, _local_shape
from .sglang import SglangGeneratorSession, SglangLoader, SglangParameterBinding
from .wire import CollectiveControl, decode_control

logger = logging.getLogger("modelexpress_rl.collective.integrations.sglang_plugin")
_PR3304_MANIFEST_VERSION = "miles-nccl-m2n-manifest-v1"

_TARGET = (
    "sglang.srt.managers.scheduler_components.weight_updater."
    "SchedulerWeightUpdaterManager.update_weights_from_distributed"
)


@dataclass(slots=True)
class _ManagerState:
    manager: Any
    session: SglangGeneratorSession
    rendezvous: CollectiveRendezvous
    channel: Any


@dataclass(slots=True)
class _RefitBindingLifecycle:
    bindings: Any
    active: bool = True
    first_start: bool = True
    closed: bool = False

    def start(self) -> None:
        if self.closed:
            raise RuntimeError("SGLang refit bindings are closed")
        if self.first_start:
            self.first_start = False
            return
        if self.active:
            raise RuntimeError("SGLang refit bindings are already active")
        self.bindings.start_round()
        self.active = True

    def finish(self) -> None:
        if not self.active:
            raise RuntimeError("SGLang refit bindings are not active")
        self.bindings.finish_round()
        self.active = False

    def fail(self) -> None:
        if self.closed:
            return
        if self.active:
            self.bindings.abort_round()
            self.active = False
        self.closed = True

    def cleanup(self) -> None:
        if self.closed:
            return
        if self.active:
            self.bindings.abort_round()
            self.active = False
        self.closed = True


# Retain the slotted manager to prevent id reuse until explicit lifecycle cleanup.
_STATES_BY_MANAGER_ID: dict[int, _ManagerState] = {}
_STATE_LOCK = Lock()


def _state(manager: Any) -> _ManagerState | None:
    with _STATE_LOCK:
        state = _STATES_BY_MANAGER_ID.get(id(manager))
        if state is not None and state.manager is not manager:
            raise RuntimeError("ModelExpress manager identity collision")
        return state


def _publish_state(
    manager: Any,
    session: SglangGeneratorSession,
    rendezvous: CollectiveRendezvous,
    channel: Any,
) -> None:
    state = _ManagerState(
        manager=manager,
        session=session,
        rendezvous=rendezvous,
        channel=channel,
    )
    with _STATE_LOCK:
        existing = _STATES_BY_MANAGER_ID.get(id(manager))
        if existing is not None:
            if existing.manager is manager:
                raise RuntimeError(
                    "the ModelExpress generator session is already prepared"
                )
            raise RuntimeError("ModelExpress manager identity collision")
        _STATES_BY_MANAGER_ID[id(manager)] = state


def _discard_state(manager: Any, session: SglangGeneratorSession) -> None:
    with _STATE_LOCK:
        state = _STATES_BY_MANAGER_ID.get(id(manager))
        if state is not None and state.manager is manager and state.session is session:
            del _STATES_BY_MANAGER_ID[id(manager)]


def _take_state(manager: Any) -> _ManagerState | None:
    with _STATE_LOCK:
        state = _STATES_BY_MANAGER_ID.get(id(manager))
        if state is None:
            return None
        if state.manager is not manager:
            raise RuntimeError("ModelExpress manager identity collision")
        return _STATES_BY_MANAGER_ID.pop(id(manager))


def _close_resources(
    session: SglangGeneratorSession | None,
    rendezvous: CollectiveRendezvous,
    channel: Any,
    *,
    suppress_errors: bool,
) -> None:
    first_error = None
    for name, resource in (
        ("session", session),
        ("rendezvous", rendezvous),
        ("channel", channel),
    ):
        if resource is None:
            continue
        try:
            resource.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
            logger.warning(
                "closing ModelExpress collective %s failed",
                name,
                exc_info=True,
            )
    if first_error is not None and not suppress_errors:
        raise first_error


def _local_tp_rank(manager: Any) -> int:
    ps = getattr(manager.tp_worker, "ps", None)
    rank = getattr(ps, "tp_rank", None)
    if rank is None:
        rank = getattr(manager.tp_worker.model_runner, "tp_rank", None)
    if rank is None:
        raise RuntimeError("SGLang worker does not expose its TP rank")
    return int(rank)


def _sglang_refit_api():
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.weight_sync.destination_adapter import (
        DESTINATION_ADAPTER_API,
        BlockFp8Spec,
        CanonicalRefitTensor,
        SglangRefitTopology,
        bind_canonical_refit_tensors,
    )

    if DESTINATION_ADAPTER_API != 2:
        raise RuntimeError(
            "ModelExpress requires SGLang destination adapter API version 2, "
            f"got {DESTINATION_ADAPTER_API!r}"
        )
    return (
        BlockFp8Spec,
        CanonicalRefitTensor,
        SglangRefitTopology,
        bind_canonical_refit_tensors,
        get_parallel,
    )


def _static_expert_placement(manager: Any) -> bool:
    args = manager.tp_worker.server_args
    return not (
        bool(getattr(args, "enable_eplb", False))
        or bool(getattr(args, "elastic_ep_backend", None))
        or int(getattr(args, "ep_num_redundant_experts", 0) or 0) != 0
        or getattr(args, "init_expert_location", "trivial") != "trivial"
    )


def _pr3304_bindings(
    manager: Any,
    control: CollectiveControl,
) -> tuple[dict[str, SglangParameterBinding], _RefitBindingLifecycle] | None:
    manifest = control.destination_manifest
    if manifest is None:
        if control.semantic_manifest_version == _PR3304_MANIFEST_VERSION:
            raise ValueError(
                "PR 3304 prepare requires exact destination manifest semantics"
            )
        return None
    if control.semantic_manifest_version != _PR3304_MANIFEST_VERSION:
        raise ValueError(
            "destination manifest semantics are supported only for "
            f"{_PR3304_MANIFEST_VERSION}"
        )
    if control.plan is None:
        raise ValueError("destination manifest requires a collective plan")
    plan_entries = list(control.plan.bulk)
    if [entry.name for entry in manifest] != [entry.name for entry in plan_entries]:
        raise ValueError(
            "destination manifest must exactly match collective plan order"
        )
    (
        BlockFp8Spec,
        CanonicalRefitTensor,
        SglangRefitTopology,
        bind_canonical_refit_tensors,
        get_parallel,
    ) = _sglang_refit_api()
    import torch

    runner = manager.tp_worker.model_runner
    ps = manager.tp_worker.ps
    parallel = get_parallel()
    descriptors = []
    manifest_by_name = {}
    for entry, destination in zip(plan_entries, manifest, strict=True):
        local_shape = _local_shape(
            entry.global_shape,
            entry.dst_mesh,
            entry.dst_placements,
        )
        if destination.dtype != entry.dtype:
            raise ValueError(
                f"{entry.name}: destination dtype differs from collective plan"
            )
        if destination.local_shape != local_shape:
            raise ValueError(
                f"{entry.name}: destination local shape differs from collective plan"
            )
        quantization = (
            BlockFp8Spec(
                quant_method=destination.quantization.quant_method,
                activation_scheme=destination.quantization.activation_scheme,
                weight_block_size=destination.quantization.weight_block_size,
                weight_dtype=_torch_dtype(torch, destination.quantization.weight_dtype),
                scale_dtype=_torch_dtype(torch, destination.quantization.scale_dtype),
                scale_format=destination.quantization.scale_format,
            )
            if destination.quantization is not None
            else None
        )
        descriptor = CanonicalRefitTensor(
            name=entry.name,
            local_shape=destination.local_shape,
            dtype=_torch_dtype(torch, destination.dtype),
            family=destination.family,
            pair_id=destination.pair_id,
            tensor_role=destination.tensor_role,
            quantization=quantization,
        )
        descriptors.append(descriptor)
        manifest_by_name[entry.name] = destination

    topology = SglangRefitTopology(
        tp_rank=int(ps.tp_rank),
        tp_size=int(ps.tp_size),
        moe_ep_rank=int(ps.moe_ep_rank),
        moe_ep_size=int(ps.moe_ep_size),
        moe_tp_rank=int(parallel.moe_tp_rank),
        moe_tp_size=int(parallel.moe_tp_size),
        dp_rank=int(ps.dp_rank if ps.dp_rank is not None else 0),
        dp_size=int(ps.dp_size),
        pp_rank=int(ps.pp_rank),
        pp_size=int(ps.pp_size),
        static_expert_placement=_static_expert_placement(manager),
    )
    adapter = bind_canonical_refit_tensors(
        runner.model,
        descriptors,
        topology=topology,
        device=runner.device,
    )
    adapter.prepare_round()
    lifecycle = _RefitBindingLifecycle(adapter)
    bindings = {}
    try:
        for descriptor in descriptors:
            destination = adapter.destinations[descriptor.name]
            metadata = destination.metadata
            expected = manifest_by_name[descriptor.name]
            if (
                metadata.canonical_name != descriptor.name
                or metadata.parameter_name != expected.parameter
                or metadata.recipe != expected.recipe
                or metadata.family != expected.family
                or metadata.tensor_role != expected.tensor_role
                or metadata.quantization != descriptor.quantization
                or tuple(metadata.receive_shape) != expected.local_shape
                or metadata.receive_dtype != descriptor.dtype
                or tuple(metadata.receive_shape) != tuple(destination.receive.shape)
                or metadata.receive_dtype != destination.receive.dtype
                or tuple(metadata.live_shape) != tuple(destination.live.shape)
                or metadata.live_dtype != destination.live.dtype
                or bool(metadata.receive_aliases_live)
                != (destination.receive.data_ptr() == destination.live.data_ptr())
                or bool(metadata.requires_install) != (destination.install is not None)
            ):
                raise ValueError(
                    f"{descriptor.name}: SGLang-resolved destination semantics differ "
                    "from the hashed PR 3304 manifest"
                )
            install = (
                None
                if destination.install is None
                else lambda _source, _live, callback=destination.install: callback()
            )
            bindings[descriptor.name] = SglangParameterBinding(
                live=destination.receive,
                receive=destination.receive if install is not None else None,
                install=install,
            )
    except BaseException:
        lifecycle.fail()
        raise
    return bindings, lifecycle


def _torch_dtype(torch: Any, dtype: str):
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "float32": torch.float32,
        }[dtype]
    except (AttributeError, KeyError) as error:
        raise ValueError(f"unsupported PR 3304 destination dtype {dtype!r}") from error


def _output(success: bool, message: str):
    from sglang.srt.managers.io_struct import UpdateWeightsFromDistributedReqOutput

    return UpdateWeightsFromDistributedReqOutput(success=success, message=message)


def _prepare(manager: Any, control: CollectiveControl):
    if _state(manager) is not None:
        raise RuntimeError("the ModelExpress generator session is already prepared")
    if control.plan is None or control.topology is None:
        raise ValueError("prepare is missing its plan or topology")
    if control.generator_slot_offset is None or control.endpoint is None:
        raise ValueError("prepare is missing its engine topology or endpoint")

    runner = manager.tp_worker.model_runner
    model = runner.model
    import torch

    resolved = _pr3304_bindings(manager, control)
    lifecycle = None
    if resolved is None:
        bindings = {}
        for entry in control.plan.bulk:
            receive = torch.empty(
                _local_shape(
                    entry.global_shape,
                    entry.dst_mesh,
                    entry.dst_placements,
                ),
                dtype=torch.bfloat16,
                device=runner.device,
            )

            def install(source, _live, *, name=entry.name):
                model.load_weights([(name, source)])

            bindings[entry.name] = SglangParameterBinding(
                live=receive,
                receive=receive,
                install=install,
            )
    else:
        bindings, lifecycle = resolved

    channel = None
    rendezvous = None
    session = None
    try:
        layer_groups = tuple((entry.name,) for entry in control.plan.bulk)
        loader = SglangLoader(
            plan=control.plan,
            bindings=bindings,
            layer_groups=layer_groups,
            start=lifecycle.start if lifecycle is not None else None,
            finish=lifecycle.finish if lifecycle is not None else None,
            fail=lifecycle.fail if lifecycle is not None else None,
            cleanup=lifecycle.cleanup if lifecycle is not None else None,
        )
        local_index = control.generator_slot_offset + _local_tp_rank(manager)
        try:
            slot_id = control.topology.generator_slots[local_index]
        except IndexError as error:
            raise RuntimeError(
                "the explicit generator topology does not contain generator index "
                f"{local_index}"
            ) from error
        channel = auth.with_auth(grpc.insecure_channel(_endpoint(control.endpoint)))
        rendezvous = CollectiveRendezvous(channel)
        session = SglangGeneratorSession.create(
            rendezvous=rendezvous,
            topology=control.topology,
            loader=loader,
            slot_id=slot_id,
            worker_id=f"sglang-{slot_id}-{uuid4().hex}",
            index_in_role=local_index,
            semantic_manifest_version=control.semantic_manifest_version,
            semantic_manifest_digest=control.semantic_manifest_digest,
            safe_point=nullcontext,
            layer_groups=layer_groups,
            device=runner.device,
        )
        session.prepare()
        _publish_state(manager, session, rendezvous, channel)
    except BaseException:
        if session is not None:
            _discard_state(manager, session)
        _close_resources(
            session,
            rendezvous,
            channel,
            suppress_errors=True,
        )
        if lifecycle is not None:
            lifecycle.cleanup()
        raise
    return _output(True, f"prepared ModelExpress collective slot {slot_id}")


def _run_round(manager: Any, control: CollectiveControl):
    state = _state(manager)
    if state is None:
        raise RuntimeError("the ModelExpress generator session is not prepared")
    verification_enabled = envs.MX_MILES_VERIFY_TENSOR_EQUALITY
    if verification_enabled != (control.tensor_digests is not None):
        failed_state = _take_state(manager)
        if failed_state is not None:
            _close_resources(
                failed_state.session,
                failed_state.rendezvous,
                failed_state.channel,
                suppress_errors=True,
            )
        raise ValueError(
            "trainer and SGLang disagree on MX_MILES_VERIFY_TENSOR_EQUALITY"
        )
    state.session.run_round(
        version=str(control.version),
        operation_id=str(control.operation_id),
        tensor_digests=control.tensor_digests,
    )
    return _output(True, f"completed ModelExpress collective {control.operation_id}")


def _close(manager: Any):
    state = _take_state(manager)
    if state is not None:
        _close_resources(
            state.session,
            state.rendezvous,
            state.channel,
            suppress_errors=False,
        )
    return _output(True, "closed ModelExpress collective session")


def around_update_weights_from_distributed(
    original,
    manager: Any,
    recv_req: Any,
):
    """Intercept marked bridge commands and preserve all stock requests."""
    try:
        control = decode_control(getattr(recv_req, "group_name", None))
    except BaseException as error:
        return _output(False, f"invalid ModelExpress collective control: {error}")
    if control is None:
        return original(manager, recv_req)
    try:
        if control.action != "close" and not getattr(
            manager,
            "_weight_update_in_progress",
            False,
        ):
            raise RuntimeError(
                "ModelExpress collective commands require an open "
                "begin_weight_update session"
            )
        if control.action == "prepare":
            return _prepare(manager, control)
        if control.action == "run_round":
            return _run_round(manager, control)
        return _close(manager)
    except BaseException as error:
        logger.exception("ModelExpress collective %s failed", control.action)
        return _output(
            False, f"ModelExpress collective {control.action} failed: {error}"
        )


def register_modelexpress_miles_collective() -> None:
    """Register the worker-local hook through SGLang's supported plugin API."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        _TARGET,
        around_update_weights_from_distributed,
        HookType.AROUND,
    )


__all__ = [
    "around_update_weights_from_distributed",
    "register_modelexpress_miles_collective",
]
