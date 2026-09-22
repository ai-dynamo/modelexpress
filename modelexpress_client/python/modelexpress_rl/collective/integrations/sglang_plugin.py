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

    layer_groups = tuple((entry.name,) for entry in control.plan.bulk)
    loader = SglangLoader(
        plan=control.plan,
        bindings=bindings,
        layer_groups=layer_groups,
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
    session = None
    try:
        session = SglangGeneratorSession.create(
            rendezvous=rendezvous,
            topology=control.topology,
            loader=loader,
            slot_id=slot_id,
            worker_id=f"sglang-{slot_id}-{uuid4().hex}",
            index_in_role=local_index,
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
