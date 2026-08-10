# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-side contract for an upstream SGLang live-refit endpoint."""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from modelexpress.engines.sglang.loader import get_sglang_loader_state
from modelexpress.refit.reshard.receiver import ReshardTopologyChanged
from modelexpress.refit.timing import RefitTimingRecorder, use_refit_timing

from .receiver import SglangReshardReceiver

logger = logging.getLogger("modelexpress.engines.sglang.refit.worker")

_receivers: dict[int, SglangReshardReceiver] = {}
_receiver_configs: dict[int, tuple[Any, ...]] = {}
_installed_versions: dict[int, int] = {}
_locks: dict[int, threading.Lock] = {}


@dataclass(frozen=True)
class SglangRefitRequest:
    """One whole-model refit request accepted by a SGLang worker."""

    target_training_step: int
    logical_group: str = "model"
    expected_layout_signature: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SglangRefitRequest":
        allowed = {
            "target_training_step",
            "logical_group",
            "expected_layout_signature",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unsupported SGLang refit request fields: {unknown}")
        return cls(**payload)

    def validate(self) -> None:
        if isinstance(self.target_training_step, bool) or not isinstance(
            self.target_training_step, int
        ):
            raise ValueError("target_training_step must be an integer")
        if self.target_training_step < 0:
            raise ValueError("target_training_step must be non-negative")
        if self.logical_group != "model":
            raise ValueError(
                "SGLang live refit currently supports only logical_group='model'"
            )
        if self.expected_layout_signature is not None and not isinstance(
            self.expected_layout_signature, str
        ):
            raise ValueError("expected_layout_signature must be a string")


@dataclass(frozen=True)
class SglangRefitResponse:
    """JSON-serializable result returned to the upstream SGLang endpoint."""

    success: bool
    target_training_step: int
    installed_training_step: int | None
    layout_signature: str | None
    metrics: dict[str, Any] | None = None
    timing: dict[str, Any] | None = None
    error: str | None = None
    receiver_poisoned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _receiver_for(
    *,
    device_id: int,
    num_trainer_sources: int,
    listen_port: int,
    timeout: float,
) -> SglangReshardReceiver:
    state = get_sglang_loader_state(device_id)
    ctx = state.context
    config = (
        id(state.model),
        num_trainer_sources,
        listen_port,
        timeout,
    )
    existing = _receivers.get(device_id)
    if existing is not None:
        if _receiver_configs[device_id] != config:
            raise RuntimeError(
                "SGLang live-refit receiver configuration changed after first use; "
                "restart the worker to rebuild its cached transfer plan"
            )
        return existing

    receiver = SglangReshardReceiver(
        model=state.model,
        tensor_registry=state.tensors,
        model_config=ctx.model_config,
        model_name=ctx.identity.model_name,
        mx_server=ctx.mx_client.server_url,
        agent_name=f"sglang-refit-{ctx.global_rank}",
        local_rank=ctx.device_id,
        global_rank=ctx.global_rank,
        num_trainer_sources=num_trainer_sources,
        device=ctx.target_device,
        listen_port=listen_port,
        timeout=timeout,
        mx_client=ctx.mx_client,
    )
    _receivers[device_id] = receiver
    _receiver_configs[device_id] = config
    return receiver


def run_sglang_live_refit(
    request: SglangRefitRequest | Mapping[str, Any],
    *,
    device_id: int,
    num_trainer_sources: int,
    listen_port: int,
    timeout: float = 1200.0,
) -> SglangRefitResponse:
    """Execute one serialized, fail-closed refit on the current worker.

    This is the worker entrypoint an upstream SGLang HTTP/RPC handler should
    invoke on every model worker. It deliberately has no cohort-generation or
    partial-update fields: the current MX rendezvous cannot verify either.
    """
    if not isinstance(request, SglangRefitRequest):
        request = SglangRefitRequest.from_mapping(request)
    request.validate()
    if num_trainer_sources <= 0:
        raise ValueError("num_trainer_sources must be positive")

    lock = _locks.setdefault(device_id, threading.Lock())
    with lock:
        receiver: SglangReshardReceiver | None = None
        timing: dict[str, Any] | None = None
        recorder = RefitTimingRecorder(
            backend="sglang-reshard-nixl",
            version=request.target_training_step,
            rank=device_id,
        )
        try:
            receiver = _receiver_for(
                device_id=device_id,
                num_trainer_sources=num_trainer_sources,
                listen_port=listen_port,
                timeout=timeout,
            )
            recorder.rank = getattr(receiver, "_global_rank", device_id)
            signature = receiver.layout_signature
            if (
                request.expected_layout_signature is not None
                and request.expected_layout_signature != signature
            ):
                raise RuntimeError(
                    "SGLang refit layout signature mismatch: "
                    f"expected={request.expected_layout_signature}, actual={signature}"
                )
            if receiver.poisoned:
                raise RuntimeError(
                    "SGLang receiver is poisoned by a prior in-place install failure"
                )

            current = _installed_versions.get(device_id)
            if current is not None and request.target_training_step < current:
                raise RuntimeError(
                    f"refusing version rollback from training step {current} to "
                    f"{request.target_training_step}"
                )
            if current == request.target_training_step:
                recorder.set_cold(False)
                recorder.mark_not_applicable(
                    "rollout_readiness",
                    reason="idempotent version already installed",
                )
                timing = recorder.emit(logger)
                return SglangRefitResponse(
                    success=True,
                    target_training_step=request.target_training_step,
                    installed_training_step=current,
                    layout_signature=signature,
                    metrics={"idempotent": True},
                    timing=timing,
                )

            cold = receiver._plan is None
            recorder.set_cold(cold)
            with use_refit_timing(recorder):
                try:
                    metrics = receiver.update_weights(
                        request.target_training_step,
                        timeout=timeout,
                    )
                except ReshardTopologyChanged:
                    logger.warning(
                        "SGLang refit topology changed; rebuilding the receiver "
                        "before retrying step %s",
                        request.target_training_step,
                    )
                    receiver.close()
                    _receivers.pop(device_id, None)
                    _receiver_configs.pop(device_id, None)
                    recorder.set_cold(True)
                    receiver = _receiver_for(
                        device_id=device_id,
                        num_trainer_sources=num_trainer_sources,
                        listen_port=listen_port,
                        timeout=timeout,
                    )
                    metrics = receiver.update_weights(
                        request.target_training_step,
                        timeout=timeout,
                    )
            _installed_versions[device_id] = request.target_training_step
            if not cold:
                for stage in (
                    "control_discovery",
                    "source_preparation",
                    "setup_registration",
                    "transfer_planning",
                ):
                    if not recorder.has_measurements(stage):
                        recorder.mark_not_applicable(
                            stage, reason="cached stable-topology plan"
                        )
            recorder.mark_not_applicable(
                "post_install", reason="Qwen3 BF16 has no refit-derived tensors"
            )
            recorder.mark_not_applicable(
                "rollout_readiness",
                reason="upstream SGLang endpoint owns readiness/barrier",
            )
            timing = recorder.emit(logger)
            return SglangRefitResponse(
                success=True,
                target_training_step=request.target_training_step,
                installed_training_step=request.target_training_step,
                layout_signature=signature,
                metrics=metrics,
                timing=timing,
            )
        except BaseException as exc:
            timing = recorder.emit(logger)
            logger.exception(
                "SGLang live refit failed closed for training step %s",
                request.target_training_step,
            )
            return SglangRefitResponse(
                success=False,
                target_training_step=request.target_training_step,
                installed_training_step=_installed_versions.get(device_id),
                layout_signature=(
                    receiver.layout_signature if receiver is not None else None
                ),
                timing=timing,
                error=f"{type(exc).__name__}: {exc}",
                receiver_poisoned=bool(receiver and receiver.poisoned),
            )


__all__ = [
    "SglangRefitRequest",
    "SglangRefitResponse",
    "run_sglang_live_refit",
]
