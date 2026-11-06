# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native vLLM endpoint and worker extensions for artifact startup signaling."""

import logging
from weakref import ReferenceType, ref

from ... import envs
from ..startup import StartupGate

logger = logging.getLogger(__name__)
_gates: dict[int, tuple[ReferenceType, StartupGate]] = {}


def prepare_artifact_gate(ctx) -> bool:
    """Bind publishers to a worker's config without modifying that config."""
    config = getattr(getattr(ctx, "adapter", None), "vllm_config", None)
    if config is None:
        return False
    parallel = getattr(config, "parallel_config", None)
    if getattr(parallel, "distributed_executor_backend", None) == "external_launcher":
        logger.warning(
            "External-launcher startup is not qualified; skipping artifacts"
        )
        return False
    key = id(config)
    entry = _gates.get(key)
    if entry is None:
        try:
            config_ref = ref(config, lambda reference: _gates.pop(key, None))
        except TypeError:
            logger.warning("Unsupported vLLM config lifetime; skipping artifacts")
            return False
        gate = StartupGate()
        gate.prepare()
        _gates[key] = (config_ref, gate)
    else:
        _, gate = entry
    ctx.artifact_ready_fn = gate.is_ready
    return True


class ArtifactStartupWorkerExtension:
    """Add a control RPC through vLLM's worker_extension_cls interface."""

    def mx_artifact_startup_complete(self) -> bool:
        entry = _gates.get(id(self.vllm_config))
        if entry is None:
            return False
        _, gate = entry
        gate.complete(lambda: None)
        return True


class ArtifactStartupPlugin:
    """Release publishers from the native post-engine app-state callback."""

    name = "modelexpress"
    required_tasks = None

    def attach_router(self, app) -> None:
        """This lifecycle plugin adds no HTTP routes."""

    async def init_state(self, engine_client, state, args) -> None:
        if (
            engine_client is None
            or not envs.MX_ARTIFACT_TRANSFER
            or envs.MX_ARTIFACT_READY_URL.strip()
        ):
            return
        try:
            results = await engine_client.collective_rpc(
                "mx_artifact_startup_complete",
                timeout=envs.MX_ARTIFACT_READY_TIMEOUT_SECS,
            )
            if not isinstance(results, list) or not results or not all(
                result is True for result in results
            ):
                logger.warning("Not all vLLM workers accepted artifact startup")
        except Exception:
            logger.warning(
                "Native vLLM artifact startup signal failed; configure "
                "ArtifactStartupWorkerExtension or an explicit "
                "MX_ARTIFACT_READY_URL. Artifacts remain gated.",
                exc_info=True,
            )
