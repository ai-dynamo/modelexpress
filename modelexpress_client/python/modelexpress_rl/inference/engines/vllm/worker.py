# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM worker that initializes ModelExpress transport before vLLM.

ModelExpress creates its NIXL/UCX transport before vLLM creates NCCL
communicators. This keeps CUDA-device and network-rail selection deterministic
for both transports while allowing the ModelExpress client itself to be created
later, after vLLM has constructed the model runner.
"""

from __future__ import annotations

from typing import Any

from vllm.v1.worker.gpu_worker import Worker as VllmWorker

from ...bootstrap import _ModelExpressGeneratorBootstrap


def _get_vllm_bootstrap_device_id(vllm_config: Any, local_rank: int) -> int:
    """Resolve the CUDA device using the same mapping vLLM will install."""
    # A Ray worker's local rank is not necessarily its physical or visible CUDA
    # device. Apply vLLM's assignment first so NIXL and vLLM bind the same GPU.
    assigned_gpu_ids = vllm_config.parallel_config.assigned_physical_gpu_ids
    if assigned_gpu_ids is not None:
        from vllm.platforms.interface import set_assigned_physical_gpu_ids

        set_assigned_physical_gpu_ids(assigned_gpu_ids)

    from vllm.platforms import current_platform

    return int(current_platform.logical_device_id_to_visible_device_id(local_rank))


class _VllmGeneratorBootstrap(_ModelExpressGeneratorBootstrap):
    """Resolve vLLM's pre-init device assignment and create MX transport."""

    def __init__(self, vllm_config: Any, local_rank: int) -> None:
        super().__init__(
            device_id=_get_vllm_bootstrap_device_id(vllm_config, local_rank)
        )


class ModelExpressVllmWorker(VllmWorker):
    """Initialize ModelExpress transport before vLLM creates NCCL groups.

    The generator client is initialized later by the framework, so the early
    transport is registered process-locally. Client initialization claims it
    and assumes ownership; no bootstrap object crosses the framework API.
    """

    def __init__(
        self,
        vllm_config: Any,
        local_rank: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        bootstrap = _VllmGeneratorBootstrap(vllm_config, local_rank)
        try:
            # Publish only after NIXL initialization succeeds, but before the
            # base worker can create NCCL state.
            bootstrap.register_default()
            self._model_express_bootstrap = bootstrap
            super().__init__(vllm_config, local_rank, *args, **kwargs)
        except BaseException:
            # Avoid retaining listeners or CUDA registrations when vLLM fails
            # before the generator client can claim the transport.
            bootstrap.close()
            raise


__all__ = ["ModelExpressVllmWorker"]
