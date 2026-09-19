# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM worker that bootstraps ModelExpress before engine initialization."""

from __future__ import annotations

from typing import Any

from vllm.v1.worker.gpu_worker import Worker as VllmWorker

from ...bootstrap import ModelExpressGeneratorBootstrap


def _get_vllm_bootstrap_device_id(vllm_config: Any, local_rank: int) -> int:
    """Resolve vLLM's visible CUDA device before worker initialization."""
    assigned_gpu_ids = vllm_config.parallel_config.assigned_physical_gpu_ids
    if assigned_gpu_ids is not None:
        from vllm.platforms.interface import set_assigned_physical_gpu_ids

        set_assigned_physical_gpu_ids(assigned_gpu_ids)

    from vllm.platforms import current_platform

    return int(current_platform.logical_device_id_to_visible_device_id(local_rank))


class VllmGeneratorBootstrap(ModelExpressGeneratorBootstrap):
    """Resolve vLLM's pre-init device assignment and create MX transport."""

    def __init__(self, vllm_config: Any, local_rank: int) -> None:
        super().__init__(
            device_id=_get_vllm_bootstrap_device_id(vllm_config, local_rank)
        )


class ModelExpressVllmWorker(VllmWorker):
    """Initialize ModelExpress transport before vLLM creates NCCL groups."""

    def __init__(
        self,
        vllm_config: Any,
        local_rank: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._model_express_bootstrap = VllmGeneratorBootstrap(vllm_config, local_rank)
        super().__init__(vllm_config, local_rank, *args, **kwargs)


__all__ = ["ModelExpressVllmWorker"]
