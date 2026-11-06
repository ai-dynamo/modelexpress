# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backport a post-warmup artifact signal without importing workers at plugin load."""

import logging
from functools import wraps
from importlib import import_module

from ... import envs
from ..startup import StartupGate

logger = logging.getLogger(__name__)
gate = StartupGate()


def _synchronize() -> None:
    from vllm.distributed import get_world_group

    get_world_group().barrier()


def install_startup_hook(ctx) -> bool:
    """Wrap the resolved worker at loader time, before startup warmup runs."""
    if (
        not envs.MX_ARTIFACT_TRANSFER
        or envs.MX_DISABLE_PATCHES
        or envs.MX_ARTIFACT_READY_URL.strip()
    ):
        return False
    config = getattr(getattr(ctx, "adapter", None), "vllm_config", None)
    parallel = getattr(config, "parallel_config", None)
    worker_cls = getattr(parallel, "worker_cls", "auto")
    if worker_cls == "auto":
        worker_cls = "vllm.v1.worker.gpu_worker.Worker"
    try:
        if isinstance(worker_cls, str):
            module_name, class_name = worker_cls.rsplit(".", 1)
            worker_cls = getattr(import_module(module_name), class_name)
        original = getattr(worker_cls, "compile_or_warm_up_model", None)
    except (ImportError, AttributeError, ValueError):
        original = None
    if not callable(original):
        logger.warning(
            "No supported vLLM startup hook; using HTTP artifact readiness"
        )
        return False

    if not getattr(original, "__modelexpress_startup_hook__", False):
        @wraps(original)
        def warmup(self, *args, **kwargs):
            gate.prepare()
            result = original(self, *args, **kwargs)
            gate.complete(_synchronize)
            return result

        warmup.__modelexpress_startup_hook__ = True
        worker_cls.compile_or_warm_up_model = warmup
    if not gate.available:
        gate.prepare()
    return True
