# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang safe-point binding for exact-version RL refits.

The engine owns pause, TP result aggregation, post-load hooks and fleet resume.
Any apply failure fences this binding until the worker process is replaced.
"""

from __future__ import annotations

import json
import logging
import threading
import time

from modelexpress_rl.inference.client import (
    ModelExpressGeneratorClient,
    ModelExpressGeneratorConfig,
)
from modelexpress_rl.inference.plan import WeightSource
from modelexpress_rl.version import WeightVersionRef
from modelexpress.refit.timing import RefitTimingRecorder, use_refit_timing
from .context import SglangGeneratorContext

logger = logging.getLogger("modelexpress.sglang.refit")


class SglangLiveRefit:
    def __init__(self, runner, *, model_name, timeout=600.0):
        parallel_state = getattr(runner, "ps", runner)
        self._rank = getattr(parallel_state, "tp_rank", 0)
        self._client = ModelExpressGeneratorClient.initialize(
            ModelExpressGeneratorConfig(
                engine_context=SglangGeneratorContext(runner, enable_full_tensor=True),
                model_name=model_name,
                source_order=(WeightSource.TRAINER,),
                rpc_timeout_seconds=timeout,
            )
        )
        self._lock = threading.Lock()
        self._poisoned = False
        self._step = None
        self._version_id = None

    def update(self, *, version_id, training_step):
        with self._lock:
            result = dict(
                success=False,
                target_training_step=training_step,
                installed_training_step=self._step,
                version_id=version_id,
                layout_signature=None,
                receiver_poisoned=self._poisoned,
                metrics={},
                timing=None,
                error=None,
            )
            handle = None
            applying = False
            cold = self._step is None
            started = time.perf_counter()
            recorder = RefitTimingRecorder(
                backend="sglang-rl-nixl",
                version=training_step,
                rank=self._rank,
                cold=cold,
            )
            with use_refit_timing(recorder):
                try:
                    if self._poisoned:
                        raise RuntimeError(
                            "SGLang refit is poisoned; restart the worker"
                        )
                    if not version_id or training_step < 0:
                        raise ValueError(
                            "an exact MX version and nonnegative training step are required"
                        )
                    if self._step is not None and training_step <= self._step:
                        if (
                            training_step == self._step
                            and version_id == self._version_id
                        ):
                            result["success"] = True
                            return result
                        raise RuntimeError(
                            "refusing stale or conflicting SGLang refit version"
                        )
                    handle = self._client.stage_weight(
                        version=WeightVersionRef(version_id)
                    )
                    staged_at = time.perf_counter()
                    applying = True
                    install_metrics = self._client.apply_weight(handle) or {}
                    applying = False
                    self._step, self._version_id = training_step, version_id
                    result.update(
                        success=True,
                        installed_training_step=training_step,
                        metrics={
                            **handle.metrics,
                            **install_metrics,
                            "stage_s": staged_at - started,
                            "total_s": time.perf_counter() - started,
                        },
                    )
                except BaseException as error:
                    if applying:
                        self._poisoned = True
                    result.update(
                        error=f"{type(error).__name__}: {error}",
                        receiver_poisoned=self._poisoned,
                    )
                    logger.exception("SGLang RL refit failed closed")
                finally:
                    if handle is not None:
                        try:
                            handle.release()
                        except Exception as error:
                            result.update(
                                success=False, error=f"staged release failed: {error}"
                            )
            recorder.mark_not_applicable(
                "rollout_readiness", reason="Miles owns fleet activation"
            )
            result["timing"] = recorder.emit(logger)
            logger.info(
                "[MX_RL_REFIT] %s",
                json.dumps(
                    {
                        "rank": self._rank,
                        "cold": cold,
                        **result,
                        "elapsed_s": time.perf_counter() - started,
                    }
                ),
            )
            return result

    def close(self):
        self._client.close()
