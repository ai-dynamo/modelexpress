# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelStreamer loading strategy: stream safetensors via runai-model-streamer.

Supports object storage (S3, GCS, Azure Blob) and local filesystem paths.
File resolution is delegated to runai_model_streamer.list_safetensors(),
which handles all backends transparently.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import time
from typing import Iterator

import torch

from ..adapter import EngineAdapter, StrategyFailed
from .base import LoadContext, LoadStrategy, _as_load_result, register_tensors
from .context import LoadResult

logger = logging.getLogger("modelexpress.strategy_model_streamer")

_LOG_INTERVAL_SECS = 30


def _patch_vllm_s3_format_check() -> None:
    """Allow 'mx' as a valid load format when the model path is an object storage URI.

    vllm's `try_verify_and_update_config` rejects any `load_format` other than
    'runai_streamer'/'runai_streamer_sharded' when `model_config.model_weights`
    is an object-storage URI. The guard is the only place inside that method
    that consults `model_weights`, so we temporarily detach the attribute for
    the duration of the call and restore it afterwards. This keeps
    `load_format == 'mx'` truthful throughout verification.

    Co-located with ModelStreamerStrategy because that strategy is the only
    consumer of object-storage URIs in the mx pathway. Installed eagerly from
    `vllm_loader` at module-import time, before vllm runs verification.

    No-ops gracefully if the vllm version does not have this check.
    """
    try:
        from vllm.config import VllmConfig
        from vllm.transformers_utils.runai_utils import is_runai_obj_uri
    except ImportError:
        return

    original = VllmConfig.try_verify_and_update_config

    def patched(self: VllmConfig) -> None:
        if (
            self.load_config.load_format == "mx"
            and hasattr(self.model_config, "model_weights")
            and is_runai_obj_uri(self.model_config.model_weights)
        ):
            saved = self.model_config.model_weights
            del self.model_config.model_weights
            try:
                original(self)
            finally:
                self.model_config.model_weights = saved
        else:
            original(self)

    VllmConfig.try_verify_and_update_config = patched
    logger.debug("Patched VllmConfig.try_verify_and_update_config to allow 'mx' for object storage URIs")


class ModelStreamerStrategy(LoadStrategy):
    """Load weights by streaming safetensors via runai-model-streamer.

    Activated by setting MX_MODEL_URI (gate only). The actual URI used to
    stream weights is taken from `model_config.model_weights` if set
    (object-storage URIs: s3://, gs://, az://) and falls back to
    `model_config.model` otherwise (local paths, HF-resolved snapshots).
    This mirrors vllm's runai_streamer_loader.
    """

    name = "model_streamer"
    requires = (EngineAdapter.apply_weight_iter,)

    def is_available(self, ctx: LoadContext) -> bool:
        if not super().is_available(ctx):
            return False
        if importlib.util.find_spec("runai_model_streamer") is None:
            logger.info(
                f"[Worker {ctx.global_rank}] runai_model_streamer not installed, skipping"
            )
            return False

        model_uri = os.environ.get("MX_MODEL_URI", "")
        if not model_uri:
            logger.info(
                f"[Worker {ctx.global_rank}] MX_MODEL_URI not set, skipping model streamer"
            )
            return False
        return True

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        result = _as_load_result(result)
        model_uri = getattr(ctx.model_config, "model_weights", None) or ctx.model_config.model

        logger.info(f"[Worker {ctx.global_rank}] Attempting model streamer loading from {model_uri}")
        try:
            weights_iter = self._stream_weights(model_uri, ctx)
        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] Model streamer loading failed, falling through: {e}"
            )
            raise StrategyFailed(str(e), mutated=False) from e

        try:
            result = ctx.adapter.apply_weight_iter(result, weights_iter)
            logger.info(f"[Worker {ctx.global_rank}] Model streamer weight loading complete")
            result = ctx.adapter.after_weight_iter_load(result)
        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] Model streamer loading failed, falling through: {e}"
            )
            raise StrategyFailed(str(e), mutated=True) from e

        register_tensors(result, ctx)
        return result

    def _stream_weights(
        self, model_uri: str, ctx: LoadContext
    ) -> Iterator[tuple[str, torch.Tensor]]:
        from runai_model_streamer import SafetensorsStreamer, list_safetensors

        file_uris = list_safetensors(model_uri)
        if not file_uris:
            raise FileNotFoundError(f"No safetensors files found at {model_uri}")

        logger.info(
            f"[Worker {ctx.global_rank}] Streaming {len(file_uris)} safetensors files "
            f"from {model_uri}"
        )

        tp_size = getattr(ctx.identity, "tensor_parallel_size", 1) or 1
        distributed = (
            tp_size > 1
            and ctx.target_device.type == "cuda"
            and os.environ.get("MX_MS_DISTRIBUTED", "0").lower() in ("1", "true")
        )
        stream_kwargs: dict = {}
        if distributed:
            stream_kwargs["device"] = f"cuda:{ctx.device_id}"
            stream_kwargs["is_distributed"] = True

        start = time.perf_counter()
        with SafetensorsStreamer() as streamer:
            streamer.stream_files(file_uris, **stream_kwargs)
            total_tensors = sum(
                len(meta) for meta in streamer.files_to_tensors_metadata.values()
            )
            count = 0
            last_log = start
            for name, tensor in streamer.get_tensors():
                count += 1
                now = time.perf_counter()
                if now - last_log >= _LOG_INTERVAL_SECS or count == total_tensors:
                    pct = count / total_tensors * 100 if total_tensors else 0
                    elapsed = now - start
                    logger.info(
                        f"[Worker {ctx.global_rank}] Streaming: "
                        f"{count}/{total_tensors} tensors ({pct:.0f}%) "
                        f"in {elapsed:.0f}s"
                    )
                    last_log = now
                # clone() ensures safety when RUNAI_STREAMER_MEMORY_LIMIT=0
                # (single-buffer mode reuses memory on next iteration)
                yield name, tensor.clone()
        elapsed = time.perf_counter() - start
        logger.info(f"[Worker {ctx.global_rank}] Streamed all weights in {elapsed:.1f}s")
