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
import torch.nn as nn

from .base import LoadContext, LoadStrategy, register_tensors, publish_metadata
from ..tensor_utils import capture_tensor_attrs

logger = logging.getLogger("modelexpress.strategy_model_streamer")

_LOG_INTERVAL_SECS = 30


_REMOTE_SCHEMES = ("s3://", "gs://", "az://")


def _resolve_model_uri(uri: str) -> str:
    """Resolve MX_MODEL_URI to a path that runai-model-streamer can read.

    - s3://, gs://, az:// -> pass through (remote storage)
    - /absolute/path -> pass through (local filesystem)
    - org/model-name -> resolve via HuggingFace Hub cache (HF_HUB_CACHE)
    """
    if any(uri.startswith(s) for s in _REMOTE_SCHEMES) or os.path.isabs(uri):
        return uri

    try:
        from huggingface_hub import scan_cache_dir
        cache_dir = os.environ.get("HF_HUB_CACHE")
        if not cache_dir:
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                cache_dir = os.path.join(hf_home, "hub")
        if cache_dir:
            cache_info = scan_cache_dir(cache_dir)
            for repo in cache_info.repos:
                if repo.repo_id == uri:
                    rev = max(repo.revisions, key=lambda r: r.last_modified)
                    logger.info(f"Resolved HF model '{uri}' -> {rev.snapshot_path}")
                    return str(rev.snapshot_path)
    except Exception as e:
        logger.warning(f"Failed to resolve HF cache for '{uri}': {e}")

    return uri


class ModelStreamerStrategy(LoadStrategy):
    """Load weights by streaming safetensors via runai-model-streamer.

    Activated by setting MX_MODEL_URI. Supported formats:
      - Remote: s3://bucket/model, gs://bucket/model, az://container/model
      - Local absolute path: /models/deepseek-ai/DeepSeek-V3
      - HF model ID: deepseek-ai/DeepSeek-V3 (resolved via HF_HUB_CACHE)
    """

    name = "model_streamer"

    def is_available(self, ctx: LoadContext) -> bool:
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

    def load(self, model: nn.Module, ctx: LoadContext) -> bool:
        model_uri = _resolve_model_uri(os.environ["MX_MODEL_URI"])

        logger.info(f"[Worker {ctx.global_rank}] Attempting model streamer loading from {model_uri}")
        try:
            weights_iter = self._stream_weights(model_uri, ctx)
            model.load_weights(weights_iter)
            logger.info(f"[Worker {ctx.global_rank}] Model streamer weight loading complete")
        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] Model streamer loading failed, falling through: {e}"
            )
            return False

        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        with capture_tensor_attrs():
            process_weights_after_loading(model, ctx.model_config, ctx.target_device)

        register_tensors(model, ctx)
        publish_metadata(ctx)
        return True

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

        start = time.perf_counter()
        with SafetensorsStreamer() as streamer:
            streamer.stream_files(file_uris)
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


