# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelStreamer loading strategy: stream safetensors from S3 to GPU via runai-model-streamer."""

from __future__ import annotations

import importlib.util
import json
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


class ModelStreamerStrategy(LoadStrategy):
    """Load weights by streaming safetensors from S3 via runai-model-streamer."""

    name = "model_streamer"

    def is_available(self, ctx: LoadContext) -> bool:
        s3_uri = os.environ.get("MX_S3_URI", "")
        if not s3_uri:
            logger.info(f"[Worker {ctx.global_rank}] MX_S3_URI not set, skipping model streamer")
            return False
        if importlib.util.find_spec("runai_model_streamer") is None:
            logger.info(
                f"[Worker {ctx.global_rank}] runai_model_streamer not installed, skipping"
            )
            return False
        return True

    def load(self, model: nn.Module, ctx: LoadContext) -> bool:
        s3_uri = os.environ["MX_S3_URI"]
        logger.info(f"[Worker {ctx.global_rank}] Attempting model streamer loading from {s3_uri}")
        try:
            weights_iter = self._stream_weights(s3_uri, ctx)
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
        self, s3_uri: str, ctx: LoadContext
    ) -> Iterator[tuple[str, torch.Tensor]]:
        from runai_model_streamer import SafetensorsStreamer

        file_uris = self._resolve_s3_safetensors(s3_uri)
        logger.info(
            f"[Worker {ctx.global_rank}] Streaming {len(file_uris)} safetensors files "
            f"from {s3_uri}"
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
                        f"[Worker {ctx.global_rank}] S3 streaming: "
                        f"{count}/{total_tensors} tensors ({pct:.0f}%) "
                        f"in {elapsed:.0f}s"
                    )
                    last_log = now
                yield name, tensor.clone()
        elapsed = time.perf_counter() - start
        logger.info(f"[Worker {ctx.global_rank}] Streamed all weights in {elapsed:.1f}s")

    @staticmethod
    def _resolve_s3_safetensors(s3_uri: str) -> list[str]:
        """Resolve safetensors file URIs from an S3 prefix.

        Tries model.safetensors.index.json first, then falls back to
        listing all .safetensors files under the prefix.
        """
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Expected s3:// URI, got: {s3_uri}")

        import boto3
        from botocore.exceptions import ClientError

        path = s3_uri.removeprefix("s3://")
        bucket, _, prefix = path.partition("/")
        prefix = prefix.rstrip("/")

        s3 = boto3.client("s3")

        index_key = f"{prefix}/model.safetensors.index.json"
        try:
            resp = s3.get_object(Bucket=bucket, Key=index_key)
            index = json.loads(resp["Body"].read())
            weight_map = index.get("weight_map", {})
            filenames = sorted(set(weight_map.values()))
            if filenames:
                return [f"s3://{bucket}/{prefix}/{fn}" for fn in filenames]
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                pass
            else:
                raise

        paginator = s3.get_paginator("list_objects_v2")
        uris: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".safetensors"):
                    uris.append(f"s3://{bucket}/{key}")

        if not uris:
            raise FileNotFoundError(f"No .safetensors files found at {s3_uri}")

        return sorted(uris)
