# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelExpress strategy-chain entrypoint for TensorRT-LLM."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from ... import configure_trtllm_logging
from ...load_strategy import (
    LoadResult,
    LoadStrategyChain,
    publish_metadata,
    register_tensors,
    unpublish_metadata,
)
from ...metrics import enable_metrics, metrics
from .adapter import TrtllmAdapter, build_trtllm_load_context

logger = logging.getLogger("modelexpress.engines.trtllm.loader")


class MxModelLoader:
    """Run TRT-LLM weight loading through the shared MX strategy chain."""

    def __init__(
        self,
        *,
        model_config: Any,
        load_config: Any,
        checkpoint_loader: Any,
        checkpoint_dir: str,
        native_loader_kwargs: dict[str, Any],
        mapping: Any,
        source_identity: Any,
        prepare_post_transform_receiver: Any,
        transform_protocol_version: int,
        p2p_enabled: bool,
        mx_server_url: str | None,
    ) -> None:
        configure_trtllm_logging()
        self._ctx = build_trtllm_load_context(
            model_config=model_config,
            load_config=load_config,
            checkpoint_loader=checkpoint_loader,
            checkpoint_dir=checkpoint_dir,
            native_loader_kwargs=native_loader_kwargs,
            mapping=mapping,
            source_identity=source_identity,
            prepare_post_transform_receiver=prepare_post_transform_receiver,
            transform_protocol_version=transform_protocol_version,
            p2p_enabled=p2p_enabled,
            mx_server_url=mx_server_url,
        )

        # Off the load path, but still before any strategy decision: a run
        # that records nothing must leave the exporter up, so a scrape can
        # prove it came up. No-op unless enabled; never raises.
        enable_metrics()

    @property
    def adapter(self) -> TrtllmAdapter:
        return self._ctx.adapter

    @property
    def model(self) -> torch.nn.Module | None:
        return self.adapter.current_model

    @property
    def p2p_succeeded(self) -> bool:
        return self.adapter.rdma_loaded

    @property
    def transform_protocol_version(self) -> int | None:
        return self.adapter.rdma_transform_protocol_version

    def load_model(self, model: torch.nn.Module) -> Any:
        """Load one TRT-LLM shard and return its native weight-loader value."""
        load_start = time.perf_counter()
        logger.info(
            "[Worker %s] TRT-LLM MxModelLoader starting (model=%s, p2p_enabled=%s)",
            self._ctx.global_rank,
            self._ctx.identity.model_name,
            self._ctx.p2p_enabled,
        )
        self.adapter.current_model = model
        # One phase only. TRT-LLM hands in an initialized model, installs no
        # cache artifacts here, and publishes from publish_model(), which the
        # engine calls after post-load processing -- outside this window.
        # Recording a publish phase there would put time outside the L0 span it
        # is supposed to be a part of, and the phases would stop summing to the
        # whole.
        with metrics.time_load("trtllm", self._ctx.identity.model_name, "main"):
            with metrics.time_load_phase(
                "trtllm", self._ctx.identity.model_name, "chain"
            ):
                value = LoadStrategyChain.run(model, self._ctx)
        total_time = time.perf_counter() - load_start
        logger.info(
            "[Worker %s] TRT-LLM MxModelLoader.load_model() COMPLETE in %.2fs",
            self._ctx.global_rank,
            total_time,
        )
        # TRT-LLM's native loader returns a ConsumableWeightsDict. Preserve
        # that value unchanged on fallback. A successful P2P transfer writes
        # the complete tensor catalog directly into the model, so the engine
        # must not run its mapping pipeline again.
        return {} if self.p2p_succeeded else value

    def publish_model(self, model: torch.nn.Module) -> None:
        """Publish a native-loaded model after TRT post-load processing."""
        try:
            result = LoadResult(value=model, model=model)
            self.adapter.current_model = model
            self._ctx.accelerator_backend.synchronize()
            register_tensors(result, self._ctx)
            publish_metadata(self._ctx)
        except Exception as exc:  # noqa: BLE001 - publish is best effort
            logger.warning(
                "[Worker %s] Failed to publish TRT-LLM model; "
                "worker will continue without P2P serving: %s",
                self._ctx.global_rank,
                exc,
            )

    def cleanup(self) -> None:
        """Release MX metadata, worker-server, NIXL, and client resources."""
        try:
            unpublish_metadata(self._ctx)
        except Exception as exc:  # noqa: BLE001 - cleanup is best effort
            logger.warning("Failed to unpublish TRT-LLM metadata: %s", exc)
        if self._ctx.nixl_manager is not None:
            try:
                self._ctx.nixl_manager.shutdown()
            except Exception as exc:  # noqa: BLE001 - cleanup is best effort
                logger.warning("Failed to shut down TRT-LLM NIXL manager: %s", exc)
            finally:
                self._ctx.nixl_manager = None
                self._ctx.tensors = {}
        close = getattr(self._ctx.mx_client, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - cleanup is best effort
                logger.warning("Failed to close TRT-LLM MX client: %s", exc)
