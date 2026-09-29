# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress Custom Model Loader for vLLM.

This loader hooks into vLLM's weight loading pipeline to perform RDMA transfers
of fully-processed model tensors. Registration happens AFTER
process_weights_after_loading() so that all final tensors are captured.
Tensor discovery uses named_parameters() and named_buffers(); bare tensor
attributes created during post-processing (e.g. FP8 scales, MLA projections)
are auto-promoted to non-persistent buffers via capture_tensor_attrs().

Uses LoadStrategyChain to auto-detect the best loading strategy:
    1. RDMA (P2P GPU transfer via NIXL) - if a source is already serving
    2. ServerCache (stream weights from ModelExpress Server) - set MODEL_EXPRESS_NO_SHARED_STORAGE=1
    3. InstantTensor (fast local safetensors, direct I/O + GDS) - set MX_INSTANT_TENSOR=0 to disable
    4. ModelStreamer (S3/GCS/Azure/local via runai-model-streamer) - set MX_MODEL_URI
    5. GDS (GPUDirect Storage) - direct file-to-GPU, bypassing CPU
    6. Default (vLLM DefaultModelLoader) - standard CPU-staged loading

Usage:
    --load-format modelexpress
    --load-format mx  (backward-compatible alias)
"""

from __future__ import annotations

import logging
import threading
import time

import torch
import torch.nn as nn

from ... import configure_vllm_logging, envs, model_prefetch
from ...load_strategy import (
    LoadContext,
    drain_tensor_readers,
    publish_metadata,
    run_load_strategy_chain,
    unpublish_metadata,
)
from ...metrics import enable_metrics, metrics
from ...nixl_transfer import NixlTransferManager
from ...vmm.runtime import log_arena_post_load, maybe_enter_vmm_arena
from .adapter import _is_speculative_draft, build_vllm_load_context
from .artifacts import (
    _vllm_health_ready,
    install_vllm_cache_artifacts,
    schedule_vllm_cache_artifact_publish,
)

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.utils.torch_utils import set_default_torch_dtype

logger = logging.getLogger(__name__)


# Global storage for tensor metadata, keyed by device_id (local CUDA ordinal).
_tensor_registry: dict[int, dict[str, torch.Tensor]] = {}
_nixl_managers: dict[int, NixlTransferManager] = {}
_loader_registry: dict[int, MxModelLoader] = {}
# Main-pass publication gates awaiting this device's speculative draft pass.
_draft_publication_gates: dict[int, _DraftPublicationGate] = {}

# Upper bound on how long a main load's publication waits for its draft pass.
# vLLM loads the drafter right after the target, so this only matters when the
# expected draft never comes through this loader (e.g. a draft-specific load
# format); the target must still publish.
_DRAFT_PUBLICATION_GRACE_SECS = 600.0


class _DraftPublicationGate:
    """Keeps a main load undiscoverable until its draft has joined the manifest.

    Opens when the draft pass on the same device finishes, or after
    ``_DRAFT_PUBLICATION_GRACE_SECS`` from the end of the main pass.
    """

    def __init__(self, grace_secs: float = _DRAFT_PUBLICATION_GRACE_SECS):
        self._released = threading.Event()
        self._grace_secs = grace_secs
        self._deadline: float | None = None
        self._expired_logged = False

    def arm(self) -> None:
        self._deadline = time.monotonic() + self._grace_secs

    def release(self) -> None:
        self._released.set()

    def is_open(self) -> bool:
        if self._released.is_set():
            return True
        if self._deadline is None or time.monotonic() < self._deadline:
            return False
        if not self._expired_logged:
            self._expired_logged = True
            logger.warning(
                "Speculative draft pass did not reach the ModelExpress loader "
                "within %.0fs of the target load; publishing the target without "
                "draft tensors",
                self._grace_secs,
            )
        return True


def _expects_draft_pass(vllm_config) -> bool:
    """True when vLLM will run a second, draft load_model on this worker.

    Mirrors _is_speculative_draft: ngram and similar methods alias the draft
    config to the target's (runner "generate") and never load a draft.
    """
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None:
        return False
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    return getattr(draft_model_config, "runner_type", None) == "draft"


def get_model_loader(device_id: int) -> MxModelLoader | None:
    """Return the ModelExpress loader that completed this device's main load."""
    return _loader_registry.get(device_id)


class MxModelLoader(BaseModelLoader):
    """
    Auto-detecting model loader for ModelExpress.

    Uses LoadStrategyChain to find the best available loading strategy
    (RDMA P2P, GDS, or default disk loading), then registers tensors
    with NIXL and publishes metadata so future nodes can discover this
    one as a source.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        configure_vllm_logging()
        # Unconditionally, and off the load path: a run that skips P2P and falls
        # back to a local or HuggingFace path must still bring the exporter up,
        # or it produces output byte-identical to MX_METRICS_ENABLED=0 -- the run
        # you most need to diagnose. No-op unless enabled; never raises.
        enable_metrics()
        self._ctx: LoadContext | None = None

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        """Load model, auto-detecting the best loading strategy.

        `prefix` is vLLM's BaseModelLoader.load_model argument for initializing
        a model subtree. ModelExpress does not interpret it; it is passed through
        to vLLM's initialize_model().
        """
        load_start = time.perf_counter()

        is_speculative_draft = _is_speculative_draft(vllm_config, model_config)
        if is_speculative_draft and envs.MX_LOAD_STRATEGY_CHAIN == "RL":
            raise ValueError(
                "RL initial loading does not support speculative draft models"
            )

        ctx = build_vllm_load_context(vllm_config, model_config)
        ctx.p2p_role = "draft" if is_speculative_draft else "main"
        draft_gate: _DraftPublicationGate | None = None
        if is_speculative_draft:
            main_loader = _loader_registry.get(ctx.device_id)
            main_ctx = main_loader._ctx if main_loader is not None else None
            if main_ctx is not None and main_ctx.identity == ctx.identity:
                # Same checkpoint (MTP): join the main load's publication
                # through its NIXL agent instead of binding a second one.
                ctx.shared_nixl_manager = _nixl_managers.get(ctx.device_id)
            else:
                # A draft from a different checkpoint (e.g. EAGLE) has its own
                # SourceIdentity, so peers could never find its tensors in the
                # target's publication. Keep it out of P2P.
                ctx.p2p_enabled = False
            draft_gate = _draft_publication_gates.pop(ctx.device_id, None)
        if envs.MX_ARTIFACT_READY_URL.strip():
            # The engine is not healthy until the draft has loaded too, so this
            # already keeps the main publication hidden until then.
            ctx.source_ready_fn = lambda: _vllm_health_ready(ctx)
        elif ctx.p2p_role == "main" and _expects_draft_pass(vllm_config):
            main_gate = _DraftPublicationGate()
            _draft_publication_gates[ctx.device_id] = main_gate
            ctx.source_ready_fn = main_gate.is_open
        if ctx.p2p_role == "main" and ctx.p2p_enabled:
            self._ctx = ctx

        logger.info(
            f"[Worker {ctx.global_rank}] MxModelLoader starting "
            f"(model={ctx.identity.model_name}, p2p_enabled={ctx.p2p_enabled}, "
            f"p2p_role={ctx.p2p_role})"
        )
        try:
            model = self._load_model(
                vllm_config, model_config, prefix, ctx, is_speculative_draft
            )
        finally:
            if draft_gate is not None:
                draft_gate.release()
            main_gate_pending = _draft_publication_gates.get(ctx.device_id)
            if ctx.p2p_role == "main" and main_gate_pending is not None:
                main_gate_pending.arm()

        total_time = time.perf_counter() - load_start
        logger.info(
            f"[Worker {ctx.global_rank}] MxModelLoader.load_model() COMPLETE "
            f"in {total_time:.2f}s"
        )
        return model.eval()

    def _load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str,
        ctx: LoadContext,
        is_speculative_draft: bool,
    ) -> nn.Module:

        # A speculative draft loads through this same path and finishes far
        # sooner than the model the user asked for, so timing them together
        # makes the p99 of neither meaningful. Asked directly rather than
        # inferred from p2p_enabled: that flag happens to agree today, but it is
        # a capability switch and any future reason to clear it would silently
        # relabel real loads as drafts.
        model_role = "draft" if is_speculative_draft else "main"

        # L0 wraps everything below, and the four L1 phases inside it are
        # disjoint, so their sum is bounded by the total by construction. The
        # timers only bracket existing calls; nothing here changes load order.
        model_id = ctx.identity.model_name
        with metrics.time_load("vllm", model_id, model_role):
            with maybe_enter_vmm_arena(ctx):
                if ctx.p2p_enabled and ctx.p2p_role == "main":
                    with metrics.time_load_phase("vllm", model_id, "artifact_install"):
                        install_vllm_cache_artifacts(ctx)
                with set_default_torch_dtype(model_config.dtype):
                    with ctx.target_device:
                        with metrics.time_load_phase("vllm", model_id, "model_init"):
                            model = initialize_model(
                                vllm_config=vllm_config,
                                model_config=model_config,
                                prefix=prefix,
                            )

                    with metrics.time_load_phase("vllm", model_id, "chain"):
                        model = run_load_strategy_chain(model, ctx)

                    if ctx.p2p_enabled and ctx.p2p_role == "draft":
                        if ctx.nixl_manager is not None:
                            # Same checkpoint, same SourceIdentity: extend the
                            # main load's manifest instead of registering a
                            # second source under a colliding mx_source_id.
                            _tensor_registry.setdefault(ctx.device_id, {}).update(
                                ctx.tensors
                            )
                    elif ctx.p2p_enabled:
                        _loader_registry[ctx.device_id] = self
                        _tensor_registry[ctx.device_id] = ctx.tensors
                        if ctx.nixl_manager is not None:
                            _nixl_managers[ctx.device_id] = ctx.nixl_manager
                        else:
                            _nixl_managers.pop(ctx.device_id, None)

                        # Scheduling the publish, not completing it: the work is
                        # handed to a background thread, so this phase measures
                        # the handoff and never the upload.
                        with metrics.time_load_phase("vllm", model_id, "publish"):
                            schedule_vllm_cache_artifact_publish(ctx)

            log_arena_post_load(ctx)

        return model

    def download_model(self, model_config: ModelConfig) -> None:
        """Download the model so it can be loaded immediately."""
        if envs.MX_LOAD_STRATEGY_CHAIN == "RL":
            logger.info(
                "RL initial load is selected; leaving weight acquisition to "
                "the RL strategy chain"
            )
            return

        if model_prefetch.is_enabled():
            # Without shared storage this would pull the full weight set from
            # Hugging Face before any strategy runs, defeating P2P-first and
            # failing outright when the worker is offline. The strategy chain
            # decides where the weights come from.
            logger.info(
                "MODEL_EXPRESS_NO_SHARED_STORAGE is set; leaving weight "
                "acquisition to the ModelExpress strategy chain"
            )
            return

        import copy

        disk_config = copy.copy(self.load_config)
        try:
            disk_config.load_format = "auto"
        except AttributeError:
            object.__setattr__(disk_config, "load_format", "auto")
        DefaultModelLoader(disk_config).download_model(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into an already-initialized model (standalone API)."""
        import copy

        disk_config = copy.copy(self.load_config)
        try:
            disk_config.load_format = "auto"
        except AttributeError:
            object.__setattr__(disk_config, "load_format", "auto")
        DefaultModelLoader(disk_config).load_weights(model, model_config)

    @property
    def nixl_manager(self) -> NixlTransferManager | None:
        """Access the NIXL manager for external use."""
        if self._ctx is not None:
            return self._ctx.nixl_manager
        return None

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        """Access the registered tensor dict."""
        if self._ctx is not None:
            return self._ctx.tensors
        return {}

    @property
    def worker_id(self) -> str | None:
        """Return the inference P2P worker ID after a completed main load."""
        if self._ctx is not None:
            return self._ctx.worker_id
        return None

    def unpublish_runtime_tensors(self) -> None:
        """Withdraw this loader's runtime tensors before an active refit."""
        if self._ctx is not None:
            drain_tensor_readers(
                self._ctx,
                timeout=envs.MX_TRANSFER_TIMEOUT,
            )
            unpublish_metadata(self._ctx)

    def publish_runtime_tensors(self, version_id: str) -> None:
        """Publish this loader's installed runtime tensors at an exact version."""
        if self._ctx is not None:
            self._ctx.identity.revision = version_id
            publish_metadata(self._ctx)
