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
    2. ModelStreamer (S3/GCS/Azure/local via runai-model-streamer) - set MX_MODEL_URI
    3. GDS (GPUDirect Storage) - direct file-to-GPU, bypassing CPU
    4. Default (vLLM DefaultModelLoader) - standard CPU-staged loading

Usage:
    --load-format mx  (auto-detect: RDMA -> ModelStreamer -> GDS -> default)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager, nullcontext
from typing import ContextManager, Iterator

import torch
import torch.nn as nn

from ... import configure_vllm_logging
from ...load_strategy import LoadContext, LoadStrategyChain
from ...nixl_transfer import NixlTransferManager
from .adapter import build_vllm_load_context

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.utils.torch_utils import set_default_torch_dtype

logger = logging.getLogger("modelexpress.engines.vllm.loader")


def _patch_vllm_s3_format_check() -> None:
    """Allow 'mx' as a valid load format when model weights use object storage.

    vLLM's verification path only allows object-storage `model_weights` for its
    native RunAI load formats. The MX loader still delegates the ModelStreamer
    path back to vLLM after strategy selection, so verification needs to accept
    the temporary `load_format == "mx"` configuration.
    """
    try:
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
    logger.debug(
        "Patched VllmConfig.try_verify_and_update_config to allow 'mx' "
        "for object storage URIs"
    )


# Global storage for tensor metadata, keyed by device_id (local CUDA ordinal).
_tensor_registry: dict[int, dict[str, torch.Tensor]] = {}
_nixl_managers: dict[int, NixlTransferManager] = {}

# Active VmmArenas keyed by device_id. Held at module scope so the arenas
# (and their physical CUDA mappings) outlive load_model and survive for the
# lifetime of the loaded model. Cleanup happens at process exit; see
# arena.close() handling in tests for explicit teardown.
_vmm_arenas: dict[int, "object"] = {}


# Install eagerly: vllm calls try_verify_and_update_config during engine init,
# before LoadStrategyChain.run() would lazily import the strategy module.
_patch_vllm_s3_format_check()


@contextmanager
def _maybe_enter_vmm_arena(ctx: LoadContext) -> Iterator[None]:
    """If MX_VMM_ARENA=1 is set, install a VmmArena hook around the load
    envelope. Otherwise yield without installing anything.

    Engine integration seam. The arena core (`modelexpress.vmm`
    subpackage: `arena.py`, `backend.py`, `hook.py`, `_alloc_ext.cpp`)
    is engine-agnostic - it only requires PyTorch's
    CUDAPluggableAllocator interface. This function is the only
    vLLM-aware piece: it reads the MX_VMM_ARENA* env vars, identifies
    vLLM's load envelope and target device, manages per-device arena
    lifetime, and wraps `vmm.use_arena` around the load body. To add a
    different PyTorch-based engine, write an equivalent helper in a
    sibling `engines/<engine>/` module that knows the engine's load
    envelope and device-context shape; reuse `use_arena` as-is.
    Non-PyTorch engines would need a different dispatch shim - the
    bump allocator and cuMem* backend would still be reusable, but the
    CUDAPluggableAllocator hook layer would not apply.

    Tunable:
        MX_VMM_ARENA=1                 enable. The only knob.

    The arena reserves 16 TiB of VA per device (VA only, no physical
    commit until cuMemMap). Each `mx_malloc(size)` becomes one
    `cuMemCreate(size_aligned)` + `cuMemMap(next_va)` + `cuMemSetAccess`
    at the bump pointer; `mx_free(va)` does `cuMemUnmap + cuMemRelease`
    on the matching allocation. No chunked sub-allocation. PyTorch's
    caching allocator amortizes tensor allocations into pool segments
    before reaching us, so one plugin call = one physical handle.

    MX_VMM_ARENA=1 is the only knob. End-of-load registration goes
    through `NixlTransferManager.register_arena` (see nixl_transfer.py),
    which calls `cuMemGetHandleForAddressRange` + `ibv_reg_dmabuf_mr`
    over the full bump range and produces one MR for the entire arena.
    `MX_POOL_REG=1` remains compatible but is no longer required for
    the single-MR property.

    Lifecycle and failure handling:
    - The arena is constructed inside `ctx.target_device` so the backend
      sees the right CUDA context on multi-GPU workers.
    - The arena is published to the module-level `_vmm_arenas` dict only
      after the wrapped load body completes successfully. If
      initialize_model or LoadStrategyChain.run raises, the freshly
      created arena is closed and not retained.
    - If `_vmm_arenas` already has an arena for this device (second load
      on the same worker), the prior arena is closed before installing a
      new one. Silently corrupts the prior model's tensors if any are
      still in use; we log a WARNING. Hot-swap-safe arena lifetime tied
      to engine teardown is a TODO.
    - PyTorch's caching allocator may not return every freed segment to
      us during load. The bump pointer therefore tracks cumulative
      allocation, not peak live size. With 16 TiB reserved, this is
      bounded only by HBM (we never exhaust VA).
    """
    if os.environ.get("MX_VMM_ARENA") != "1":
        yield
        return

    # The previous chunked-arena design accepted MX_VMM_ARENA_BYTES and
    # MX_VMM_ARENA_CHUNK_BYTES env vars. The current design ignores both
    # (16 TiB VA reserve is unconditional, no chunked sub-allocation).
    # Warn on first entry so an operator who carried the old env vars
    # forward from a pre-refactor manifest sees one clear message rather
    # than silent behavior change.
    for stale_var in ("MX_VMM_ARENA_BYTES", "MX_VMM_ARENA_CHUNK_BYTES"):
        if os.environ.get(stale_var):
            logger.warning(
                "[Worker %d] %s is set but no longer honored; the new VMM "
                "arena reserves 16 TiB of VA unconditionally and uses one "
                "cuMemCreate per allocation. Drop the env var from your "
                "manifest to silence this warning.",
                ctx.global_rank,
                stale_var,
            )

    # The modelexpress.vmm._alloc_ext C extension is built optional (see
    # setup.py). If a working compiler wasn't available at install time,
    # the .so is absent and the arena machinery cannot be installed.
    # Fall back to the non-arena path with a clear warning rather than
    # crashing the load.
    from ...vmm.hook import ARENA_AVAILABLE

    if not ARENA_AVAILABLE:
        logger.warning(
            "[Worker %d] MX_VMM_ARENA=1 set but the modelexpress.vmm._alloc_ext "
            "C extension is unavailable; falling back to the non-arena load "
            "path. Pool-reg (MX_POOL_REG=1) still works. Reinstall "
            "modelexpress with a working C++ compiler to enable the arena "
            "fast path.",
            ctx.global_rank,
        )
        yield
        return

    # Lazy imports - keep the cuda-python dependency optional for users
    # who don't enable the arena. Import from submodules directly so test
    # monkeypatches on `modelexpress.vmm.{backend,hook}.X` are picked up.
    from ...vmm.arena import VmmArena
    from ...vmm.backend import CudaVmmBackend
    from ...vmm.hook import use_arena

    if ctx.device_id in _vmm_arenas:
        # Pre-existing arena from a prior load_model on the same worker.
        # Replacing it silently corrupts any still-live tensors that
        # point into the old arena's VA range; vLLM's typical lifecycle
        # only re-enters load_model when the prior model has been torn
        # down, but there's no programmatic guarantee. Log the
        # replacement so an audit can catch a hot-swap-while-serving
        # situation.
        logger.warning(
            "[Worker %d] Replacing existing VmmArena on device %d. Any "
            "tensors still backed by the prior arena's VA range will see "
            "corrupted memory once its close() releases the per-allocation "
            "handles. This is safe only if the prior model has been fully "
            "torn down. TODO: tie arena lifetime to engine teardown.",
            ctx.global_rank,
            ctx.device_id,
        )
        old = _vmm_arenas.pop(ctx.device_id)
        try:
            old.close()
        except Exception as e:
            logger.warning(
                "[Worker %d] failed to close prior VmmArena: %s",
                ctx.global_rank,
                e,
            )

    # CudaVmmBackend requires a CUDA context on the calling thread. On
    # multi-GPU workers the current device may not match ctx.device_id
    # until vLLM enters ctx.target_device, so we enter it here just for
    # backend construction. The caller is expected to enter
    # ctx.target_device again around initialize_model; torch device
    # contexts are reentrant so this is fine.
    with ctx.target_device:
        backend = CudaVmmBackend(device=ctx.device_id)

    arena = VmmArena(backend=backend, device=ctx.device_id)
    logger.info(
        "[Worker %d] VmmArena enabled: base=0x%x reserved=%d granularity=%d",
        ctx.global_rank,
        arena.base,
        arena.total_bytes,
        arena.granularity,
    )

    # Only publish the arena into the module-level dict AFTER the body
    # completes successfully. On exception the arena gets closed and
    # is not retained, so a retry-on-different-strategy or upstream
    # error-handling starts from a clean state.
    #
    # Also stash the arena on ctx.vmm_arena so the strategy chain's
    # register_tensors can pass it to NixlTransferManager.register_arena
    # for single-MR-via-dmabuf registration over the full bump range.
    ctx.vmm_arena = arena
    published = False
    try:
        with use_arena(arena, device=ctx.device_id):
            yield
        _vmm_arenas[ctx.device_id] = arena
        published = True
    finally:
        if not published:
            ctx.vmm_arena = None
            try:
                arena.close()
            except Exception as e:
                logger.warning(
                    "[Worker %d] failed to close VmmArena after load error: %s",
                    ctx.global_rank,
                    e,
                )


def _log_arena_post_load(ctx: LoadContext) -> None:
    """Log arena state after load completes. The single-MR registration
    via `cuMemGetHandleForAddressRange` + `ibv_reg_dmabuf_mr` over
    `[base, base+used_bytes)` already ran inside `LoadStrategyChain` via
    `NixlTransferManager.register_arena`; this hook is purely
    diagnostic. Empirically validated on Blackwell + ConnectX over
    InfiniBand: the registration succeeds over a VA range with
    mid-range holes from prior `cuMemUnmaps`, and the dmabuf pin keeps
    live tensor pages addressable to the HCA."""
    arena = _vmm_arenas.get(ctx.device_id)
    if arena is None:
        return
    base, used = arena.registered_range()
    logger.info(
        "[Worker %d] VmmArena post-load: base=0x%x used=%d live_allocs=%d mapped=%d",
        ctx.global_rank,
        base,
        used,
        arena.live_allocation_count,
        arena.mapped_bytes,
    )


@register_model_loader("mx")
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
        self._ctx: LoadContext | None = None

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        """Load model, auto-detecting the best loading strategy."""
        load_start = time.perf_counter()

        ctx = build_vllm_load_context(vllm_config, model_config)
        self._ctx = ctx

        logger.info(f"[Worker {ctx.global_rank}] MxModelLoader starting (model={ctx.identity.model_name})")

        with _maybe_enter_vmm_arena(ctx):
            with set_default_torch_dtype(model_config.dtype):
                with ctx.target_device:
                    model = initialize_model(
                        vllm_config=vllm_config, model_config=model_config
                    )

                model = LoadStrategyChain.run(model, ctx)

                # Update global registries
                _tensor_registry[ctx.device_id] = ctx.tensors
                if ctx.nixl_manager is not None:
                    _nixl_managers[ctx.device_id] = ctx.nixl_manager
                else:
                    _nixl_managers.pop(ctx.device_id, None)

        _log_arena_post_load(ctx)

        total_time = time.perf_counter() - load_start
        logger.info(
            f"[Worker {ctx.global_rank}] MxModelLoader.load_model() COMPLETE "
            f"in {total_time:.2f}s"
        )
        return model.eval()

    def download_model(self, model_config: ModelConfig) -> None:
        """Download the model so it can be loaded immediately."""
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
