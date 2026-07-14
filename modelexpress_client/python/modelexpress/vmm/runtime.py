# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-agnostic VMM arena lifecycle helpers.

`maybe_enter_vmm_arena(ctx)` is the integration seam each engine adapter
calls around its weight-load envelope. The arena core
(``modelexpress.vmm.arena``, ``backend``, ``hook``, ``_alloc_ext``) only
requires PyTorch's ``CUDAPluggableAllocator`` interface; this module
adds the env-var handling, per-device arena lifetime tracking, and the
``use_arena`` wrapping that every PyTorch-based engine adapter would
otherwise re-implement. Engine modules call into this; this module does
not call back into engine code.

``log_arena_post_load(ctx)`` is the corresponding diagnostic hook that
runs after the load body returns.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from .. import envs

if TYPE_CHECKING:
    from ..load_strategy.context import LoadContext
    from .arena import VmmArena

logger = logging.getLogger(__name__)


# Active VmmArenas keyed by device_id. Held at module scope so the
# arenas (and their physical CUDA mappings) outlive the load envelope
# and survive for the lifetime of the loaded model. Cleanup happens at
# process exit; see arena.close() handling in tests for explicit
# teardown.
_vmm_arenas: dict[int, "VmmArena"] = {}


@contextmanager
def maybe_enter_vmm_arena(ctx: "LoadContext") -> Iterator[None]:
    """If ``MX_VMM_ARENA=1`` is set, install a VmmArena hook around the
    load envelope. Otherwise yield without installing anything.

    Engine integration seam. The caller is the engine adapter (vLLM,
    SGLang, TRT-LLM, ...) that owns the load lifecycle; this helper
    handles every step that does not depend on engine internals:

    - reads the ``MX_VMM_ARENA`` env var and warns on stale
      ``MX_VMM_ARENA_BYTES`` / ``MX_VMM_ARENA_CHUNK_BYTES``;
    - probes ``ARENA_AVAILABLE`` (whether the optional
      ``modelexpress.vmm._alloc_ext`` C extension built at install time)
      and yields a no-op fallback when it is unavailable;
    - constructs ``CudaVmmBackend`` and ``VmmArena`` inside
      ``ctx.target_device`` so the backend sees the right CUDA context
      on multi-GPU workers;
    - manages the module-level ``_vmm_arenas`` dict keyed by
      ``ctx.device_id`` and publishes the arena after the wrapped body
      completes successfully;
    - wraps ``vmm.use_arena`` around the yielded scope so allocations
      issued by the engine inside the ``with`` land in the arena's
      bump range.

    The fields read off ``ctx`` (``device_id``, ``target_device``,
    ``global_rank``, ``vmm_arena``) are populated by the engine
    adapter, so this function does not need to know which engine
    invoked it.

    Tunable:
        MX_VMM_ARENA=1                 enable. The only knob.

    The arena reserves 16 TiB of VA per device (VA only, no physical
    commit until cuMemMap). Each ``mx_malloc(size)`` becomes one
    ``cuMemCreate(size_aligned)`` + ``cuMemMap(next_va)`` +
    ``cuMemSetAccess`` at the bump pointer; ``mx_free(va)`` does
    ``cuMemUnmap + cuMemRelease`` on the matching allocation. No
    chunked sub-allocation. PyTorch's caching allocator amortizes
    tensor allocations into pool segments before reaching us, so one
    plugin call == one physical handle.

    End-of-load registration goes through
    ``NixlTransferManager.register_arena``, which calls
    ``cuMemGetHandleForAddressRange`` + ``ibv_reg_dmabuf_mr`` over the
    full bump range and produces one MR for the entire arena.
    ``MX_POOL_REG=1`` remains compatible but is no longer required for
    the single-MR property.

    Lifecycle and failure handling:

    - The arena is constructed inside ``ctx.target_device`` so the
      backend sees the right CUDA context on multi-GPU workers.
    - The arena is published to the module-level ``_vmm_arenas`` dict
      only after the wrapped body completes successfully. If the body
      raises, the freshly created arena is closed and not retained.
    - If ``_vmm_arenas`` already has an arena for this device (second
      load on the same worker), the prior arena is closed before
      installing a new one. Silently corrupts the prior model's tensors
      if any are still in use; we log a WARNING. Hot-swap-safe arena
      lifetime tied to engine teardown is a TODO.
    - PyTorch's caching allocator may not return every freed segment
      to us during load. The bump pointer therefore tracks cumulative
      allocation, not peak live size. With 16 TiB reserved, this is
      bounded only by HBM (we never exhaust VA).
    """
    if not envs.MX_VMM_ARENA:
        yield
        return

    if not ctx.accelerator_backend.supports_vmm():
        logger.warning(
            "[Worker %d] MX_VMM_ARENA=1 set but %s does not support VMM "
            "arena; falling back to the non-arena load path.",
            ctx.global_rank,
            ctx.accelerator_backend.name,
        )
        yield
        return

    # The previous chunked-arena design accepted MX_VMM_ARENA_BYTES and
    # MX_VMM_ARENA_CHUNK_BYTES env vars. The current design ignores
    # both (16 TiB VA reserve is unconditional, no chunked
    # sub-allocation). Warn on first entry so an operator who carried
    # the old env vars forward from a pre-refactor manifest sees one
    # clear message rather than silent behavior change.
    for stale_var in ("MX_VMM_ARENA_BYTES", "MX_VMM_ARENA_CHUNK_BYTES"):
        if envs.is_set(stale_var):
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
    from .hook import ARENA_AVAILABLE

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
    # who don't enable the arena. Import from submodules directly so
    # test monkeypatches on `modelexpress.vmm.{backend,hook}.X` are
    # picked up.
    from .arena import VmmArena
    from .backend import CudaVmmBackend
    from .hook import use_arena

    if ctx.device_id in _vmm_arenas:
        # Pre-existing arena from a prior load on the same worker.
        # Replacing it silently corrupts any still-live tensors that
        # point into the old arena's VA range; the typical engine
        # lifecycle only re-enters the load path when the prior model
        # has been torn down, but there is no programmatic guarantee.
        # Log the replacement so an audit can catch a
        # hot-swap-while-serving situation.
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
    # until the engine enters ctx.target_device, so we enter it here
    # just for backend construction. The caller is expected to enter
    # ctx.target_device again around the body; torch device contexts
    # are reentrant so this is fine.
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
    # completes successfully. On exception the arena gets closed and is
    # not retained, so a retry-on-different-strategy or upstream
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


def log_arena_post_load(ctx: "LoadContext") -> None:
    """Log arena state after the load envelope returns.

    The single-MR registration via ``cuMemGetHandleForAddressRange`` +
    ``ibv_reg_dmabuf_mr`` over ``[base, base+used_bytes)`` already ran
    inside ``LoadStrategyChain`` via
    ``NixlTransferManager.register_arena``; this hook is purely
    diagnostic. Empirically validated on Blackwell + ConnectX over
    InfiniBand: the registration succeeds over a VA range with
    mid-range holes from prior ``cuMemUnmap`` calls, and the dmabuf pin
    keeps live tensor pages addressable to the HCA.
    """
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
