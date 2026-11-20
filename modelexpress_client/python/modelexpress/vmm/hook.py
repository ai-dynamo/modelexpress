# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDAPluggableAllocator wiring for the VmmArena.

Glues a VmmArena into PyTorch's allocator machinery so that any CUDA
allocation made inside `use_arena(arena, device)` lands in the arena
rather than going through the default caching allocator.

Sequence:
    arena = VmmArena(total_bytes=N, backend=CudaVmmBackend(device=0))
    with use_arena(arena, device=0):
        # All torch.empty / model init / load_weights /
        # process_weights_after_loading allocations land in the arena.
        ...
    base, mapped = arena.registered_range()
    # nixl.register_memory((base, mapped, ...))   <- one MR for the whole load
    arena.close()
"""

from __future__ import annotations

import importlib
import logging
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    import torch

    from .arena import VmmArena

logger = logging.getLogger(__name__)


def _probe_extension():
    """Check whether the _vmm_alloc_ext C extension is importable.

    The extension is built as optional (see setup.py): a working C++
    compiler at install time is preferred but not required. If the
    .so is absent or fails to load, the arena allocator becomes a
    runtime no-op and callers fall back to the pool-reg path.

    Returns (module_or_None, available_bool, error_or_None).
    """
    try:
        ext = importlib.import_module("modelexpress.vmm._alloc_ext")
        return ext, True, None
    except ImportError as e:
        return None, False, e


_vmm_alloc_ext, ARENA_AVAILABLE, _import_error = _probe_extension()

# Python-side mirror of the active arena, used for nesting checks and
# diagnostics. The C extension owns the active capsule used by mx_malloc /
# mx_free; this variable is not on the allocator hot path.
_active_arena: "VmmArena | None" = None
_active_lock = threading.Lock()

# Initialized lazily inside install_pluggable_allocator. The allocator is
# kept alive at module scope because PyTorch holds C-level references to it
# and we don't want it garbage-collected. The MemPool is NOT kept here: each
# use_arena scope gets its own pool and drops it on exit, see use_arena.
_pluggable_alloc = None  # torch.cuda.CUDAPluggableAllocator
# Pools from scopes that could not release on exit (release_idle=False).
# Kept alive so they are never destroyed inside a caller's own pool scope.
_retained_pools: list = []
_callbacks_initialized = False
# Separate lock for init so concurrent first-time use_arena calls don't
# double-initialize the pluggable allocator. Kept separate
# from _active_lock to avoid holding the active-arena lock across the
# (potentially slow) PyTorch init path.
_init_lock = threading.Lock()


def pool_release_safe_in_pool_scope() -> bool:
    """Whether a MemPool can be destroyed while the caller sits inside
    another allocate-to-pool scope.

    Up to torch 2.12, emptying a pool asserts captures_underway.empty(),
    which counts every active mempool diversion, so destroying our pool
    inside vLLM's sleep-mode pool aborts the process. From torch 2.13 the
    assert is !is_capture_context(), which only checks for a real CUDA
    stream capture.
    """
    import torch

    return torch.__version__ >= (2, 13)


def arena_supported_torch() -> bool:
    """Whether this torch routes cached-segment frees to the owning allocator.

    Before torch 2.8, release_block frees a segment with cudaFree once its
    MemPool is destroyed, so an empty_cache() outside a MemPoolContext
    would cudaFree arena memory that mx_free must unmap. From torch 2.8
    the free is dispatched to the allocator that owns the segment.
    """
    import torch

    return torch.__version__ >= (2, 8)


class ArenaUnavailableError(RuntimeError):
    """Raised when arena machinery is requested but _vmm_alloc_ext failed
    to build or import, or the installed torch is older than 2.8. Catch
    this in callers that want a graceful fallback to the non-arena path."""


def _mx_malloc(size: int, device: int, stream: int) -> int:
    """Test helper mirroring the C extension's mx_malloc dispatch."""
    del device, stream
    arena = _active_arena
    if arena is None:
        logger.error("mx_malloc(%d) called with no active arena", size)
        return 0
    try:
        return arena.malloc(int(size))
    except Exception as e:
        logger.error("arena.malloc(%d) failed: %s", size, e)
        return 0


def _mx_free(ptr: int, size: int, device: int, stream: int) -> None:
    """Test helper mirroring the C extension's mx_free dispatch."""
    del size, device, stream
    arena = _active_arena
    if arena is None:
        logger.warning(
            "mx_free(0x%x) called with no active arena, ignoring", ptr
        )
        return
    try:
        arena.free(int(ptr))
    except Exception as e:
        logger.error("arena.free(0x%x) failed: %s", ptr, e)


def _ensure_callbacks_initialized() -> None:
    """Build the CUDAPluggableAllocator once per process.

    Thread-safe via _init_lock: concurrent first-time callers serialize
    on the lock and the double-check pattern ensures only one PyTorch
    init runs. Subsequent calls fast-path through the flag check
    without acquiring the lock.

    Raises ArenaUnavailableError if the _vmm_alloc_ext C extension is
    not importable or torch is older than 2.8. Callers should catch this
    and fall back to the non-arena path rather than crashing.
    """
    global _pluggable_alloc, _callbacks_initialized
    if _callbacks_initialized:
        return

    with _init_lock:
        # Double-check after lock acquisition: another thread may have
        # raced ahead and finished the init while we were waiting.
        if _callbacks_initialized:
            return

        if not ARENA_AVAILABLE:
            raise ArenaUnavailableError(
                "_vmm_alloc_ext C extension is not available "
                f"(import failed: {_import_error}). The arena allocator "
                "fast path is disabled. Install a C++ compiler and reinstall "
                "modelexpress, or unset MX_VMM_ARENA to use the pool-reg path."
            )

        if not arena_supported_torch():
            import torch

            raise ArenaUnavailableError(
                f"The VMM arena requires torch >= 2.8 (found {torch.__version__}): "
                "on older torch, empty_cache() can free arena memory with "
                "cudaFree instead of the arena's allocator. Upgrade torch, or "
                "unset MX_VMM_ARENA to use the pool-reg path."
            )

        cumem = _vmm_alloc_ext  # set by _probe_extension at import time
        from torch.cuda import CUDAPluggableAllocator

        _pluggable_alloc = CUDAPluggableAllocator(
            cumem.__file__, "mx_malloc", "mx_free"
        )
        _callbacks_initialized = True
        logger.debug("CUDAPluggableAllocator initialized for VmmArena hook")


def install_pluggable_allocator() -> None:
    """Eagerly initialize the pluggable allocator.

    Calling this is optional; use_arena will initialize on first use.
    """
    _ensure_callbacks_initialized()


@contextmanager
def use_arena(
    arena: "VmmArena",
    device: "torch.device | int",
    release_idle: bool = True,
) -> Iterator[None]:
    """Direct subsequent CUDA allocations on `device` into `arena`.

    Exit semantics: on context exit, the scope's MemPool is dropped and
    then the active arena is cleared. The arena itself is NOT closed - call
    arena.close() separately when done with the loaded weights (typically
    after NIXL registration and/or transfer).

    The pool is dropped while the arena is still active on purpose.
    torch.cuda.empty_cache() never releases cached segments of a private
    pool that is still referenced, so a pool kept alive past the load
    would hold every temporary loading buffer as idle arena memory for the
    life of the process. Destroying the pool frees its fully idle segments
    through the pluggable allocator, and mx_free only reaches the arena
    while it is the active one. Segments that still hold live tensors stay
    allocated.

    Pass release_idle=False when the caller itself runs inside another
    allocate-to-pool scope (vLLM sleep mode wraps model loading in its own
    pool) and pool_release_safe_in_pool_scope() is False. Destroying a
    MemPool there trips an allocator assert on those torch versions, so the
    pool is kept alive instead and its idle segments stay mapped, as before.

    Only one arena can be active at a time per process; nested or
    concurrent use_arena calls raise RuntimeError.
    """
    global _active_arena
    import torch

    _ensure_callbacks_initialized()
    from torch.cuda.memory import MemPool

    with _active_lock:
        if _active_arena is not None:
            raise RuntimeError(
                "another VmmArena is already active; use_arena does not "
                "support nesting or concurrent activation"
            )
        # Update the C extension first, then the Python mirror. If the
        # extension call raises (bad capsule, type error), _active_arena
        # stays None and subsequent use_arena() calls aren't permanently
        # wedged with a stale "already active" error.
        _vmm_alloc_ext.set_active_arena(arena.capsule)
        _active_arena = arena

    # Created only once this scope owns the arena: a pool built before the
    # nesting check would be destroyed inside the outer scope's pool.
    pool = None
    try:
        pool = MemPool(allocator=_pluggable_alloc.allocator())
        with torch.cuda.use_mem_pool(pool, device=device):
            yield
    finally:
        # Drop the last reference while the arena is still active, so the
        # pool's idle segments are freed back into this arena (see the
        # docstring). use_mem_pool has already released its own reference.
        if pool is not None and not release_idle:
            _retained_pools.append(pool)
        del pool
        with _active_lock:
            _vmm_alloc_ext.set_active_arena(None)
            _active_arena = None


def is_active() -> bool:
    """Whether use_arena is currently in scope."""
    return _active_arena is not None


def active_arena() -> "VmmArena | None":
    """Return the currently active arena, if any."""
    return _active_arena
