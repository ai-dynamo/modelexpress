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

    from .vmm_arena import VmmArena

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
        ext = importlib.import_module("modelexpress._vmm_alloc_ext")
        return ext, True, None
    except ImportError as e:
        return None, False, e


_vmm_alloc_ext, ARENA_AVAILABLE, _import_error = _probe_extension()

# Python-side mirror of the active arena, used for nesting checks and
# diagnostics. The C extension owns the active capsule used by mx_malloc /
# mx_free; this variable is not on the allocator hot path.
_active_arena: "VmmArena | None" = None
_active_lock = threading.Lock()

# Initialized lazily inside install_pluggable_allocator. We keep these alive
# at module scope because PyTorch's CUDAPluggableAllocator and MemPool hold
# C-level references to the underlying allocator and we don't want them
# garbage-collected.
_pluggable_alloc = None  # torch.cuda.CUDAPluggableAllocator
_mem_pool = None  # torch.cuda.memory.MemPool
_callbacks_initialized = False
# Separate lock for init so concurrent first-time use_arena calls don't
# double-initialize the pluggable allocator and mempool. Kept separate
# from _active_lock to avoid holding the active-arena lock across the
# (potentially slow) PyTorch init path.
_init_lock = threading.Lock()


class ArenaUnavailableError(RuntimeError):
    """Raised when arena machinery is requested but _vmm_alloc_ext failed
    to build or import. Catch this in callers that want a graceful
    fallback to the non-arena path."""


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
    """Build the CUDAPluggableAllocator and MemPool once per process.

    Thread-safe via _init_lock: concurrent first-time callers serialize
    on the lock and the double-check pattern ensures only one PyTorch
    init runs. Subsequent calls fast-path through the flag check
    without acquiring the lock.

    Raises ArenaUnavailableError if the _vmm_alloc_ext C extension is
    not importable. Callers should catch this and fall back to the
    non-arena path rather than crashing.
    """
    global _pluggable_alloc, _mem_pool, _callbacks_initialized
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

        cumem = _vmm_alloc_ext  # set by _probe_extension at import time
        from torch.cuda import CUDAPluggableAllocator
        from torch.cuda.memory import MemPool

        _pluggable_alloc = CUDAPluggableAllocator(
            cumem.__file__, "mx_malloc", "mx_free"
        )
        _mem_pool = MemPool(allocator=_pluggable_alloc.allocator())
        _callbacks_initialized = True
        logger.debug(
            "CUDAPluggableAllocator + MemPool initialized for VmmArena hook"
        )


def install_pluggable_allocator() -> None:
    """Eagerly initialize the pluggable allocator and mempool.

    Calling this is optional; use_arena will initialize on first use.
    """
    _ensure_callbacks_initialized()


@contextmanager
def use_arena(arena: "VmmArena", device: "torch.device | int") -> Iterator[None]:
    """Direct subsequent CUDA allocations on `device` into `arena`.

    Exit semantics: on context exit, the active arena is cleared. The
    arena itself is NOT closed - call arena.close() separately when done
    with the loaded weights (typically after NIXL registration and/or
    transfer).

    Only one arena can be active at a time per process; nested or
    concurrent use_arena calls raise RuntimeError.
    """
    global _active_arena
    import torch

    _ensure_callbacks_initialized()

    with _active_lock:
        if _active_arena is not None:
            raise RuntimeError(
                "another VmmArena is already active; use_arena does not "
                "support nesting or concurrent activation"
            )
        _active_arena = arena
        _vmm_alloc_ext.set_active_arena(arena.capsule)

    try:
        with torch.cuda.use_mem_pool(_mem_pool, device=device):
            yield
    finally:
        with _active_lock:
            _vmm_alloc_ext.set_active_arena(None)
            _active_arena = None


def is_active() -> bool:
    """Whether use_arena is currently in scope."""
    return _active_arena is not None


def active_arena() -> "VmmArena | None":
    """Return the currently active arena, if any."""
    return _active_arena
