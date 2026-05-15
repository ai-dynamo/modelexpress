# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-allocation-handle VMM arena for ModelExpress.

Reserves one large CUDA VMM VA range (default 16 TiB) up front and bumps
forward through it, where each `malloc(size)` becomes one
`cuMemCreate(size_aligned)` + `cuMemMap` + `cuMemSetAccess` at the
current bump offset. `free(va)` does `cuMemUnmap` + `cuMemRelease` for
that specific allocation. The VA range is registered with NIXL once at
end-of-load via `cuMemGetHandleForAddressRange` + `register_memory`,
collapsing whatever subset of the VA is mapped at registration time
into one MR via dmabuf.

The hot path lives in `_vmm_alloc_ext`: bump arithmetic, lifecycle state,
and the VA -> (size, handle) lookup are C++ atomics plus a small mutex-
protected map. The actual cuMem* calls stay behind the Python
`VmmBackend` protocol so unit tests can use `_StubBackend` without CUDA
and the extension does not link against CUDA.
"""

from __future__ import annotations

import enum
import importlib
import logging
import threading
from typing import Protocol

logger = logging.getLogger(__name__)

try:
    _vmm_alloc_ext = importlib.import_module("modelexpress.vmm._alloc_ext")
except ImportError as e:  # pragma: no cover - covered through vmm_hook fallback tests
    _vmm_alloc_ext = None
    _EXT_IMPORT_ERROR = e
else:
    _EXT_IMPORT_ERROR = None


class _ArenaState(enum.Enum):
    """Lifecycle states for VmmArena.

    OPEN: normal operation, malloc and free allowed.
    CLOSED: close() has been called. Further malloc raises; close() is
        idempotent.
    """

    OPEN = "open"
    CLOSED = "closed"


# Default 16 TiB VA reservation. CUDA virtual addresses are 49 bits on
# Hopper/Ada/Blackwell, giving ~512 TiB of address space per process; a
# 16 TiB reserve costs nothing for VA-only operations because no
# physical memory is committed until cuMemMap. Sized to comfortably
# cover cumulative allocation for any model that fits in HBM, with
# headroom for FP8 / NVFP4 post-processing transforms that allocate
# replacement tensors without freeing the originals through us
# (PyTorch's caching allocator may hold the freed segments).
DEFAULT_VA_RESERVE = 16 * 1024 * 1024 * 1024 * 1024  # 16 TiB

# Skeleton-phase fake base address. Outside any plausible real CUDA VA
# so a leaked stub address is obvious. The real backend returns the
# actual VA from cuMemAddressReserve.
_SKELETON_FAKE_BASE = 0x10_0000_0000_0000  # 16 TiB

# Default backend allocation granularity used by the stub (matches the
# 2 MiB CUDA driver granularity observed on Ada/Blackwell so test
# numbers line up with production expectations).
_DEFAULT_STUB_GRANULARITY = 2 * 1024 * 1024


class VmmArenaError(Exception):
    """Raised on arena allocation or lifecycle errors."""


class VmmBackend(Protocol):
    """Abstracts the CUDA VMM driver calls.

    Production implementation: `CudaVmmBackend` in `vmm_backend.py`.
    Test implementation: `_StubBackend` below.
    """

    def reserve(self, total_bytes: int) -> int:
        """Reserve total_bytes of VA. Returns base address."""
        ...

    def allocate(self, va: int, size: int) -> int:
        """Allocate `size` bytes of physical memory and map at `va`.

        Returns an opaque handle that must be passed back to
        `deallocate` to release the physical memory. `size` is
        guaranteed to be a multiple of `allocation_granularity()`.
        """
        ...

    def deallocate(self, va: int, size: int, handle: int) -> None:
        """Unmap the chunk at `va` and release the physical handle."""
        ...

    def release_reserve(self, base: int, total_bytes: int) -> None:
        """Free the reserved VA range. Called from close() after all
        allocations have been deallocated."""
        ...

    def allocation_granularity(self) -> int:
        """Backend driver granularity (multiple of which sizes are rounded)."""
        ...


class _StubBackend:
    """Test-only backend. Records calls without touching CUDA.

    Tests inject this directly via the `backend=` constructor argument.
    Returns synthetic handles starting at 1. Mutating methods are locked
    so concurrent malloc/free tests remain valid under free-threaded
    Python builds.
    """

    def __init__(self, granularity: int = _DEFAULT_STUB_GRANULARITY) -> None:
        self._granularity = granularity
        self.reserved: list[tuple[int, int]] = []  # (base, total)
        self.allocations: list[tuple[int, int, int]] = []  # (va, size, handle)
        self.deallocations: list[tuple[int, int, int]] = []  # (va, size, handle)
        self.released: list[tuple[int, int]] = []  # (base, total)
        self._next_handle = 1
        self._lock = threading.Lock()

    def reserve(self, total_bytes: int) -> int:
        with self._lock:
            base = _SKELETON_FAKE_BASE + sum(s for _, s in self.reserved)
            self.reserved.append((base, total_bytes))
            return base

    def allocate(self, va: int, size: int) -> int:
        with self._lock:
            handle = self._next_handle
            self._next_handle += 1
            self.allocations.append((va, size, handle))
            return handle

    def deallocate(self, va: int, size: int, handle: int) -> None:
        with self._lock:
            self.deallocations.append((va, size, handle))

    def release_reserve(self, base: int, total_bytes: int) -> None:
        with self._lock:
            self.released.append((base, total_bytes))

    def allocation_granularity(self) -> int:
        return self._granularity


class VmmArena:
    """Per-allocation-handle VMM arena.

    Lifecycle:
        arena = VmmArena()  # 16 TiB VA reserve by default
        va1 = arena.malloc(size1)
        va2 = arena.malloc(size2)
        arena.free(va1)
        ...
        base, used = arena.registered_range()
        # cuMemGetHandleForAddressRange + register_memory(base, used)
        arena.close()

    The per-arena C++ capsule owns bump arithmetic, lifecycle state, and
    live allocation records. Python remains the backend layer for reserve,
    cuMem allocation/deallocation, and VA reserve release.
    """

    def __init__(
        self,
        total_bytes: int = DEFAULT_VA_RESERVE,
        device: int = 0,
        backend: VmmBackend | None = None,
    ) -> None:
        if _vmm_alloc_ext is None:
            raise VmmArenaError(
                "_vmm_alloc_ext C extension is not available; "
                f"import failed: {_EXT_IMPORT_ERROR}"
            )
        if total_bytes <= 0:
            raise ValueError("total_bytes must be positive")
        if backend is None:
            raise ValueError(
                "backend is required: pass CudaVmmBackend(device=...) for "
                "production or _StubBackend() for tests. The previous "
                "implicit-stub default made production callers silently "
                "fall through to fake addresses if the backend kwarg was "
                "forgotten."
            )

        self._device = device
        self._backend: VmmBackend = backend
        self._granularity = self._backend.allocation_granularity()
        if self._granularity <= 0 or (self._granularity & (self._granularity - 1)) != 0:
            raise VmmArenaError(
                f"backend.allocation_granularity() must be a positive power of 2, got {self._granularity}"
            )

        # Round reserve up to granularity for clean bump arithmetic.
        rounded = ((total_bytes + self._granularity - 1) // self._granularity) * self._granularity
        self._total_bytes = rounded
        self._base = self._backend.reserve(rounded)
        try:
            self._capsule = _vmm_alloc_ext.arena_create(
                self._base,
                self._total_bytes,
                self._granularity,
                self._perform_cuda_alloc,
                self._perform_cuda_dealloc,
            )
        except BaseException:
            # arena_create raised after reserve() succeeded; release the
            # VA range so it doesn't leak.
            try:
                self._backend.release_reserve(self._base, self._total_bytes)
            except Exception:
                logger.warning(
                    "failed to roll back reserved VA range for "
                    "base=0x%x total=%d after arena_create failure",
                    self._base,
                    self._total_bytes,
                    exc_info=True,
                )
            raise

        logger.debug(
            "VmmArena reserved base=0x%x total=%d granularity=%d device=%d",
            self._base,
            self._total_bytes,
            self._granularity,
            self._device,
        )

    @property
    def capsule(self):
        """Opaque C extension state used by vmm_hook.set_active_arena."""
        return self._capsule

    @property
    def base(self) -> int:
        return self._base

    @property
    def total_bytes(self) -> int:
        """Total VA reserved. Always a multiple of granularity."""
        return self._total_bytes

    @property
    def device(self) -> int:
        return self._device

    @property
    def granularity(self) -> int:
        """Backend allocation granularity (cuMemCreate size multiple)."""
        return self._granularity

    @property
    def used_bytes(self) -> int:
        """Cumulative bytes allocated (bump pointer). The range
        [base, base + used_bytes) is what gets registered with NIXL."""
        return int(_vmm_alloc_ext.arena_get_used_bytes(self._capsule))

    @property
    def mapped_bytes(self) -> int:
        """Sum of currently-live allocations (after subtracting frees).
        Tracks actual physical VRAM consumed by the arena right now."""
        return int(_vmm_alloc_ext.arena_get_mapped_bytes(self._capsule))

    @property
    def live_allocation_count(self) -> int:
        return int(_vmm_alloc_ext.arena_get_live_count(self._capsule))

    @property
    def closed(self) -> bool:
        return bool(_vmm_alloc_ext.arena_is_closed(self._capsule))

    @property
    def state(self) -> _ArenaState:
        return _ArenaState.CLOSED if self.closed else _ArenaState.OPEN

    def _perform_cuda_alloc(self, va: int, size: int) -> int:
        return self._backend.allocate(int(va), int(size))

    def _perform_cuda_dealloc(self, va: int, size: int, handle: int) -> None:
        self._backend.deallocate(int(va), int(size), int(handle))

    def malloc(self, size: int, alignment: int | None = None) -> int:
        """Allocate `size` bytes from the arena. Returns the VA.

        `size` is rounded up to backend granularity (the minimum
        cuMemCreate size). `alignment` is accepted for API compatibility
        but is implicitly satisfied because every allocation lands at a
        granularity-aligned offset from base.
        """
        if size <= 0:
            raise ValueError("size must be positive")
        if alignment is not None and alignment > self._granularity:
            raise ValueError(
                f"alignment {alignment} exceeds backend granularity {self._granularity}; "
                "PyTorch's caching allocator should round up before reaching us"
            )

        try:
            return int(_vmm_alloc_ext.arena_malloc(self._capsule, int(size)))
        except RuntimeError as e:
            msg = str(e)
            if msg.startswith("arena is closed") or "would exceed reserved range" in msg:
                raise VmmArenaError(msg) from e
            raise

    def free(self, ptr: int) -> None:
        """Free the allocation at `ptr`. No-op if the arena doesn't
        recognize the pointer (PyTorch's caching allocator may
        occasionally free pointers from before our hook was installed).
        """
        _vmm_alloc_ext.arena_free(self._capsule, int(ptr))

    def registered_range(self) -> tuple[int, int]:
        """Return (base, used_bytes) suitable for NIXL register_memory.

        The range covers all VA that has ever been allocated since arena
        init, including holes from subsequent frees.
        """
        base, used = _vmm_alloc_ext.arena_registered_range(self._capsule)
        return (int(base), int(used))

    def close(self) -> None:
        """Deallocate all remaining allocations and free the VA reserve.
        Idempotent.
        """
        drained = _vmm_alloc_ext.arena_close_and_drain(self._capsule)
        if drained is None:
            return

        first_error: Exception | None = None
        for va, size, handle in drained:
            try:
                self._backend.deallocate(int(va), int(size), int(handle))
            except Exception as e:
                logger.warning(
                    "backend.deallocate failed during close for va=0x%x size=%d",
                    int(va),
                    int(size),
                    exc_info=True,
                )
                if first_error is None:
                    first_error = e

        try:
            self._backend.release_reserve(self._base, self._total_bytes)
        except Exception as e:
            logger.warning(
                "backend.release_reserve failed for base=0x%x total=%d",
                self._base,
                self._total_bytes,
                exc_info=True,
            )
            if first_error is None:
                first_error = e

        logger.debug(
            "VmmArena closed base=0x%x total=%d used=%d",
            self._base,
            self._total_bytes,
            self.used_bytes,
        )
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "VmmArena":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
