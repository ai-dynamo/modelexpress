# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-allocation-handle VmmArena.

These tests exercise the allocator math without touching CUDA, via the
`_StubBackend` that records reserve/allocate/deallocate/release_reserve
calls without making them. A separate CUDA-backed test module
(test_vmm_backend.py) covers the real cuMem* calls when CUDA is
available.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from modelexpress.vmm.arena import (
    DEFAULT_VA_RESERVE,
    VmmArena,
    VmmArenaError,
    _ArenaState,
    _StubBackend,
)


# Stub backend default granularity (2 MiB, matching CUDA driver
# granularity observed on Ada/Blackwell).
_GRAN = 2 * 1024 * 1024


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic(self):
        arena = VmmArena(backend=_StubBackend())
        assert arena.total_bytes == DEFAULT_VA_RESERVE
        assert arena.used_bytes == 0
        assert arena.mapped_bytes == 0
        assert arena.live_allocation_count == 0
        assert arena.granularity == _GRAN
        assert not arena.closed
        assert arena.state == _ArenaState.OPEN

    def test_default_reserve_is_16_tib(self):
        assert DEFAULT_VA_RESERVE == 16 * 1024 * 1024 * 1024 * 1024

    def test_explicit_total_bytes(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        assert arena.total_bytes == 1 << 30

    def test_rounds_total_up_to_granularity(self):
        # 300 MiB is not a multiple of 2 MiB; rounds up to 300 MiB
        # exactly (300 MiB IS a multiple of 2 MiB). Use a non-multiple
        # value to actually trigger rounding.
        arena = VmmArena(total_bytes=300 * 1024 * 1024 + 1, backend=_StubBackend())
        assert arena.total_bytes == 300 * 1024 * 1024 + _GRAN

    def test_exact_granularity_multiple_not_padded(self):
        arena = VmmArena(total_bytes=512 * 1024 * 1024, backend=_StubBackend())
        assert arena.total_bytes == 512 * 1024 * 1024

    def test_rejects_zero_total(self):
        with pytest.raises(ValueError):
            VmmArena(total_bytes=0, backend=_StubBackend())

    def test_rejects_negative_total(self):
        with pytest.raises(ValueError):
            VmmArena(total_bytes=-1, backend=_StubBackend())


# ---------------------------------------------------------------------------
# Backend interactions at construction time
# ---------------------------------------------------------------------------


class TestBackendOnConstruction:
    def test_reserve_called_once(self):
        backend = _StubBackend()
        arena = VmmArena(total_bytes=1 << 30, backend=backend)
        assert len(backend.reserved) == 1
        base, size = backend.reserved[0]
        assert base == arena.base
        assert size == arena.total_bytes

    def test_no_allocations_yet(self):
        backend = _StubBackend()
        VmmArena(total_bytes=1 << 30, backend=backend)
        assert len(backend.allocations) == 0


# ---------------------------------------------------------------------------
# malloc behavior
# ---------------------------------------------------------------------------


class TestMalloc:
    def test_returns_granularity_aligned_addresses(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        a = arena.malloc(1024)
        b = arena.malloc(1024)
        # Every malloc rounds size up to granularity, so a and b are
        # both granularity-aligned offsets from base.
        assert (a - arena.base) % _GRAN == 0
        assert (b - arena.base) % _GRAN == 0
        assert b > a
        assert b - a >= _GRAN  # smallest possible step is one granularity unit

    def test_first_malloc_at_base(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        a = arena.malloc(1024)
        assert a == arena.base

    def test_size_rounds_up_to_granularity(self):
        backend = _StubBackend()
        arena = VmmArena(total_bytes=1 << 30, backend=backend)
        arena.malloc(1024)
        # Backend sees the granularity-rounded size, not the raw request.
        _va, size, _handle = backend.allocations[0]
        assert size == _GRAN

    def test_alignment_compatible_passes_through(self):
        # alignment <= granularity is acceptable; the bump pointer is
        # already granularity-aligned so any smaller alignment is
        # implicitly satisfied.
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        arena.malloc(1, alignment=256)
        arena.malloc(1, alignment=4096)  # 4 KiB < 2 MiB granularity

    def test_alignment_exceeding_granularity_rejected(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        with pytest.raises(ValueError, match="exceeds backend granularity"):
            arena.malloc(1, alignment=_GRAN * 2)

    def test_tracks_live_allocation_count(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        arena.malloc(100)
        arena.malloc(200)
        assert arena.live_allocation_count == 2

    def test_each_malloc_is_one_backend_call(self):
        backend = _StubBackend()
        arena = VmmArena(total_bytes=1 << 30, backend=backend)
        for _ in range(10):
            arena.malloc(1024)
        # No internal sub-allocation: 10 plugin calls = 10 backend
        # allocations.
        assert len(backend.allocations) == 10

    def test_used_bytes_advances_by_aligned_size(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        arena.malloc(100)
        assert arena.used_bytes == _GRAN
        arena.malloc(200)
        assert arena.used_bytes == 2 * _GRAN

    def test_overflow_raises(self):
        arena = VmmArena(total_bytes=4 * _GRAN, backend=_StubBackend())
        arena.malloc(_GRAN)
        arena.malloc(_GRAN)
        arena.malloc(_GRAN)
        arena.malloc(_GRAN)
        with pytest.raises(VmmArenaError, match="exceed"):
            arena.malloc(1)

    def test_overflow_at_boundary(self):
        arena = VmmArena(total_bytes=_GRAN, backend=_StubBackend())
        arena.malloc(_GRAN)
        with pytest.raises(VmmArenaError):
            arena.malloc(1)

    def test_after_close_raises(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        arena.close()
        with pytest.raises(VmmArenaError, match="closed"):
            arena.malloc(100)

    def test_zero_size_rejected(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        with pytest.raises(ValueError):
            arena.malloc(0)

    def test_negative_size_rejected(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        with pytest.raises(ValueError):
            arena.malloc(-1)

    def test_addresses_are_unique(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        addrs = [arena.malloc(1024) for _ in range(100)]
        assert len(set(addrs)) == 100


# ---------------------------------------------------------------------------
# free behavior (load-bearing now, not a no-op)
# ---------------------------------------------------------------------------


class TestFree:
    def test_free_calls_backend_deallocate(self):
        backend = _StubBackend()
        arena = VmmArena(total_bytes=1 << 30, backend=backend)
        va = arena.malloc(1024)
        arena.free(va)
        assert len(backend.deallocations) == 1
        d_va, _d_size, _d_handle = backend.deallocations[0]
        assert d_va == va

    def test_free_reduces_mapped_bytes(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        va1 = arena.malloc(1024)
        va2 = arena.malloc(1024)
        assert arena.mapped_bytes == 2 * _GRAN
        arena.free(va1)
        assert arena.mapped_bytes == _GRAN
        arena.free(va2)
        assert arena.mapped_bytes == 0

    def test_free_does_not_decrease_used_bytes(self):
        """Bump pointer is high-water; free reclaims VRAM but VA is not reused."""
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        va = arena.malloc(1024)
        used_after_alloc = arena.used_bytes
        arena.free(va)
        assert arena.used_bytes == used_after_alloc

    def test_free_unknown_ptr_is_silent(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        # PyTorch's caching allocator may free pointers we don't
        # recognize if our plugin was installed mid-stream. Must not
        # raise.
        arena.free(0xDEADBEEF)
        arena.free(arena.base)  # never malloc'd this either

    def test_double_free_is_silent(self):
        backend = _StubBackend()
        arena = VmmArena(total_bytes=1 << 30, backend=backend)
        va = arena.malloc(1024)
        arena.free(va)
        arena.free(va)
        # Only one deallocate call; second pop returns None and no-ops.
        assert len(backend.deallocations) == 1

    def test_free_then_realloc_does_not_reuse_va(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        va1 = arena.malloc(1024)
        arena.free(va1)
        va2 = arena.malloc(1024)
        assert va2 > va1  # bump moved past the freed slot


# ---------------------------------------------------------------------------
# Registration interface
# ---------------------------------------------------------------------------


class TestRegisteredRange:
    def test_returns_base_and_used_bytes(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        arena.malloc(1024)
        base, used = arena.registered_range()
        assert base == arena.base
        assert used == _GRAN

    def test_empty_arena_zero_used(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        base, used = arena.registered_range()
        assert base == arena.base
        assert used == 0

    def test_grows_with_each_allocation(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        arena.malloc(1024)
        _, u1 = arena.registered_range()
        arena.malloc(1024)
        _, u2 = arena.registered_range()
        assert u2 == u1 + _GRAN

    def test_does_not_shrink_on_free(self):
        """The registered range covers all VA ever allocated, including
        holes from frees. End-of-load registration via dmabuf pins the
        currently-mapped subset."""
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        va1 = arena.malloc(1024)
        arena.malloc(1024)
        _, u_before = arena.registered_range()
        arena.free(va1)
        _, u_after = arena.registered_range()
        assert u_after == u_before


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_close_idempotent(self):
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        arena.close()
        arena.close()  # must not raise
        assert arena.closed

    def test_close_releases_reserve(self):
        backend = _StubBackend()
        arena = VmmArena(total_bytes=1 << 30, backend=backend)
        arena.close()
        assert len(backend.released) == 1
        base, size = backend.released[0]
        assert base == arena.base
        assert size == arena.total_bytes

    def test_close_deallocates_live_allocations(self):
        backend = _StubBackend()
        arena = VmmArena(total_bytes=1 << 30, backend=backend)
        arena.malloc(1024)
        arena.malloc(1024)
        arena.close()
        # Both allocations released by close.
        assert len(backend.deallocations) == 2

    def test_close_does_not_double_deallocate_already_freed(self):
        backend = _StubBackend()
        arena = VmmArena(total_bytes=1 << 30, backend=backend)
        va = arena.malloc(1024)
        arena.free(va)
        arena.close()
        # One deallocate from free, none from close (allocation gone).
        assert len(backend.deallocations) == 1

    def test_context_manager(self):
        backend = _StubBackend()
        with VmmArena(total_bytes=1 << 30, backend=backend) as arena:
            arena.malloc(1024)
            assert not arena.closed
        assert arena.closed
        assert len(backend.released) == 1

    def test_context_manager_closes_on_exception(self):
        backend = _StubBackend()
        with pytest.raises(RuntimeError):
            with VmmArena(total_bytes=1 << 30, backend=backend) as arena:
                arena.malloc(1024)
                raise RuntimeError("synthetic")
        assert arena.closed
        assert len(backend.released) == 1


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class _FailingAllocateBackend(_StubBackend):
    """Stub backend whose allocate raises on the n-th call."""

    def __init__(self, fail_on_call: int = 1):
        super().__init__()
        self._fail_on_call = fail_on_call
        self._call_count = 0

    def allocate(self, va: int, size: int) -> int:
        self._call_count += 1
        if self._call_count == self._fail_on_call:
            raise RuntimeError(f"synthetic allocate failure on call {self._call_count}")
        return super().allocate(va, size)


class _FailingReleaseBackend(_StubBackend):
    """Stub backend whose release_reserve raises."""

    def release_reserve(self, base: int, total_bytes: int) -> None:
        super().release_reserve(base, total_bytes)
        raise RuntimeError("synthetic release_reserve failure")


class _FailingDeallocateBackend(_StubBackend):
    """Stub backend whose deallocate raises."""

    def deallocate(self, va: int, size: int, handle: int) -> None:
        super().deallocate(va, size, handle)
        raise RuntimeError("synthetic deallocate failure")


class TestFailureHandling:
    def test_malloc_failure_leaks_bump_slot_only(self):
        """If backend.allocate raises, the C hot path intentionally leaves
        the reserved VA slot behind but records no live allocation. This
        keeps rollback out of the atomic bump path; the 16 TiB reserve makes
        failed-allocation VA exhaustion unrealistic."""
        backend = _FailingAllocateBackend(fail_on_call=1)
        arena = VmmArena(total_bytes=1 << 30, backend=backend)
        with pytest.raises(RuntimeError, match="synthetic allocate failure"):
            arena.malloc(1024)
        assert arena.used_bytes == _GRAN
        assert arena.mapped_bytes == 0
        assert arena.live_allocation_count == 0

    def test_malloc_after_failure_still_works(self):
        """Per-allocation failures do not poison the arena. Subsequent
        malloc proceeds normally after the leaked VA slot."""
        backend = _FailingAllocateBackend(fail_on_call=1)
        arena = VmmArena(total_bytes=1 << 30, backend=backend)
        with pytest.raises(RuntimeError):
            arena.malloc(1024)
        # Next call (call 2) succeeds after the intentionally leaked slot.
        va = arena.malloc(1024)
        assert va == arena.base + _GRAN

    @pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnraisableExceptionWarning"
    )
    def test_deallocate_failure_logged_not_raised(self):
        """A backend deallocate error during free does not propagate to
        the caller, but it is no longer silently swallowed: the error
        surfaces via sys.unraisablehook (which pytest captures as an
        unraisable-exception warning) instead of PyErr_Clear()."""
        backend = _FailingDeallocateBackend()
        arena = VmmArena(total_bytes=1 << 30, backend=backend)
        va = arena.malloc(1024)
        arena.free(va)  # must not raise
        # The deallocate was attempted.
        assert len(backend.deallocations) == 1
        # And the allocation tracking was dropped.
        assert arena.live_allocation_count == 0

    def test_close_sets_state_even_when_release_raises(self):
        """release_reserve raising must not leave state in OPEN.
        close() re-raises the cleanup error after state transition."""
        backend = _FailingReleaseBackend()
        arena = VmmArena(total_bytes=1 << 30, backend=backend)
        with pytest.raises(RuntimeError, match="synthetic release_reserve"):
            arena.close()
        assert arena.state == _ArenaState.CLOSED
        # Second close is a no-op.
        arena.close()


# ---------------------------------------------------------------------------
# Realistic load-shaped workload (smoke test)
# ---------------------------------------------------------------------------


class TestRealisticWorkload:
    def test_many_allocations(self):
        """Simulate ~200 tensor-shaped allocations like a real load."""
        backend = _StubBackend()
        arena = VmmArena(total_bytes=2 * 1024 * 1024 * 1024, backend=backend)
        addrs = [arena.malloc(64 * 1024) for _ in range(200)]
        assert len(addrs) == 200
        assert addrs == sorted(set(addrs))  # unique, ascending
        # Each malloc -> one backend allocate call.
        assert len(backend.allocations) == 200

    def test_alloc_free_alloc_pattern(self):
        """Simulate the FP8 post-processing pattern: allocate BF16,
        free BF16, allocate FP8 replacement (smaller). mapped_bytes
        tracks the live state; used_bytes is the cumulative high-water."""
        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        # Pretend BF16 tensor.
        bf16 = arena.malloc(8 * 1024 * 1024)
        assert arena.mapped_bytes == 8 * 1024 * 1024
        # Free BF16 (process_weights_after_loading replaced it).
        arena.free(bf16)
        assert arena.mapped_bytes == 0
        # Allocate FP8 replacement (half the size).
        fp8 = arena.malloc(4 * 1024 * 1024)
        assert arena.mapped_bytes == 4 * 1024 * 1024
        # used_bytes covers both the (now-freed) BF16 and the FP8.
        assert arena.used_bytes == 12 * 1024 * 1024
        assert fp8 > bf16

    def test_close_releases_live_allocations(self):
        backend = _StubBackend()
        arena = VmmArena(total_bytes=1 << 30, backend=backend)
        for _ in range(50):
            arena.malloc(1024)
        # Free half of them.
        for va, _, _ in list(backend.allocations[:25]):
            arena.free(va)
        assert arena.live_allocation_count == 25
        arena.close()
        # close deallocated the remaining 25; 25 + 25 = 50 total deallocates.
        assert len(backend.deallocations) == 50


# ---------------------------------------------------------------------------
# Free-threaded hot-path regression coverage
# ---------------------------------------------------------------------------


class TestConcurrentHotPath:
    def test_concurrent_mallocs_are_unique(self):
        backend = _StubBackend()
        arena = VmmArena(total_bytes=512 * _GRAN, backend=backend)

        def alloc_one(i: int) -> int:
            return arena.malloc(1024 + i)

        with ThreadPoolExecutor(max_workers=16) as executor:
            addrs = list(executor.map(alloc_one, range(200)))

        assert len(addrs) == 200
        assert len(set(addrs)) == 200
        assert min(addrs) == arena.base
        assert max(addrs) < arena.base + arena.used_bytes
        assert arena.used_bytes == 200 * _GRAN
        assert arena.mapped_bytes == 200 * _GRAN
        assert arena.live_allocation_count == 200
        assert len(backend.allocations) == 200

    def test_concurrent_free_reduces_live_state_once(self):
        backend = _StubBackend()
        arena = VmmArena(total_bytes=512 * _GRAN, backend=backend)
        addrs = [arena.malloc(1024) for _ in range(200)]

        with ThreadPoolExecutor(max_workers=16) as executor:
            list(executor.map(arena.free, addrs + addrs))

        assert arena.used_bytes == 200 * _GRAN
        assert arena.mapped_bytes == 0
        assert arena.live_allocation_count == 0
        assert len(backend.deallocations) == 200

    def test_multiple_arenas_keep_independent_c_state(self):
        backend1 = _StubBackend()
        backend2 = _StubBackend()
        arena1 = VmmArena(total_bytes=16 * _GRAN, backend=backend1)
        arena2 = VmmArena(total_bytes=16 * _GRAN, backend=backend2)

        a1 = arena1.malloc(1024)
        a2 = arena2.malloc(1024)

        assert a1 == arena1.base
        assert a2 == arena2.base
        assert arena1.used_bytes == _GRAN
        assert arena2.used_bytes == _GRAN
        assert len(backend1.allocations) == 1
        assert len(backend2.allocations) == 1

        arena1.free(a1)
        assert arena1.mapped_bytes == 0
        assert arena2.mapped_bytes == _GRAN

        arena2.free(a2)
        assert arena2.mapped_bytes == 0


# ---------------------------------------------------------------------------
# Lifecycle barrier: close-while-allocating
# ---------------------------------------------------------------------------


class _BlockingAllocBackend(_StubBackend):
    """Backend whose `allocate` blocks on a `threading.Event` so a
    second thread can call `close()` while a malloc is in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.completed = threading.Event()

    def allocate(self, va: int, size: int) -> int:
        self.entered.set()
        # Wait up to a few seconds for the test driver to signal release.
        # If the driver forgets to release us we want a clean test failure,
        # not an infinite hang.
        if not self.release.wait(timeout=5.0):
            raise RuntimeError("blocking allocate timed out waiting for release")
        try:
            return super().allocate(va, size)
        finally:
            self.completed.set()


class TestCloseRace:
    """Verify the in-flight counter in `_vmm_alloc_ext` blocks
    `close_and_drain` from running its snapshot while a malloc that
    won the OPEN state check is still mid-callback. Pre-fix this
    race could let close drain an empty map while the malloc later
    inserted a record into a closed arena, corrupting state and
    enabling use-after-free against the freed VA reserve.
    """

    def test_close_waits_for_in_flight_malloc(self):
        backend = _BlockingAllocBackend()
        arena = VmmArena(total_bytes=16 * _GRAN, backend=backend)

        with ThreadPoolExecutor(max_workers=2) as executor:
            malloc_future = executor.submit(arena.malloc, 1024)

            # Wait until the allocator callback is parked inside the
            # backend's allocate, holding an InFlightGuard.
            assert backend.entered.wait(timeout=5.0)

            # Kick off close. It must not return until the in-flight
            # malloc has inserted its record.
            close_future = executor.submit(arena.close)

            # Close is now spinning on in_flight; release the malloc.
            backend.release.set()
            assert backend.completed.wait(timeout=5.0)

            va = malloc_future.result(timeout=5.0)
            close_future.result(timeout=5.0)

        # The arena drained the record the malloc inserted: backend saw
        # one deallocate, the live record is gone, mapped_bytes is 0.
        assert va == arena.base
        assert arena.closed
        assert arena.live_allocation_count == 0
        assert arena.mapped_bytes == 0
        assert len(backend.deallocations) == 1
        d_va, d_size, _ = backend.deallocations[0]
        assert d_va == va
        assert d_size == _GRAN
        # Backend reserve was released exactly once at arena close.
        assert len(backend.released) == 1

    def test_close_bails_subsequent_malloc(self):
        """After close fires, mallocs that arrive AFTER the state CAS
        see CLOSED and raise immediately, even with the in-flight
        counter pattern in place."""
        backend = _StubBackend()
        arena = VmmArena(total_bytes=16 * _GRAN, backend=backend)
        arena.malloc(1024)
        arena.close()
        with pytest.raises(VmmArenaError, match="closed"):
            arena.malloc(1024)


# ---------------------------------------------------------------------------
# Drain rollback on Python-list construction failure
# ---------------------------------------------------------------------------


class TestDrainRollback:
    """If `PyList_New` or `Py_BuildValue` fails inside `close_and_drain`,
    the C extension rolls the state back to OPEN and leaves the
    allocation map intact so the caller can retry close. Triggering a
    Python allocation failure from a unit test is impractical, so this
    class covers the surrounding contract: a successful drain hands
    back every live record, and a re-entered close after a successful
    drain is a no-op.
    """

    def test_drain_returns_all_live_records(self):
        backend = _StubBackend()
        arena = VmmArena(total_bytes=16 * _GRAN, backend=backend)
        vas = [arena.malloc(1024) for _ in range(4)]
        arena.free(vas[1])  # one freed; three should remain at close

        arena.close()

        # 1 free + 3 close-time deallocs = 4 total backend.deallocations.
        assert len(backend.deallocations) == 4
        assert arena.closed

    def test_double_close_is_noop(self):
        backend = _StubBackend()
        arena = VmmArena(total_bytes=16 * _GRAN, backend=backend)
        arena.malloc(1024)
        arena.close()
        # Second close must not re-call backend.release_reserve.
        arena.close()
        assert len(backend.released) == 1
