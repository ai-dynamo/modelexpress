# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CUDAPluggableAllocator hook around VmmArena (Phase 5).

The C extension itself can be loaded and exercised with mocked callbacks
on any host (no CUDA needed). Tests that route through PyTorch's
CUDAPluggableAllocator + MemPool require CUDA and skip otherwise.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# C extension loadable on any host where it built
# ---------------------------------------------------------------------------


class TestExtensionLoad:
    """Verify the C extension's import + init_module surface. Skipped when
    the .so is absent (no-compiler install path); the optional-extension
    fallback is covered separately by TestOptionalExtension."""

    @pytest.fixture(autouse=True)
    def _ensure_arena_available(self):
        from modelexpress.vmm import hook as vmm_hook

        if not vmm_hook.ARENA_AVAILABLE:
            pytest.skip("C extension not built; arena fast path disabled")

    def test_can_import(self):
        from modelexpress.vmm import _alloc_ext as _vmm_alloc_ext

        assert hasattr(_vmm_alloc_ext, "init_module")

    def test_init_module_accepts_callables(self):
        from modelexpress.vmm import _alloc_ext as _vmm_alloc_ext

        called: dict[str, list] = {"malloc": [], "free": []}

        def malloc_cb(size, device, stream):
            called["malloc"].append((size, device, stream))
            return 0

        def free_cb(ptr, size, device, stream):
            called["free"].append((ptr, size, device, stream))

        _vmm_alloc_ext.init_module(malloc_cb, free_cb)
        # init_module returns None on success.

    def test_init_module_rejects_non_callable(self):
        from modelexpress.vmm import _alloc_ext as _vmm_alloc_ext

        with pytest.raises(TypeError):
            _vmm_alloc_ext.init_module(42, 43)


# ---------------------------------------------------------------------------
# vmm_hook plumbing on non-CUDA host (uses stub backend)
# ---------------------------------------------------------------------------


class TestVmmHookPlumbing:
    """Verify the malloc/free dispatch into the active arena.

    These tests don't go through PyTorch's MemPool - they call the
    module-private dispatch functions directly. That's enough to verify
    the active-arena handoff without needing CUDA.
    """

    def test_malloc_with_no_arena_returns_zero(self):
        from modelexpress.vmm import hook as vmm_hook

        # Reset state
        vmm_hook._active_arena = None
        addr = vmm_hook._mx_malloc(1024, 0, 0)
        assert addr == 0

    def test_malloc_dispatches_to_active_arena(self):
        from modelexpress.vmm import hook as vmm_hook
        from modelexpress.vmm.arena import VmmArena, _StubBackend

        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        vmm_hook._active_arena = arena
        try:
            addr = vmm_hook._mx_malloc(1024, 0, 0)
            assert addr == arena.base
            # arena saw the allocation and mapped one granularity-sized handle.
            assert arena.live_allocation_count == 1
            assert arena.mapped_bytes == arena.granularity
        finally:
            arena.close()
            vmm_hook._active_arena = None

    def test_free_with_no_arena_is_silent(self):
        from modelexpress.vmm import hook as vmm_hook

        vmm_hook._active_arena = None
        # Must not raise
        vmm_hook._mx_free(0xDEADBEEF, 1024, 0, 0)

    def test_free_dispatches_to_arena(self):
        from modelexpress.vmm import hook as vmm_hook
        from modelexpress.vmm.arena import VmmArena, _StubBackend

        arena = VmmArena(total_bytes=1 << 30, backend=_StubBackend())
        vmm_hook._active_arena = arena
        try:
            addr = vmm_hook._mx_malloc(1024, 0, 0)
            used_before = arena.used_bytes
            mapped_before = arena.mapped_bytes
            assert arena.live_allocation_count == 1

            vmm_hook._mx_free(addr, 1024, 0, 0)

            # The bump pointer is cumulative, but physical mapping is released.
            assert arena.used_bytes == used_before
            assert arena.mapped_bytes == mapped_before - arena.granularity
            assert arena.live_allocation_count == 0
        finally:
            arena.close()
            vmm_hook._active_arena = None


# ---------------------------------------------------------------------------
# Real CUDA-backed end-to-end (Phase 5 validation)
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        from cuda.bindings import driver  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _cuda_available(),
    reason="requires CUDA + cuda-python",
)
class TestUseArenaWithMemPool:
    @pytest.fixture(autouse=True)
    def _cuda_init(self):
        import torch

        torch.cuda.init()
        torch.cuda.set_device(0)
        # Force primary context binding on this thread (set_device alone
        # is insufficient - cuCtxGetCurrent returns 0 until a real CUDA
        # op runs).
        _ = torch.zeros(1, device="cuda:0")
        torch.cuda.synchronize()
        yield 0

    def test_torch_alloc_inside_use_arena_lands_in_arena(self):
        """Inside use_arena, torch.empty allocations must come from the arena."""
        import torch

        from modelexpress.vmm.arena import VmmArena
        from modelexpress.vmm.backend import CudaVmmBackend
        from modelexpress.vmm.hook import use_arena

        backend = CudaVmmBackend(device=0)
        gran = backend.allocation_granularity()
        arena = VmmArena(
            total_bytes=gran * 8,
            backend=backend,
        )
        try:
            with use_arena(arena, device=0):
                t = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda:0")
                # Tensor data_ptr should be inside the arena's range.
                base, _ = arena.registered_range()
                assert base <= t.data_ptr() < base + arena.total_bytes
                # And the arena should have recorded the allocation.
                assert arena.live_allocation_count >= 1
        finally:
            arena.close()

    def test_outside_use_arena_uses_default_allocator(self):
        """Without use_arena, normal cudaMalloc path is unaffected."""
        import torch

        # Allocate before any use_arena scope so the caching allocator path
        # is exercised.
        t = torch.empty(1024, dtype=torch.uint8, device="cuda:0")
        # No assertion about arena state; just that the allocation works.
        assert t.numel() == 1024

    def test_nesting_raises(self):
        from modelexpress.vmm.arena import VmmArena
        from modelexpress.vmm.backend import CudaVmmBackend
        from modelexpress.vmm.hook import use_arena

        backend = CudaVmmBackend(device=0)
        gran = backend.allocation_granularity()
        a1 = VmmArena(total_bytes=gran * 4, backend=backend)
        a2 = VmmArena(total_bytes=gran * 4, backend=backend)
        try:
            with use_arena(a1, device=0):
                with pytest.raises(RuntimeError, match="already active"):
                    with use_arena(a2, device=0):
                        pass
        finally:
            a1.close()
            a2.close()


# ---------------------------------------------------------------------------
# Build-optional / runtime fallback (no CUDA needed)
# ---------------------------------------------------------------------------


class TestOptionalExtension:
    """Verify the build-optional + runtime-fallback machinery.

    setup.py builds _vmm_alloc_ext as Extension(optional=True) and the
    custom BuildExtension catches compiler errors. vmm_hook probes for
    the .so at import time and exposes ARENA_AVAILABLE. When MX_VMM_ARENA=1
    is set but the extension is missing, the vllm loader falls back to
    nullcontext rather than crashing.
    """

    def test_arena_available_flag_is_bool(self):
        from modelexpress.vmm import hook as vmm_hook

        assert isinstance(vmm_hook.ARENA_AVAILABLE, bool)

    def test_arena_unavailable_error_subclasses_runtimeerror(self):
        from modelexpress.vmm.hook import ArenaUnavailableError

        assert issubclass(ArenaUnavailableError, RuntimeError)

    def test_ensure_callbacks_raises_when_unavailable(self, monkeypatch):
        """Simulate a missing/unbuilt C extension and verify the hook
        raises ArenaUnavailableError instead of an opaque ImportError."""
        from modelexpress.vmm import hook as vmm_hook
        from modelexpress.vmm.hook import ArenaUnavailableError

        # Reset _callbacks_initialized so the guard doesn't short-circuit.
        monkeypatch.setattr(vmm_hook, "_callbacks_initialized", False)
        monkeypatch.setattr(vmm_hook, "ARENA_AVAILABLE", False)
        monkeypatch.setattr(
            vmm_hook,
            "_import_error",
            ImportError("simulated missing _vmm_alloc_ext"),
        )

        with pytest.raises(ArenaUnavailableError, match="not available"):
            vmm_hook._ensure_callbacks_initialized()

    def test_vllm_loader_falls_back_to_nullcontext(self, monkeypatch, caplog):
        """When MX_VMM_ARENA=1 but ARENA_AVAILABLE=False, the vllm loader
        helper should yield without installing arena machinery and emit a
        warning, not crash."""
        from modelexpress.vmm import hook as vmm_hook
        from modelexpress.engines.vllm import loader as vllm_loader

        monkeypatch.setenv("MX_VMM_ARENA", "1")
        monkeypatch.setattr(vmm_hook, "ARENA_AVAILABLE", False)

        # Minimal LoadContext stub: the helper only reads global_rank/device_id
        # before bailing out on the unavailable path.
        class _Ctx:
            global_rank = 0
            device_id = 0

        with caplog.at_level("WARNING", logger=vllm_loader.logger.name):
            # Entering the context manager must not raise; the body must
            # execute; exiting must clean up without raising. This is the
            # behavioral contract of nullcontext-equivalent.
            entered = False
            with vllm_loader._maybe_enter_vmm_arena(_Ctx()):
                entered = True
            assert entered

        # And we must have surfaced a warning so the user can see why.
        assert any(
            "MX_VMM_ARENA=1 set but the modelexpress.vmm._alloc_ext" in rec.message
            for rec in caplog.records
        ), (
            "expected fallback warning in caplog; got: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_vllm_loader_no_op_when_env_unset(self, monkeypatch):
        """Baseline: with MX_VMM_ARENA unset, the helper yields without
        installing arena machinery, regardless of ARENA_AVAILABLE."""
        from modelexpress.engines.vllm import loader as vllm_loader

        monkeypatch.delenv("MX_VMM_ARENA", raising=False)

        class _Ctx:
            global_rank = 0
            device_id = 0

        entered = False
        with vllm_loader._maybe_enter_vmm_arena(_Ctx()):
            entered = True
        assert entered


# ---------------------------------------------------------------------------
# Loader lifecycle: publish-after-success, replace warn
# ---------------------------------------------------------------------------


class _StubTargetDevice:
    """No-op CM that pretends to be ctx.target_device."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _StubCtx:
    global_rank = 0
    device_id = 0
    target_device = _StubTargetDevice()
    vmm_arena = None


class _StubCudaVmmBackend:
    """Stub CudaVmmBackend with a fixed granularity, no CUDA required."""

    def __init__(self, device=0, granularity=2 * 1024 * 1024):
        self.device = device
        self._granularity = granularity
        self.reserved: list[tuple[int, int]] = []
        self.allocations: list[tuple[int, int, int]] = []
        self.deallocations: list[tuple[int, int, int]] = []
        self.released: list[tuple[int, int]] = []
        self._next_handle = 1

    def reserve(self, total_bytes):
        base = 0x2000_0000
        self.reserved.append((base, total_bytes))
        return base

    def allocate(self, va, size):
        handle = self._next_handle
        self._next_handle += 1
        self.allocations.append((va, size, handle))
        return handle

    def deallocate(self, va, size, handle):
        self.deallocations.append((va, size, handle))

    def release_reserve(self, base, total_bytes):
        self.released.append((base, total_bytes))

    def allocation_granularity(self):
        return self._granularity


class TestLoaderLifecycle:
    """Verify _maybe_enter_vmm_arena's lifecycle invariants without CUDA.

    Stubs out CudaVmmBackend (no cuda-python dependency) and use_arena
    (which would need PyTorch + CUDA). The vmm_hook.ARENA_AVAILABLE flag
    is True in the dev environment because the .so is built; if a future
    CI legitimately can't build it, these tests skip.
    """

    @pytest.fixture(autouse=True)
    def _ensure_arena_available(self):
        from modelexpress.vmm import hook as vmm_hook

        if not vmm_hook.ARENA_AVAILABLE:
            pytest.skip("C extension not built; loader tests need ARENA_AVAILABLE")

    def _patch_loader_deps(self, monkeypatch, use_arena_factory=None):
        """Replace CudaVmmBackend + use_arena with stubs."""
        from contextlib import contextmanager

        from modelexpress.engines.vllm import loader as vllm_loader

        monkeypatch.setattr(vllm_loader, "_vmm_arenas", {})
        monkeypatch.setattr(
            "modelexpress.vmm.backend.CudaVmmBackend", _StubCudaVmmBackend
        )

        if use_arena_factory is None:

            @contextmanager
            def _stub_use_arena(arena, device):
                yield

            monkeypatch.setattr("modelexpress.vmm.hook.use_arena", _stub_use_arena)
        else:
            monkeypatch.setattr("modelexpress.vmm.hook.use_arena", use_arena_factory)

        return vllm_loader

    def test_arena_published_on_successful_body(self, monkeypatch):
        """When the load body returns normally, the arena ends up in
        _vmm_arenas keyed by device_id."""
        monkeypatch.setenv("MX_VMM_ARENA", "1")
        vllm_loader = self._patch_loader_deps(monkeypatch)

        with vllm_loader._maybe_enter_vmm_arena(_StubCtx()):
            pass  # success

        assert 0 in vllm_loader._vmm_arenas

    def test_arena_not_published_on_body_exception(self, monkeypatch):
        """When the load body raises, the freshly-created arena is closed
        and NOT retained in _vmm_arenas. Re-running the helper starts
        from a clean slate."""
        monkeypatch.setenv("MX_VMM_ARENA", "1")
        vllm_loader = self._patch_loader_deps(monkeypatch)

        class _LoadError(RuntimeError):
            pass

        with pytest.raises(_LoadError):
            with vllm_loader._maybe_enter_vmm_arena(_StubCtx()):
                raise _LoadError("synthetic load failure")

        assert 0 not in vllm_loader._vmm_arenas

    def test_replace_arena_logs_warning(self, monkeypatch, caplog):
        """When _vmm_arenas already has an arena for the device, the
        helper logs a clear WARNING about replacement before closing it."""
        from modelexpress.engines.vllm import loader as vllm_loader

        monkeypatch.setenv("MX_VMM_ARENA", "1")
        vllm_loader_mod = self._patch_loader_deps(monkeypatch)

        # Pre-populate with a stub "old" arena.
        class _StubOldArena:
            closed = False

            def close(self):
                self.closed = True

        old_arena = _StubOldArena()
        vllm_loader_mod._vmm_arenas[0] = old_arena

        with caplog.at_level("WARNING", logger=vllm_loader_mod.logger.name):
            with vllm_loader_mod._maybe_enter_vmm_arena(_StubCtx()):
                pass

        assert old_arena.closed
        assert any(
            "Replacing existing VmmArena" in rec.message for rec in caplog.records
        ), (
            "expected replacement WARNING; got: "
            f"{[r.message for r in caplog.records]}"
        )
