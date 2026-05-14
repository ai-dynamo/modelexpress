# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real cuMem-backed VmmBackend implementation for the VmmArena.

Imports cuda-python lazily so this module can be loaded on non-CUDA hosts
without crashing at import time. The backend itself only works on a host
with CUDA + GPUs; instantiating CudaVmmBackend will raise if cuda-python
is not installed or the runtime is not available.

Allocation shape: each ``backend.allocate`` call creates and maps exactly
one physical chunk at the requested VA, matching the per-PyTorch-segment
shape. Multi-chunk-per-allocation patterns (e.g. for IPC export of
sub-handles) are out of scope here; PyTorch's caching allocator already
amortizes tensor allocations into segments before reaching us, so one
plugin call == one physical handle.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CudaUnavailableError(RuntimeError):
    """Raised when cuda-python or the CUDA runtime is not available."""


class CudaBackendError(RuntimeError):
    """Raised on a non-success CUresult from a driver call."""

    def __init__(self, op: str, result_code, detail: str | None = None) -> None:
        msg = f"{op} failed with CUDA result {result_code}"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)
        self.op = op
        self.result_code = result_code


def _import_cuda():
    """Lazy import of cuda-python driver bindings.

    Raises CudaUnavailableError if the package is missing.
    """
    try:
        from cuda.bindings import driver as cu
    except ImportError as e:
        raise CudaUnavailableError(
            "cuda-python is not installed; CudaVmmBackend requires it. "
            "Install with `pip install cuda-python`."
        ) from e
    return cu


class CudaVmmBackend:
    """VmmBackend implementation using cuMem* APIs.

    Intended use: pass into VmmArena(backend=CudaVmmBackend(device=0)) for
    production. The backend handles:
      - VA reservation via cuMemAddressReserve
      - Per-allocation physical chunk creation via cuMemCreate
      - Mapping and access setup via cuMemMap + cuMemSetAccess
      - Per-allocation teardown via cuMemUnmap + cuMemRelease
      - VA reserve cleanup via cuMemAddressFree

    The gpuDirectRDMACapable flag is set on every chunk allocation so the
    range is registerable with NIXL/UCX for RDMA.

    A CUDA context must already exist on the calling thread when methods
    are invoked. The simplest way is to call torch.cuda.set_device(device)
    or instantiate any CUDA tensor before using the backend; the backend's
    constructor verifies a context exists.

    The backend itself holds no per-allocation state (no chunk list). The
    VmmArena tracks live allocations in its own va -> _Allocation map and
    feeds (va, size, handle) tuples to deallocate when freeing.
    """

    def __init__(self, device: int = 0) -> None:
        self._cu = _import_cuda()
        self._device = device
        self._ensure_context()
        self._granularity = self._compute_granularity()
        logger.debug(
            "CudaVmmBackend initialized device=%d granularity=%d",
            self._device,
            self._granularity,
        )

    @property
    def device(self) -> int:
        return self._device

    # ---- VmmBackend protocol ------------------------------------------------

    def reserve(self, total_bytes: int) -> int:
        rounded = self._round_up(total_bytes, self._granularity)
        ptr = self._call(
            "cuMemAddressReserve",
            self._cu.cuMemAddressReserve(rounded, self._granularity, 0, 0),
        )
        base = int(ptr)
        logger.debug(
            "cuMemAddressReserve base=0x%x rounded=%d (requested=%d)",
            base,
            rounded,
            total_bytes,
        )
        return base

    def allocate(self, va: int, size: int) -> int:
        """Create one physical chunk of `size` bytes and map at `va`.

        Returns the cuMemGenericAllocationHandle as int. The caller
        (VmmArena) tracks it and passes it back to deallocate.

        Rollback discipline: cuMemCreate first; on cuMemMap or
        cuMemSetAccess failure, release the handle. On cuMemSetAccess
        failure after cuMemMap succeeded, also unmap before propagating.
        """
        if size % self._granularity != 0:
            raise ValueError(
                f"size {size} is not a multiple of granularity {self._granularity}"
            )
        prop = self._allocation_prop()
        handle = self._call("cuMemCreate", self._cu.cuMemCreate(size, prop, 0))
        try:
            self._call("cuMemMap", self._cu.cuMemMap(va, size, 0, handle, 0))
            try:
                access = self._access_desc()
                self._call(
                    "cuMemSetAccess",
                    self._cu.cuMemSetAccess(va, size, [access], 1),
                )
            except Exception:
                self._best_effort(
                    "cuMemUnmap (rollback)", self._cu.cuMemUnmap, va, size
                )
                raise
        except Exception:
            self._best_effort(
                "cuMemRelease (rollback)", self._cu.cuMemRelease, handle
            )
            raise
        logger.debug(
            "cuMemMap va=0x%x size=%d handle=%d", va, size, int(handle)
        )
        return int(handle)

    def deallocate(self, va: int, size: int, handle: int) -> None:
        """Unmap and release one allocation. Best-effort: errors are
        accumulated and raised at the end so callers see all failures.
        """
        errors: list[tuple[str, BaseException]] = []
        try:
            self._call("cuMemUnmap", self._cu.cuMemUnmap(va, size))
        except Exception as e:
            errors.append(("cuMemUnmap", e))
        try:
            self._call("cuMemRelease", self._cu.cuMemRelease(handle))
        except Exception as e:
            errors.append((f"cuMemRelease(handle=0x{handle:x})", e))
        logger.debug("deallocate va=0x%x size=%d handle=%d errors=%d",
                     va, size, handle, len(errors))
        if errors:
            details = "; ".join(f"{op}: {e}" for op, e in errors)
            raise CudaBackendError(
                "deallocate", None, f"{len(errors)} failure(s): {details}"
            )

    def release_reserve(self, base: int, total_bytes: int) -> None:
        """Free the VA reserve at `base`. The arena guarantees all
        per-allocation chunks have been deallocated before this is
        called. Any error is wrapped and re-raised.
        """
        try:
            self._call(
                "cuMemAddressFree",
                self._cu.cuMemAddressFree(base, total_bytes),
            )
        except Exception as e:
            raise CudaBackendError(
                "release_reserve", None, f"cuMemAddressFree failed: {e}"
            ) from e
        logger.debug("cuMemAddressFree base=0x%x total=%d", base, total_bytes)

    def allocation_granularity(self) -> int:
        """Backend driver granularity (cuMemCreate size multiple)."""
        return self._granularity

    # ---- Internals ----------------------------------------------------------

    def _best_effort(self, op: str, fn, *args) -> None:
        """Call fn(*args), log any failure, never raise.

        Used in rollback paths where a primary error is already in flight
        and we don't want to mask it with a secondary cleanup failure.
        """
        try:
            ret = fn(*args)
            err = ret[0] if isinstance(ret, tuple) else ret
            if err is not None and err != self._cu.CUresult.CUDA_SUCCESS:
                try:
                    _, name = self._cu.cuGetErrorString(err)
                    detail = (
                        name.decode() if isinstance(name, bytes) else str(name)
                    )
                except Exception:
                    detail = str(err)
                logger.warning("%s failed during rollback: %s", op, detail)
        except Exception as e:
            logger.warning("%s raised during rollback: %s", op, e)

    @staticmethod
    def _round_up(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

    def _ensure_context(self) -> None:
        """Verify a CUDA context exists on this thread.

        We don't create one ourselves; the caller is expected to have set
        one up (e.g. via torch.cuda.set_device or any CUDA tensor op).
        cuMemAddressReserve will fail with CUDA_ERROR_NOT_INITIALIZED if
        no context exists, which is a confusing error - this check turns
        it into something diagnosable.
        """
        err, ctx = self._cu.cuCtxGetCurrent()
        if err == self._cu.CUresult.CUDA_ERROR_NOT_INITIALIZED:
            raise CudaUnavailableError(
                "CUDA driver not initialized; "
                "call torch.cuda.init() and torch.cuda.set_device() before "
                "creating CudaVmmBackend"
            )
        if err != self._cu.CUresult.CUDA_SUCCESS:
            raise CudaBackendError("cuCtxGetCurrent", err)
        if int(ctx) == 0:
            raise CudaUnavailableError(
                "no CUDA context on current thread; "
                "call torch.cuda.set_device() or initialize CUDA before "
                "creating CudaVmmBackend"
            )

    def _allocation_prop(self):
        prop = self._cu.CUmemAllocationProp()
        prop.type = self._cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = (
            self._cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        )
        prop.location.id = self._device
        # GPUDirect RDMA capable so NIXL/UCX can register the resulting
        # range with the HCA without bouncing through host memory.
        prop.allocFlags.gpuDirectRDMACapable = 1
        return prop

    def _access_desc(self):
        access = self._cu.CUmemAccessDesc()
        access.location.type = (
            self._cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        )
        access.location.id = self._device
        access.flags = (
            self._cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        )
        return access

    def _compute_granularity(self) -> int:
        prop = self._allocation_prop()
        gran = self._call(
            "cuMemGetAllocationGranularity",
            self._cu.cuMemGetAllocationGranularity(
                prop,
                self._cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            ),
        )
        return int(gran)

    def _call(self, op: str, result):
        """Unwrap a (CUresult, *rest) tuple into rest, raising on error.

        cuda-python returns a tuple from every driver call. The first
        element is CUresult; success is CUresult.CUDA_SUCCESS. Subsequent
        elements are output arguments (handles, pointers, etc.). For
        success we return either the single output or a tuple of them.
        """
        err, *rest = result
        if err != self._cu.CUresult.CUDA_SUCCESS:
            try:
                _, name = self._cu.cuGetErrorString(err)
                detail = name.decode() if isinstance(name, bytes) else str(name)
            except Exception:
                detail = None
            raise CudaBackendError(op, err, detail)
        if not rest:
            return None
        if len(rest) == 1:
            return rest[0]
        return tuple(rest)
