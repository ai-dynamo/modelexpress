# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VMM arena allocator subpackage.

Layout:
- ``arena``: ``VmmArena`` reserves a large CUDA VA range and bump-assigns
  per-allocation addresses backed by individual ``cuMemCreate`` handles.
- ``backend``: ``CudaVmmBackend`` wraps the CUDA driver VMM calls
  (``cuMemAddressReserve``, ``cuMemCreate``, ``cuMemMap``, etc.).
- ``hook``: ``use_arena`` context manager wires the arena into PyTorch's
  ``CUDAPluggableAllocator`` via the ``_alloc_ext`` C extension.
- ``_alloc_ext``: optional C extension hot path for the malloc/free shims.

See ``README.md`` in this directory for the user-facing overview.
"""

from .arena import (
    DEFAULT_VA_RESERVE,
    VmmArena,
    VmmArenaError,
    VmmBackend,
)
from .backend import (
    CudaBackendError,
    CudaUnavailableError,
    CudaVmmBackend,
)
from .hook import (
    ARENA_AVAILABLE,
    ArenaUnavailableError,
    active_arena,
    install_pluggable_allocator,
    is_active,
    use_arena,
)
from .runtime import (
    log_arena_post_load,
    maybe_enter_vmm_arena,
)

__all__ = [
    "ARENA_AVAILABLE",
    "ArenaUnavailableError",
    "CudaBackendError",
    "CudaUnavailableError",
    "CudaVmmBackend",
    "DEFAULT_VA_RESERVE",
    "VmmArena",
    "VmmArenaError",
    "VmmBackend",
    "active_arena",
    "install_pluggable_allocator",
    "is_active",
    "log_arena_post_load",
    "maybe_enter_vmm_arena",
    "use_arena",
]
