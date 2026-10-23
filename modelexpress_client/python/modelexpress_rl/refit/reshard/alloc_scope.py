# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Where NIXL-registered refit receive and staging buffers are allocated from.

Registering a buffer with NIXL constrains how it may be allocated, and the
constraint is a property of the accelerator rather than of the refit. This module
holds the family-agnostic selection so callers do not have to name a specific
allocator; :mod:`modelexpress_rl.refit.reshard.cuda_pool` holds the one
accelerator-specific implementation that exists.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, ContextManager

if TYPE_CHECKING:
    import torch

    from modelexpress.accelerators import AcceleratorBackend


def registered_buffer_alloc_scope(
    backend: "AcceleratorBackend",
    *,
    device: "torch.device | None" = None,
) -> ContextManager[None]:
    """Return the allocation scope ``backend`` needs for NIXL-registered buffers.

    A backend that reports ``requires_classic_alloc_pool()`` gets
    :func:`~modelexpress_rl.refit.reshard.cuda_pool.classic_cuda_alloc`; every
    other backend gets a no-op and allocates normally, which is the correct answer
    here rather than a fallback. CUDA is the only dedicated pool implementation;
    fail clearly if another backend requests one instead of silently using CUDA
    code.

    ``device`` is the device the buffers are allocated on. A pool scope applies
    to one device, so pass it whenever it may differ from the backend's current
    device; ``None`` or a device without an index means the current device.
    """
    if backend.requires_classic_alloc_pool():
        if backend.name != "cuda":
            raise NotImplementedError(
                f"No registered-buffer allocation pool is implemented for "
                f"backend {backend.name!r}"
            )
        # Imported per call rather than at module scope so this selection carries
        # no import-time dependency on the one accelerator-specific
        # implementation. Note this does not by itself keep cuda_pool out of a
        # non-CUDA process: the package __init__ re-exports classic_cuda_alloc,
        # so importing anything under refit.reshard loads that module anyway.
        from modelexpress_rl.refit.reshard.cuda_pool import classic_cuda_alloc

        return classic_cuda_alloc(device)
    return nullcontext()


__all__ = ["registered_buffer_alloc_scope"]
