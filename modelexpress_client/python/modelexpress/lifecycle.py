# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pause and resume the P2P source-serving role across memory-invalidation
windows (sleep/wake, CRIU checkpoint/restore, planned eviction).

Composes ``publish_metadata`` / ``unpublish_metadata`` / ``register_tensors``
with MxClient and NIXL agent teardown — the steps ``unpublish_metadata``
deliberately leaves to the caller.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from .load_strategy.base import (
    publish_metadata,
    register_tensors,
    unpublish_metadata,
)
from .metadata.client_factory import create_metadata_client

if TYPE_CHECKING:
    import torch.nn as nn

    from .load_strategy.context import LoadContext

logger = logging.getLogger("modelexpress.lifecycle")


def pause_serving(ctx: LoadContext) -> None:
    """Release MX network state ahead of a weight-memory invalidation.

    Order (intentional): mark STALE and stop heartbeat/worker server
    via ``unpublish_metadata`` so peers stop pulling, then close the
    MxClient channel, then shut down the NIXL agent.

    Clears ``ctx.mx_client`` and ``ctx.nixl_manager``. Retains
    ``ctx.tensors`` so :func:`resume_serving` can re-register the same
    set without re-walking the model (tensor objects survive
    GMS unmap/remap and CRIU + cuda-checkpoint).

    Teardown failures are logged, not raised — a partial teardown is
    safer than letting memory be invalidated with NIXL still pinned.
    """
    unpublish_metadata(ctx)

    if ctx.mx_client is not None:
        try:
            ctx.mx_client.close()
        except Exception as e:
            logger.warning(
                "[Worker %s] MxClient close failed during pause: %s",
                ctx.global_rank, e,
            )
        ctx.mx_client = None

    if ctx.nixl_manager is not None:
        try:
            ctx.nixl_manager.shutdown()
        except Exception as e:
            logger.warning(
                "[Worker %s] NIXL manager shutdown failed during pause: %s",
                ctx.global_rank, e,
            )
        ctx.nixl_manager = None


def resume_serving(
    ctx: LoadContext,
    model: nn.Module,
    *,
    worker_id: str | None = None,
) -> None:
    """Re-enter the P2P serving network after a weight-memory restore.

    Order: fresh MxClient (honors ``MX_METADATA_BACKEND``), new
    ``worker_id`` unless the caller supplied one, NIXL re-registration
    with ``reuse_discovered=True``, then republish metadata.

    ``worker_id=None`` mints a fresh id — the right default for
    relocation failover (different pod). Pass an explicit ``worker_id``
    to preserve identity for in-place wake.

    ``model`` is only consulted by :func:`register_tensors`' defensive
    re-discovery path if ``ctx.tensors`` is unexpectedly empty;
    ``reuse_discovered=True`` otherwise skips the walk, avoiding
    torch.compile / CUDA-graph artifacts attached post-warmup.
    """
    ctx.mx_client = create_metadata_client(worker_rank=ctx.worker_rank)
    ctx.worker_id = worker_id if worker_id is not None else uuid.uuid4().hex[:8]

    register_tensors(model, ctx, reuse_discovered=True)
    publish_metadata(ctx)
