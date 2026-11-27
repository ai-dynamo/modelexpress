# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-domain weight loading over an isolated TCP context."""

from .. import envs
from ..adapter import StrategyFailed
from ..topology import adaptive_enabled
from .base import register_tensors
from .context import LoadContext, LoadResult, P2PFailure
from .nixl_p2p_strategy import NixlP2PStrategy


class TcpStrategy(NixlP2PStrategy):
    name = "tcp"
    channel = "tcp"

    def is_available(self, ctx: LoadContext) -> bool:
        # TODO: Support cross-domain TCP loading from LIBFABRIC sources.
        # NIXL lacks per-agent provider selection; FI_PROVIDER is process-wide
        # and read once during libfabric initialization.
        return (
            adaptive_enabled()
            and envs.MX_NIXL_BACKEND == "UCX"
            and super().is_available(ctx)
        )

    def prepare(self, result: LoadResult, ctx: LoadContext) -> None:
        if self._state(ctx).last_failure is P2PFailure.MODEL:
            raise StrategyFailed(
                "P2P model preparation, discovery or processing failed; use a storage loader",
                mutated=False,
            )

    def finalize(self, result: LoadResult, ctx: LoadContext) -> None:
        """Hand TCP-received weights to the normal publication agent."""
        self.rollback(ctx)
        if result.model_for_publish is not None:
            register_tensors(result, ctx)
