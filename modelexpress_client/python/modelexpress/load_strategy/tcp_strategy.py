# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-domain weight loading over an isolated TCP context."""

from .. import envs
from ..topology import adaptive_enabled
from .context import LoadContext
from .nixl_p2p_strategy import NixlP2PStrategy


class TcpStrategy(NixlP2PStrategy):
    name = "tcp"
    channel = "tcp"

    def is_available(self, ctx: LoadContext) -> bool:
        # TODO: Support cross-domain TCP loading from LIBFABRIC sources.
        return (
            adaptive_enabled()
            and envs.MX_NIXL_BACKEND == "UCX"
            and super().is_available(ctx)
        )
