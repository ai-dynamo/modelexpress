# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""P2P weight loading over the configured NIXL transport."""

from .nixl_p2p_strategy import NixlP2PStrategy


class RdmaStrategy(NixlP2PStrategy):
    """Receive weights using the existing automatic transport selection."""

    name = "rdma"
    channel = "auto"
