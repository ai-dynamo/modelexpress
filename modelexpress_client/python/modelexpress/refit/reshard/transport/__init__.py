# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Transport backends for the reshard-refit weight pull.

``base`` defines the engine-neutral read interface (:class:`Transport`,
:class:`ReadDescriptor`) plus an in-memory reference implementation for tests;
``nixl`` is the production NIXL/RDMA backend. The receiver depends only on the
``Transport`` protocol, so a non-NIXL backend is a new module here."""

from modelexpress.refit.reshard.transport.base import (
    InMemoryReferenceTransport,
    ReadDescriptor,
    Transport,
)
from modelexpress.refit.reshard.transport.nixl import NixlReshardTransport

__all__ = [
    "InMemoryReferenceTransport",
    "NixlReshardTransport",
    "ReadDescriptor",
    "Transport",
]
