# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""No-gather sharded weight resharding (trainer -> inference), vended by MX.

A framework-neutral core for slice-level weight transfer: capture which slice of
each full source tensor an engine's own weight loader reads (geometry.py),
intersect those slices against the published source shards (slice_plan.py), and
emit the exact byte segments to RDMA-pull (plan.py). No all-gather and no
per-model conversion specs - the engine's real loaders define the reshard.

Overlap is arbitrary per-dim (not just dim-0). Any tensor whose slice can't be
expressed as a box raises ``UnsupportedReshard`` and falls back to a full
(non-sliced) pull, so a sync never aborts over one awkward tensor.
"""

from modelexpress.refit.reshard.geometry import (
    LazyWeight,
    OpChain,
    RecordedCopy,
    UnsupportedReshard,
    capture_geometry,
)
from modelexpress.refit.reshard.transfer_plan import (
    SourceInfo,
    TransferPlan,
    execute_transfer,
    plan_transfer,
)
from modelexpress.refit.reshard.slice_plan import (
    PullSegment,
    Shard,
    intersect,
    op_chain_to_box,
    paired_runs,
    plan_pull,
)
from modelexpress.refit.reshard.transport import (
    InMemoryReferenceTransport,
    NixlReshardTransport,
    ReadDescriptor,
    Transport,
)
from modelexpress.refit.reshard.cuda_pool import classic_cuda_alloc
from modelexpress.refit.reshard.receiver import ReshardReceiver
from modelexpress.refit.reshard.rendezvous import (
    MxReshardRendezvous,
    PublishedShard,
    PublishedTensor,
    gather_sources,
    wrap_rendezvous_blob,
)

__all__ = [
    "InMemoryReferenceTransport",
    "LazyWeight",
    "MxReshardRendezvous",
    "NixlReshardTransport",
    "OpChain",
    "PublishedShard",
    "PublishedTensor",
    "PullSegment",
    "ReadDescriptor",
    "RecordedCopy",
    "ReshardReceiver",
    "Shard",
    "SourceInfo",
    "Transport",
    "TransferPlan",
    "UnsupportedReshard",
    "capture_geometry",
    "classic_cuda_alloc",
    "execute_transfer",
    "gather_sources",
    "intersect",
    "op_chain_to_box",
    "paired_runs",
    "plan_pull",
    "plan_transfer",
    "wrap_rendezvous_blob",
]
