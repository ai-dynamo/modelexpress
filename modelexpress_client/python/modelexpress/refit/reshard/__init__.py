# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""No-gather sharded weight resharding (trainer -> inference), vended by MX.

A framework-neutral core for slice-level weight transfer: capture which slice of
each full source tensor an engine's own weight loader reads (geometry.py),
intersect those slices against the published source shards (slice_plan.py), and
emit the exact byte segments to RDMA-pull (plan.py). No all-gather and no
per-model conversion specs - the engine's real loaders define the reshard.

Overlap is arbitrary per-dim (not just dim-0). Descriptor-heavy strided copies
can pull a complete dim-0-sharded source into contiguous staging and replay the
captured views locally. Any tensor whose placement cannot be represented safely
raises ``UnsupportedReshard``. The receiver fails the update before transfer
because a general fallback path is not implemented.
"""

from modelexpress.refit.reshard.geometry import (
    LazyWeight,
    OpChain,
    RecordedCopy,
    UnsupportedReshard,
    capture_geometry,
)
from modelexpress.refit.reshard.types import IncompleteRefit
from modelexpress.refit.reshard.transfer_plan import (
    FullPullSource,
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
from modelexpress.refit.reshard.megatron import (
    MegatronTargetLayout,
    MegatronTargetSpec,
    lower_megatron_target,
)
from modelexpress.refit.reshard.megatron_receiver import MegatronReshardReceiver
from modelexpress.refit.reshard.megatron_aliases import (
    MegatronAliasInput,
    build_hf_aliases,
)
from modelexpress.refit.reshard.megatron_publisher import (
    MegatronPublishedTensorSpec,
    publish_megatron_reshard_view,
    publish_registered_shard_table,
)
from modelexpress.refit.reshard.rendezvous import (
    MxReshardRendezvous,
    PublishedShard,
    PublishedTensor,
    RendezvousPayload,
    gather_sources,
    wrap_rendezvous_blob,
)

__all__ = [
    "InMemoryReferenceTransport",
    "FullPullSource",
    "IncompleteRefit",
    "LazyWeight",
    "MegatronAliasInput",
    "MegatronPublishedTensorSpec",
    "MegatronReshardReceiver",
    "MegatronTargetLayout",
    "MegatronTargetSpec",
    "MxReshardRendezvous",
    "NixlReshardTransport",
    "OpChain",
    "PublishedShard",
    "PublishedTensor",
    "RendezvousPayload",
    "PullSegment",
    "ReadDescriptor",
    "RecordedCopy",
    "ReshardReceiver",
    "Shard",
    "SourceInfo",
    "Transport",
    "TransferPlan",
    "UnsupportedReshard",
    "build_hf_aliases",
    "capture_geometry",
    "classic_cuda_alloc",
    "execute_transfer",
    "gather_sources",
    "intersect",
    "lower_megatron_target",
    "op_chain_to_box",
    "paired_runs",
    "plan_pull",
    "plan_transfer",
    "publish_megatron_reshard_view",
    "publish_registered_shard_table",
    "wrap_rendezvous_blob",
]
