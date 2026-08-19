# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NCCL M2N collective refit: the client-side control plane.

A sibling of the NIXL pull path in ``modelexpress.refit.reshard``, not a mode
of it. Nothing here imports NIXL, and nothing here imports torch, NCCL or CUDA
either -- the plan contract and the rendezvous are deliberately testable
without a GPU. See ``docs/NCCL_M2N_REFIT.md``.
"""

from .backend import DEFAULT_LAYER_GROUP, NcclM2nReceiver, NcclM2nSender
from .client import RefitClientGenerator, RefitClientTrainer
from .comm import CommunicatorCache, LaneCommunicator, LaneKey, NcclUnavailableError
from .plan import (
    PlanCoverageError,
    build_mesh,
    build_param_plan,
    default_placements,
    default_shard_dim,
    generator_rank_offset,
    grouped_expert_name,
    is_bulk_param,
    is_expert_param,
    plan_digest,
    validate_coverage,
)
from .rendezvous import (
    CollectiveRendezvous,
    EpochChangedError,
    GroupNotReadyError,
    LaneMembership,
    Membership,
    RendezvousError,
)
from .spi import Loader, LocalParamSpec, Publisher, RefitCtx, resolve_specs
from .types import (
    MeshSpec,
    MiscParam,
    ParamPlan,
    Placement,
    PlacementKind,
    ReshardPlan,
    Role,
)

__all__ = [
    "CollectiveRendezvous",
    "CommunicatorCache",
    "DEFAULT_LAYER_GROUP",
    "EpochChangedError",
    "GroupNotReadyError",
    "LaneCommunicator",
    "LaneKey",
    "LaneMembership",
    "Loader",
    "LocalParamSpec",
    "Membership",
    "MeshSpec",
    "MiscParam",
    "NcclM2nReceiver",
    "NcclM2nSender",
    "NcclUnavailableError",
    "ParamPlan",
    "Placement",
    "PlacementKind",
    "PlanCoverageError",
    "Publisher",
    "RefitClientGenerator",
    "RefitClientTrainer",
    "RefitCtx",
    "RendezvousError",
    "ReshardPlan",
    "Role",
    "build_mesh",
    "build_param_plan",
    "default_placements",
    "default_shard_dim",
    "generator_rank_offset",
    "grouped_expert_name",
    "is_bulk_param",
    "is_expert_param",
    "plan_digest",
    "resolve_specs",
    "validate_coverage",
]
