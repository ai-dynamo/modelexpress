# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer-inference weight synchronization via NIXL RDMA.

Sub-packages
------------
protocol/    Pure data types and serialization.  No torch dependency.
             Safe to import on any process (trainer, server, worker).

engine/      LazyWeight bake pass and WeightLoaderAdapter ABC.
             Requires torch.  One adapter per inference engine.

planner/     Op-chain resolver (torch) and region router (pure math).
             LocalPlanner routes on the client; ServerPlanner offloads to MX.

transport/   NixlExecutor: execute pre-built RdmaDescriptor lists via NIXL.

roles/       PullRole (inference pulls from trainer) and
             PushRole (trainer pushes to inference workers).

adapters/    Model-specific naming bridges (MoEAdapter, etc.).

Quick start (PULL mode)::

    from modelexpress.weight_transfer import PullRole, MoEAdapter
    from modelexpress.weight_transfer.planner import ServerPlanner

    adapter = MyVllmAdapter(num_experts=64)
    planner = ServerPlanner(mx_client)
    role = PullRole(adapter, nixl_manager, device_id=0, planner=planner)

    table = fetch_trainer_table(mx_client, model_name)
    role.initialize(model, table)   # bake once
    role.sync()                     # call each training step
"""

# Protocol types (no torch)
from .protocol import (
    OpSpec,
    OpChain,
    TrainerShard,
    TrainerTensor,
    TrainerTable,
    InferenceShard,
    InferenceTable,
    ResolvedRegion,
    RdmaDescriptor,
    SyncMode,
    encode_trainer_table,
    decode_trainer_table,
    encode_inference_table,
    decode_inference_table,
)

# Engine layer
from .engine import BakeRecorder, LazyWeight, RecordedCopy, WeightLoaderAdapter

# Roles
from .roles import PullRole, PushRole

# Adapters
from .adapters import MoEAdapter

__all__ = [
    # protocol
    "OpSpec",
    "OpChain",
    "TrainerShard",
    "TrainerTensor",
    "TrainerTable",
    "InferenceShard",
    "InferenceTable",
    "ResolvedRegion",
    "RdmaDescriptor",
    "SyncMode",
    "encode_trainer_table",
    "decode_trainer_table",
    "encode_inference_table",
    "decode_inference_table",
    # engine
    "BakeRecorder",
    "LazyWeight",
    "RecordedCopy",
    "WeightLoaderAdapter",
    # roles
    "PullRole",
    "PushRole",
    # adapters
    "MoEAdapter",
]
