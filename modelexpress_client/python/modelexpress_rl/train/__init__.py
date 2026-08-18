# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer-side ModelExpress RL integrations."""

from .adapter import (
    CompletionFence,
    NixlMetadataProvider,
    StagedWeightVersionShardData,
    TrainerEngineAdapter,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionShardManifest,
    WeightVersionShardManifestPublisher,
)
from .client import ModelExpressTrainerClient, StagedWeightVersionShard
from .manifest import WeightVersionShardManifestService

__all__ = [
    "CompletionFence",
    "ModelExpressTrainerClient",
    "NixlMetadataProvider",
    "StagedWeightVersionShard",
    "StagedWeightVersionShardData",
    "TrainerEngineAdapter",
    "TrainerStagingMode",
    "WeightPayloadFormat",
    "WeightVersionShardManifest",
    "WeightVersionShardManifestPublisher",
    "WeightVersionShardManifestService",
]
