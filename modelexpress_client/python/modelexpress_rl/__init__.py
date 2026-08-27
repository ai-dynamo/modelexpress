# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelExpress clients and protobuf bindings for RL weight refit."""

from .control import ModelExpressControlClient, WeightVersion, WeightVersionState
from .inference import (
    ModelExpressGeneratorClient,
    ModelExpressGeneratorConfig,
    ObjectStorageGeneratorConfig,
    SglangGeneratorContext,
    VllmGeneratorContext,
)
from .train import (
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    ObjectStorageConfig,
    TrainerStagingMode,
    WeightPayloadFormat,
)
from .object_storage import ObjectStorageSource, ObjectStorageType
from .version import WeightVersionRef

__all__ = [  # noqa: RUF022 - grouped by public API role, not alphabetically.
    # Framework-facing clients.
    "ModelExpressControlClient",
    "ModelExpressGeneratorClient",
    "ModelExpressTrainerClient",
    # Configuration fixed when a worker client is initialized.
    "ModelExpressGeneratorConfig",
    "ModelExpressTrainerConfig",
    "ObjectStorageConfig",
    "ObjectStorageGeneratorConfig",
    "SglangGeneratorContext",
    "TrainerStagingMode",
    "WeightPayloadFormat",
    "VllmGeneratorContext",
    # Version values shared across the control, trainer, and generator clients.
    "ObjectStorageSource",
    "ObjectStorageType",
    "WeightVersion",
    "WeightVersionRef",
    "WeightVersionState",
]
