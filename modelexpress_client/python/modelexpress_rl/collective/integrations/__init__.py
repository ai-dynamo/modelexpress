# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework adapters for the NCCL M2N collective refit path."""

from .miles import (
    CollectiveTopology,
    MilesPublisher,
    MilesTrainerSession,
    MilesTransferCoordinator,
)
from .miles_native import (
    MilesNativePublisher,
    MilesNativeTensorRecord,
    MilesSourceBinding,
    MilesSourceRecipe,
    inventory_miles_native_tensors,
)
from .sglang import (
    SglangGeneratorSession,
    SglangLoader,
    SglangParameterBinding,
)

__all__ = [
    "CollectiveTopology",
    "MilesNativePublisher",
    "MilesNativeTensorRecord",
    "MilesPublisher",
    "MilesSourceBinding",
    "MilesSourceRecipe",
    "MilesTrainerSession",
    "MilesTransferCoordinator",
    "SglangGeneratorSession",
    "SglangLoader",
    "SglangParameterBinding",
    "inventory_miles_native_tensors",
]
