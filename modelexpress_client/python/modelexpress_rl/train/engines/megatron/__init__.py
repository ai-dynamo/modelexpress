# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Megatron integration for ModelExpress RL refit publication."""

from .adapter import MegatronTrainerAdapter
from .aliases import MegatronAliasInput, MegatronTensorSpec, build_hf_aliases
from .publisher import (
    MegatronReshardManifest,
    build_megatron_reshard_manifest,
)
from .selection import build_megatron_tensor_specs, model_express_role_for_mapping

__all__ = [
    "MegatronAliasInput",
    "MegatronReshardManifest",
    "MegatronTensorSpec",
    "MegatronTrainerAdapter",
    "build_hf_aliases",
    "build_megatron_tensor_specs",
    "build_megatron_reshard_manifest",
    "model_express_role_for_mapping",
]
