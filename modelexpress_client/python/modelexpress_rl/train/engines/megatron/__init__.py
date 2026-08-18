# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Megatron integration for ModelExpress RL refit publication."""

from .adapter import MegatronTrainerAdapter
from .aliases import MegatronAliasInput, MegatronTensorSpec, build_hf_aliases
from .publisher import (
    MegatronPublishedTensorSpec,
    MegatronReshardManifest,
    build_megatron_reshard_manifest,
    publish_megatron_reshard_view,
    publish_registered_shard_table,
)

__all__ = [
    "MegatronAliasInput",
    "MegatronPublishedTensorSpec",
    "MegatronReshardManifest",
    "MegatronTensorSpec",
    "MegatronTrainerAdapter",
    "build_hf_aliases",
    "build_megatron_reshard_manifest",
    "publish_megatron_reshard_view",
    "publish_registered_shard_table",
]
