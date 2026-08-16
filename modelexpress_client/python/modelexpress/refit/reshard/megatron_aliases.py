# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for Megatron source aliases."""

from modelexpress_rl.train.engines.megatron.aliases import (
    MegatronAliasInput,
    MegatronTensorSpec,
    build_hf_aliases,
)

__all__ = ["MegatronAliasInput", "MegatronTensorSpec", "build_hf_aliases"]
