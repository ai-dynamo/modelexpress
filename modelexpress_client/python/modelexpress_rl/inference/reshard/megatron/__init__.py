# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-side planning for Megatron-native source tensors."""

from .layout import MegatronTargetLayout, MegatronTargetSpec, lower_megatron_target
from .receiver import MegatronReshardReceiver

__all__ = [
    "MegatronReshardReceiver",
    "MegatronTargetLayout",
    "MegatronTargetSpec",
    "lower_megatron_target",
]
