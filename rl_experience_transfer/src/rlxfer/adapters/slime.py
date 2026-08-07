# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""slime v0.3.1 rollout adapter."""

from typing import ClassVar

from .base import GroupedSampleAdapter


class SlimeAdapter(GroupedSampleAdapter):
    """Convert slime ``Sample`` groups and ``RolloutFnTrainOutput`` objects."""

    framework_name: ClassVar[str] = "slime"
    distribution_name: ClassVar[str] = "slime"
    import_name: ClassVar[str] = "slime"
    extra_name: ClassVar[str] = "slime"
    sample_module: ClassVar[str] = "slime.utils.types"
    rollout_module: ClassVar[str] = "slime.rollout.base_types"
