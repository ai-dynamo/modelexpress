# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MILES v0.2.1 rollout adapter."""

from typing import Any, ClassVar

from .base import GroupedSampleAdapter, construct_record


class MilesAdapter(GroupedSampleAdapter):
    """Convert MILES ``Sample`` groups and ``RolloutFnTrainOutput`` objects."""

    framework_name: ClassVar[str] = "miles"
    distribution_name: ClassVar[str] = "miles"
    import_name: ClassVar[str] = "miles"
    extra_name: ClassVar[str] = "miles"
    sample_module: ClassVar[str] = "miles.utils.types"
    rollout_module: ClassVar[str] = "miles.rollout.base_types"

    def _prepare_state(self, state: dict[str, Any]) -> dict[str, Any]:
        types = self._require(self.sample_module)
        if isinstance(state.get("adapter"), dict):
            state["adapter"] = construct_record(types.AdapterRef, state["adapter"])
        if isinstance(state.get("reward_spec"), dict):
            state["reward_spec"] = construct_record(types.RewardSpec, state["reward_spec"])
        return state
