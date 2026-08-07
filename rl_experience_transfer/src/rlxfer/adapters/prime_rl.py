# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PRIME-RL adapter for TrainingBatch/TrainingSample at commit 2873bf2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np

from rlxfer.model import ExperienceBatch, SampleIdentity, TensorPayload, Trajectory

from .base import (
    BaseAdapter,
    IncompatibleExperienceError,
    as_list,
    canonical_rewards,
    construct_record,
    native_fields,
    restore_native_value,
    safe_native_value,
)


class PrimeRLAdapter(BaseAdapter):
    """Convert PRIME-RL ``TrainingBatch`` objects without importing PRIME at startup."""

    framework_name: ClassVar[str] = "prime_rl"
    distribution_name: ClassVar[str] = "prime-rl"
    import_name: ClassVar[str] = "prime_rl"
    extra_name: ClassVar[str] = "prime-rl"

    def from_framework(self, native: Any) -> ExperienceBatch:
        batch_state = native_fields(native)
        examples = batch_state.get("examples")
        if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
            raise TypeError("PRIME-RL TrainingBatch.examples must be a sequence")

        trajectories = tuple(
            self._to_trajectory(example, index=index) for index, example in enumerate(examples)
        )
        batch = self._batch(
            trajectories=trajectories,
            extensions={
                self.framework_name: {
                    "step": batch_state.get("step"),
                    "run_idx": batch_state.get("run_idx"),
                }
            },
        )
        self.validate_compatible(batch)
        return batch

    def to_framework(self, batch: ExperienceBatch) -> Any:
        self.validate_compatible(batch)
        types = self._require("prime_rl.transport.types")
        sample_type = types.TrainingSample
        examples = []
        for trajectory in batch.trajectories:
            extension = trajectory.extensions.get(self.framework_name)
            if not isinstance(extension, Mapping):
                raise IncompatibleExperienceError(
                    "PRIME-RL reconstruction needs extensions['prime_rl'] for each trajectory; "
                    "prompt/completion boundaries are otherwise ambiguous"
                )
            state = restore_native_value(extension)
            if not isinstance(state, Mapping):
                raise IncompatibleExperienceError("invalid PRIME-RL native sample extension")
            state = dict(state)
            routed = state.get("routed_experts")
            if isinstance(routed, Mapping):
                state["routed_experts"] = construct_record(types.RoutedExperts, routed)
            mm_kwargs = state.get("mm_kwargs")
            if isinstance(mm_kwargs, Mapping):
                state["mm_kwargs"] = {
                    str(key): construct_record(types.EncodedTensor, value)
                    if isinstance(value, Mapping)
                    else value
                    for key, value in mm_kwargs.items()
                }
            examples.append(construct_record(sample_type, state))

        batch_extension = batch.extensions.get(self.framework_name, {})
        if not isinstance(batch_extension, Mapping):
            raise IncompatibleExperienceError("extensions['prime_rl'] must be a mapping")
        return types.TrainingBatch(
            examples=examples,
            step=int(batch_extension.get("step", 0)),
            run_idx=batch_extension.get("run_idx"),
        )

    def validate_compatible(self, batch: ExperienceBatch) -> None:
        self._validate_batch(batch)
        for index, trajectory in enumerate(batch.trajectories):
            extension = trajectory.extensions.get(self.framework_name)
            if not isinstance(extension, Mapping):
                raise IncompatibleExperienceError(
                    f"trajectory {index} lacks PRIME-RL prompt/completion metadata"
                )
            required = {
                "prompt_ids",
                "prompt_mask",
                "completion_ids",
                "completion_mask",
                "completion_logprobs",
                "completion_temperatures",
                "env_name",
            }
            missing = sorted(required.difference(extension))
            if missing:
                raise IncompatibleExperienceError(
                    f"trajectory {index} is missing PRIME-RL fields: {', '.join(missing)}"
                )
            prompt_ids = as_list(extension["prompt_ids"], field="prompt_ids")
            prompt_mask = as_list(extension["prompt_mask"], field="prompt_mask")
            if len(prompt_mask) != len(prompt_ids):
                raise IncompatibleExperienceError(
                    f"trajectory {index} prompt_mask length {len(prompt_mask)} does not match "
                    f"prompt_ids length {len(prompt_ids)}"
                )
            completion_ids = as_list(extension["completion_ids"], field="completion_ids")
            for field in ("completion_mask", "completion_logprobs", "completion_temperatures"):
                values = as_list(extension[field], field=field)
                if len(values) != len(completion_ids):
                    raise IncompatibleExperienceError(
                        f"trajectory {index} {field} length {len(values)} does not match "
                        f"completion_ids length {len(completion_ids)}"
                    )
            teacher = extension.get("teacher_logprobs")
            if teacher is not None:
                teacher_length = len(as_list(teacher, field="teacher_logprobs"))
                expected_lengths = (len(completion_ids), len(prompt_ids) + len(completion_ids))
                if teacher_length not in expected_lengths:
                    raise IncompatibleExperienceError(
                        f"trajectory {index} teacher_logprobs must align with the completion "
                        "or full sequence"
                    )

    def _to_trajectory(self, native: Any, *, index: int) -> Trajectory:
        state = native_fields(native)
        as_list(state.get("prompt_ids"), field="prompt_ids")
        completion_ids = as_list(state.get("completion_ids"), field="completion_ids")
        completion_logprobs = as_list(state.get("completion_logprobs"), field="completion_logprobs")
        return Trajectory(
            identity=SampleIdentity(producer_id="prime_rl-adapter", sequence_number=index),
            tokens=TensorPayload(np.asarray(completion_ids), name="tokens"),
            rewards=canonical_rewards(state.get("reward")),
            log_probs=TensorPayload(np.asarray(completion_logprobs), name="log_probs")
            if completion_logprobs
            else None,
            advantages=TensorPayload(
                np.full(len(completion_ids), float(state["advantage"])), name="advantages"
            )
            if state.get("advantage") is not None
            else None,
            extensions={self.framework_name: safe_native_value(state)},
        )
