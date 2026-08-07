# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo RL adapter for BatchedDataDict and v0.7 PromptGroupRecord rollouts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from rlxfer.errors import MissingDependencyError
from rlxfer.model import ExperienceBatch, SampleIdentity, TensorPayload, Trajectory

from .base import (
    BaseAdapter,
    IncompatibleExperienceError,
    as_list,
    construct_record,
    is_sequence,
    native_fields,
    restore_native_value,
    safe_native_value,
)


class NemoRLAdapter(BaseAdapter):
    """Convert NeMo RL rollout containers without exposing NeMo transport internals."""

    framework_name: ClassVar[str] = "nemo_rl"
    distribution_name: ClassVar[str] = "nemo-rl"
    import_name: ClassVar[str] = "nemo_rl"
    extra_name: ClassVar[str] = "nemo-rl"

    def from_framework(self, native: Any) -> ExperienceBatch:
        if isinstance(native, tuple) and len(native) == 2 and isinstance(native[0], Mapping):
            return self._from_batched(native[0], metrics=native[1])
        if isinstance(native, Mapping):
            return self._from_batched(native)
        if _is_prompt_group(native):
            return self._from_prompt_groups([native], single=True)
        if is_sequence(native):
            records = list(native)
            if all(_is_prompt_group(record) for record in records):
                return self._from_prompt_groups(records, single=False)
        raise TypeError(
            "NeMo RL adapter expects BatchedDataDict, (BatchedDataDict, metrics), "
            "PromptGroupRecord, or a PromptGroupRecord sequence"
        )

    def to_framework(self, batch: ExperienceBatch) -> Any:
        self.validate_compatible(batch)
        extension = batch.extensions[self.framework_name]
        if not isinstance(extension, Mapping):
            raise IncompatibleExperienceError("extensions['nemo_rl'] must be a mapping")
        kind = extension.get("kind")
        if kind == "batched_data_dict":
            module = self._require("nemo_rl.distributed.batched_data_dict")
            fields = restore_native_value(extension.get("fields"))
            if not isinstance(fields, Mapping):
                raise IncompatibleExperienceError("NeMo BatchedDataDict fields are malformed")
            return module.BatchedDataDict(dict(fields))
        if kind == "prompt_group_records":
            try:
                interfaces = self._require("nemo_rl.experience.interfaces")
            except MissingDependencyError as error:
                raise IncompatibleExperienceError(
                    "PromptGroupRecord reconstruction requires NeMo RL v0.7 or newer; "
                    "v0.5 supports BatchedDataDict rollouts only"
                ) from error
            records_value = restore_native_value(extension.get("records"))
            if not isinstance(records_value, list):
                raise IncompatibleExperienceError("NeMo prompt-group records are malformed")
            records = []
            for record_value in records_value:
                if not isinstance(record_value, Mapping):
                    raise IncompatibleExperienceError("NeMo prompt-group record is malformed")
                state = dict(record_value)
                completions_value = state.get("completions")
                if not isinstance(completions_value, list):
                    raise IncompatibleExperienceError("NeMo completions must be a list")
                state["completions"] = [
                    construct_record(interfaces.Completion, completion)
                    for completion in completions_value
                    if isinstance(completion, Mapping)
                ]
                if len(state["completions"]) != len(completions_value):
                    raise IncompatibleExperienceError("NeMo completion record is malformed")
                records.append(construct_record(interfaces.PromptGroupRecord, state))
            return records[0] if extension.get("single") else records
        raise IncompatibleExperienceError(f"unsupported NeMo RL native kind {kind!r}")

    def validate_compatible(self, batch: ExperienceBatch) -> None:
        self._validate_batch(batch)
        extension = batch.extensions.get(self.framework_name)
        if not isinstance(extension, Mapping):
            raise IncompatibleExperienceError(
                "NeMo RL reconstruction requires extensions['nemo_rl']"
            )
        kind = extension.get("kind")
        if kind == "batched_data_dict":
            fields = restore_native_value(extension.get("fields"))
            if not isinstance(fields, Mapping):
                raise IncompatibleExperienceError("NeMo BatchedDataDict fields are malformed")
            for ids_name, lengths_name in (
                ("input_ids", "input_lengths"),
                ("output_ids", "unpadded_sequence_lengths"),
            ):
                if ids_name in fields and lengths_name not in fields:
                    raise IncompatibleExperienceError(
                        f"NeMo field {ids_name!r} requires {lengths_name!r}"
                    )
            return
        if kind == "prompt_group_records":
            records = restore_native_value(extension.get("records"))
            if not isinstance(records, list):
                raise IncompatibleExperienceError("NeMo prompt-group records must be a list")
            for record in records:
                if not isinstance(record, Mapping) or not isinstance(
                    record.get("completions"), list
                ):
                    raise IncompatibleExperienceError(
                        "each NeMo prompt group must contain a completions list"
                    )
            return
        raise IncompatibleExperienceError(f"unsupported NeMo RL native kind {kind!r}")

    def _from_batched(self, native: Mapping[str, Any], metrics: Any = None) -> ExperienceBatch:
        state = native_fields(native)
        tensors = {
            name: TensorPayload(value, name=name)
            for name, value in state.items()
            if _is_tensor(value)
        }
        batch = self._batch(
            trajectories=tuple(_batched_trajectories(state)),
            tensors=tensors,
            payload={"native_kind": "batched_data_dict"},
            extensions={
                self.framework_name: {
                    "kind": "batched_data_dict",
                    "fields": safe_native_value(state),
                    "metrics": safe_native_value(metrics),
                }
            },
        )
        self.validate_compatible(batch)
        return batch

    def _from_prompt_groups(self, records: list[Any], *, single: bool) -> ExperienceBatch:
        trajectories: list[Trajectory] = []
        layout: list[list[int]] = []
        safe_records: list[Any] = []
        for record in records:
            state = native_fields(record)
            completions = state.get("completions")
            if not is_sequence(completions):
                raise TypeError("PromptGroupRecord.completions must be a sequence")
            indexes: list[int] = []
            for completion_index, completion in enumerate(completions):
                indexes.append(len(trajectories))
                trajectories.append(
                    _completion_trajectory(
                        completion,
                        prompt_index=state.get("prompt_idx"),
                        completion_index=completion_index,
                    )
                )
            layout.append(indexes)
            safe_records.append(safe_native_value(state))

        batch = self._batch(
            trajectories=tuple(trajectories),
            payload={"native_kind": "prompt_group_records"},
            extensions={
                self.framework_name: {
                    "kind": "prompt_group_records",
                    "single": single,
                    "groups": layout,
                    "records": safe_records,
                }
            },
        )
        self.validate_compatible(batch)
        return batch


def _is_prompt_group(value: Any) -> bool:
    try:
        return "completions" in native_fields(value) and "prompt_idx" in native_fields(value)
    except TypeError:
        return False


def _is_tensor(value: Any) -> bool:
    return isinstance(value, np.ndarray) or (
        hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "stride")
    )


def _batched_trajectories(state: Mapping[str, Any]) -> list[Trajectory]:
    ids_name, lengths_name = _batch_pair(state)
    if ids_name is None or lengths_name is None:
        return []
    ids = state[ids_name]
    lengths = as_list(state[lengths_name], field=lengths_name)
    try:
        row_count = len(ids)
    except TypeError as error:
        raise TypeError(f"{ids_name} must be a two-dimensional batch") from error
    if len(lengths) != row_count:
        raise ValueError(
            f"{lengths_name} length {len(lengths)} must equal {ids_name} rows {row_count}"
        )
    trajectories = []
    for index, length_value in enumerate(lengths):
        if isinstance(length_value, (bool, float, complex)) or not hasattr(
            length_value, "__index__"
        ):
            raise ValueError(f"{lengths_name}[{index}] must be an integer")
        length = int(length_value)
        if length < 0:
            raise ValueError(f"{lengths_name}[{index}] must be non-negative")
        token_row = _slice_row(ids, index, length)
        kwargs: dict[str, Any] = {
            "identity": SampleIdentity(producer_id="nemo_rl-adapter", sequence_number=index),
            "tokens": TensorPayload(token_row, name="tokens"),
            "extensions": {"nemo_rl": {"batch_index": index}},
        }
        for source, destination in (
            ("token_mask", "attention_mask"),
            ("generation_logprobs", "log_probs"),
            ("prev_logprobs", "log_probs"),
            ("reference_policy_logprobs", "reference_log_probs"),
            ("values", "values"),
            ("advantages", "advantages"),
            ("returns", "returns"),
        ):
            if source in state and destination not in kwargs:
                kwargs[destination] = TensorPayload(
                    _slice_row(state[source], index, length), name=destination
                )
        trajectories.append(Trajectory(**kwargs))
    return trajectories


def _batch_pair(state: Mapping[str, Any]) -> tuple[str | None, str | None]:
    if "input_ids" in state and "input_lengths" in state:
        return "input_ids", "input_lengths"
    if "output_ids" in state and "unpadded_sequence_lengths" in state:
        return "output_ids", "unpadded_sequence_lengths"
    return None, None


def _slice_row(value: Any, index: int, length: int) -> Any:
    try:
        row = value[index]
        width = len(row)
    except (IndexError, TypeError) as error:
        raise ValueError("batched tensor must contain indexable one-dimensional rows") from error
    if length > width:
        raise ValueError(f"sequence length {length} exceeds padded row width {width}")
    return row[:length]


def _completion_trajectory(native: Any, *, prompt_index: Any, completion_index: int) -> Trajectory:
    state = native_fields(native)
    messages = state.get("message_log")
    tokens: list[Any] = []
    fallback_tokens: list[Any] = []
    log_probs: list[Any] = []
    response_parts: list[str] = []
    if is_sequence(messages):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            token_ids = message.get("token_ids")
            generation_logprobs = message.get("generation_logprobs")
            if generation_logprobs is not None:
                generated = as_list(generation_logprobs, field="generation_logprobs")
                message_tokens = as_list(token_ids, field="token_ids")
                if len(message_tokens) != len(generated):
                    raise ValueError(
                        "NeMo completion token_ids and generation_logprobs must have equal length"
                    )
                tokens.extend(message_tokens)
                log_probs.extend(generated)
            elif token_ids is not None and message.get("role") == "assistant":
                fallback_tokens.extend(as_list(token_ids, field="token_ids"))
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                response_parts.append(message["content"])
    reward = state.get("reward")
    if not log_probs:
        tokens = fallback_tokens
    return Trajectory(
        identity=SampleIdentity(
            request_id=str(prompt_index) if prompt_index is not None else None,
            producer_id="nemo_rl-adapter",
            sequence_number=completion_index,
        ),
        response="".join(response_parts) or None,
        tokens=TensorPayload(np.asarray(tokens), name="tokens") if tokens else None,
        rewards={"reward": float(reward)} if isinstance(reward, (int, float)) else {},
        log_probs=TensorPayload(np.asarray(log_probs), name="log_probs") if log_probs else None,
        terminal=not bool(state.get("truncated", False)),
        truncated=bool(state.get("truncated", False)),
        extensions={"nemo_rl": safe_native_value(state)},
    )
