# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small framework-adapter contract and shared optional-import helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from base64 import b64decode, b64encode
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from importlib import import_module, metadata
from typing import Any, ClassVar, Protocol, TypeGuard, cast, runtime_checkable

import numpy as np

from rlxfer.compatibility import CompatibilityRequirements, ensure_compatible
from rlxfer.errors import CompatibilityError, MissingDependencyError
from rlxfer.model import (
    ExperienceBatch,
    ExperienceMetadata,
    PolicyVersion,
    SampleIdentity,
    TensorPayload,
    Trajectory,
)

from .compat import SUPPORT, verify_framework_version


class IncompatibleExperienceError(CompatibilityError):
    """Raised when a batch cannot safely be consumed by an adapter."""


@runtime_checkable
class ExperienceAdapter(Protocol):
    """Transport-independent conversion contract for one RL framework."""

    framework_name: ClassVar[str]

    @property
    def framework_version(self) -> str:
        """Return the detected framework version or ``"unavailable"``."""
        ...

    @property
    def adapter_version(self) -> str:
        """Return the rlxfer adapter implementation version."""
        ...

    def from_framework(self, native: Any) -> ExperienceBatch:
        """Convert a native rollout batch to canonical experience."""
        ...

    def to_framework(self, batch: ExperienceBatch) -> Any:
        """Reconstruct the native training input."""
        ...

    def validate_compatible(self, batch: ExperienceBatch) -> None:
        """Reject experience that this adapter cannot consume safely."""
        ...


class BaseAdapter(ABC):
    """Common version detection and dependency diagnostics."""

    framework_name: ClassVar[str]
    distribution_name: ClassVar[str]
    import_name: ClassVar[str]
    extra_name: ClassVar[str]

    @property
    def adapter_version(self) -> str:
        """Return this package's adapter implementation version."""

        return SUPPORT[self.framework_name].adapter_version

    def _verify_framework_version(self) -> None:
        verify_framework_version(self.framework_name, self.framework_version)

    @property
    def framework_version(self) -> str:
        try:
            return metadata.version(self.distribution_name)
        except metadata.PackageNotFoundError:
            try:
                module = import_module(self.import_name)
            except ModuleNotFoundError:
                return "unavailable"
            return str(getattr(module, "__version__", "unknown"))

    def _require(self, module_name: str) -> Any:
        try:
            return import_module(module_name)
        except ModuleNotFoundError as error:
            message = (
                f"{self.framework_name} native conversion requires {self.distribution_name!r}; "
                f"install rl-experience-transfer[{self.extra_name}] in an isolated environment"
            )
            raise MissingDependencyError(message) from error

    def _batch(self, **values: Any) -> ExperienceBatch:
        return ExperienceBatch(
            metadata=ExperienceMetadata(
                producer_id=f"{self.framework_name}-adapter",
                producer_framework=self.framework_name,
                producer_framework_version=self.framework_version,
            ),
            **values,
        )

    def _validate_batch(self, batch: ExperienceBatch) -> None:
        self._verify_framework_version()
        batch.validate(
            consumer_framework=self.framework_name,
            consumer_framework_version=self.framework_version,
        )

    @abstractmethod
    def from_framework(self, native: Any) -> ExperienceBatch:
        """Convert native rollout data to a canonical batch."""

    @abstractmethod
    def to_framework(self, batch: ExperienceBatch) -> Any:
        """Reconstruct native training data."""

    @abstractmethod
    def validate_compatible(self, batch: ExperienceBatch) -> None:
        """Validate framework-specific training requirements."""


class GroupedSampleAdapter(BaseAdapter):
    """Shared adapter for the intentionally compatible slime/MILES rollout shape."""

    sample_module: ClassVar[str]
    rollout_module: ClassVar[str]
    compatible_namespaces: ClassVar[tuple[str, ...]] = ("slime", "miles")

    def __init__(
        self,
        cross_framework_requirements: CompatibilityRequirements | None = None,
    ) -> None:
        self.cross_framework_requirements = cross_framework_requirements

    def from_framework(self, native: Any) -> ExperienceBatch:
        if is_sequence(native):
            groups_value: Any = native
            metrics: Any = None
        else:
            output = native_fields(native)
            groups_value = output.get("samples")
            metrics = output.get("metrics")
        groups = _sample_groups(groups_value)

        trajectories: list[Trajectory] = []
        layout: list[list[int]] = []
        for group in groups:
            indexes: list[int] = []
            for sample in group:
                indexes.append(len(trajectories))
                trajectories.append(self._sample_to_trajectory(sample))
            layout.append(indexes)

        batch = self._batch(
            trajectories=tuple(trajectories),
            extensions={
                self.framework_name: {
                    "groups": layout,
                    "metrics": safe_native_value(metrics),
                }
            },
        )
        self.validate_compatible(batch)
        return batch

    def to_framework(self, batch: ExperienceBatch) -> Any:
        self.validate_compatible(batch)
        sample_type = self._require(self.sample_module).Sample
        output_type = self._require(self.rollout_module).RolloutFnTrainOutput
        samples = []
        for trajectory in batch.trajectories:
            extension = self._sample_extension(trajectory)
            state = restore_native_value(extension)
            if not isinstance(state, Mapping):
                raise IncompatibleExperienceError("native sample extension must be a mapping")
            samples.append(construct_record(sample_type, self._prepare_state(dict(state))))

        extension = self._batch_extension(batch)
        layout = extension.get("groups")
        if not isinstance(layout, list):
            raise IncompatibleExperienceError("grouped rollout extension lacks a groups list")
        groups = [[samples[int(index)] for index in group] for group in layout]
        metrics = restore_native_value(extension.get("metrics"))
        return output_type(samples=groups, metrics=metrics)

    def validate_compatible(self, batch: ExperienceBatch) -> None:
        self._validate_batch(batch)
        if batch.metadata.producer_framework != self.framework_name:
            requirements = self.cross_framework_requirements
            if requirements is None:
                raise IncompatibleExperienceError(
                    f"cross-framework {batch.metadata.producer_framework} -> "
                    f"{self.framework_name} conversion is unsafe without explicit consumer "
                    "requirements"
                )
            if requirements.consumer_framework != self.framework_name:
                raise IncompatibleExperienceError(
                    "cross-framework requirements target a different consumer framework"
                )
            ensure_compatible(batch, requirements)
        for index, trajectory in enumerate(batch.trajectories):
            state = restore_native_value(self._sample_extension(trajectory))
            if not isinstance(state, Mapping):
                raise IncompatibleExperienceError(f"trajectory {index} native state is malformed")
            tokens = as_list(state.get("tokens"), field="tokens")
            response_length = state.get("response_length", 0)
            if not isinstance(response_length, int) or not 0 <= response_length <= len(tokens):
                raise IncompatibleExperienceError(
                    f"trajectory {index} response_length {response_length!r} is invalid for "
                    f"{len(tokens)} tokens"
                )
            for field in ("loss_mask", "rollout_log_probs", "teacher_log_probs"):
                value = state.get(field)
                if value is not None and len(as_list(value, field=field)) != response_length:
                    raise IncompatibleExperienceError(
                        f"trajectory {index} {field} length must equal response_length"
                    )

        extension = self._batch_extension(batch)
        layout = extension.get("groups")
        if not isinstance(layout, list) or any(not isinstance(group, list) for group in layout):
            raise IncompatibleExperienceError("grouped rollout extension lacks valid group nesting")
        flattened = [int(index) for group in layout for index in group]
        if sorted(flattened) != list(range(len(batch.trajectories))):
            raise IncompatibleExperienceError(
                "group nesting must reference every trajectory exactly once"
            )

    def _sample_to_trajectory(self, native: Any) -> Trajectory:
        state = native_fields(native)
        tokens = as_list(state.get("tokens"), field="tokens")
        response_length = state.get("response_length", 0)
        if not isinstance(response_length, int) or not 0 <= response_length <= len(tokens):
            raise ValueError(
                f"response_length {response_length!r} is invalid for {len(tokens)} tokens"
            )
        log_probs_value = state.get("rollout_log_probs")
        log_probs = (
            as_list(log_probs_value, field="rollout_log_probs")
            if log_probs_value is not None
            else None
        )
        status_value = getattr(state.get("status"), "value", state.get("status"))
        rewards = canonical_rewards(state.get("reward"))
        index_value = state.get("index")
        rollout_id = state.get("rollout_id", index_value)
        versions = state.get("weight_versions")
        policy_version = None
        if isinstance(versions, Sequence) and not isinstance(versions, (str, bytes)) and versions:
            latest_version = versions[-1]
            if isinstance(latest_version, (str, int)) and not isinstance(latest_version, bool):
                policy_version = PolicyVersion(latest_version)
        sequence_number = (
            index_value
            if isinstance(index_value, int)
            and not isinstance(index_value, bool)
            and index_value >= 0
            else None
        )
        response_tokens = tokens[-response_length:] if response_length else []
        return Trajectory(
            identity=SampleIdentity(
                request_id=str(rollout_id) if rollout_id is not None else None,
                producer_id=f"{self.framework_name}-adapter",
                sequence_number=sequence_number,
            ),
            policy_version=policy_version,
            prompt=state.get("prompt") if isinstance(state.get("prompt"), str) else None,
            response=state.get("response") if isinstance(state.get("response"), str) else None,
            tokens=TensorPayload(np.asarray(response_tokens), name="tokens"),
            rewards=rewards,
            log_probs=TensorPayload(np.asarray(log_probs), name="log_probs")
            if log_probs is not None
            else None,
            terminal=status_value == "completed",
            truncated=status_value == "truncated",
            extensions={self.framework_name: safe_native_value(state)},
        )

    def _sample_extension(self, trajectory: Trajectory) -> Any:
        for namespace in (self.framework_name, *self.compatible_namespaces):
            extension = trajectory.extensions.get(namespace)
            if isinstance(extension, Mapping):
                return extension
        raise IncompatibleExperienceError(
            f"{self.framework_name} needs a slime/MILES native sample extension"
        )

    def _batch_extension(self, batch: ExperienceBatch) -> Mapping[str, Any]:
        for namespace in (self.framework_name, *self.compatible_namespaces):
            extension = batch.extensions.get(namespace)
            if isinstance(extension, Mapping):
                return extension
        raise IncompatibleExperienceError(
            f"{self.framework_name} needs preserved slime/MILES group nesting"
        )

    def _prepare_state(self, state: dict[str, Any]) -> dict[str, Any]:
        return state


def native_fields(value: Any) -> dict[str, Any]:
    """Return public constructor fields from a mapping, dataclass, or msgspec struct."""
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    struct_fields = getattr(type(value), "__struct_fields__", ())
    if struct_fields:
        return {str(name): getattr(value, name) for name in struct_fields}
    state = getattr(value, "__dict__", None)
    if isinstance(state, Mapping):
        return {str(key): item for key, item in state.items() if not str(key).startswith("_")}
    raise TypeError(f"expected a mapping or native record, got {type(value).__name__}")


def safe_native_value(value: Any) -> Any:
    """Remove framework class instances while preserving supported tensor/array leaves."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"$rlxfer.bytes": b64encode(value).decode("ascii")}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return TensorPayload(value)
    module = type(value).__module__.split(".", 1)[0]
    if module == "torch" and hasattr(value, "shape") and hasattr(value, "dtype"):
        return TensorPayload(value)
    if isinstance(value, Enum):
        return safe_native_value(value.value)
    if isinstance(value, Mapping):
        return {str(key): safe_native_value(item) for key, item in value.items()}
    if is_sequence(value):
        return [safe_native_value(item) for item in value]
    return {key: safe_native_value(item) for key, item in native_fields(value).items()}


def restore_native_value(value: Any) -> Any:
    """Restore tensor and byte leaves from a framework extension."""
    if isinstance(value, TensorPayload):
        return value.data
    if isinstance(value, Mapping):
        if set(value) == {"$rlxfer.bytes"}:
            encoded = value["$rlxfer.bytes"]
            if not isinstance(encoded, str):
                raise TypeError("invalid encoded native byte payload")
            return b64decode(encoded.encode("ascii"))
        return {str(key): restore_native_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [restore_native_value(item) for item in value]
    return value


def as_list(value: Any, *, field: str) -> list[Any]:
    """Normalize a one-dimensional tensor, array, or sequence without importing torch."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise TypeError(f"{field} must be one-dimensional, got shape {value.shape}")
        return cast(list[Any], value.tolist())
    if hasattr(value, "detach") and hasattr(value, "reshape"):
        shape = getattr(value, "shape", ())
        if len(shape) != 1:
            raise TypeError(f"{field} must be one-dimensional, got shape {tuple(shape)}")
        return list(value.detach().cpu().tolist())
    if is_sequence(value):
        result = list(value)
        if any(is_sequence(item) for item in result):
            raise TypeError(f"{field} must be one-dimensional")
        return result
    raise TypeError(f"{field} must be a one-dimensional tensor, array, or sequence")


def construct_record(record_type: type[Any], state: Mapping[str, Any]) -> Any:
    """Construct a native record, preferring its version-aware ``from_dict`` hook."""
    from_dict = getattr(record_type, "from_dict", None)
    if callable(from_dict):
        return from_dict(dict(state))
    return record_type(**dict(state))


def is_sequence(value: Any) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def canonical_rewards(value: Any) -> dict[str, float | TensorPayload]:
    if isinstance(value, (int, float)):
        return {"reward": float(value)}
    if isinstance(value, Mapping):
        return {
            str(key): float(item) for key, item in value.items() if isinstance(item, (int, float))
        }
    return {}


def _sample_groups(value: Any) -> list[list[Any]]:
    if not is_sequence(value):
        raise TypeError("RolloutFnTrainOutput.samples must be a sequence of groups")
    groups: list[list[Any]] = []
    for group in value:
        if not is_sequence(group):
            raise TypeError("each rollout sample group must be a sequence")
        groups.append(list(group))
    return groups
