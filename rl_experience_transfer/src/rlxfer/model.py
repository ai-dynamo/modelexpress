# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport- and framework-independent reinforcement-learning experience model."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from time import time
from typing import Any, NoReturn, SupportsInt, TypeVar, cast
from uuid import UUID, uuid4

from .errors import SchemaValidationError

SCHEMA_VERSION = "1.0"
_NAMESPACE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*$")
_ValidatedT = TypeVar("_ValidatedT")
_INTEGER_DTYPES = {"uint8", "int8", "int16", "int32", "int64", "long"}


@dataclass(slots=True)
class TensorPayload:
    """A tensor/array plus the metadata needed to reconstruct it."""

    data: object
    name: str = ""
    shape: tuple[int, ...] = ()
    dtype: str = ""
    device: str = ""
    stride: tuple[int, ...] | None = None
    layout: str = "strided"
    nbytes: int | None = None

    def __post_init__(self) -> None:
        actual_shape = _shape_of(self.data)
        if not self.shape and actual_shape is not None:
            self.shape = actual_shape
        if not self.dtype:
            self.dtype = _dtype_of(self.data) or ""
        if not self.device:
            self.device = str(getattr(self.data, "device", "cpu"))
        if self.stride is None:
            self.stride = _stride_of(self.data)
        if self.nbytes is None:
            self.nbytes = _nbytes_of(self.data)

    def validate(self, *, field_path: str = "tensor") -> None:
        """Validate metadata against the attached tensor-like object."""

        if not isinstance(self.dtype, str) or not self.dtype:
            _invalid(field_path + ".dtype", "non-empty dtype", self.dtype)
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in self.shape
        ):
            _invalid(field_path + ".shape", "non-negative integer dimensions", self.shape)
        actual_shape = _shape_of(self.data)
        if actual_shape is None:
            _invalid(field_path + ".data", "tensor or NumPy-compatible array", type(self.data))
        if actual_shape != self.shape:
            _invalid(field_path + ".shape", actual_shape, self.shape)
        actual_dtype = _dtype_of(self.data)
        if not isinstance(self.device, str) or not self.device:
            _invalid(field_path + ".device", "non-empty device", self.device)
        if not isinstance(self.layout, str) or not self.layout:
            _invalid(field_path + ".layout", "non-empty layout", self.layout)
        if self.stride is not None and (
            len(self.stride) != len(self.shape)
            or any(isinstance(size, bool) or not isinstance(size, int) for size in self.stride)
        ):
            _invalid(field_path + ".stride", f"{len(self.shape)} integer dimensions", self.stride)
        if self.nbytes is not None and (
            isinstance(self.nbytes, bool) or not isinstance(self.nbytes, int) or self.nbytes < 0
        ):
            _invalid(field_path + ".nbytes", "non-negative integer", self.nbytes)
        comparisons = (
            ("dtype", actual_dtype, _normalize_dtype(self.dtype), actual_dtype is not None),
            ("device", _device_of(self.data), self.device, True),
            ("layout", _layout_of(self.data), self.layout, True),
            ("stride", _stride_of(self.data), self.stride, self.stride is not None),
            ("nbytes", _nbytes_of(self.data), self.nbytes, self.nbytes is not None),
        )
        for name, actual, declared, required in comparisons:
            if (
                required
                and declared != actual
                and (actual is not None or name in {"stride", "nbytes"})
            ):
                _invalid(f"{field_path}.{name}", actual, getattr(self, name))


@dataclass(frozen=True, slots=True)
class PolicyVersion:
    """Identity of the policy that generated an experience."""

    version: str | int
    policy_id: str = "default"
    model_id: str | None = None

    def validate(self, *, field_path: str = "policy_version") -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id:
            _invalid(field_path + ".policy_id", "non-empty string", self.policy_id)
        if isinstance(self.version, bool) or not isinstance(self.version, (str, int)):
            _invalid(field_path + ".version", "string or non-negative integer", self.version)
        if isinstance(self.version, str) and not self.version:
            _invalid(field_path + ".version", "non-empty string", self.version)
        if isinstance(self.version, int) and self.version < 0:
            _invalid(field_path + ".version", "non-negative integer", self.version)
        if self.model_id is not None and (not isinstance(self.model_id, str) or not self.model_id):
            _invalid(field_path + ".model_id", "non-empty string or null", self.model_id)


@dataclass(frozen=True, slots=True)
class SampleIdentity:
    """Stable identifiers used for tracing and idempotent consumption."""

    sample_id: str = field(default_factory=lambda: str(uuid4()))
    request_id: str | None = None
    producer_id: str | None = None
    sequence_number: int | None = None
    idempotency_key: str | None = None

    def validate(self, *, field_path: str = "identity") -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            _invalid(field_path + ".sample_id", "non-empty string", self.sample_id)
        _validate_optional_non_negative_int(self, "sequence_number", field_path)
        _validate_optional_strings(
            self, ("request_id", "producer_id", "idempotency_key"), field_path
        )


@dataclass(slots=True)
class Transition:
    """One environment transition; values may be JSON data or tensor payloads."""

    observation: object | None = None
    action: object | None = None
    reward: float | None = None
    next_observation: object | None = None
    terminal: bool = False
    truncated: bool = False
    log_probability: float | None = None
    reference_log_probability: float | None = None
    value: float | None = None
    advantage: float | None = None
    return_value: float | None = None
    extensions: dict[str, object] = field(default_factory=dict)

    def validate(self, *, field_path: str = "transition") -> None:
        _validate_completion_flags(self, field_path)
        _validate_optional_finite_numbers(
            self,
            (
                "reward",
                "log_probability",
                "reference_log_probability",
                "value",
                "advantage",
                "return_value",
            ),
            field_path,
        )
        for name in ("observation", "action", "next_observation"):
            _validate_nested(getattr(self, name), f"{field_path}.{name}")
        _validate_extensions(self.extensions, field_path + ".extensions")


@dataclass(slots=True)
class Trajectory:
    """A variable-length rollout trajectory with token-aligned tensor fields."""

    identity: SampleIdentity | None = None
    policy_version: PolicyVersion | None = None
    transitions: tuple[Transition, ...] = ()
    prompt: str | None = None
    response: str | None = None
    tokens: TensorPayload | None = None
    attention_mask: TensorPayload | None = None
    rewards: dict[str, float | TensorPayload] = field(default_factory=dict)
    per_token_rewards: TensorPayload | None = None
    log_probs: TensorPayload | None = None
    reference_log_probs: TensorPayload | None = None
    values: TensorPayload | None = None
    advantages: TensorPayload | None = None
    returns: TensorPayload | None = None
    terminal: bool = False
    truncated: bool = False
    generation_metadata: dict[str, object] = field(default_factory=dict)
    extensions: dict[str, object] = field(default_factory=dict)

    def validate(self, *, field_path: str = "trajectory") -> None:
        if self.identity is not None:
            _validated(self.identity, SampleIdentity, field_path + ".identity", nullable=True)
        if self.policy_version is not None:
            _validated(
                self.policy_version,
                PolicyVersion,
                field_path + ".policy_version",
                nullable=True,
            )
        _validate_completion_flags(self, field_path)
        _validate_optional_strings(self, ("prompt", "response"), field_path, non_empty=False)
        for index, transition in enumerate(self.transitions):
            path = f"{field_path}.transitions[{index}]"
            _validated(transition, Transition, path)
        if self.attention_mask is not None and self.tokens is None:
            _invalid(field_path + ".attention_mask", "tokens to define alignment", "tokens missing")
        token_shape: tuple[int, ...] | None = None
        if self.tokens is not None:
            _validated(self.tokens, TensorPayload, field_path + ".tokens", nullable=True)
            token_shape = self.tokens.shape
            if _normalize_dtype(self.tokens.dtype) not in _INTEGER_DTYPES:
                _invalid(field_path + ".tokens.dtype", "integer token IDs", self.tokens.dtype)
        aligned = {
            "attention_mask": self.attention_mask,
            "per_token_rewards": self.per_token_rewards,
            "log_probs": self.log_probs,
            "reference_log_probs": self.reference_log_probs,
            "values": self.values,
            "advantages": self.advantages,
            "returns": self.returns,
        }
        for name, tensor in aligned.items():
            if tensor is None:
                continue
            tensor = _validated(tensor, TensorPayload, f"{field_path}.{name}", nullable=True)
            if token_shape is None:
                _invalid(f"{field_path}.{name}", "tokens to define alignment", "tokens missing")
            if tensor.shape != token_shape:
                _invalid(f"{field_path}.{name}.shape", token_shape, tensor.shape)
            if name == "attention_mask" and _normalize_dtype(tensor.dtype) not in (
                _INTEGER_DTYPES | {"bool"}
            ):
                _invalid(
                    field_path + ".attention_mask.dtype",
                    "boolean or integer mask",
                    tensor.dtype,
                )
        for name, reward in self.rewards.items():
            if not isinstance(name, str) or not name:
                _invalid(field_path + ".rewards", "non-empty reward names", name)
            if isinstance(reward, TensorPayload):
                reward.validate(field_path=f"{field_path}.rewards[{name!r}]")
            elif not _is_finite_number(reward):
                _invalid(
                    f"{field_path}.rewards[{name!r}]", "finite number or TensorPayload", reward
                )
        _validate_nested(self.generation_metadata, field_path + ".generation_metadata")
        _validate_extensions(self.extensions, field_path + ".extensions")


@dataclass(slots=True)
class Episode:
    """One or more trajectories belonging to the same environment episode."""

    trajectories: tuple[Trajectory, ...] = ()
    episode_id: str = field(default_factory=lambda: str(uuid4()))
    terminal: bool = False
    truncated: bool = False
    rewards: dict[str, float] = field(default_factory=dict)
    extensions: dict[str, object] = field(default_factory=dict)

    def validate(self, *, field_path: str = "episode") -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            _invalid(field_path + ".episode_id", "non-empty string", self.episode_id)
        if not self.trajectories:
            _invalid(field_path + ".trajectories", "at least one trajectory", self.trajectories)
        _validate_completion_flags(self, field_path)
        for index, trajectory in enumerate(self.trajectories):
            path = f"{field_path}.trajectories[{index}]"
            _validated(trajectory, Trajectory, path)
        for name, reward in self.rewards.items():
            if not isinstance(name, str) or not name or not _is_finite_number(reward):
                _invalid(f"{field_path}.rewards[{name!r}]", "named finite reward", reward)
        _validate_extensions(self.extensions, field_path + ".extensions")


@dataclass(slots=True)
class ExperienceMetadata:
    """Routing, provenance, and semantic compatibility metadata for a batch."""

    producer_id: str
    producer_framework: str
    producer_framework_version: str
    experience_id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: str = SCHEMA_VERSION
    created_at: float = field(default_factory=time)
    sequence_number: int | None = None
    idempotency_key: str | None = None
    policy_version: PolicyVersion | None = None
    algorithm: str | None = None
    tokenizer_id: str | None = None
    model_id: str | None = None
    reward_definition: str | None = None
    sequence_format: str | None = None
    padding: str | None = None
    chat_template: str | None = None
    truncation: str | None = None
    requires_reference_log_probs: bool = False
    generation: dict[str, object] = field(default_factory=dict)

    def validate(
        self,
        *,
        consumer_framework: str | None = None,
        consumer_framework_version: str | None = None,
    ) -> None:
        context = self.validation_context(consumer_framework, consumer_framework_version)
        try:
            UUID(self.experience_id)
        except (ValueError, AttributeError, TypeError):
            _invalid("metadata.experience_id", "UUID string", self.experience_id, **context)
        if self.schema_version != SCHEMA_VERSION:
            _invalid("metadata.schema_version", SCHEMA_VERSION, self.schema_version, **context)
        for name in ("producer_id", "producer_framework", "producer_framework_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                _invalid(f"metadata.{name}", "non-empty string", value, **context)
        if not _is_finite_number(self.created_at) or self.created_at < 0:
            _invalid(
                "metadata.created_at", "non-negative Unix timestamp", self.created_at, **context
            )
        _validate_optional_non_negative_int(self, "sequence_number", "metadata", context=context)
        _validate_optional_strings(self, ("idempotency_key",), "metadata", context=context)
        if self.policy_version is not None:
            _validated(
                self.policy_version,
                PolicyVersion,
                "metadata.policy_version",
                nullable=True,
                context=context,
            )
        _validate_optional_strings(
            self,
            (
                "algorithm",
                "tokenizer_id",
                "model_id",
                "reward_definition",
                "sequence_format",
                "padding",
                "chat_template",
                "truncation",
            ),
            "metadata",
            context=context,
        )
        if not isinstance(self.requires_reference_log_probs, bool):
            _invalid(
                "metadata.requires_reference_log_probs",
                "boolean",
                self.requires_reference_log_probs,
                **context,
            )
        _validate_nested(self.generation, "metadata.generation", **context)

    def validation_context(
        self,
        consumer_framework: str | None = None,
        consumer_framework_version: str | None = None,
    ) -> dict[str, str | None]:
        """Return error context suitable for :class:`SchemaValidationError`."""

        return {
            "producer_framework": self.producer_framework,
            "producer_framework_version": self.producer_framework_version,
            "consumer_framework": consumer_framework,
            "consumer_framework_version": consumer_framework_version,
            "experience_id": self.experience_id,
        }


@dataclass(slots=True)
class ExperienceBatch:
    """Canonical transfer unit shared by producers, consumers, and adapters."""

    metadata: ExperienceMetadata
    trajectories: tuple[Trajectory, ...] = ()
    episodes: tuple[Episode, ...] = ()
    tensors: dict[str, TensorPayload] = field(default_factory=dict)
    payload: dict[str, object] = field(default_factory=dict)
    extensions: dict[str, object] = field(default_factory=dict)

    @property
    def experience_id(self) -> str:
        """Return the globally unique ID carried by this batch."""

        return self.metadata.experience_id

    def validate(
        self,
        *,
        consumer_framework: str | None = None,
        consumer_framework_version: str | None = None,
    ) -> None:
        """Validate structure before serialization or expensive transfer work."""

        if not isinstance(self.metadata, ExperienceMetadata):
            _invalid("metadata", "ExperienceMetadata", type(self.metadata))
        self.metadata.validate(
            consumer_framework=consumer_framework,
            consumer_framework_version=consumer_framework_version,
        )
        context = self.metadata.validation_context(consumer_framework, consumer_framework_version)
        if not (self.trajectories or self.episodes or self.tensors or self.payload):
            _invalid(
                "batch",
                "at least one trajectory, episode, tensor, or payload value",
                "empty",
                **context,
            )
        sample_ids: set[str] = set()
        for index, trajectory in enumerate(self.trajectories):
            _validated(trajectory, Trajectory, f"trajectories[{index}]", context=context)
            _record_sample_id(trajectory, f"trajectories[{index}]", sample_ids, context)
        for index, episode in enumerate(self.episodes):
            _validated(episode, Episode, f"episodes[{index}]", context=context)
            for trajectory in episode.trajectories:
                _record_sample_id(
                    trajectory, f"episodes[{index}].trajectories", sample_ids, context
                )
        for name, tensor in self.tensors.items():
            if not isinstance(name, str) or not name:
                _invalid("tensors", "non-empty tensor names", name, **context)
            _validated(tensor, TensorPayload, f"tensors[{name!r}]", context=context)
        _validate_nested(self.payload, "payload", **context)
        _validate_extensions(self.extensions, "extensions", **context)


@dataclass(slots=True)
class TransferDescriptor:
    """Transport-neutral catalog of externally transferred tensor buffers."""

    experience_id: str
    transfer_id: str = field(default_factory=lambda: str(uuid4()))
    strategy: str = "external"
    tensor_names: tuple[str, ...] = ()
    byte_sizes: dict[str, int] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)
    metadata_bytes: int = 0
    extensions: dict[str, object] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        """Return metadata plus tensor bytes described by this transfer."""

        return self.metadata_bytes + sum(self.byte_sizes.values())

    def validate(self) -> None:
        for name, value in (
            ("experience_id", self.experience_id),
            ("transfer_id", self.transfer_id),
        ):
            try:
                UUID(value)
            except (ValueError, AttributeError, TypeError):
                _invalid(name, "UUID string", value, experience_id=self.experience_id)
        if not self.strategy:
            _invalid(
                "strategy", "non-empty string", self.strategy, experience_id=self.experience_id
            )
        if (
            isinstance(self.metadata_bytes, bool)
            or not isinstance(self.metadata_bytes, int)
            or self.metadata_bytes < 0
        ):
            _invalid(
                "metadata_bytes",
                "non-negative integer",
                self.metadata_bytes,
                experience_id=self.experience_id,
            )
        if len(set(self.tensor_names)) != len(self.tensor_names):
            _invalid(
                "tensor_names", "unique names", self.tensor_names, experience_id=self.experience_id
            )
        expected_names = set(self.tensor_names)
        if set(self.byte_sizes) != expected_names:
            _invalid(
                "byte_sizes", expected_names, set(self.byte_sizes), experience_id=self.experience_id
            )
        for name, size in self.byte_sizes.items():
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                _invalid(
                    f"byte_sizes[{name!r}]",
                    "non-negative integer",
                    size,
                    experience_id=self.experience_id,
                )
        if not set(self.checksums).issubset(expected_names):
            _invalid(
                "checksums",
                f"keys within {expected_names}",
                set(self.checksums),
                experience_id=self.experience_id,
            )
        _validate_extensions(self.extensions, "extensions", experience_id=self.experience_id)


def _validated(
    value: object,
    expected_type: type[_ValidatedT],
    field_path: str,
    *,
    nullable: bool = False,
    context: Mapping[str, str | None] | None = None,
) -> _ValidatedT:
    expected = expected_type.__name__ + (" or null" if nullable else "")
    if not isinstance(value, expected_type):
        _invalid(field_path, expected, type(value), **(context or {}))
    try:
        cast(Any, value).validate(field_path=field_path)
    except SchemaValidationError as error:
        if context:
            raise _with_context(error, context) from error
        raise
    return value


def _record_sample_id(
    trajectory: Trajectory,
    field_path: str,
    seen: set[str],
    context: Mapping[str, str | None],
) -> None:
    if trajectory.identity is None:
        return
    sample_id = trajectory.identity.sample_id
    if sample_id in seen:
        _invalid(
            field_path + ".identity.sample_id",
            "unique sample ID within batch",
            sample_id,
            **context,
        )
    seen.add(sample_id)


def _validate_extensions(
    extensions: Mapping[str, object],
    field_path: str,
    **context: str | None,
) -> None:
    if not isinstance(extensions, Mapping):
        _invalid(field_path, "mapping of extension namespaces", type(extensions), **context)
    for namespace, value in extensions.items():
        if not isinstance(namespace, str) or _NAMESPACE.fullmatch(namespace) is None:
            _invalid(
                f"{field_path}.<namespace>",
                "namespace matching [A-Za-z][A-Za-z0-9_.-]*",
                namespace,
                **context,
            )
        _validate_nested(value, f"{field_path}.{namespace}", **context)


def _validate_completion_flags(value: object, field_path: str) -> None:
    flags = tuple(getattr(value, name) for name in ("terminal", "truncated"))
    for name, flag in zip(("terminal", "truncated"), flags, strict=True):
        if not isinstance(flag, bool):
            _invalid(f"{field_path}.{name}", "boolean", flag)
    if all(flags):
        _invalid(field_path, "terminal and truncated not both true", flags)


def _validate_optional_finite_numbers(
    value: object,
    names: Iterable[str],
    field_path: str,
) -> None:
    for name in names:
        item = getattr(value, name)
        if item is not None and not _is_finite_number(item):
            _invalid(f"{field_path}.{name}", "finite number or null", item)


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _validate_optional_non_negative_int(
    value: object,
    name: str,
    field_path: str,
    *,
    context: Mapping[str, str | None] | None = None,
) -> None:
    item = getattr(value, name)
    if item is not None and (isinstance(item, bool) or not isinstance(item, int) or item < 0):
        _invalid(f"{field_path}.{name}", "non-negative integer", item, **(context or {}))


def _validate_optional_strings(
    value: object,
    names: Iterable[str],
    field_path: str,
    *,
    non_empty: bool = True,
    context: Mapping[str, str | None] | None = None,
) -> None:
    for name in names:
        item = getattr(value, name)
        if item is not None and (not isinstance(item, str) or (non_empty and not item)):
            expected = "non-empty string or null" if non_empty else "string or null"
            actual = item if non_empty else type(item).__name__
            _invalid(f"{field_path}.{name}", expected, actual, **(context or {}))


def _validate_nested(value: object, field_path: str, **context: str | None) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        _invalid(field_path, "finite JSON number", value, **context)
    if isinstance(value, TensorPayload):
        try:
            value.validate(field_path=field_path)
        except SchemaValidationError as error:
            raise _with_context(error, context) from error
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _invalid(field_path, "mapping with string keys", key, **context)
            _validate_nested(item, f"{field_path}.{key}", **context)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_nested(item, f"{field_path}[{index}]", **context)
        return
    _invalid(field_path, "JSON-safe value or TensorPayload", type(value).__name__, **context)


def _shape_of(value: object) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None or isinstance(shape, (str, bytes)) or not isinstance(shape, Iterable):
        return None
    try:
        return tuple(int(cast(SupportsInt, size)) for size in cast(Iterable[object], shape))
    except (TypeError, ValueError):
        return None


def _dtype_of(value: object) -> str | None:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else _normalize_dtype(str(dtype))


def _normalize_dtype(dtype: str) -> str:
    return dtype.removeprefix("torch.").removeprefix("numpy.")


def _device_of(value: object) -> str | None:
    if _shape_of(value) is None:
        return None
    return str(getattr(value, "device", "cpu"))


def _layout_of(value: object) -> str | None:
    layout = getattr(value, "layout", None)
    if layout is not None:
        return str(layout).removeprefix("torch.")
    if getattr(value, "strides", None) is not None:
        return "strided"
    return None


def _stride_of(value: object) -> tuple[int, ...] | None:
    stride: object = getattr(value, "stride", None)
    if callable(stride):
        stride = stride()
        return _integer_stride(stride)
    if stride is not None:
        return _integer_stride(stride)
    stride = getattr(value, "strides", None)
    if stride is None or isinstance(stride, (str, bytes)) or not isinstance(stride, Iterable):
        return None
    try:
        byte_stride = tuple(int(cast(SupportsInt, size)) for size in cast(Iterable[object], stride))
        itemsize = int(cast(SupportsInt, cast(Any, value).itemsize))
    except (TypeError, ValueError, AttributeError):
        return None
    if itemsize <= 0 or any(size % itemsize for size in byte_stride):
        return None
    return tuple(size // itemsize for size in byte_stride)


def _integer_stride(value: object) -> tuple[int, ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return None
    try:
        return tuple(int(cast(SupportsInt, size)) for size in cast(Iterable[object], value))
    except (TypeError, ValueError):
        return None


def _nbytes_of(value: object) -> int | None:
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    element_size = getattr(value, "element_size", None)
    numel = getattr(value, "numel", None)
    if callable(element_size) and callable(numel):
        return int(element_size()) * int(numel())
    return None


def _invalid(
    field: str,
    expected: object,
    actual: object,
    **context: str | None,
) -> NoReturn:
    raise SchemaValidationError(field=field, expected=expected, actual=actual, **context)


def _with_context(
    error: SchemaValidationError,
    context: Mapping[str, str | None],
) -> SchemaValidationError:
    return SchemaValidationError(
        field=error.field,
        expected=error.expected,
        actual=error.actual,
        producer_framework=context.get("producer_framework"),
        producer_framework_version=context.get("producer_framework_version"),
        consumer_framework=context.get("consumer_framework"),
        consumer_framework_version=context.get("consumer_framework_version"),
        experience_id=context.get("experience_id"),
    )
