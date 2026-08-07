# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Semantic compatibility checks between experience producers and consumers."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import CompatibilityError
from .model import ExperienceBatch, PolicyVersion


@dataclass(frozen=True, slots=True)
class CompatibilityRequirements:
    """Consumer semantics that must match before a transfer begins."""

    consumer_framework: str
    consumer_framework_version: str
    algorithm: str | None = None
    tokenizer_id: str | None = None
    model_id: str | None = None
    policy_version: PolicyVersion | None = None
    reward_definition: str | None = None
    sequence_format: str | None = None
    padding: str | None = None
    chat_template: str | None = None
    truncation: str | None = None
    requires_reference_log_probs: bool = False
    max_policy_lag: int | None = None

    def __post_init__(self) -> None:
        for name in ("consumer_framework", "consumer_framework_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.max_policy_lag is not None and (
            isinstance(self.max_policy_lag, bool)
            or not isinstance(self.max_policy_lag, int)
            or self.max_policy_lag < 0
        ):
            raise ValueError("max_policy_lag must be a non-negative integer or None")
        if self.policy_version is not None:
            self.policy_version.validate()
        if self.max_policy_lag is not None:
            if self.policy_version is None:
                raise ValueError("max_policy_lag requires policy_version")
            if not isinstance(self.policy_version.version, int):
                raise ValueError("max_policy_lag requires an integer policy version")
        if not isinstance(self.requires_reference_log_probs, bool):
            raise ValueError("requires_reference_log_probs must be a boolean")


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    """One actionable incompatibility."""

    field: str
    producer_value: object
    consumer_value: object
    reason: str
    action: str

    def __str__(self) -> str:
        return (
            f"{self.field}: {self.reason} "
            f"(producer={self.producer_value!r}, consumer={self.consumer_value!r}); "
            f"action: {self.action}"
        )


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Complete compatibility result; all detected problems are retained."""

    experience_id: str
    producer_framework: str
    producer_framework_version: str
    consumer_framework: str
    consumer_framework_version: str
    issues: tuple[CompatibilityIssue, ...] = ()

    @property
    def compatible(self) -> bool:
        """Whether the experience is safe for this consumer."""

        return not self.issues

    def raise_if_incompatible(self) -> None:
        """Raise an actionable error when any requirement is unmet."""

        if self.compatible:
            return
        detail = "; ".join(str(issue) for issue in self.issues)
        raise CompatibilityError(
            f"experience {self.experience_id} from "
            f"{self.producer_framework}@{self.producer_framework_version} is incompatible with "
            f"{self.consumer_framework}@{self.consumer_framework_version}: {detail}"
        )


def check_compatibility(
    batch: ExperienceBatch,
    requirements: CompatibilityRequirements,
) -> CompatibilityReport:
    """Validate schema and compare every declared consumer semantic requirement."""

    batch.validate(
        consumer_framework=requirements.consumer_framework,
        consumer_framework_version=requirements.consumer_framework_version,
    )
    metadata = batch.metadata
    issues: list[CompatibilityIssue] = []
    semantic_fields = (
        "algorithm",
        "tokenizer_id",
        "model_id",
        "reward_definition",
        "sequence_format",
        "padding",
        "chat_template",
        "truncation",
    )
    for field_name in semantic_fields:
        producer_value = getattr(metadata, field_name)
        consumer_value = getattr(requirements, field_name)
        issue = _semantic_issue(field_name, producer_value, consumer_value)
        if issue is not None:
            issues.append(issue)
    policy_issue = _policy_issue(metadata.policy_version, requirements)
    if policy_issue is not None:
        issues.append(policy_issue)
    trajectories = batch.trajectories + tuple(
        trajectory for episode in batch.episodes for trajectory in episode.trajectories
    )
    for index, trajectory in enumerate(trajectories):
        if trajectory.policy_version in {None, metadata.policy_version}:
            continue
        issue = _policy_issue(
            trajectory.policy_version,
            requirements,
            field=f"trajectories[{index}].policy_version",
        )
        if issue is not None:
            issues.append(issue)
    if requirements.requires_reference_log_probs and not _has_reference_log_probs(batch):
        issues.append(
            CompatibilityIssue(
                "reference_log_probs",
                "absent",
                "required",
                "the consumer algorithm requires reference-policy log probabilities",
                "generate reference log probabilities during rollout before publishing",
            )
        )
    return CompatibilityReport(
        experience_id=metadata.experience_id,
        producer_framework=metadata.producer_framework,
        producer_framework_version=metadata.producer_framework_version,
        consumer_framework=requirements.consumer_framework,
        consumer_framework_version=requirements.consumer_framework_version,
        issues=tuple(issues),
    )


def ensure_compatible(batch: ExperienceBatch, requirements: CompatibilityRequirements) -> None:
    """Raise :class:`CompatibilityError` unless ``batch`` meets ``requirements``."""

    check_compatibility(batch, requirements).raise_if_incompatible()


def _semantic_issue(field: str, producer: object, consumer: object) -> CompatibilityIssue | None:
    if consumer is None or producer == consumer:
        return None
    missing = producer is None
    return CompatibilityIssue(
        field,
        producer,
        consumer,
        "producer did not declare a required semantic" if missing else "semantic values differ",
        (
            f"set producer metadata.{field} to the consumer-compatible value"
            if missing
            else "use matching rollout/training configuration or reject this experience"
        ),
    )


def _policy_issue(
    producer: PolicyVersion | None,
    requirements: CompatibilityRequirements,
    *,
    field: str = "policy_version",
) -> CompatibilityIssue | None:
    consumer = requirements.policy_version
    if consumer is None or producer == consumer:
        return None
    if producer is None:
        return CompatibilityIssue(
            field,
            producer,
            consumer,
            "policy identity is missing or differs",
            "use a matching policy or configure max_policy_lag for matching integer versions",
        )
    lag = requirements.max_policy_lag
    matching_identity = (
        producer.policy_id == consumer.policy_id and producer.model_id == consumer.model_id
    )
    if (
        lag is not None
        and matching_identity
        and isinstance(producer.version, int)
        and isinstance(consumer.version, int)
        and 0 <= consumer.version - producer.version <= lag
    ):
        return None
    if not matching_identity:
        reason = "policy identity is missing or differs"
    elif isinstance(producer.version, int) and isinstance(consumer.version, int):
        delta = consumer.version - producer.version
        reason = "producer policy is newer than the consumer" if delta < 0 else "policy is stale"
    else:
        reason = "non-numeric policy versions must match exactly"
    return CompatibilityIssue(
        field,
        producer,
        consumer,
        reason,
        "use a matching policy or configure max_policy_lag for matching integer versions",
    )


def _has_reference_log_probs(batch: ExperienceBatch) -> bool:
    trajectories = batch.trajectories + tuple(
        trajectory for episode in batch.episodes for trajectory in episode.trajectories
    )
    return (
        bool(trajectories)
        and all(trajectory.reference_log_probs is not None for trajectory in trajectories)
    ) or any(name in batch.tensors for name in ("reference_log_probs", "ref_log_probs"))
