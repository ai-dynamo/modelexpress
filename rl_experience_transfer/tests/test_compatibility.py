# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rlxfer.compatibility import (
    CompatibilityRequirements,
    check_compatibility,
    ensure_compatible,
)
from rlxfer.errors import CompatibilityError
from rlxfer.model import (
    ExperienceBatch,
    ExperienceMetadata,
    PolicyVersion,
    TensorPayload,
    Trajectory,
)


def _batch(*, with_reference: bool = True) -> ExperienceBatch:
    tokens = TensorPayload(np.array([1, 2, 3], dtype=np.int64))
    reference = (
        TensorPayload(np.array([-0.2, -0.3, -0.4], dtype=np.float32)) if with_reference else None
    )
    metadata = ExperienceMetadata(
        producer_id="worker-1",
        producer_framework="slime",
        producer_framework_version="0.1",
        policy_version=PolicyVersion(4, policy_id="actor", model_id="tiny"),
        algorithm="grpo",
        tokenizer_id="tokenizer-v1",
        model_id="tiny",
        reward_definition="unit-test-reward-v1",
        sequence_format="prompt-response",
        padding="right",
        chat_template="chat-v1",
        truncation="right:64",
    )
    return ExperienceBatch(
        metadata=metadata,
        trajectories=(Trajectory(tokens=tokens, reference_log_probs=reference),),
    )


def _requirements() -> CompatibilityRequirements:
    return CompatibilityRequirements(
        consumer_framework="miles",
        consumer_framework_version="0.3",
        algorithm="grpo",
        tokenizer_id="tokenizer-v1",
        model_id="tiny",
        policy_version=PolicyVersion(4, policy_id="actor", model_id="tiny"),
        reward_definition="unit-test-reward-v1",
        sequence_format="prompt-response",
        padding="right",
        chat_template="chat-v1",
        truncation="right:64",
        requires_reference_log_probs=True,
    )


@pytest.mark.unit
def test_matching_cross_framework_semantics_are_compatible() -> None:
    report = check_compatibility(_batch(), _requirements())

    assert report.compatible
    assert report.issues == ()
    ensure_compatible(_batch(), _requirements())


@pytest.mark.unit
def test_report_collects_every_mismatch_and_rejection_is_actionable() -> None:
    requirements = CompatibilityRequirements(
        consumer_framework="prime_rl",
        consumer_framework_version="0.9",
        algorithm="ppo",
        tokenizer_id="other-tokenizer",
        model_id="other-model",
        policy_version=PolicyVersion(99),
        reward_definition="other-reward",
        sequence_format="response-only",
        padding="left",
        chat_template="other-chat",
        truncation="left:32",
        requires_reference_log_probs=True,
    )
    batch = _batch(with_reference=False)

    report = check_compatibility(batch, requirements)

    assert not report.compatible
    assert {issue.field for issue in report.issues} == {
        "algorithm",
        "tokenizer_id",
        "model_id",
        "policy_version",
        "reward_definition",
        "sequence_format",
        "padding",
        "chat_template",
        "truncation",
        "reference_log_probs",
    }
    with pytest.raises(CompatibilityError) as caught:
        report.raise_if_incompatible()
    message = str(caught.value)
    assert batch.experience_id in message
    assert "action:" in message
    assert "slime@0.1" in message
    assert "prime_rl@0.9" in message


@pytest.mark.unit
def test_missing_required_producer_metadata_is_not_treated_as_wildcard() -> None:
    batch = _batch()
    batch.metadata.tokenizer_id = None

    report = check_compatibility(batch, _requirements())

    issue = next(item for item in report.issues if item.field == "tokenizer_id")
    assert "did not declare" in issue.reason
    assert "metadata.tokenizer_id" in issue.action


@pytest.mark.unit
def test_policy_staleness_is_bounded_and_requires_matching_identity() -> None:
    requirements = replace(
        _requirements(),
        policy_version=PolicyVersion(6, policy_id="actor", model_id="tiny"),
        max_policy_lag=2,
    )
    batch = _batch()
    assert check_compatibility(batch, requirements).compatible

    for policy, reason in (
        (PolicyVersion(3, policy_id="actor", model_id="tiny"), "stale"),
        (PolicyVersion(7, policy_id="actor", model_id="tiny"), "newer"),
        (PolicyVersion(5, policy_id="other", model_id="tiny"), "identity"),
    ):
        batch.metadata.policy_version = policy
        issue = next(
            item
            for item in check_compatibility(batch, requirements).issues
            if item.field == "policy_version"
        )
        assert reason in issue.reason

    with pytest.raises(ValueError, match="requires policy_version"):
        CompatibilityRequirements("trainer", "1", max_policy_lag=1)

    batch.metadata.policy_version = PolicyVersion(6, policy_id="actor", model_id="tiny")
    batch.trajectories[0].policy_version = PolicyVersion(2, policy_id="actor", model_id="tiny")
    assert any(
        issue.field == "trajectories[0].policy_version"
        for issue in check_compatibility(batch, requirements).issues
    )
