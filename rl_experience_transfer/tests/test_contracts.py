# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rlxfer.api import ExperienceProducer
from rlxfer.compatibility import CompatibilityRequirements
from rlxfer.contracts import ConsumerContract, SchemaMigrationRegistry
from rlxfer.errors import CompatibilityError, MigrationError
from rlxfer.model import (
    SCHEMA_VERSION,
    ExperienceBatch,
    ExperienceMetadata,
    TensorPayload,
    Trajectory,
)
from rlxfer.transports.memory import InMemoryTransport


def _batch(version: str = SCHEMA_VERSION) -> ExperienceBatch:
    return ExperienceBatch(
        metadata=ExperienceMetadata("rollout", "test", "1", schema_version=version),
        trajectories=(Trajectory(tokens=TensorPayload(np.arange(3, dtype=np.int64))),),
    )


def _contract(*required_fields: str) -> ConsumerContract:
    return ConsumerContract(
        CompatibilityRequirements("trainer", "1"),
        required_fields=frozenset(required_fields),
    )


def _set_version(batch: ExperienceBatch, version: str) -> ExperienceBatch:
    return replace(batch, metadata=replace(batch.metadata, schema_version=version))


@pytest.mark.unit
def test_contract_migrates_explicit_path_and_checks_nested_fields() -> None:
    migrations = SchemaMigrationRegistry(
        {
            ("0.8", "0.9"): lambda batch: _set_version(batch, "0.9"),
            ("0.9", SCHEMA_VERSION): lambda batch: _set_version(batch, SCHEMA_VERSION),
        }
    )

    migrated = _contract("trajectories.tokens").negotiate(_batch("0.8"), migrations)

    assert migrated.metadata.schema_version == SCHEMA_VERSION
    assert migrated.experience_id == migrated.metadata.experience_id


@pytest.mark.unit
def test_contract_rejects_missing_fields_before_transport() -> None:
    transport = InMemoryTransport()
    producer = ExperienceProducer(transport, consumer_contract=_contract("trajectories.log_probs"))

    with pytest.raises(CompatibilityError, match=r"trajectories\.log_probs"):
        producer.publish(_batch())

    assert transport.health().queue_depth == 0
    ExperienceProducer(
        transport,
        consumer_contract=_contract("metadata.idempotency_key"),
    ).publish(_batch(), idempotency_key="caller-key")
    assert transport.health().queue_depth == 1


@pytest.mark.unit
def test_migration_requires_a_path_valid_version_and_stable_identity() -> None:
    batch = _batch("0.9")
    with pytest.raises(CompatibilityError, match="does not support schema"):
        _contract().negotiate(batch)

    migrations = SchemaMigrationRegistry()
    with pytest.raises(MigrationError, match="no schema migration path"):
        migrations.migrate(batch)

    migrations.register(
        "0.9",
        SCHEMA_VERSION,
        lambda value: replace(
            _set_version(value, SCHEMA_VERSION),
            metadata=replace(
                value.metadata,
                schema_version=SCHEMA_VERSION,
                experience_id=ExperienceMetadata("other", "test", "1").experience_id,
            ),
        ),
    )
    with pytest.raises(MigrationError, match="changed experience_id"):
        migrations.migrate(batch)
