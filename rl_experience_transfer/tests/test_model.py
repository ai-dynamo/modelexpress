# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import cast
from uuid import UUID

import numpy as np
import pytest

from rlxfer.errors import SchemaValidationError
from rlxfer.model import (
    SCHEMA_VERSION,
    Episode,
    ExperienceBatch,
    ExperienceMetadata,
    PolicyVersion,
    SampleIdentity,
    TensorPayload,
    Trajectory,
    TransferDescriptor,
    Transition,
)
from rlxfer.observability import InMemoryMetrics, structured_log


def _metadata() -> ExperienceMetadata:
    return ExperienceMetadata(
        producer_id="rollout-1",
        producer_framework="nemo_rl",
        producer_framework_version="0.4",
        policy_version=PolicyVersion(7, model_id="tiny-model"),
    )


def _trajectory(length: int = 4, sample_id: str | None = None) -> Trajectory:
    identity = SampleIdentity(sample_id=sample_id) if sample_id is not None else SampleIdentity()
    return Trajectory(
        identity=identity,
        policy_version=PolicyVersion(7, model_id="tiny-model"),
        prompt="question",
        response="answer",
        tokens=TensorPayload(np.arange(length, dtype=np.int64)),
        attention_mask=TensorPayload(np.ones(length, dtype=np.bool_)),
        log_probs=TensorPayload(np.linspace(-1.0, -0.1, length, dtype=np.float32)),
        rewards={"task": 1.0},
        transitions=(Transition(observation={"step": 0}, action=1, reward=1.0),),
    )


@pytest.mark.unit
def test_tensor_metadata_and_variable_length_batch_validate() -> None:
    first = _trajectory(3)
    second = _trajectory(5)
    batch = ExperienceBatch(metadata=_metadata(), trajectories=(first, second))

    batch.validate()

    assert SCHEMA_VERSION == "1.0"
    assert first.tokens is not None
    assert first.tokens.shape == (3,)
    assert first.tokens.dtype == "int64"
    assert first.tokens.device == "cpu"
    assert first.tokens.nbytes == 24
    assert not hasattr(batch, "__dict__")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field_name", "false_value"),
    [
        ("stride", (99, 1)),
        ("layout", "sparse"),
        ("nbytes", 1),
        ("device", "cuda:99"),
    ],
)
def test_tensor_payload_rejects_metadata_that_disagrees_with_data(
    field_name: str,
    false_value: object,
) -> None:
    payload = TensorPayload(np.arange(6, dtype=np.float32).reshape(2, 3))
    setattr(payload, field_name, false_value)

    with pytest.raises(SchemaValidationError) as caught:
        payload.validate()

    assert caught.value.field == f"tensor.{field_name}"


@pytest.mark.unit
def test_nested_extensions_accept_json_and_wrapped_tensors() -> None:
    extension_tensor = TensorPayload(np.array([1.5, 2.5], dtype=np.float32))
    batch = ExperienceBatch(
        metadata=_metadata(),
        trajectories=(_trajectory(),),
        payload={"nested": [1, {"tensor": extension_tensor}]},
        extensions={"nemo_rl": {"loss_mask": extension_tensor, "flags": (True, None)}},
    )

    batch.validate()


@pytest.mark.unit
def test_alignment_error_carries_complete_framework_context() -> None:
    trajectory = _trajectory(4)
    trajectory.attention_mask = TensorPayload(np.ones(3, dtype=np.int64))
    batch = ExperienceBatch(metadata=_metadata(), trajectories=(trajectory,))

    with pytest.raises(SchemaValidationError) as caught:
        batch.validate(consumer_framework="prime_rl", consumer_framework_version="0.2")

    error = caught.value
    assert error.field == "trajectories[0].attention_mask.shape"
    assert error.expected == (4,)
    assert error.actual == (3,)
    assert error.producer_framework == "nemo_rl"
    assert error.producer_framework_version == "0.4"
    assert error.consumer_framework == "prime_rl"
    assert error.consumer_framework_version == "0.2"
    assert error.experience_id == batch.experience_id
    assert "nemo_rl@0.4" in str(error)
    assert "prime_rl@0.2" in str(error)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [("experience_id", "not-a-uuid"), ("schema_version", "2.0")],
)
def test_metadata_rejects_invalid_uuid_or_schema(field_name: str, field_value: str) -> None:
    metadata = _metadata()
    setattr(metadata, field_name, field_value)
    batch = ExperienceBatch(metadata=metadata, trajectories=(_trajectory(),))

    with pytest.raises(SchemaValidationError) as caught:
        batch.validate()

    assert caught.value.field == f"metadata.{field_name}"


@pytest.mark.unit
def test_extension_namespace_and_nested_raw_array_are_rejected() -> None:
    batch = ExperienceBatch(
        metadata=_metadata(),
        trajectories=(_trajectory(),),
        extensions={"not a namespace": {"raw": np.ones(1)}},
    )
    with pytest.raises(SchemaValidationError, match="namespace"):
        batch.validate()

    batch.extensions = {"slime": {"raw": np.ones(1)}}
    with pytest.raises(SchemaValidationError, match="TensorPayload"):
        batch.validate()


@pytest.mark.unit
def test_duplicate_sample_ids_and_invalid_episode_are_rejected() -> None:
    duplicated = str(UUID("89cb7285-3459-4c62-801d-e812f67a69f8"))
    batch = ExperienceBatch(
        metadata=_metadata(),
        trajectories=(_trajectory(sample_id=duplicated), _trajectory(sample_id=duplicated)),
    )
    with pytest.raises(SchemaValidationError, match="unique sample ID"):
        batch.validate()

    empty_episode = Episode(trajectories=())
    with pytest.raises(SchemaValidationError, match="at least one trajectory"):
        ExperienceBatch(metadata=_metadata(), episodes=(empty_episode,)).validate()


@pytest.mark.unit
def test_transfer_descriptor_validates_catalog_and_size() -> None:
    descriptor = TransferDescriptor(
        experience_id=_metadata().experience_id,
        tensor_names=("tokens", "mask"),
        byte_sizes={"tokens": 16, "mask": 2},
        metadata_bytes=100,
        checksums={"tokens": "sha256:123"},
    )

    descriptor.validate()
    assert descriptor.total_bytes == 118

    descriptor.byte_sizes.pop("mask")
    with pytest.raises(SchemaValidationError) as caught:
        descriptor.validate()
    assert caught.value.field == "byte_sizes"


@pytest.mark.unit
def test_metrics_and_structured_logging_never_expose_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    metrics = InMemoryMetrics()
    metrics.increment("produced_batches", attributes={"transport": "memory"})
    metrics.observe("transfer_latency", 0.25)
    counters, observations = metrics.snapshot()
    assert counters == {"produced_batches{transport=memory}": 1}
    assert observations == {"transfer_latency": (0.25,)}

    logger = logging.getLogger("rlxfer-test")
    with caplog.at_level(logging.INFO, logger="rlxfer-test"):
        structured_log(
            logger,
            "batch_produced",
            experience_id="safe-id",
            prompt="secret prompt",
            tokens=[1, 2, 3],
            nested={"response_text": "secret answer"},
        )
    fields = cast(dict[str, object], vars(caplog.records[-1])["rlxfer"])
    assert fields["experience_id"] == "safe-id"
    assert fields["prompt"] == "<redacted>"
    assert fields["tokens"] == "<redacted>"
    assert fields["nested"] == {"response_text": "<redacted>"}
    assert "secret" not in str(fields)
