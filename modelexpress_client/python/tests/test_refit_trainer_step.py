# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from modelexpress.refit_trainer_step import (
    annotate_trainer_step_ownership,
    publish_trainer_step_source,
    trainer_step_replacement_tensor,
    trainer_step_source_provenance,
    trainer_step_tensor_for_range,
)
from modelexpress.resharding import SliceOwnership


def _range_slices(tensor_range):
    return tuple(slice(start, end) for start, end in tensor_range)


def test_trainer_step_source_range_matches_full_replacement_slice():
    shape = (7, 4)
    tensor_range = ((2, 6), (0, 4))

    full = trainer_step_replacement_tensor(
        shape,
        dtype=torch.float32,
        device="cpu",
    )
    source = trainer_step_tensor_for_range(
        shape,
        tensor_range,
        dtype=torch.float32,
        device="cpu",
    )

    assert source.shape == (4, 4)
    torch.testing.assert_close(source, full[_range_slices(tensor_range)])


def test_trainer_step_changes_parameter_from_initial_values():
    shape = (5, 3)
    tensor_range = ((0, 5), (0, 3))

    initial = trainer_step_tensor_for_range(
        shape,
        tensor_range,
        dtype=torch.float32,
        device="cpu",
        step_count=0,
    )
    after_step = trainer_step_tensor_for_range(
        shape,
        tensor_range,
        dtype=torch.float32,
        device="cpu",
        step_count=1,
    )

    assert not torch.allclose(initial, after_step)


def test_trainer_step_preserves_requested_dtype():
    payload = trainer_step_replacement_tensor(
        (6, 4),
        dtype=torch.bfloat16,
        device="cpu",
    )

    assert payload.dtype == torch.bfloat16
    assert payload.shape == (6, 4)


def test_trainer_step_provenance_is_explicit():
    provenance = trainer_step_source_provenance(step_count=2, learning_rate=0.25)

    assert provenance["source_payload_generator"] == "torch.optim.SGD"
    assert provenance["optimizer_step_count"] == 2
    assert provenance["learning_rate"] == 0.25
    assert provenance["optimizer_step_publisher_used"] is True
    assert provenance["synthetic_training_objective_used"] is True
    assert provenance["static_replacement_formula_used"] is False
    assert provenance["real_rl_training_loop_used"] is False


def _ownership(source_range=((1, 5), (0, 4))):
    return SliceOwnership(
        model_name="tiny-trainer-step",
        model_version="step-17",
        tensor_name="lm_head.weight",
        global_shape=(7, 4),
        dtype="float32",
        source_range=source_range,
        worker_id="trainer-rank7-worker",
        source_id="trainer-rank7",
        worker_rank=7,
        layout_tags={"trainer_layout": "fsdp-row-shard-poc"},
        element_size_bytes=4,
    )


def test_trainer_step_ownership_annotation_fills_publication_metadata():
    owner = _ownership()

    annotated = annotate_trainer_step_ownership(
        owner,
        step_count=3,
        learning_rate=0.5,
    )

    assert annotated.source_lease == "trainer-rank7-step-17-optimizer-step-3"
    assert annotated.nixl_descriptor_id == annotated.source_lease
    assert annotated.layout_tags["trainer_layout"] == "fsdp-row-shard-poc"
    assert annotated.layout_tags["trainer_update_source"] == (
        "torch.optim.SGD-step-publisher"
    )
    assert annotated.layout_tags["source_payload_generator"] == "torch.optim.SGD"
    assert annotated.layout_tags["optimizer_step_count"] == 3
    assert annotated.layout_tags["learning_rate"] == "0.5"
    assert annotated.layout_tags["optimizer_step_publisher"] is True
    assert annotated.layout_tags["static_replacement_formula"] is False


def test_trainer_step_publication_carries_tensor_ownership_and_artifact_metadata():
    owner = _ownership()
    full = trainer_step_replacement_tensor(
        owner.global_shape,
        dtype=torch.float32,
        device="cpu",
        step_count=2,
        learning_rate=0.25,
    )

    publication = publish_trainer_step_source(
        owner,
        dtype=torch.float32,
        device=torch.device("cpu"),
        step_count=2,
        learning_rate=0.25,
    )

    torch.testing.assert_close(
        publication.tensor,
        full[_range_slices(owner.source_range)],
    )
    assert publication.ownership.layout_tags["optimizer_step_publisher"] is True
    assert publication.provenance["optimizer_step_count"] == 2
    metadata = publication.to_artifact_metadata()
    assert metadata["ownership"]["source_id"] == "trainer-rank7"
    assert metadata["source_payload_provenance"]["learning_rate"] == 0.25
    assert metadata["tensor_shape"] == [4, 4]
    assert metadata["tensor_dtype"] == "float32"
    assert metadata["tensor_bytes"] == 64
