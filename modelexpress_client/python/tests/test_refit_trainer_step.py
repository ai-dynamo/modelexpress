# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from modelexpress.refit_trainer_step import (
    annotate_trainer_step_ownership,
    publish_trainer_loop_step,
    publish_trainer_step_source,
    trainer_loop_model_version,
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


def _loop_ownerships():
    shape = (6, 4)
    return [
        SliceOwnership(
            model_name="tiny-trainer-loop",
            model_version="base-step",
            tensor_name="lm_head.weight",
            global_shape=shape,
            dtype="float32",
            source_range=((0, 3), (0, 4)),
            worker_id="trainer-rank0-worker",
            source_id="trainer-rank0",
            worker_rank=0,
            layout_tags={"trainer_layout": "fsdp-row-shard-poc"},
            element_size_bytes=4,
        ),
        SliceOwnership(
            model_name="tiny-trainer-loop",
            model_version="base-step",
            tensor_name="lm_head.weight",
            global_shape=shape,
            dtype="float32",
            source_range=((3, 6), (0, 4)),
            worker_id="trainer-rank1-worker",
            source_id="trainer-rank1",
            worker_rank=1,
            layout_tags={"trainer_layout": "fsdp-row-shard-poc"},
            element_size_bytes=4,
        ),
    ]


def _assemble_loop_publications(loop_step):
    shape = loop_step.source_publications[0].ownership.global_shape
    assembled = torch.empty(shape, dtype=loop_step.source_publications[0].tensor.dtype)
    for publication in loop_step.source_publications:
        assembled[_range_slices(publication.ownership.source_range)] = (
            publication.tensor
        )
    return assembled


def test_trainer_loop_step_publication_versions_all_source_ranks():
    loop_step = publish_trainer_loop_step(
        _loop_ownerships(),
        dtype=torch.float32,
        device=torch.device("cpu"),
        step_index=3,
        learning_rate=0.25,
    )

    expected_model_version = trainer_loop_model_version("base-step", 3)
    assert loop_step.model_version == expected_model_version
    assert loop_step.step_index == 3
    assert [pub.ownership.source_id for pub in loop_step.source_publications] == [
        "trainer-rank0",
        "trainer-rank1",
    ]
    assert all(
        pub.ownership.model_version == expected_model_version
        for pub in loop_step.source_publications
    )
    assert all(pub.ownership.source_lease for pub in loop_step.source_publications)
    assert all(
        pub.ownership.layout_tags["trainer_loop_publisher"] is True
        for pub in loop_step.source_publications
    )
    assert all(
        pub.provenance["trainer_loop_publisher_used"] is True
        for pub in loop_step.source_publications
    )
    assert loop_step.provenance["synthetic_trainer_loop_smoke_used"] is True
    assert loop_step.provenance["real_rl_training_loop_used"] is False
    metadata = loop_step.to_artifact_metadata()
    assert metadata["source_publication_count"] == 2
    assert metadata["total_tensor_bytes"] == 96


def test_trainer_loop_step_reconstructs_expected_step_version():
    step1 = publish_trainer_loop_step(
        _loop_ownerships(),
        dtype=torch.float32,
        device=torch.device("cpu"),
        step_index=1,
    )
    step2 = publish_trainer_loop_step(
        _loop_ownerships(),
        dtype=torch.float32,
        device=torch.device("cpu"),
        step_index=2,
    )

    assembled_step2 = _assemble_loop_publications(step2)
    expected_step2 = trainer_step_replacement_tensor(
        (6, 4),
        dtype=torch.float32,
        device="cpu",
        step_count=2,
    )

    torch.testing.assert_close(assembled_step2, expected_step2)
    assert not torch.allclose(_assemble_loop_publications(step1), assembled_step2)


def test_trainer_loop_step_custom_model_version_resets_step_scoped_metadata():
    owners = _loop_ownerships()
    owners[0] = replace(
        owners[0],
        source_lease="stale-lease",
        nixl_descriptor_id="stale-descriptor",
    )

    loop_step = publish_trainer_loop_step(
        owners,
        dtype=torch.float32,
        device=torch.device("cpu"),
        step_index=4,
        model_version="extern-trainer-step-000004",
    )

    assert loop_step.base_model_version == "base-step"
    assert loop_step.model_version == "extern-trainer-step-000004"
    assert all(
        pub.ownership.model_version == "extern-trainer-step-000004"
        for pub in loop_step.source_publications
    )
    assert all(
        "extern-trainer-step-000004" in pub.ownership.source_lease
        for pub in loop_step.source_publications
    )
    assert all(
        pub.ownership.nixl_descriptor_id == pub.ownership.source_lease
        for pub in loop_step.source_publications
    )
    assert all(
        pub.ownership.source_lease != "stale-lease"
        for pub in loop_step.source_publications
    )


def test_trainer_loop_step_rejects_incoherent_ownership_batch():
    owners = _loop_ownerships()

    with pytest.raises(ValueError, match="at least one source ownership"):
        publish_trainer_loop_step(
            [],
            dtype=torch.float32,
            device=torch.device("cpu"),
            step_index=1,
        )
    with pytest.raises(ValueError, match="step_index must be positive"):
        publish_trainer_loop_step(
            owners,
            dtype=torch.float32,
            device=torch.device("cpu"),
            step_index=0,
        )
    with pytest.raises(ValueError, match="span model names"):
        publish_trainer_loop_step(
            [owners[0], replace(owners[1], model_name="other-model")],
            dtype=torch.float32,
            device=torch.device("cpu"),
            step_index=1,
        )
    with pytest.raises(ValueError, match="span base model versions"):
        publish_trainer_loop_step(
            [owners[0], replace(owners[1], model_version="other-base")],
            dtype=torch.float32,
            device=torch.device("cpu"),
            step_index=1,
        )
