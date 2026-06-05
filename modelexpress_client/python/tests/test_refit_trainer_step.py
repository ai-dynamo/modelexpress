# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from modelexpress.refit_trainer_step import (
    DistributedTrainerContext,
    annotate_trainer_step_ownership,
    publish_distributed_trainer_loop_step,
    publish_distributed_trainer_step_source,
    publish_trainer_loop_step,
    publish_trainer_step_source,
    trainer_loop_model_version,
    trainer_step_replacement_tensor,
    trainer_step_source_provenance,
    trainer_step_tensor_for_range,
    trainer_update_parameters_from_ownerships,
)
from modelexpress.refit_distributed_trainer_publication_smoke import (
    _covers_full_rows,
    _ownership_for_rank,
    _row_shard,
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


def test_distributed_trainer_step_publication_marks_real_trainer_process():
    owner = _loop_ownerships()[0]
    context = DistributedTrainerContext(
        backend="gloo",
        rank=0,
        world_size=2,
        local_rank=0,
    )
    expected = trainer_step_replacement_tensor(
        owner.global_shape,
        dtype=torch.float32,
        device="cpu",
        step_count=2,
        learning_rate=0.25,
    )

    publication = publish_distributed_trainer_step_source(
        owner,
        dtype=torch.float32,
        device=torch.device("cpu"),
        step_count=2,
        learning_rate=0.25,
        distributed_context=context,
        synchronize_distributed=False,
    )

    torch.testing.assert_close(
        publication.tensor,
        expected[_range_slices(owner.source_range)],
    )
    assert publication.ownership.layout_tags["trainer_update_source"] == (
        "torch.distributed+torch.optim.SGD-trainer-loop"
    )
    assert publication.ownership.layout_tags["real_distributed_trainer_loop"] is True
    assert publication.ownership.layout_tags["synthetic_trainer_loop_smoke"] is False
    assert publication.ownership.layout_tags["distributed_trainer_backend"] == "gloo"
    assert publication.ownership.layout_tags["distributed_trainer_rank"] == 0
    assert publication.provenance["real_distributed_trainer_loop_used"] is True
    assert publication.provenance["synthetic_trainer_loop_smoke_used"] is False
    assert publication.provenance["torch_distributed_data_transfer_used"] is False
    assert publication.provenance["trainer_tensor_payload_transfer"] == "mx-nixl"


def test_distributed_trainer_loop_publication_is_rank_scoped():
    owner = _loop_ownerships()[1]
    context = DistributedTrainerContext(
        backend="gloo",
        rank=1,
        world_size=2,
        local_rank=0,
    )

    loop_step = publish_distributed_trainer_loop_step(
        [owner],
        dtype=torch.float32,
        device=torch.device("cpu"),
        step_index=3,
        learning_rate=0.25,
        distributed_context=context,
        synchronize_distributed=False,
    )

    publication = loop_step.source_publications[0]
    assert loop_step.model_version == trainer_loop_model_version("base-step", 3)
    assert loop_step.provenance["real_distributed_trainer_loop_used"] is True
    assert loop_step.provenance["synthetic_trainer_loop_smoke_used"] is False
    assert loop_step.provenance["torch_distributed_world_size"] == 2
    assert loop_step.provenance["torch_distributed_rank"] == 1
    assert publication.ownership.source_id == "trainer-rank1"
    assert publication.ownership.layout_tags["trainer_loop_publisher"] is True
    assert publication.ownership.layout_tags["trainer_loop_step_index"] == 3
    assert publication.ownership.layout_tags["real_distributed_trainer_loop"] is True
    assert publication.ownership.layout_tags["synthetic_trainer_loop_smoke"] is False


def test_distributed_trainer_publication_rejects_wrong_rank_owner():
    owner = _loop_ownerships()[1]
    context = DistributedTrainerContext(
        backend="gloo",
        rank=0,
        world_size=2,
        local_rank=0,
    )

    with pytest.raises(ValueError, match="worker_rank must match"):
        publish_distributed_trainer_step_source(
            owner,
            dtype=torch.float32,
            device=torch.device("cpu"),
            distributed_context=context,
            synchronize_distributed=False,
        )


def test_distributed_trainer_smoke_row_shards_cover_full_tensor():
    payloads = [
        {
            "source_range": [
                [_row_shard(rank, 3, 8)[0], _row_shard(rank, 3, 8)[1]],
                [0, 4],
            ]
        }
        for rank in range(3)
    ]

    assert [_row_shard(rank, 3, 8) for rank in range(3)] == [
        (0, 3),
        (3, 6),
        (6, 8),
    ]
    assert _covers_full_rows(payloads, (8, 4)) is True


def test_distributed_trainer_smoke_ownership_marks_rank_owner():
    owner = _ownership_for_rank(
        model_name="mx-dist-unit",
        model_version="base",
        tensor_name="lm_head.weight",
        shape=(8, 4),
        dtype=torch.float32,
        rank=1,
        world_size=2,
    )

    assert owner.source_id == "trainer-rank1"
    assert owner.worker_rank == 1
    assert owner.source_range == ((4, 8), (0, 4))
    assert owner.layout_tags["trainer_layout"] == "torch-distributed-row-shard-poc"
    assert owner.layout_tags["source_tensor_owner"] == (
        "torch.distributed-trainer-rank"
    )


def test_trainer_update_parameters_are_inferred_from_source_ownership_metadata():
    owners = [
        replace(
            owner,
            layout_tags={
                **owner.layout_tags,
                "optimizer_step_count": 2,
                "learning_rate": "0.25",
            },
        )
        for owner in _loop_ownerships()
    ]

    params = trainer_update_parameters_from_ownerships(owners)

    assert params.step_count == 2
    assert params.learning_rate == 0.25

    loop_params = trainer_update_parameters_from_ownerships(
        [
            replace(
                owner,
                layout_tags={
                    **owner.layout_tags,
                    "optimizer_step_count": 1,
                    "trainer_loop_step_index": 4,
                    "learning_rate": "0.5",
                },
            )
            for owner in _loop_ownerships()
        ]
    )
    assert loop_params.step_count == 4
    assert loop_params.learning_rate == 0.5

    with pytest.raises(ValueError, match="inconsistent optimizer step counts"):
        trainer_update_parameters_from_ownerships(
            [
                replace(
                    owners[0],
                    layout_tags={**owners[0].layout_tags, "optimizer_step_count": 2},
                ),
                replace(
                    owners[1],
                    layout_tags={**owners[1].layout_tags, "optimizer_step_count": 3},
                ),
            ]
        )

    with pytest.raises(ValueError, match="inconsistent learning rates"):
        trainer_update_parameters_from_ownerships(
            [
                replace(
                    owners[0],
                    layout_tags={**owners[0].layout_tags, "learning_rate": "0.25"},
                ),
                replace(
                    owners[1],
                    layout_tags={**owners[1].layout_tags, "learning_rate": "0.5"},
                ),
            ]
        )


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
        pub.ownership.layout_tags["trainer_update_source"]
        == "torch.optim.SGD-trainer-loop-smoke"
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
