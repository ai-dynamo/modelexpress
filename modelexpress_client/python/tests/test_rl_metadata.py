# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from modelexpress import p2p_pb2
from modelexpress.rl_metadata import (
    RL_SCHEMA_VERSION_KEY,
    RlSourceCandidate,
    RlSourceMetadata,
    RlSourceRole,
    build_rl_query_identities,
    candidates_from_response,
    complete_model_versions,
    complete_source_groups,
    get_rl_source_metadata,
    latest_complete_model_version,
    latest_model_version,
    retained_model_versions,
    select_rl_source_candidates,
    try_get_rl_source_metadata,
    with_rl_source_metadata,
)


def _base_identity() -> p2p_pb2.SourceIdentity:
    return p2p_pb2.SourceIdentity(
        mx_version="0.3.0",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name="test-model",
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        expert_parallel_size=0,
        dtype="bfloat16",
        revision="abc123",
    )


def _ref(
    *,
    source_id: str,
    worker_id: str,
    rank: int,
) -> p2p_pb2.SourceInstanceRef:
    return p2p_pb2.SourceInstanceRef(
        mx_source_id=source_id,
        worker_id=worker_id,
        model_name="test-model",
        worker_rank=rank,
    )


def _candidate(
    source_id: str,
    *,
    version: int,
    role: RlSourceRole = RlSourceRole.TRAINER,
    rank: int = 0,
    world_size: int = 1,
    retain_latest_k: int = 1,
) -> RlSourceCandidate:
    return RlSourceCandidate(
        source_id,
        f"{source_id}-worker-r{rank}",
        "test-model",
        rank,
        RlSourceMetadata(
            version,
            role,
            world_size=world_size,
            retain_latest_k=retain_latest_k,
        ),
    )


def test_rl_source_metadata_round_trip_preserves_shape_registry_case():
    metadata = RlSourceMetadata(
        model_version=7,
        role=RlSourceRole.TRAINER,
        world_size=4,
        retain_latest_k=2,
        shape_registry={
            "Model.Layer.W": {
                "shape": [2, 4],
                "dtype": "BF16",
            },
        },
    )

    parsed = RlSourceMetadata.from_extra_parameters(metadata.to_extra_parameters())

    assert parsed == metadata
    assert parsed.shape_registry["Model.Layer.W"]["dtype"] == "BF16"


def test_with_rl_source_metadata_does_not_mutate_base_identity():
    identity = _base_identity()
    metadata = RlSourceMetadata(
        model_version=3,
        role=RlSourceRole.INFERENCE_REPLICA,
        world_size=8,
    )

    updated = with_rl_source_metadata(identity, metadata)

    assert RL_SCHEMA_VERSION_KEY not in identity.extra_parameters
    assert get_rl_source_metadata(updated) == metadata
    assert updated.model_name == identity.model_name


def test_with_rl_source_metadata_clears_stale_shape_registry():
    identity = with_rl_source_metadata(
        _base_identity(),
        RlSourceMetadata(
            model_version=1,
            role=RlSourceRole.TRAINER,
            world_size=1,
            shape_registry={"stale": {"shape": [1]}},
        ),
    )

    updated = with_rl_source_metadata(
        identity,
        RlSourceMetadata(
            model_version=2,
            role=RlSourceRole.TRAINER,
            world_size=1,
        ),
    )

    assert get_rl_source_metadata(updated).shape_registry == {}


def test_try_get_rl_source_metadata_returns_none_for_plain_identity():
    assert try_get_rl_source_metadata(_base_identity()) is None


def test_rl_source_metadata_rejects_invalid_values():
    with pytest.raises(ValueError, match="model_version"):
        RlSourceMetadata(
            model_version=-1,
            role=RlSourceRole.TRAINER,
            world_size=1,
        )

    with pytest.raises(ValueError, match="world_size"):
        RlSourceMetadata(
            model_version=1,
            role=RlSourceRole.TRAINER,
            world_size=0,
        )


def test_build_rl_query_identities_uses_role_specific_metadata():
    identities = build_rl_query_identities(
        _base_identity(),
        model_version=11,
        world_size=4,
        roles=(RlSourceRole.INFERENCE_REPLICA, RlSourceRole.TRAINER),
    )

    assert [get_rl_source_metadata(identity).role for identity in identities] == [
        RlSourceRole.INFERENCE_REPLICA,
        RlSourceRole.TRAINER,
    ]
    assert {get_rl_source_metadata(identity).model_version for identity in identities} == {11}


def test_candidates_from_response_annotates_list_sources_refs():
    metadata = RlSourceMetadata(
        model_version=5,
        role=RlSourceRole.TRAINER,
        world_size=2,
    )
    response = p2p_pb2.ListSourcesResponse(
        instances=[
            _ref(source_id="source-a", worker_id="worker-a", rank=0),
            _ref(source_id="source-a", worker_id="worker-b", rank=1),
        ],
    )

    candidates = candidates_from_response(response, metadata)

    assert candidates == [
        RlSourceCandidate("source-a", "worker-a", "test-model", 0, metadata),
        RlSourceCandidate("source-a", "worker-b", "test-model", 1, metadata),
    ]


def test_candidates_from_response_can_parse_rl_metadata_from_ref_identity():
    metadata = RlSourceMetadata(
        model_version=9,
        role=RlSourceRole.INFERENCE_REPLICA,
        world_size=2,
    )
    identity = with_rl_source_metadata(_base_identity(), metadata)
    plain_identity = _base_identity()
    response = p2p_pb2.ListSourcesResponse(
        instances=[
            p2p_pb2.SourceInstanceRef(
                mx_source_id="source-a",
                worker_id="worker-a",
                model_name="test-model",
                worker_rank=0,
                identity=identity,
            ),
            p2p_pb2.SourceInstanceRef(
                mx_source_id="source-b",
                worker_id="worker-b",
                model_name="test-model",
                worker_rank=1,
                identity=plain_identity,
            ),
            _ref(source_id="legacy-source", worker_id="legacy-worker", rank=0),
        ],
    )

    candidates = candidates_from_response(response)

    assert candidates == [
        RlSourceCandidate("source-a", "worker-a", "test-model", 0, metadata),
    ]


def test_select_candidates_uses_latest_version_and_prefers_replica_role():
    candidates = [
        RlSourceCandidate(
            "trainer-v7",
            "trainer-r0",
            "test-model",
            0,
            RlSourceMetadata(7, RlSourceRole.TRAINER, world_size=1),
        ),
        RlSourceCandidate(
            "replica-v8",
            "replica-r0",
            "test-model",
            0,
            RlSourceMetadata(8, RlSourceRole.INFERENCE_REPLICA, world_size=1),
        ),
        RlSourceCandidate(
            "trainer-v8",
            "trainer-r0",
            "test-model",
            0,
            RlSourceMetadata(8, RlSourceRole.TRAINER, world_size=1),
        ),
    ]

    selected = select_rl_source_candidates(candidates, receiver_rank=0)

    assert [candidate.mx_source_id for candidate in selected] == [
        "replica-v8",
        "trainer-v8",
    ]
    assert latest_model_version(candidates) == 8


def test_retained_model_versions_uses_latest_version_window():
    candidates = [
        _candidate("trainer-v7", version=7, retain_latest_k=5),
        _candidate("trainer-v8", version=8, retain_latest_k=5),
        _candidate("trainer-v9", version=9, retain_latest_k=2),
    ]

    assert retained_model_versions(candidates) == {8, 9}


def test_complete_model_versions_require_distinct_ranks_per_role():
    candidates = [
        RlSourceCandidate(
            "trainer-v7",
            "trainer-r0",
            "test-model",
            0,
            RlSourceMetadata(7, RlSourceRole.TRAINER, world_size=2),
        ),
        RlSourceCandidate(
            "trainer-v7",
            "trainer-r1",
            "test-model",
            1,
            RlSourceMetadata(7, RlSourceRole.TRAINER, world_size=2),
        ),
        RlSourceCandidate(
            "replica-v8",
            "replica-r0",
            "test-model",
            0,
            RlSourceMetadata(8, RlSourceRole.INFERENCE_REPLICA, world_size=2),
        ),
    ]

    assert complete_model_versions(candidates) == {7}
    assert complete_source_groups(candidates) == {(7, RlSourceRole.TRAINER)}
    assert latest_complete_model_version(candidates) == 7


def test_select_candidates_skips_incomplete_preferred_role_group():
    candidates = [
        _candidate("trainer-v8-r0", version=8, rank=0, world_size=2),
        _candidate("trainer-v8-r1", version=8, rank=1, world_size=2),
        _candidate(
            "replica-v8-r0",
            version=8,
            role=RlSourceRole.INFERENCE_REPLICA,
            rank=0,
            world_size=2,
        ),
    ]

    selected = select_rl_source_candidates(
        candidates,
        receiver_rank=0,
        same_rank_only=False,
    )

    assert [candidate.mx_source_id for candidate in selected] == [
        "trainer-v8-r0",
        "trainer-v8-r1",
    ]


def test_select_candidates_falls_back_to_latest_complete_version():
    candidates = [
        RlSourceCandidate(
            "trainer-v7",
            "trainer-r0",
            "test-model",
            0,
            RlSourceMetadata(7, RlSourceRole.TRAINER, world_size=2),
        ),
        RlSourceCandidate(
            "trainer-v7",
            "trainer-r1",
            "test-model",
            1,
            RlSourceMetadata(7, RlSourceRole.TRAINER, world_size=2),
        ),
        RlSourceCandidate(
            "trainer-v8",
            "trainer-r0",
            "test-model",
            0,
            RlSourceMetadata(8, RlSourceRole.TRAINER, world_size=2),
        ),
    ]

    selected = select_rl_source_candidates(
        candidates,
        receiver_rank=1,
        same_rank_only=False,
    )

    assert [candidate.worker_id for candidate in selected] == [
        "trainer-r1",
        "trainer-r0",
    ]


def test_select_candidates_retention_ignores_incomplete_newer_version():
    candidates = [
        _candidate("trainer-v8", version=8),
        _candidate("trainer-v9", version=9, retain_latest_k=2),
        _candidate("trainer-v10", version=10, world_size=2),
    ]

    selected = select_rl_source_candidates(
        candidates,
        receiver_rank=0,
        model_version=8,
    )

    assert [candidate.mx_source_id for candidate in selected] == ["trainer-v8"]


def test_select_candidates_can_request_specific_version():
    candidates = [
        _candidate(
            "replica-v8",
            version=8,
            role=RlSourceRole.INFERENCE_REPLICA,
            retain_latest_k=2,
        ),
        _candidate("trainer-v7", version=7),
    ]

    selected = select_rl_source_candidates(
        candidates,
        receiver_rank=0,
        model_version=7,
    )

    assert [candidate.mx_source_id for candidate in selected] == ["trainer-v7"]


def test_select_candidates_rejects_exact_version_outside_retention_window():
    candidates = [
        _candidate("trainer-v7", version=7, retain_latest_k=3),
        _candidate("trainer-v8", version=8, retain_latest_k=2),
        _candidate("trainer-v9", version=9, retain_latest_k=2),
    ]

    assert (
        select_rl_source_candidates(candidates, receiver_rank=0, model_version=7)
        == []
    )
    assert [
        candidate.mx_source_id
        for candidate in select_rl_source_candidates(
            candidates,
            receiver_rank=0,
            model_version=8,
        )
    ] == ["trainer-v8"]


def test_select_candidates_enforces_same_rank_by_default():
    candidates = [
        RlSourceCandidate(
            "same-rank",
            "worker-r1",
            "test-model",
            1,
            RlSourceMetadata(3, RlSourceRole.TRAINER, world_size=2),
        ),
        RlSourceCandidate(
            "other-rank",
            "worker-r0",
            "test-model",
            0,
            RlSourceMetadata(3, RlSourceRole.TRAINER, world_size=2),
        ),
    ]

    assert [
        candidate.mx_source_id
        for candidate in select_rl_source_candidates(candidates, receiver_rank=1)
    ] == ["same-rank"]
    assert [
        candidate.mx_source_id
        for candidate in select_rl_source_candidates(
            candidates,
            receiver_rank=1,
            same_rank_only=False,
        )
    ] == ["same-rank", "other-rank"]
