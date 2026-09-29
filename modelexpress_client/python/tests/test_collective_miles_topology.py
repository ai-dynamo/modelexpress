# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure MILES topology projection for the PR 3304 reference layout.

Covers the exact TP1/CP1/PP8/EP4/ETP1 trainer to four TP8/EP8 rollout
projection, including lane-local mesh coordinates, deterministic ordering,
atomic residual fallback, and fail-closed validation.

Live tensor discovery, storage binding, and collective execution are covered
by the MILES publisher, SGLang binding, and backend test modules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest
from pytest import param

from modelexpress_rl.collective import Placement, plan_digest
from modelexpress_rl.collective.integrations.miles_topology import (
    MilesTensorSpec,
    MilesTrainerTopology,
    build_miles_reshard_plan,
)

_PP_SIZE = 8
_EP_SIZE = 4
_ROLLOUT_ENGINE_SIZES = (8, 8, 8, 8)
_WORLD_SLOT_BY_EP = (2, 0, 3, 1)


def _world_rank(pp_rank: int, ep_rank: int) -> int:
    """Assign world ranks independently of PP-major topology order."""
    return _WORLD_SLOT_BY_EP[ep_rank] * _PP_SIZE + pp_rank


def _trainer_lane(pp_rank: int) -> tuple[int, ...]:
    return tuple(_world_rank(pp_rank, ep_rank) for ep_rank in range(_EP_SIZE))


def _topologies() -> list[MilesTrainerTopology]:
    return [
        MilesTrainerTopology(
            world_rank=_world_rank(pp, ep),
            pp_rank=pp,
            pp_size=_PP_SIZE,
            tp_rank=0,
            tp_size=1,
            cp_rank=0,
            cp_size=1,
            dense_dp_rank=ep,
            dense_dp_size=_EP_SIZE,
            ep_rank=ep,
            ep_size=_EP_SIZE,
            etp_rank=0,
            etp_size=1,
            expert_dp_rank=0,
            expert_dp_size=1,
            independent_dp_rank=0,
            independent_dp_size=1,
        )
        for pp in range(_PP_SIZE)
        for ep in range(_EP_SIZE)
    ]


def _specs() -> tuple[list[MilesTensorSpec], list[tuple[str, ...]]]:
    specs: list[MilesTensorSpec] = []
    update_units: list[tuple[str, ...]] = []
    for pp in range(_PP_SIZE):
        layer = pp
        dense_owner = _world_rank(pp, 0)
        dense_fc1 = f"native.layers.{layer}.dense.linear_fc1.weight"
        dense_fc2 = f"native.layers.{layer}.dense.linear_fc2.weight"
        update_units.extend(((dense_fc1,), (dense_fc2,)))
        specs.extend(
            [
                MilesTensorSpec(
                    world_rank=dense_owner,
                    source_name=dense_fc1,
                    canonical_name=f"model.layers.{layer}.mlp.gate_proj.weight",
                    family="dense",
                    global_shape=(2048, 4096),
                ),
                MilesTensorSpec(
                    world_rank=dense_owner,
                    source_name=dense_fc1,
                    canonical_name=f"model.layers.{layer}.mlp.up_proj.weight",
                    family="dense",
                    global_shape=(2048, 4096),
                ),
                MilesTensorSpec(
                    world_rank=dense_owner,
                    source_name=dense_fc2,
                    canonical_name=f"model.layers.{layer}.mlp.down_proj.weight",
                    family="dense",
                    global_shape=(4096, 2048),
                ),
            ]
        )

        expert_fc1_names = tuple(
            f"native.layers.{layer}.experts.ep{ep}.linear_fc1.weight"
            for ep in range(_EP_SIZE)
        )
        expert_fc2_names = tuple(
            f"native.layers.{layer}.experts.ep{ep}.linear_fc2.weight"
            for ep in range(_EP_SIZE)
        )
        update_units.extend((expert_fc1_names, expert_fc2_names))
        for ep in range(_EP_SIZE):
            owner = _world_rank(pp, ep)
            specs.extend(
                [
                    MilesTensorSpec(
                        world_rank=owner,
                        source_name=expert_fc1_names[ep],
                        canonical_name=f"model.layers.{layer}.mlp.experts.gate_proj.weight",
                        family="routed_expert",
                        global_shape=(256, 1024, 4096),
                    ),
                    MilesTensorSpec(
                        world_rank=owner,
                        source_name=expert_fc1_names[ep],
                        canonical_name=f"model.layers.{layer}.mlp.experts.up_proj.weight",
                        family="routed_expert",
                        global_shape=(256, 1024, 4096),
                    ),
                    MilesTensorSpec(
                        world_rank=owner,
                        source_name=expert_fc2_names[ep],
                        canonical_name=f"model.layers.{layer}.mlp.experts.down_proj.weight",
                        family="routed_expert",
                        global_shape=(256, 4096, 1024),
                    ),
                ]
            )

    incomplete = "native.layers.0.incomplete.linear_fc1.weight"
    unsupported = "native.layers.0.attention.weight"
    specs.append(
        MilesTensorSpec(
            world_rank=_world_rank(0, 0),
            source_name=incomplete,
            canonical_name="model.layers.0.mlp.shared_expert.gate_proj.weight",
            family="dense",
            global_shape=(2048, 4096),
        )
    )
    update_units.append((incomplete, unsupported))
    update_units.append(("native.final_norm.weight",))
    return specs, update_units


def _duplicate_first_parallel_coordinate(
    records: list[MilesTrainerTopology],
) -> list[MilesTrainerTopology]:
    duplicate = replace(
        records[0],
        pp_rank=records[1].pp_rank,
        dense_dp_rank=records[1].dense_dp_rank,
        ep_rank=records[1].ep_rank,
    )
    return [duplicate, *records[1:]]


def test_reference_topology_projects_exact_pr3304_lane_coordinates() -> None:
    specs, update_units = _specs()

    result = build_miles_reshard_plan(
        _topologies(),
        specs,
        update_units=update_units,
        rollout_engine_sizes=_ROLLOUT_ENGINE_SIZES,
        pp_wave_size=2,
    )

    assert result.plan.source_partition_count == _PP_SIZE
    assert result.trainer_lanes == tuple(_trainer_lane(pp) for pp in range(_PP_SIZE))
    assert result.waves == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert len(result.plan.bulk) == _PP_SIZE * 6
    assert [entry.name for entry in result.plan.bulk] == sorted(
        (entry.name for entry in result.plan.bulk),
        key=lambda name: (int(name.split(".")[2]), name),
    )

    by_name = {entry.name: entry for entry in result.plan.bulk}
    expected_destination_ranks = [
        list(range(start, start + _ROLLOUT_ENGINE_SIZES[0]))
        for start in range(
            _EP_SIZE,
            _EP_SIZE + sum(_ROLLOUT_ENGINE_SIZES),
            _ROLLOUT_ENGINE_SIZES[0],
        )
    ]
    for pp in range(_PP_SIZE):
        stage_routes = [route for route in result.routes if route.partition_id == pp]
        topology_by_world = {
            topology.world_rank: topology for topology in _topologies()
        }
        assert [
            topology_by_world[world_rank].ep_rank
            for world_rank in result.trainer_lanes[pp]
        ] == [0, 1, 2, 3]
        assert {
            route.source_world_ranks
            for route in stage_routes
            if route.family == "dense"
        } == {(_world_rank(pp, 0),)}
        assert {
            route.source_world_ranks
            for route in stage_routes
            if route.family == "routed_expert"
        } == {_trainer_lane(pp)}
        expert_gate_route = next(
            route
            for route in stage_routes
            if route.canonical_name == f"model.layers.{pp}.mlp.experts.gate_proj.weight"
        )
        assert expert_gate_route.source_names_by_world == tuple(
            (
                _world_rank(pp, ep),
                (f"native.layers.{pp}.experts.ep{ep}.linear_fc1.weight",),
            )
            for ep in range(_EP_SIZE)
        )

        dense_gate = by_name[f"model.layers.{pp}.mlp.gate_proj.weight"]
        dense_up = by_name[f"model.layers.{pp}.mlp.up_proj.weight"]
        dense_down = by_name[f"model.layers.{pp}.mlp.down_proj.weight"]
        expert_gate = by_name[f"model.layers.{pp}.mlp.experts.gate_proj.weight"]
        expert_up = by_name[f"model.layers.{pp}.mlp.experts.up_proj.weight"]
        expert_down = by_name[f"model.layers.{pp}.mlp.experts.down_proj.weight"]

        for entry in (dense_gate, dense_up, dense_down):
            assert entry.src_mesh.shape == (1, 1)
            assert entry.src_mesh.rank_offset == 0
            assert entry.src_mesh.nested() == [[0]]
        for entry in (expert_gate, expert_up, expert_down):
            assert entry.src_mesh.shape == (1, 4)
            assert entry.src_mesh.rank_offset == 0
            assert entry.src_mesh.nested() == [[0, 1, 2, 3]]
        for entry in (
            dense_gate,
            dense_up,
            dense_down,
            expert_gate,
            expert_up,
            expert_down,
        ):
            assert entry.dst_mesh.shape == (4, 8)
            assert entry.dst_mesh.rank_offset == 4
            assert entry.dst_mesh.nested() == expected_destination_ranks
            assert entry.group_key == f"pp-wave-{pp // 2}"

        for entry in (dense_gate, dense_up, expert_gate, expert_up, expert_down):
            assert entry.src_placements == (
                Placement.replicate(),
                Placement.shard(0),
            )
            assert entry.dst_placements == (
                Placement.replicate(),
                Placement.shard(0),
            )
        assert dense_down.src_placements == (
            Placement.replicate(),
            Placement.shard(1),
        )
        assert dense_down.dst_placements == (
            Placement.replicate(),
            Placement.shard(1),
        )

    assert result.layer_groups == tuple(
        tuple(entry.name for entry in result.plan.bulk if entry.partition_id in wave)
        for wave in result.waves
    )
    assert result.residual_update_units == (
        ("native.final_norm.weight",),
        (
            "native.layers.0.incomplete.linear_fc1.weight",
            "native.layers.0.attention.weight",
        ),
    )
    assert "model.layers.0.mlp.shared_expert.gate_proj.weight" not in by_name


def test_reference_plan_is_stable_across_input_order() -> None:
    specs, update_units = _specs()
    expected = build_miles_reshard_plan(
        _topologies(),
        specs,
        update_units=update_units,
        rollout_engine_sizes=_ROLLOUT_ENGINE_SIZES,
        pp_wave_size=2,
    )
    actual = build_miles_reshard_plan(
        list(reversed(_topologies())),
        list(reversed(specs)),
        update_units=list(reversed(update_units)),
        rollout_engine_sizes=_ROLLOUT_ENGINE_SIZES,
        pp_wave_size=2,
    )

    assert actual == expected
    assert plan_digest(actual.plan) == plan_digest(expected.plan)


def test_dense_owners_follow_tp_coordinates_under_permuted_world_ranks() -> None:
    topologies = [
        MilesTrainerTopology(
            world_rank=1 - parallel_rank,
            pp_rank=0,
            pp_size=1,
            tp_rank=parallel_rank,
            tp_size=2,
            cp_rank=0,
            cp_size=1,
            dense_dp_rank=0,
            dense_dp_size=1,
            ep_rank=parallel_rank,
            ep_size=2,
            etp_rank=0,
            etp_size=1,
            expert_dp_rank=0,
            expert_dp_size=1,
            independent_dp_rank=0,
            independent_dp_size=1,
        )
        for parallel_rank in range(2)
    ]
    specs = [
        MilesTensorSpec(
            world_rank=1 - parallel_rank,
            source_name=f"native.tp{parallel_rank}.linear_fc1.weight",
            canonical_name=f"model.layers.0.mlp.{component}_proj.weight",
            family="dense",
            global_shape=(8, 4),
        )
        for parallel_rank in range(2)
        for component in ("gate", "up")
    ]

    result = build_miles_reshard_plan(
        topologies,
        specs,
        update_units=[
            (
                "native.tp0.linear_fc1.weight",
                "native.tp1.linear_fc1.weight",
            )
        ],
        rollout_engine_sizes=(2,),
        pp_wave_size=1,
    )

    assert result.trainer_lanes == ((1, 0),)
    gate_route = next(
        route
        for route in result.routes
        if route.canonical_name == "model.layers.0.mlp.gate_proj.weight"
    )
    assert gate_route.source_world_ranks == (1, 0)
    assert gate_route.source_names_by_world == (
        (1, ("native.tp0.linear_fc1.weight",)),
        (0, ("native.tp1.linear_fc1.weight",)),
    )


def test_dense_source_offset_tracks_coordinate_order_not_world_rank() -> None:
    baseline_topologies = _topologies()
    topology_by_world = {
        topology.world_rank: topology for topology in baseline_topologies
    }
    topologies = [
        replace(
            topology,
            dense_dp_rank=(topology.ep_rank - 1) % _EP_SIZE,
        )
        for topology in baseline_topologies
    ]
    specs, update_units = _specs()
    remapped_specs = [
        replace(
            spec,
            world_rank=_world_rank(
                topology_by_world[spec.world_rank].pp_rank,
                1,
            ),
        )
        if spec.family == "dense"
        else spec
        for spec in specs
    ]

    result = build_miles_reshard_plan(
        topologies,
        remapped_specs,
        update_units=update_units,
        rollout_engine_sizes=_ROLLOUT_ENGINE_SIZES,
        pp_wave_size=2,
    )

    dense = next(
        entry
        for entry in result.plan.bulk
        if entry.name == "model.layers.0.mlp.gate_proj.weight"
    )
    assert dense.src_mesh.rank_offset == 1
    assert dense.src_mesh.nested() == [[1]]


def test_missing_dense_up_keeps_the_fused_fc1_unit_residual() -> None:
    specs, update_units = _specs()
    incomplete = [
        spec
        for spec in specs
        if spec.canonical_name != "model.layers.0.mlp.up_proj.weight"
    ]

    result = build_miles_reshard_plan(
        _topologies(),
        incomplete,
        update_units=update_units,
        rollout_engine_sizes=_ROLLOUT_ENGINE_SIZES,
        pp_wave_size=2,
    )

    names = {entry.name for entry in result.plan.bulk}
    assert "model.layers.0.mlp.gate_proj.weight" not in names
    assert "model.layers.0.mlp.up_proj.weight" not in names
    assert ("native.layers.0.dense.linear_fc1.weight",) in (
        result.residual_update_units
    )


def test_fused_gate_up_rejects_incompatible_global_shapes() -> None:
    specs, update_units = _specs()
    incompatible = [
        replace(spec, global_shape=(4096, 4096))
        if spec.canonical_name == "model.layers.0.mlp.up_proj.weight"
        else spec
        for spec in specs
    ]

    with pytest.raises(
        ValueError,
        match="model.layers.0.mlp.*fused gate/up.*global_shape",
    ):
        build_miles_reshard_plan(
            _topologies(),
            incompatible,
            update_units=update_units,
            rollout_engine_sizes=_ROLLOUT_ENGINE_SIZES,
            pp_wave_size=2,
        )


def test_fused_gate_up_rejects_incompatible_dtypes() -> None:
    specs, update_units = _specs()
    incompatible = [
        replace(spec, dtype="float16")
        if spec.canonical_name == "model.layers.0.mlp.up_proj.weight"
        else spec
        for spec in specs
    ]

    with pytest.raises(
        ValueError,
        match="model.layers.0.mlp.*fused gate/up.*dtype",
    ):
        build_miles_reshard_plan(
            _topologies(),
            incompatible,
            update_units=update_units,
            rollout_engine_sizes=_ROLLOUT_ENGINE_SIZES,
            pp_wave_size=2,
        )


def test_fused_gate_up_rejects_different_native_source_mapping() -> None:
    specs, update_units = _specs()
    native_fc1 = "native.layers.0.dense.linear_fc1.weight"
    alias_fc1 = "native.layers.0.dense.linear_fc1.alias.weight"
    incompatible = [
        replace(spec, source_name=alias_fc1)
        if spec.canonical_name == "model.layers.0.mlp.up_proj.weight"
        else spec
        for spec in specs
    ]
    paired_units = [unit for unit in update_units if unit != (native_fc1,)]
    paired_units.append((native_fc1, alias_fc1))

    with pytest.raises(
        ValueError,
        match="model.layers.0.mlp.*fused gate/up.*source mapping",
    ):
        build_miles_reshard_plan(
            _topologies(),
            incompatible,
            update_units=paired_units,
            rollout_engine_sizes=_ROLLOUT_ENGINE_SIZES,
            pp_wave_size=2,
        )


def test_missing_expert_up_keeps_all_ep_fc1_sources_residual() -> None:
    specs, update_units = _specs()
    incomplete = [
        spec
        for spec in specs
        if spec.canonical_name != "model.layers.0.mlp.experts.up_proj.weight"
    ]
    expert_fc1 = tuple(
        f"native.layers.0.experts.ep{ep}.linear_fc1.weight" for ep in range(_EP_SIZE)
    )

    result = build_miles_reshard_plan(
        _topologies(),
        incomplete,
        update_units=update_units,
        rollout_engine_sizes=_ROLLOUT_ENGINE_SIZES,
        pp_wave_size=2,
    )

    names = {entry.name for entry in result.plan.bulk}
    assert "model.layers.0.mlp.experts.gate_proj.weight" not in names
    assert "model.layers.0.mlp.experts.up_proj.weight" not in names
    assert expert_fc1 in result.residual_update_units


def test_missing_down_keeps_a_grouped_dense_ffn_unit_residual() -> None:
    specs, update_units = _specs()
    dense_fc1 = "native.layers.0.dense.linear_fc1.weight"
    dense_fc2 = "native.layers.0.dense.linear_fc2.weight"
    grouped_units = [
        unit for unit in update_units if unit not in ((dense_fc1,), (dense_fc2,))
    ]
    grouped_units.append((dense_fc1, dense_fc2))
    incomplete = [
        spec
        for spec in specs
        if spec.canonical_name != "model.layers.0.mlp.down_proj.weight"
    ]

    result = build_miles_reshard_plan(
        _topologies(),
        incomplete,
        update_units=grouped_units,
        rollout_engine_sizes=_ROLLOUT_ENGINE_SIZES,
        pp_wave_size=2,
    )

    names = {entry.name for entry in result.plan.bulk}
    assert "model.layers.0.mlp.gate_proj.weight" not in names
    assert "model.layers.0.mlp.up_proj.weight" not in names
    assert "model.layers.0.mlp.down_proj.weight" not in names
    assert (dense_fc1, dense_fc2) in result.residual_update_units


def test_one_decoder_layer_cannot_span_multiple_pp_partitions() -> None:
    specs, update_units = _specs()
    conflicting = [
        replace(
            spec,
            canonical_name=spec.canonical_name.replace(
                "model.layers.1.",
                "model.layers.0.",
            ),
        )
        if spec.family == "routed_expert"
        and spec.canonical_name.startswith("model.layers.1.")
        else spec
        for spec in specs
        if not (
            spec.family == "routed_expert"
            and spec.canonical_name.startswith("model.layers.0.")
        )
    ]

    with pytest.raises(
        ValueError,
        match="global decoder layer 0.*PP partitions 0 and 1",
    ):
        build_miles_reshard_plan(
            _topologies(),
            conflicting,
            update_units=update_units,
            rollout_engine_sizes=_ROLLOUT_ENGINE_SIZES,
            pp_wave_size=2,
        )


@pytest.mark.parametrize(
    "mutate,match",
    [
        param(
            lambda records: [],
            "without trainer topology",
            id="empty-topology",
        ),
        param(
            lambda records: records[:-1],
            "topology describes 32 ranks",
            id="missing-rank",
        ),
        param(
            lambda records: [
                replace(records[0], pp_size=7),
                *records[1:],
            ],
            "inconsistent trainer pp sizes",
            id="inconsistent-pp-size",
        ),
        param(
            lambda records: [
                replace(records[0], world_rank=records[1].world_rank),
                *records[1:],
            ],
            "duplicate world ranks",
            id="duplicate-world-rank",
        ),
        param(
            lambda records: [
                replace(records[0], ep_rank=records[0].ep_size),
                *records[1:],
            ],
            "invalid trainer ep rank 4 for size 4",
            id="out-of-range-ep-rank",
        ),
        param(
            _duplicate_first_parallel_coordinate,
            "unique coordinates",
            id="duplicate-parallel-coordinate",
        ),
        param(
            lambda records: [replace(record, etp_size=2) for record in records],
            "requires trainer ETP=1",
            id="unsupported-etp-size",
        ),
    ],
)
def test_malformed_or_inconsistent_trainer_topology_fails_closed(
    mutate: Callable[
        [list[MilesTrainerTopology]],
        list[MilesTrainerTopology],
    ],
    match: str,
) -> None:
    specs, update_units = _specs()

    with pytest.raises(ValueError, match=match):
        build_miles_reshard_plan(
            mutate(_topologies()),
            specs,
            update_units=update_units,
            rollout_engine_sizes=_ROLLOUT_ENGINE_SIZES,
            pp_wave_size=2,
        )


@pytest.mark.parametrize(
    "rollout_engine_sizes,pp_wave_size,match",
    [
        param((), 2, "positive rollout engine sizes", id="no-rollout-engines"),
        param((8, 4), 2, "homogeneous rollout engine sizes", id="mixed-engine-sizes"),
        param(
            _ROLLOUT_ENGINE_SIZES, 0, "pp_wave_size must be positive", id="zero-wave"
        ),
    ],
)
def test_invalid_rollout_topology_fails_closed(
    rollout_engine_sizes: tuple[int, ...],
    pp_wave_size: int,
    match: str,
) -> None:
    specs, update_units = _specs()

    with pytest.raises(ValueError, match=match):
        build_miles_reshard_plan(
            _topologies(),
            specs,
            update_units=update_units,
            rollout_engine_sizes=rollout_engine_sizes,
            pp_wave_size=pp_wave_size,
        )


def test_missing_expert_owner_fails_closed_with_parameter_name() -> None:
    specs, update_units = _specs()
    canonical_name = "model.layers.3.mlp.experts.gate_proj.weight"
    missing_owner = _world_rank(3, 2)
    incomplete_specs = [
        spec
        for spec in specs
        if not (
            spec.canonical_name == canonical_name and spec.world_rank == missing_owner
        )
    ]

    with pytest.raises(
        ValueError,
        match=rf"{canonical_name}.*expected source owners",
    ):
        build_miles_reshard_plan(
            _topologies(),
            incomplete_specs,
            update_units=update_units,
            rollout_engine_sizes=_ROLLOUT_ENGINE_SIZES,
            pp_wave_size=2,
        )


def test_no_complete_atomic_update_unit_fails_closed() -> None:
    specs, _ = _specs()

    with pytest.raises(
        ValueError,
        match="no complete atomic MILES update unit",
    ):
        build_miles_reshard_plan(
            _topologies(),
            specs,
            update_units=[("native.final_norm.weight",)],
            rollout_engine_sizes=_ROLLOUT_ENGINE_SIZES,
            pp_wave_size=2,
        )
