# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan contract for the NCCL M2N collective path.

Torch-free by design, so these run anywhere. The properties worth protecting
are the ones whose violation is silent at refit time: incomplete coverage,
duplicated parameters, and a digest that fails to notice a plan change.
"""

import pytest
from modelexpress_rl.collective import envs
from modelexpress_rl.collective import (
    MeshSpec,
    MiscParam,
    ParamPlan,
    Placement,
    PlanCoverageError,
    ReshardPlan,
    build_mesh,
    build_param_plan,
    default_placements,
    default_shard_dim,
    generator_rank_offset,
    grouped_expert_name,
    is_bulk_param,
    plan_digest,
    validate_coverage,
)


def param(name="w", *, shape=(8, 4), dtype="bfloat16", partition=0, group_key=None):
    return ParamPlan(
        name=name,
        global_shape=shape,
        dtype=dtype,
        partition_id=partition,
        src_mesh=MeshSpec(shape=(2,), rank_offset=0),
        src_placements=(Placement.shard(0),),
        dst_mesh=MeshSpec(shape=(2,), rank_offset=2),
        dst_placements=(Placement.shard(0),),
        group_key=group_key,
    )


class TestPlacement:
    def test_shard_requires_a_dim_and_replicate_forbids_one(self):
        assert Placement.shard(1).canonical() == "S1"
        assert Placement.replicate().canonical() == "R"
        with pytest.raises(ValueError):
            Placement(Placement.shard(0).kind, None)
        with pytest.raises(ValueError):
            Placement(Placement.replicate().kind, 0)


class TestMeshSpec:
    def test_ranks_are_lane_local_and_start_at_the_offset(self):
        assert MeshSpec(shape=(2, 3), rank_offset=4).ranks() == [4, 5, 6, 7, 8, 9]

    def test_degenerate_shapes_are_rejected(self):
        with pytest.raises(ValueError):
            MeshSpec(shape=())
        with pytest.raises(ValueError):
            MeshSpec(shape=(0,))
        with pytest.raises(ValueError):
            MeshSpec(shape=(2,), rank_offset=-1)


class TestParamPlanValidation:
    def test_a_placement_per_mesh_axis_is_required(self):
        with pytest.raises(ValueError, match="placements for a"):
            ParamPlan(
                name="w",
                global_shape=(8, 4),
                dtype="bfloat16",
                partition_id=0,
                src_mesh=MeshSpec(shape=(2, 2)),
                src_placements=(Placement.shard(0),),
                dst_mesh=MeshSpec(shape=(2,)),
                dst_placements=(Placement.shard(0),),
            )

    def test_sharding_a_dim_the_tensor_lacks_is_rejected(self):
        with pytest.raises(ValueError, match="does not have"):
            ParamPlan(
                name="w",
                global_shape=(8,),
                dtype="bfloat16",
                partition_id=0,
                src_mesh=MeshSpec(shape=(2,)),
                src_placements=(Placement.shard(3),),
                dst_mesh=MeshSpec(shape=(2,)),
                dst_placements=(Placement.replicate(),),
            )

    def test_an_uneven_split_is_rejected(self):
        # 5 rows across 2 ranks has no whole-tensor answer, and NCCL would
        # move a differently-shaped tile than the receiver expects.
        with pytest.raises(ValueError, match="does not divide evenly"):
            ParamPlan(
                name="w",
                global_shape=(5, 4),
                dtype="bfloat16",
                partition_id=0,
                src_mesh=MeshSpec(shape=(2,)),
                src_placements=(Placement.shard(0),),
                dst_mesh=MeshSpec(shape=(1,)),
                dst_placements=(Placement.replicate(),),
            )

    def test_sharding_one_tensor_dim_on_two_mesh_axes_is_rejected(self):
        with pytest.raises(ValueError, match="two mesh axes"):
            ParamPlan(
                name="w",
                global_shape=(8, 4),
                dtype="bfloat16",
                partition_id=0,
                src_mesh=MeshSpec(shape=(2, 2)),
                src_placements=(Placement.shard(0), Placement.shard(0)),
                dst_mesh=MeshSpec(shape=(2,)),
                dst_placements=(Placement.replicate(),),
            )

    def test_a_valid_plan_is_accepted(self):
        assert param().name == "w"

    def test_an_empty_dtype_is_rejected(self):
        with pytest.raises(ValueError, match="dtype"):
            param(dtype="")


class TestReshardPlanValidation:
    def test_invalid_misc_records_are_rejected(self):
        with pytest.raises(ValueError, match="name"):
            MiscParam("", (4,), "bfloat16")
        with pytest.raises(ValueError, match="global_shape"):
            MiscParam("m", (), "bfloat16")
        with pytest.raises(ValueError, match="global_shape"):
            MiscParam("m", (4, 0), "bfloat16")
        with pytest.raises(ValueError, match="dtype"):
            MiscParam("m", (4,), "")

    def test_source_partitions_must_exist(self):
        with pytest.raises(ValueError, match="source_partition_count must be positive"):
            ReshardPlan(source_partition_count=0)
        with pytest.raises(ValueError, match="partition_id must be less than"):
            ReshardPlan(bulk=[param(partition=1)], source_partition_count=1)


class TestCoverage:
    def test_a_plan_that_partitions_the_model_passes(self):
        plan = ReshardPlan(
            bulk=[param("a"), param("b")],
            misc=[MiscParam("c", (4,), "bfloat16")],
        )
        validate_coverage(plan, ["a", "b", "c"])

    def test_a_missing_parameter_is_rejected(self):
        # The worst failure this gate prevents: 'c' never moves, the
        # destination keeps its previous value, and the refit reports success.
        plan = ReshardPlan(bulk=[param("a")], misc=[MiscParam("b", (4,), "bfloat16")])
        with pytest.raises(PlanCoverageError, match="does not cover"):
            validate_coverage(plan, ["a", "b", "c"])

    def test_a_parameter_in_both_lists_is_rejected(self):
        plan = ReshardPlan(bulk=[param("a")], misc=[MiscParam("a", (4,), "bfloat16")])
        with pytest.raises(PlanCoverageError, match="more than once"):
            validate_coverage(plan, ["a"])

    def test_a_duplicate_within_the_bulk_list_is_rejected(self):
        plan = ReshardPlan(bulk=[param("a"), param("a")])
        with pytest.raises(PlanCoverageError, match="more than once"):
            validate_coverage(plan, ["a"])

    def test_a_parameter_the_model_does_not_have_is_rejected(self):
        plan = ReshardPlan(bulk=[param("a"), param("ghost")])
        with pytest.raises(PlanCoverageError, match="does not have"):
            validate_coverage(plan, ["a"])

    def test_duplicate_expected_names_are_rejected(self):
        with pytest.raises(PlanCoverageError, match="model parameter list"):
            validate_coverage(ReshardPlan(bulk=[param("a")]), ["a", "a"])


class TestDigest:
    def test_the_digest_depends_on_bulk_operation_order(self):
        # Both sides execute this list in order. If different orders shared a
        # digest, they could reach READY and enter different NCCL collectives.
        a = ReshardPlan(bulk=[param("a"), param("b")])
        b = ReshardPlan(bulk=[param("b"), param("a")])
        assert plan_digest(a) != plan_digest(b)

    def test_the_digest_depends_on_misc_ordering(self):
        # The misc order is the broadcast payload layout, so two orders are
        # two different plans even with identical contents.
        a = ReshardPlan(misc=[MiscParam("a", (4,), "f"), MiscParam("b", (4,), "f")])
        b = ReshardPlan(misc=[MiscParam("b", (4,), "f"), MiscParam("a", (4,), "f")])
        assert plan_digest(a) != plan_digest(b)

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda p: ReshardPlan(bulk=[param("a", dtype="float16")]), id="dtype"),
            pytest.param(lambda p: ReshardPlan(bulk=[param("a", shape=(16, 4))]), id="shape"),
            pytest.param(lambda p: ReshardPlan(bulk=[param("a", group_key="k")]), id="group_key"),
            pytest.param(lambda p: ReshardPlan(bulk=[param("z")]), id="name"),
            pytest.param(
                lambda p: ReshardPlan(bulk=[param("a")], source_partition_count=2),
                id="partition_count",
            ),
        ],
    )
    def test_any_meaningful_change_moves_the_digest(self, mutate):
        base = ReshardPlan(bulk=[param("a")])
        assert plan_digest(mutate(base)) != plan_digest(base)

    def test_a_changed_partition_moves_the_digest(self):
        base = ReshardPlan(bulk=[param("a", partition=0)], source_partition_count=2)
        moved = ReshardPlan(bulk=[param("a", partition=1)], source_partition_count=2)
        assert plan_digest(moved) != plan_digest(base)

    def test_structurally_distinct_records_cannot_alias_through_delimiters(self):
        # These produced the same string under the old unescaped ``|`` format.
        left = ReshardPlan(misc=[MiscParam("x|1", (2,), "f")])
        right = ReshardPlan(misc=[MiscParam("x", (1,), "2|f")])
        assert plan_digest(left) != plan_digest(right)

    def test_a_changed_placement_moves_the_digest(self):
        base = ReshardPlan(bulk=[param("a")])
        shifted = ParamPlan(
            name="a",
            global_shape=(8, 4),
            dtype="bfloat16",
            partition_id=0,
            src_mesh=MeshSpec(shape=(2,), rank_offset=0),
            src_placements=(Placement.shard(1),),
            dst_mesh=MeshSpec(shape=(2,), rank_offset=2),
            dst_placements=(Placement.shard(0),),
        )
        assert plan_digest(ReshardPlan(bulk=[shifted])) != plan_digest(base)

    def test_a_changed_rank_offset_moves_the_digest(self):
        # The offset is where this side's ranks start in the lane; getting it
        # wrong sends tiles to the wrong ranks.
        base = ReshardPlan(bulk=[param("a")])
        moved = ParamPlan(
            name="a",
            global_shape=(8, 4),
            dtype="bfloat16",
            partition_id=0,
            src_mesh=MeshSpec(shape=(2,), rank_offset=0),
            src_placements=(Placement.shard(0),),
            dst_mesh=MeshSpec(shape=(2,), rank_offset=4),
            dst_placements=(Placement.shard(0),),
        )
        assert plan_digest(ReshardPlan(bulk=[moved])) != plan_digest(base)


class TestDefaultDerivation:
    def test_the_bulk_set_is_ffn_projections(self):
        assert is_bulk_param("model.layers.0.mlp.gate_proj.weight")
        assert is_bulk_param("model.layers.0.mlp.up_proj.weight")
        assert is_bulk_param("model.layers.0.mlp.down_proj.weight")
        assert is_bulk_param("model.layers.0.mlp.experts.3.gate_proj.weight")

    def test_non_ffn_and_shared_experts_ride_the_misc_path(self):
        assert not is_bulk_param("model.embed_tokens.weight")
        assert not is_bulk_param("model.layers.0.self_attn.q_proj.weight")
        assert not is_bulk_param("model.layers.0.input_layernorm.weight")
        assert not is_bulk_param("model.layers.0.mlp.shared_expert.gate_proj.weight")
        # An FP8 scale sibling is not the weight it belongs to.
        assert not is_bulk_param("model.layers.0.mlp.gate_proj.weight_scale_inv")

    def test_column_and_row_parallel_projections_shard_different_dims(self):
        assert default_shard_dim("x.gate_proj.weight") == 0
        assert default_shard_dim("x.up_proj.weight") == 0
        assert default_shard_dim("x.down_proj.weight") == 1
        assert default_shard_dim("x.embed_tokens.weight") is None

    def test_per_expert_weights_collapse_into_one_grouped_entry(self):
        assert (
            grouped_expert_name("model.layers.0.mlp.experts.7.gate_proj.weight")
            == "model.layers.0.mlp.experts.gate_proj.weight"
        )
        assert grouped_expert_name("model.layers.0.mlp.gate_proj.weight") is None
        # An FP8 scale sibling must not be folded into the weight group.
        assert grouped_expert_name("model.layers.0.mlp.experts.7.gate_proj.weight_scale_inv") is None


class TestDefaultMesh:
    def test_size_one_dims_are_dropped_and_survivors_reversed(self):
        # (tp, ep, dp, pp) emit order, reversed into a row-major grid, so tp
        # ends up innermost -- consecutive ranks differ in tp.
        mesh, axis_of = build_mesh(rank_count=8, tp_size=4, dp_size=2)
        assert mesh.shape == (2, 4)
        assert axis_of == {"dp": 0, "tp": 1}

    def test_a_single_rank_side_has_no_active_axes(self):
        mesh, axis_of = build_mesh(rank_count=1)
        assert mesh.shape == (1,)
        assert axis_of == {}

    def test_undeclared_parallelism_is_rejected_rather_than_guessed(self):
        # Four ranks with every size left at 1 is ambiguous -- four data
        # replicas? -- so it is rejected instead of silently flattened.
        with pytest.raises(ValueError, match="does not account for"):
            build_mesh(rank_count=4)

    def test_a_single_active_dim_gives_a_one_dimensional_mesh(self):
        mesh, axis_of = build_mesh(rank_count=4, dp_size=4)
        assert mesh.shape == (4,)
        assert axis_of == {"dp": 0}

    def test_a_parallelism_product_that_misses_ranks_is_rejected(self):
        with pytest.raises(ValueError, match="does not account for"):
            build_mesh(rank_count=8, tp_size=3)

    @pytest.mark.parametrize("sizes", [(-2, -2, 1, 1), (1, 1, 0, 4)])
    def test_non_positive_parallelism_sizes_are_rejected(self, sizes):
        with pytest.raises(ValueError, match="parallelism sizes must be positive"):
            build_mesh(
                rank_count=4,
                tp_size=sizes[0],
                ep_size=sizes[1],
                dp_size=sizes[2],
                pp_size=sizes[3],
            )

    def test_expert_params_shard_the_expert_dim_on_the_ep_axis(self):
        _, axis_of = build_mesh(rank_count=8, ep_size=8)
        placements = default_placements(
            "model.layers.0.mlp.experts.gate_proj.weight", axis_of, ndim=3
        )
        assert placements[axis_of["ep"]].canonical() == "S0"

    def test_expert_params_apply_both_ep_and_tp_placements(self):
        _, axis_of = build_mesh(rank_count=8, tp_size=2, ep_size=4)
        placements = default_placements(
            "model.layers.0.mlp.experts.gate_proj.weight", axis_of, ndim=3
        )
        assert placements[axis_of["ep"]].canonical() == "S0"
        assert placements[axis_of["tp"]].canonical() == "S1"

    def test_one_dimensional_params_replicate(self):
        _, axis_of = build_mesh(rank_count=8, tp_size=8)
        placements = default_placements("model.layers.0.input_layernorm.weight", axis_of, ndim=1)
        assert all(p.canonical() == "R" for p in placements)

    def test_the_default_derivation_produces_a_valid_plan(self):
        src, src_axes = build_mesh(rank_count=4, tp_size=4)
        dst, dst_axes = build_mesh(rank_count=2, rank_offset=4, tp_size=2)
        entry = build_param_plan(
            name="model.layers.0.mlp.gate_proj.weight",
            global_shape=(8, 4),
            dtype="bfloat16",
            partition_id=0,
            src_mesh=src,
            src_axis_of=src_axes,
            dst_mesh=dst,
            dst_axis_of=dst_axes,
        )
        assert entry.src_placements[src_axes["tp"]].canonical() == "S0"
        assert entry.dst_placements[dst_axes["tp"]].canonical() == "S0"


class TestGeneratorRankOffset:
    def test_generators_start_after_the_partition_s_trainers(self):
        # Must agree with the server's assignment rule, or every rank builds a
        # mesh that disagrees with the rank it was handed.
        assert generator_rank_offset(4, 1) == 4
        assert generator_rank_offset(4, 2) == 2
        assert generator_rank_offset(8, 4) == 2

    def test_an_uneven_partitioning_is_rejected(self):
        with pytest.raises(ValueError, match="not divisible"):
            generator_rank_offset(5, 2)
        with pytest.raises(ValueError, match="must be positive"):
            generator_rank_offset(4, 0)
        with pytest.raises(ValueError, match="trainer_count must be positive"):
            generator_rank_offset(0, 1)


class TestEnvironment:
    @pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
    def test_non_finite_deadlines_are_rejected(self, monkeypatch, value):
        monkeypatch.setenv("MX_NCCL_REFIT_GROUP_TIMEOUT_S", value)
        with pytest.raises(ValueError, match="must be positive"):
            _ = envs.MX_NCCL_REFIT_GROUP_TIMEOUT_S

    def test_registration_ttl_defaults_to_three_heartbeats(self, monkeypatch):
        monkeypatch.delenv("MX_NCCL_REFIT_REGISTRATION_TTL_S", raising=False)
        monkeypatch.setenv("MX_HEARTBEAT_INTERVAL_SECS", "7")
        assert envs.MX_NCCL_REFIT_REGISTRATION_TTL_S == 21
