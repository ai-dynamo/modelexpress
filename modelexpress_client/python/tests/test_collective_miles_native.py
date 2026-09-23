# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from modelexpress_rl.collective import MeshSpec, ParamPlan, Placement, ReshardPlan
from modelexpress_rl.collective.integrations import (
    CollectiveTopology,
    MilesNativePublisher,
    MilesSourceRecipe,
    inventory_miles_native_tensors,
)
from modelexpress_rl.collective.integrations.miles_topology import (
    MilesReshardTopologyPlan,
    MilesSourceRoute,
    MilesTensorSpec,
    MilesTrainerTopology,
    build_miles_reshard_plan,
)


class FakeCudaTensor(torch.Tensor):
    @staticmethod
    def wrap(tensor: torch.Tensor, *, device: str = "cuda:0") -> FakeCudaTensor:
        if device != "cuda:0":
            raise ValueError("the focused fake supports cuda:0 only")
        result = torch.Tensor._make_subclass(FakeCudaTensor, tensor, False)
        return result

    @property
    def device(self) -> torch.device:
        return torch.device("cuda:0")


class MetadataOnlyTensor:
    shape = (8, 4)
    dtype = torch.bfloat16
    partition_dim = -1
    partition_stride = 1

    def data_ptr(self):
        raise AssertionError("metadata inventory must not inspect storage")


def _cuda(values, *, dtype=torch.bfloat16) -> FakeCudaTensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach().clone().to(dtype=dtype)
    else:
        tensor = torch.tensor(values, dtype=dtype)
    return FakeCudaTensor.wrap(tensor)


def _native_name(layer: int, suffix: str) -> str:
    return f"module.module.decoder.layers.{layer}.mlp.{suffix}"


def test_inventory_is_metadata_only_and_applies_pr3304_partition_corrections() -> None:
    dense_fc1 = MetadataOnlyTensor()
    dense_fc2 = MetadataOnlyTensor()
    expert_fc1 = MetadataOnlyTensor()

    records = inventory_miles_native_tensors(
        [
            (_native_name(0, "linear_fc2.weight"), dense_fc2),
            (_native_name(0, "linear_fc1.weight"), dense_fc1),
            (_native_name(0, "experts.linear_fc1.weight12"), expert_fc1),
            (_native_name(0, "attention.linear_qkv.weight"), MetadataOnlyTensor()),
        ],
        source_keys={
            _native_name(0, "linear_fc1.weight"): "local.dense.fc1",
            _native_name(0, "linear_fc2.weight"): "local.dense.fc2",
            _native_name(0, "experts.linear_fc1.weight12"): "local.expert.12.fc1",
        },
    )

    assert [record.native_name for record in records] == [
        _native_name(0, "experts.linear_fc1.weight12"),
        _native_name(0, "linear_fc1.weight"),
        _native_name(0, "linear_fc2.weight"),
    ]
    by_name = {record.native_name: record for record in records}
    assert by_name[_native_name(0, "linear_fc1.weight")].partition_dim == 0
    assert by_name[_native_name(0, "linear_fc1.weight")].partition_stride == 2
    assert by_name[_native_name(0, "linear_fc2.weight")].partition_dim == 1
    assert by_name[_native_name(0, "linear_fc2.weight")].partition_stride == 1
    assert by_name[_native_name(0, "experts.linear_fc1.weight12")].expert_id == 12
    assert by_name[_native_name(0, "linear_fc1.weight")].source_key == "local.dense.fc1"


@pytest.mark.parametrize(
    ("suffix", "partition_dim", "partition_stride"),
    [
        ("linear_fc1.weight", 1, 1),
        ("linear_fc2.weight", -1, 2),
    ],
)
def test_inventory_rejects_contradictory_dense_partition_metadata(
    suffix,
    partition_dim,
    partition_stride,
) -> None:
    tensor = MetadataOnlyTensor()
    tensor.partition_dim = partition_dim
    tensor.partition_stride = partition_stride

    with pytest.raises(ValueError, match="partition"):
        inventory_miles_native_tensors([(_native_name(0, suffix), tensor)])


def _bf16_topology() -> MilesReshardTopologyPlan:
    topologies = [
        MilesTrainerTopology(
            world_rank=11,
            pp_rank=0,
            pp_size=1,
            tp_rank=0,
            tp_size=1,
            cp_rank=0,
            cp_size=1,
            dense_dp_rank=0,
            dense_dp_size=2,
            ep_rank=0,
            ep_size=2,
            etp_rank=0,
            etp_size=1,
            expert_dp_rank=0,
            expert_dp_size=1,
            independent_dp_rank=0,
            independent_dp_size=1,
        ),
        MilesTrainerTopology(
            world_rank=4,
            pp_rank=0,
            pp_size=1,
            tp_rank=0,
            tp_size=1,
            cp_rank=0,
            cp_size=1,
            dense_dp_rank=1,
            dense_dp_size=2,
            ep_rank=1,
            ep_size=2,
            etp_rank=0,
            etp_size=1,
            expert_dp_rank=0,
            expert_dp_size=1,
            independent_dp_rank=0,
            independent_dp_size=1,
        ),
    ]
    dense_fc1 = _native_name(0, "linear_fc1.weight")
    dense_fc2 = _native_name(0, "linear_fc2.weight")
    specs = [
        MilesTensorSpec(
            world_rank=11,
            source_name=dense_fc1,
            canonical_name=f"model.layers.0.mlp.{component}_proj.weight",
            family="dense",
            global_shape=(4, 3),
        )
        for component in ("gate", "up")
    ]
    specs.append(
        MilesTensorSpec(
            world_rank=11,
            source_name=dense_fc2,
            canonical_name="model.layers.0.mlp.down_proj.weight",
            family="dense",
            global_shape=(3, 4),
        )
    )
    expert_fc1_names = []
    expert_fc2_names = []
    for owner, expert_ids in ((11, (0, 1)), (4, (2, 3))):
        for expert_id in expert_ids:
            fc1 = _native_name(0, f"experts.linear_fc1.weight{expert_id}")
            fc2 = _native_name(0, f"experts.linear_fc2.weight{expert_id}")
            expert_fc1_names.append(fc1)
            expert_fc2_names.append(fc2)
            specs.extend(
                [
                    MilesTensorSpec(
                        world_rank=owner,
                        source_name=fc1,
                        canonical_name=(
                            f"model.layers.0.mlp.experts.{component}_proj.weight"
                        ),
                        family="routed_expert",
                        global_shape=(4, 2, 3),
                    )
                    for component in ("gate", "up")
                ]
            )
            specs.append(
                MilesTensorSpec(
                    world_rank=owner,
                    source_name=fc2,
                    canonical_name="model.layers.0.mlp.experts.down_proj.weight",
                    family="routed_expert",
                    global_shape=(4, 3, 2),
                )
            )
    return build_miles_reshard_plan(
        topologies,
        specs,
        update_units=[
            (dense_fc1,),
            (dense_fc2,),
            tuple(expert_fc1_names),
            tuple(expert_fc2_names),
        ],
        rollout_engine_sizes=(2,),
        pp_wave_size=1,
    )


def _collective_topology(
    topology: MilesReshardTopologyPlan,
    *,
    generator_count: int | None = None,
) -> CollectiveTopology:
    trainer_slots = tuple(
        f"trainer-{world_rank}"
        for lane in topology.trainer_lanes
        for world_rank in lane
    )
    if generator_count is None:
        generator_ranks = topology.plan.bulk[0].dst_mesh.ranks()
        generator_count = len(generator_ranks)
    return CollectiveTopology(
        model_name="test-model",
        trainer_slots=trainer_slots,
        generator_slots=tuple(f"generator-{index}" for index in range(generator_count)),
        source_partition_count=topology.plan.source_partition_count,
        m2n_abi_version="test-abi",
    )


def _rank_11_weights(marker: int = 0) -> dict[str, FakeCudaTensor]:
    dense_fc1 = _cuda(
        [
            [1 + marker, 2, 3],
            [4, 5, 6],
            [7 + marker, 8, 9],
            [10, 11, 12],
            [21 + marker, 22, 23],
            [24, 25, 26],
            [27 + marker, 28, 29],
            [30, 31, 32],
        ]
    )
    dense_fc2 = _cuda(
        [
            [41 + marker, 42, 43, 44],
            [45, 46, 47, 48],
            [49, 50, 51, 52],
        ]
    )
    expert_fc1_0 = _cuda(
        [
            [61 + marker, 62, 63],
            [64, 65, 66],
            [71 + marker, 72, 73],
            [74, 75, 76],
        ]
    )
    expert_fc1_1 = _cuda(
        [
            [81 + marker, 82, 83],
            [84, 85, 86],
            [91 + marker, 92, 93],
            [94, 95, 96],
        ]
    )
    expert_fc2_0 = _cuda([[101 + marker, 102], [103, 104], [105, 106]])
    expert_fc2_1 = _cuda([[111 + marker, 112], [113, 114], [115, 116]])
    return {
        "local.dense.fc1": dense_fc1,
        "local.dense.fc2": dense_fc2,
        "local.expert.0.fc1": expert_fc1_0,
        "local.expert.1.fc1": expert_fc1_1,
        "local.expert.0.fc2": expert_fc2_0,
        "local.expert.1.fc2": expert_fc2_1,
    }


def _rank_11_records(weights: dict[str, FakeCudaTensor]):
    native_by_key = {
        "local.dense.fc1": _native_name(0, "linear_fc1.weight"),
        "local.dense.fc2": _native_name(0, "linear_fc2.weight"),
        "local.expert.0.fc1": _native_name(0, "experts.linear_fc1.weight0"),
        "local.expert.1.fc1": _native_name(0, "experts.linear_fc1.weight1"),
        "local.expert.0.fc2": _native_name(0, "experts.linear_fc2.weight0"),
        "local.expert.1.fc2": _native_name(0, "experts.linear_fc2.weight1"),
    }
    return inventory_miles_native_tensors(
        [(native_by_key[key], tensor) for key, tensor in weights.items()],
        source_keys={native: key for key, native in native_by_key.items()},
    )


def test_publisher_prepares_native_dense_and_expert_sources_and_rebinds_specs() -> None:
    topology = _bf16_topology()
    first_weights = _rank_11_weights()
    publisher = MilesNativePublisher(
        topology=topology,
        collective_topology=_collective_topology(topology),
        source_partition=0,
        source_world_rank=11,
        inventory=_rank_11_records(first_weights),
        device="cuda:0",
    )

    specs = publisher.local_params()
    assert all(spec.base is None for spec in specs.values())
    assert publisher.capture() == topology.plan
    assert publisher.parameter_names() == topology.plan.parameter_names()

    publisher.refresh(first_weights)
    publisher.start_new_round("version-1")

    assert publisher.local_params() == specs
    assert publisher.local_params()[
        "model.layers.0.mlp.gate_proj.weight"
    ].base.tolist() == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ]
    assert publisher.local_params()[
        "model.layers.0.mlp.up_proj.weight"
    ].base.tolist() == [
        [21, 22, 23],
        [24, 25, 26],
        [27, 28, 29],
        [30, 31, 32],
    ]
    assert publisher.local_params()[
        "model.layers.0.mlp.experts.gate_proj.weight"
    ].base.tolist() == [
        [[61, 62, 63], [64, 65, 66]],
        [[81, 82, 83], [84, 85, 86]],
    ]
    assert publisher.local_params()[
        "model.layers.0.mlp.experts.up_proj.weight"
    ].base.tolist() == [
        [[71, 72, 73], [74, 75, 76]],
        [[91, 92, 93], [94, 95, 96]],
    ]
    assert publisher.local_params()[
        "model.layers.0.mlp.experts.down_proj.weight"
    ].base.tolist() == [
        [[101, 102], [103, 104], [105, 106]],
        [[111, 112], [113, 114], [115, 116]],
    ]
    assert {binding.recipe for binding in publisher.source_bindings()} == {
        MilesSourceRecipe.DENSE_FC1_GATE,
        MilesSourceRecipe.DENSE_FC1_UP,
        MilesSourceRecipe.DENSE_FC2,
        MilesSourceRecipe.EXPERT_FC1_GATE,
        MilesSourceRecipe.EXPERT_FC1_UP,
        MilesSourceRecipe.EXPERT_FC2,
    }

    first_gate = specs["model.layers.0.mlp.gate_proj.weight"].base
    second_weights = _rank_11_weights(marker=200)
    publisher.refresh(second_weights)
    publisher.start_new_round("version-2")

    assert publisher.local_params() == specs
    assert specs["model.layers.0.mlp.gate_proj.weight"].base is not first_gate
    assert specs["model.layers.0.mlp.gate_proj.weight"].base[0, 0].item() == 201


def test_publisher_uses_supplied_manifest_recipe_as_single_authority() -> None:
    topology = _bf16_topology()
    weights = _rank_11_weights()
    recipes = {
        entry.name: {
            "model.layers.0.mlp.gate_proj.weight": MilesSourceRecipe.DENSE_FC1_UP,
            "model.layers.0.mlp.up_proj.weight": MilesSourceRecipe.DENSE_FC1_GATE,
            "model.layers.0.mlp.down_proj.weight": MilesSourceRecipe.DENSE_FC2,
            "model.layers.0.mlp.experts.gate_proj.weight": (
                MilesSourceRecipe.EXPERT_FC1_GATE
            ),
            "model.layers.0.mlp.experts.up_proj.weight": (
                MilesSourceRecipe.EXPERT_FC1_UP
            ),
            "model.layers.0.mlp.experts.down_proj.weight": (
                MilesSourceRecipe.EXPERT_FC2
            ),
        }[entry.name]
        for entry in topology.plan.bulk
    }
    publisher = MilesNativePublisher(
        topology=topology,
        collective_topology=_collective_topology(topology),
        source_partition=0,
        source_world_rank=11,
        inventory=_rank_11_records(weights),
        device="cuda:0",
        source_recipes=recipes,
    )

    publisher.refresh(weights)

    assert publisher.local_params()[
        "model.layers.0.mlp.gate_proj.weight"
    ].base.tolist() == [
        [21, 22, 23],
        [24, 25, 26],
        [27, 28, 29],
        [30, 31, 32],
    ]
    assert (
        next(
            binding
            for binding in publisher.source_bindings()
            if binding.canonical_name == "model.layers.0.mlp.gate_proj.weight"
        ).recipe
        is MilesSourceRecipe.DENSE_FC1_UP
    )


def test_publisher_rejects_incomplete_manifest_recipe_mapping() -> None:
    topology = _bf16_topology()
    weights = _rank_11_weights()

    with pytest.raises(ValueError, match="must exactly cover"):
        MilesNativePublisher(
            topology=topology,
            collective_topology=_collective_topology(topology),
            source_partition=0,
            source_world_rank=11,
            inventory=_rank_11_records(weights),
            device="cuda:0",
            source_recipes={
                "model.layers.0.mlp.gate_proj.weight": (
                    MilesSourceRecipe.DENSE_FC1_GATE
                )
            },
        )


def test_publisher_emits_specs_only_for_entries_owned_by_this_world_rank() -> None:
    topology = _bf16_topology()
    rank_4_names = {
        _native_name(0, "experts.linear_fc1.weight2"): "local.expert.2.fc1",
        _native_name(0, "experts.linear_fc1.weight3"): "local.expert.3.fc1",
        _native_name(0, "experts.linear_fc2.weight2"): "local.expert.2.fc2",
        _native_name(0, "experts.linear_fc2.weight3"): "local.expert.3.fc2",
    }
    tensors = {
        "local.expert.2.fc1": _cuda(torch.zeros(4, 3)),
        "local.expert.3.fc1": _cuda(torch.zeros(4, 3)),
        "local.expert.2.fc2": _cuda(torch.zeros(3, 2)),
        "local.expert.3.fc2": _cuda(torch.zeros(3, 2)),
    }
    records = inventory_miles_native_tensors(
        [(native, tensors[key]) for native, key in rank_4_names.items()],
        source_keys=rank_4_names,
    )

    publisher = MilesNativePublisher(
        topology=topology,
        collective_topology=_collective_topology(topology),
        source_partition=0,
        source_world_rank=4,
        inventory=records,
        device="cuda:0",
    )

    assert list(publisher.local_params()) == [
        "model.layers.0.mlp.experts.down_proj.weight",
        "model.layers.0.mlp.experts.gate_proj.weight",
        "model.layers.0.mlp.experts.up_proj.weight",
    ]


def test_refresh_fails_atomically_on_missing_or_drifted_native_storage() -> None:
    topology = _bf16_topology()
    weights = _rank_11_weights()
    publisher = MilesNativePublisher(
        topology=topology,
        collective_topology=_collective_topology(topology),
        source_partition=0,
        source_world_rank=11,
        inventory=_rank_11_records(weights),
        device="cuda:0",
    )
    with pytest.raises(RuntimeError, match="refresh"):
        publisher.start_new_round("version-without-refresh")

    publisher.refresh(weights)
    publisher.start_new_round("version-1")
    previous = {name: spec.base for name, spec in publisher.local_params().items()}

    missing = dict(weights)
    del missing["local.expert.1.fc2"]
    with pytest.raises(KeyError, match="local.expert.1.fc2"):
        publisher.refresh(missing)
    assert {
        name: spec.base for name, spec in publisher.local_params().items()
    } == previous

    drifted = dict(weights)
    drifted["local.dense.fc1"] = _cuda(torch.zeros(8, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="dtype"):
        publisher.refresh(drifted)
    assert {
        name: spec.base for name, spec in publisher.local_params().items()
    } == previous


def test_publisher_rejects_native_inventory_layer_drift() -> None:
    topology = _bf16_topology()
    weights = _rank_11_weights()
    records = list(_rank_11_records(weights))
    records[0] = replace(records[0], layer=7)

    with pytest.raises(ValueError, match="layer"):
        MilesNativePublisher(
            topology=topology,
            collective_topology=_collective_topology(topology),
            source_partition=0,
            source_world_rank=11,
            inventory=records,
            device="cuda:0",
        )


def test_publisher_rejects_expert_partition_metadata_drift() -> None:
    topology = _bf16_topology()
    weights = _rank_11_weights()
    records = list(_rank_11_records(weights))
    expert_index = next(
        index
        for index, record in enumerate(records)
        if record.family == "routed_expert" and record.projection == "fc1"
    )
    records[expert_index] = replace(
        records[expert_index],
        partition_dim=1,
        partition_stride=1,
    )

    with pytest.raises(ValueError, match="expert native partition metadata"):
        MilesNativePublisher(
            topology=topology,
            collective_topology=_collective_topology(topology),
            source_partition=0,
            source_world_rank=11,
            inventory=records,
            device="cuda:0",
        )


def _fp8_topology(
    *,
    drop_last_scale: bool = False,
    scale_first: bool = False,
) -> MilesReshardTopologyPlan:
    native_by_component = {
        "gate": (
            _native_name(0, "experts.linear_fc1.weight0"),
            _native_name(0, "experts.linear_fc1.weight1"),
        ),
        "up": (
            _native_name(0, "experts.linear_fc1.weight0"),
            _native_name(0, "experts.linear_fc1.weight1"),
        ),
        "down": (
            _native_name(0, "experts.linear_fc2.weight0"),
            _native_name(0, "experts.linear_fc2.weight1"),
        ),
    }
    entries = []
    routes = []
    for component in ("gate", "up", "down"):
        weight_name = f"model.layers.0.mlp.experts.{component}_proj.weight"
        weight_shape = (2, 128, 128)
        scale_shape = (2, 1, 1)
        pair_entries = (
            ("weight", weight_name, "float8_e4m3fn", weight_shape),
            ("scale", f"{weight_name}_scale_inv", "float32", scale_shape),
        )
        if scale_first:
            pair_entries = tuple(reversed(pair_entries))
        for role, name, dtype, shape in pair_entries:
            if drop_last_scale and component == "down" and role == "scale":
                continue
            entries.append(
                ParamPlan(
                    name=name,
                    global_shape=shape,
                    dtype=dtype,
                    partition_id=0,
                    src_mesh=MeshSpec(shape=(1, 1), rank_offset=0),
                    src_placements=(Placement.replicate(), Placement.shard(0)),
                    dst_mesh=MeshSpec(shape=(1, 1), rank_offset=1),
                    dst_placements=(Placement.replicate(), Placement.shard(0)),
                    group_key="pp-wave-0",
                )
            )
            routes.append(
                MilesSourceRoute(
                    canonical_name=name,
                    family="routed_expert",
                    partition_id=0,
                    source_world_ranks=(7,),
                    source_names_by_world=((7, native_by_component[component]),),
                )
            )
    return MilesReshardTopologyPlan(
        plan=ReshardPlan(bulk=entries, source_partition_count=1),
        trainer_lanes=((7,),),
        routes=tuple(routes),
        residual_update_units=(),
        waves=((0,),),
        layer_groups=(tuple(entry.name for entry in entries),),
    )


def test_fp8_weight_scale_pairs_are_atomic_and_quantized_once_per_projection() -> None:
    weights = {
        "fc1.0": _cuda(torch.ones(256, 128)),
        "fc1.1": _cuda(torch.full((256, 128), 2)),
        "fc2.0": _cuda(torch.full((128, 128), 3)),
        "fc2.1": _cuda(torch.full((128, 128), 4)),
    }
    names = {
        _native_name(0, "experts.linear_fc1.weight0"): "fc1.0",
        _native_name(0, "experts.linear_fc1.weight1"): "fc1.1",
        _native_name(0, "experts.linear_fc2.weight0"): "fc2.0",
        _native_name(0, "experts.linear_fc2.weight1"): "fc2.1",
    }
    records = inventory_miles_native_tensors(
        [(native, weights[key]) for native, key in names.items()],
        source_keys=names,
    )
    quantized = []

    def quantizer(logical):
        quantized.append(logical.clone())
        marker = logical.flatten()[0].item()
        return (
            _cuda(
                torch.full(logical.shape, marker, dtype=torch.float8_e4m3fn),
                dtype=torch.float8_e4m3fn,
            ),
            _cuda(
                torch.full(
                    (
                        logical.shape[0],
                        logical.shape[1] // 128,
                        logical.shape[2] // 128,
                    ),
                    marker,
                    dtype=torch.float32,
                ),
                dtype=torch.float32,
            ),
        )

    topology = _fp8_topology()
    publisher = MilesNativePublisher(
        topology=topology,
        collective_topology=_collective_topology(topology),
        source_partition=0,
        source_world_rank=7,
        inventory=records,
        device="cuda:0",
        fp8_quantizer=quantizer,
    )

    publisher.refresh(weights)
    publisher.start_new_round("version-1")

    assert len(quantized) == 3
    bindings = publisher.source_bindings()
    assert {(binding.pair_id, binding.tensor_role) for binding in bindings} == {
        (
            f"model.layers.0.mlp.experts.{component}_proj.weight",
            role,
        )
        for component in ("gate", "up", "down")
        for role in ("weight", "scale")
    }
    specs = publisher.local_params()
    for component in ("gate", "up", "down"):
        weight_name = f"model.layers.0.mlp.experts.{component}_proj.weight"
        assert specs[weight_name].base.dtype == torch.float8_e4m3fn
        assert specs[f"{weight_name}_scale_inv"].base.dtype == torch.float32


def test_fp8_pair_quantization_is_independent_of_scale_first_plan_order() -> None:
    weights = {
        "fc1.0": _cuda(torch.ones(256, 128)),
        "fc1.1": _cuda(torch.full((256, 128), 2)),
        "fc2.0": _cuda(torch.full((128, 128), 3)),
        "fc2.1": _cuda(torch.full((128, 128), 4)),
    }
    names = {
        _native_name(0, "experts.linear_fc1.weight0"): "fc1.0",
        _native_name(0, "experts.linear_fc1.weight1"): "fc1.1",
        _native_name(0, "experts.linear_fc2.weight0"): "fc2.0",
        _native_name(0, "experts.linear_fc2.weight1"): "fc2.1",
    }
    records = inventory_miles_native_tensors(
        [(native, weights[key]) for native, key in names.items()],
        source_keys=names,
    )
    quantized = []

    def quantizer(logical):
        quantized.append(logical.clone())
        return (
            _cuda(
                torch.zeros(logical.shape, dtype=torch.float8_e4m3fn),
                dtype=torch.float8_e4m3fn,
            ),
            _cuda(
                torch.zeros(
                    (
                        logical.shape[0],
                        logical.shape[1] // 128,
                        logical.shape[2] // 128,
                    ),
                    dtype=torch.float32,
                ),
                dtype=torch.float32,
            ),
        )

    topology = _fp8_topology(scale_first=True)
    publisher = MilesNativePublisher(
        topology=topology,
        collective_topology=_collective_topology(topology),
        source_partition=0,
        source_world_rank=7,
        inventory=records,
        device="cuda:0",
        fp8_quantizer=quantizer,
    )

    publisher.refresh(weights)

    assert len(quantized) == 3
    assert all(
        binding.tensor_role == "scale" for binding in publisher.source_bindings()[::2]
    )


def test_fp8_publisher_rejects_an_incomplete_atomic_module() -> None:
    topology = _fp8_topology(drop_last_scale=True)
    weights = {
        "fc1.0": _cuda(torch.ones(256, 128)),
        "fc1.1": _cuda(torch.ones(256, 128)),
        "fc2.0": _cuda(torch.ones(128, 128)),
        "fc2.1": _cuda(torch.ones(128, 128)),
    }
    names = {
        _native_name(0, "experts.linear_fc1.weight0"): "fc1.0",
        _native_name(0, "experts.linear_fc1.weight1"): "fc1.1",
        _native_name(0, "experts.linear_fc2.weight0"): "fc2.0",
        _native_name(0, "experts.linear_fc2.weight1"): "fc2.1",
    }
    records = inventory_miles_native_tensors(
        [(native, weights[key]) for native, key in names.items()],
        source_keys=names,
    )

    with pytest.raises(ValueError, match="incomplete"):
        MilesNativePublisher(
            topology=topology,
            collective_topology=_collective_topology(topology),
            source_partition=0,
            source_world_rank=7,
            inventory=records,
            device="cuda:0",
            fp8_quantizer=lambda logical: (logical, logical),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda topology: replace(
                topology,
                routes=tuple(
                    replace(route, source_world_ranks=(99,))
                    if route.canonical_name.endswith("gate_proj.weight")
                    else route
                    for route in topology.routes
                ),
            ),
            "ownership",
        ),
        (
            lambda topology: replace(
                topology,
                routes=topology.routes[:-1],
            ),
            "exactly cover",
        ),
    ],
)
def test_publisher_fails_closed_on_topology_ownership_drift(mutate, message) -> None:
    weights = _rank_11_weights()
    with pytest.raises(ValueError, match=message):
        MilesNativePublisher(
            topology=(topology := mutate(_bf16_topology())),
            collective_topology=_collective_topology(topology),
            source_partition=0,
            source_world_rank=11,
            inventory=_rank_11_records(weights),
            device="cuda:0",
        )


def test_publisher_requires_valid_collective_topology_during_initialization() -> None:
    topology = _bf16_topology()
    weights = _rank_11_weights()

    with pytest.raises(ValueError, match="destination mesh"):
        MilesNativePublisher(
            topology=topology,
            collective_topology=_collective_topology(topology, generator_count=1),
            source_partition=0,
            source_world_rank=11,
            inventory=_rank_11_records(weights),
            device="cuda:0",
        )
