# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MILES/SGLang collective control wire."""

import pytest

from modelexpress_rl.collective.integrations import wire
from modelexpress_rl.collective.integrations.miles import CollectiveTopology
from modelexpress_rl.collective.integrations.wire import (
    CollectiveControl,
    decode_control,
    encode_control,
)
from modelexpress_rl.collective.types import (
    MeshSpec,
    ParamPlan,
    Placement,
    ReshardPlan,
)


def _plan() -> ReshardPlan:
    return ReshardPlan(
        bulk=[
            ParamPlan(
                name="model.layers.0.weight",
                global_shape=(12, 4),
                dtype="bfloat16",
                partition_id=0,
                src_mesh=MeshSpec((1,)),
                src_placements=(Placement.shard(0),),
                dst_mesh=MeshSpec((6,), rank_offset=1),
                dst_placements=(Placement.shard(0),),
                group_key="model.layers.0.weight",
            )
        ],
        source_partition_count=2,
    )


def _topology() -> CollectiveTopology:
    return CollectiveTopology(
        model_name="qwen",
        trainer_slots=("trainer-0", "trainer-1"),
        generator_slots=tuple(f"generator-{rank}" for rank in range(6)),
        source_partition_count=2,
        m2n_abi_version="miles-sglang-bf16-replicated-v1",
    )


def test_control_wire_round_trips_the_exact_plan_and_topology():
    control = CollectiveControl(
        action="prepare",
        plan=_plan(),
        topology=_topology(),
        generator_slot_offset=2,
        endpoint="mx:50051",
    )

    decoded = decode_control(encode_control(control))

    assert decoded == control


def test_control_wire_leaves_stock_sglang_group_names_untouched():
    assert decode_control("miles-pp-0") is None


def test_run_round_control_preserves_the_unverified_production_path():
    control = CollectiveControl(
        action="run_round",
        version="7",
        operation_id="operation-7",
    )

    assert decode_control(encode_control(control)) == control


def test_control_wire_rejects_an_oversized_payload_before_json_decode(monkeypatch):
    monkeypatch.setattr(
        wire.json,
        "loads",
        lambda _value: pytest.fail("oversized payload reached json.loads"),
    )
    value = wire.CONTROL_PREFIX + "x" * wire.MAX_CONTROL_CHARS

    with pytest.raises(ValueError, match="control exceeds"):
        decode_control(value)
