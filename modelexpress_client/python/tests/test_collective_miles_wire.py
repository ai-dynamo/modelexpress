# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MILES/SGLang collective control wire."""

import json

import pytest

from modelexpress_rl.collective.integrations import wire
from modelexpress_rl.collective.integrations.miles import CollectiveTopology
from modelexpress_rl.collective.integrations.wire import (
    CollectiveControl,
    decode_control,
    encode_control,
    plan_from_wire,
    plan_to_wire,
    topology_from_wire,
    topology_to_wire,
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


def test_run_round_control_round_trips_a_plan_free_control():
    control = CollectiveControl(
        action="run_round",
        version="7",
        operation_id="operation-7",
    )

    assert decode_control(encode_control(control)) == control


@pytest.mark.parametrize(
    "key",
    [
        "name",
        "global_shape",
        "dtype",
        "partition_id",
        "src_mesh",
        "src_placements",
        "dst_mesh",
        "dst_placements",
        "group_key",
    ],
)
def test_plan_decoder_rejects_a_bulk_entry_missing_a_required_key(key):
    payload = plan_to_wire(_plan())
    del payload["bulk"][0][key]

    with pytest.raises(ValueError, match="missing"):
        plan_from_wire(payload)


def test_plan_decoder_rejects_a_plan_missing_bulk_or_partition_count():
    with pytest.raises(ValueError, match="missing 'bulk'"):
        plan_from_wire({"source_partition_count": 2})

    payload = plan_to_wire(_plan())
    del payload["source_partition_count"]

    with pytest.raises(ValueError, match="missing 'source_partition_count'"):
        plan_from_wire(payload)


def test_plan_decoder_rejects_geometry_the_encoder_would_not_emit():
    payload = plan_to_wire(_plan())
    payload["source_partition_count"] = 0

    with pytest.raises(ValueError, match="source_partition_count"):
        plan_from_wire(payload)

    payload = plan_to_wire(_plan())
    payload["bulk"][0]["partition_id"] = 7

    with pytest.raises(ValueError, match="partition_id"):
        plan_from_wire(payload)


def test_topology_decoder_rejects_missing_keys():
    payload = topology_to_wire(_topology())
    del payload["m2n_abi_version"]

    with pytest.raises(ValueError, match="missing 'm2n_abi_version'"):
        topology_from_wire(payload)


def test_decode_control_rejects_a_payload_without_an_action():
    encoded = wire.CONTROL_PREFIX + json.dumps({"version": "7"})

    with pytest.raises(ValueError, match="missing 'action'"):
        decode_control(encoded)


def test_control_wire_rejects_an_oversized_payload_before_json_decode(monkeypatch):
    monkeypatch.setattr(
        wire.json,
        "loads",
        lambda _value: pytest.fail("oversized payload reached json.loads"),
    )
    value = wire.CONTROL_PREFIX + "x" * wire.MAX_CONTROL_CHARS

    with pytest.raises(ValueError, match="control exceeds"):
        decode_control(value)


def test_decode_control_coerces_scalar_fields_to_strings():
    encoded = wire.CONTROL_PREFIX + json.dumps(
        {"action": "run_round", "version": 7, "operation_id": 3}
    )

    decoded = decode_control(encoded)

    assert decoded is not None
    assert decoded.version == "7"
    assert decoded.operation_id == "3"
    assert decoded.endpoint is None
    assert decode_control(encode_control(decoded)) == decoded


@pytest.mark.parametrize(
    "shape",
    [
        "124",  # a string would otherwise iterate into per-digit dims
        5,  # a bare int is not iterable geometry
        None,
        [True, 4],  # bools are not integers on this wire
        ["12", 4],
    ],
)
def test_plan_decoder_rejects_a_non_list_of_ints_global_shape(shape):
    payload = plan_to_wire(_plan())
    payload["bulk"][0]["global_shape"] = shape

    with pytest.raises(ValueError, match="global_shape"):
        plan_from_wire(payload)


@pytest.mark.parametrize(
    "partition_id",
    [None, True, "0", 0.0],
)
def test_plan_decoder_rejects_a_non_integer_partition_id(partition_id):
    payload = plan_to_wire(_plan())
    payload["bulk"][0]["partition_id"] = partition_id

    with pytest.raises(ValueError, match="partition_id.*must be an integer"):
        plan_from_wire(payload)


@pytest.mark.parametrize("key", ["name", "dtype"])
def test_plan_decoder_rejects_a_null_string_field(key):
    payload = plan_to_wire(_plan())
    payload["bulk"][0][key] = None

    with pytest.raises(ValueError, match=f"{key}.*must be a string"):
        plan_from_wire(payload)


@pytest.mark.parametrize("mesh", ["src_mesh", "dst_mesh"])
def test_plan_decoder_rejects_non_integer_mesh_fields(mesh):
    payload = plan_to_wire(_plan())
    payload["bulk"][0][mesh]["shape"] = "6"

    with pytest.raises(ValueError, match="shape.*must be a list"):
        plan_from_wire(payload)

    payload = plan_to_wire(_plan())
    payload["bulk"][0][mesh]["rank_offset"] = False

    with pytest.raises(ValueError, match="rank_offset.*must be an integer"):
        plan_from_wire(payload)


def test_plan_decoder_rejects_a_string_placement_list_and_bool_dim():
    payload = plan_to_wire(_plan())
    payload["bulk"][0]["src_placements"] = "S0"

    with pytest.raises(ValueError, match="src_placements.*must be a list"):
        plan_from_wire(payload)

    payload = plan_to_wire(_plan())
    payload["bulk"][0]["src_placements"] = [{"kind": "SHARD", "dim": True}]

    with pytest.raises(ValueError, match="dim.*must be an integer or null"):
        plan_from_wire(payload)


def test_plan_decoder_casts_a_non_null_group_key_to_str_and_keeps_null():
    payload = plan_to_wire(_plan())
    payload["bulk"][0]["group_key"] = 7

    decoded = plan_from_wire(payload)

    assert decoded.bulk[0].group_key == "7"

    payload = plan_to_wire(_plan())
    payload["bulk"][0]["group_key"] = None

    decoded = plan_from_wire(payload)

    assert decoded.bulk[0].group_key is None


@pytest.mark.parametrize("key", ["trainer_slots", "generator_slots"])
def test_topology_decoder_rejects_a_string_slot_list(key):
    payload = topology_to_wire(_topology())
    payload[key] = "trainer-0"  # would otherwise iterate into characters

    with pytest.raises(ValueError, match=f"{key}.*must be a list"):
        topology_from_wire(payload)

    payload = topology_to_wire(_topology())
    payload[key] = ["trainer-0", 1]

    with pytest.raises(ValueError, match=f"{key}.*list of strings"):
        topology_from_wire(payload)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model_name", None),
        ("m2n_abi_version", 2),
        ("receiver_protocol", False),
    ],
)
def test_topology_decoder_rejects_non_string_scalar_fields(key, value):
    payload = topology_to_wire(_topology())
    payload[key] = value

    with pytest.raises(ValueError, match=f"{key}.*must be a string"):
        topology_from_wire(payload)


def test_topology_decoder_rejects_a_non_integer_partition_count():
    payload = topology_to_wire(_topology())
    payload["source_partition_count"] = True

    with pytest.raises(ValueError, match="source_partition_count.*must be an integer"):
        topology_from_wire(payload)


@pytest.mark.parametrize("offset", [None, True, "2"])
def test_decode_control_rejects_a_non_integer_generator_slot_offset(offset):
    encoded = wire.CONTROL_PREFIX + json.dumps(
        {"action": "run_round", "version": "7", "operation_id": "op-7"}
    )
    raw = json.loads(encoded.removeprefix(wire.CONTROL_PREFIX))
    raw["generator_slot_offset"] = offset

    with pytest.raises(ValueError, match="generator_slot_offset"):
        decode_control(wire.CONTROL_PREFIX + json.dumps(raw))
