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
    tensor_equality_receipt,
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


def test_prepare_control_round_trips_an_opaque_semantic_manifest_identity():
    control = CollectiveControl(
        action="prepare",
        plan=_plan(),
        topology=_topology(),
        generator_slot_offset=2,
        endpoint="mx:50051",
        semantic_manifest_version="native-records-v1",
        semantic_manifest_digest="ab" * 32,
    )

    decoded = decode_control(encode_control(control))

    assert decoded == control


def test_prepare_control_omits_semantic_manifest_fields_for_legacy_callers():
    control = CollectiveControl(
        action="prepare",
        plan=_plan(),
        topology=_topology(),
        generator_slot_offset=2,
        endpoint="mx:50051",
    )

    payload = wire.json.loads(encode_control(control).removeprefix(wire.CONTROL_PREFIX))

    assert "semantic_manifest_version" not in payload
    assert "semantic_manifest_digest" not in payload
    assert decode_control(encode_control(control)) == control


@pytest.mark.parametrize(
    ("version", "digest", "message"),
    [
        (None, "01" * 32, "must be supplied together"),
        ("native-records-v1", None, "must be supplied together"),
        ("", "01" * 32, "version must be a non-empty string"),
        ("native-records-v1", "not-a-digest", "64 lowercase hexadecimal"),
    ],
)
def test_prepare_control_rejects_an_invalid_semantic_manifest_identity(
    version,
    digest,
    message,
):
    with pytest.raises(ValueError, match=message):
        CollectiveControl(
            action="prepare",
            plan=_plan(),
            topology=_topology(),
            generator_slot_offset=2,
            endpoint="mx:50051",
            semantic_manifest_version=version,
            semantic_manifest_digest=digest,
        )


def test_non_prepare_control_rejects_a_semantic_manifest_identity():
    with pytest.raises(ValueError, match="valid only for prepare"):
        CollectiveControl(
            action="run_round",
            version="7",
            operation_id="operation-7",
            semantic_manifest_version="native-records-v1",
            semantic_manifest_digest="01" * 32,
        )


def test_control_wire_leaves_stock_sglang_group_names_untouched():
    assert decode_control("miles-pp-0") is None


def test_run_round_control_round_trips_the_sorted_tensor_digest_map():
    control = CollectiveControl(
        action="run_round",
        version="7",
        operation_id="operation-7",
        tensor_digests=(
            ("model.layers.0.weight", "01" * 32),
            ("model.layers.1.weight", "ab" * 32),
        ),
    )

    assert decode_control(encode_control(control)) == control


def test_run_round_control_preserves_the_unverified_production_path():
    control = CollectiveControl(
        action="run_round",
        version="7",
        operation_id="operation-7",
    )

    assert decode_control(encode_control(control)) == control


def test_tensor_equality_receipt_is_deterministic_and_operation_scoped():
    tensor_digests = (
        ("model.a", "01" * 32),
        ("model.b", "ab" * 32),
    )

    receipt = tensor_equality_receipt(
        version="7",
        operation_id="operation-7",
        tensor_digests=tensor_digests,
    )

    assert receipt == (
        "aeebffe8bc55e919ded1ac3cbb267874e6e315871efa1d9802245d8444ac4fb6"
    )
    assert receipt != tensor_equality_receipt(
        version="7",
        operation_id="operation-8",
        tensor_digests=tensor_digests,
    )


@pytest.mark.parametrize(
    ("tensor_digests", "message"),
    [
        ((), "requires tensor digests"),
        (
            (
                ("model.layers.1.weight", "ab" * 32),
                ("model.layers.0.weight", "01" * 32),
            ),
            "sorted by tensor name",
        ),
        (
            (
                ("model.layers.0.weight", "01" * 32),
                ("model.layers.0.weight", "ab" * 32),
            ),
            "unique tensor names",
        ),
        (
            (("model.layers.0.weight", "not-a-digest"),),
            "64 lowercase hexadecimal",
        ),
    ],
)
def test_run_round_control_rejects_invalid_tensor_digest_maps(
    tensor_digests,
    message,
):
    with pytest.raises(ValueError, match=message):
        CollectiveControl(
            action="run_round",
            version="7",
            operation_id="operation-7",
            tensor_digests=tensor_digests,
        )


def test_control_wire_rejects_an_oversized_payload_before_json_decode(monkeypatch):
    monkeypatch.setattr(
        wire.json,
        "loads",
        lambda _value: pytest.fail("oversized payload reached json.loads"),
    )
    value = wire.CONTROL_PREFIX + "x" * wire.MAX_CONTROL_CHARS

    with pytest.raises(ValueError, match="control exceeds"):
        decode_control(value)


def test_control_wire_rejects_excessive_digest_entries_before_normalization():
    excessive = [["model.weight", "01" * 32]] * (wire.MAX_TENSOR_DIGESTS + 1)
    value = wire.CONTROL_PREFIX + wire.json.dumps(
        {
            "action": "run_round",
            "version": "7",
            "operation_id": "operation-7",
            "tensor_digests": excessive,
        },
        separators=(",", ":"),
    )

    with pytest.raises(ValueError, match="entry limit"):
        decode_control(value)


def test_control_wire_rejects_oversized_tensor_names():
    oversized_name = "x" * (wire.MAX_TENSOR_NAME_CHARS + 1)
    value = wire.CONTROL_PREFIX + wire.json.dumps(
        {
            "action": "run_round",
            "version": "7",
            "operation_id": "operation-7",
            "tensor_digests": [[oversized_name, "01" * 32]],
        },
        separators=(",", ":"),
    )

    with pytest.raises(ValueError, match="name exceeds"):
        decode_control(value)
