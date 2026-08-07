# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from rlxfer.errors import IntegrityError, SerializationError
from rlxfer.model import ExperienceBatch, ExperienceMetadata, TensorPayload
from rlxfer.serialization import (
    JsonExperienceSerializer,
    SerializationLimits,
    SerializedExperience,
    validate_metadata,
)


def _batch(array: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> ExperienceBatch:
    return ExperienceBatch(
        metadata=ExperienceMetadata(
            producer_id="rollout-1",
            producer_framework="nemo_rl",
            producer_framework_version="0.test",
            experience_id="01031b86-18f3-4de1-97bc-a7aece118d41",
            created_at=1.0,
        ),
        tensors={"tokens": TensorPayload(array, name="tokens")},
        extensions={
            "nemo_rl": {
                "nested": [{"loss_mask": TensorPayload(np.array([1, 0, 1], dtype=np.uint8))}]
            }
        },
    )


@pytest.mark.unit
def test_external_tensor_round_trip_is_byte_exact_and_deterministic() -> None:
    array = np.arange(24, dtype=np.int16).reshape(4, 6)[:, ::2]
    serializer = JsonExperienceSerializer(inline_threshold=0)

    first = serializer.serialize(_batch(array))
    second = serializer.serialize(_batch(array))
    restored = serializer.deserialize(first)

    assert first.metadata == second.metadata
    assert len(first.buffers) == 2
    assert first.buffers[0].path == ("tensors", "tokens", "data")
    assert first.buffers[0].shape == array.shape
    assert first.buffers[0].stride == (3, 1)
    assert first.buffers[0].dtype == array.dtype.str
    assert first.buffers[0].materialize() == array.tobytes(order="C")

    tokens = restored.tensors["tokens"]
    assert isinstance(tokens.data, np.ndarray)
    assert tokens.data.dtype == array.dtype
    assert tokens.data.shape == array.shape
    assert tokens.data.tobytes() == array.tobytes(order="C")
    assert np.array_equal(tokens.data, array)
    assert tokens.stride == (3, 1)
    assert tokens.layout == "strided"
    assert tokens.data.strides == (6, 2)
    assert tokens.data.flags.c_contiguous
    nested = restored.extensions["nemo_rl"]
    assert isinstance(nested, dict)
    nested_tensor = nested["nested"][0]["loss_mask"]
    assert isinstance(nested_tensor, TensorPayload)
    assert isinstance(nested_tensor.data, np.ndarray)
    assert np.array_equal(nested_tensor.data, np.array([1, 0, 1], dtype=np.uint8))


@pytest.mark.unit
def test_small_tensor_is_inline_and_metadata_can_be_prevalidated() -> None:
    serializer = JsonExperienceSerializer(inline_threshold=1024)
    encoded = serializer.serialize(_batch(np.array([[3.5, -1.0]], dtype=np.float32)))

    assert encoded.buffers == ()
    descriptors = validate_metadata(encoded.metadata)
    assert len(descriptors) == 2
    assert descriptors[0].data is not None
    assert descriptors[0].wire_device == "cpu"
    restored = serializer.deserialize(encoded)
    restored_data = restored.tensors["tokens"].data
    assert isinstance(restored_data, np.ndarray)
    assert np.array_equal(
        restored_data,
        np.array([[3.5, -1.0]], dtype=np.float32),
    )


@pytest.mark.unit
def test_checksum_corruption_is_rejected_before_batch_construction() -> None:
    serializer = JsonExperienceSerializer(inline_threshold=0)
    encoded = serializer.serialize(_batch(np.arange(4, dtype=np.int64)))
    segment = encoded.buffers[0]
    corrupted = bytes([segment.materialize()[0] ^ 0xFF]) + segment.materialize()[1:]
    payload = SerializedExperience(
        metadata=encoded.metadata,
        buffers=(replace(segment, data=corrupted, owner=None), *encoded.buffers[1:]),
    )

    with pytest.raises(IntegrityError, match="checksum mismatch"):
        serializer.deserialize(payload)


@pytest.mark.unit
def test_malformed_catalog_size_and_schema_are_rejected() -> None:
    serializer = JsonExperienceSerializer(inline_threshold=1024)
    encoded = serializer.serialize(_batch(np.arange(4, dtype=np.int32)))
    document = json.loads(encoded.metadata)
    document["tensors"][0]["nbytes"] += 1
    malformed = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()

    with pytest.raises(IntegrityError, match="catalog size mismatch"):
        validate_metadata(malformed)

    document["tensors"][0]["nbytes"] -= 1
    document["schema_version"] = "99.0"
    malformed = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
    with pytest.raises(SerializationError, match="unsupported schema version"):
        serializer.deserialize(SerializedExperience(malformed))


@pytest.mark.unit
def test_external_catalog_and_supplied_segment_must_agree() -> None:
    serializer = JsonExperienceSerializer(inline_threshold=0)
    encoded = serializer.serialize(_batch(np.arange(4, dtype=np.float64)))
    segment = encoded.buffers[0]
    payload = SerializedExperience(
        metadata=encoded.metadata,
        buffers=(replace(segment, original_device="xpu:0"), *encoded.buffers[1:]),
    )

    with pytest.raises(IntegrityError, match="metadata disagrees with catalog"):
        serializer.deserialize(payload)


@pytest.mark.unit
def test_numpy_stride_uses_element_units_and_normalizes_wire_layout() -> None:
    array = np.arange(30, dtype=np.float32).reshape(5, 6)[:, ::2]
    payload = TensorPayload(array)
    serializer = JsonExperienceSerializer()

    assert payload.stride == (6, 2)
    encoded = serializer.serialize(_batch(array))
    restored = serializer.deserialize(encoded).tensors["tokens"]

    assert encoded.buffers[0].stride == (3, 1)
    assert isinstance(restored.data, np.ndarray)
    assert restored.stride == (3, 1)
    assert tuple(size // restored.data.itemsize for size in restored.data.strides) == (3, 1)
    assert restored.layout == "strided"
    assert np.array_equal(restored.data, array)


@pytest.mark.unit
def test_metadata_byte_depth_and_item_limits_apply_in_both_directions() -> None:
    encoded = JsonExperienceSerializer().serialize(_batch(np.arange(4, dtype=np.int16)))

    byte_limited = JsonExperienceSerializer(
        limits=replace(SerializationLimits(), max_metadata_bytes=len(encoded.metadata) - 1)
    )
    with pytest.raises(SerializationError, match=r"metadata size .* exceeds byte limit"):
        byte_limited.deserialize(encoded)
    with pytest.raises(SerializationError, match=r"metadata size .* exceeds byte limit"):
        byte_limited.serialize(_batch(np.arange(4, dtype=np.int16)))

    nested: object = "leaf"
    for _ in range(20):
        nested = [nested]
    deep_batch = _batch(np.arange(4, dtype=np.int16))
    deep_batch.extensions = {"nemo_rl": {"nested": nested}}
    depth_limited = JsonExperienceSerializer(limits=replace(SerializationLimits(), max_depth=12))
    with pytest.raises(SerializationError, match="nesting depth exceeds limit"):
        depth_limited.serialize(deep_batch)

    document = json.loads(encoded.metadata)
    root = document["root"]
    for _ in range(20):
        root = {"$type": "list", "items": [root]}
    document["root"] = root
    deep_metadata = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
    with pytest.raises(SerializationError, match="nesting depth exceeds limit"):
        depth_limited.deserialize(SerializedExperience(deep_metadata))

    item_limited = JsonExperienceSerializer(limits=replace(SerializationLimits(), max_items=20))
    with pytest.raises(SerializationError, match="item count exceeds limit"):
        item_limited.serialize(_batch(np.arange(4, dtype=np.int16)))
    with pytest.raises(SerializationError, match="item count exceeds limit"):
        item_limited.deserialize(encoded)


@pytest.mark.unit
def test_tensor_count_and_byte_limits_apply_before_restore() -> None:
    batch = _batch(np.arange(4, dtype=np.int16))
    defaults = SerializationLimits()

    with pytest.raises(SerializationError, match="tensor count exceeds limit"):
        JsonExperienceSerializer(limits=replace(defaults, max_tensor_count=1)).serialize(batch)
    with pytest.raises(SerializationError, match="per-tensor byte limit"):
        JsonExperienceSerializer(limits=replace(defaults, max_tensor_bytes=7)).serialize(batch)
    with pytest.raises(SerializationError, match=r"total tensor size .* exceeds byte limit"):
        JsonExperienceSerializer(
            limits=replace(defaults, max_tensor_bytes=16, max_total_tensor_bytes=10)
        ).serialize(batch)

    encoded = JsonExperienceSerializer().serialize(batch)
    with pytest.raises(SerializationError, match="tensor count exceeds limit"):
        JsonExperienceSerializer(limits=replace(defaults, max_tensor_count=1)).deserialize(encoded)


@pytest.mark.unit
def test_non_contiguous_torch_stride_is_normalized_when_torch_is_available() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.arange(30, dtype=torch.int64).reshape(5, 6)[:, ::2]
    batch = _batch(np.arange(1, dtype=np.int64))
    batch.tensors = {"tokens": TensorPayload(tensor)}
    serializer = JsonExperienceSerializer()

    encoded = serializer.serialize(batch)
    restored = serializer.deserialize(encoded).tensors["tokens"]

    assert encoded.buffers[0].stride == (3, 1)
    assert isinstance(restored.data, torch.Tensor)
    assert restored.data.is_contiguous()
    assert restored.data.stride() == (3, 1)
    assert restored.stride == (3, 1)
    assert restored.layout == "strided"
    assert torch.equal(restored.data, tensor)
