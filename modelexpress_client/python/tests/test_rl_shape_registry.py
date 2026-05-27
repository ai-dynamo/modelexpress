# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from modelexpress.rl_shape_registry import (
    allocate_tensors_from_shape_registry,
    shape_registry_from_tensors,
    torch_dtype_from_string,
)


def test_shape_registry_round_trip_allocates_tensors_on_requested_device():
    tensors = {
        "w1": torch.zeros((2, 3), dtype=torch.bfloat16),
        "w2": torch.ones((1,), dtype=torch.float32),
    }

    registry = shape_registry_from_tensors(tensors)
    allocated = allocate_tensors_from_shape_registry(registry, device="cpu")

    assert allocated["w1"].shape == (2, 3)
    assert allocated["w1"].dtype == torch.bfloat16
    assert allocated["w1"].device.type == "cpu"
    assert allocated["w2"].shape == (1,)
    assert allocated["w2"].dtype == torch.float32


def test_shape_registry_from_tensors_preserves_layout_metadata():
    registry = shape_registry_from_tensors(
        {"experts.w": torch.zeros((4, 2), dtype=torch.float16)},
        tensor_metadata={
            "experts.w": {
                "global_shape": [8, 2],
                "shard_offsets": [4, 0],
                "expert_ids": [4, 5, 6, 7],
                "expert_axis": 0,
            }
        },
    )

    assert registry == {
        "experts.w": {
            "shape": [4, 2],
            "dtype": "torch.float16",
            "global_shape": [8, 2],
            "shard_offsets": [4, 0],
            "expert_ids": [4, 5, 6, 7],
            "expert_axis": 0,
        }
    }


def test_shape_registry_from_tensors_rejects_invalid_layout_metadata():
    tensors = {"w": torch.zeros((2, 3), dtype=torch.float32)}

    with pytest.raises(ValueError, match="unknown tensors"):
        shape_registry_from_tensors(tensors, tensor_metadata={"missing": {}})

    with pytest.raises(ValueError, match="tensor shape"):
        shape_registry_from_tensors(
            tensors,
            tensor_metadata={"w": {"shape": [3, 2]}},
        )

    with pytest.raises(ValueError, match="tensor dtype"):
        shape_registry_from_tensors(
            tensors,
            tensor_metadata={"w": {"dtype": "torch.float16"}},
        )


def test_torch_dtype_from_string_accepts_torch_prefix():
    assert torch_dtype_from_string("torch.bfloat16") is torch.bfloat16
    assert torch_dtype_from_string("float16") is torch.float16


def test_torch_dtype_from_string_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="unsupported tensor dtype"):
        torch_dtype_from_string("not_a_dtype")
