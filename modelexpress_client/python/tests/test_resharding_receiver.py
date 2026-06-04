# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from modelexpress.resharding import SliceOwnership, plan_segments
from modelexpress.resharding_receiver import (
    build_receiver_requests_from_runtime_tensors,
    install_segment_payloads_into_runtime_tensors,
)


def _owner(source_range, *, source_id):
    return SliceOwnership(
        model_name="qwen",
        model_version="step-7",
        tensor_name="weight",
        global_shape=(6, 4),
        dtype="float32",
        source_range=source_range,
        worker_id=f"{source_id}-worker",
        source_id=source_id,
        source_lease=f"{source_id}-lease",
        nixl_descriptor_id=f"{source_id}-desc",
        layout_tags={"storage_layout": "row-major"},
    )


def _slice_tensor(tensor, tensor_range):
    return tensor[tuple(slice(start, end) for start, end in tensor_range)]


@pytest.mark.parametrize("runtime_framework", ["vllm", "sglang"])
def test_receiver_installs_multisource_segments_into_runtime_owned_tensor(
    runtime_framework,
):
    model = nn.Module()
    model.weight = nn.Parameter(torch.empty((3, 4), dtype=torch.float32))
    target_tensors = dict(model.named_parameters())
    requested_range = ((2, 5), (0, 4))

    requests = build_receiver_requests_from_runtime_tensors(
        target_tensors,
        model_name="qwen",
        model_version="step-7",
        runtime_framework=runtime_framework,
        requested_ranges={"weight": requested_range},
    )
    assert len(requests) == 1
    assert requests[0].runtime_framework == runtime_framework
    assert requests[0].requested_range == requested_range
    assert requests[0].target_shape == (3, 4)
    assert requests[0].destination_strides == (4, 1)

    plans = plan_segments(
        [
            _owner(((0, 3), (0, 4)), source_id="trainer-rank0"),
            _owner(((3, 6), (0, 4)), source_id="trainer-rank1"),
        ],
        requests,
    )
    assert [plan.source_id for plan in plans] == [
        "trainer-rank0",
        "trainer-rank1",
    ]

    global_weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    payloads = [
        (plan, _slice_tensor(global_weight, plan.source_range).clone())
        for plan in plans
    ]
    installed = install_segment_payloads_into_runtime_tensors(
        payloads,
        target_tensors,
        target_ranges={"weight": requested_range},
    )

    assert [segment.source_id for segment in installed] == [
        "trainer-rank0",
        "trainer-rank1",
    ]
    torch.testing.assert_close(model.weight, global_weight[2:5, :])


def test_receiver_rejects_payload_dtype_mismatch_without_explicit_cast():
    target = {"weight": torch.empty((1, 4), dtype=torch.float32)}
    requests = build_receiver_requests_from_runtime_tensors(
        target,
        model_name="qwen",
        model_version="step-7",
        runtime_framework="vllm",
    )
    plans = plan_segments(
        [_owner(((0, 6), (0, 4)), source_id="trainer-rank0")],
        requests,
    )

    with pytest.raises(TypeError, match="payload dtype"):
        install_segment_payloads_into_runtime_tensors(
            [(plans[0], torch.ones((1, 4), dtype=torch.float16))],
            target,
        )
