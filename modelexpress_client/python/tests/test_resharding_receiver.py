# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from modelexpress.resharding import QuantizationScope, SliceOwnership, plan_segments
from modelexpress.resharding_manifest import (
    TensorManifestEntry,
    classify_qwen_moe_tensor,
)
from modelexpress.resharding_receiver import (
    build_receiver_requests_from_runtime_tensors,
    install_global_required_quantization_payloads_into_runtime_tensors,
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


def test_receiver_installs_global_required_quantization_fallback_tensor():
    tensor_name = "model.layers.0.mlp.experts.w1.weight_scale_inv"
    entry = classify_qwen_moe_tensor(
        tensor_name,
        {"shape": (2, 3), "dtype": "float32"},
        model_name="qwen3-moe-fp8",
        model_version="step-7",
    )
    target_key = f"vllm:{tensor_name}"
    target_tensors = {target_key: torch.empty((2, 3), dtype=torch.float32)}
    payload = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    installed = install_global_required_quantization_payloads_into_runtime_tensors(
        [(entry, payload)],
        target_tensors,
        runtime_framework="vllm",
    )

    assert len(installed) == 1
    assert installed[0].tensor_name == tensor_name
    assert installed[0].target_key == target_key
    assert installed[0].quantization_scope == QuantizationScope.GLOBAL_REQUIRED
    assert installed[0].tensor_family == "quantization-global-required-fallback"
    assert installed[0].runtime_framework == "vllm"
    torch.testing.assert_close(target_tensors[target_key], payload)


def test_receiver_rejects_non_global_required_quantization_fallback_tensor():
    entry = classify_qwen_moe_tensor(
        "model.layers.0.mlp.experts.w1.weight",
        {"shape": (2, 3), "dtype": "float32"},
        model_name="qwen3-moe",
        model_version="step-7",
    )

    with pytest.raises(ValueError, match="expected global-required"):
        install_global_required_quantization_payloads_into_runtime_tensors(
            [(entry, torch.ones((2, 3), dtype=torch.float32))],
            {entry.tensor_name: torch.empty((2, 3), dtype=torch.float32)},
        )


def test_receiver_rejects_quantization_fallback_target_dtype_mismatch():
    tensor_name = "model.layers.0.mlp.experts.w1.weight_scale_inv"
    entry = classify_qwen_moe_tensor(
        tensor_name,
        {"shape": (2, 3), "dtype": "float32"},
        model_name="qwen3-moe-fp8",
        model_version="step-7",
    )

    with pytest.raises(TypeError, match="expected manifest dtype float32"):
        install_global_required_quantization_payloads_into_runtime_tensors(
            [(entry, torch.ones((2, 3), dtype=torch.float32))],
            {entry.tensor_name: torch.empty((2, 3), dtype=torch.float16)},
        )


def test_receiver_installs_real_qwen_fp8_global_required_manifest_entry():
    artifact = _find_repo_artifact(
        "artifacts/resharding/qwen3-30b-a3b-fp8-moe-manifest.json.gz"
    )
    if artifact is None:
        pytest.skip("real Qwen FP8 manifest artifact is not available")

    with gzip.open(artifact, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    entry = next(
        TensorManifestEntry.from_dict(raw_entry)
        for raw_entry in payload["entries"]
        if raw_entry["quantization_scope"] == QuantizationScope.GLOBAL_REQUIRED.value
    )
    target_key = f"sglang:{entry.tensor_name}"
    target_tensors = {target_key: torch.empty(entry.global_shape, dtype=torch.float32)}
    fallback_payload = torch.arange(
        target_tensors[target_key].numel(),
        dtype=torch.float32,
    ).reshape(entry.global_shape)

    installed = install_global_required_quantization_payloads_into_runtime_tensors(
        [(entry, fallback_payload)],
        target_tensors,
        runtime_framework="sglang",
    )

    assert installed[0].tensor_name == entry.tensor_name
    assert installed[0].target_key == target_key
    assert installed[0].shape == entry.global_shape
    assert installed[0].bytes == target_tensors[target_key].numel() * 4
    assert installed[0].runtime_framework == "sglang"
    torch.testing.assert_close(target_tensors[target_key], fallback_payload)


def _find_repo_artifact(relative_path: str) -> Path | None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / relative_path
        if candidate.exists():
            return candidate
    return None
