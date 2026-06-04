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
    begin_runtime_refit_transaction,
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


def _versioned_owner(tensor_name, shape, *, model_version, source_id="trainer-rank0"):
    return SliceOwnership(
        model_name="qwen",
        model_version=model_version,
        tensor_name=tensor_name,
        global_shape=shape,
        dtype="float32",
        source_range=tuple((0, dim) for dim in shape),
        worker_id=f"{source_id}-worker",
        source_id=source_id,
        source_lease=f"{source_id}-{model_version}-lease",
        nixl_descriptor_id=f"{source_id}-{model_version}-desc",
        layout_tags={"storage_layout": "row-major", "trainer_layout": "fsdp"},
    )


def test_runtime_refit_transaction_rolls_back_multilayer_runtime_install():
    tensor_names = [
        "model.layers.0.mlp.experts.w1.weight",
        "model.layers.1.mlp.experts.w1.weight",
    ]
    previous_by_name = {
        tensor_names[0]: torch.arange(12, dtype=torch.float32).reshape(3, 4),
        tensor_names[1]: torch.arange(12, 24, dtype=torch.float32).reshape(3, 4),
    }
    runtime_by_name = {
        name: tensor.clone() for name, tensor in previous_by_name.items()
    }
    runtime_by_target = {
        f"vllm:{name}": tensor for name, tensor in runtime_by_name.items()
    }

    requests = build_receiver_requests_from_runtime_tensors(
        runtime_by_name,
        model_name="qwen",
        model_version="step-8",
        runtime_framework="vllm",
        target_id_prefix="vllm",
        layout_tags_by_tensor={name: {"moe_expert_axis": 0} for name in tensor_names},
    )
    plans = plan_segments(
        [
            _versioned_owner(
                name, tuple(runtime_by_name[name].shape), model_version="step-8"
            )
            for name in tensor_names
        ],
        requests,
    )
    payload_by_name = {
        tensor_names[0]: torch.full((3, 4), 80.0, dtype=torch.float32),
        tensor_names[1]: torch.full((3, 4), 81.0, dtype=torch.float32),
    }
    payloads = [(plan, payload_by_name[plan.tensor_name]) for plan in plans]

    transaction = begin_runtime_refit_transaction(
        runtime_by_target,
        previous_model_version="step-7",
        target_model_version="step-8",
    )
    installed = install_segment_payloads_into_runtime_tensors(
        payloads, runtime_by_target
    )

    assert len(transaction.snapshots) == 2
    assert [item.tensor_name for item in installed] == tensor_names
    for name in tensor_names:
        torch.testing.assert_close(runtime_by_name[name], payload_by_name[name])

    transaction.rollback()

    assert transaction.rolled_back is True
    assert transaction.committed is False
    for name in tensor_names:
        torch.testing.assert_close(runtime_by_name[name], previous_by_name[name])


def test_runtime_refit_transaction_commit_keeps_new_version_and_blocks_rollback():
    tensor_name = "model.layers.0.mlp.experts.w2.weight"
    runtime = torch.zeros((2, 4), dtype=torch.float32)
    runtime_by_name = {tensor_name: runtime}
    runtime_by_target = {f"sglang:{tensor_name}": runtime}
    requests = build_receiver_requests_from_runtime_tensors(
        runtime_by_name,
        model_name="qwen",
        model_version="step-9",
        runtime_framework="sglang",
        target_id_prefix="sglang",
    )
    plans = plan_segments(
        [_versioned_owner(tensor_name, (2, 4), model_version="step-9")],
        requests,
    )
    payload = torch.full((2, 4), 9.0, dtype=torch.float32)

    transaction = begin_runtime_refit_transaction(
        runtime_by_target,
        previous_model_version="step-8",
        target_model_version="step-9",
    )
    install_segment_payloads_into_runtime_tensors(
        [(plans[0], payload)], runtime_by_target
    )
    transaction.commit()

    assert transaction.committed is True
    assert transaction.rolled_back is False
    assert transaction.to_dict()["target_model_version"] == "step-9"
    torch.testing.assert_close(runtime, payload)
    with pytest.raises(RuntimeError, match="cannot rollback a committed"):
        transaction.rollback()


def test_runtime_refit_transaction_rejects_same_version():
    with pytest.raises(ValueError, match="must differ"):
        begin_runtime_refit_transaction(
            {"weight": torch.zeros((1,), dtype=torch.float32)},
            previous_model_version="step-7",
            target_model_version="step-7",
        )


def test_runtime_refit_transaction_rejects_dtype_change_before_rollback():
    runtime_by_target = {"vllm:weight": torch.ones((2,), dtype=torch.float32)}
    transaction = begin_runtime_refit_transaction(
        runtime_by_target,
        previous_model_version="step-7",
        target_model_version="step-8",
    )
    runtime_by_target["vllm:weight"] = torch.ones((2,), dtype=torch.float16)

    with pytest.raises(ValueError, match="changed dtype"):
        transaction.rollback()
