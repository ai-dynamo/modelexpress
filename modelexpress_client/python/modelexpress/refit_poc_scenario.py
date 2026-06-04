# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic refit scenario shared by planner and GPU POC runners."""

from __future__ import annotations

from .resharding import SliceOwnership, SliceRequest, range_extents

MODEL_NAME = "qwen3-moe-refit-poc"
MODEL_VERSION = "trainer-step-000001"
TENSOR_NAME = "model.layers.0.mlp.experts.w1.weight"
GLOBAL_SHAPE = (8, 4)
REQUEST_RANGE = ((2, 6), (0, 4))
PRIMARY_SOURCE_IDS = ("trainer-rank0", "trainer-rank1")
FAILED_PRIMARY_SOURCE_IDS = ("trainer-rank0",)
ALTERNATE_SOURCE_IDS = ("trainer-rank2-alt",)
CONTROL_PLANE_SYNTHETIC = "synthetic"
CONTROL_PLANE_LIVE_MX = "live-mx"


def primary_ownerships() -> list[SliceOwnership]:
    return [
        SliceOwnership(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            tensor_name=TENSOR_NAME,
            global_shape=GLOBAL_SHAPE,
            dtype="float32",
            source_range=((0, 3), (0, 4)),
            worker_id="rank0",
            source_id="trainer-rank0",
            worker_rank=0,
            source_lease="lease-rank0-primary",
            nixl_descriptor_id="nixl-rank0-primary",
            layout_tags={
                "trainer_layout": "fsdp",
                "tp": 1,
                "moe_expert_axis": 0,
                "storage_layout": "row-major",
            },
        ),
        SliceOwnership(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            tensor_name=TENSOR_NAME,
            global_shape=GLOBAL_SHAPE,
            dtype="float32",
            source_range=((3, 8), (0, 4)),
            worker_id="rank1",
            source_id="trainer-rank1",
            worker_rank=1,
            source_lease="lease-rank1-primary",
            nixl_descriptor_id="nixl-rank1-primary",
            layout_tags={
                "trainer_layout": "fsdp",
                "tp": 1,
                "moe_expert_axis": 0,
                "storage_layout": "row-major",
            },
        ),
    ]


def alternate_ownerships() -> list[SliceOwnership]:
    return [
        SliceOwnership(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            tensor_name=TENSOR_NAME,
            global_shape=GLOBAL_SHAPE,
            dtype="float32",
            source_range=((0, 3), (0, 4)),
            worker_id="rank2",
            source_id="trainer-rank2-alt",
            worker_rank=2,
            source_lease="lease-rank2-alt",
            nixl_descriptor_id="nixl-rank2-alt",
            layout_tags={
                "trainer_layout": "fsdp-replica",
                "tp": 1,
                "moe_expert_axis": 0,
                "storage_layout": "row-major",
            },
        )
    ]


def ownerships_by_rank() -> dict[int, SliceOwnership]:
    return {
        owner.worker_rank: owner
        for owner in [*primary_ownerships(), *alternate_ownerships()]
    }


def inference_request(
    *,
    requested_range=REQUEST_RANGE,
    target_id: str = "inference-rank3",
    runtime_framework: str = "vllm",
) -> SliceRequest:
    return SliceRequest(
        tensor_name=TENSOR_NAME,
        requested_range=requested_range,
        target_shape=range_extents(requested_range),
        dtype="float32",
        target_id=target_id,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        runtime_framework=runtime_framework,
        layout_tags={
            "target_layout": "tp2-ep2",
            "tp": 2,
            "ep": 2,
            "moe_expert_axis": 0,
            "storage_layout": "row-major",
        },
    )
