# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON serialization for all protocol types.  No torch dependency."""

from __future__ import annotations

import base64
import json
from dataclasses import asdict

from .types import (
    InferenceShard,
    InferenceTable,
    M2nDescriptor,
    RdmaDescriptor,
    ResolvedRegion,
    TrainerShard,
    TrainerTable,
    TrainerTensor,
)


def encode_trainer_table(table: TrainerTable) -> bytes:
    d = {
        "step": table.step,
        "agents": [base64.b64encode(a).decode() for a in table.agents],
        "tensors": [
            {
                "name": tt.name,
                "dtype": tt.dtype,
                "shape": tt.shape,
                "shards": [asdict(s) for s in tt.shards],
            }
            for tt in table.tensors
        ],
    }
    return json.dumps(d, separators=(",", ":")).encode()


def decode_trainer_table(data: bytes) -> TrainerTable:
    d = json.loads(data)
    return TrainerTable(
        step=d.get("step", 0),
        agents=[base64.b64decode(a) for a in d["agents"]],
        tensors=[
            TrainerTensor(
                name=t["name"],
                dtype=t["dtype"],
                shape=t["shape"],
                shards=[TrainerShard(**s) for s in t["shards"]],
            )
            for t in d["tensors"]
        ],
    )


def encode_inference_table(table: InferenceTable) -> bytes:
    d = {
        "worker_rank": table.worker_rank,
        "agents": [base64.b64encode(a).decode() for a in table.agents],
        "shards": [asdict(s) for s in table.shards],
    }
    return json.dumps(d, separators=(",", ":")).encode()


def decode_inference_table(data: bytes) -> InferenceTable:
    d = json.loads(data)
    return InferenceTable(
        worker_rank=d.get("worker_rank", 0),
        agents=[base64.b64decode(a) for a in d["agents"]],
        shards=[InferenceShard(**s) for s in d["shards"]],
    )


def encode_resolved_regions(regions: list[ResolvedRegion]) -> bytes:
    d = [
        {
            "tensor_name": r.tensor_name,
            "src_elem_runs": r.src_elem_runs,
            "dst_addr": r.dst_addr,
            "dst_elem_runs": r.dst_elem_runs,
            "element_size": r.element_size,
            "dst_device_id": r.dst_device_id,
        }
        for r in regions
    ]
    return json.dumps(d, separators=(",", ":")).encode()


def decode_resolved_regions(data: bytes) -> list[ResolvedRegion]:
    return [ResolvedRegion(**r) for r in json.loads(data)]


def encode_rdma_descriptors(descs: list[RdmaDescriptor]) -> bytes:
    return json.dumps([asdict(d) for d in descs], separators=(",", ":")).encode()


def decode_rdma_descriptors(data: bytes) -> list[RdmaDescriptor]:
    return [RdmaDescriptor(**d) for d in json.loads(data)]


def encode_m2n_descriptors(descs: list[M2nDescriptor]) -> bytes:
    return json.dumps([asdict(d) for d in descs], separators=(",", ":")).encode()


def decode_m2n_descriptors(data: bytes) -> list[M2nDescriptor]:
    return [M2nDescriptor(**d) for d in json.loads(data)]
