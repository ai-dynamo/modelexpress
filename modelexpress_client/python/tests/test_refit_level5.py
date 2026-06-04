# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from modelexpress.refit_level5 import (
    CHECKPOINT_ENGINE_STRATEGY,
    MX_NIXL_STRATEGY,
    NCCL_RESHARD_STRATEGY,
    build_level5_timing_table,
    build_level5_table_from_artifacts,
    normalize_level5_artifact,
    unmeasured_level5_row,
    _materialize_full_tensor_from_padded_gather,
    _pad_trainer_source,
    _padded_trainer_collective_bytes,
)
from modelexpress.refit_poc_scenario import primary_ownerships


def _artifact(strategy, *, checksum=556224.0, placement_scope="same-node-single-pod"):
    return {
        "schema_version": 1,
        "result": "pass",
        "mode": f"mode-{strategy}",
        "strategy": strategy,
        "placement_scope": placement_scope,
        "metrics": {
            "trainer_to_inference_bytes": 64,
            "inference_side_fanout_bytes": 0,
            "trainer_collective_bytes": 0,
            "checkpoint_storage_bytes": 0,
            "redundant_cross_boundary_factor": 1.0,
            "segment_count": 2,
            "source_count_per_target_tensor": {"weight": 2},
            "registration_duration_ms": 1.0,
            "publish_duration_ms": 2.0,
            "metadata_query_duration_ms": 3.0,
            "planner_duration_ms": 4.0,
            "raw_read_duration_ms": 5.0,
            "activation_install_duration_ms": 6.0,
        },
        "validation": {
            "allclose": True,
            "checksum": checksum,
            "expected_checksum": 556224.0,
            "max_abs_error": 0.0,
        },
        "proof": {"actual_nixl_reads_used": strategy == MX_NIXL_STRATEGY},
    }


def test_padded_trainer_gather_handles_non_aligned_source_shapes():
    owners = primary_ownerships()
    rank0 = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    rank1 = torch.arange(20, dtype=torch.float32).reshape(5, 4) + 100

    padded = [
        _pad_trainer_source(rank0, owners),
        _pad_trainer_source(rank1, owners),
    ]
    full = _materialize_full_tensor_from_padded_gather(
        padded,
        owners,
        torch.device("cpu"),
    )

    assert padded[0].shape == padded[1].shape == (5, 4)
    torch.testing.assert_close(full[0:3, :], rank0)
    torch.testing.assert_close(full[3:8, :], rank1)
    assert _padded_trainer_collective_bytes(owners, trainer_rank_count=2) == 160


def test_normalize_level5_artifact_preserves_required_metrics():
    row = normalize_level5_artifact(
        _artifact(MX_NIXL_STRATEGY),
        source_artifact="mx.json",
    )

    assert row["strategy"] == MX_NIXL_STRATEGY
    assert row["source_artifact"] == "mx.json"
    assert row["pass"] is True
    assert row["allclose"] is True
    assert row["checksum_matches"] is True
    assert row["trainer_to_inference_bytes"] == 64
    assert row["read_duration_ms"] == 5.0
    assert row["missing_timing_fields"] == []


def test_normalize_level5_artifact_fails_checksum_mismatch():
    row = normalize_level5_artifact(_artifact(MX_NIXL_STRATEGY, checksum=1.0))

    assert row["pass"] is False
    assert row["checksum_matches"] is False


def test_level5_table_requires_all_three_rows():
    table = build_level5_table_from_artifacts(
        mx_artifact=_artifact(MX_NIXL_STRATEGY),
        nccl_artifact=None,
        checkpoint_artifact=None,
    )

    assert table["result"] == "fail"
    assert table["level5_synthetic_smoke_pass"] is False
    assert table["level5_full_model_claim_safe"] is False
    assert table["failed_rows"] == [
        NCCL_RESHARD_STRATEGY,
        CHECKPOINT_ENGINE_STRATEGY,
    ]
    assert {row["strategy"] for row in table["rows"]} == {
        MX_NIXL_STRATEGY,
        NCCL_RESHARD_STRATEGY,
        CHECKPOINT_ENGINE_STRATEGY,
    }


def test_level5_table_passes_for_comparable_checksum_gated_rows():
    rows = [
        normalize_level5_artifact(_artifact(MX_NIXL_STRATEGY)),
        normalize_level5_artifact(_artifact(NCCL_RESHARD_STRATEGY)),
        normalize_level5_artifact(_artifact(CHECKPOINT_ENGINE_STRATEGY)),
    ]

    table = build_level5_timing_table(rows)

    assert table["result"] == "pass"
    assert table["level5_synthetic_smoke_pass"] is True
    assert table["production_competitive_claim_safe"] is False
    assert table["placement_scopes"] == ["same-node-single-pod"]


def test_level5_table_rejects_mixed_placement_scope():
    rows = [
        normalize_level5_artifact(
            _artifact(MX_NIXL_STRATEGY, placement_scope="cross-node")
        ),
        normalize_level5_artifact(_artifact(NCCL_RESHARD_STRATEGY)),
        normalize_level5_artifact(_artifact(CHECKPOINT_ENGINE_STRATEGY)),
    ]

    table = build_level5_timing_table(rows)

    assert table["result"] == "fail"
    assert table["comparable_placement_scope"] is False


def test_unmeasured_row_records_block_reason():
    row = unmeasured_level5_row(
        NCCL_RESHARD_STRATEGY,
        reason="capacity blocked",
        source_artifact="capacity.json",
    )

    assert row["pass"] is False
    assert row["measured"] is False
    assert row["block_reason"] == "capacity blocked"
    assert row["source_artifact"] == "capacity.json"
