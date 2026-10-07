# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation contracts shared by the online driver and offline report."""

import math


def ranks(rows, config):
    assert isinstance(rows, list) and len(rows) == config["tp"], (
        "Incomplete rank results"
    )
    by_rank = {r["rank"]: r for r in rows}
    assert set(by_rank) == set(range(config["tp"])), "Missing or duplicate ranks"
    for row in by_rank.values():
        assert "error" not in row and not row["phase"].endswith("-failed"), row
    return by_rank


def hashes(rows, config):
    result = {}
    for rank, row in ranks(rows, config).items():
        tensors = row["tensors"]
        assert tensors, "Empty tensor inventory"
        count = config["expected_tensors_per_rank"]
        if count is not None:
            assert len(tensors) == count, (rank, len(tensors), count)
        result[rank] = tensors
    return result


def scales(rows, config):
    for row in ranks(rows, config).values():
        expected = config["expected_host_scales_per_rank"]
        if expected is not None:
            assert len(row["scales"]) == expected
        assert all(
            x["gpu"] == x["host"] and (x["cpu"] is None or x["gpu"] == x["cpu"])
            for x in row["scales"]
        ), row


def refit(rows, config, role):
    for row in ranks(rows, config).values():
        assert (
            row["phase"]
            == row["version"]
            == row["serving_version"]
            == config["target_version"]
        ), row
        assert row["source"] == config["sources"][role]
        assert row["weight_addresses_preserved"]


def inference(rows):
    assert rows
    for row in rows:
        assert row["token_ids"] and row["logprob_count"] == len(row["token_ids"]), row


def duration(value):
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    assert math.isfinite(value) and value >= 0, "Invalid latency measurement"
    return value


def latency(rows, config, load_seconds, rpc_seconds):
    assert len(load_seconds) == 1, "Expected one fresh model-load measurement"
    per_rank = {}
    for rank, row in ranks(rows, config).items():
        per_rank[rank] = {
            key: duration(row[key])
            for key in ["stage_seconds", "install_seconds", "total_seconds"]
        }
        assert math.isclose(
            row["total_seconds"],
            row["stage_seconds"] + row["install_seconds"],
            rel_tol=1e-6,
            abs_tol=1e-6,
        ), "Total must equal measured stage plus install time"
    return {
        "model_load_seconds": duration(load_seconds[0]),
        "refit_rpc_seconds": duration(rpc_seconds),
        "per_rank": per_rank,
        "slowest_rank_total_seconds": max(
            row["total_seconds"] for row in per_rank.values()
        ),
    }
