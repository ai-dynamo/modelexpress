# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused CPU checks for the standalone M2N microbench."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "m2n_microbench_under_test",
    HERE / "m2n_microbench.py",
)
assert SPEC is not None and SPEC.loader is not None
microbench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = microbench
SPEC.loader.exec_module(microbench)


def _case(**overrides):
    values = {
        "profile": "smoke",
        "source_partitions": 2,
        "destination_fanout": 4,
        "streams": 2,
        "schedule": "synthetic",
        "grouping": "layer",
        "bucket_bytes": 4 * 1024 * 1024,
        "drain": "per-group",
        "group_keys": "unique",
        "warmup": 1,
        "repetitions": 2,
        "timeout_s": 10.0,
    }
    values.update(overrides)
    return microbench.Case(**values)


def test_qwen_3b_manifest_preserves_count_and_embedding_bytes():
    manifest = microbench.manifest_for("qwen-2.5-3b")

    assert len(manifest) == 290
    embedding = next(
        item for item in manifest if item.name == "model.embed_tokens.weight"
    )
    assert embedding.shape == (151_936, 2_048)
    assert embedding.bytes == 622_329_856
    assert sum(item.bytes for item in manifest) == 6_171_702_272


@pytest.mark.parametrize("grouping", ("singleton", "layer", "bucket"))
def test_groupings_cover_every_tensor_once(grouping):
    case = _case(grouping=grouping)
    manifest = microbench.manifest_for(case.profile)
    groups = microbench.group_manifest(case, manifest)

    assert [item.name for group in groups for item in group] == [
        item.name for item in manifest
    ]


@pytest.mark.parametrize(
    ("drain", "expected"),
    (
        ("per-tensor", 8),
        ("per-group", 6),
        ("end", 2),
    ),
)
def test_mock_reports_the_selected_drain_schedule(drain, expected):
    report = microbench.run_mock(
        _case(grouping="layer", drain=drain, warmup=0, repetitions=1)
    )

    [round_record] = report["rounds"]
    assert round_record["drain_count"] == expected
    assert round_record["verified_exact"] is True


def test_mock_reports_per_rank_sender_drains():
    entries = microbench.build_entries(_case(), microbench.manifest_for("smoke"))

    per_tensor = microbench.expected_role_drains(entries, _case(drain="per-tensor"))
    per_group = microbench.expected_role_drains(entries, _case(drain="per-group"))
    end = microbench.expected_role_drains(entries, _case(drain="end"))

    assert per_tensor == {"sender_per_rank": 4, "receiver_per_rank": 8}
    assert per_group == {"sender_per_rank": 3, "receiver_per_rank": 6}
    assert end == {"sender_per_rank": 1, "receiver_per_rank": 2}


def test_miles_current_schedule_uses_shipping_singleton_role_drains():
    case = _case(schedule="miles-current", grouping="layer", drain="end")
    report = microbench.run_mock(case)
    entries = microbench.build_entries(case, microbench.manifest_for(case.profile))

    assert report["schedule"]["classification"] == "shipping MILES schedule"
    assert report["plan"]["group_count"] == len(entries)
    assert report["rounds"][0]["role_drain_counts"] == {
        "sender_per_rank": 1,
        "receiver_per_rank": len(entries),
    }


def test_synthetic_schedule_is_explicitly_labelled():
    report = microbench.run_mock(_case(schedule="synthetic"))

    assert report["schedule"]["classification"] == "synthetic scheduling experiment"


def test_shared_group_keys_are_deterministic_and_report_capability():
    case = _case(group_keys="shared")
    manifest = microbench.manifest_for(case.profile)
    entries = microbench.build_entries(case, manifest)
    report = microbench.run_mock(case)

    assert len({entry.group_key for entry in entries}) == report["plan"]["group_count"]
    assert report["plan"]["native_fusion_supported"] is False
    assert "does not forward" in report["plan"]["group_key_behavior"]


def test_case_identity_and_json_output_are_deterministic(tmp_path):
    case = _case(warmup=0, repetitions=1)
    first = microbench.run_mock(case)
    second = microbench.run_mock(case)
    output = tmp_path / "result.json"

    microbench._write_json(output, first)

    assert first["case_id"] == second["case_id"]
    assert json.loads(output.read_text())["case_id"] == first["case_id"]
    assert first["manifest"]["matches_production_plan"] is False


def test_gpu_environment_preflight_requires_cumem_and_rejects_forced_id(monkeypatch):
    monkeypatch.delenv("NCCL_CUMEM_ENABLE", raising=False)
    monkeypatch.delenv("NCCL_COMM_ID", raising=False)
    with pytest.raises(RuntimeError, match="NCCL_CUMEM_ENABLE=1"):
        microbench._require_gpu_environment()

    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")
    monkeypatch.setenv("NCCL_COMM_ID", "forced")
    with pytest.raises(RuntimeError, match="NCCL_COMM_ID"):
        microbench._require_gpu_environment()


def test_abort_group_closes_all_lanes_after_failure():
    class Cache:
        def __init__(self):
            self.groups = []

        def abort_group(self, group_id):
            self.groups.append(group_id)
            return 2

    cache = Cache()

    assert microbench._abort_group(cache, "round-failure") == 2
    assert cache.groups == ["round-failure"]


def test_vram_envelope_tracks_destination_and_partition_ownership():
    case = _case(source_partitions=2)
    entries = microbench.build_entries(case, microbench.manifest_for(case.profile))
    envelope = microbench._vram_envelope(case, entries)

    assert sum(envelope["source_buffer_bytes_by_partition"]) == sum(
        entry.tensor.bytes for entry in entries
    )
    assert envelope["destination_buffer_bytes_per_rank"] == sum(
        entry.tensor.bytes for entry in entries
    )
    assert envelope["verification_temporary_bytes"] == 0


def test_import_provenance_tracks_the_loaded_module_path():
    provenance = microbench._module_provenance("json", None)

    assert provenance["available"] is True
    assert provenance["module_file"].endswith("json/__init__.py")
    assert "revision" in provenance


@pytest.mark.parametrize(
    "overrides",
    (
        {"source_partitions": 3},
        {"destination_fanout": 3},
        {"streams": 3},
        {"repetitions": 0},
        {"timeout_s": 0},
    ),
)
def test_invalid_case_rejects_before_scheduling(overrides):
    with pytest.raises(ValueError):
        microbench.run_mock(_case(**overrides))
