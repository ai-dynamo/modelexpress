# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A bad reconstructed checkpoint must stop the live protocol before resume."""

import json
import sys
from types import SimpleNamespace

import pytest
from harness.runner import RefitRunner
from scenarios.delta.scenario import DeltaScenario


@pytest.mark.parametrize("failure", ["checkpoint", "runtime"])
@pytest.mark.parametrize("target_version", ["nemotron-test-d1", "custom-update"])
def test_publisher_mismatch_stops_before_resume(
    tmp_path, monkeypatch, target_version, failure
):
    config = {
        "run": "nemotron-test",
        "publication_path": str(tmp_path / "publication.json"),
        "initial_version": "nemotron-test-base",
        "target_version": target_version,
        "sources": {"s3": "OBJECT_STORAGE", "peer": "GENERATOR"},
        "resource_prefix": "mx-test",
        "roles": ["s3"],
        "tp": 1,
        "revision": "revision",
        "embedding": "embedding",
        "expected_tensors_per_rank": None,
        "expected_host_scales_per_rank": None,
    }
    publication = tmp_path / "publication.json"
    publication.write_text(
        json.dumps(
            {
                "run": config["run"],
                "model_revision": config["revision"],
                "tensor": "embedding",
                "expected_sha256": "a" * 64,
            }
        )
    )
    calls = []

    def post(url, *, json, timeout):
        route = url.rsplit("/", 1)[1]
        calls.append(route)
        result = []
        if route == "generate":
            result = [{"token_ids": [1], "logprob_count": 1}]
        elif route == "rpc":
            method = json["method"]
            row = {"rank": 0, "phase": "init"}
            if method == "hotload":
                row.update(
                    phase=config["target_version"],
                    version=config["target_version"],
                    serving_version=config["target_version"],
                    source="OBJECT_STORAGE",
                    weight_addresses_preserved=True,
                )
            elif method == "tensor_hashes":
                row["tensors"] = {"embedding": {"sha256": "before"}}
            elif method == "verify_checkpoint":
                row.update(
                    version=config["target_version"],
                    verified=True,
                    sha256="same",
                    expected_sha256="same",
                    checkpoint_tensor="embedding",
                    checkpoint_sha256=("b" if failure == "checkpoint" else "a") * 64,
                )
            elif method == "verify_runtime_tensor":
                row.update(
                    version=config["target_version"],
                    checkpoint_tensor="embedding",
                    expected_sha256="c" * 64,
                    actual_sha256="d" * 64,
                    verified=False,
                )
            result = [row]
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"ok": True, "result": result},
            raise_for_status=lambda: None,
        )

    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(
            post=post,
            get=lambda *a, **k: SimpleNamespace(status_code=200),
            RequestException=OSError,
        ),
    )
    with pytest.raises(
        AssertionError,
        match=(
            "published embedding"
            if failure == "checkpoint"
            else "Installed runtime tensor"
        ),
    ):
        RefitRunner(config, tmp_path / "driver", DeltaScenario(config)).run()
    assert "resume" not in calls
    assert (tmp_path / "driver/FAIL").exists()
    assert not (tmp_path / "driver/PASS").exists()


def test_shared_protocol_runs_an_alternate_scenario(tmp_path, monkeypatch):
    config = {
        "scenario": "fixture",
        "initial_version": "initial-42",
        "target_version": "updated-99",
        "sources": {"receiver": "GENERATOR"},
        "resource_prefix": "fixture",
        "roles": ["receiver"],
        "tp": 1,
        "publication_path": str(tmp_path / "publication.json"),
        "expected_tensors_per_rank": None,
        "expected_host_scales_per_rank": None,
    }
    publication = tmp_path / "publication.json"
    publication.write_text("{}")
    events = []

    def post(url, *, json, timeout):
        route = url.rsplit("/", 1)[1]
        events.append(route)
        result = {}
        if route == "generate":
            result = [{"token_ids": [1], "logprob_count": 1}]
        elif route == "rpc":
            method, kwargs = json["method"], json["kwargs"]
            events.append(method)
            row = {"rank": 0, "phase": "init"}
            if method == "hotload_init":
                assert kwargs == {
                    "initial_version_id": "initial-42",
                    "source": "GENERATOR",
                }
            elif method == "hotload":
                assert kwargs == {"version_id": "updated-99"}
                row.update(
                    phase="updated-99",
                    version="updated-99",
                    serving_version="updated-99",
                    source="GENERATOR",
                    weight_addresses_preserved=True,
                )
            elif method == "tensor_hashes":
                # This scenario preserves weights; the shared driver must allow it.
                row["tensors"] = {"weight": {"sha256": "unchanged"}}
            else:
                pytest.fail(f"Unexpected delta-specific RPC: {method}")
            result = [row]
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"ok": True, "result": result},
            raise_for_status=lambda: None,
        )

    def verify(rpc, role, evidence):
        assert role == "receiver" and evidence == {}
        events.append("verify-scenario")

    def inventory(before, after, evidence):
        assert before == after
        events.append("validate-scenario")

    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(
            post=post,
            get=lambda *a, **k: SimpleNamespace(status_code=200),
            RequestException=OSError,
        ),
    )
    case = SimpleNamespace(verify_refit=verify, validate_inventory=inventory)
    RefitRunner(config, tmp_path / "driver", case).run()
    assert (
        events.index("pause")
        < events.index("hotload")
        < events.index("verify-scenario")
    )
    assert events.index("validate-scenario") < events.index("resume")
    assert (tmp_path / "driver/PASS").exists()
