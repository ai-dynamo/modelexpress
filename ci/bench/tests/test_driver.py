# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A bad reconstructed checkpoint must stop the live protocol before resume."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("target_version", ["nemotron-test-d1", "custom-update"])
def test_publisher_mismatch_stops_before_resume(tmp_path, monkeypatch, target_version):
    config = {
        "run": "nemotron-test",
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
            method, kwargs = json["method"], json["kwargs"]
            row = {"rank": 0, "phase": "init"}
            if method == "hotload":
                row.update(
                    phase=config["target_version"],
                    version=config["target_version"],
                    serving_version=config["target_version"],
                    source="OBJECT_STORAGE",
                    weight_addresses_preserved=True,
                )
            elif method == "hotload_verify_checkpoint":
                if kwargs["version_id"].startswith("hashes:"):
                    row["tensors"] = {"embedding": {"sha256": "before"}}
                else:
                    row.update(
                        version=config["target_version"],
                        verified=True,
                        sha256="same",
                        expected_sha256="same",
                        checkpoint_tensor="embedding",
                        checkpoint_sha256="b" * 64,
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
    monkeypatch.setitem(sys.modules, "config", SimpleNamespace(CONFIG=config))
    monkeypatch.setitem(
        sys.modules,
        "pathlib",
        SimpleNamespace(
            Path=lambda path: (
                publication
                if path == "/tmp/mx-delta/report.json"
                else tmp_path / "driver"
            )
        ),
    )
    monkeypatch.syspath_prepend(str(ROOT / "common"))
    with pytest.raises(AssertionError, match="published embedding"):
        runpy.run_path(str(ROOT / "common/run_bench.py"))
    assert "resume" not in calls
    assert (tmp_path / "driver/FAIL").exists()
    assert not (tmp_path / "driver/PASS").exists()
