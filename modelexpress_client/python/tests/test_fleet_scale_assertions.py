# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression cases for fleet readiness with historical worker generations."""

import importlib
from pathlib import Path
from subprocess import CompletedProcess

import pytest


@pytest.fixture
def fleet_test(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "ci/k8s/client")
    )
    return importlib.import_module("test_fleet_scale")


@pytest.mark.parametrize("retired", [False, True])
def test_ready_fleet_accepts_retired_stale_generation(monkeypatch, fleet_test, retired):
    rows = [f"worker-{rank} Ready" for rank in range(15)]
    if retired:
        rows.append("retired-worker Stale")
    monkeypatch.setattr(
        fleet_test, "kubectl",
        lambda *args, **kwargs: CompletedProcess(args, 0, stdout="\n".join(rows)),
    )
    fleet_test.test_fleet_crs_published_and_ready("test-fleet", 15)


@pytest.mark.parametrize(
    "rows",
    [
        [f"worker-{rank} Ready" for rank in range(14)] + ["retired-worker Stale"],
        [f"worker-{rank} Ready" for rank in range(15)] + ["worker-new Initializing"],
        [f"worker-{rank} Ready" for rank in range(15)] + ["worker-new Unknown"],
    ],
)
def test_unhealthy_fleet_is_rejected(monkeypatch, fleet_test, rows):
    monkeypatch.setattr(
        fleet_test, "kubectl",
        lambda *args, **kwargs: CompletedProcess(args, 0, stdout="\n".join(rows)),
    )
    with pytest.raises(AssertionError):
        fleet_test.test_fleet_crs_published_and_ready("test-fleet", 15)
