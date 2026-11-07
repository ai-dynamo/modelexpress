# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared lifecycle contracts for every RL GPU E2E lane."""

from pathlib import Path
import os
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def workflow(name):
    return yaml.safe_load((ROOT / f".github/workflows/{name}").read_text())


@pytest.mark.parametrize("lane", ["generator-only", "vime", "primerl"])
def test_rl_build_and_gpu_lifecycle_are_consistent(lane):
    parent = workflow("modelexpress-rl-ci.yml")["jobs"]
    build, call = parent[f"build-{lane}"], parent[f"test-{lane}"]
    assert build["needs"] == "validate"
    assert build["runs-on"] == "prod-modelexpress-builder-amd-v1"
    assert build["timeout-minutes"] == 60
    assert "worker_image" in build["outputs"]
    assert "containerimage.digest" in str(build["steps"])
    assert call["with"]["worker_image"] == f"${{{{ needs.build-{lane}.outputs.worker_image }}}}"
    assert call["with"]["sha"] == "${{ needs.validate.outputs.sha }}"
    assert call["with"]["harness_sha"] == "${{ needs.validate.outputs.harness_sha }}"
    child = workflow(f"modelexpress-rl-{lane}-gpu-ci.yml")
    inputs = child.get("on", child.get(True))["workflow_call"]["inputs"]
    for name in ["sha", "harness_sha", "worker_image", "gpu_limit", "gpu_slot"]:
        assert inputs[name]["required"] is True
    assert inputs["upload_evidence"]["default"] is True
    assert set(child["jobs"]) == {"test", "cleanup"}
    test = child["jobs"]["test"]
    assert test["runs-on"] == "prod-deploy-tester-v1"
    assert test["timeout-minutes"] == 120
    assert "modelexpress-rl-gpu-" in test["concurrency"]["group"]
    assert test["concurrency"]["cancel-in-progress"] is False
    steps = test["steps"]
    assert steps[0]["with"]["ref"] == "${{ inputs.harness_sha }}"
    names = [step.get("name", "") for step in steps]
    collect, upload, cleanup = [names.index(name) for name in [
        "Collect evidence", "Upload evidence", "Cleanup owned workload",
    ]]
    assert collect < upload < cleanup == len(steps) - 1
    assert steps[collect]["if"] == steps[cleanup]["if"] == "always()"
    assert "inputs.upload_evidence" in steps[upload]["if"]
    fallback = child["jobs"]["cleanup"]
    assert fallback["needs"] == "test"
    assert "always()" in fallback["if"]
    assert "concurrency" not in fallback
    assert fallback["steps"][0]["with"]["ref"] == "${{ inputs.harness_sha }}"


def test_rl_status_waits_for_all_lanes_without_parent_cleanup():
    jobs = workflow("modelexpress-rl-ci.yml")["jobs"]
    assert "cleanup" not in jobs
    assert set(jobs["status-check"]["needs"]) == {
        "validate", "build-generator-only", "test-generator-only", "build-vime", "test-vime",
        "build-primerl", "test-primerl",
    }


def test_vime_collection_retains_full_logs_in_the_selected_namespace(tmp_path):
    fake = tmp_path / "kubectl"
    fake.write_text('''#!/bin/bash
echo "$*" >> "$CALLS"
if [[ " $* " == *" get pods -n "* ]]; then echo pod/rollout;
elif [[ " $* " == *" logs "* ]]; then
  for ((i=0; i<300; i++)); do echo "line-$i"; done
else echo fixture;
fi
''')
    fake.chmod(0o755)
    output = tmp_path / "evidence"
    result = subprocess.run(
        ["bash", "ci/rl/vime/cluster.sh", "collect"], cwd=ROOT,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
             "CALLS": str(tmp_path / "calls"), "RESULTS_DIR": str(output),
             "NAMESPACE": "mx-ci-vime-fixture", "KUBE_CONTEXT": "fixture"},
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "rollout.log").read_text().splitlines() == [f"line-{i}" for i in range(300)]
    calls = (tmp_path / "calls").read_text().splitlines()
    assert all("--context=fixture" in call and "-n mx-ci-vime-fixture" in call for call in calls)
    assert "--tail=-1" in calls[-1]


@pytest.mark.parametrize("lane,resource", [
    ("vime", "pods,jobs"), ("vime", "events"), ("vime", "pods -n"),
    ("primerl", "pods,jobs"), ("primerl", "events"),
])
@pytest.mark.parametrize("error,expected,calls", [
    ("unexpected EOF", 0, 2),
    ("Forbidden", 17, 1),
])
def test_evidence_collection_retries_only_transient_reads(tmp_path, lane, resource, error, expected, calls):
    fake = tmp_path / "kubectl"
    fake.write_text(f'''#!/bin/bash
if [[ " $* " == *" get {resource}"* ]]; then
  n=$(cat "$CALLS" 2>/dev/null || echo 0)
  n=$((n + 1)); echo "$n" > "$CALLS"
  if [ "$n" -eq 1 ]; then echo '{error}' >&2; exit 17; fi
fi
if [[ " $* " == *" get pods -n "* ]]; then echo pod/worker; else echo fixture; fi
''')
    fake.chmod(0o755)
    # Avoid real sleeps while exercising the production retry helper.
    bash_env = tmp_path / "bash-env"
    bash_env.write_text("sleep() { :; }\n")
    output = tmp_path / "evidence"
    result = subprocess.run(
        ["bash", f"ci/rl/{lane}/cluster.sh", "collect"], cwd=ROOT,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "BASH_ENV": str(bash_env),
             "CALLS": str(tmp_path / "calls"), "RESULTS_DIR": str(output),
             "NAMESPACE": "mx-ci-fixture", "KUBE_CONTEXT": "fixture",
             "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"},
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == expected, result.stderr
    assert int((tmp_path / "calls").read_text()) == calls
    if expected == 0:
        assert (output / "events.txt").read_text().strip() == "fixture"
