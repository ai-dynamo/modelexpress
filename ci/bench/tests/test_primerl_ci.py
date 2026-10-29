# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the pinned, two-GPU PrimeRL integration workload."""

import os
from pathlib import Path
import shlex
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT / "ci/rl/primerl"


def test_primerl_image_disables_hosted_monitor_uploads():
    environment = {}
    for line in (CASE / "Dockerfile").read_text().splitlines():
        if line.startswith("ENV "):
            environment.update(value.split("=", 1) for value in shlex.split(line[4:]))
    assert environment["PRIME_RUNS_MODE"] == "disabled"
    assert environment["WANDB_MODE"] == "offline"


def test_primerl_job_has_two_gpus_and_durable_evidence():
    manifest = yaml.safe_load((CASE / "workload.yaml").read_text())
    job, collector = manifest["items"]
    assert job["kind"] == "Job"
    assert job["spec"]["backoffLimit"] == 0
    pod = job["spec"]["template"]["spec"]
    worker = pod["containers"][0]
    assert worker["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert worker["resources"]["requests"]["nvidia.com/gpu"] == "2"
    assert worker["command"] == ["bash", "/opt/mx-ci/run.sh"]
    assert {"name": "checkpoints", "emptyDir": {}} in pod["volumes"]
    assert {"name": "checkpoints", "mountPath": "/checkpoints"} in worker["volumeMounts"]
    assert collector["kind"] == "Pod"
    assert "nvidia.com/gpu" not in collector["spec"]["containers"][0]["resources"]["requests"]
    assert pod["volumes"][0] == collector["spec"]["volumes"][0]


@pytest.mark.parametrize("terminal,expected", [("Complete", 0), ("Failed", 1)])
def test_primerl_wait_reports_job_failure_without_retry(tmp_path, terminal, expected):
    fake = tmp_path / "kubectl"
    fake.write_text(
        "#!/bin/bash\n"
        "echo \"$*\" >> \"$CALLS\"\n"
        f"echo '{{\"status\":{{\"conditions\":[{{\"type\":\"{terminal}\",\"status\":\"True\"}}]}}}}'\n"
    )
    fake.chmod(0o755)
    result = subprocess.run(
        ["bash", str(CASE / "cluster.sh"), "run"], cwd=ROOT,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
             "CALLS": str(tmp_path / "calls"), "NAMESPACE": "mx-ci-primerl-123-1",
             "KUBE_CONTEXT": "fixture", "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"},
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == expected, result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert len(calls) == 1
    assert "--context=fixture" in calls[0]
    assert "-n mx-ci-primerl-123-1" in calls[0]


def test_primerl_is_required_uses_rl_admission_and_cleans_on_failure():
    rl = yaml.safe_load((ROOT / ".github/workflows/modelexpress-rl-ci.yml").read_text())
    assert {"build-primerl", "test-primerl"} <= set(rl["jobs"]["status-check"]["needs"])
    assert rl["jobs"]["build-primerl"]["runs-on"] == "prod-modelexpress-builder-amd-v1"
    selected = rl["jobs"]["test-primerl"]
    assert "issue_comment" in selected["if"]
    assert selected["with"]["gpu_slot"] == "${{ needs.validate.outputs.primerl_gpu_slot }}"
    assert selected["with"]["worker_image"] == "${{ needs.build-primerl.outputs.worker_image }}"
    workflow = yaml.safe_load((ROOT / ".github/workflows/modelexpress-rl-primerl-gpu-ci.yml").read_text())
    assert "build" not in workflow["jobs"]
    smoke = workflow["jobs"]["test"]
    assert "modelexpress-rl-gpu-" in smoke["concurrency"]["group"]
    assert smoke["concurrency"]["cancel-in-progress"] is False
    collect = next(step for step in smoke["steps"] if step.get("name") == "Collect evidence")
    assert collect["if"] == "always()"
    cleanup = smoke["steps"][-1]
    assert cleanup["if"] == "always()"
    assert cleanup["run"].endswith("cluster.sh cleanup")
    assert "always()" in workflow["jobs"]["cleanup"]["if"]


@pytest.mark.parametrize("comment,failed,status", [
    (False, "test-primerl", 1),
    (False, "build-primerl", 1),
    (False, "test-vime", 1),
    (False, None, 0),
    (True, "test-primerl", 0),
    (True, "test-generator-only", 1),
    (True, "build-generator-only", 1),
])
def test_rl_status_gates_only_selected_workloads(comment, failed, status):
    import json

    workflow = yaml.safe_load((ROOT / ".github/workflows/modelexpress-rl-ci.yml").read_text())
    gate = workflow["jobs"]["status-check"]
    needs = {name: {"result": "success"} for name in gate["needs"]}
    if comment:
        for name in ["build-vime", "test-vime", "build-primerl", "test-primerl"]:
            needs[name]["result"] = "skipped"
    if failed:
        needs[failed]["result"] = "failure"
    steps = [step for step in gate["steps"] if not comment or "if" not in step]
    script = "\n".join(step["run"] for step in steps).replace("${{ toJson(needs) }}", json.dumps(needs))
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        capture_output=True, text=True,
    )
    assert result.returncode == status, result.stderr


@pytest.mark.parametrize("gpus", [1, 2])
def test_primerl_preflight_pins_checkpoint_and_preserves_upstream_recipe(tmp_path, monkeypatch, gpus):
    script = (CASE / "run.sh").read_text().split("python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    path = tmp_path / "configs/ci/integration/reverse-text/start.toml"
    path.parent.mkdir(parents=True)
    original = ('max_steps = 15\n[ckpt]\n[model]\nname = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"\n'
                '[orchestrator.renderer]\nname = "prime-qwen3"\n')
    path.write_text(original)
    output = tmp_path / "evidence/outputs"
    output.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTEST_OUTPUT_DIR", str(output))
    monkeypatch.setenv("MX_TEST_SHA", "a" * 40)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: gpus)))
    monkeypatch.setattr("importlib.metadata.version", lambda _: "fixture")
    downloads = []

    def download(model, **kwargs):
        downloads.append((model, kwargs))
        return kwargs["local_dir"]

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    if gpus != 2:
        with pytest.raises(AssertionError, match="exactly two"):
            exec(compile(script, "primerl-preflight", "exec"), {})
        assert path.read_text() == original
        assert not downloads
    else:
        exec(compile(script, "primerl-preflight", "exec"), {})
        resolved = tomllib.loads(path.read_text())
        assert resolved["max_steps"] == 15
        assert resolved["orchestrator"]["renderer"]["name"] == "prime-qwen3"
        manifest = yaml.safe_load((CASE / "workload.yaml").read_text())
        mounts = manifest["items"][0]["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
        mount = Path(next(volume["mountPath"] for volume in mounts if volume["name"] == "checkpoints"))
        checkpoint = Path(resolved["ckpt"]["output_dir"])
        # PrimeRL --clean deletes this directory; it must not delete the volume mount.
        assert checkpoint.is_relative_to(mount) and checkpoint != mount
        assert resolved["deployment"] == {"type": "single_node", "num_train_gpus": 1, "num_infer_gpus": 1}
        assert downloads == [("PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT", {
            "revision": "c97a910849ec6aa962add3dc253a0817d61c0210",
            "local_dir": resolved["model"]["name"],
        })]
        assert (output.parent / "runtime.json").exists()
