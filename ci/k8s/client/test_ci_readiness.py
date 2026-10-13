# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def run_shell(tmp_path, body, command):
    fake = tmp_path / "kubectl"
    fake.write_text("#!/bin/bash\n" + command)
    fake.chmod(0o755)
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c",
         'source .github/scripts/ci-kubectl.sh\nsleep() { SECONDS=$((SECONDS + $1)); }\n' + body],
        cwd=ROOT,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
             "FIXTURE": str(tmp_path), "NAMESPACE": "fixture"},
        capture_output=True, text=True, timeout=10,
    )


@pytest.mark.parametrize("error,operation,status,calls", [
    ("connection refused", "get", 0, 2),
    ("connection reset by peer", "get", 0, 2),
    ("unexpected EOF", "get", 0, 2),
    ("context deadline exceeded", "get", 0, 2),
    ("Client.Timeout exceeded", "get", 0, 2),
    ("request canceled (Client.Timeout or context cancellation while reading body)", "get", 0, 2),
    ("i/o timeout", "get", 0, 2),
    ("TLS handshake timeout", "get", 0, 2),

    ("Forbidden", "get", 17, 1),
    ("connection refused", "delete", 17, 1),
])
def test_polling_retry_boundaries(tmp_path, error, operation, status, calls):
    command = f'''count=$(cat "$FIXTURE/calls" 2>/dev/null || echo 0)
count=$((count + 1)); echo "$count" > "$FIXTURE/calls"
if [ "$count" -eq 1 ]; then echo '{error}' >&2; exit 17; fi
echo '{{"items":[]}}'
'''
    result = run_shell(tmp_path, f"ci_kubectl {operation} pods", command)
    assert result.returncode == status, result.stderr
    assert int((tmp_path / "calls").read_text()) == calls
    assert error in result.stderr
    if status == 0:
        assert json.loads(result.stdout) == {"items": []}



def test_response_body_timeout_retries_are_bounded(tmp_path):
    command = '''count=$(cat "$FIXTURE/calls" 2>/dev/null || echo 0)
count=$((count + 1)); echo "$count" > "$FIXTURE/calls"
echo 'request canceled (Client.Timeout or context cancellation while reading body)' >&2
exit 17
'''
    result = run_shell(tmp_path, "ci_kubectl get pods", command)
    assert result.returncode == 17
    assert int((tmp_path / "calls").read_text()) == 4


def test_all_replicas_schedule_before_startup(tmp_path):
    command = '''if [[ " $* " == *" job,statefulset,deployment "* ]]; then
  echo '{"items":[{"spec":{"replicas":2}}]}'; exit 0
fi
count=$(cat "$FIXTURE/calls" 2>/dev/null || echo 0)
count=$((count + 1)); echo "$count" > "$FIXTURE/calls"
if [ "$count" -lt 3 ]; then
  echo '{"items":[{"status":{"conditions":[{"type":"PodScheduled","status":"True"}]}},{"status":{"conditions":[{"type":"PodScheduled","status":"False"}]}}]}'
else
  echo '{"items":[{"status":{"conditions":[{"type":"PodScheduled","status":"True"}]}},{"status":{"conditions":[{"type":"PodScheduled","status":"True"}]}}]}'
fi
'''
    result = run_shell(tmp_path, 'wait_for_scheduling Target mx-target job-name=mx-target 30\necho "startup-start=$SECONDS"', command)
    assert result.returncode == 0, result.stderr
    assert "scheduled (2/2 pods)" in result.stdout
    assert int(result.stdout.rsplit("startup-start=", 1)[1]) >= 10
    assert int((tmp_path / "calls").read_text()) == 3


def test_scheduling_deadline_does_not_pass_an_empty_workload(tmp_path):
    result = run_shell(tmp_path, 'wait_for_scheduling Source mx-source job-name=mx-source 10', 'echo \'{"items":[]}\'\n')
    assert result.returncode != 0
    assert "did not schedule within 10s" in result.stderr


def test_worker_crash_fails_while_other_replica_is_pending(tmp_path):
    command = '''if [[ " $* " == *" job,statefulset,deployment "* ]]; then
  echo '{"spec":{"replicas":2}}'; exit 0
fi
echo '{"items":[{"metadata":{"name":"failed-worker"},"status":{"containerStatuses":[{"state":{"waiting":{"reason":"CrashLoopBackOff"}}}]}}]}'
'''
    result = run_shell(tmp_path, 'wait_for_scheduling Source mx-source job-name=mx-source 30', command)
    assert result.returncode != 0
    assert "pod failed while waiting for scheduling: failed-worker" in result.stderr
    assert "did not schedule within" not in result.stderr


@pytest.mark.parametrize("scenario,status,deletes", [
    ("success", 0, 2),
    ("reset", 0, 4),
    ("wrong_owner", 1, 0),
    ("missing", 0, 0),
    ("api_failure", 1, 0),
    ("api_failure_after_delete", 1, 2),
    ("stuck_finalizers", 1, 2),
])
def test_namespace_cleanup_requires_owner_and_confirmed_absence(tmp_path, scenario, status, deletes):
    command = f'''echo "$*" >> "$FIXTURE/commands"
scenario={scenario}
if [[ " $* " == *" get namespace "* ]]; then
  if [ "$scenario" = api_failure ] || {{ [ "$scenario" = api_failure_after_delete ] && [ -f "$FIXTURE/deleted" ]; }}; then
    echo 'connection refused' >&2; exit 1
  fi
  if [ "$scenario" = missing ] || {{ [ -f "$FIXTURE/deleted" ] && [ "$scenario" != stuck_finalizers ]; }}; then exit 0; fi
  owner=123-2
  if [ "$scenario" = wrong_owner ]; then owner=other-run; fi
  echo '{{"metadata":{{"labels":{{"ci.modelexpress.nvidia.com/run-id":"'$owner'"}}}}}}'; exit 0
fi
if [[ " $* " == *" delete namespace "* ]]; then
  if [ "$scenario" = reset ] && [ ! -f "$FIXTURE/reset" ]; then
    touch "$FIXTURE/reset"; echo 'connection reset by peer' >&2; exit 1
  fi
  touch "$FIXTURE/deleted"
fi
'''
    body = 'export GITHUB_RUN_ID=123 GITHUB_RUN_ATTEMPT=2\nexport -f sleep\nbash .github/scripts/ci-namespace.sh cleanup mx-ci-fixture-123-2'
    result = run_shell(tmp_path, body, command)
    assert result.returncode == status, result.stdout + result.stderr
    commands = (tmp_path / "commands").read_text().splitlines()
    assert sum(" delete " in f" {line} " for line in commands) == deletes
    assert not any("finalize" in line or "replace" in line for line in commands)
    if scenario == "api_failure_after_delete":
        assert "is gone" not in result.stdout
    if scenario == "stuck_finalizers":
        assert "finalizers were preserved" in result.stderr


def test_cleanup_waits_for_host_pods(tmp_path):
    (tmp_path / ".kubeconfig-host").write_text("fixture")
    command = '''echo "$*" >> "$FIXTURE/commands"
if [[ " $* " == *" get namespace "* ]]; then exit 0; fi
if [[ " $* " == *" get pods "* ]]; then
  echo '{"items":[{"metadata":{"name":"remaining-host-pod"}}]}'; exit 0
fi
exit 1
'''
    body = f'export GITHUB_RUN_ID=123 GITHUB_RUN_ATTEMPT=2 GITHUB_WORKSPACE="{tmp_path}"\nexport -f sleep\nbash .github/scripts/ci-namespace.sh cleanup mx-ci-fixture-123-2'
    result = run_shell(tmp_path, body, command)
    assert result.returncode != 0
    assert "host pods" in result.stderr
    assert "delete" not in (tmp_path / "commands").read_text()


def test_namespace_creation_records_run_attempt_ownership(tmp_path):
    command = '''if [[ " $* " == *" get namespace "* ]]; then exit 0; fi
if [[ " $* " == *" create namespace "* ]]; then
  echo '{"apiVersion":"v1","kind":"Namespace","metadata":{"name":"mx-ci-fixture-123-2"}}'; exit 0
fi
if [[ " $* " == *" apply "* ]]; then cat > "$FIXTURE/applied.json"; exit 0; fi
exit 1
'''
    result = run_shell(tmp_path, 'export GITHUB_RUN_ID=123 GITHUB_RUN_ATTEMPT=2\nbash .github/scripts/ci-namespace.sh create mx-ci-fixture-123-2 dynamo', command)
    assert result.returncode == 0, result.stderr
    labels = json.loads((tmp_path / "applied.json").read_text())["metadata"]["labels"]
    assert labels["ci.modelexpress.nvidia.com/run-id"] == "123-2"
    assert labels["ci.modelexpress.nvidia.com/workload-type"] == "dynamo"


def test_liveness_refreshes_replaced_pods(tmp_path):
    command = '''if [[ " $* " == *" get pods "* ]]; then
  count=$(cat "$FIXTURE/calls" 2>/dev/null || echo 0)
  count=$((count + 1)); echo "$count" > "$FIXTURE/calls"
  if [ "$count" -eq 1 ]; then
    echo '{"items":[{"metadata":{"name":"old"}},{"metadata":{"name":"terminating","deletionTimestamp":"now"}}]}'
  else
    echo '{"items":[{"metadata":{"name":"replacement"}}]}'
  fi
  exit 0
fi
if [[ " $* " == *" old "* ]]; then echo 'pod old not found' >&2; exit 1; fi
if [[ " $* " == *" replacement "* ]]; then exit 0; fi
exit 1
'''
    result = run_shell(tmp_path, 'wait_for_worker_liveness worker 1 30', command)
    assert result.returncode == 0, result.stderr
    assert "pod old not found" in result.stderr
    assert int((tmp_path / "calls").read_text()) == 2


@pytest.mark.parametrize("pool", ["a100a", "rl"])
def test_admission_waits_for_previous_labeled_namespace(tmp_path, pool):
    namespace = json.dumps({"items": [{"metadata": {"name": "mx-ci-bench-old", "labels": {
        "ci.modelexpress.nvidia.com/gpu-lane": "slot-0",
        "ci.modelexpress.nvidia.com/gpu-pool": pool,
    }}}]})
    command = f'''if [[ " $* " == *" get namespaces "* ]]; then
  count=$(cat "$FIXTURE/calls" 2>/dev/null || echo 0)
  count=$((count + 1)); echo "$count" > "$FIXTURE/calls"
  if [ "$count" -eq 1 ]; then echo '{namespace}'; else echo '{{"items":[]}}'; fi
else echo '{{"items":[]}}'; fi
'''
    result = run_shell(tmp_path, f'export GITHUB_WORKSPACE=. KUBE_CONTEXT=fixture MX_CI_GPU_LANE=slot-0\nwait_for_gpu_pool_cleanup {pool}', command)
    assert result.returncode == 0, result.stderr
    assert "Waiting for GPU pool" in result.stdout
    assert int((tmp_path / "calls").read_text()) == 2


def test_admission_does_not_treat_api_failure_as_free_capacity(tmp_path):
    result = run_shell(tmp_path, 'wait_for_gpu_pool_cleanup a100a', 'echo Forbidden >&2; exit 17')
    assert result.returncode != 0
    assert "Forbidden" in result.stderr


@pytest.mark.parametrize("lane,pool,status", [
    ("slot-0", "a100a", 1),
    ("slot-1", "a100a", 0),
    ("slot-0", "a100b", 0),
    (None, "a100a", 0),
])
def test_orphan_host_pods_require_matching_mirrored_slot_and_pool(tmp_path, lane, pool, status):
    pods = json.dumps({"items": [{"metadata": {"namespace": "mx-ci", "labels": {
        "vcluster.loft.sh/namespace": "mx-ci-vllm-old",
        "vcluster.loft.sh/ns-label-mx-ci-cluster-x-9c8bc18f28": lane,
        "vcluster.loft.sh/ns-label-mx-ci-cluster-x-dc1dfe5427": pool,
    }}}]})
    command = f'''echo "$*" >> "$FIXTURE/commands"
if [[ " $* " == *" get namespaces "* ]]; then echo '{{"items":[]}}'; else echo '{pods}'; fi
'''
    result = run_shell(tmp_path, 'export GITHUB_WORKSPACE=. MX_CI_GPU_LANE=slot-0\nwait_for_gpu_pool_cleanup a100a', command)
    assert result.returncode == status, result.stderr
    calls = (tmp_path / "commands").read_text()
    assert " -A " not in calls
    assert "-n mx-ci" in calls
    assert "ns-label-mx-ci-cluster-x-9c8bc18f28=slot-0" in calls
    if status:
        assert "after 900s" in result.stderr


@pytest.mark.parametrize("lane,other,status", [
    ("slot-0", "slot-1", 0),
    ("slot-0", "slot-0", 1),
    ("slot-0", "rdma", 0),
    ("slot-0", "slot-9", 0),
    ("slot-0", None, 0),
    ("all", "slot-0", 0),
    ("all", "all", 1),
])
@pytest.mark.parametrize("pool", ["a100a", "rl"])
def test_cleanup_guard_only_blocks_exact_labeled_slot(tmp_path, lane, other, status, pool):
    namespaces = json.dumps({"items": [{"metadata": {"name": "mx-ci-vllm-old", "labels": {
        "ci.modelexpress.nvidia.com/gpu-lane": other,
        "ci.modelexpress.nvidia.com/gpu-pool": pool,
    }}}]})
    command = f'''echo "$*" >> "$FIXTURE/commands"
if [[ " $* " == *" get namespaces "* ]]; then echo '{namespaces}'; else echo '{{"items":[]}}'; fi
'''
    result = run_shell(tmp_path, f'export GITHUB_WORKSPACE=. KUBE_CONTEXT=fixture MX_CI_GPU_LANE={lane} MX_CI_GPU_CONCURRENCY=4\nwait_for_gpu_pool_cleanup {pool}', command)
    assert result.returncode == status, result.stderr
    calls = (tmp_path / "commands").read_text()
    assert f"ci.modelexpress.nvidia.com/gpu-lane={lane}" in calls
    if pool == "rl":
        assert "get pods" not in calls
    if status:
        assert "after 900s" in result.stderr


@pytest.mark.parametrize("pool", ["a100a", "rl"])
def test_cleanup_accepts_large_cluster_pod_list(tmp_path, pool):
    pods = {"items": [{"metadata": {"namespace": "other-workload", "annotations": {"fixture": "x" * 2_000_000}}, "spec": {}, "status": {"phase": "Running"}}]}
    (tmp_path / "pods.json").write_text(json.dumps(pods))
    command = '''if [[ " $* " == *" get namespaces "* ]]; then echo '{"items":[]}'; else cat "$FIXTURE/pods.json"; fi
'''
    result = run_shell(tmp_path, f'export GITHUB_WORKSPACE=. KUBE_CONTEXT=fixture MX_CI_GPU_LANE=slot-15 MX_CI_GPU_CONCURRENCY=16\nwait_for_gpu_pool_cleanup {pool}', command)
    assert result.returncode == 0, result.stderr
    assert "no previous CI workload" in result.stdout



@pytest.mark.parametrize("operation", ["health", "vram"])
def test_empty_pod_list_recovers_when_replacement_appears(tmp_path, operation):
    command = '''if [[ " $* " == *" -o json "* ]]; then
  count=$(cat "$FIXTURE/calls" 2>/dev/null || echo 0)
  count=$((count + 1)); echo "$count" > "$FIXTURE/calls"
  if [ "$count" -eq 1 ]; then echo '{"items":[]}'; else echo '{"items":[{"metadata":{"name":"replacement"}}]}'; fi
  exit 0
fi
if [[ " $* " == *" jsonpath="* ]]; then echo 'array index out of bounds: index 0, length 0' >&2; exit 17; fi
if [[ " $* " == *" nvidia-smi "* ]]; then echo 123; fi
exit 0
'''
    body = 'source .github/actions/run-mx-p2p-test/vram-sampling.sh\ntest_pod=; test_peak=0; test_samples=0\n'
    if operation == "health":
        body += 'wait_for_health Target fixture worker 8000 test_pod 30\necho "pod=$test_pod"'
    else:
        body += 'sample_peak_vram Target fixture worker test_pod test_peak test_samples\n[ "$test_samples" -eq 0 ]\nsample_peak_vram Target fixture worker test_pod test_peak test_samples\necho "pod=$test_pod peak=$test_peak samples=$test_samples"'
    result = run_shell(tmp_path, body, command)
    assert result.returncode == 0, result.stderr
    assert "pod=replacement" in result.stdout
    assert int((tmp_path / "calls").read_text()) == 2
    if operation == "vram":
        assert "peak=123 samples=1" in result.stdout


@pytest.mark.parametrize("operation", ["health", "vram"])
def test_pod_lookup_api_errors_still_fail(tmp_path, operation):
    body = 'source .github/actions/run-mx-p2p-test/vram-sampling.sh\ntest_pod=; test_peak=0; test_samples=0\n'
    body += 'wait_for_health Target fixture worker 8000 test_pod 30' if operation == "health" else 'sample_peak_vram Target fixture worker test_pod test_peak test_samples'
    result = run_shell(tmp_path, body, 'echo Forbidden >&2; exit 17')
    assert result.returncode != 0
    assert "Forbidden" in result.stderr


def test_cleanup_waits_for_slow_vcluster_namespace_finalization(tmp_path):
    command = '''if [[ " $* " == *" get namespace "* ]]; then
  if [ -f "$FIXTURE/deleted" ]; then
    reads=$(cat "$FIXTURE/reads" 2>/dev/null || echo 0)
    reads=$((reads + 1)); echo "$reads" > "$FIXTURE/reads"
    if [ "$reads" -ge 100 ]; then exit 0; fi
  fi
  echo '{"metadata":{"labels":{"ci.modelexpress.nvidia.com/run-id":"123-2"}}}'; exit 0
fi
if [[ " $* " == *" delete namespace "* ]]; then touch "$FIXTURE/deleted"; fi
'''
    body = 'export GITHUB_RUN_ID=123 GITHUB_RUN_ATTEMPT=2\nexport -f sleep\nbash .github/scripts/ci-namespace.sh cleanup mx-ci-fixture-123-2'
    result = run_shell(tmp_path, body, command)
    assert result.returncode == 0, result.stdout + result.stderr
    assert int((tmp_path / "reads").read_text()) == 100
    assert "host pods are gone" in result.stdout
