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
def test_admission_waits_for_previous_namespace_and_physical_pods(tmp_path, pool):
    command = '''if [[ " $* " == *" get namespaces "* ]]; then
  count=$(cat "$FIXTURE/calls" 2>/dev/null || echo 0)
  count=$((count + 1)); echo "$count" > "$FIXTURE/calls"
  if [ "$count" -eq 1 ]; then echo '{"items":[{"metadata":{"name":"mx-ci-bench-old"}}]}'; else echo '{"items":[]}'; fi
  exit 0
fi
if [ "$(cat "$FIXTURE/calls")" -eq 1 ]; then
  echo '{"items":[{"metadata":{"namespace":"mx-ci-bench-old"},"spec":{"nodeSelector":{"agentpool":"a100a"},"containers":[{"resources":{"requests":{"nvidia.com/gpu":"1"}}}]},"status":{"phase":"Pending"}}]}'
else echo '{"items":[]}'; fi
'''
    result = run_shell(tmp_path, f'export GITHUB_WORKSPACE="{tmp_path}" KUBE_CONTEXT=fixture\nwait_for_gpu_pool_cleanup {pool}', command)
    assert result.returncode == 0, result.stderr
    assert "Waiting for GPU pool" in result.stdout
    assert int((tmp_path / "calls").read_text()) == 2


def test_admission_does_not_treat_api_failure_as_free_capacity(tmp_path):
    result = run_shell(tmp_path, 'wait_for_gpu_pool_cleanup a100a', 'echo Forbidden >&2; exit 17')
    assert result.returncode != 0
    assert "Forbidden" in result.stderr


def test_admission_waits_for_host_pods_after_virtual_namespace_disappears(tmp_path):
    command = '''if [[ " $* " == *" get namespaces "* ]]; then echo '{"items":[]}'; exit 0; fi
 echo '{"items":[{"spec":{"nodeSelector":{"agentpool":"a100a"},"containers":[{"resources":{"limits":{"nvidia.com/gpu":"1"}}}]},"status":{"phase":"Running"}}]}'
'''
    result = run_shell(tmp_path, 'export GITHUB_WORKSPACE=.\nwait_for_gpu_pool_cleanup a100a', command)
    assert result.returncode != 0
    assert "after 900s" in result.stderr
