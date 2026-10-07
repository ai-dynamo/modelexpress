# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise namespace ownership and failure cleanup without a cluster."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("profile", ["nemotron", "kimi"])
@pytest.mark.parametrize("owner", ["123-1", "someone-else"])
def test_cleanup_never_deletes_foreign_namespace_and_retains_owned_on_failure(
    tmp_path, owner, profile
):
    executable = tmp_path / "kubectl"
    log = tmp_path / "calls"
    executable.write_text("""#!/usr/bin/env python3
import os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['CALLS']).open('a') as out:
    out.write(' '.join(args) + '\\n')
if 'get' in args:
    print(os.environ['OWNER'] if any('go-template' in arg for arg in args) else 'namespace/test')
if 'apply' in args:
    sys.exit(1)
""")
    executable.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "CALLS": str(log),
        "OWNER": owner,
        "MODEL_PROFILE": profile,
        "NAMESPACE": "test",
        "KUBE_CONTEXT": "ci",
        "RUN_ID": profile + "-123-1",
        "RESULTS_DIR": str(tmp_path / "results"),
        "SERVER_IMAGE": "registry/server@sha256:" + "a" * 64,
        "WORKER_IMAGE": "registry/worker@sha256:" + "b" * 64,
        "MX_BENCH_S3_ROLE_ARN": "arn:aws:iam::123:role/test",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/ci.py"), "cleanup"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    calls = log.read_text()
    if owner == "123-1":
        assert f"delete pod mx-{profile}-123-1-control mx-{profile}-123-1-s3" in calls
        assert "delete namespace test" not in calls
    else:
        assert "delete" not in calls and "apply" not in calls


@pytest.mark.parametrize("profile,gpus", [("nemotron", 1), ("kimi", 8)])
def test_setup_sizes_quota_from_selected_profile(tmp_path, profile, gpus):
    executable = tmp_path / "kubectl"
    log = tmp_path / "calls"
    executable.write_text("""#!/usr/bin/env python3
import os, sys
from pathlib import Path
with Path(os.environ['CALLS']).open('a') as out:
    out.write(' '.join(sys.argv[1:]) + '\\n')
sys.exit(0)
""")
    executable.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "CALLS": str(log),
        "MODEL_PROFILE": profile,
        "NAMESPACE": "test",
        "KUBE_CONTEXT": "ci",
        "RUN_ID": profile + "-123-1",
        "RESULTS_DIR": str(tmp_path / "results"),
        "SERVER_IMAGE": "registry/server@sha256:" + "a" * 64,
        "WORKER_IMAGE": "registry/worker@sha256:" + "b" * 64,
        "MX_BENCH_S3_ROLE_ARN": "arn:aws:iam::123:role/test",
        "NGC_API_KEY": "fake",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/ci.py"), "setup"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "apply" not in log.read_text()
    assert (
        f"create quota bench-gpu-budget --hard=requests.nvidia.com/gpu={gpus},limits.nvidia.com/gpu={gpus}"
        in log.read_text()
    )


def test_cleanup_failure_can_retry_with_existing_results(tmp_path):
    executable = tmp_path / "kubectl"
    log = tmp_path / "calls"
    attempts = tmp_path / "attempts"
    executable.write_text("""#!/usr/bin/env python3
import os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['CALLS']).open('a') as out:
    out.write(' '.join(args) + '\\n')
if 'get' in args:
    print('123-1' if any('go-template' in arg for arg in args) else ('namespace/test' if 'namespace' in args else '{"items": []}'))
if 'wait' in args and any('Succeeded' in arg for arg in args):
    attempts = Path(os.environ['ATTEMPTS'])
    previous = int(attempts.read_text()) if attempts.exists() else 0
    attempts.write_text(str(previous + 1))
    sys.exit(1 if previous == 0 else 0)
if 'logs' in args:
    print('{"verified_absent": true}')
""")
    executable.chmod(0o755)
    results = tmp_path / "results"
    results.mkdir()
    (results / "bench-driver.log").write_text("saved evidence")
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "CALLS": str(log),
        "ATTEMPTS": str(attempts),
        "MODEL_PROFILE": "nemotron",
        "NAMESPACE": "test",
        "KUBE_CONTEXT": "ci",
        "RUN_ID": "nemotron-123-1",
        "RESULTS_DIR": str(results),
        "SERVER_IMAGE": "registry/server@sha256:" + "a" * 64,
        "WORKER_IMAGE": "registry/worker@sha256:" + "b" * 64,
        "MX_BENCH_S3_ROLE_ARN": "arn:aws:iam::123:role/test",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    command = [sys.executable, str(ROOT / "scripts/ci.py"), "cleanup"]
    failed = subprocess.run(
        command, env=env, capture_output=True, text=True, check=False
    )
    assert failed.returncode != 0
    assert "delete namespace test" not in log.read_text()
    assert "mx-nemotron-123-1-control-cleanup --ignore-not-found" in log.read_text()
    retried = subprocess.run(
        command, env=env, capture_output=True, text=True, check=False
    )
    assert retried.returncode == 0, retried.stderr
    assert "delete namespace test" in log.read_text()
    assert (results / "bench-driver.log").read_text() == "saved evidence"
    assert attempts.read_text() == "2"
