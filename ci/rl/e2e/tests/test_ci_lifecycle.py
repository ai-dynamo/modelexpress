"""Exercise namespace ownership and failure cleanup without a cluster."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("profile", ["nemotron", "kimi"])
@pytest.mark.parametrize("owner", ["123-1", "someone-else"])
def test_cleanup_never_deletes_foreign_namespace_and_reaps_owned_on_failure(
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
        "MX_E2E_S3_ROLE_ARN": "arn:aws:iam::123:role/test",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/ci.sh"), "cleanup"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    calls = log.read_text()
    if owner == "123-1":
        assert f"delete pod mx-{profile}-123-1-control mx-{profile}-123-1-s3" in calls
        assert "delete namespace test" in calls
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
# Stop at deployment; all earlier operations are offline stubs.
sys.exit(1 if 'apply' in sys.argv else 0)
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
        "MX_E2E_S3_ROLE_ARN": "arn:aws:iam::123:role/test",
        "NGC_API_KEY": "fake",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/ci.sh"), "setup"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert (
        f"create quota e2e-gpu-budget --hard=requests.nvidia.com/gpu={gpus},limits.nvidia.com/gpu={gpus}"
        in log.read_text()
    )
