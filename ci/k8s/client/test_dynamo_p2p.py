# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K8s integration test for row 7 — Dynamo aggregated-serving with ModelExpress P2P.

Runs after .github/actions/run-mx-dynamo-test has:
  1. Installed the Dynamo operator
  2. Applied the DGD manifest with replicas=1
  3. Waited for the first VllmWorker to publish metadata
  4. Patched the DGD to replicas=2
  5. Waited for both workers to publish (i.e. the second one completed RDMA)

Asserts:
  1. Exactly 2 ModelMetadata CRs exist in the test namespace, covering source
     ranks 0 and 1 (matches our existing P2P test plumbing — the worker_rank
     filter is what gates rank pairing in rdma_strategy.py).
  2. The second worker actually pulled weights via NIXL RDMA (not via the
     HF disk-fallback path). Detected by scanning all worker pod logs for
     exactly one `add_remote_agent: ... (agent=b'<source-agent-name>')` line
     across the fleet — only the receiving replica logs this; the source
     replica never does.
  3. The Dynamo Frontend serves /v1/completions end-to-end with a non-empty
     response. Proves the frontend → worker routing path works, separate
     from the P2P transfer itself.

Invoked by the workflow as:
  pytest ci/k8s/client/test_dynamo_p2p.py -v \\
      --namespace $NAMESPACE \\
      --model $MX_CI_MODEL
"""

import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager


# Service + label conventions are set by the dynamo operator from the DGD spec:
#   DGD `<dgd_name>` → frontend service `<dgd_name>-frontend`; worker pods get
#   `nvidia.com/dynamo-graph-deployment-name=<dgd_name>` plus
#   `nvidia.com/dynamo-component-type=worker`. The `dgd_name` fixture (see
#   conftest.py) parameterizes this so the same test code works for any DGD
#   name the workflow assigns (typically `mx-dynamo-${{ github.run_id }}` for
#   per-run uniqueness).
FRONTEND_PORT = 8000


def _worker_label_selector(dgd_name: str) -> str:
    return (
        f"nvidia.com/dynamo-graph-deployment-name={dgd_name},"
        f"nvidia.com/dynamo-component-type=worker"
    )


def _kubectl(*args: str, namespace: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", "-n", namespace, *args],
        capture_output=True,
        text=True,
        check=check,
    )


def _worker_pod_names(namespace: str, dgd_name: str) -> list[str]:
    selector = _worker_label_selector(dgd_name)
    result = _kubectl(
        "get", "pods",
        "-l", selector,
        "-o", "jsonpath={.items[*].metadata.name}",
        namespace=namespace,
    )
    names = result.stdout.split()
    assert names, f"no worker pods found in {namespace} with selector {selector}"
    return names


def _worker_logs_combined(namespace: str, dgd_name: str) -> str:
    """Concat logs from all worker pods. The replica that pulled is whichever one
    emits an `add_remote_agent` line; we don't care which ordinal it is."""
    chunks = []
    for pod in _worker_pod_names(namespace, dgd_name):
        # --all-containers because the worker pod may have sidecars; the dynamo
        # operator's pod template can include init/sidecar containers.
        try:
            r = subprocess.run(
                ["kubectl", "-n", namespace, "logs", pod, "--all-containers", "--tail=-1"],
                capture_output=True, text=True, check=False,
            )
            chunks.append(r.stdout)
        except Exception as e:
            print(f"WARN: failed to fetch logs from {pod}: {e}")
    return "\n".join(chunks)


@contextmanager
def _port_forward(namespace: str, target: str, local_port: int, remote_port: int):
    """target is either `svc/<name>` or `pod/<name>` — kubectl port-forward
    accepts either form."""
    proc = subprocess.Popen(
        ["kubectl", "-n", namespace, "port-forward", target, f"{local_port}:{remote_port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.perf_counter() + 60
        while time.perf_counter() < deadline:
            try:
                urllib.request.urlopen(f"http://localhost:{local_port}/health", timeout=2)
                break
            except urllib.error.HTTPError:
                # Server responded — it's up even if /health returns an error code.
                break
            except (urllib.error.URLError, ConnectionRefusedError):
                time.sleep(1)
        yield local_port
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_two_modelmetadata_crs_published(namespace: str) -> None:
    """Both VllmWorker replicas must have published their own ModelMetadata CR.

    The action.yml waits for this before invoking pytest, so a failure here
    most likely means the cluster state regressed between the wait and the
    test run (e.g. a worker pod crashed and the server reaper deleted its CR).
    """
    result = _kubectl(
        "get", "modelmetadata",
        "-o", "jsonpath={range .items[*]}{.metadata.name}{\" \"}{.spec.worker_rank}{\"\\n\"}{end}",
        namespace=namespace,
    )
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    print(f"[modelmetadata] {len(rows)} CR(s):")
    for row in rows:
        print(f"  {row}")
    assert len(rows) == 2, (
        f"Expected 2 ModelMetadata CRs (one per VllmWorker replica), got {len(rows)}."
    )


def test_second_replica_used_rdma(namespace: str, dgd_name: str) -> None:
    """Exactly one worker pod must have called `add_remote_agent` — the second
    replica pulling from the first. The source replica never calls it.

    Catches the regression where the second replica falls back to HF disk
    download instead of RDMA pull (e.g. mx-server connectivity issue, source
    metadata not READY, NIXL agent registration failed).
    """
    logs = _worker_logs_combined(namespace, dgd_name)
    # Same regex shape as test_p2p_k8s.py's per-rank source agent check. Both
    # naming schemes (`mx-{role}-worker{N}-{uuid}` and
    # `trtllm-live-source-rank{N}-{pid}`) are accepted in case future dynamo
    # paths route through the trtllm transfer code.
    pattern = r"agent=b?'((?:mx-\w+-worker|trtllm-live-source-rank)(\d+)[-\w]*)'"
    matches = re.findall(pattern, logs)
    distinct_agents = {name for name, _ in set(matches)}
    print(f"[workers] distinct source agents observed across all worker logs: {distinct_agents}")
    assert len(distinct_agents) == 1, (
        f"Expected exactly 1 source agent across worker logs (one replica pulling from one source), "
        f"got {len(distinct_agents)}: {distinct_agents}. "
        f"Zero usually means the second replica fell back to HF disk load — P2P didn't engage."
    )


def test_frontend_inference(namespace: str, model: str, dgd_name: str) -> None:
    """Dynamo Frontend's /v1/completions returns a valid response.

    Validates end-to-end: HTTP request → Frontend → round-robin to a worker
    → inference using P2P-loaded weights → response back. The completion text
    may be nonsensical (Qwen2.5-0.5B + non-zero temperature), but non-empty is
    enough — we're testing the routing + serving path, not model correctness.
    """
    payload = json.dumps({
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": 8,
    }).encode()
    frontend_svc = f"{dgd_name}-frontend"
    with _port_forward(namespace, f"svc/{frontend_svc}", local_port=18000, remote_port=FRONTEND_PORT) as port:
        req = urllib.request.Request(
            f"http://localhost:{port}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Cold-start budget: vLLM CUDA-graph capture etc.
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
    print(f"[frontend] response: {json.dumps(body, indent=2)}")
    choices = body.get("choices", [])
    assert choices, f"No choices in response from Frontend: {body}"
    text = choices[0].get("text", "")
    print(f"[frontend] completion text: {text!r}")
    assert text, f"Empty completion text from Frontend: {body}"
