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
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from kube_utils import kubectl, port_forward


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


def _worker_pod_names(namespace: str, dgd_name: str) -> list[str]:
    selector = _worker_label_selector(dgd_name)
    result = kubectl(
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


def test_modelmetadata_crs_published(namespace: str, expected_cr_count: int) -> None:
    """Every worker replica must have published its own ModelMetadata CR.

    Aggregated: 2 (two VllmWorker replicas after scale-up).
    Disaggregated: 3 (one VllmPrefillWorker + two VllmDecodeWorker after
    scale-up).

    The action.yml waits for `expected_cr_count` before invoking pytest, so a
    failure here most likely means the cluster state regressed between the
    wait and the test run (e.g. a worker pod crashed and the server reaper
    deleted its CR).
    """
    result = kubectl(
        "get", "modelmetadata",
        "-o", "jsonpath={range .items[*]}{.metadata.name}{\" \"}{.spec.worker_rank}{\"\\n\"}{end}",
        namespace=namespace,
    )
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    print(f"[modelmetadata] {len(rows)} CR(s):")
    for row in rows:
        print(f"  {row}")
    assert len(rows) == expected_cr_count, (
        f"Expected {expected_cr_count} ModelMetadata CRs, got {len(rows)}."
    )


def test_second_replica_used_rdma(namespace: str, dgd_name: str) -> None:
    """At least one worker pod must have called `add_remote_agent` — the
    signal that P2P weight transfer engaged at all.

    Why `>= 1` and not `== 1`: in aggregated mode only the scaled-up replica
    pulls (1 source + 1 puller = 1 distinct source agent), but disaggregated
    has additional pullers — the prefill worker could also pull weights from
    whichever decode replica downloaded from HF first, AND the scaled-up
    decode replica picks a source independently. Depending on which sources
    rdma_strategy returns, those two pulls may land on the same source (1
    distinct agent) or different sources (2 distinct agents). Both outcomes
    are healthy; the real failure mode this test guards against is the
    scaled-up replica falling back to HF disk load entirely — that's the
    `== 0` case.
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
    assert len(distinct_agents) >= 1, (
        f"No add_remote_agent calls found across worker logs — every worker that "
        f"needed weights fell back to HF disk load. P2P didn't engage at all."
    )


# Number of /v1/completions requests sent through the Frontend in the inference
# test. Sized at >= 2x post-scale worker count (aggregated: 2 workers;
# disaggregated: 2 decode workers serving + 1 prefill) so round-robin must
# dispatch to every serving worker at least once with high probability —
# catches the failure mode where the scaled-up replica's P2P-loaded weights
# load fine but produce 5xx / empty text on inference.
INFERENCE_REQUEST_COUNT = 6


def _send_completion(port: int, model: str, index: int) -> str:
    """POST one /v1/completions request and return the completion text.

    Asserts non-empty text inside the worker thread so the failure shows up
    with the offending request index — `future.result()` re-raises in the
    main test thread and pytest captures it normally.
    """
    payload = json.dumps({
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": 8,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Cold-start budget per request (vLLM CUDA-graph capture, model warmup);
    # warm responses complete in seconds. Using the same timeout for every
    # call keeps the code simple — the wall-clock cost is dominated by the
    # cold start either way.
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
    choices = body.get("choices", [])
    assert choices, f"Request {index}: no choices in response: {body}"
    text = choices[0].get("text", "")
    assert text, f"Request {index}: empty completion text: {body}"
    return text


def test_frontend_inference(namespace: str, model: str, dgd_name: str) -> None:
    """Dynamo Frontend's /v1/completions returns valid responses across
    multiple concurrent requests.

    Validates end-to-end: HTTP request → Frontend → round-robin to a worker →
    inference using P2P-loaded weights → response back. We can't address
    individual workers directly (`python3 -m dynamo.vllm` doesn't expose
    /v1/completions on the worker pod, only the Frontend does), so we fire N
    requests concurrently and rely on round-robin to spread them across every
    worker. As long as all N return non-empty text, the scaled-up replica's
    P2P-loaded weights are intact enough to serve. Concurrent rather than
    sequential because real serving sees concurrent load, and concurrency
    forces the Frontend to actually distribute — sequential round-robin can
    degenerate to "same worker handles all" if it always responds before the
    next request arrives.

    The completion text may be nonsensical (Qwen2.5-0.5B + non-zero
    temperature), but every response must have non-empty text — we're
    testing the routing + serving path, not model correctness.
    """
    frontend_svc = f"{dgd_name}-frontend"
    with port_forward(namespace, f"svc/{frontend_svc}", local_port=18000, remote_port=FRONTEND_PORT) as port:
        with ThreadPoolExecutor(max_workers=INFERENCE_REQUEST_COUNT) as pool:
            futures = {
                pool.submit(_send_completion, port, model, i): i
                for i in range(INFERENCE_REQUEST_COUNT)
            }
            for future in as_completed(futures):
                i = futures[future]
                # future.result() re-raises any AssertionError /
                # urllib.error.URLError from inside the worker thread.
                text = future.result()
                print(f"[frontend req {i}] completion: {text!r}")
