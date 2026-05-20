# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared K8s P2P CI test — used by all inference engine frameworks.

Runs after the workflow has applied the source and target Jobs and both pods
have reached Running state.  Asserts:
  1. Target pod logs contain the framework-specific P2P transfer marker (--p2p-marker).
  2. Target connected to `--tp-size` distinct source NIXL agents covering ranks
     0..tp_size-1 (catches TP collapse and source-rank-pairing regressions).
  3. Both source and target servers respond to /v1/completions — confirms weights
     are loaded and the model is serving correctly on each.

Invoked by the workflow as:
  pytest ci/k8s/client/test_p2p_k8s.py -v \
      --namespace $NAMESPACE \
      --model $MX_CI_MODEL \
      --source-port $SOURCE_PORT \
      --worker-port $WORKER_PORT \
      --tp-size $TP_SIZE \
      [--p2p-marker "framework-specific transfer complete string"]

--p2p-marker defaults:
  vLLM:    "RDMA transfer complete"             (emitted by vLLM's RdmaStrategy)
  TRT-LLM: "ModelExpress P2P transfer complete" (printed by trtllm_p2p_launcher.py)

--tp-size default is 1 — every existing TP=1 P2P test still asserts exactly
one source agent, which is the correct expectation and adds a free safety net.

Known gaps for multi-node TP (manifest scaffolded at
ci/k8s/client/vllm/manifest-azure-multi-node-tp2.yaml but not yet wired in):

  1. `_pod_name` returns `.items[0]` from a label-filtered list, which has no
     ordinal ordering. Multi-node manifests are StatefulSets where pod-0 is
     the Ray head + vLLM API server and pod-1 is a Ray worker (no HTTP
     endpoint, no model). The inference + log-fetch helpers need to pin to
     pod-0 — either filter by `statefulset.kubernetes.io/pod-name=<sts>-0`
     or extend `_pod_name` with an `ordinal` arg.

  2. Log-scanning tests (`test_rdma_transfer_logged`,
     `test_per_rank_source_agents`) fetch logs from a single pod. With
     vLLM `--distributed-executor-backend=ray` the rank-1 worker process
     lives in pod-1, and Ray does not forward worker stdout into the head
     pod's container logs by default. So rank 1's `add_remote_agent: ...`
     line and rank 1's `[Worker 1] RDMA transfer complete` line both live
     in pod-1's logs, not pod-0's. Need to concat logs from every pod in
     the StatefulSet before running the regexes.

  3. `_assert_inference` port-forwards to whichever pod `_pod_name` picks.
     Same pod-0 pinning as (1) — pod-1 doesn't serve HTTP.

  None of this affects TP=1 or single-node TP=2: those are Jobs with a
  single pod per role, so `.items[0]` is unambiguous and all logs live in
  one place. The cleanest fix is a helper like `_pods_in_order(ns, job)`
  that returns ordinal-sorted pod names, plus an `ordinal` arg on the
  existing helpers defaulting to 0 (backward-compatible).
"""

import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager


def _kubectl(*args: str, namespace: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", "-n", namespace, *args],
        capture_output=True,
        text=True,
        check=check,
    )


def _pod_name(namespace: str, job_name: str) -> str:
    result = _kubectl(
        "get", "pods",
        "-l", f"job-name={job_name}",
        "-o", "jsonpath={.items[0].metadata.name}",
        namespace=namespace,
    )
    name = result.stdout.strip()
    assert name, f"{job_name} pod not found"
    return name


@contextmanager
def _port_forward(namespace: str, pod: str, local_port: int, remote_port: int):
    proc = subprocess.Popen(
        ["kubectl", "-n", namespace, "port-forward", pod, f"{local_port}:{remote_port}"],
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
                break  # server responded — it's up even if /health returns an error code
            except (urllib.error.URLError, ConnectionRefusedError):
                time.sleep(1)
        yield local_port
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def _assert_inference(namespace: str, job_name: str, model: str, remote_port: int, local_port: int) -> None:
    pod = _pod_name(namespace, job_name)
    print(f"\n[{job_name}] pod={pod} remote_port={remote_port} local_port={local_port}")
    # TODO: replace with a more complex prompt that exercises multi-token reasoning
    # to better validate model correctness beyond a single-word completion.
    payload = json.dumps({
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": 8,
    }).encode()
    with _port_forward(namespace, pod, local_port=local_port, remote_port=remote_port) as port:
        req = urllib.request.Request(
            f"http://localhost:{port}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # First inference per pod triggers TRT-LLM cold-start (CUDA graph
        # capture + JIT compile) which can exceed 60s, especially when source
        # and target share a node and contend for GPU scheduling.
        # Subsequent requests are sub-second.
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
    print(f"[{job_name}] response: {json.dumps(body, indent=2)}")
    choices = body.get("choices", [])
    assert choices, f"No choices in response from {job_name}: {body}"
    text = choices[0].get("text", "")
    print(f"[{job_name}] completion text: {text!r}")
    assert text, f"Empty completion text from {job_name}: {body}"


def test_rdma_transfer_logged(namespace: str, p2p_marker: str) -> None:
    """Target pod logs must contain the framework's P2P transfer marker.

    Absence means the target loaded weights via a fallback path, not RDMA.
    """
    pod = _pod_name(namespace, "mx-target")
    print(f"\n[mx-target] pod={pod}")
    result = _kubectl("logs", pod, "-c", "mx-target", "--tail=-1", namespace=namespace)
    marker_lines = [l for l in result.stdout.splitlines() if any(k in l for k in ("RDMA", "P2P", "transfer"))]
    print(f"[mx-target] transfer log lines:\n" + "\n".join(marker_lines))
    assert p2p_marker in result.stdout, (
        f"P2P marker {p2p_marker!r} not found in target logs.\n"
        f"Last 50 log lines:\n" + "\n".join(result.stdout.splitlines()[-50:])
    )


def test_per_rank_source_agents(namespace: str, tp_size: int) -> None:
    """Target must connect to one distinct source NIXL agent per target rank.

    Each source rank publishes exactly one NIXL agent. The shared
    nixl_transfer.py:386 logs `add_remote_agent: ... (agent=b'<name>')` for
    every target→source connection in both engine paths (vLLM via
    rdma_strategy.py and TRT-LLM via trtllm_live_transfer.py), though the
    agent-name format differs per engine (see regex below). Failure modes:

      - TP collapse: fewer than `tp_size` add_remote_agent lines.
      - Source-rank collapse (e.g. all target ranks pulling from source rank 0
        because the rank filter regressed in _find_source_instances): same agent
        name repeats across target ranks → distinct-count < tp_size.
      - Wrong source-rank set: source ranks observed don't cover 0..tp_size-1.

    Inference alone can't catch source-rank collapse because TP shards have the
    same shape per rank, so the load succeeds with wrong values and produces
    plausible-but-garbage text — which the current non-empty assertion passes.
    """
    pod = _pod_name(namespace, "mx-target")
    result = _kubectl("logs", pod, "-c", "mx-target", "--tail=-1", namespace=namespace)
    # Two NIXL agent naming schemes are in use:
    #   vLLM    (load_strategy/base.py:90)    — mx-{role}-worker{rank}-{uuid8}
    #   TRT-LLM (trtllm_live_transfer.py:107) — trtllm-live-source-rank{rank}-{pid}
    # Both encode source rank as the integer after `worker` / `rank`; one regex
    # captures (full_agent_name, source_rank) for either format so the test
    # stays engine-agnostic.
    matches = re.findall(
        r"agent=b?'((?:mx-\w+-worker|trtllm-live-source-rank)(\d+)[-\w]*)'",
        result.stdout,
    )

    distinct_pairs = set(matches)
    distinct_agents = {name for name, _ in distinct_pairs}
    source_ranks = sorted({int(r) for _, r in distinct_pairs})
    print(f"[mx-target] distinct source agents: {distinct_agents}  source_ranks={source_ranks}")

    assert len(distinct_agents) == tp_size, (
        f"Expected {tp_size} distinct source agent(s), got {len(distinct_agents)}: "
        f"{distinct_agents}. Fewer means TP collapse (target rank didn't run) or "
        f"source-rank collapse (multiple target ranks pulled from the same source)."
    )
    assert source_ranks == list(range(tp_size)), (
        f"Expected source ranks {list(range(tp_size))}, got {source_ranks}."
    )


def test_source_inference_produces_output(namespace: str, model: str, source_port: int) -> None:
    """Source server must return a valid completion response."""
    _assert_inference(namespace, "mx-source", model, remote_port=source_port, local_port=18001)


def test_target_inference_produces_output(namespace: str, model: str, worker_port: int) -> None:
    """Target server must return a valid completion response after P2P transfer."""
    _assert_inference(namespace, "mx-target", model, remote_port=worker_port, local_port=18000)
