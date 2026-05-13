# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared K8s P2P CI test — used by all inference engine frameworks.

Runs after the workflow has applied the source and target Jobs and both pods
have reached Running state.  Asserts:
  1. Target pod logs contain the framework-specific P2P transfer marker (--p2p-marker).
  2. Both source and target servers respond to /v1/completions — confirms weights
     are loaded and the model is serving correctly on each.

Invoked by the workflow as:
  pytest ci/k8s/client/test_p2p_k8s.py -v \
      --namespace $NAMESPACE \
      --model $MX_CI_MODEL \
      --source-port $SOURCE_PORT \
      --worker-port $WORKER_PORT \
      [--p2p-marker "framework-specific transfer complete string"]

--p2p-marker defaults:
  vLLM:    "RDMA transfer complete"             (emitted by vLLM's RdmaStrategy)
  TRT-LLM: "ModelExpress P2P transfer complete" (printed by trtllm_p2p_launcher.py)
"""

import json
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


def test_source_inference_produces_output(namespace: str, model: str, source_port: int) -> None:
    """Source server must return a valid completion response."""
    _assert_inference(namespace, "mx-source", model, remote_port=source_port, local_port=18001)


def test_target_inference_produces_output(namespace: str, model: str, worker_port: int) -> None:
    """Target server must return a valid completion response after P2P transfer."""
    _assert_inference(namespace, "mx-target", model, remote_port=worker_port, local_port=18000)
