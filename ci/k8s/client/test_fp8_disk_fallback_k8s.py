# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP8 disk-fallback regression test: disk fallback after vLLM retry reinit.

The workflow applies a source Job first, waits for it to publish ModelMetadata,
then force-deletes that source before applying the target. The target should:

  1. discover the stale READY source metadata,
  2. prepare itself as an RDMA target,
  3. fail the receive because the source pod is gone,
  4. reinitialize the vLLM model, clearing CompilationConfig state,
  5. fall back to vLLM's native disk loader,
  6. serve inference successfully.

This is the CI shape that guards the vLLM CompilationConfig cleanup
needed when a failed, mutated strategy falls through to disk fallback.
"""

import json
import re
import urllib.request

from kube_utils import kubectl, port_forward


TARGET_JOB_NAME = "mx-target"
TARGET_LOCAL_PORT = 18000


def _target_pod(namespace: str) -> str:
    result = kubectl(
        "get", "pods",
        "-l", f"job-name={TARGET_JOB_NAME}",
        "-o", "jsonpath={.items[0].metadata.name}",
        namespace=namespace,
    )
    pod = result.stdout.strip()
    assert pod, f"{TARGET_JOB_NAME} pod not found"
    return pod


def _target_logs(namespace: str) -> str:
    pod = _target_pod(namespace)
    result = kubectl("logs", pod, "-c", TARGET_JOB_NAME, "--tail=-1", namespace=namespace)
    return result.stdout


def test_target_reinitialized_before_disk_fallback(namespace: str) -> None:
    """Target logs must show the retry reinit path before disk fallback."""
    logs = _target_logs(namespace)

    registration_error = "NIXL registration failed"
    assert registration_error not in logs, (
        "Target swallowed a NIXL registration failure instead of reaching the "
        "dead-peer receive path. Last 80 log lines:\n"
        + "\n".join(logs.splitlines()[-80:])
    )

    ordered_patterns = [
        ("source selection", r"Trying source worker "),
        (
            "LIBFABRIC NIXL agent creation",
            r"NIXL agent '[^'\n]+' created on device \d+ \(backend=LIBFABRIC\)",
        ),
        ("target tensor registration", r"\[TIMING\] register_tensors:"),
        ("receive start", r"Receiving \d+ tensors from source"),
        (
            "receive-stage failure",
            r"Strategy rdma failed, trying next: RDMA receive failed:",
        ),
        ("vLLM reinitialization", r"Re-initializing vLLM model after failed strategy"),
        ("default strategy", r"Trying strategy: default"),
        ("disk load start", r"Loading weights from disk\.\.\."),
        ("disk load completion", r"Weights loaded from disk"),
        ("loader completion", r"MxModelLoader\.load_model\(\) COMPLETE"),
    ]
    cursor = 0
    for label, pattern in ordered_patterns:
        match = re.search(pattern, logs[cursor:])
        assert match is not None, (
            "Target did not exercise the ordered dead-peer receive, "
            "reinit, and disk-fallback path. "
            f"Missing or out-of-order marker: {label}. "
            "Last 80 log lines:\n"
            + "\n".join(logs.splitlines()[-80:])
        )
        cursor += match.end()


def test_target_inference_produces_output(
    namespace: str,
    model: str,
    worker_port: int,
) -> None:
    """Target must serve after the reinitialized native disk load completes."""
    pod = _target_pod(namespace)
    payload = json.dumps({
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": 8,
    }).encode()

    with port_forward(namespace, pod, local_port=TARGET_LOCAL_PORT, remote_port=worker_port) as port:
        req = urllib.request.Request(
            f"http://localhost:{port}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())

    choices = body.get("choices", [])
    assert choices and choices[0].get("text"), f"No completion text in response: {body}"
