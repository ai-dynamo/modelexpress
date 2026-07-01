# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared K8s P2P CI test — used by all inference engine frameworks.

Runs after the workflow has applied the source and target Jobs and both pods
have reached Running state.  Asserts:
  1. Target pod logs contain the framework-specific P2P transfer marker (--p2p-marker).
  2. Target connected to source NIXL agents covering ranks 0..tp_size-1, with a
     distinct-agent count in [tp_size, tp_size * dp_size] (catches TP collapse
     and source-rank-pairing regressions; the range accommodates interchangeable
     DP replicas that share a rank — see --dp-size).
  3. Both source and target servers respond to /v1/completions — confirms weights
     are loaded and the model is serving correctly on each.
  4. When enabled by the workflow, target peak and final VRAM do not materially
     exceed source VRAM.

Invoked by the workflow as:
  pytest ci/k8s/client/test_p2p_k8s.py -v \
      --namespace $NAMESPACE \
      --model $MX_CI_MODEL \
      --source-port $SOURCE_PORT \
      --worker-port $WORKER_PORT \
      --tp-size $TP_SIZE \
      [--dp-size $DP_SIZE] \
      [--p2p-marker "framework-specific transfer complete string"]

--p2p-marker defaults:
  vLLM:    "RDMA transfer complete"             (emitted by vLLM's RdmaStrategy)
  TRT-LLM: "ModelExpress P2P transfer complete" (printed by trtllm_p2p_launcher.py)

--tp-size default is 1 — every existing TP=1 P2P test still asserts exactly
one source agent, which is the correct expectation and adds a free safety net.

--dp-size default is 1 — with a single copy per rank the accepted range
collapses to exactly tp_size, so TP-only jobs are unaffected. Set it to the
target's data-parallel size (e.g. 2 for the DP=2 job) so the range widens to
[tp_size, tp_size * dp_size] for the interchangeable replicas DP publishes.
"""

import json
import re
import urllib.request

import pytest

from kube_utils import kubectl, port_forward


def _pod_name(namespace: str, job_name: str, ordinal: int | None = None) -> str:
    """Resolve the pod name for the given job/StatefulSet label.

    `ordinal=None`: filter by `job-name=<job_name>` and return `.items[0]`.
    Original Job-pod behavior, unchanged.

    `ordinal=N`: prefer pods that have BOTH the `job-name` label and
    `apps.kubernetes.io/pod-index=N` (K8s auto-applies the latter to
    StatefulSet pods). If no match (e.g., a single-pod Job whose pod
    doesn't carry the ordinal label), fall back to the plain
    job-name selector — which returns the only pod that exists. So
    callers can safely pass `ordinal=0` even when they don't know
    whether they're hitting a StatefulSet or a single-pod Job; the
    StatefulSet case is pinned to pod-0, the Job case is a no-op.
    """
    if ordinal is not None:
        # Probe with `-o name` first (returns "pod/<name>" or empty for no
        # match — exits 0 either way, unlike a jsonpath into an empty
        # items array which crashes with "array index out of bounds" and
        # would raise CalledProcessError before we get a chance to check.
        ordinal_selector = f"job-name={job_name},apps.kubernetes.io/pod-index={ordinal}"
        probe = kubectl(
            "get", "pods",
            "-l", ordinal_selector,
            "-o", "name",
            namespace=namespace,
        )
        probed = probe.stdout.strip().removeprefix("pod/")
        if probed:
            return probed
        # No StatefulSet pod with that ordinal — fall through to plain
        # job-name selector (single-pod Job case).
    result = kubectl(
        "get", "pods",
        "-l", f"job-name={job_name}",
        "-o", "jsonpath={.items[0].metadata.name}",
        namespace=namespace,
    )
    name = result.stdout.strip()
    assert name, f"{job_name} pod not found"
    return name


def _all_pod_logs(namespace: str, job_name: str, container: str) -> str:
    """Concat the logs of every pod under the given `job-name` label.

    Multi-pod StatefulSets distribute work across replicas. With
    `--distributed-executor-backend=ray` on vLLM, rank-1's worker process
    lives in pod-1 and Ray doesn't forward worker stdout to the head
    pod's container logs by default. So lines like rank-1's
    `[Worker 1] RDMA transfer complete` or remote-agent load logs only
    appear in pod-1's logs, not pod-0's.

    For single-pod Jobs (existing TP=1 / single-node TP>=2 cases) this
    returns the same content as a single-pod `kubectl logs` would —
    `.items[*]` has exactly one element and the regex/assertions in the
    callers operate on one concatenated string either way.
    """
    pod_list = kubectl(
        "get", "pods",
        "-l", f"job-name={job_name}",
        "-o", "jsonpath={.items[*].metadata.name}",
        namespace=namespace,
    ).stdout.split()
    chunks = []
    for pod in pod_list:
        r = kubectl("logs", pod, "-c", container, "--tail=-1", namespace=namespace)
        chunks.append(r.stdout)
    return "\n".join(chunks)


def _ready_artifact_source_types(namespace: str) -> set[str]:
    result = kubectl(
        "get", "modelmetadata",
        "-o", "json",
        namespace=namespace,
    )
    payload = json.loads(result.stdout)
    source_types = set()
    for item in payload.get("items", []):
        spec = item.get("spec") or {}
        status = item.get("status") or {}
        worker = status.get("worker") or {}
        artifact = worker.get("artifactSource") or {}
        source_type = spec.get("sourceType")
        if (
            source_type
            and source_type != "weights"
            and worker.get("status") == "Ready"
            and artifact.get("artifactId")
        ):
            source_types.add(source_type)
    return source_types


def _assert_inference(namespace: str, job_name: str, model: str, remote_port: int, local_port: int) -> None:
    # Pin to pod-0 explicitly. For multi-node StatefulSets, only the head
    # pod (apps.kubernetes.io/pod-index=0) runs the vLLM HTTP API server;
    # the worker pod has no HTTP listener. For single-pod Jobs, _pod_name
    # falls back to the job-name selector since the ordinal label doesn't
    # exist there — so this is a no-op for the existing single-node case.
    pod = _pod_name(namespace, job_name, ordinal=0)
    print(f"\n[{job_name}] pod={pod} remote_port={remote_port} local_port={local_port}")
    # TODO: replace with a more complex prompt that exercises multi-token reasoning
    # to better validate model correctness beyond a single-word completion.
    payload = json.dumps({
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": 8,
    }).encode()
    with port_forward(namespace, pod, local_port=local_port, remote_port=remote_port) as port:
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
    # Concat logs across every pod under the job/StatefulSet label so the
    # marker is found regardless of which rank/replica emitted it. For
    # single-pod Jobs this is identical to logs of pod-0; for multi-node
    # StatefulSets it captures the rank-1 worker's logs from pod-1, which
    # Ray's distributed executor doesn't forward to the head pod's stdout.
    logs = _all_pod_logs(namespace, "mx-target", "mx-target")
    marker_lines = [line for line in logs.splitlines() if any(k in line for k in ("RDMA", "P2P", "transfer"))]
    print("[mx-target] transfer log lines:\n" + "\n".join(marker_lines))
    assert p2p_marker in logs, (
        f"P2P marker {p2p_marker!r} not found in target logs.\n"
        f"Last 50 log lines:\n" + "\n".join(logs.splitlines()[-50:])
    )


def test_per_rank_source_agents(
    namespace: str, tp_size: int, dp_size: int, transport: str
) -> None:
    """Target's source NIXL agents must cover every rank without mis-pairing.

    NIXL-only: the assertion scans for remote-agent log lines that the shared
    NixlTransferManager emits. The Mooncake TransferEngine path
    (transport=transfer_engine) transfers via batch_transfer_sync_read and
    never registers NIXL agents, so there is nothing to count — skip it there.

    Each source rank publishes exactly one NIXL agent. The shared
    NixlTransferManager.receive_from_source (in modelexpress.nixl_transfer)
    logs either `add_remote_agent: ... (agent=b'<name>')` for central metadata
    or `Using pre-loaded remote agent <name>` for P2P metadata.

    Two orthogonal checks:

      1. source_ranks == range(tp_size). This is the deterministic invariant:
         the observed source ranks must cover exactly 0..tp_size-1, no matter
         which replica each target core selected. It catches TP collapse (a
         target rank never ran → missing rank) and source-rank collapse (the
         rank filter in _find_source_instances regressed so a target rank
         pulled from the wrong rank). Inference alone can't catch the latter:
         shards share a shape per rank, so a wrong-rank load succeeds with
         garbage values and still produces plausible text.

      2. tp_size <= distinct_agents <= tp_size * dp_size. Distinct-agent count
         is NOT deterministic under DP: DP replicas are interchangeable full
         copies at the SAME worker_rank, so a rank can expose up to dp_size
         agents and each target core of that rank picks one at random (default
         `random` selector). The dp_size cores at a rank therefore land on
         1..dp_size distinct replicas → the total is anywhere in
         [tp_size, tp_size * dp_size]. Asserting == tp_size is only correct
         when dp_size == 1 (pure TP, one candidate per rank); it is a coin-flip
         flake under DP > 1. The lower bound still catches full collapse; the
         upper bound catches agents outside the expected rank set.
    """
    if transport != "nixl":
        pytest.skip(
            f"per-rank NIXL agent assertion does not apply to transport={transport!r}"
        )
    # Same multi-pod concat as test_rdma_transfer_logged — under multi-node
    # TP each rank's remote-agent line lives in its own pod's logs.
    logs = _all_pod_logs(namespace, "mx-target", "mx-target")
    matches = []
    for pattern in (
        r"agent=b?'?((?:mx-\w+-worker|trtllm-live-source-rank)(\d+)[-\w]*)'?",
        r"Using pre-loaded remote agent ((?:mx-\w+-worker|trtllm-live-source-rank)(\d+)[-\w]*)",
    ):
        matches.extend(re.findall(pattern, logs))

    distinct_pairs = set(matches)
    distinct_agents = {name for name, _ in distinct_pairs}
    source_ranks = sorted({int(r) for _, r in distinct_pairs})
    print(f"[mx-target] distinct source agents: {distinct_agents}  source_ranks={source_ranks}")

    lo, hi = tp_size, tp_size * dp_size
    assert lo <= len(distinct_agents) <= hi, (
        f"Expected between {lo} and {hi} distinct source agent(s) "
        f"(tp_size={tp_size}, dp_size={dp_size}), got {len(distinct_agents)}: "
        f"{distinct_agents}. Below {lo} means TP collapse (a target rank didn't "
        f"run) or source-rank collapse (a rank's target cores all pulled from a "
        f"single source); above {hi} means agents outside the expected rank set."
    )
    assert source_ranks == list(range(tp_size)), (
        f"Expected source ranks {list(range(tp_size))}, got {source_ranks}."
    )


def test_artifact_transfer(
    namespace: str,
    require_artifact_transfer: bool,
    expected_artifact_sources: int,
    expected_artifact_source_types: set[str],
) -> None:
    if not require_artifact_transfer:
        pytest.skip("artifact transfer assertion not enabled")

    source_types = _ready_artifact_source_types(namespace)
    assert len(source_types) >= expected_artifact_sources, (
        f"Expected at least {expected_artifact_sources} ready artifact source(s), "
        f"got {len(source_types)}: {sorted(source_types)}"
    )
    assert expected_artifact_source_types <= source_types, (
        f"Expected ready artifact source types {sorted(expected_artifact_source_types)}, "
        f"got {sorted(source_types)}"
    )

    logs = _all_pod_logs(namespace, "mx-target", "mx-target")
    install_lines = [line for line in logs.splitlines() if "vLLM artifact install complete" in line]
    print("[mx-target] artifact install lines:\n" + "\n".join(install_lines))
    assert install_lines, "Target did not log vLLM artifact installation"
    for source_type in expected_artifact_source_types:
        assert f"name={source_type}" in logs, (
            f"Target did not log vLLM artifact installation for {source_type}"
        )


def test_source_inference_produces_output(namespace: str, model: str, source_port: int) -> None:
    """Source server must return a valid completion response."""
    _assert_inference(namespace, "mx-source", model, remote_port=source_port, local_port=18001)


def test_target_inference_produces_output(namespace: str, model: str, worker_port: int) -> None:
    """Target server must return a valid completion response after P2P transfer."""
    _assert_inference(namespace, "mx-target", model, remote_port=worker_port, local_port=18000)


def _assert_target_vram_matches_source(
    *,
    metric: str,
    source_vram_mib: int | None,
    target_vram_mib: int | None,
    vram_tolerance_percent: float,
) -> None:
    if source_vram_mib is None and target_vram_mib is None:
        pytest.skip("VRAM measurements were not collected for this P2P job")
    assert source_vram_mib is not None, f"Missing source {metric} VRAM measurement"
    assert target_vram_mib is not None, f"Missing target {metric} VRAM measurement"

    delta_mib = target_vram_mib - source_vram_mib
    allowed_delta_mib = source_vram_mib * vram_tolerance_percent / 100.0
    print(
        f"{metric.title()} VRAM: "
        f"source={source_vram_mib} MiB "
        f"target={target_vram_mib} MiB "
        f"delta={delta_mib} MiB "
        f"allowed_delta={allowed_delta_mib:.1f} MiB "
        f"tolerance={vram_tolerance_percent}%"
    )
    assert delta_mib <= allowed_delta_mib, (
        f"Target RDMA path used too much extra {metric} VRAM: "
        f"source={source_vram_mib} MiB, "
        f"target={target_vram_mib} MiB, "
        f"delta={delta_mib} MiB, "
        f"allowed_delta={allowed_delta_mib:.1f} MiB "
        f"({vram_tolerance_percent}% of source {metric} VRAM)."
    )


def test_target_peak_vram_matches_source(
    source_peak_vram_mib: int | None,
    target_peak_vram_mib: int | None,
    vram_tolerance_percent: float,
) -> None:
    """Target RDMA load must not use materially more peak VRAM than source load."""
    _assert_target_vram_matches_source(
        metric="peak",
        source_vram_mib=source_peak_vram_mib,
        target_vram_mib=target_peak_vram_mib,
        vram_tolerance_percent=vram_tolerance_percent,
    )


def test_target_final_vram_matches_source(
    source_final_vram_mib: int | None,
    target_final_vram_mib: int | None,
    vram_tolerance_percent: float,
) -> None:
    """Target RDMA load must not use materially more final VRAM than source load."""
    _assert_target_vram_matches_source(
        metric="final",
        source_vram_mib=source_final_vram_mib,
        target_vram_mib=target_final_vram_mib,
        vram_tolerance_percent=vram_tolerance_percent,
    )
