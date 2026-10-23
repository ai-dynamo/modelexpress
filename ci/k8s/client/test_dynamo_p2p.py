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
  1. Exactly 2 weight-source ModelMetadata CRs exist in the test namespace, covering source
     ranks 0 and 1 (matches our existing P2P test plumbing — the worker_rank
     filter is what gates rank pairing in rdma_strategy.py).
  2. The second worker actually pulled weights via NIXL RDMA (not via the
     HF disk-fallback path). Detected by scanning all worker pod logs for a
     remote source agent loaded through either central or P2P metadata.
  3. AIPerf scores from the source captured before scaling do not significantly
     regress on the explicitly routed target. Raw response metadata proves
     the intended decode worker served every request.

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

from kube_utils import completion_response, kubectl, port_forward
from inference_parity import artifact_dir, compare, profile, read_responses


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
    emits a remote-agent line; we don't care which ordinal it is."""
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
        "-o", "jsonpath={range .items[*]}{.metadata.name}{\" \"}{.spec.worker_rank}{\"|\"}{.spec.sourceType}{\"\\n\"}{end}",
        namespace=namespace,
    )
    rows = [
        line.rsplit("|", 1)[0] for line in result.stdout.splitlines()
        if line.strip() and line.rsplit("|", 1)[-1] in ("", "weights")
    ]
    print(f"[modelmetadata] {len(rows)} CR(s):")
    for row in rows:
        print(f"  {row}")
    assert len(rows) == expected_cr_count, (
        f"Expected {expected_cr_count} ModelMetadata CRs, got {len(rows)}."
    )


def test_second_replica_used_rdma(namespace: str, dgd_name: str) -> None:
    """At least one worker pod must have loaded a remote NIXL agent.

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
    # Central metadata calls add_remote_agent; P2P metadata preloads the same
    # agent while fetching metadata directly from the source worker.
    matches = []
    for pattern in (
        r"agent=b?'?((?:mx-\w+-worker|trtllm-live-source-rank)(\d+)[-\w]*)'?",
        r"Using pre-loaded remote agent ((?:mx-\w+-worker|trtllm-live-source-rank)(\d+)[-\w]*)",
    ):
        matches.extend(re.findall(pattern, logs))
    distinct_agents = {name for name, _ in set(matches)}
    print(f"[workers] distinct source agents observed across all worker logs: {distinct_agents}")
    assert len(distinct_agents) >= 1, (
        f"No remote source agents found across worker logs — every worker that "
        f"needed weights fell back to HF disk load. P2P didn't engage at all."
    )


def _probe_worker(port: int, model: str, worker_id: int | None = None) -> int:
    nvext = {"extra_fields": ["worker_id"]}
    if worker_id is not None:
        nvext.update(backend_instance_id=worker_id, decode_worker_id=worker_id)
    payload = json.dumps({
        "model": model, "messages": [{"role": "user", "content": "Reply with A."}],
        "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
        "nvext": nvext,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    body = completion_response(req)
    actual = body["nvext"]["worker_id"]["decode_worker_id"]
    assert isinstance(actual, int), f"Missing serving worker ID: {body}"
    assert worker_id is None or actual == worker_id, f"Routed to worker {actual}, expected {worker_id}"
    return actual


def _wait_for_worker_route(port: int, model: str, worker_id: int) -> None:
    for attempt in range(90):
        try:
            _probe_worker(port, model, worker_id)
            return
        except urllib.error.HTTPError as error:
            message = error.read().decode()
            if error.code != 400 or f"nvext.decode_worker_id={worker_id} does not identify a known worker" not in message:
                raise
            assert attempt < 89, f"Worker {worker_id} was discovered but never became routable: {message}"
            time.sleep(2)


def _discovered_worker_ids(port: int) -> set[int]:
    request = urllib.request.Request(f"http://localhost:{port}/health")
    body = completion_response(request)
    return {
        instance["instance_id"] for instance in body["instances"]
        if instance["endpoint"] == "generate"
    }


def test_frontend_inference(
    namespace: str, model: str, dgd_name: str, capture_source_baseline: bool,
) -> None:
    """Capture the source before scale-up, then explicitly route to each target."""
    directory = artifact_dir(namespace)
    frontend_svc = f"{dgd_name}-frontend"
    with port_forward(namespace, f"svc/{frontend_svc}", local_port=18000, remote_port=FRONTEND_PORT) as port:
        url = f"http://localhost:{port}"
        if capture_source_baseline:
            source_id = _probe_worker(port, model)
            profile(url, model, directory / "source", worker_id=source_id)
            initial_ids = _discovered_worker_ids(port)
            assert source_id in initial_ids, "Serving source missing from frontend discovery"
            (directory / "source-worker.json").write_text(json.dumps(source_id) + "\n")
            (directory / "initial-workers.json").write_text(json.dumps(sorted(initial_ids)) + "\n")
            return

        source_id = json.loads((directory / "source-worker.json").read_text())
        source = read_responses(directory / "source", worker_id=source_id)
        initial_ids = set(json.loads((directory / "initial-workers.json").read_text()))
        for _ in range(20):
            serving_ids = _discovered_worker_ids(port)
            if initial_ids <= serving_ids and len(serving_ids - initial_ids) == 1:
                break
            time.sleep(2)
        assert initial_ids <= serving_ids and len(serving_ids - initial_ids) == 1, (
            f"Expected original workers {initial_ids} and one target, observed {serving_ids}"
        )
        (directory / "serving-workers.json").write_text(json.dumps(sorted(serving_ids)) + "\n")
        for target_id in sorted(serving_ids - initial_ids):
            _wait_for_worker_route(port, model, target_id)
            target = profile(url, model, directory / f"target-{target_id}", worker_id=target_id)
            compare(source, target, directory / f"comparison-{target_id}.json")
