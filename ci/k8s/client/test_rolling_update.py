# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fleet scale and rolling-update test: CRD backend, NIXL-over-TCP P2P.

run-mx-fleet-test has already:
  1. Deployed mx-server with the Kubernetes CRD backend.
  2. Applied the fleet workers Deployment at replicas=1.
  3. Waited for the source worker to reach Ready status.
  4. Scaled in waves to fleet_size total replicas.
  5. Waited for all fleet_size workers to reach Ready status.
  6. Rolled the Deployment with maxUnavailable=1 and maxSurge=0.
  7. Waited for all replacement workers to reach Ready and all original
     worker generations to become Stale.

Asserts:
  1. Exactly fleet_size weight-source CRs are Ready and every weight CR that
     was Ready before the rollout is now Stale.
  2. Every current replacement pod logged "RDMA transfer complete" and none
     logged "Weights loaded from disk". A replacement therefore found a live
     source even while terminated generations remained in metadata.

Inference correctness from P2P-loaded weights is intentionally NOT checked
here — it is covered thoroughly by the framework P2P tests.
A single request through the fleet Service would hit only one pod (and
non-deterministically), adding no scale-specific signal.

Invoked by the workflow as:
  pytest ci/k8s/client/test_rolling_update.py -v \\
      --namespace $NAMESPACE \\
      --model $MX_CI_MODEL \\
      --expected-cr-count $FLEET_SIZE
"""

import json
import os

from kube_utils import kubectl


def test_fleet_generations_after_rolling_update(
    namespace: str, expected_cr_count: int,
) -> None:
    """Only replacement generations are Ready; replaced generations are Stale."""
    result = kubectl("get", "modelmetadata", "-o", "json", namespace=namespace)
    items = json.loads(result.stdout).get("items", [])
    weight_rows = []
    for item in items:
        source_type = (item.get("spec") or {}).get("sourceType")
        if source_type not in (None, "", "weights"):
            continue
        name = (item.get("metadata") or {}).get("name", "<unnamed>")
        status = ((item.get("status") or {}).get("worker") or {}).get("status")
        weight_rows.append((name, status))

    print(f"[modelmetadata] {len(weight_rows)} weight CR(s):")
    for name, status in weight_rows:
        print(f"  {name} {status}")

    ready = [name for name, status in weight_rows if status == "Ready"]
    status_by_name = dict(weight_rows)
    pre_roll_ready = os.environ.get("PRE_ROLL_READY_CRS", "").splitlines()
    assert len(pre_roll_ready) == expected_cr_count, (
        f"Expected {expected_cr_count} captured pre-roll Ready CR names, "
        f"got {len(pre_roll_ready)}: {pre_roll_ready}"
    )
    not_stale = {
        name: status_by_name.get(name, "Missing")
        for name in pre_roll_ready
        if status_by_name.get(name) != "Stale"
    }
    unexpected = [
        f"{name}={status}" for name, status in weight_rows
        if status not in ("Ready", "Stale")
    ]
    assert len(ready) == expected_cr_count, (
        f"Expected exactly {expected_cr_count} live Ready generations after rollout, "
        f"got {len(ready)}."
    )
    assert not not_stale, (
        "Pre-roll Ready generations did not transition to Stale: "
        + ", ".join(f"{name}={status}" for name, status in not_stale.items())
    )
    assert not unexpected, "Unexpected worker generation states: " + ", ".join(unexpected)


def test_rolling_replacements_pair_with_live_sources(
    namespace: str, expected_cr_count: int, p2p_marker: str,
) -> None:
    """Every live replacement must complete P2P without disk fallback."""
    pods = _worker_pod_names(namespace)
    assert len(pods) == expected_cr_count, (
        f"Expected {expected_cr_count} live replacement pods, got {len(pods)}: {pods}"
    )

    missing_p2p = []
    disk_fallbacks = []
    for pod in pods:
        logs = kubectl("logs", pod, "--tail=-1", namespace=namespace).stdout
        if p2p_marker not in logs:
            missing_p2p.append(pod)
        if "Weights loaded from disk" in logs:
            disk_fallbacks.append(pod)

    assert not missing_p2p, (
        f"Replacement pods missing P2P marker {p2p_marker!r}: {missing_p2p}"
    )
    assert not disk_fallbacks, (
        "Replacement pods fell back to disk instead of pairing with a live source: "
        + ", ".join(disk_fallbacks)
    )


def _worker_pod_names(namespace: str) -> list[str]:
    result = kubectl(
        "get", "pods",
        "-l", "app=mx-fleet-worker",
        "-o", "jsonpath={.items[*].metadata.name}",
        namespace=namespace,
    )
    names = result.stdout.split()
    assert names, f"No worker pods found in {namespace} with label app=mx-fleet-worker"
    return names
