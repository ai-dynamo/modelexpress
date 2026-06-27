# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fleet scale test (row 20): ~15 workers, CRD backend, NIXL-over-TCP P2P.

run-mx-fleet-test has already:
  1. Deployed mx-server with the Kubernetes CRD backend.
  2. Applied the fleet workers Deployment at replicas=1.
  3. Waited for the source worker to publish 1 ModelMetadata CR.
  4. Scaled in waves to fleet_size total replicas.
  5. Waited for all fleet_size ModelMetadata CRs to appear.

Asserts:
  1. At least fleet_size ModelMetadata CRs exist and every one is
     status.worker.status == "Ready" — confirms the fleet scaled and the
     CRD backend is tracking all live workers (the reaper has not evicted
     any during scale-up).
  2. At least fleet_size - 1 worker pods logged "RDMA transfer complete"
     — the one source downloads from HF and does not log this; every
     other worker must have pulled via NIXL (TCP transport on a100a MIG).

Inference correctness from P2P-loaded weights is intentionally NOT checked
here — it is covered thoroughly by test_p2p_k8s.py and test_dynamo_p2p.py.
A single request through the fleet Service would hit only one pod (and
non-deterministically), adding no scale-specific signal.

Invoked by the workflow as:
  pytest ci/k8s/client/test_fleet_scale.py -v \\
      --namespace $NAMESPACE \\
      --model $MX_CI_MODEL \\
      --expected-cr-count $FLEET_SIZE
"""

import subprocess

from kube_utils import kubectl


def test_fleet_crs_published_and_ready(namespace: str, expected_cr_count: int) -> None:
    """At least fleet_size ModelMetadata CRs must exist, all of them Ready.

    Two conditions in one check because they describe the same healthy state
    and neither is meaningful alone:
      - Count >= expected_cr_count (not ==) tolerates a Deployment briefly
        running a pod or two above the desired replica count (surge pods);
        those still publish valid CRs. The real failure is too FEW CRs — the
        reaper evicted a live worker, or some workers never published.
      - Every CR's status.worker.status == Ready confirms the CRD backend is
        tracking all live workers; a non-Ready CR (Unknown or Stale) means a
        worker crashed after publishing or the reaper wrongly evicted it.
    A pure Ready check without the count floor would pass green on a fleet
    that never scaled (e.g. 3 Ready CRs), so the count floor is load-bearing.
    """
    result = kubectl(
        "get", "modelmetadata",
        "-o", "jsonpath={range .items[*]}{.metadata.name}{\" \"}{.status.worker.status}{\"\\n\"}{end}",
        namespace=namespace,
    )
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    print(f"[modelmetadata] {len(rows)} CR(s):")
    for row in rows:
        print(f"  {row}")
    assert len(rows) >= expected_cr_count, (
        f"Expected at least {expected_cr_count} ModelMetadata CRs, got {len(rows)}."
    )
    not_ready = [r for r in rows if not r.endswith(" Ready")]
    assert not not_ready, (
        f"{len(not_ready)}/{len(rows)} CRs are not Ready: " + ", ".join(not_ready)
    )


def test_fleet_p2p_engagement(namespace: str, expected_cr_count: int) -> None:
    """At least fleet_size - 1 workers must have logged P2P transfer.

    The source worker downloads from HF and never logs the marker; every
    other worker must pull via NIXL (TCP transport). Counting occurrences
    of "RDMA transfer complete" across all pod logs — each receiving worker
    logs it exactly once — gives the number of successful P2P transfers.
    A count below fleet_size - 1 means at least one non-source worker fell
    back to the HF disk path, which is the failure mode this test guards.
    """
    all_logs = ""
    for pod in _worker_pod_names(namespace):
        try:
            r = subprocess.run(
                ["kubectl", "-n", namespace, "logs", pod, "--tail=-1"],
                capture_output=True, text=True, check=False,
            )
            all_logs += r.stdout
        except Exception as exc:
            print(f"WARN: failed to fetch logs from {pod}: {exc}")

    marker = "RDMA transfer complete"
    transfer_count = all_logs.count(marker)
    min_expected = expected_cr_count - 1
    print(f"[fleet] '{marker}' occurrences across all worker logs: {transfer_count}")
    assert transfer_count >= min_expected, (
        f"Expected at least {min_expected} P2P transfers (fleet_size - 1), "
        f"got {transfer_count}. At least one non-source worker fell back to "
        f"HF disk load."
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
