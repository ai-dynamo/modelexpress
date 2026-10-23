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
  3. AIPerf MMLU scores do not significantly regress from the source baseline
     captured before target deployment. Raw output differences are diagnostic.
  4. When enabled by the workflow, target peak and final VRAM do not materially
     exceed source VRAM.
  5. When configured, source and target report the expected token-pool capacity.
  6. When --expect-mtp is set, every target rank transfers both the main model
     and the MTP draft through P2P, and both workers sustain 128-token decoding.

Invoked by the workflow as:
  pytest ci/k8s/client/test_p2p_k8s.py -v \
      --namespace $NAMESPACE \
      --model $MX_CI_MODEL \
      --source-port $SOURCE_PORT \
      --worker-port $WORKER_PORT \
      --tp-size $TP_SIZE \
      [--dp-size $DP_SIZE] \
      [--p2p-marker "framework-specific transfer complete string"] \
      [--expected-max-total-tokens N] \
      [--expect-mtp]

--p2p-marker defaults:
  vLLM: "RDMA transfer complete" (emitted by vLLM's RdmaStrategy)

--tp-size default is 1 — every existing TP=1 P2P test still asserts exactly
one source agent, which is the correct expectation and adds a free safety net.

--dp-size default is 1 — with a single copy per rank the accepted range
collapses to exactly tp_size, so TP-only jobs are unaffected. Set it to the
target's data-parallel size (e.g. 2 for the DP=2 job) so the range widens to
[tp_size, tp_size * dp_size] for the interchangeable replicas DP publishes.
"""

import json
import re
import time

import pytest

from kube_utils import kubectl, port_forward
from inference_parity import artifact_dir, compare, compare_weight_digests, long_decode, profile, read_responses, replay_mismatch


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
            "--field-selector=status.phase!=Failed",
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
        "--field-selector=status.phase!=Failed",
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
    return "\n".join(_pod_logs_by_pod(namespace, job_name, container).values())


def _pod_logs_by_pod(namespace: str, job_name: str, container: str) -> dict[str, str]:
    """Same fetch as `_all_pod_logs`, keyed by pod name instead of concatenated.

    Assertions about "every pod did X" cannot be expressed on the concatenated
    string: one matching line anywhere satisfies a substring check no matter how
    many pods are missing it.
    """
    pod_list = kubectl(
        "get", "pods",
        "-l", f"job-name={job_name}",
        "--field-selector=status.phase!=Failed",
        "-o", "jsonpath={.items[*].metadata.name}",
        namespace=namespace,
    ).stdout.split()
    out: dict[str, str] = {}
    for pod in pod_list:
        r = kubectl("logs", pod, "-c", container, "--tail=-1", namespace=namespace)
        out[pod] = r.stdout
    return out


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


def _deduplicate_log_handlers(lines: list[str]) -> list[str]:
    """Collapse paired vLLM/MX renderings, even when handlers interleave."""
    vllm_format = re.compile(
        r"^(.*?)INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[([^\]]+:\d+)\] (.*)$"
    )
    mx_format = re.compile(
        r"^(.*?)\[\d{4}-(\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] INFO (\S+:\d+): (.*)$"
    )
    result = []
    unpaired = {}
    for line in lines:
        match = vllm_format.match(line) or mx_format.match(line)
        if match:
            record = (match.groups(), match.re)
            other_format = mx_format if match.re == vllm_format else vllm_format
            counterpart = (match.groups(), other_format)
            # Consume one cross-handler pair; preserve real repeated records.
            if unpaired.get(counterpart, 0):
                unpaired[counterpart] -= 1
                continue
            unpaired[record] = unpaired.get(record, 0) + 1
        result.append(line)
    return result


def test_mtp_load_phases(namespace: str, expect_mtp: bool, tp_size: int) -> None:
    """Every target rank must finish main and draft loads through RDMA."""
    if not expect_mtp:
        pytest.skip("MTP phase assertion not enabled")

    logs = _all_pod_logs(namespace, "mx-target", "mx-target")
    transfer_marker = "RDMA transfer complete"
    start_marker = "MxModelLoader starting"
    complete_marker = "MxModelLoader.load_model() COMPLETE"
    phase_lines = [
        line
        for line in logs.splitlines()
        if any(marker in line for marker in (start_marker, transfer_marker, complete_marker))
    ]
    print("[mx-target] MTP load phases:\n" + "\n".join(phase_lines))
    for rank in range(tp_size):
        rank_logs = "\n".join(_deduplicate_log_handlers([
            line for line in logs.splitlines() if f"[Worker {rank}]" in line
        ]))
        cursor = 0
        for role in ("main", "draft"):
            start = rank_logs.find(start_marker, cursor)
            assert start >= 0, f"Rank {rank}: missing {role} load"
            header = rank_logs[start:].splitlines()[0]
            assert f"p2p_enabled=True, p2p_role={role}" in header, (
                f"Rank {rank}: expected P2P-enabled {role} load, got {header}"
            )
            end = rank_logs.find(complete_marker, start)
            assert end > start, f"Rank {rank}: incomplete {role} load"
            phase = rank_logs[start:end]
            assert phase.count(start_marker) == 1, (
                f"Rank {rank}: another load started before {role} completed"
            )
            assert transfer_marker in phase, f"Rank {rank}: no {role} RDMA transfer"
            assert "Streaming weights from" not in phase, (
                f"Rank {rank}: {role} fell back to ModelStreamer"
            )
            cursor = end + len(complete_marker)


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


EFFECTIVENESS_TIMEOUT_SECS = 180
EFFECTIVENESS_POLL_SECS = 10

# Either outcome ends the wait: the hit line means the cache was reused, the
# warning means it was not. Both are emitted from the same call site, so seeing
# one proves the publisher thread ran and the snapshot is no longer premature.
_EFFECTIVENESS_MARKERS = (
    "which ModelExpress installed",
    "was not reused and the engine recompiled",
)


_COMPILE_INSTALL_MARKER = "vLLM artifact install complete: name=torch_compile_cache"


def _pods_pending_effectiveness(per_pod: dict[str, str]) -> set[str]:
    """Pods that installed a torch.compile cache but have not logged the check.

    The granularity is per pod, not per worker rank: `mark_publish_scheduled` is
    documented as pod-scoped, so in a TP>1 pod exactly one rank schedules the
    publisher and therefore exactly one rank runs the check. Requiring a line per
    rank would fail every multi-rank deployment by construction.
    """
    return {
        pod
        for pod, logs in per_pod.items()
        if _COMPILE_INSTALL_MARKER in logs
        and not any(marker in logs for marker in _EFFECTIVENESS_MARKERS)
    }


def _await_effectiveness_check(namespace: str) -> dict[str, str]:
    """Re-read target logs until every installing pod logged the check.

    Returns the freshest per-pod logs either way; the caller still asserts on
    content, so a timeout surfaces as a named-pod assertion failure rather than a
    bare TimeoutError with no context.
    """
    per_pod = _pod_logs_by_pod(namespace, "mx-target", "mx-target")
    pending = _pods_pending_effectiveness(per_pod)
    if not pending:
        return per_pod
    deadline = time.monotonic() + EFFECTIVENESS_TIMEOUT_SECS
    while time.monotonic() < deadline:
        time.sleep(EFFECTIVENESS_POLL_SECS)
        per_pod = _pod_logs_by_pod(namespace, "mx-target", "mx-target")
        pending = _pods_pending_effectiveness(per_pod)
        if not pending:
            print("[mx-target] every installing pod logged the effectiveness check")
            return per_pod
    print(
        f"[mx-target] {len(pending)} pod(s) still missing the effectiveness check "
        f"after {EFFECTIVENESS_TIMEOUT_SECS}s: {sorted(pending)}"
    )
    return per_pod


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

    # The effectiveness check runs on each target's own publisher thread, gated on
    # /health plus a cache-settle interval. The composite action only waits for the
    # *source* to publish before starting targets, so at this point a target's
    # publisher may not have fired yet. Poll rather than asserting on one snapshot.
    per_pod = _await_effectiveness_check(namespace)
    logs = "\n".join(per_pod.values())

    install_lines = [line for line in logs.splitlines() if "artifact install complete" in line]
    print("[mx-target] artifact install lines:\n" + "\n".join(install_lines))
    assert install_lines, "Target did not log artifact installation"
    for source_type in expected_artifact_source_types:
        assert f"name={source_type}" in logs, (
            f"Target did not log artifact installation for {source_type}"
        )

    # A successful install only means the bytes arrived. The engine still has to
    # pick the installed cache, and it will not if the source was compiled under
    # a different configuration - it silently recompiles instead. Without this
    # assertion the test passes on a deployment that gets no artifact benefit.
    unused_lines = [
        line
        for line in logs.splitlines()
        if "was not reused and the engine recompiled" in line
    ]
    assert not unused_lines, (
        "Target installed a torch.compile cache that vLLM did not use, so it "
        "recompiled. The source and target compile configurations differ; "
        "partition the artifact source pool with "
        "MX_ARTIFACT_COMPILE_CONFIG_DIGEST.\n" + "\n".join(unused_lines)
    )

    # Per pod, not across the concatenation: one matching line anywhere would
    # otherwise satisfy the check no matter how many pods skipped it.
    pending = _pods_pending_effectiveness(per_pod)
    assert not pending, (
        f"{len(pending)} target pod(s) installed a torch.compile cache but never "
        f"logged the effectiveness check: {sorted(pending)}. The engine may not "
        "have reached the publish path, or the check regressed."
    )

    # Every target in this fleet runs the same configuration, so each artifact
    # type must resolve to one identity. Divergence means identity construction
    # is not deterministic across pods - the failure mode that turns an artifact
    # pool into a set of singletons.
    ids_by_type: dict[str, set[str]] = {}
    for line in install_lines:
        fields = dict(
            token.split("=", 1)
            for token in line.split()
            if token.count("=") >= 1 and token.split("=", 1)[0] in {"name", "mx_source_id"}
        )
        if "name" in fields and "mx_source_id" in fields:
            ids_by_type.setdefault(fields["name"], set()).add(fields["mx_source_id"])
    for source_type, source_ids in sorted(ids_by_type.items()):
        assert len(source_ids) == 1, (
            f"Targets resolved {len(source_ids)} distinct mx_source_ids for "
            f"{source_type}: {sorted(source_ids)}. Identically configured "
            "workers must agree on the artifact identity."
        )


def test_source_target_inference_matches(
    namespace: str, model: str, source_port: int, worker_port: int, accuracy_pilot: bool,
    capture_source_baseline: bool,
) -> None:
    """P2P-loaded target must not significantly regress native MMLU accuracy."""
    directory = artifact_dir(namespace)
    source_pod = _pod_name(namespace, "mx-source", ordinal=0)
    if capture_source_baseline:
        with port_forward(namespace, source_pod, local_port=18001, remote_port=source_port) as port:
            profile(f"http://localhost:{port}", model, directory / "source")
        return
    target_pod = _pod_name(namespace, "mx-target", ordinal=0)
    with (
        port_forward(namespace, source_pod, local_port=18001, remote_port=source_port) as source_api,
        port_forward(namespace, target_pod, local_port=18000, remote_port=worker_port) as target_api,
    ):
        source_url = f"http://localhost:{source_api}"
        target_url = f"http://localhost:{target_api}"
        source = (
            read_responses(directory / "source") if (directory / "source").exists()
            else profile(source_url, model, directory / "source")
        )
        target = profile(target_url, model, directory / "target")
        if accuracy_pilot:
            profile(source_url, model, directory / "source-64", samples_per_subject=8)
            repeat = profile(source_url, model, directory / "source-repeat")
            compare(source, repeat, directory / "native-repeat-comparison.json")
        try:
            compare(source, target, directory / "comparison.json")
        except AssertionError:
            cases = sorted(source.keys() & target.keys())
            case = next(
                (case for case in cases if source[case]["passed"] and not target[case]["passed"]),
                next((case for case in cases if source[case] != target[case]), None),
            )
            for side, pod, url, records in (
                ("source", source_pod, source_url, source),
                ("target", target_pod, target_url, target),
            ):
                try:
                    if case is not None:
                        replay_mismatch(url, model, records[case], directory / f"{side}-replays.json")
                    logs = kubectl("logs", pod, namespace=namespace).stdout
                    (directory / f"{side}-worker.log").write_text(logs)
                    info = json.loads(kubectl("get", "pod", pod, "-o", "json", namespace=namespace).stdout)
                    gpu = kubectl("exec", pod, "--", "nvidia-smi", "-L", namespace=namespace, check=False)
                    details = {
                        "case": case, "pod": pod, "node": info["spec"]["nodeName"],
                        "containers": info["status"].get("containerStatuses", []),
                        "args": info["spec"]["containers"][0].get("args", []),
                        "gpu": gpu.stdout, "gpu_error": gpu.stderr,
                    }
                    (directory / f"{side}-worker.json").write_text(json.dumps(details, indent=2) + "\n")
                except Exception as error:
                    (directory / f"{side}-diagnostic-error.txt").write_text(str(error) + "\n")
            raise


def test_finalized_tensor_bytes_match(namespace: str) -> None:
    """Audited workers must agree byte-for-byte independently of dataset scores."""
    pods = [_pod_name(namespace, job, ordinal=0) for job in ("mx-source", "mx-target")]
    info = json.loads(kubectl("get", "pod", *pods, "-o", "json", namespace=namespace).stdout)
    expected = any(
        env["name"] == "MX_CI_WEIGHT_DIGESTS" and env.get("value") == "1"
        for pod in info["items"] for container in pod["spec"]["containers"] for env in container.get("env", [])
    )
    logs = [
        kubectl("logs", pod, namespace=namespace).stdout for pod in pods
    ]
    directory = artifact_dir(namespace)
    directory.mkdir(parents=True, exist_ok=True)
    if not compare_weight_digests(*logs, directory / "weight-comparison.json"):
        assert not expected, "Enabled CI weight audit did not produce tensor digests"
        pytest.skip("Worker image does not provide finalized tensor digests")


def test_mtp_long_decode(
    namespace: str,
    model: str,
    source_port: int,
    worker_port: int,
    expect_mtp: bool,
) -> None:
    """Main/draft execution sustains decoding beyond short MMLU answers."""
    if not expect_mtp:
        pytest.skip("MTP long-decode assertion not enabled")
    for side, job, remote, local in (
        ("source", "mx-source", source_port, 18003),
        ("target", "mx-target", worker_port, 18004),
    ):
        pod = _pod_name(namespace, job, ordinal=0)
        with port_forward(namespace, pod, local_port=local, remote_port=remote) as port:
            long_decode(f"http://localhost:{port}", model, artifact_dir(namespace) / f"{side}-long-decode.json")


def test_configured_token_pool_capacity(
    namespace: str,
    expected_max_total_tokens: int,
) -> None:
    """Source and target must use the configured SGLang token-pool cap."""
    if expected_max_total_tokens == 0:
        pytest.skip("Token-pool capacity assertion not enabled")

    for job_name in ("mx-source", "mx-target"):
        logs = _all_pod_logs(namespace, job_name, job_name)
        capacities = {
            int(value)
            for value in re.findall(r"\bmax_total_num_tokens=(\d+)\b", logs)
        }
        assert capacities == {expected_max_total_tokens}, (
            f"{job_name} reported token-pool capacities {sorted(capacities)}, "
            f"expected only {expected_max_total_tokens}"
        )


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
