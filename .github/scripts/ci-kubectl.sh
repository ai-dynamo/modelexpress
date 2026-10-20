#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared CI helpers for bounded Kubernetes reads, scheduling, and worker liveness.
# Source from the repository root; readiness helpers use NAMESPACE.

ci_kubectl() {
  local attempt output status error_file
  error_file=$(mktemp)
  for attempt in 1 2 3 4; do
    status=0
    output=$(kubectl --request-timeout=10s "$@" 2>"$error_file") || status=$?
    cat "$error_file" >&2
    if [ "$status" -eq 0 ]; then
      printf '%s\n' "$output"
      rm -f "$error_file"
      return 0
    fi
    if [ "$1" != get ] || [ "$attempt" -eq 4 ] || \
       ! grep -qiE 'connection refused|connection reset by peer|unexpected EOF|context deadline exceeded|Client\.Timeout|context cancellation while reading body|i/o timeout|TLS handshake timeout' "$error_file"; then
      rm -f "$error_file"
      return "$status"
    fi
    echo "Retrying interrupted Kubernetes API read (${attempt}/3)" >&2
    sleep 2
  done
}

ci_workload_pods() {
  local label="$1" workload="$2" selector="$3" pods failed controllers
  pods=$(ci_kubectl get pods -n "$NAMESPACE" -l "$selector" -o json) || return 1
  pods=$(printf '%s\n' "$pods" | jq '
      def transient_rdma_admission:
        .status.phase == "Failed" and
        .status.reason == "UnexpectedAdmissionError" and
        ((.status.message // "") | test("no healthy devices present.*rdma/ib"; "i"));
      {failed: [.items[] | select(
        (.status.phase == "Failed" and (transient_rdma_admission | not)) or
        any(.status.containerStatuses[]?, .status.initContainerStatuses[]?;
          ((.state.waiting.reason // .state.terminated.reason // "") |
            test("Error|CrashLoopBackOff|OOMKilled")))
      ) | .metadata.name],
       retryable: any(.items[]; transient_rdma_admission),
       items: [.items[] | select(.status.phase != "Failed" and .metadata.deletionTimestamp == null)]}') || return 1
  failed=$(printf '%s\n' "$pods" | jq -r '.failed[]')
  if [ -n "$failed" ]; then
    echo "ERROR: $label pod failed: $failed" >&2
    return 1
  fi
  if [ "$(printf '%s\n' "$pods" | jq '.retryable')" = true ]; then
    controllers=$(ci_kubectl get job,statefulset,deployment "$workload" -n "$NAMESPACE" --ignore-not-found -o json) || return 1
    if printf '%s\n' "$controllers" | jq -e '(.items // [.]) | any(.[];
      (.kind == "Job" and (.status.failed // 0) > (.spec.backoffLimit // 0)) or
      any(.status.conditions[]?; (.type == "Failed" or .type == "FailureTarget") and .status == "True"))' >/dev/null; then
      echo "ERROR: $label workload exhausted its RDMA admission retry" >&2
      return 1
    fi
  fi
  printf '%s\n' "$pods" | jq '{items}'
}

wait_for_scheduling() {
  local label="$1" workload="$2" selector="$3" timeout="$4"
  local deadline=$((SECONDS + timeout)) controllers expected pods scheduled
  while [ "$SECONDS" -lt "$deadline" ]; do
    controllers=$(ci_kubectl get job,statefulset,deployment "$workload" -n "$NAMESPACE" --ignore-not-found -o json) || return 1
    expected=$(printf '%s\n' "$controllers" | jq '(.items // [.]) | map(.spec.replicas // .spec.parallelism // 1) | add // 0')
    pods=$(ci_workload_pods "$label" "$workload" "$selector") || return 1
    scheduled=$(printf '%s\n' "$pods" | jq '[.items[] | select(any(.status.conditions[]?; .type == "PodScheduled" and .status == "True"))] | length')
    if [ "$expected" -gt 0 ] && [ "$scheduled" -ge "$expected" ]; then
      echo "$label scheduled (${scheduled}/${expected} pods); starting the startup deadline"
      return 0
    fi
    echo "$label waiting for scheduling (${scheduled}/${expected} pods)"
    sleep 5
  done
  echo "ERROR: $label did not schedule within ${timeout}s" >&2
  ci_kubectl get pods -n "$NAMESPACE" -l "$selector" -o wide || true
  ci_kubectl get events -n "$NAMESPACE" --sort-by=.lastTimestamp | tail -20 || true
  return 1
}

wait_for_worker_liveness() {
  local selector="$1" expected="$2" timeout="$3"
  local deadline=$((SECONDS + timeout)) pods names pod ready count
  while [ "$SECONDS" -lt "$deadline" ]; do
    pods=$(ci_kubectl get pods -n "$NAMESPACE" -l "$selector" -o json) || return 1
    names=$(printf '%s\n' "$pods" | jq -r '.items[] | select(.metadata.deletionTimestamp == null) | .metadata.name')
    count=$(printf '%s\n' "$pods" | jq '[.items[] | select(.metadata.deletionTimestamp == null)] | length')
    ready=0
    for pod in $names; do
      if kubectl --request-timeout=10s exec -n "$NAMESPACE" "$pod" -- curl -fsS -m 5 http://localhost:9090/live >/dev/null; then
        ready=$((ready + 1))
      fi
    done
    if [ "$count" -eq "$expected" ] && [ "$ready" -eq "$expected" ]; then
      echo "All $expected active workers /live OK"
      return 0
    fi
    echo "Worker liveness: $ready/$expected ready ($count active pods); retrying in 10s"
    sleep 10
  done
  echo "ERROR: workers did not become live within ${timeout}s" >&2
  return 1
}

# GitHub releases a cancelled job's queue slot before independent cleanup finishes.
# Wait for the previous CI namespaces and physical pods before allocating a new workload.
wait_for_gpu_pool_cleanup() {
  local pool="$1" deadline=$((SECONDS + 900)) namespaces pods remaining
  local lane="${MX_CI_GPU_LANE:-slot-0}"
  local slot_selector="ci.modelexpress.nvidia.com/gpu-lane=$lane"
  # vCluster mirrors namespace labels using SHA256(label key)[:10].
  local host_lane="vcluster.loft.sh/ns-label-${VCLUSTER_NAME:-mx-ci-cluster}-x-9c8bc18f28"
  local host_pool="vcluster.loft.sh/ns-label-${VCLUSTER_NAME:-mx-ci-cluster}-x-dc1dfe5427"
  while [ "$SECONDS" -lt "$deadline" ]; do
    pods='{"items":[]}'
    if [[ "$pool" == a100* ]]; then
      namespaces=$(ci_kubectl get namespaces -l "$slot_selector,ci.modelexpress.nvidia.com/gpu-pool=$pool" -o json) || return 1
      pods=$(ci_kubectl get pods --kubeconfig="$GITHUB_WORKSPACE/.kubeconfig-host" \
        -n "${VCLUSTER_NAMESPACE:-mx-ci}" -l "$host_lane=$lane,$host_pool=$pool" -o json) || return 1
    else
      namespaces=$(ci_kubectl get namespaces --context="$KUBE_CONTEXT" -l "$slot_selector" -o json) || return 1
    fi
    remaining=$(printf '%s\n%s\n' "$namespaces" "$pods" | jq -s \
      --arg pool "$pool" --arg lane "$lane" --arg host_lane "$host_lane" --arg host_pool "$host_pool" '
      ([.[0].items[] |
        select(.metadata.name | startswith("mx-ci-")) |
        select(.metadata.labels["ci.modelexpress.nvidia.com/gpu-lane"] == $lane) |
        select(($pool | startswith("a100") | not) or .metadata.labels["ci.modelexpress.nvidia.com/gpu-pool"] == $pool)] | length) +
      ([.[1].items[] |
        select(.metadata.labels[$host_lane] == $lane and .metadata.labels[$host_pool] == $pool)] | length)') || return 1
    if [ "$remaining" -eq 0 ]; then
      echo "GPU pool $pool lane $lane has no previous CI workload awaiting cleanup"
      return 0
    fi
    echo "Waiting for GPU pool $pool lane $lane cleanup ($remaining namespaces/pods remain)"
    sleep 10
  done
  echo "ERROR: previous workload still occupies GPU pool $pool lane $lane after 900s" >&2
  return 1
}
