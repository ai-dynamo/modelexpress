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
       ! grep -qiE 'connection refused|connection reset by peer|unexpected EOF|context deadline exceeded|Client\.Timeout exceeded|i/o timeout|TLS handshake timeout' "$error_file"; then
      rm -f "$error_file"
      return "$status"
    fi
    echo "Retrying interrupted Kubernetes API read (${attempt}/3)" >&2
    sleep 2
  done
}

wait_for_scheduling() {
  local label="$1" workload="$2" selector="$3" timeout="$4"
  local deadline=$((SECONDS + timeout)) controllers expected pods scheduled failed
  while [ "$SECONDS" -lt "$deadline" ]; do
    controllers=$(ci_kubectl get job,statefulset,deployment "$workload" -n "$NAMESPACE" --ignore-not-found -o json) || return 1
    expected=$(printf '%s\n' "$controllers" | jq '(.items // [.]) | map(.spec.replicas // .spec.parallelism // 1) | add // 0')
    pods=$(ci_kubectl get pods -n "$NAMESPACE" -l "$selector" -o json) || return 1
    failed=$(printf '%s\n' "$pods" | jq -r '.items[] | select(.status.phase == "Failed" or any(.status.containerStatuses[]?, .status.initContainerStatuses[]?; .state.waiting.reason == "CrashLoopBackOff" or .state.terminated.reason == "OOMKilled" or .state.terminated.reason == "Error")) | .metadata.name')
    if [ -n "$failed" ]; then
      echo "ERROR: $label pod failed while waiting for scheduling: $failed" >&2
      return 1
    fi
    scheduled=$(printf '%s\n' "$pods" | jq '[.items[] | select(.metadata.deletionTimestamp == null) | select(any(.status.conditions[]?; .type == "PodScheduled" and .status == "True"))] | length')
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
  local lane="${MX_CI_GPU_LANE:-all}" concurrency="${MX_CI_GPU_CONCURRENCY:-4}"
  [[ "$concurrency" == 1 || "$concurrency" == 2 || "$concurrency" == 4 ]] || { echo "MX_CI_GPU_CONCURRENCY must be 1, 2, or 4" >&2; return 1; }
  if [ "$concurrency" = 1 ]; then lane=all; fi
  while [ "$SECONDS" -lt "$deadline" ]; do
    if [[ "$pool" == a100* ]]; then
      namespaces=$(ci_kubectl get namespaces -l "ci.modelexpress.nvidia.com/gpu-pool=$pool" -o json) || return 1
      pods=$(ci_kubectl get pods -A --kubeconfig="$GITHUB_WORKSPACE/.kubeconfig-host" -o json) || return 1
    else
      namespaces=$(ci_kubectl get namespaces --context="$KUBE_CONTEXT" -o json) || return 1
      pods=$(ci_kubectl get pods -A --context="$KUBE_CONTEXT" -o json) || return 1
    fi
    remaining=$(jq -n --arg pool "$pool" --arg lane "$lane" --argjson namespaces "$namespaces" --argjson pods "$pods" '
      def classify($name):
        if ($name | test("^mx-ci-(fleet|rolling)-")) then "tcp"
        elif ($name | test("^mx-ci-h100-vime-")) then "vime"
        elif ($name | test("^mx-ci-bench-")) then "refit"
        elif ($name | test("-mla-")) then "mla"
        elif ($name | test("-(dp2|ep2)-")) then "parallel"
        else "rdma" end;
      def selected($name):
        ([$namespaces.items[] | select(.metadata.name == $name) | .metadata.labels["ci.modelexpress.nvidia.com/gpu-lane"] | select(. != null and length > 0)][0] // classify($name)) as $actual |
        $lane == "all" or $actual == $lane or (($lane | startswith("slot-")) != ($actual | startswith("slot-")));
      def rl($name): $name | startswith("mx-ci-bench-") or startswith("mx-ci-h100-vime-");
      ([$namespaces.items[] | select(selected(.metadata.name)) | select(($pool | startswith("a100")) or rl(.metadata.name))] | length) +
      ([$pods.items[] |
        (.metadata.labels["vcluster.loft.sh/namespace"] // .metadata.namespace // "") as $ns |
        select(selected($ns)) |
        if ($pool | startswith("a100")) then
          select($ns | startswith("mx-ci-")) |
          select(.status.phase != "Succeeded" and .status.phase != "Failed") |
          select(.spec.nodeSelector.agentpool == $pool or ((.spec.nodeName // "") | startswith("aks-" + $pool + "-"))) |
          select(any(.spec.containers[]?, .spec.initContainers[]?; (.resources.requests["nvidia.com/gpu"] // .resources.limits["nvidia.com/gpu"] // "0" | tonumber) > 0))
        else select(rl($ns)) end] | length)') || return 1
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
