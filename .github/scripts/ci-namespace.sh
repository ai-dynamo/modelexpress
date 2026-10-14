#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Create or clean up an mx-ci namespace owned by the current run and attempt.
# Usage: bash .github/scripts/ci-namespace.sh create|cleanup NAMESPACE [plain|dynamo]
# Cleanup preserves finalizers and verifies virtual and available host pod removal.
set -euo pipefail
source .github/scripts/ci-kubectl.sh

operation="$1"
namespace="$2"
owner="${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}"
[[ "$namespace" == mx-ci-* ]] || { echo "Refusing non-CI namespace $namespace" >&2; exit 1; }
read_owned_namespace() {
  info=$(ci_kubectl get namespace "$namespace" --ignore-not-found -o json) || return $?
  if [ -n "$info" ]; then
    actual=$(printf '%s\n' "$info" | jq -r '.metadata.labels["ci.modelexpress.nvidia.com/run-id"] // ""')
    [ "$actual" = "$owner" ] || { echo "Refusing namespace $namespace owned by '$actual', expected '$owner'" >&2; return 1; }
  fi
}
read_owned_namespace

wait_for_host_cleanup() {
  local host_kubeconfig="${GITHUB_WORKSPACE:-.}/.kubeconfig-host"
  [ -f "$host_kubeconfig" ] || return 0
  local deadline=$((SECONDS + 180)) pods count
  while [ "$SECONDS" -lt "$deadline" ]; do
    pods=$(ci_kubectl get pods --kubeconfig="$host_kubeconfig" \
      -n "${VCLUSTER_NAMESPACE:-mx-ci}" -l "vcluster.loft.sh/namespace=$namespace" -o json) || return 1
    count=$(printf '%s\n' "$pods" | jq '.items | length')
    if [ "$count" -eq 0 ]; then return 0; fi
    echo "Waiting for $count host pods from $namespace to be removed"
    sleep 5
  done
  echo "ERROR: host pods for $namespace still hold resources after virtual cleanup" >&2
  return 1
}

if [ "$operation" = create ]; then
  if [ -z "$info" ]; then
    kubectl create namespace "$namespace" --dry-run=client -o json \
      | jq --arg owner "$owner" --arg kind "${3:-plain}" --arg pool "${MX_CI_GPU_POOL:-}" --arg lane "${MX_CI_GPU_LANE:-}" \
          '.metadata.labels = {"ci.modelexpress.nvidia.com/run-id": $owner, "ci.modelexpress.nvidia.com/workload-type": $kind, "ci.modelexpress.nvidia.com/gpu-pool": $pool, "ci.modelexpress.nvidia.com/gpu-lane": $lane}' \
      | kubectl --request-timeout=10s apply -f -
  fi
  exit 0
fi
[ "$operation" = cleanup ] || { echo "Unknown namespace operation: $operation" >&2; exit 1; }
[ -n "$info" ] || { wait_for_host_cleanup; exit 0; }

# Stop workloads before namespace finalization so stuck CR finalizers cannot retain GPUs.
kind=$(printf '%s\n' "$info" | jq -r '.metadata.labels["ci.modelexpress.nvidia.com/workload-type"] // "plain"')
resources=deployment,statefulset,job
if [ "$kind" = dynamo ]; then
  resources=dgd,$resources
fi
for attempt in 1 2 3 4; do
  read_owned_namespace
  [ -n "$info" ] || { wait_for_host_cleanup; exit 0; }
  status=0
  output=$(kubectl --request-timeout=10s delete "$resources" --all -n "$namespace" --ignore-not-found --wait=false 2>&1) || status=$?
  printf '%s\n' "$output"
  if [ "$status" -eq 0 ]; then
    output=$(kubectl --request-timeout=10s delete namespace "$namespace" --ignore-not-found --wait=false 2>&1) || status=$?
    printf '%s\n' "$output"
  fi
  if [ "$status" -eq 0 ]; then break; fi
  if [ "$attempt" -eq 4 ] || ! printf '%s\n' "$output" | grep -qiE 'connection refused|connection reset by peer|unexpected EOF|context deadline exceeded|Client\.Timeout exceeded|i/o timeout|TLS handshake timeout'; then exit "$status"; fi
  sleep 2
done

deadline=$((SECONDS + 600))
while [ "$SECONDS" -lt "$deadline" ]; do
  read_owned_namespace
  if [ -z "$info" ]; then
    wait_for_host_cleanup
    echo "Namespace $namespace and its host pods are gone"
    exit 0
  fi
  sleep 5
done
echo "ERROR: namespace $namespace still exists after 600s of cleanup; finalizers were preserved" >&2
ci_kubectl get namespace "$namespace" -o json | jq '{spec, status}' >&2 || true
ci_kubectl get pods -n "$namespace" -o wide >&2 || true
exit 1
