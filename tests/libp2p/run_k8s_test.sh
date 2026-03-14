#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-node DHT test on microk8s.
# Zero-config: peers discover each other via headless Service DNS.
#
# Usage: ./run_k8s_test.sh [--cleanup-only]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KUBECTL="/snap/bin/microk8s kubectl"
NAMESPACE="dht-test"
IMAGE="localhost:32000/dht-test:latest"
TIMEOUT=180

log() { echo "[$(date +%H:%M:%S)] $*"; }

cleanup() {
    log "Cleaning up namespace $NAMESPACE..."
    $KUBECTL delete namespace "$NAMESPACE" --ignore-not-found --wait=false 2>/dev/null || true
    for i in $(seq 1 30); do
        if ! $KUBECTL get namespace "$NAMESPACE" &>/dev/null; then
            break
        fi
        sleep 1
    done
}

if [[ "${1:-}" == "--cleanup-only" ]]; then
    cleanup
    exit 0
fi

# Step 1: Build and push image
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
log "Building container image..."
docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$REPO_ROOT"
docker push "$IMAGE"

# Step 2: Clean slate
cleanup

# Step 3: Deploy everything (namespace, headless service, peers, test job)
log "Deploying DHT cluster..."
$KUBECTL apply -f "$SCRIPT_DIR/k8s-dht-test.yaml"

# Step 4: Wait for peers
log "Waiting for peer pods to be ready..."
$KUBECTL rollout status statefulset/dht-peer -n "$NAMESPACE" --timeout="${TIMEOUT}s"

# Step 5: Show cluster state
log "DHT cluster state:"
$KUBECTL get pods -n "$NAMESPACE" -o wide

# Step 6: Wait for test job to complete
log "Waiting for test job to complete..."
$KUBECTL wait --for=condition=Complete job/dht-test -n "$NAMESPACE" --timeout="${TIMEOUT}s" 2>/dev/null || true

# Get test pod name and exit code
TEST_POD=$($KUBECTL get pods -n "$NAMESPACE" -l job-name=dht-test -o jsonpath='{.items[0].metadata.name}')
EXIT_CODE=$($KUBECTL get pod "$TEST_POD" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}' 2>/dev/null || echo "unknown")

echo ""
echo "=========================================="
echo "TEST LOGS"
echo "=========================================="
$KUBECTL logs "$TEST_POD" -n "$NAMESPACE"
echo "=========================================="

echo ""
log "Pod placement:"
$KUBECTL get pods -n "$NAMESPACE" -o custom-columns="NAME:.metadata.name,NODE:.spec.nodeName,IP:.status.podIP,STATUS:.status.phase" --no-headers

if [[ "$EXIT_CODE" == "0" ]]; then
    echo ""
    log "RESULT: ALL TESTS PASSED (exit code 0)"
else
    echo ""
    log "RESULT: TESTS FAILED (exit code $EXIT_CODE)"
fi

log "Cleaning up..."
cleanup

exit "${EXIT_CODE:-1}"
