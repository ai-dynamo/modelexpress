#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# DHT metadata backend integration test on microk8s.
#
# Usage: ./run_k8s_metadata_test.sh [--cleanup-only]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KUBECTL="/snap/bin/microk8s kubectl"
NAMESPACE="dht-metadata-test"
IMAGE="localhost:32000/dht-metadata-test:latest"
TIMEOUT=300

log() { echo "[$(date +%H:%M:%S)] $*"; }

cleanup() {
    log "Cleaning up namespace $NAMESPACE..."
    $KUBECTL delete namespace "$NAMESPACE" --ignore-not-found --wait=false 2>/dev/null || true
    for i in $(seq 1 60); do
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

# Step 3: Deploy
log "Deploying metadata test cluster (3 peers + 1 publisher + 1 test job)..."
$KUBECTL apply -f "$SCRIPT_DIR/k8s-metadata-test.yaml"

# Step 4: Wait for peer deployment
log "Waiting for peer pods..."
$KUBECTL rollout status deployment/dht-peer -n "$NAMESPACE" --timeout="${TIMEOUT}s"

# Step 5: Wait for publisher
log "Waiting for publisher pod..."
$KUBECTL rollout status deployment/metadata-publisher -n "$NAMESPACE" --timeout="${TIMEOUT}s"

# Step 6: Show cluster state
log "Cluster state:"
$KUBECTL get pods -n "$NAMESPACE" -o wide

# Step 7: Wait for test job
log "Waiting for test job to complete (timeout ${TIMEOUT}s)..."
$KUBECTL wait --for=condition=Complete job/metadata-test -n "$NAMESPACE" --timeout="${TIMEOUT}s" 2>/dev/null || true

# Get results
TEST_POD=$($KUBECTL get pods -n "$NAMESPACE" -l job-name=metadata-test -o jsonpath='{.items[0].metadata.name}')
EXIT_CODE=$($KUBECTL get pod "$TEST_POD" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}' 2>/dev/null || echo "unknown")

echo ""
echo "=========================================="
echo "TEST LOGS"
echo "=========================================="
$KUBECTL logs "$TEST_POD" -n "$NAMESPACE" -c test
echo "=========================================="

echo ""
log "Pod placement:"
$KUBECTL get pods -n "$NAMESPACE" -o custom-columns="NAME:.metadata.name,NODE:.spec.nodeName,IP:.status.podIP,STATUS:.status.phase" --no-headers

if [[ "$EXIT_CODE" == "0" ]]; then
    echo ""
    log "RESULT: ALL TESTS PASSED"
else
    echo ""
    log "RESULT: TESTS FAILED (exit code $EXIT_CODE)"
fi

log "Cleaning up..."
cleanup

exit "${EXIT_CODE:-1}"
