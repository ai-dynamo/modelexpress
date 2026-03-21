#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-node DHT test on microk8s.
# Zero-config: peers discover each other via headless Service DNS.
#
# Usage: ./run_k8s_test.sh [--replicas N] [--soak] [--cleanup-only]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KUBECTL="/snap/bin/microk8s kubectl"
NAMESPACE="dht-test"
IMAGE="localhost:32000/dht-test:latest"
REPLICAS=3
TIMEOUT=600
TEST_ROLE="test"

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

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cleanup-only)
            cleanup
            exit 0
            ;;
        --replicas)
            REPLICAS="$2"
            shift 2
            ;;
        --soak)
            TEST_ROLE="soak"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Soak test needs a decent cluster
if [[ "$TEST_ROLE" == "soak" && "$REPLICAS" -lt 20 ]]; then
    log "Soak test requires at least 20 replicas, setting to 200"
    REPLICAS=200
fi

# Step 1: Build and push image
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
log "Building container image..."
docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$REPO_ROOT"
docker push "$IMAGE"

# Step 2: Clean slate
cleanup

# Step 3: Deploy everything (namespace, headless service, peers, test job)
# Patch replica count and MIN_PEERS from YAML defaults to match requested scale.
# For small clusters (<=10), wait for all-1 peers. For large clusters, cap at 25
# because busybox nslookup truncates large DNS responses (UDP packet size limit).
# The Python bootstrap_from_dns handles full discovery via proper DNS resolution.
if [ "$REPLICAS" -le 10 ]; then
    MIN_PEERS=$((REPLICAS - 1))
    [ "$MIN_PEERS" -lt 1 ] && MIN_PEERS=1
else
    MIN_PEERS=25
fi

# For soak test, increase job timeout (rollout restart takes time)
if [[ "$TEST_ROLE" == "soak" ]]; then
    TIMEOUT=1200
fi

log "Deploying DHT cluster ($REPLICAS peer replicas, MIN_PEERS=$MIN_PEERS, role=$TEST_ROLE)..."
sed -e "s/replicas: 3/replicas: $REPLICAS/" \
    -e "/name: MIN_PEERS/{n;s/value: \"2\"/value: \"$MIN_PEERS\"/;}" \
    -e "s/\"--role\", \"test\"/\"--role\", \"$TEST_ROLE\"/" \
    "$SCRIPT_DIR/k8s-dht-test.yaml" | $KUBECTL apply -f -

# Step 4: Wait for deployment rollout
log "Waiting for $REPLICAS peer pods to be ready (timeout ${TIMEOUT}s)..."
$KUBECTL rollout status deployment/dht-peer -n "$NAMESPACE" --timeout="${TIMEOUT}s"

# Step 5: Show cluster state
log "DHT cluster state:"
$KUBECTL get pods -n "$NAMESPACE" -o wide | head -20
TOTAL_PODS=$($KUBECTL get pods -n "$NAMESPACE" --no-headers | wc -l)
log "Total pods: $TOTAL_PODS"

# Show distribution across nodes
log "Pod distribution across nodes:"
$KUBECTL get pods -n "$NAMESPACE" -o custom-columns="NODE:.spec.nodeName" --no-headers | sort | uniq -c | sort -rn

# Step 6: Wait for test job to complete
log "Waiting for test job to complete (timeout ${TIMEOUT}s)..."
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
log "Pod placement (first 20):"
$KUBECTL get pods -n "$NAMESPACE" -o custom-columns="NAME:.metadata.name,NODE:.spec.nodeName,IP:.status.podIP,STATUS:.status.phase" --no-headers | head -20

if [[ "$EXIT_CODE" == "0" ]]; then
    echo ""
    log "RESULT: ALL TESTS PASSED (exit code 0, $REPLICAS peers, role=$TEST_ROLE)"
else
    echo ""
    log "RESULT: TESTS FAILED (exit code $EXIT_CODE, $REPLICAS peers, role=$TEST_ROLE)"
fi

log "Cleaning up..."
cleanup

exit "${EXIT_CODE:-1}"
