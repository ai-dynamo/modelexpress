#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Kubernetes test script for gRPC file transfer without shared storage
# This script tests the --no-shared-storage feature on Kubernetes using a kind cluster.
#
# Modes:
#   - cli: Test using CLI flags (--no-shared-storage --transfer-chunk-size)
#   - env: Test using environment variables (MODEL_EXPRESS_NO_SHARED_STORAGE, etc.)
#   - all: Run both tests (default)

set -e

# Configuration
RELEASE_NAME="modelexpress-test"
NAMESPACE="modelexpress-test"
IMAGE_NAME="modelexpress"
IMAGE_TAG="test"
TEST_MODEL="google-t5/t5-small"
TIMEOUT_SECONDS=600
CLEANUP=true
TEST_MODE="all"  # "cli", "env", or "all"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cleanup)
            CLEANUP=false
            shift
            ;;
        --model)
            TEST_MODEL="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        --mode)
            TEST_MODE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-cleanup    Don't cleanup resources after test"
            echo "  --model NAME    Model to use for testing (default: google-t5/t5-small)"
            echo "  --timeout SECS  Timeout in seconds (default: 600)"
            echo "  --mode MODE     Test mode: 'cli', 'env', or 'all' (default: all)"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup() {
    if [ "$CLEANUP" = true ]; then
        log_info "Cleaning up Kubernetes resources..."
        kubectl delete job grpc-transfer-test-cli -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
        kubectl delete job grpc-transfer-test-env -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
        helm uninstall "$RELEASE_NAME" -n "$NAMESPACE" 2>/dev/null || true
        kubectl delete namespace "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
        log_info "Cleanup complete"
    else
        log_warning "Skipping cleanup (--no-cleanup specified)"
        log_info "To cleanup manually, run:"
        echo "  kubectl delete job grpc-transfer-test-cli grpc-transfer-test-env -n $NAMESPACE"
        echo "  helm uninstall $RELEASE_NAME -n $NAMESPACE"
        echo "  kubectl delete namespace $NAMESPACE"
    fi
}

trap cleanup EXIT

check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi

    if ! command -v helm &> /dev/null; then
        log_error "helm not found. Please install helm."
        exit 1
    fi

    if ! command -v docker &> /dev/null; then
        log_error "docker not found. Please install docker."
        exit 1
    fi

    # Install kind if not available
    if ! command -v kind &> /dev/null; then
        log_info "Installing kind..."
        curl -Lo /tmp/kind "https://kind.sigs.k8s.io/dl/v0.31.0/kind-linux-amd64" 2>/dev/null
        chmod +x /tmp/kind
        sudo mv /tmp/kind /usr/local/bin/kind
        log_success "kind installed"
    fi

    # Check for kind cluster, create if needed
    if ! kind get clusters 2>/dev/null | grep -q .; then
        log_info "Creating kind cluster..."
        kind create cluster --wait 5m
        # Fix kubeconfig for Docker-in-Docker environments
        CONTAINER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' kind-control-plane)
        kubectl config set-cluster kind-kind --server="https://${CONTAINER_IP}:6443" --insecure-skip-tls-verify=true >/dev/null
        log_success "kind cluster created"
    else
        log_info "Using existing kind cluster"
    fi

    log_success "Prerequisites met"
}

build_and_load_image() {
    log_info "Building Docker image..."
    cd "$(dirname "$0")"

    # Build the image locally
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" . 2>&1 | tail -5

    # Load into kind cluster
    log_info "Loading image into kind cluster..."
    kind load docker-image "${IMAGE_NAME}:${IMAGE_TAG}" 2>&1 | tail -3

    log_success "Docker image built and loaded: ${IMAGE_NAME}:${IMAGE_TAG}"
}

deploy_server() {
    log_info "Creating namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    log_info "Deploying ModelExpress server with local storage (no PVC)..."
    helm upgrade --install "$RELEASE_NAME" ./helm \
        -n "$NAMESPACE" \
        -f ./helm/values-local-storage.yaml \
        --set image.repository="$IMAGE_NAME" \
        --set image.tag="$IMAGE_TAG" \
        --wait \
        --timeout "${TIMEOUT_SECONDS}s"

    log_success "Server deployed"

    # Get service name
    SERVICE_NAME=$(kubectl get svc -n "$NAMESPACE" -l "app.kubernetes.io/instance=$RELEASE_NAME" -o jsonpath='{.items[0].metadata.name}')
    log_info "Server service: $SERVICE_NAME"

    # Wait for pod to be ready
    log_info "Waiting for server pod to be ready..."
    kubectl wait --for=condition=ready pod \
        -l "app.kubernetes.io/instance=$RELEASE_NAME" \
        -n "$NAMESPACE" \
        --timeout="${TIMEOUT_SECONDS}s"

    log_success "Server is ready"
}

# Run test using CLI flags
run_test_cli_flags() {
    log_info "========================================"
    log_info "Test: Using CLI flags"
    log_info "========================================"
    log_info "Testing: --no-shared-storage --transfer-chunk-size 65536"

    SERVICE_NAME=$(kubectl get svc -n "$NAMESPACE" -l "app.kubernetes.io/instance=$RELEASE_NAME" -o jsonpath='{.items[0].metadata.name}')

    # Create Job that uses CLI flags
    kubectl apply -n "$NAMESPACE" -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: grpc-transfer-test-cli
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: client
        image: ${IMAGE_NAME}:${IMAGE_TAG}
        imagePullPolicy: IfNotPresent
        command: ["/bin/sh", "-c"]
        args:
        - |
          echo "=== gRPC Transfer Test (CLI Flags) ==="
          echo "Downloading model using CLI flags..."

          /app/modelexpress-cli \\
              --no-shared-storage \\
              --transfer-chunk-size 65536 \\
              --endpoint "http://${SERVICE_NAME}:8001" \\
              model download "${TEST_MODEL}"

          DOWNLOAD_RESULT=\$?
          echo ""

          # Verify files were transferred
          FILE_COUNT=\$(find /cache -type f 2>/dev/null | wc -l)

          if [ \$DOWNLOAD_RESULT -eq 0 ] && [ "\$FILE_COUNT" -gt 0 ]; then
              TOTAL_BYTES=\$(du -sb /cache 2>/dev/null | cut -f1)
              echo "=== RESULT: SUCCESS ==="
              echo "Files: \$FILE_COUNT"
              echo "Bytes: \$TOTAL_BYTES"
              exit 0
          else
              echo "=== RESULT: FAILED ==="
              echo "Files: \$FILE_COUNT"
              echo "Download exit code: \$DOWNLOAD_RESULT"
              exit 1
          fi
        env:
        - name: MODEL_EXPRESS_CACHE_DIRECTORY
          value: "/cache"
        - name: HOME
          value: "/cache"
        volumeMounts:
        - name: cache
          mountPath: /cache
      volumes:
      - name: cache
        emptyDir:
          sizeLimit: 2Gi
EOF

    log_info "Waiting for CLI flags test job to complete..."
    if kubectl wait --for=condition=complete job/grpc-transfer-test-cli -n "$NAMESPACE" --timeout="${TIMEOUT_SECONDS}s" 2>/dev/null; then
        log_info "Job output:"
        kubectl logs job/grpc-transfer-test-cli -n "$NAMESPACE" | tail -20
        log_success "CLI flags test PASSED"
        return 0
    else
        log_error "CLI flags test FAILED"
        log_info "Job output:"
        kubectl logs job/grpc-transfer-test-cli -n "$NAMESPACE" 2>/dev/null | tail -30 || true
        return 1
    fi
}

# Run test using environment variables
run_test_env_vars() {
    log_info "========================================"
    log_info "Test: Using environment variables"
    log_info "========================================"
    log_info "Testing: MODEL_EXPRESS_NO_SHARED_STORAGE=true MODEL_EXPRESS_TRANSFER_CHUNK_SIZE=65536"

    SERVICE_NAME=$(kubectl get svc -n "$NAMESPACE" -l "app.kubernetes.io/instance=$RELEASE_NAME" -o jsonpath='{.items[0].metadata.name}')

    # Create Job that uses environment variables (no CLI flags)
    kubectl apply -n "$NAMESPACE" -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: grpc-transfer-test-env
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: client
        image: ${IMAGE_NAME}:${IMAGE_TAG}
        imagePullPolicy: IfNotPresent
        command: ["/bin/sh", "-c"]
        args:
        - |
          echo "=== gRPC Transfer Test (Environment Variables) ==="
          echo "MODEL_EXPRESS_NO_SHARED_STORAGE=\$MODEL_EXPRESS_NO_SHARED_STORAGE"
          echo "MODEL_EXPRESS_TRANSFER_CHUNK_SIZE=\$MODEL_EXPRESS_TRANSFER_CHUNK_SIZE"
          echo ""
          echo "Downloading model using environment variables..."

          # Note: NOT using --no-shared-storage flag, relying on env var
          /app/modelexpress-cli \\
              --endpoint "http://${SERVICE_NAME}:8001" \\
              model download "${TEST_MODEL}"

          DOWNLOAD_RESULT=\$?
          echo ""

          # Verify files were transferred
          FILE_COUNT=\$(find /cache -type f 2>/dev/null | wc -l)

          if [ \$DOWNLOAD_RESULT -eq 0 ] && [ "\$FILE_COUNT" -gt 0 ]; then
              TOTAL_BYTES=\$(du -sb /cache 2>/dev/null | cut -f1)
              echo "=== RESULT: SUCCESS ==="
              echo "Files: \$FILE_COUNT"
              echo "Bytes: \$TOTAL_BYTES"
              exit 0
          else
              echo "=== RESULT: FAILED ==="
              echo "Files: \$FILE_COUNT"
              echo "Download exit code: \$DOWNLOAD_RESULT"
              exit 1
          fi
        env:
        - name: MODEL_EXPRESS_NO_SHARED_STORAGE
          value: "true"
        - name: MODEL_EXPRESS_TRANSFER_CHUNK_SIZE
          value: "65536"
        - name: MODEL_EXPRESS_CACHE_DIRECTORY
          value: "/cache"
        - name: HOME
          value: "/cache"
        volumeMounts:
        - name: cache
          mountPath: /cache
      volumes:
      - name: cache
        emptyDir:
          sizeLimit: 2Gi
EOF

    log_info "Waiting for environment variables test job to complete..."
    if kubectl wait --for=condition=complete job/grpc-transfer-test-env -n "$NAMESPACE" --timeout="${TIMEOUT_SECONDS}s" 2>/dev/null; then
        log_info "Job output:"
        kubectl logs job/grpc-transfer-test-env -n "$NAMESPACE" | tail -20
        log_success "Environment variables test PASSED"
        return 0
    else
        log_error "Environment variables test FAILED"
        log_info "Job output:"
        kubectl logs job/grpc-transfer-test-env -n "$NAMESPACE" 2>/dev/null | tail -30 || true
        return 1
    fi
}

# Main
log_info "gRPC File Transfer Kubernetes Test"
log_info "Mode: $TEST_MODE"
log_info "Model: $TEST_MODEL"
echo ""

check_prerequisites
build_and_load_image
deploy_server

tests_passed=0
tests_failed=0

echo ""

case "$TEST_MODE" in
    cli)
        if run_test_cli_flags; then
            tests_passed=$((tests_passed + 1))
        else
            tests_failed=$((tests_failed + 1))
        fi
        ;;
    env)
        if run_test_env_vars; then
            tests_passed=$((tests_passed + 1))
        else
            tests_failed=$((tests_failed + 1))
        fi
        ;;
    all)
        # Run both tests
        echo ""
        if run_test_cli_flags; then
            tests_passed=$((tests_passed + 1))
        else
            tests_failed=$((tests_failed + 1))
        fi

        echo ""
        if run_test_env_vars; then
            tests_passed=$((tests_passed + 1))
        else
            tests_failed=$((tests_failed + 1))
        fi
        ;;
    *)
        log_error "Unknown test mode: $TEST_MODE"
        log_info "Valid modes: cli, env, all"
        exit 1
        ;;
esac

echo ""
log_info "========================================"
log_info "Test Summary"
log_info "========================================"
echo -e "Tests passed: ${GREEN}$tests_passed${NC}"
echo -e "Tests failed: ${RED}$tests_failed${NC}"

if [ $tests_failed -eq 0 ]; then
    log_success "All tests PASSED!"
    exit 0
else
    log_error "Some tests FAILED!"
    exit 1
fi
