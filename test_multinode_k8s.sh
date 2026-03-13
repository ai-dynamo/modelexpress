#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the ModelExpress Docker image, distribute it to all cluster nodes,
# and run the multi-node Kubernetes integration tests.
#
# The actual test logic lives in workspace-tests/tests/k8s_multinode_tests.rs.
# This script only handles image preparation, which can't be done from Rust.
#
# Prerequisites:
#   - Docker
#   - A multi-node Kubernetes cluster (at least 2 schedulable nodes) with kubeconfig accessible
#   - The modelexpress:multinode-test image must be available on all cluster nodes
#
# Image distribution:
#   - microk8s: auto-detected, uses `microk8s images import` to distribute to all nodes
#   - Other clusters: push the image to your registry and ensure nodes can pull it,
#     or use your cluster's image distribution mechanism. Use --skip-build to skip
#     the Docker build step if the image is already available.
#
# The tests themselves are cluster-agnostic: they discover nodes dynamically via
# the Kubernetes API and only require standard kubeconfig access.
#
# Usage:
#   ./test_multinode_k8s.sh [--skip-build] [-- cargo test args...]
#
# Examples:
#   ./test_multinode_k8s.sh                              # Build + run all tests
#   ./test_multinode_k8s.sh --skip-build                 # Skip image build, just run tests
#   ./test_multinode_k8s.sh -- --test cross_node         # Run only cross_node tests

set -e
cd "$(dirname "$0")"

SKIP_BUILD=false
CARGO_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build) SKIP_BUILD=true; shift ;;
        --) shift; CARGO_ARGS=("$@"); break ;;
        *) CARGO_ARGS=("$@"); break ;;
    esac
done

# Step 1: Build and distribute image
if [ "$SKIP_BUILD" = false ]; then
    echo "Building Docker image modelexpress:multinode-test..."
    docker build -t modelexpress:multinode-test .

    if command -v /snap/bin/microk8s &>/dev/null; then
        echo "Distributing image to microk8s cluster nodes..."
        docker save modelexpress:multinode-test | /snap/bin/microk8s images import -
    fi
fi

# Step 2: Run the Rust k8s tests (ignored by default, --ignored enables them)
cargo test --test k8s_multinode_tests -- --ignored "${CARGO_ARGS[@]}"
