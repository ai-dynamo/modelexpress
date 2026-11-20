#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${NAMESPACE:?}" "${KUBE_CONTEXT:?}"
kubectl() { command kubectl --context="$KUBE_CONTEXT" "$@"; }
export -f kubectl
source .github/scripts/ci-kubectl.sh

case "$1" in
  setup)
    wait_for_gpu_pool_cleanup rl
    bash .github/scripts/ci-namespace.sh create "$NAMESPACE" dynamo
    kubectl create quota vime-delta-refit-gpu-budget -n "$NAMESPACE" \
      --hard=requests.nvidia.com/gpu=3,limits.nvidia.com/gpu=3
    kubectl create secret docker-registry nvcr-imagepullsecret -n "$NAMESPACE" \
      --docker-server=nvcr.io --docker-username='$oauthtoken' --docker-password="${NGC_API_KEY:?}"
    kubectl create secret generic mx-s3-creds -n "$NAMESPACE" \
      --from-literal=AWS_ACCESS_KEY_ID=delta-refit --from-literal=AWS_SECRET_ACCESS_KEY=delta-refit-smoke
    kubectl create secret generic hf-token-secret -n "$NAMESPACE" \
      --from-literal=HF_TOKEN="${HF_TOKEN:-}"
    kubectl wait -n "$NAMESPACE" --for=create pvc/shared-model-cache --timeout=2m
    kubectl wait -n "$NAMESPACE" --for=jsonpath='{.status.phase}'=Bound pvc/shared-model-cache --timeout=5m
    sed "s|WORKER_IMAGE|${WORKER_IMAGE:?}|g" ci/rl/model-download.yaml \
      | kubectl create -n "$NAMESPACE" -f -
    kubectl wait -n "$NAMESPACE" --for=condition=complete job/vime-delta-refit-model-download --timeout=20m
    ;;
  run)
    bash ci/rl/smoke_vime_delta_refit.sh
    ;;
  collect)
    mkdir -p "${RESULTS_DIR:?}"
    ci_kubectl get pods,jobs,deployments,services,dynamographdeployments -n "$NAMESPACE" -o yaml \
      > "$RESULTS_DIR/workloads.yaml"
    ci_kubectl get events -n "$NAMESPACE" --sort-by=.lastTimestamp > "$RESULTS_DIR/events.txt"
    pods=$(ci_kubectl get pods -n "$NAMESPACE" -o name)
    for pod in $pods; do
      kubectl logs -n "$NAMESPACE" "$pod" --all-containers --timestamps --tail=-1 \
        > "$RESULTS_DIR/${pod#pod/}.log" 2>&1 || true
    done
    ;;
  cleanup)
    bash .github/scripts/ci-namespace.sh cleanup "$NAMESPACE"
    ;;
  *) exit 2 ;;
esac
