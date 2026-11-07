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
    bash .github/scripts/ci-namespace.sh create "$NAMESPACE"
    kubectl create quota primerl-gpu-budget -n "$NAMESPACE" \
      --hard=requests.nvidia.com/gpu=2,limits.nvidia.com/gpu=2
    kubectl create secret docker-registry nvcr-imagepullsecret -n "$NAMESPACE" \
      --docker-server=nvcr.io --docker-username='$oauthtoken' --docker-password="${NGC_API_KEY:?}"
    kubectl create secret generic hf-token-secret -n "$NAMESPACE" \
      --from-literal=HF_TOKEN="${HF_TOKEN:-}"
    kubectl wait -n "$NAMESPACE" --for=create pvc/shared-model-cache --timeout=2m
    kubectl wait -n "$NAMESPACE" --for=jsonpath='{.status.phase}'=Bound pvc/shared-model-cache --timeout=5m
    envsubst '${WORKER_IMAGE} ${GITHUB_RUN_ID} ${GITHUB_RUN_ATTEMPT} ${MX_TEST_SHA}' \
      < ci/rl/primerl/workload.yaml | kubectl create -n "$NAMESPACE" -f -
    ;;
  run)
    deadline=$((SECONDS + 2100))
    while [ "$SECONDS" -lt "$deadline" ]; do
      job=$(ci_kubectl get job primerl-refit -n "$NAMESPACE" -o json)
      if printf '%s\n' "$job" | jq -e 'any(.status.conditions[]?; .type == "Failed" and .status == "True")' >/dev/null; then
        echo "ERROR: PrimeRL reverse-text job failed" >&2
        exit 1
      fi
      if printf '%s\n' "$job" | jq -e 'any(.status.conditions[]?; .type == "Complete" and .status == "True")' >/dev/null; then
        exit 0
      fi
      sleep 10
    done
    echo "ERROR: PrimeRL reverse-text job timed out" >&2
    exit 1
    ;;
  collect)
    mkdir -p "${RESULTS_DIR:?}"
    ci_kubectl get pods,jobs -n "$NAMESPACE" -o yaml > "$RESULTS_DIR/workloads.yaml"
    ci_kubectl get events -n "$NAMESPACE" --sort-by=.lastTimestamp > "$RESULTS_DIR/events.txt"
    kubectl logs -n "$NAMESPACE" job/primerl-refit --all-containers --timestamps --tail=-1 \
      > "$RESULTS_DIR/worker.log" 2>&1 || true
    kubectl wait -n "$NAMESPACE" --for=condition=Ready pod/primerl-evidence --timeout=2m
    kubectl cp "$NAMESPACE/primerl-evidence:/models/mx-primerl-ci/${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}/." \
      "$RESULTS_DIR/run"
    ;;
  cleanup)
    bash .github/scripts/ci-namespace.sh cleanup "$NAMESPACE"
    ;;
  *) exit 2 ;;
esac
