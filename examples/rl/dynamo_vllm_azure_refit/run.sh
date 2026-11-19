#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ "${1:-}" == --help ]]; then
  cat <<'USAGE'
Run the Azure full-checkpoint, delta-update, and restart example.

Set NAMESPACE, RUN_ID, WORKER_IMAGE, and DYNAMO_FRONTEND_IMAGE.
MODEL_NAME and MODEL_SUBPATH default to Qwen/Qwen3-0.6B.
Use a dedicated namespace with hf-token-secret, nvcr-imagepullsecret,
and shared-model-cache containing the complete model checkpoint.
SCENARIO=blob (default) uses one GPU and restores from Blob after restart.
SCENARIO=p2p uses two GPUs with RDMA. It checks Blob-to-generator-to-peer
updates and restores a replacement generator from the surviving peer.
Requires the Dynamo v1beta1 operator, kubectl, and envsubst.
The worker image must be built with this directory's Dockerfile and the
same Dynamo revision as the frontend. See docs/rl/CONFIGURATION.md.

This run creates Azurite, Redis, ModelExpress, and a Dynamo deployment.
It deletes one generator pod after updates to verify the selected recovery path.
Resources remain in the namespace for inspection after success or failure.
USAGE
  exit 0
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
namespace=${NAMESPACE:?NAMESPACE is required}
export RUN_ID=${RUN_ID:?RUN_ID is required}
export WORKER_IMAGE=${WORKER_IMAGE:?WORKER_IMAGE is required}
export DYNAMO_FRONTEND_IMAGE=${DYNAMO_FRONTEND_IMAGE:?DYNAMO_FRONTEND_IMAGE is required}
export MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-0.6B}
export MODEL_SUBPATH=${MODEL_SUBPATH:-Qwen/Qwen3-0.6B}
scenario=${SCENARIO:-blob}
case "$scenario" in
  blob)
    export EXPECTED_WORKERS=1 SOURCE_ORDER=OBJECT_STORAGE P2P_METADATA=0 RDMA_DEVICES=0
    restart_strategy=desired_version_object_storage
    ;;
  p2p)
    export EXPECTED_WORKERS=2 SOURCE_ORDER=GENERATOR,OBJECT_STORAGE P2P_METADATA=1 RDMA_DEVICES=1
    restart_strategy=desired_version_p2p
    ;;
  *) echo "SCENARIO must be blob or p2p" >&2; exit 1 ;;
esac
if [[ "$namespace" == default || "$namespace" == kube-* ]]; then
  echo "Use a dedicated namespace for this example" >&2
  exit 1
fi
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$ ]]; then
  echo "RUN_ID must be 1-64 letters, digits, underscores, or hyphens" >&2
  exit 1
fi
k=(kubectl --namespace "$namespace")

wait_for_job() {
  local job=$1 deadline=$((SECONDS + $2)) conditions
  local job_error="Timed out waiting for job $1"
  while (( SECONDS < deadline )); do
    if ! conditions=$("${k[@]}" get "job/$job" --request-timeout=30s \
      -o jsonpath='{range .status.conditions[?(@.status=="True")]}{.type}{"\n"}{end}'); then
      job_error="Could not read status for job $job"
      break
    fi
    if grep -Fxq Failed <<< "$conditions"; then
      job_error="Job $job failed"
      break
    fi
    if grep -Fxq Complete <<< "$conditions"; then
      return 0
    fi
    sleep 5
  done
  echo "$job_error" >&2
  "${k[@]}" logs "job/$job" --all-containers --prefix --ignore-errors --request-timeout=30s >&2 || true
  return 1
}

echo "Using context $(kubectl config current-context), namespace $namespace"
"${k[@]}" get secret hf-token-secret nvcr-imagepullsecret >/dev/null
"${k[@]}" get persistentvolumeclaim shared-model-cache >/dev/null
existing=$("${k[@]}" get dynamographdeployment mx-azure-refit --ignore-not-found -o name)
if [[ -n "$existing" ]]; then
  echo "An Azure refit run already exists here; use a fresh example namespace" >&2
  exit 1
fi

"${k[@]}" create configmap mx-azure-refit-scripts \
  --from-file="$here/publish_versions.py" --from-file="$here/orchestrate.py" \
  --dry-run=client -o yaml | "${k[@]}" apply -f -
# envsubst needs literal variable names; preserve the worker's command substitution.
# shellcheck disable=SC2016
variables='${RUN_ID} ${WORKER_IMAGE} ${DYNAMO_FRONTEND_IMAGE} ${MODEL_NAME} ${MODEL_SUBPATH}'
# shellcheck disable=SC2016
variables+=' ${EXPECTED_WORKERS} ${SOURCE_ORDER} ${P2P_METADATA} ${RDMA_DEVICES}'
envsubst "$variables" < "$here/stack.yaml" | "${k[@]}" apply -f -
"${k[@]}" rollout status deployment/mx-azure-refit-azurite --timeout=5m
"${k[@]}" rollout status deployment/mx-azure-refit-server --timeout=5m
"${k[@]}" wait --for=condition=Ready dynamographdeployment/mx-azure-refit --timeout=20m

envsubst "$variables" < "$here/test-job.yaml" | "${k[@]}" apply -f -
wait_for_job mx-azure-refit-active 1800
"${k[@]}" logs job/mx-azure-refit-active -c publish-versions
"${k[@]}" logs job/mx-azure-refit-active -c orchestrate
desired=$("${k[@]}" get configmap mx-azure-refit-desired-version -o jsonpath='{.data.desired_version_uid}')
if [[ "$desired" != "$RUN_ID-v3" ]]; then
  echo "Desired version did not advance to $RUN_ID-v3" >&2
  exit 1
fi

selector='nvidia.com/dynamo-graph-deployment-name=mx-azure-refit,nvidia.com/dynamo-component-type=worker'
read -r -a workers <<< "$("${k[@]}" get pods -l "$selector" -o jsonpath='{.items[*].metadata.name}')"
if [[ ${#workers[@]} -ne "$EXPECTED_WORKERS" ]]; then
  echo "Expected $EXPECTED_WORKERS generator pods" >&2
  exit 1
fi
worker=${workers[0]}
logs=""
for pod in "${workers[@]}"; do
  logs+=$("${k[@]}" logs "$pod" -c vllm-engine)
  logs+=$'\n'
done
for number in 1 2 3; do
  if ! grep -F "active refit version=$RUN_ID-v$number trying source=OBJECT_STORAGE" <<< "$logs" >/dev/null; then
    echo "Version $RUN_ID-v$number did not use Azure object storage" >&2
    exit 1
  fi
  if [[ "$scenario" == p2p ]] && ! grep -F \
    "weight update version=$RUN_ID-v$number prepared source=GENERATOR method=RuntimeTensorNixlUpdateMethod" \
    <<< "$logs" >/dev/null; then
    echo "Version $RUN_ID-v$number did not exercise generator P2P" >&2
    exit 1
  fi
done
"${k[@]}" delete pod "$worker" --wait=true

replacement=""
for _ in $(seq 1 120); do
  while IFS= read -r pod; do
    [[ -z "$pod" || " ${workers[*]} " == *" $pod "* ]] && continue
    # A listed pod can disappear or start terminating before this lookup.
    state=$("${k[@]}" get pod "$pod" --ignore-not-found \
      -o jsonpath='{.metadata.deletionTimestamp}{"|"}{.status.conditions[?(@.type=="Ready")].status}') || continue
    if [[ "$state" == '|True' ]]; then
      replacement=$pod
      break 2
    fi
  done < <("${k[@]}" get pods -l "$selector" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')
  sleep 5
done
if [[ -z "$replacement" ]]; then
  echo "Replacement generator did not become Ready" >&2
  exit 1
fi
logs=$("${k[@]}" logs "$replacement" -c vllm-engine)
if ! grep -F "Trying strategy: $restart_strategy" <<< "$logs" >/dev/null || \
    grep -F "Strategy $restart_strategy " <<< "$logs" >/dev/null; then
  echo "Replacement did not restore through $restart_strategy" >&2
  exit 1
fi
envsubst "$variables" < "$here/verify-job.yaml" | "${k[@]}" apply -f -
wait_for_job mx-azure-refit-restart-verify 1200
"${k[@]}" logs job/mx-azure-refit-restart-verify
echo "E2E PASS: Azure full checkpoint, two deltas, and $scenario replacement recovery"
