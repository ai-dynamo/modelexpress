#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
namespace=${NAMESPACE:?NAMESPACE is required}
run_id=${RUN_ID:?RUN_ID is required}
model_name=${MODEL_NAME:-Qwen/Qwen3-0.6B}
model_subpath=${MODEL_SUBPATH:-Qwen/Qwen3-0.6B}
worker_image=${WORKER_IMAGE:?WORKER_IMAGE is required}
dynamo_frontend_image=${DYNAMO_FRONTEND_IMAGE:?DYNAMO_FRONTEND_IMAGE is required}
k=(kubectl --namespace "$namespace")

echo "Using context $(kubectl config current-context), namespace $namespace"
"${k[@]}" get secret hf-token-secret mx-minio-creds nvcr-imagepullsecret >/dev/null
"${k[@]}" get persistentvolumeclaim shared-model-cache >/dev/null

"${k[@]}" create configmap mx-s3-refit-scripts \
  --from-file="$here/publish_versions.py" \
  --from-file="$here/orchestrate.py" \
  --dry-run=client -o yaml | "${k[@]}" apply -f -
"${k[@]}" delete job mx-s3-refit-active mx-s3-refit-restart-verify \
  --ignore-not-found --wait=true

export RUN_ID="$run_id"
export MODEL_NAME="$model_name"
export MODEL_SUBPATH="$model_subpath"
export WORKER_IMAGE="$worker_image"
export DYNAMO_FRONTEND_IMAGE="$dynamo_frontend_image"

envsubst < "$here/stack.yaml" | "${k[@]}" apply -f -
"${k[@]}" rollout status deployment/mx-s3-refit-minio --timeout=5m
"${k[@]}" rollout status deployment/mx-s3-refit-server --timeout=5m
"${k[@]}" wait --for=condition=Ready \
  dynamographdeployment/mx-s3-refit --timeout=20m

envsubst < "$here/test-job.yaml" | "${k[@]}" apply -f -
"${k[@]}" wait --for=condition=complete job/mx-s3-refit-active --timeout=30m
"${k[@]}" logs job/mx-s3-refit-active -c publish-versions
"${k[@]}" logs job/mx-s3-refit-active -c orchestrate

expected_version="${run_id}-v3"
actual_desired=$("${k[@]}" get configmap mx-s3-refit-desired-version \
  -o jsonpath='{.data.desired_version_uid}')
if [[ "$actual_desired" != "$expected_version" ]]; then
  echo "desired version is $actual_desired, expected $expected_version" >&2
  exit 1
fi

worker_selector="nvidia.com/dynamo-graph-deployment-name=mx-s3-refit,nvidia.com/dynamo-component-type=worker"
worker_pods=()
while IFS= read -r pod; do
  [[ -n "$pod" ]] && worker_pods+=("$pod")
done < <("${k[@]}" get pods -l "$worker_selector" \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | sort)
if [[ ${#worker_pods[@]} -ne 2 ]]; then
  echo "found ${#worker_pods[@]} Dynamo workers, expected 2" >&2
  exit 1
fi

for number in 1 2 3; do
  version="${run_id}-v${number}"
  saw_object_storage=false
  saw_generator=false
  for pod in "${worker_pods[@]}"; do
    if "${k[@]}" logs "$pod" -c vllm-engine | \
      grep -F "active refit version=$version trying source=OBJECT_STORAGE" \
      >/dev/null; then
      saw_object_storage=true
    fi
    if "${k[@]}" logs "$pod" -c vllm-engine | \
      grep -F "weight update version=$version prepared source=GENERATOR method=RuntimeTensorNixlUpdateMethod" \
      >/dev/null; then
      saw_generator=true
    fi
  done
  if [[ "$saw_object_storage" != true || "$saw_generator" != true ]]; then
    echo "version $version did not exercise both S3 and generator P2P" >&2
    exit 1
  fi
done

worker_pod=${worker_pods[0]}
survivor_pod=${worker_pods[1]}
"${k[@]}" delete pod "$worker_pod" --wait=true

replacement=""
for _ in $(seq 1 120); do
  while IFS= read -r pod; do
    [[ -z "$pod" || "$pod" == "$survivor_pod" ]] && continue
    deletion_timestamp=$("${k[@]}" get pod "$pod" \
      -o jsonpath='{.metadata.deletionTimestamp}' 2>/dev/null || true)
    ready=$("${k[@]}" get pod "$pod" \
      -o jsonpath='{range .status.conditions[?(@.type=="Ready")]}{.status}{end}' \
      2>/dev/null || true)
    if [[ -z "$deletion_timestamp" && "$ready" == "True" ]]; then
      replacement=$pod
      break 2
    fi
  done < <("${k[@]}" get pods -l "$worker_selector" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null | sort)
  sleep 5
done
if [[ -z "$replacement" ]]; then
  echo "replacement Dynamo worker did not become Ready" >&2
  exit 1
fi
replacement_logs=$("${k[@]}" logs "$replacement" -c vllm-engine)
if ! grep -F "Trying strategy: desired_version_p2p" \
  <<<"$replacement_logs" >/dev/null; then
  echo "replacement did not attempt desired-version P2P" >&2
  exit 1
fi
if grep -F "Strategy desired_version_p2p" \
  <<<"$replacement_logs" >/dev/null; then
  echo "replacement failed desired-version P2P instead of using the survivor" >&2
  exit 1
fi

envsubst < "$here/verify-job.yaml" | "${k[@]}" apply -f -
"${k[@]}" wait --for=condition=complete \
  job/mx-s3-refit-restart-verify --timeout=20m
"${k[@]}" logs job/mx-s3-refit-restart-verify

echo "E2E PASS: active full-plus-delta refit and replacement inheritance"
