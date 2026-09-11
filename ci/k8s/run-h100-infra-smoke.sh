#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
namespace="mx-ci-h100-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"
smoke_id="${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"
storage_class=${H100_STORAGE_CLASS:-dgxc-enterprise-file}

cleanup() {
  rc=$?
  trap - EXIT

  if ! owner=$(kubectl get namespace "$namespace" --ignore-not-found \
    -o go-template='{{ index .metadata.labels "ci.modelexpress.nvidia.com/run-id" }}'); then
    echo "Unable to verify ownership of namespace '$namespace'" >&2
    exit 1
  fi
  if [[ "$owner" == "$smoke_id" ]]; then
    if (( rc != 0 )); then
      kubectl -n "$namespace" get pods,pvc,resourcequota -o wide || true
      kubectl -n "$namespace" describe pods || true
      kubectl -n "$namespace" logs pod/trainer || true
      kubectl -n "$namespace" logs pod/rollout || true
      kubectl -n "$namespace" get events --sort-by=.lastTimestamp || true
    fi

    if ! kubectl delete namespace "$namespace" --wait=true --timeout=5m; then
      rc=1
    fi
  elif [[ -n "$owner" ]]; then
    echo "Refusing to delete namespace '$namespace' owned by '$owner'" >&2
    rc=1
  fi

  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

profile=$(kubectl -n kube-system get configmap mx-ci-profile \
  -o jsonpath='{.data.profile}')
if [[ "$profile" != aws-h100 ]]; then
  echo "Expected the dedicated aws-h100 vCluster, found '$profile'" >&2
  exit 1
fi

kubectl wait crd/dynamographdeployments.nvidia.com \
  --for=condition=Established --timeout=60s
served=$(kubectl get crd dynamographdeployments.nvidia.com \
  -o jsonpath='{.spec.versions[?(@.name=="v1beta1")].served}')
if [[ "$served" != true ]]; then
  echo "DynamoGraphDeployment v1beta1 is not served" >&2
  exit 1
fi

kubectl create namespace "$namespace" --dry-run=client -o yaml \
  | kubectl label --local -f - \
      nscleanup/enabled=true \
      nscleanup/ttl=7200 \
      ci.modelexpress.nvidia.com/run-id="$smoke_id" \
      -o yaml \
  | kubectl create -f -

VLLM_ENGINE_IMAGE=nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 \
  DYNAMO_SIDECAR_IMAGE=nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 \
  MODEL_NAME=Qwen/Qwen3-0.6B \
  envsubst '${VLLM_ENGINE_IMAGE} ${DYNAMO_SIDECAR_IMAGE} ${MODEL_NAME}' \
  < "$repo_root/examples/rl/dynamo_vllm_refit/dgd.yaml" \
  | kubectl -n "$namespace" apply --dry-run=server -f - >/dev/null

SMOKE_ID="$smoke_id" H100_STORAGE_CLASS="$storage_class" \
  envsubst '${SMOKE_ID} ${H100_STORAGE_CLASS}' \
  < "$repo_root/ci/k8s/h100-infra-smoke.yaml" \
  | kubectl -n "$namespace" create -f -

kubectl -n "$namespace" wait pvc/h100-smoke-shared \
  --for=jsonpath='{.status.phase}'=Bound --timeout=3m

deadline=$((SECONDS + 600))
while true; do
  trainer_phase=$(kubectl -n "$namespace" get pod/trainer -o jsonpath='{.status.phase}')
  rollout_phase=$(kubectl -n "$namespace" get pod/rollout -o jsonpath='{.status.phase}')
  if [[ "$trainer_phase" == Failed || "$rollout_phase" == Failed ]]; then
    echo "H100 smoke pod failed: trainer=$trainer_phase rollout=$rollout_phase" >&2
    exit 1
  fi
  if [[ "$trainer_phase" == Succeeded && "$rollout_phase" == Succeeded ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "H100 smoke timed out: trainer=$trainer_phase rollout=$rollout_phase" >&2
    exit 1
  fi
  sleep 5
done

trainer_node=$(kubectl -n "$namespace" get pod/trainer \
  -o jsonpath='{.spec.nodeName}')
rollout_node=$(kubectl -n "$namespace" get pod/rollout \
  -o jsonpath='{.spec.nodeName}')
if [[ -z "$trainer_node" || -z "$rollout_node" || "$trainer_node" != "$rollout_node" ]]; then
  echo "Expected trainer and rollout on the same H100 node" >&2
  exit 1
fi

pool=$(kubectl get node "$trainer_node" \
  -o jsonpath='{.metadata.labels.ai-dynamo\.github\.com/ci-pool}')
if [[ "$pool" != h100 ]]; then
  echo "Node '$trainer_node' is not in the H100 CI pool" >&2
  exit 1
fi

trainer_log=$(kubectl -n "$namespace" logs pod/trainer)
rollout_log=$(kubectl -n "$namespace" logs pod/rollout)
printf '%s\n' "$trainer_log" "$rollout_log"
grep -Fqx H100_TRAINER_OK <<< "$trainer_log"
grep -Fqx H100_ROLLOUT_OK <<< "$rollout_log"

gpu_records=$(printf '%s\n' "$trainer_log" "$rollout_log" \
  | grep '^GPU_RECORD ' || true)
gpu_uuids=$(printf '%s\n' "$gpu_records" \
  | grep -oE 'GPU-[[:xdigit:]-]+' || true)
unique_gpu_count=$(printf '%s\n' "$gpu_uuids" \
  | sed '/^$/d' \
  | sort -u \
  | wc -l)
if [[ "$unique_gpu_count" -ne 3 ]]; then
  echo "Expected 3 unique H100 UUIDs, found $unique_gpu_count" >&2
  exit 1
fi

echo "H100_INFRA_SMOKE_OK node=$trainer_node"
