#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
example=$root/examples/rl/vime_dynamo_delta_refit
namespace=${NAMESPACE:?NAMESPACE is required}
context=${KUBE_CONTEXT:?KUBE_CONTEXT is required}
worker_image=${WORKER_IMAGE:?WORKER_IMAGE is required}
trainer_image=${TRAINER_IMAGE:?TRAINER_IMAGE is required}
model_subpath=${MODEL_SUBPATH:?MODEL_SUBPATH is required}
k=(kubectl --kubeconfig=/teleport/kubeconfig.yaml --context="$context" --namespace="$namespace")
logs=$(mktemp -d)
trap 'rm -rf "$logs"' EXIT

# Check the namespace prerequisites.
"${k[@]}" get secret mx-minio-creds nvcr-imagepullsecret >/dev/null
"${k[@]}" get persistentvolumeclaim shared-model-cache >/dev/null

# Deploy the example stack with its TP1 rollout worker on one H100.
sed -e "s|WORKER_IMAGE|$worker_image|g" -e "s|MODEL_SUBPATH|$model_subpath|g" \
  "$example/stack.yaml" \
  | sed '/^        tolerations:$/i\        priorityClassName: ci-nightly-high-priority\n        nodeSelector: {kubernetes.io/arch: amd64, nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3}' \
  | "${k[@]}" create -f -

# Wait for storage, ModelExpress, and rollout readiness.
"${k[@]}" rollout status deployment/vime-delta-refit-minio --timeout=5m
"${k[@]}" rollout status deployment/vime-delta-refit-mx --timeout=5m
"${k[@]}" wait --for=condition=Ready dynamographdeployment/vime-delta-refit --timeout=20m

# Run ten TP2 trainer steps, with full HF checkpoints at v4 and v8.
sed -e "s|TRAINER_IMAGE|$trainer_image|g" -e "s|MODEL_SUBPATH|$model_subpath|g" \
  "$example/trainer.yaml" \
  | kubectl patch --local --type=strategic -f - \
      -p '{"spec":{"activeDeadlineSeconds":1800,"priorityClassName":"ci-nightly-high-priority","nodeSelector":{"kubernetes.io/arch":"amd64","nvidia.com/gpu.product":"NVIDIA-H100-80GB-HBM3"},"containers":[{"name":"trainer","env":[{"name":"NUM_ROLLOUT","value":"10"},{"name":"FULL_HF_CHECKPOINT_INTERVAL","value":"4"}]}]}}' \
      -o yaml \
  | "${k[@]}" create -f -

# Capture the trainer log and require successful completion.
"${k[@]}" wait --for=condition=Ready pod/vime-delta-refit-trainer --timeout=20m
"${k[@]}" logs --follow pod/vime-delta-refit-trainer | tee "$logs/trainer.log"
"${k[@]}" wait --for=jsonpath='{.status.phase}'=Succeeded pod/vime-delta-refit-trainer --timeout=1m

# Verify all installs and the full-checkpoint cadence.
worker=$("${k[@]}" get pod \
  -l nvidia.com/dynamo-graph-deployment-name=vime-delta-refit,nvidia.com/dynamo-component=VLLMWorker \
  -o jsonpath='{.items[0].metadata.name}')
"${k[@]}" logs "$worker" -c vllm-engine | tee "$logs/vllm.log"
grep -qx 'TRAINING COMPLETE' "$logs/trainer.log"
mapfile -t version_ids < <(
  sed -n 's/.*ModelExpress weight update finished version=\([^ ]*\).*/\1/p' "$logs/vllm.log" \
    | awk '!seen[$0]++'
)
test "${#version_ids[@]}" -eq 10
"${k[@]}" exec deployment/vime-delta-refit-mx -c modelexpress -- python3 -c '
import sys

from modelexpress_rl import ModelExpressControlClient, WeightPayloadFormat
version_ids = ["vime-delta-refit-v0", *sys.argv[1:]]
with ModelExpressControlClient.connect(server_url="127.0.0.1:8101") as control:
    actual = [control.get_weight_version(version_id).payload_format for version_id in version_ids]
expected = [WeightPayloadFormat.FULL_TENSOR]
expected += [
    WeightPayloadFormat.FULL_HF_CHECKPOINT if version % 4 == 0 else WeightPayloadFormat.XOR_DELTA
    for version in range(1, 11)
]
assert actual == expected, actual
' "${version_ids[@]}"
echo "SMOKE PASS"
