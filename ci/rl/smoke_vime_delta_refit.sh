#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
example=$root/examples/rl/vime_dynamo_delta_refit
namespace=${NAMESPACE:?NAMESPACE is required}
worker_image=${WORKER_IMAGE:?WORKER_IMAGE is required}
trainer_image=${TRAINER_IMAGE:?TRAINER_IMAGE is required}
model_subpath=${MODEL_SUBPATH:?MODEL_SUBPATH is required}
k=(kubectl --namespace "$namespace")
logs=$(mktemp -d)
trap 'rm -rf "$logs"' EXIT

# Check the CI namespace prerequisites.
"${k[@]}" get secret mx-minio-creds nvcr-imagepullsecret >/dev/null
"${k[@]}" get persistentvolumeclaim shared-model-cache >/dev/null

# Deploy the example stack in CI's prepared disposable namespace.
sed -e "s|WORKER_IMAGE|$worker_image|g" -e "s|MODEL_SUBPATH|$model_subpath|g" \
  "$example/stack.yaml" \
  | sed '/^        tolerations:$/i\        nodeSelector: {agentpool: a100b}' \
  | sed '/^        volumes:$/i\        - {key: no-datadog, operator: Exists, effect: NoExecute}' \
  | "${k[@]}" create -f -

# Wait for MinIO, ModelExpress, and the rollout worker.
"${k[@]}" rollout status deployment/vime-delta-refit-minio --timeout=5m
"${k[@]}" rollout status deployment/vime-delta-refit-mx --timeout=5m
"${k[@]}" wait --for=condition=Ready dynamographdeployment/vime-delta-refit --timeout=15m

# Run ten CI steps with a full Hugging Face checkpoint at versions 5 and 10.
sed -e "s|TRAINER_IMAGE|$trainer_image|g" -e "s|MODEL_SUBPATH|$model_subpath|g" \
  "$example/trainer.yaml" \
  | kubectl patch --local --type=strategic -f - \
      -p '{"spec":{"activeDeadlineSeconds":1800,"nodeSelector":{"agentpool":"a100b"},"tolerations":[{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"},{"key":"no-datadog","operator":"Exists","effect":"NoExecute"}],"containers":[{"name":"trainer","env":[{"name":"NUM_ROLLOUT","value":"10"},{"name":"FULL_HF_CHECKPOINT_INTERVAL","value":"5"}]}]}}' \
      -o yaml \
  | "${k[@]}" create -f -

# Capture the trainer log and require successful completion.
"${k[@]}" wait --for=condition=Ready pod/vime-delta-refit-trainer --timeout=20m
"${k[@]}" logs --follow pod/vime-delta-refit-trainer | tee "$logs/trainer.log"
"${k[@]}" wait --for=jsonpath='{.status.phase}'=Succeeded pod/vime-delta-refit-trainer --timeout=1m

# Verify training, the final install, and the periodic full-checkpoint cadence.
worker=$("${k[@]}" get pod \
  -l nvidia.com/dynamo-graph-deployment-name=vime-delta-refit,nvidia.com/dynamo-component=VLLMWorker \
  -o jsonpath='{.items[0].metadata.name}')
"${k[@]}" logs "$worker" -c vllm-engine | tee "$logs/vllm.log"
grep -qx 'TRAINING COMPLETE' "$logs/trainer.log"
grep -Fq 'ModelExpress weight update finished version=vime-delta-refit-v10' "$logs/vllm.log"
"${k[@]}" exec deployment/vime-delta-refit-mx -c modelexpress -- python3 -c '
from modelexpress_rl import ModelExpressControlClient, WeightPayloadFormat
with ModelExpressControlClient.connect(server_url="127.0.0.1:8101") as control:
    actual = [control.get_weight_version(f"vime-delta-refit-v{version}").payload_format for version in range(1, 11)]
expected = [WeightPayloadFormat.XOR_DELTA] * 4 + [WeightPayloadFormat.FULL_HF_CHECKPOINT]
expected += [WeightPayloadFormat.XOR_DELTA] * 4 + [WeightPayloadFormat.FULL_HF_CHECKPOINT]
assert actual == expected, actual
'
echo "SMOKE PASS"
