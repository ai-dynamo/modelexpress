#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Read the namespace, images, and model path from the environment.
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
namespace=${NAMESPACE:?NAMESPACE is required}
worker_image=${WORKER_IMAGE:?WORKER_IMAGE is required}
trainer_image=${TRAINER_IMAGE:?TRAINER_IMAGE is required}
model_subpath=${MODEL_SUBPATH:?MODEL_SUBPATH is required}
k=(kubectl --namespace "$namespace")

# Check the namespace prerequisites before creating resources.
echo "Using context $(kubectl config current-context), namespace $namespace"
"${k[@]}" get secret mx-minio-creds nvcr-imagepullsecret >/dev/null
"${k[@]}" get persistentvolumeclaim shared-model-cache >/dev/null

# Deploy MinIO, ModelExpress, Dynamo, and the rollout worker.
sed -e "s|WORKER_IMAGE|$worker_image|g" -e "s|MODEL_SUBPATH|$model_subpath|g" \
  "$here/stack.yaml" | "${k[@]}" create -f -

# Wait for the services and rollout worker to become ready.
"${k[@]}" rollout status deployment/vime-delta-refit-minio --timeout=5m
"${k[@]}" rollout status deployment/vime-delta-refit-mx --timeout=5m
"${k[@]}" wait --for=condition=Ready dynamographdeployment/vime-delta-refit --timeout=15m

# Start the two-GPU Vime trainer.
sed -e "s|TRAINER_IMAGE|$trainer_image|g" -e "s|MODEL_SUBPATH|$model_subpath|g" \
  "$here/trainer.yaml" | "${k[@]}" create -f -

# Stream the training and refit output.
"${k[@]}" wait --for=condition=Ready pod/vime-delta-refit-trainer --timeout=20m
"${k[@]}" logs --follow pod/vime-delta-refit-trainer

# Confirm the trainer completed successfully.
"${k[@]}" wait --for=jsonpath='{.status.phase}'=Succeeded pod/vime-delta-refit-trainer --timeout=1m
