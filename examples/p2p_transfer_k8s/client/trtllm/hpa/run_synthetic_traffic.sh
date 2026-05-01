#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

NAMESPACE="${NAMESPACE:-zheluo}"
MANIFEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${MANIFEST_DIR}/kimi-agg-autoscale-aiperf.yaml"

echo "Namespace: ${NAMESPACE}"
echo "Waiting for the OpenAI frontend service to exist..."
kubectl get svc -n "${NAMESPACE}" kimi-agg-autoscale-frontend >/dev/null

echo "Waiting for the frontend /v1/models endpoint through an in-cluster curl pod..."
kubectl run kimi-agg-frontend-probe -n "${NAMESPACE}" --rm -i --restart=Never --image=curlimages/curl:8.10.1 -- \
  sh -c 'until curl -sf http://kimi-agg-autoscale-frontend:8000/v1/models; do sleep 5; done'

echo "Creating AIPerf Mooncake/synthetic traffic job..."
JOB_REF="$(kubectl create -n "${NAMESPACE}" -f "${MANIFEST}" -o name)"
JOB_NAME="${JOB_REF#job.batch/}"
echo "Created ${JOB_REF}"

echo "Waiting for ${JOB_NAME} to complete..."
kubectl wait --for=condition=complete -n "${NAMESPACE}" "${JOB_REF}" --timeout="${AIPERF_TIMEOUT:-90m}"

echo "AIPerf logs:"
kubectl logs -n "${NAMESPACE}" "job/${JOB_NAME}"
