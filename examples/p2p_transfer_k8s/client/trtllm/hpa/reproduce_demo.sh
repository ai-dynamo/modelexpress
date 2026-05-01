#!/usr/bin/env bash
# Reproduce the HPA demo flow from README.md and collect evidence.
#
# Required:
#   NAMESPACE=<namespace> ./reproduce_demo.sh
#
# Optional:
#   DGD_FILE=./kimi-agg-autoscale-dgd.yaml
#   HPA_FILE=./kimi-agg-autoscale-hpa.yaml
#   MODEL_NAME=nvidia/Kimi-K2.5-NVFP4
#   SCALE_TO=2
#   WORLD_SIZE=8
#   WAIT_SOURCE_TIMEOUT=45m
#   WAIT_SCALE_TIMEOUT=20m
#   APPLY_MODEL_METADATA_CRD=0
#   SKIP_MANUAL_SCALE=1

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-${1:-}}"
CRD_FILE="${CRD_FILE:-${SCRIPT_DIR}/../../../server/kubernetes_backend/crd-modelmetadata.yaml}"
RBAC_FILE="${RBAC_FILE:-${SCRIPT_DIR}/../../../server/kubernetes_backend/rbac-modelmetadata.yaml}"
DGD_FILE="${DGD_FILE:-${SCRIPT_DIR}/kimi-agg-autoscale-dgd.yaml}"
HPA_FILE="${HPA_FILE:-${SCRIPT_DIR}/kimi-agg-autoscale-hpa.yaml}"
MODEL_NAME="${MODEL_NAME:-nvidia/Kimi-K2.5-NVFP4}"
SCALE_TO="${SCALE_TO:-2}"
WORLD_SIZE="${WORLD_SIZE:-8}"
WAIT_SOURCE_TIMEOUT="${WAIT_SOURCE_TIMEOUT:-45m}"
WAIT_SCALE_TIMEOUT="${WAIT_SCALE_TIMEOUT:-20m}"
APPLY_MODEL_METADATA_CRD="${APPLY_MODEL_METADATA_CRD:-1}"
SKIP_MANUAL_SCALE="${SKIP_MANUAL_SCALE:-0}"

usage() {
  cat <<'EOF'
Usage:
  NAMESPACE=<namespace> ./reproduce_demo.sh

This script applies:
  1. ModelMetadata CRD/RBAC for the in-DGD ModelExpress service
  2. kimi-agg-autoscale DGD with ModelExpress + Frontend + source replicas=1
  3. DGDSA + HPA
  4. Optional manual DGDSA scale patch to SCALE_TO, unless SKIP_MANUAL_SCALE=1
  5. collect_demo_metrics.sh

Review and customize the manifests before running. In particular update node
affinity, Dynamo NATS/etcd FQDNs, and ResourceClaimTemplate.
EOF
}

if [[ -z "${NAMESPACE}" || "${NAMESPACE}" == "-h" || "${NAMESPACE}" == "--help" ]]; then
  usage
  exit 1
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

wait_for_modelmetadata_sources() {
  local deadline count
  deadline=$((SECONDS + 2700))
  while (( SECONDS < deadline )); do
    count="$(kubectl get modelmetadata -n "${NAMESPACE}" -o json 2>/dev/null \
      | jq -r --arg model "${MODEL_NAME}" '[.items[] | select(.spec.modelName == $model and .status.worker.backendType == "nixl" and (.status.worker.tensorCount // 0) > 0)] | length' \
      | tr -d ' ')"
    echo "MX published ModelMetadata workers: ${count}/${WORLD_SIZE}"
    if [[ "${count}" == "${WORLD_SIZE}" ]]; then
      return 0
    fi
    sleep 30
  done
  echo "Timed out waiting for ${WORLD_SIZE} published ModelMetadata records" >&2
  return 1
}

require_cmd kubectl
require_cmd jq

echo "Namespace: ${NAMESPACE}"
echo "DGD image:"
grep -n 'image: "nvcr.io/nvidian/dynamo-dev/kavink:dynamo-trtllm-mx' "${DGD_FILE}" || true

if [[ "${APPLY_MODEL_METADATA_CRD}" != "0" ]]; then
  echo "Applying ModelMetadata CRD from ${CRD_FILE}"
  kubectl apply -f "${CRD_FILE}"
fi

echo "Applying ModelExpress RBAC from ${RBAC_FILE}"
kubectl apply -n "${NAMESPACE}" -f "${RBAC_FILE}"
kubectl patch rolebinding -n "${NAMESPACE}" modelexpress-metadata \
  --type=json \
  -p="[{\"op\":\"replace\",\"path\":\"/subjects/0/namespace\",\"value\":\"${NAMESPACE}\"}]"

echo "Applying source DGD from ${DGD_FILE}"
kubectl apply -n "${NAMESPACE}" -f "${DGD_FILE}"

echo "Waiting for initial source DGD readiness, timeout ${WAIT_SOURCE_TIMEOUT}"
kubectl wait -n "${NAMESPACE}" --for=condition=Ready dgd/kimi-agg-autoscale --timeout="${WAIT_SOURCE_TIMEOUT}"

echo "Waiting for all source ranks to publish to ModelExpress"
wait_for_modelmetadata_sources

echo "Applying DGDSA + HPA from ${HPA_FILE}"
kubectl apply -n "${NAMESPACE}" -f "${HPA_FILE}"
kubectl get dgdsa,hpa -n "${NAMESPACE}"

if [[ "${SKIP_MANUAL_SCALE}" != "1" ]]; then
  echo "Patching DGDSA to replicas=${SCALE_TO}"
  kubectl patch dgdsa -n "${NAMESPACE}" kimi-agg-autoscale-source \
    --type=merge -p "{\"spec\":{\"replicas\":${SCALE_TO}}}"

  echo "Waiting for scaled DGD readiness, timeout ${WAIT_SCALE_TIMEOUT}"
  kubectl wait -n "${NAMESPACE}" --for=condition=Ready dgd/kimi-agg-autoscale --timeout="${WAIT_SCALE_TIMEOUT}"
fi

echo "Collecting demo metrics"
NAMESPACE="${NAMESPACE}" "${SCRIPT_DIR}/collect_demo_metrics.sh"
