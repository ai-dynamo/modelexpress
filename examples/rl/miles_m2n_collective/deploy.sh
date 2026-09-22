#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly TEMPLATE="${SCRIPT_DIR}/k8s.yaml.template"

: "${KUBE_CONTEXT:=$(kubectl config current-context)}"
: "${NAMESPACE:=modelexpress-miles-m2n}"
: "${RUN_ID:=$(date -u +%Y%m%d-%H%M%S)}"
: "${RUNTIME_PULL_SECRET:=nvcr-imagepullsecret}"
: "${MX_SERVER_IMAGE:=nvcr.io/nvidian/dynamo-dev/modelexpress-server@sha256:fea73f36110fc47ab60725319c0ec1593315d90ca072368a11969c4b04c10df6}"
: "${GPU_PRODUCT:=NVIDIA-H100-80GB-HBM3}"
: "${MX_NCCL_REFIT_NUM_STREAMS:=2}"
: "${MX_NCCL_REFIT_TRANSFER_TIMEOUT_S:=600}"
: "${ACTOR_GPUS:=2}"
: "${PIPELINE_PARALLEL_SIZE:=2}"
: "${ROLLOUT_GPUS:=2}"
: "${ROLLOUT_GPUS_PER_ENGINE:=1}"
: "${TOTAL_GPUS:=4}"

require_runtime_image() {
    if [[ -z "${RUNTIME_IMAGE:-}" ]]; then
        echo "RUNTIME_IMAGE must name the pushed MILES runtime image" >&2
        exit 2
    fi
}

validate_topology() {
    if (( ACTOR_GPUS + ROLLOUT_GPUS != TOTAL_GPUS )); then
        echo "ACTOR_GPUS + ROLLOUT_GPUS must equal TOTAL_GPUS" >&2
        exit 2
    fi
    if (( ACTOR_GPUS != PIPELINE_PARALLEL_SIZE )); then
        echo "ACTOR_GPUS must equal PIPELINE_PARALLEL_SIZE for this PP-only proof" >&2
        exit 2
    fi
    if (( ROLLOUT_GPUS % ROLLOUT_GPUS_PER_ENGINE != 0 )); then
        echo "ROLLOUT_GPUS must be divisible by ROLLOUT_GPUS_PER_ENGINE" >&2
        exit 2
    fi
}

render() {
    require_runtime_image
    validate_topology
    export NAMESPACE RUN_ID RUNTIME_IMAGE RUNTIME_PULL_SECRET MX_SERVER_IMAGE
    export GPU_PRODUCT MX_NCCL_REFIT_NUM_STREAMS MX_NCCL_REFIT_TRANSFER_TIMEOUT_S
    export ACTOR_GPUS
    export PIPELINE_PARALLEL_SIZE ROLLOUT_GPUS ROLLOUT_GPUS_PER_ENGINE TOTAL_GPUS
    envsubst < "${TEMPLATE}"
}

preflight() {
    kubectl --context "${KUBE_CONTEXT}" get namespace "${NAMESPACE}" >/dev/null
    kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
        get secret "${RUNTIME_PULL_SECRET}" >/dev/null
}

apply() {
    local rendered
    rendered="$(mktemp)"
    trap 'rm -f "${rendered}"' RETURN
    render > "${rendered}"
    kubectl --context "${KUBE_CONTEXT}" apply --dry-run=client -f "${rendered}" >/dev/null
    kubectl --context "${KUBE_CONTEXT}" apply -f "${rendered}"
    kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
        rollout status deployment/mx-server --timeout=5m
}

wait_for_run() {
    local job="miles-m2n-${RUN_ID}"
    local timeout_seconds=5400
    local poll_seconds=5
    local deadline=$((SECONDS + timeout_seconds))
    local conditions

    while (( SECONDS < deadline )); do
        if ! conditions="$(
            kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
                get "job/${job}" \
                -o 'jsonpath={range .status.conditions[*]}{.type}={.status} reason={.reason} message={.message}{"\n"}{end}' \
                2>&1
        )"; then
            echo "waiting for ${job}: ${conditions}" >&2
            sleep "${poll_seconds}"
            continue
        fi

        if [[ "${conditions}" == *"Complete=True"* ]]; then
            printf 'Job %s completed\n%s\n' "${job}" "${conditions}"
            return 0
        fi
        if [[ "${conditions}" == *"Failed=True"* ]]; then
            printf 'Job %s failed\n%s\n' "${job}" "${conditions}" >&2
            kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
                get "job/${job}" -o wide >&2 || true
            return 1
        fi

        sleep "${poll_seconds}"
    done

    echo "timed out waiting for ${job} after ${timeout_seconds}s" >&2
    kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
        get "job/${job}" -o wide >&2 || true
    return 1
}

logs() {
    kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
        logs -f "job/miles-m2n-${RUN_ID}"
}

delete_run() {
    kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
        delete job "miles-m2n-${RUN_ID}" --ignore-not-found
}

case "${1:-}" in
    render) render ;;
    preflight) preflight ;;
    apply) preflight; apply ;;
    wait) wait_for_run ;;
    logs) logs ;;
    delete) delete_run ;;
    *)
        echo "usage: RUNTIME_IMAGE=... $0 {render|preflight|apply|wait|logs|delete}" >&2
        exit 2
        ;;
esac
