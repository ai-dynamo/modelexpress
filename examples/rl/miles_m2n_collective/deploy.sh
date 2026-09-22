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
: "${MILES_WEIGHT_TRANSFER_MODE:=external}"
: "${MILES_BENCH_ROLLOUTS:=1}"
: "${MILES_BENCH_RUN_LABEL:=${RUN_ID}}"
: "${MILES_BENCH_BLOCK:=0}"
: "${MILES_BENCH_REPETITION:=0}"
: "${BENCH_NODE_NAME:=}"
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
: "${MX_MILES_VERIFY_TENSOR_EQUALITY:=0}"

if [[ "${MILES_WEIGHT_TRANSFER_MODE}" == "external" ]]; then
    SGLANG_PLUGINS_VALUE=modelexpress_miles_collective
    MX_SERVER_ADDRESS_VALUE=mx-server:8000
else
    SGLANG_PLUGINS_VALUE=
    MX_SERVER_ADDRESS_VALUE=
fi

readonly JOB_NAME="miles-m2n-${MILES_WEIGHT_TRANSFER_MODE}-${RUN_ID}-b${MILES_BENCH_BLOCK}-r${MILES_BENCH_REPETITION}"

require_runtime_image() {
    if [[ -z "${RUNTIME_IMAGE:-}" ]]; then
        echo "RUNTIME_IMAGE must name the pushed MILES runtime image" >&2
        exit 2
    fi
    if [[ ! "${RUNTIME_IMAGE}" =~ ^[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}$ ]]; then
        echo "RUNTIME_IMAGE must be digest-pinned with @sha256:<64 lowercase hex characters>" >&2
        exit 2
    fi
}

validate_benchmark() {
    if [[ "${MILES_WEIGHT_TRANSFER_MODE}" != "external" && "${MILES_WEIGHT_TRANSFER_MODE}" != "broadcast" ]]; then
        echo "MILES_WEIGHT_TRANSFER_MODE must be external or broadcast" >&2
        exit 2
    fi
    if [[ ! "${MILES_BENCH_ROLLOUTS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "MILES_BENCH_ROLLOUTS must be a positive integer" >&2
        exit 2
    fi
    if [[ ! "${MILES_BENCH_BLOCK}" =~ ^[0-9]+$ || ! "${MILES_BENCH_REPETITION}" =~ ^[0-9]+$ ]]; then
        echo "MILES_BENCH_BLOCK and MILES_BENCH_REPETITION must be non-negative integers" >&2
        exit 2
    fi
    if [[ "${MX_MILES_VERIFY_TENSOR_EQUALITY}" != "0" && "${MX_MILES_VERIFY_TENSOR_EQUALITY}" != "1" ]]; then
        echo "MX_MILES_VERIFY_TENSOR_EQUALITY must be 0 or 1" >&2
        exit 2
    fi
    if [[ "${MILES_WEIGHT_TRANSFER_MODE}" == "external" && ! "${MX_SERVER_IMAGE}" =~ ^[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}$ ]]; then
        echo "MX_SERVER_IMAGE must be digest-pinned with @sha256:<64 lowercase hex characters>" >&2
        exit 2
    fi
    if [[ ! "${RUN_ID}" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
        echo "RUN_ID must be a lowercase Kubernetes name component" >&2
        exit 2
    fi
    if [[ ! "${MILES_BENCH_RUN_LABEL}" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
        echo "MILES_BENCH_RUN_LABEL must be a lowercase Kubernetes label value" >&2
        exit 2
    fi
    if (( ${#JOB_NAME} > 63 || ${#MILES_BENCH_RUN_LABEL} > 63 )); then
        echo "derived job name and MILES_BENCH_RUN_LABEL must be at most 63 characters" >&2
        exit 2
    fi
    if [[ -n "${BENCH_NODE_NAME}" ]]; then
        if [[ ! "${BENCH_NODE_NAME}" =~ ^[a-z0-9]([-.a-z0-9]*[a-z0-9])?$ || ${#BENCH_NODE_NAME} -gt 253 ]]; then
            echo "BENCH_NODE_NAME must be a lowercase Kubernetes node name" >&2
            exit 2
        fi
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
    validate_benchmark
    validate_topology
    export NAMESPACE RUN_ID JOB_NAME RUNTIME_IMAGE RUNTIME_PULL_SECRET MX_SERVER_IMAGE
    export GPU_PRODUCT MX_NCCL_REFIT_NUM_STREAMS MX_NCCL_REFIT_TRANSFER_TIMEOUT_S
    export MX_MILES_VERIFY_TENSOR_EQUALITY SGLANG_PLUGINS_VALUE
    export MX_SERVER_ADDRESS_VALUE
    export MILES_WEIGHT_TRANSFER_MODE MILES_BENCH_ROLLOUTS MILES_BENCH_RUN_LABEL
    export MILES_BENCH_BLOCK MILES_BENCH_REPETITION
    export BENCH_NODE_NAME
    export ACTOR_GPUS
    export PIPELINE_PARALLEL_SIZE ROLLOUT_GPUS ROLLOUT_GPUS_PER_ENGINE TOTAL_GPUS
    envsubst < "${TEMPLATE}" | awk \
        -v include_external="$([[ "${MILES_WEIGHT_TRANSFER_MODE}" == "external" ]] && printf 1 || printf 0)" \
        -v include_node="$([[ -n "${BENCH_NODE_NAME}" ]] && printf 1 || printf 0)" '
            /^[[:space:]]*# BEGIN MODELEXPRESS SERVER RESOURCES$/ {
                skip_external = include_external != 1
                next
            }
            /^[[:space:]]*# END MODELEXPRESS SERVER RESOURCES$/ {
                skip_external = 0
                next
            }
            /^[[:space:]]*# BEGIN BENCH NODE SELECTOR$/ {
                skip_node = include_node != 1
                next
            }
            /^[[:space:]]*# END BENCH NODE SELECTOR$/ {
                skip_node = 0
                next
            }
            !skip_external && !skip_node
        '
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
    if [[ "${MILES_WEIGHT_TRANSFER_MODE}" == "external" ]]; then
        kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
            rollout status deployment/mx-server --timeout=5m
    fi
}

wait_for_run() {
    local timeout_seconds=5400
    local poll_seconds=5
    local deadline=$((SECONDS + timeout_seconds))
    local conditions

    while (( SECONDS < deadline )); do
        if ! conditions="$(
            kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
                get "job/${JOB_NAME}" \
                -o 'jsonpath={range .status.conditions[*]}{.type}={.status} reason={.reason} message={.message}{"\n"}{end}' \
                2>&1
        )"; then
            echo "waiting for ${JOB_NAME}: ${conditions}" >&2
            sleep "${poll_seconds}"
            continue
        fi

        if [[ "${conditions}" == *"Complete=True"* ]]; then
            printf 'Job %s completed\n%s\n' "${JOB_NAME}" "${conditions}"
            return 0
        fi
        if [[ "${conditions}" == *"Failed=True"* ]]; then
            printf 'Job %s failed\n%s\n' "${JOB_NAME}" "${conditions}" >&2
            kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
                get "job/${JOB_NAME}" -o wide >&2 || true
            return 1
        fi

        sleep "${poll_seconds}"
    done

    echo "timed out waiting for ${JOB_NAME} after ${timeout_seconds}s" >&2
    kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
        get "job/${JOB_NAME}" -o wide >&2 || true
    return 1
}

logs() {
    kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
        logs -f "job/${JOB_NAME}"
}

capture_logs() {
    local output_path="$1"
    kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
        logs "job/${JOB_NAME}" > "${output_path}"
}

parse_logs() {
    local log_path="$1"
    local output_path="$2"
    local expected_updates=$((MILES_BENCH_ROLLOUTS + 1))
    python3 "${SCRIPT_DIR}/parse_miles_timings.py" \
        "${log_path}" \
        "${output_path}" \
        --expected-updates "${expected_updates}" \
        --mode "${MILES_WEIGHT_TRANSFER_MODE}" \
        --run-label "${MILES_BENCH_RUN_LABEL}" \
        --block "${MILES_BENCH_BLOCK}" \
        --repetition "${MILES_BENCH_REPETITION}"
}

delete_run() {
    kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" \
        delete job "${JOB_NAME}" --ignore-not-found
}

case "${1:-}" in
    render) render ;;
    preflight) preflight ;;
    apply) preflight; apply ;;
    wait) wait_for_run ;;
    logs) logs ;;
    capture)
        [[ -n "${2:-}" ]] || {
            echo "usage: $0 capture LOG_PATH" >&2
            exit 2
        }
        capture_logs "$2"
        ;;
    parse)
        [[ -n "${2:-}" && -n "${3:-}" ]] || {
            echo "usage: $0 parse LOG_PATH JSON_PATH" >&2
            exit 2
        }
        parse_logs "$2" "$3"
        ;;
    delete) delete_run ;;
    *)
        echo "usage: RUNTIME_IMAGE=... $0 {render|preflight|apply|wait|logs|capture|parse|delete}" >&2
        exit 2
        ;;
esac
