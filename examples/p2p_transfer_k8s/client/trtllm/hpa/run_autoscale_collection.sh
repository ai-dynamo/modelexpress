#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Orchestrate one apples-to-apples autoscale measurement run.
#
# The script owns the measurement window:
#   1. reset the scaling adapter to SCALE_FROM replicas
#   2. start live samplers
#   3. launch the configured AIPerf Job
#   4. wait SCALE_DELAY_SECONDS
#   5. patch the scaling adapter to SCALE_TO replicas and record the timestamp
#   6. wait for AIPerf to finish
#   7. stop samplers and run collect_demo_metrics.sh
#
# Use the same TRAFFIC_PROFILE_ID, AIPERF_MANIFEST, SCALE_FROM,
# SCALE_TO, and SCALE_DELAY_SECONDS for RUN_TYPE=mx and RUN_TYPE=vanilla.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  NAMESPACE=<namespace> RUN_TYPE=mx ./run_autoscale_collection.sh
  NAMESPACE=<namespace> RUN_TYPE=vanilla DGD_NAME=<vanilla-dgd> FRONTEND_SERVICE=<svc:port> ./run_autoscale_collection.sh

Required:
  NAMESPACE

Important environment:
  RUN_TYPE              mx or vanilla. Default: mx
  DGD_NAME              Default: kimi-agg-autoscale
  SERVICE_NAME          Scaled DGD service. Default: source
  DGDSA_NAME            Default: ${DGD_NAME}-${SERVICE_NAME}
  APP_LABEL             Default: app.kubernetes.io/part-of=${DGD_NAME}
  FRONTEND_SERVICE      Default: ${DGD_NAME}-frontend:8000
  MODEL_NAME            Default: nvidia/Kimi-K2.5-NVFP4
  AIPERF_MANIFEST       Default: ./kimi-agg-autoscale-aiperf.yaml
  AIPERF_SCALE_MARKER   Regex in AIPerf logs used as the scale-delay anchor.
                        Default: Running Mooncake concurrency surge
  SCALE_TRIGGER_MODE    timer or log_marker. timer scales after
                        CONCURRENCY_RAMP_DURATION_SECONDS + SCALE_DELAY_SECONDS
                        from AIPerf job creation. Default: timer
  AIPERF_MARKER_TIMEOUT Default: 30m
  TRAFFIC_PROFILE_ID    Free-form label. Default: mooncake-kimi-concurrency-1-128
  BASELINE_CONCURRENCY  AIPerf baseline concurrency metadata. Default: 1
  SURGE_CONCURRENCY     AIPerf target concurrency. Default: 128
  BASELINE_DURATION_SECONDS  Baseline phase duration. Default: 90
  SURGE_DURATION_SECONDS     Surge phase duration. Default: 1710
  CONCURRENCY_RAMP_DURATION_SECONDS  AIPerf concurrency ramp duration.
                                      Default: BASELINE_DURATION_SECONDS
  BENCHMARK_GRACE_PERIOD_SECONDS     AIPerf benchmark grace period. Default: 900
  PROGRESS_INTERVAL_SECONDS           AIPerf progress log interval. Default: 30
  TRACE_REQUEST_COUNT   Max Mooncake rows to use. Default: 20000
  TRACE_MAX_ISL         Raw Mooncake input length cap. Default: 256000
  TRACE_MAX_OSL         Mooncake output length cap. Default: 8000
  SCALE_FROM            Default: 1
  SCALE_TO              Default: 2
  SCALE_DELAY_SECONDS   Delay after the AIPerf scale marker before scale-up.
                        Default: 60
  SAMPLE_INTERVAL_SECONDS Default: 10
  SLICE_SECONDS           Dashboard time-series slice size. Default: 30
  SOURCE_PODS_PER_REPLICA Expected source pods per replica. Default: 2
  WAIT_FOR_DGD_READY      If 1, wait for DGD Ready before pod readiness.
                          Default: 1
  PATCH_GROVE_DIRECT      If 1, also patch the Grove PodCliqueScalingGroup
                          replica count. Useful when DGD reconcile is wedged.
                          Default: 0
  PATCH_DGD_DIRECT        If 1, patch DGD service replicas directly.
                          Useful for Grove-backed DGD services without DGDScaleAdapter.
                          Default: 0
  DGD_SERVICE_KEY         DGD service key to patch when PATCH_DGD_DIRECT=1.
                          Default: ${SERVICE_NAME}
  GROVE_PCSG_NAME         Default: ${DGD_NAME}-0-${SERVICE_NAME}
  SCALE_PATCH_TARGET      dgdsa, grove, or both. Default: dgdsa
  SCALED_POD_NAME_REGEX   Regex for pods counted as scaled workers.
                          Default: source
  AIPERF_TIMEOUT        Default: 120m
  AIPERF_ARTIFACT_FILES Space-separated artifact files to copy from the
                        AIPerf result directory. Default: structured metrics
                        only; excludes huge raw input dumps.
  AIPERF_FALLBACK_ARTIFACT_ROOTS Space-separated PVC mount roots to try when
                        the logged result path is not mounted by the frontend.
                        Default: /models /model-cache
  OUT_ROOT              Default: ./metrics
  RUN_ID                Default: <utc timestamp>_<run_type>

Optional:
  HPA_NAME              Default: ${DGD_NAME}
  CLEAN_MX_METADATA     If 1 and RUN_TYPE=mx, delete existing ModelMetadata before run.
                         Default: 0
  SAMPLE_GPU            If 1, sample nvidia-smi from source pods. Default: 1
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

NAMESPACE="${NAMESPACE:-${1:-}}"
if [[ -z "${NAMESPACE}" ]]; then
  usage
  exit 1
fi

RUN_TYPE="${RUN_TYPE:-mx}"
DGD_NAME="${DGD_NAME:-kimi-agg-autoscale}"
SERVICE_NAME="${SERVICE_NAME:-source}"
DGDSA_NAME="${DGDSA_NAME:-${DGD_NAME}-${SERVICE_NAME}}"
HPA_NAME="${HPA_NAME:-${DGD_NAME}}"
APP_LABEL="${APP_LABEL:-app.kubernetes.io/part-of=${DGD_NAME}}"
FRONTEND_SERVICE="${FRONTEND_SERVICE:-${DGD_NAME}-frontend:8000}"
SOURCE_SERVICE="${SOURCE_SERVICE:-${DGD_NAME}-${SERVICE_NAME}:9090}"
MODEL_NAME="${MODEL_NAME:-nvidia/Kimi-K2.5-NVFP4}"
AIPERF_MANIFEST="${AIPERF_MANIFEST:-${SCRIPT_DIR}/kimi-agg-autoscale-aiperf.yaml}"
AIPERF_SCALE_MARKER="${AIPERF_SCALE_MARKER:-Running Mooncake concurrency surge}"
SCALE_TRIGGER_MODE="${SCALE_TRIGGER_MODE:-timer}"
AIPERF_MARKER_TIMEOUT="${AIPERF_MARKER_TIMEOUT:-30m}"
TRAFFIC_PROFILE_ID="${TRAFFIC_PROFILE_ID:-mooncake-kimi-concurrency-1-128}"
BASELINE_CONCURRENCY="${BASELINE_CONCURRENCY:-1}"
SURGE_CONCURRENCY="${SURGE_CONCURRENCY:-128}"
BASELINE_DURATION_SECONDS="${BASELINE_DURATION_SECONDS:-90}"
SURGE_DURATION_SECONDS="${SURGE_DURATION_SECONDS:-1710}"
CONCURRENCY_RAMP_DURATION_SECONDS="${CONCURRENCY_RAMP_DURATION_SECONDS:-${BASELINE_DURATION_SECONDS}}"
BENCHMARK_GRACE_PERIOD_SECONDS="${BENCHMARK_GRACE_PERIOD_SECONDS:-900}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-30}"
TRACE_REQUEST_COUNT="${TRACE_REQUEST_COUNT:-20000}"
TRACE_MAX_ISL="${TRACE_MAX_ISL:-256000}"
TRACE_MAX_OSL="${TRACE_MAX_OSL:-8000}"
SCALE_FROM="${SCALE_FROM:-1}"
SCALE_TO="${SCALE_TO:-2}"
SCALE_DELAY_SECONDS="${SCALE_DELAY_SECONDS:-60}"
SAMPLE_INTERVAL_SECONDS="${SAMPLE_INTERVAL_SECONDS:-10}"
SLICE_SECONDS="${SLICE_SECONDS:-30}"
SOURCE_PODS_PER_REPLICA="${SOURCE_PODS_PER_REPLICA:-2}"
WAIT_FOR_DGD_READY="${WAIT_FOR_DGD_READY:-1}"
PATCH_GROVE_DIRECT="${PATCH_GROVE_DIRECT:-0}"
PATCH_DGD_DIRECT="${PATCH_DGD_DIRECT:-0}"
DGD_SERVICE_KEY="${DGD_SERVICE_KEY:-${SERVICE_NAME}}"
GROVE_PCSG_NAME="${GROVE_PCSG_NAME:-${DGD_NAME}-0-${SERVICE_NAME}}"
SCALE_PATCH_TARGET="${SCALE_PATCH_TARGET:-dgdsa}"
SCALED_POD_NAME_REGEX="${SCALED_POD_NAME_REGEX:-source}"
AIPERF_TIMEOUT="${AIPERF_TIMEOUT:-120m}"
AIPERF_ARTIFACT_FILES="${AIPERF_ARTIFACT_FILES:-profile_export.jsonl profile_export_aiperf.csv profile_export_aiperf.json server_metrics_export.csv server_metrics_export.json trace_stats.json input_config.json}"
AIPERF_FALLBACK_ARTIFACT_ROOTS="${AIPERF_FALLBACK_ARTIFACT_ROOTS:-/models /model-cache}"
CLEAN_MX_METADATA="${CLEAN_MX_METADATA:-0}"
SAMPLE_GPU="${SAMPLE_GPU:-1}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${RUN_ID:-${STAMP}_${RUN_TYPE}}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/metrics}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${RUN_ID}}"
SAMPLE_DIR="${OUT_DIR}/samples"
RAW_METRICS_DIR="${SAMPLE_DIR}/frontend_metrics"
STOP_FILE="${OUT_DIR}/.sampling"

require_cmd kubectl
require_cmd jq
require_cmd perl

mkdir -p "${SAMPLE_DIR}" "${RAW_METRICS_DIR}" "${OUT_DIR}/logs"

cat > "${OUT_DIR}/run_config.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "run_type": "${RUN_TYPE}",
  "namespace": "${NAMESPACE}",
  "dgd_name": "${DGD_NAME}",
  "service_name": "${SERVICE_NAME}",
  "dgdsa_name": "${DGDSA_NAME}",
  "hpa_name": "${HPA_NAME}",
  "app_label": "${APP_LABEL}",
  "frontend_service": "${FRONTEND_SERVICE}",
  "source_service": "${SOURCE_SERVICE}",
  "model_name": "${MODEL_NAME}",
  "traffic_profile_id": "${TRAFFIC_PROFILE_ID}",
  "baseline_concurrency": ${BASELINE_CONCURRENCY},
  "surge_concurrency": ${SURGE_CONCURRENCY},
  "baseline_duration_seconds": ${BASELINE_DURATION_SECONDS},
  "surge_duration_seconds": ${SURGE_DURATION_SECONDS},
  "total_duration_seconds": $((BASELINE_DURATION_SECONDS + SURGE_DURATION_SECONDS)),
  "concurrency_ramp_duration_seconds": ${CONCURRENCY_RAMP_DURATION_SECONDS},
  "traffic_mode": "continuous_concurrency_ramp",
  "traffic_start_source": "first_request",
  "benchmark_grace_period_seconds": ${BENCHMARK_GRACE_PERIOD_SECONDS},
  "progress_interval_seconds": ${PROGRESS_INTERVAL_SECONDS},
  "trace_request_count": ${TRACE_REQUEST_COUNT},
  "trace_max_isl": ${TRACE_MAX_ISL},
  "trace_max_osl": ${TRACE_MAX_OSL},
  "aiperf_manifest": "${AIPERF_MANIFEST}",
  "aiperf_scale_marker": "${AIPERF_SCALE_MARKER}",
  "scale_trigger_mode": "${SCALE_TRIGGER_MODE}",
  "aiperf_marker_timeout": "${AIPERF_MARKER_TIMEOUT}",
  "scale_from": ${SCALE_FROM},
  "scale_to": ${SCALE_TO},
  "scale_delay_seconds": ${SCALE_DELAY_SECONDS},
  "sample_interval_seconds": ${SAMPLE_INTERVAL_SECONDS},
  "slice_seconds": ${SLICE_SECONDS},
  "source_pods_per_replica": ${SOURCE_PODS_PER_REPLICA},
  "wait_for_dgd_ready": ${WAIT_FOR_DGD_READY},
  "patch_grove_direct": ${PATCH_GROVE_DIRECT},
  "patch_dgd_direct": ${PATCH_DGD_DIRECT},
  "dgd_service_key": "${DGD_SERVICE_KEY}",
  "grove_pcsg_name": "${GROVE_PCSG_NAME}",
  "scale_patch_target": "${SCALE_PATCH_TARGET}",
  "scaled_pod_name_regex": "${SCALED_POD_NAME_REGEX}",
  "aiperf_artifact_files": "${AIPERF_ARTIFACT_FILES}",
  "aiperf_fallback_artifact_roots": "${AIPERF_FALLBACK_ARTIFACT_ROOTS}",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "run_id,run_type,traffic_profile_id,baseline_concurrency,surge_concurrency,baseline_duration_seconds,surge_duration_seconds,trace_request_count,scale_from,scale_to,scale_delay_seconds" > "${OUT_DIR}/comparison_key.csv"
echo "${RUN_ID},${RUN_TYPE},${TRAFFIC_PROFILE_ID},${BASELINE_CONCURRENCY},${SURGE_CONCURRENCY},${BASELINE_DURATION_SECONDS},${SURGE_DURATION_SECONDS},${TRACE_REQUEST_COUNT},${SCALE_FROM},${SCALE_TO},${SCALE_DELAY_SECONDS}" >> "${OUT_DIR}/comparison_key.csv"

reset_to_baseline() {
  if [[ "${SCALE_PATCH_TARGET}" == "dgdsa" || "${SCALE_PATCH_TARGET}" == "both" ]]; then
    echo "Resetting ${DGDSA_NAME} to replicas=${SCALE_FROM}"
    kubectl patch dgdsa -n "${NAMESPACE}" "${DGDSA_NAME}" \
      --type=merge -p "{\"spec\":{\"replicas\":${SCALE_FROM}}}"
  fi
  if [[ "${PATCH_DGD_DIRECT}" == "1" ]]; then
    echo "Patching DGD ${DGD_NAME}/${DGD_SERVICE_KEY} to replicas=${SCALE_FROM}"
    kubectl patch dgd -n "${NAMESPACE}" "${DGD_NAME}" \
      --type=merge -p "{\"spec\":{\"services\":{\"${DGD_SERVICE_KEY}\":{\"replicas\":${SCALE_FROM}}}}}"
  fi
  if [[ "${PATCH_GROVE_DIRECT}" == "1" || "${SCALE_PATCH_TARGET}" == "grove" || "${SCALE_PATCH_TARGET}" == "both" ]]; then
    echo "Patching Grove ${GROVE_PCSG_NAME} to replicas=${SCALE_FROM}"
    kubectl patch podcliquescalinggroup -n "${NAMESPACE}" "${GROVE_PCSG_NAME}" \
      --type=merge -p "{\"spec\":{\"replicas\":${SCALE_FROM}}}"
  fi
  if [[ "${WAIT_FOR_DGD_READY}" == "1" ]]; then
    echo "Waiting for ${DGD_NAME} to be Ready at baseline"
    if ! kubectl wait -n "${NAMESPACE}" --for=condition=Ready "dgd/${DGD_NAME}" --timeout="${WAIT_BASELINE_TIMEOUT:-45m}"; then
      echo "DGD did not report Ready; falling back to source pod readiness"
    fi
  fi
  wait_for_ready_source_pods "${SCALE_FROM}" "${WAIT_BASELINE_TIMEOUT:-45m}"
}

wait_for_ready_source_pods() {
  local expected_replicas timeout timeout_seconds deadline ready_count expected_pods
  expected_replicas="$1"
  timeout="$2"
  expected_pods=$((expected_replicas * SOURCE_PODS_PER_REPLICA))
  case "${timeout}" in
    *h) timeout_seconds=$((${timeout%h} * 3600)) ;;
    *m) timeout_seconds=$((${timeout%m} * 60)) ;;
    *s) timeout_seconds=${timeout%s} ;;
    *) timeout_seconds=${timeout} ;;
  esac
  deadline=$((SECONDS + timeout_seconds))
  echo "Waiting for ${expected_pods} ready scaled pod(s) matching /${SCALED_POD_NAME_REGEX}/ at baseline"
  while (( SECONDS < deadline )); do
    ready_count="$(
      kubectl get pods -n "${NAMESPACE}" -l "${APP_LABEL}" -o json 2>/dev/null \
        | jq -r --arg pod_regex "${SCALED_POD_NAME_REGEX}" '
            [
              .items[]
              | select((.metadata.name | test($pod_regex)) and (.status.phase == "Running"))
              | select(((.status.containerStatuses // [])[0].ready // false) == true)
            ] | length'
    )"
    if [[ "${ready_count}" -ge "${expected_pods}" ]]; then
      echo "Ready source pods: ${ready_count}/${expected_pods}"
      return 0
    fi
    echo "Ready scaled pods: ${ready_count}/${expected_pods}; sleeping 10s"
    sleep 10
  done
  echo "Timed out waiting for scaled pod readiness" >&2
  return 1
}

clean_mx_metadata() {
  if [[ "${RUN_TYPE}" != "mx" || "${CLEAN_MX_METADATA}" != "1" ]]; then
    return
  fi
  echo "Deleting existing ModelMetadata records for a clean MX baseline"
  kubectl delete modelmetadata -n "${NAMESPACE}" --all --ignore-not-found=true
}

find_frontend_pod() {
  kubectl get pods -n "${NAMESPACE}" -l "${APP_LABEL}" -o json \
    | jq -r '.items[] | select(.metadata.name | test("frontend")) | .metadata.name' \
    | head -1
}

copy_aiperf_artifact_file() {
  local pod result_dir file dest_dir remote root stripped tmp_file
  pod="$1"
  result_dir="$2"
  file="$3"
  dest_dir="$4"
  tmp_file="${dest_dir}/${file}.tmp"

  for remote in "${result_dir%/}/${file}"; do
    rm -f "${tmp_file}"
    if kubectl exec -n "${NAMESPACE}" "${pod}" -- cat "${remote}" > "${tmp_file}" 2>/dev/null; then
      mv "${tmp_file}" "${dest_dir}/${file}"
      return 0
    fi
  done

  read -r -a roots <<< "${AIPERF_FALLBACK_ARTIFACT_ROOTS}"
  for root in "${roots[@]}"; do
    if [[ "${result_dir}" == /model-cache/* ]]; then
      stripped="${result_dir#/model-cache/}"
    elif [[ "${result_dir}" == /models/* ]]; then
      stripped="${result_dir#/models/}"
    else
      stripped="${result_dir#/}"
    fi
    remote="${root%/}/${stripped}/${file}"
    rm -f "${tmp_file}"
    if kubectl exec -n "${NAMESPACE}" "${pod}" -- cat "${remote}" > "${tmp_file}" 2>/dev/null; then
      mv "${tmp_file}" "${dest_dir}/${file}"
      return 0
    fi
  done

  rm -f "${tmp_file}"
  return 1
}

copy_aiperf_artifacts() {
  local aiperf_log_file fallback_pod result_dir safe_name dest_dir file copied missing_count result_list
  aiperf_log_file="$1"
  fallback_pod="$(find_frontend_pod || true)"
  if [[ -z "${fallback_pod}" ]]; then
    echo "No frontend pod found; skipping AIPerf artifact copy" >&2
    return
  fi

  mkdir -p "${OUT_DIR}/aiperf_artifacts"
  result_list="${OUT_DIR}/aiperf_result_dirs.txt"
  if ! grep -E 'Results: ' "${aiperf_log_file}" \
    | awk '{print $NF}' \
    | sort -u > "${result_list}"; then
    echo "No AIPerf result directory found in ${aiperf_log_file}" | tee -a "${OUT_DIR}/aiperf_artifact_copy.log"
    return
  fi

  while IFS= read -r result_dir; do
        [[ -z "${result_dir}" ]] && continue
        safe_name="$(echo "${result_dir}" | sed -E 's#^/##; s#[^A-Za-z0-9._-]+#_#g')"
        dest_dir="${OUT_DIR}/aiperf_artifacts/${safe_name}"
        mkdir -p "${dest_dir}"

        copied=0
        missing_count=0
        read -r -a artifact_files <<< "${AIPERF_ARTIFACT_FILES}"
        for file in "${artifact_files[@]}"; do
          [[ -z "${file}" ]] && continue
          if copy_aiperf_artifact_file "${fallback_pod}" "${result_dir}" "${file}" "${dest_dir}"; then
            copied=$((copied + 1))
          else
            missing_count=$((missing_count + 1))
          fi
        done

        echo "Copied ${copied} AIPerf artifact(s) from ${result_dir} via ${fallback_pod}; missing ${missing_count}" \
          | tee -a "${OUT_DIR}/aiperf_artifact_copy.log"
  done < "${result_list}"
}

sample_once() {
  local ts frontend_pod metrics_file
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if [[ "${SCALE_PATCH_TARGET}" == "grove" ]]; then
    kubectl get podcliquescalinggroup -n "${NAMESPACE}" "${GROVE_PCSG_NAME}" -o json 2>/dev/null \
      | jq -r --arg ts "${ts}" '[$ts, (.spec.replicas // ""), (.status.replicas // ""), ""] | @csv' \
      >> "${SAMPLE_DIR}/replica_timeline.csv" || true
  else
    kubectl get dgdsa -n "${NAMESPACE}" "${DGDSA_NAME}" -o json 2>/dev/null \
      | jq -r --arg ts "${ts}" '[$ts, (.spec.replicas // ""), (.status.replicas // ""), (.status.selector // "")] | @csv' \
      >> "${SAMPLE_DIR}/replica_timeline.csv" || true
  fi

  kubectl get pods -n "${NAMESPACE}" -l "${APP_LABEL}" -o json 2>/dev/null \
    | jq -r --arg ts "${ts}" '
      def cond($name): (.status.conditions // [] | map(select(.type == $name)) | .[0] // {});
      .items[] |
      [
        $ts,
        .metadata.name,
        .status.phase,
        (.metadata.labels["grove.io/podcliquescalinggroup-replica-index"] // ""),
        (.metadata.creationTimestamp // ""),
        (.status.startTime // ""),
        ((.status.containerStatuses // [])[0].ready // ""),
        ((.status.containerStatuses // [])[0].restartCount // ""),
        (cond("Ready").lastTransitionTime // ""),
        (.spec.nodeName // "")
      ] | @csv' \
    >> "${SAMPLE_DIR}/pod_samples.csv" || true

  frontend_pod="$(find_frontend_pod || true)"
  if [[ -n "${frontend_pod}" ]]; then
    metrics_file="${RAW_METRICS_DIR}/${ts//[:]/-}.prom"
    kubectl exec -n "${NAMESPACE}" "${frontend_pod}" -- \
      sh -lc 'curl -sf http://127.0.0.1:8000/metrics' \
      > "${metrics_file}" 2>/dev/null || true
  fi

  if [[ "${SAMPLE_GPU}" == "1" ]]; then
    kubectl get pods -n "${NAMESPACE}" -l "${APP_LABEL}" -o json 2>/dev/null \
      | jq -r --arg pod_regex "${SCALED_POD_NAME_REGEX}" '.items[] | select(.metadata.name | test($pod_regex)) | .metadata.name' \
      | while IFS= read -r pod; do
          [[ -z "${pod}" ]] && continue
          kubectl exec -n "${NAMESPACE}" "${pod}" -- sh -lc \
            'command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true' \
            2>/dev/null \
            | awk -v ts="${ts}" -v pod="${pod}" 'NF {print ts "," pod "," $0}' \
            >> "${SAMPLE_DIR}/gpu.csv" || true
        done
  fi
}

sampler_loop() {
  echo "timestamp,spec_replicas,status_replicas,status_selector" > "${SAMPLE_DIR}/replica_timeline.csv"
  echo "timestamp,pod,phase,replica_index,created,start_time,container_ready,restarts,ready_transition,node" > "${SAMPLE_DIR}/pod_samples.csv"
  echo "timestamp,pod,gpu_index,gpu_util_percent,memory_used_mib,memory_total_mib" > "${SAMPLE_DIR}/gpu.csv"
  touch "${STOP_FILE}"
  while [[ -e "${STOP_FILE}" ]]; do
    sample_once
    sleep "${SAMPLE_INTERVAL_SECONDS}"
  done
}

start_aiperf() {
  local rendered_manifest
  rendered_manifest="${OUT_DIR}/aiperf.rendered.yaml"
  cp "${AIPERF_MANIFEST}" "${rendered_manifest}"
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  FRONTEND_SERVICE_OVERRIDE="${FRONTEND_SERVICE}" \
  SOURCE_METRICS_OVERRIDE="http://${SOURCE_SERVICE}/metrics" \
  BASELINE_CONCURRENCY_OVERRIDE="${BASELINE_CONCURRENCY}" \
  SURGE_CONCURRENCY_OVERRIDE="${SURGE_CONCURRENCY}" \
  BASELINE_DURATION_SECONDS_OVERRIDE="${BASELINE_DURATION_SECONDS}" \
  SURGE_DURATION_SECONDS_OVERRIDE="${SURGE_DURATION_SECONDS}" \
  CONCURRENCY_RAMP_DURATION_SECONDS_OVERRIDE="${CONCURRENCY_RAMP_DURATION_SECONDS}" \
  BENCHMARK_GRACE_PERIOD_SECONDS_OVERRIDE="${BENCHMARK_GRACE_PERIOD_SECONDS}" \
  PROGRESS_INTERVAL_SECONDS_OVERRIDE="${PROGRESS_INTERVAL_SECONDS}" \
  TRACE_REQUEST_COUNT_OVERRIDE="${TRACE_REQUEST_COUNT}" \
  TRACE_MAX_ISL_OVERRIDE="${TRACE_MAX_ISL}" \
  TRACE_MAX_OSL_OVERRIDE="${TRACE_MAX_OSL}" \
  perl -0pi -e '
    s/(- name: TARGET_MODEL\n\s+value: ).*/$1$ENV{MODEL_NAME_OVERRIDE}/g;
    s/(- name: ENDPOINT\n\s+value: ).*/$1$ENV{FRONTEND_SERVICE_OVERRIDE}/g;
    s/(- name: AIPERF_SERVER_METRICS_URLS\n\s+value: ).*/$1"$ENV{SOURCE_METRICS_OVERRIDE}"/g;
    s/(- name: BASELINE_CONCURRENCY\n\s+value: ).*/$1"$ENV{BASELINE_CONCURRENCY_OVERRIDE}"/g;
    s/(- name: SURGE_CONCURRENCY\n\s+value: ).*/$1"$ENV{SURGE_CONCURRENCY_OVERRIDE}"/g;
    s/(- name: BASELINE_DURATION_SECONDS\n\s+value: ).*/$1"$ENV{BASELINE_DURATION_SECONDS_OVERRIDE}"/g;
    s/(- name: SURGE_DURATION_SECONDS\n\s+value: ).*/$1"$ENV{SURGE_DURATION_SECONDS_OVERRIDE}"/g;
    s/(- name: CONCURRENCY_RAMP_DURATION_SECONDS\n\s+value: ).*/$1"$ENV{CONCURRENCY_RAMP_DURATION_SECONDS_OVERRIDE}"/g;
    s/(- name: BENCHMARK_GRACE_PERIOD_SECONDS\n\s+value: ).*/$1"$ENV{BENCHMARK_GRACE_PERIOD_SECONDS_OVERRIDE}"/g;
    s/(- name: PROGRESS_INTERVAL_SECONDS\n\s+value: ).*/$1"$ENV{PROGRESS_INTERVAL_SECONDS_OVERRIDE}"/g;
    s/(- name: TRACE_REQUEST_COUNT\n\s+value: ).*/$1"$ENV{TRACE_REQUEST_COUNT_OVERRIDE}"/g;
    s/(- name: TRACE_MAX_ISL\n\s+value: ).*/$1"$ENV{TRACE_MAX_ISL_OVERRIDE}"/g;
    s/(- name: TRACE_MAX_OSL\n\s+value: ).*/$1"$ENV{TRACE_MAX_OSL_OVERRIDE}"/g;
  ' "${rendered_manifest}"

  echo "Creating AIPerf job from rendered manifest ${rendered_manifest}" >&2
  local job_ref job_name
  job_ref="$(kubectl create -n "${NAMESPACE}" -f "${rendered_manifest}" -o name)"
  job_name="${job_ref#job.batch/}"
  echo "${job_ref}" > "${OUT_DIR}/aiperf_job_ref.txt"
  echo "${job_name}" > "${OUT_DIR}/aiperf_job_name.txt"
  kubectl get -n "${NAMESPACE}" "${job_ref}" -o yaml > "${OUT_DIR}/aiperf_job.yaml" 2>&1 || true
  echo "${job_ref}"
}

wait_for_aiperf_marker() {
  local job_name timeout_seconds deadline now
  job_name="$1"
  case "${AIPERF_MARKER_TIMEOUT}" in
    *h) timeout_seconds=$((${AIPERF_MARKER_TIMEOUT%h} * 3600)) ;;
    *m) timeout_seconds=$((${AIPERF_MARKER_TIMEOUT%m} * 60)) ;;
    *s) timeout_seconds=${AIPERF_MARKER_TIMEOUT%s} ;;
    *) timeout_seconds=${AIPERF_MARKER_TIMEOUT} ;;
  esac
  deadline=$((SECONDS + timeout_seconds))
  echo "Waiting for AIPerf scale marker /${AIPERF_SCALE_MARKER}/, timeout ${AIPERF_MARKER_TIMEOUT}"
  while (( SECONDS < deadline )); do
    if kubectl logs -n "${NAMESPACE}" "job/${job_name}" --tail=200 2>/dev/null \
      | grep -E "${AIPERF_SCALE_MARKER}" > "${OUT_DIR}/aiperf_scale_marker.log"; then
      now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "${now}" > "${OUT_DIR}/scale_marker_time.txt"
      echo "${now}" > "${OUT_DIR}/traffic_marker_time.txt"
      echo "AIPerf marker observed at ${now}"
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for AIPerf scale marker; scaling from job creation fallback" >&2
  date -u +%Y-%m-%dT%H:%M:%SZ > "${OUT_DIR}/traffic_marker_timeout.txt"
  return 1
}

patch_scale_up() {
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cat > "${OUT_DIR}/scale_event.json" <<EOF
{
  "timestamp": "${ts}",
  "dgdsa_name": "${DGDSA_NAME}",
  "from": ${SCALE_FROM},
  "to": ${SCALE_TO},
  "delay_after_traffic_marker_seconds": ${SCALE_DELAY_SECONDS},
  "scale_trigger_mode": "${SCALE_TRIGGER_MODE}",
  "aiperf_scale_marker": "${AIPERF_SCALE_MARKER}"
}
EOF
  echo "Scaling ${DGDSA_NAME} to replicas=${SCALE_TO} at ${ts}"
  if [[ "${SCALE_PATCH_TARGET}" == "dgdsa" || "${SCALE_PATCH_TARGET}" == "both" ]]; then
    kubectl patch dgdsa -n "${NAMESPACE}" "${DGDSA_NAME}" \
      --type=merge -p "{\"spec\":{\"replicas\":${SCALE_TO}}}"
  fi
  if [[ "${PATCH_DGD_DIRECT}" == "1" ]]; then
    echo "Patching DGD ${DGD_NAME}/${DGD_SERVICE_KEY} to replicas=${SCALE_TO} at ${ts}"
    kubectl patch dgd -n "${NAMESPACE}" "${DGD_NAME}" \
      --type=merge -p "{\"spec\":{\"services\":{\"${DGD_SERVICE_KEY}\":{\"replicas\":${SCALE_TO}}}}}"
  fi
  if [[ "${PATCH_GROVE_DIRECT}" == "1" || "${SCALE_PATCH_TARGET}" == "grove" || "${SCALE_PATCH_TARGET}" == "both" ]]; then
    echo "Patching Grove ${GROVE_PCSG_NAME} to replicas=${SCALE_TO} at ${ts}"
    kubectl patch podcliquescalinggroup -n "${NAMESPACE}" "${GROVE_PCSG_NAME}" \
      --type=merge -p "{\"spec\":{\"replicas\":${SCALE_TO}}}"
  fi
}

stop_sampler() {
  rm -f "${STOP_FILE}"
  if [[ -n "${SAMPLER_PID:-}" ]]; then
    wait "${SAMPLER_PID}" 2>/dev/null || true
  fi
}

trap stop_sampler EXIT

echo "Autoscale collection run: ${RUN_ID}"
echo "Output: ${OUT_DIR}"
clean_mx_metadata
reset_to_baseline

sampler_loop &
SAMPLER_PID="$!"

AIPERF_JOB_REF="$(start_aiperf)"
JOB_NAME="$(cat "${OUT_DIR}/aiperf_job_name.txt")"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUT_DIR}/aiperf_job_create_time.txt"
cp "${OUT_DIR}/aiperf_job_create_time.txt" "${OUT_DIR}/traffic_marker_time.txt"

if [[ "${SCALE_TRIGGER_MODE}" == "log_marker" ]]; then
  wait_for_aiperf_marker "${JOB_NAME}" || true
  echo "Waiting ${SCALE_DELAY_SECONDS}s after traffic marker before controlled scale-up"
  sleep "${SCALE_DELAY_SECONDS}"
else
  ramp_delay="${CONCURRENCY_RAMP_DURATION_SECONDS}"
  if [[ "${ramp_delay}" == "0" ]]; then
    ramp_delay="${BASELINE_DURATION_SECONDS}"
  fi
  trigger_delay=$((ramp_delay + SCALE_DELAY_SECONDS))
  python3 - "${OUT_DIR}/aiperf_job_create_time.txt" "${ramp_delay}" "${OUT_DIR}/scale_marker_time.txt" <<'PY'
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

created = Path(sys.argv[1]).read_text().strip()
ramp_seconds = int(sys.argv[2])
out = Path(sys.argv[3])
start = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
out.write_text((start + timedelta(seconds=ramp_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ") + "\n")
PY
  echo "Waiting ${trigger_delay}s after AIPerf job creation before controlled scale-up (${ramp_delay}s ramp + ${SCALE_DELAY_SECONDS}s delay)"
  sleep "${trigger_delay}"
fi
patch_scale_up

echo "Waiting for ${AIPERF_JOB_REF} to complete, timeout ${AIPERF_TIMEOUT}"
if kubectl wait --for=condition=complete -n "${NAMESPACE}" "${AIPERF_JOB_REF}" --timeout="${AIPERF_TIMEOUT}"; then
  echo "complete" > "${OUT_DIR}/aiperf_status.txt"
else
  echo "not_complete" > "${OUT_DIR}/aiperf_status.txt"
fi

AIPERF_LOG_FILE="${OUT_DIR}/logs/${JOB_NAME}.log"
kubectl logs -n "${NAMESPACE}" "job/${JOB_NAME}" --timestamps > "${AIPERF_LOG_FILE}" 2>&1 || true

copy_aiperf_artifacts "${AIPERF_LOG_FILE}"

stop_sampler

echo "Running final artifact collection"
RUN_TYPE="${RUN_TYPE}" \
APP_LABEL="${APP_LABEL}" \
DGD_NAME="${DGD_NAME}" \
DGDSA_NAME="${DGDSA_NAME}" \
HPA_NAME="${HPA_NAME}" \
MODEL_NAME="${MODEL_NAME}" \
OUT_DIR="${OUT_DIR}" \
NAMESPACE="${NAMESPACE}" \
"${SCRIPT_DIR}/collect_demo_metrics.sh"

if [[ -d "${OUT_DIR}/aiperf_artifacts" ]]; then
  python3 "${SCRIPT_DIR}/build_time_series.py" "${OUT_DIR}" --slice-seconds "${SLICE_SECONDS}" || true
fi

cat >> "${OUT_DIR}/summary.md" <<EOF

## Controlled Scale Event

- Traffic profile: ${TRAFFIC_PROFILE_ID}
- Baseline concurrency: ${BASELINE_CONCURRENCY}
- Surge concurrency: ${SURGE_CONCURRENCY}
- Baseline duration seconds: ${BASELINE_DURATION_SECONDS}
- Surge duration seconds: ${SURGE_DURATION_SECONDS}
- Concurrency ramp duration seconds: ${CONCURRENCY_RAMP_DURATION_SECONDS}
- Benchmark grace period seconds: ${BENCHMARK_GRACE_PERIOD_SECONDS}
- AIPerf progress interval seconds: ${PROGRESS_INTERVAL_SECONDS}
- Trace request count: ${TRACE_REQUEST_COUNT}
- Trace max ISL/OSL: ${TRACE_MAX_ISL}/${TRACE_MAX_OSL}
- Scale from/to: ${SCALE_FROM} -> ${SCALE_TO}
- Direct Grove patch: ${PATCH_GROVE_DIRECT}
- Scale marker: ${AIPERF_SCALE_MARKER}
- Scale delay after traffic marker: ${SCALE_DELAY_SECONDS}s
- Scale event file: scale_event.json
- Live samples: samples/replica_timeline.csv, samples/pod_samples.csv, samples/gpu.csv
- Time series: time_series_${SLICE_SECONDS}s.csv
- Worker routing time series: worker_routing_${SLICE_SECONDS}s.csv
- Worker selection time series: worker_selection_${SLICE_SECONDS}s.csv
- Frontend metric snapshots: samples/frontend_metrics/*.prom
EOF

echo "Run artifacts written to ${OUT_DIR}"
