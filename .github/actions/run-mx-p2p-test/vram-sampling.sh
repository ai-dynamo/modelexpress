#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

sample_peak_vram() {
  local label="$1"
  local selector="$2"
  local container="$3"
  local pod_var="$4"
  local peak_var="$5"
  local samples_var="$6"
  local current_var="${7:-}"

  local pod="${!pod_var}"
  local peak="${!peak_var}"
  local samples="${!samples_var}"

  if [ -z "${pod}" ]; then
    pod="$(kubectl get pods -n "${NAMESPACE}" \
      -l "${selector}" \
      -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  fi
  if [ -z "${pod}" ]; then
    return 0
  fi

  local sample output err err_file
  err_file="$(mktemp)"
  # Query process memory instead of whole-GPU memory.used. The CI a100a pool
  # uses MIG slices, where full-device memory.used is blocked from inside the
  # container and reports "[Insufficient Permissions]".
  output="$(kubectl exec -n "${NAMESPACE}" "${pod}" -c "${container}" -- \
      nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>"${err_file}" \
    || true)"
  err="$(tr '\n' ' ' < "${err_file}")"
  rm -f "${err_file}"

  sample="$(printf '%s\n' "${output}" \
    | awk 'BEGIN {sum=0; seen=0} NF {gsub(/[^0-9]/, "", $1); if ($1 ~ /^[0-9]+$/) {sum += $1+0; seen=1}} END {if (seen) print sum}')"
  if [[ ! "${sample}" =~ ^[0-9]+$ ]]; then
    echo "${label} VRAM sample unavailable for pod ${pod}: stdout='${output:0:120}' stderr='${err:0:120}'"
    return 0
  fi

  samples=$((samples + 1))
  if [ "${sample}" -gt "${peak}" ]; then
    peak="${sample}"
  fi

  printf -v "${pod_var}" '%s' "${pod}"
  printf -v "${peak_var}" '%s' "${peak}"
  printf -v "${samples_var}" '%s' "${samples}"
  if [ -n "${current_var}" ]; then
    printf -v "${current_var}" '%s' "${sample}"
  fi
  echo "${label} peak VRAM sample=${sample} MiB max=${peak} MiB"
}

wait_for_health() {
  local label="$1"
  local selector="$2"
  local container="$3"
  local port="$4"
  local pod_var="$5"
  local peak_var="$6"
  local samples_var="$7"
  local final_var="$8"
  local timeout="$9"
  local timeout_detail="${10:-}"

  local deadline failed pod
  deadline=$((SECONDS + timeout))
  while [ $SECONDS -lt $deadline ]; do
    pod="$(kubectl get pods -n "${NAMESPACE}" \
      -l "${selector}" \
      -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
    printf -v "${pod_var}" '%s' "${pod}"

    sample_peak_vram "${label}" "${selector}" "${container}" \
      "${pod_var}" "${peak_var}" "${samples_var}" "${final_var}"
    if [ -n "${pod}" ] && kubectl exec -n "${NAMESPACE}" "${pod}" -c "${container}" -- \
         curl -sf -m 5 "http://localhost:${port}/health" >/dev/null 2>&1; then
      echo "${label} ${pod} /health returned 200."
      break
    fi

    failed="$(kubectl get pods -n "${NAMESPACE}" -l "${selector}" --no-headers 2>/dev/null \
      | awk '$3 ~ /Error|CrashLoopBackOff|OOMKilled|ImagePullBackOff|ErrImagePull/' || true)"
    if [ -n "${failed}" ]; then
      echo "ERROR: ${label} pod entered failure state before /health returned 200: ${failed}"
      exit 1
    fi

    echo "${label} /health not ready yet (pod=${pod:-<not-created>})..."
    sleep 5
  done

  sample_peak_vram "${label}" "${selector}" "${container}" \
    "${pod_var}" "${peak_var}" "${samples_var}" "${final_var}"
  pod="${!pod_var}"
  if [ -z "${pod}" ] || ! kubectl exec -n "${NAMESPACE}" "${pod}" -c "${container}" -- \
       curl -sf -m 5 "http://localhost:${port}/health" >/dev/null 2>&1; then
    echo "ERROR: ${label} /health did not return 200 within ${timeout}s${timeout_detail}"
    exit 1
  fi
  if [ "${!samples_var}" -eq 0 ]; then
    echo "ERROR: sampled no ${label} GPU memory values with nvidia-smi"
    exit 1
  fi
}

write_vram_measurements() {
  local path="$1"
  local source_peak_mib="$2"
  local source_final_mib="$3"
  local target_peak_mib="$4"
  local target_final_mib="$5"
  local tolerance_percent="$6"

  python3 - "${path}" "${source_peak_mib}" "${source_final_mib}" \
      "${target_peak_mib}" "${target_final_mib}" "${tolerance_percent}" <<'PY'
import json
import sys

path, source_peak, source_final, target_peak, target_final, tolerance = sys.argv[1:]
measurements = {
    "source": {
        "peak_mib": int(source_peak),
        "final_mib": int(source_final),
    },
    "target": {
        "peak_mib": int(target_peak),
        "final_mib": int(target_final),
    },
    "tolerance_percent": float(tolerance),
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(measurements, f, indent=2)
    f.write("\n")
PY
}
