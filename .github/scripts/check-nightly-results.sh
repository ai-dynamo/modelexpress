#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
phase=${1:?Expected release or final}
case "$phase" in release|final) ;; *) exit 1 ;; esac
case "${RELEASE}:${RUN_TESTS}" in
  true:true|true:false|false:true|false:false) ;;
  *) echo '::error::Missing or invalid nightly mode; cannot determine required jobs'; exit 1 ;;
esac

printf '%s\n' "$RESULTS" | jq -e 'type == "object"' >/dev/null
printf '%s\n' "$RESULTS" | jq .
failed=$(printf '%s\n' "$RESULTS" | jq -r 'to_entries[] | select(.value.result != "success" and .value.result != "skipped") | .key')
mandatory=(preflight)
if [ "$RUN_TESTS" = true ]; then
  mandatory+=(core-tests inference-tests rl-tests)
fi
if [ "$RELEASE" = true ]; then
  # Core validation is required even when optional GPU suites are disabled.
  mandatory+=(core-tests build-server-image assemble-server-manifest build-wheels stage-crates package-helm-chart stage-helm-ngc stage-containers-ngc compliance)
  if [ "$phase" = final ] && [ "${SKIP_GITLAB:-false}" != true ]; then
    mandatory+=(trigger-gitlab-release)
  fi
fi
for job in "${mandatory[@]}"; do
  result=$(printf '%s\n' "$RESULTS" | jq -r --arg job "$job" '.[$job].result // "MISSING"')
  if [ "$result" != success ]; then
    printf '::error::Required nightly job %s has result %s\n' "$job" "$result"
    failed="${failed} ${job}"
  fi
done
if [ -n "$failed" ]; then
  printf '::error::Nightly jobs failed: %s\n' "$failed"
  exit 1
fi
echo 'All required nightly jobs succeeded.'
