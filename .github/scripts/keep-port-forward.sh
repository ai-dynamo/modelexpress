#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Restart kubectl port-forward with the same arguments after tunnel termination.
# Pass kubectl arguments after the script name; TERM/INT stops and reaps its child.
set -uo pipefail

child_pid=
stop() {
  trap - TERM INT EXIT
  if [ -n "$child_pid" ]; then
    kill "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  exit 0
}
trap stop TERM INT EXIT

while true; do
  echo "$(date -u +%FT%TZ) Starting kubectl port-forward"
  kubectl "$@" &
  child_pid=$!
  status=0
  wait "$child_pid" || status=$?
  child_pid=
  echo "$(date -u +%FT%TZ) Port-forward exited (${status}); restarting in 2s"
  sleep 2 &
  child_pid=$!
  wait "$child_pid" || true
  child_pid=
done
