#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start one bench_collective_refit.py process per rank against a live
# ModelExpress server, wait for all of them, then aggregate the per-rank JSON.
#
# Every rank is its own process because that is how a refit deployment is
# actually laid out, and because a single process holding every device would
# hide the per-rank control-plane cost this is here to measure.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${BENCH_TRAINERS:=2}"
: "${BENCH_GENERATORS:=2}"
: "${BENCH_ROUNDS:=5}"
: "${BENCH_PARAMS:=32}"
: "${BENCH_ROWS:=8192}"
: "${BENCH_COLS:=8192}"
: "${BENCH_PARTITIONS:=1}"
: "${BENCH_DTYPE:=bfloat16}"
: "${MX_ENDPOINT:=127.0.0.1:8001}"
: "${BENCH_RANK_BASE:=0}"
# How many of the cohort's ranks this host runs. Defaults to all of them, which
# is the single-node case; a cross-node run sets a base and a count per pod and
# aggregates the per-rank JSON afterwards.
: "${BENCH_LOCAL_RANKS:=0}"
BENCH_RUN_ID="${BENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%S)-$$}"
BENCH_OUT="${BENCH_OUT:-/tmp/mxbench/out/$BENCH_RUN_ID}"
export BENCH_TRAINERS BENCH_GENERATORS BENCH_ROUNDS BENCH_PARAMS BENCH_ROWS \
       BENCH_COLS BENCH_PARTITIONS BENCH_DTYPE MX_ENDPOINT BENCH_RUN_ID BENCH_OUT \
       BENCH_RANK_BASE

WORLD=$((BENCH_TRAINERS + BENCH_GENERATORS))
if [ "$BENCH_LOCAL_RANKS" -eq 0 ]; then BENCH_LOCAL_RANKS=$WORLD; fi
LAST=$((BENCH_RANK_BASE + BENCH_LOCAL_RANKS))
mkdir -p "$BENCH_OUT"
echo "run $BENCH_RUN_ID: ${BENCH_TRAINERS}x${BENCH_GENERATORS}, ${BENCH_PARAMS} params of ${BENCH_ROWS}x${BENCH_COLS} ${BENCH_DTYPE}, ${BENCH_ROUNDS} rounds -> $BENCH_OUT"

pids=()
for ((rank = BENCH_RANK_BASE; rank < LAST; rank++)); do
  RANK=$rank python3 "$HERE/bench_collective_refit.py" \
    >"$BENCH_OUT/rank$rank.log" 2>&1 &
  pids+=($!)
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done

# Teardown segfaults after a rank has reported are routine torch/NCCL atexit
# noise, so the run is scored on the JSON each rank wrote, not on exit codes.
if [ "$BENCH_LOCAL_RANKS" -eq "$WORLD" ]; then
  python3 "$HERE/bench_report.py" "$BENCH_OUT" || rc=1
else
  echo "ran ranks $BENCH_RANK_BASE..$((LAST - 1)) of $WORLD; collect $BENCH_OUT/rank*.json from every host and run bench_report.py over the union"
fi
exit $rc
