#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch the standalone M2N microbench. GPU mode uses one local GPU per
# participant and a process-level timeout in addition to per-lane deadlines.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:?usage: run_microbench.sh <mock|gpu> [microbench arguments]}"
shift

: "${M2N_MB_TIMEOUT_S:=180}"
: "${M2N_MB_COHORT_TIMEOUT_S:=1800}"
: "${M2N_MB_PARTITIONS:=1}"
: "${M2N_MB_FANOUT:=1}"
: "${M2N_MB_OUT:=}"

case "$MODE" in
    mock)
        exec python3 "$HERE/m2n_microbench.py" --mode mock "$@"
        ;;
    gpu)
        IFS=, read -r -a partitions <<<"$M2N_MB_PARTITIONS"
        IFS=, read -r -a fanouts <<<"$M2N_MB_FANOUT"
        if [[ "${#partitions[@]}" -ne 1 || "${#fanouts[@]}" -ne 1 ]]; then
            echo "GPU launcher takes one partition/fanout pair; loop externally for a sweep." >&2
            exit 2
        fi
        world_size=$((partitions[0] + fanouts[0]))
        output_args=()
        if [[ -n "$M2N_MB_OUT" ]]; then
            output_args=(--json-out "$M2N_MB_OUT")
        fi
        exec timeout --preserve-status "${M2N_MB_COHORT_TIMEOUT_S}s" \
            torchrun --standalone --nproc_per_node="$world_size" \
            "$HERE/m2n_microbench.py" \
            --mode gpu \
            --source-partitions "${partitions[0]}" \
            --destination-fanout "${fanouts[0]}" \
            --timeout-s "$M2N_MB_TIMEOUT_S" \
            "${output_args[@]}" \
            "$@"
        ;;
    *)
        echo "mode must be mock or gpu, got $MODE" >&2
        exit 2
        ;;
esac
