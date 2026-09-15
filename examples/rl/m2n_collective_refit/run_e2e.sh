#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One end-to-end refit: an FSDP2-sharded checkpoint into a live vLLM engine.
# Trainers and generators are separate processes on one node; the generator is
# started first because vLLM needs about a minute before it reaches the
# rendezvous, and a trainer that arrives early simply waits there.
set -u

MODEL=${MODEL:?set MODEL to a checkpoint directory}
MX_ENDPOINT=${MX_ENDPOINT:?set MX_ENDPOINT to host:port of the ModelExpress server}
T=${T:-4}
G=${G:-4}
ROUNDS=${ROUNDS:-5}
RUN=${RUN:-e2e-$(date +%s)}
OUT=${OUT:-/work/out/$RUN}
TRAINER=${TRAINER:-fsdp2}

export PYTHONPATH=${PYTHONPATH:-$(cd "$(dirname "$0")" && pwd)}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING}
export MX_NCCL_REFIT_GROUP_TIMEOUT_S=${MX_NCCL_REFIT_GROUP_TIMEOUT_S:-900}
mkdir -p "$OUT"
echo "run=$RUN model=$MODEL trainer=$TRAINER trainers=$T generators=$G rounds=$ROUNDS out=$OUT"

( CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((G-1))) \
  python3 -m mx_m2n_e2e.generator --model-dir "$MODEL" --endpoint "$MX_ENDPOINT" \
    --model-name "$RUN" --run-id "$RUN" --trainers "$T" --generators "$G" \
    --rounds "$ROUNDS" --out "$OUT" > "$OUT/generator.log" 2>&1
  echo $? > "$OUT/generator.rc" ) &

sleep 5
for rank in $(seq 0 $((T-1))); do
  # LOCAL_RANK and WORLD_SIZE are DeepSpeed's; each process pins one visible
  # device, so its local rank is always zero.
  ( CUDA_VISIBLE_DEVICES=$((G+rank)) LOCAL_DEVICE=0 RANK=$rank \
    LOCAL_RANK=0 WORLD_SIZE=$T \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT:-29555} \
    python3 -m mx_m2n_e2e.trainer --model-dir "$MODEL" --endpoint "$MX_ENDPOINT" \
      --model-name "$RUN" --run-id "$RUN" --trainers "$T" --generators "$G" \
      --rounds "$ROUNDS" --backend "$TRAINER" --out "$OUT" > "$OUT/trainer$rank.log" 2>&1
    echo $? > "$OUT/trainer$rank.rc" ) &
done
wait

echo "=== generator rc $(cat "$OUT/generator.rc" 2>/dev/null) ==="
grep -E "^\[gen\]|E2E|FAILED" "$OUT/generator.log" | grep -v "workers:" | tail -12
echo "=== trainer rcs $(cat "$OUT"/trainer*.rc 2>/dev/null | tr '\n' ' ') ==="
grep -hE "^\[trainer 0\]" "$OUT/trainer0.log" | tail -8
