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
TRAINER=${TRAINER:-fsdp2}   # fsdp2 | deepspeed | jax
DST=${DST:-replicate}

export PYTHONPATH=${PYTHONPATH:-$(cd "$(dirname "$0")" && pwd)}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING}
export MX_NCCL_REFIT_GROUP_TIMEOUT_S=${MX_NCCL_REFIT_GROUP_TIMEOUT_S:-900}
mkdir -p "$OUT"
echo "run=$RUN model=$MODEL trainer=$TRAINER dst=$DST trainers=$T generators=$G rounds=$ROUNDS out=$OUT"

( CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((G-1))) \
  python3 -m mx_m2n_e2e.generator --model-dir "$MODEL" --endpoint "$MX_ENDPOINT" \
    --model-name "$RUN" --run-id "$RUN" --trainers "$T" --generators "$G" \
    --rounds "$ROUNDS" --dst-layout "$DST" ${DIFF:+--diff-checkpoint} --out "$OUT" > "$OUT/generator.log" 2>&1
  echo $? > "$OUT/generator.rc" ) &

sleep 5
for rank in $(seq 0 $((T-1))); do
  if [ "$TRAINER" = jax ]; then
    # JAX has its own coordinator and no torch.distributed. Preallocation is
    # off because XLA otherwise claims most of the device and leaves NCCL's
    # buffers nowhere to live.
    ( CUDA_VISIBLE_DEVICES=$((G+rank)) RANK=$rank \
      MASTER_ADDR=127.0.0.1 JAX_PORT=${JAX_PORT:-29566} \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python3 -m mx_m2n_e2e.jax_trainer --model-dir "$MODEL" --endpoint "$MX_ENDPOINT" \
        --model-name "$RUN" --run-id "$RUN" --trainers "$T" --generators "$G" \
        --rounds "$ROUNDS" --dst-layout "$DST" \
        --out "$OUT" > "$OUT/trainer$rank.log" 2>&1
      echo $? > "$OUT/trainer$rank.rc" ) &
  else
    # LOCAL_RANK and WORLD_SIZE are DeepSpeed's; each process pins one visible
    # device, so its local rank is always zero.
    ( CUDA_VISIBLE_DEVICES=$((G+rank)) LOCAL_DEVICE=0 RANK=$rank \
      LOCAL_RANK=0 WORLD_SIZE=$T \
      MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT:-29555} \
      python3 -m mx_m2n_e2e.trainer --model-dir "$MODEL" --endpoint "$MX_ENDPOINT" \
        --model-name "$RUN" --run-id "$RUN" --trainers "$T" --generators "$G" \
        --rounds "$ROUNDS" --backend "$TRAINER" --dst-layout "$DST" \
        --out "$OUT" > "$OUT/trainer$rank.log" 2>&1
      echo $? > "$OUT/trainer$rank.rc" ) &
  fi
done
wait

echo "=== generator rc $(cat "$OUT/generator.rc" 2>/dev/null) ==="
grep -E "^\[gen\]|E2E|FAILED" "$OUT/generator.log" | grep -v "workers:" | tail -12
echo "=== trainer rcs $(cat "$OUT"/trainer*.rc 2>/dev/null | tr '\n' ' ') ==="
grep -hE "^\[trainer 0\]" "$OUT/trainer0.log" | tail -8
