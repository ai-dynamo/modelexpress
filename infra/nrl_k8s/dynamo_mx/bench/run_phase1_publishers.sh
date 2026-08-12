#!/usr/bin/env bash
set -euo pipefail

: "${BASE_RANK:?set BASE_RANK=0 or 4}"
: "${RENDEZVOUS_NAME:?set the shared Phase-1 identity}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
WIRE_DTYPE="${WIRE_DTYPE:-bf16}"
MX_SERVER="modelexpress-server.kavin.svc.cluster.local:8001"
ROOT="/mnt/rl-workspace/kavink/reshard_phase1"
mkdir -p "${ROOT}"

for local_rank in 0 1 2 3; do
  global_rank=$((BASE_RANK + local_rank))
  CUDA_VISIBLE_DEVICES="${local_rank}" \
    python3 /tmp/reshard_publisher_sharded.py \
      --model "${MODEL}" \
      --rendezvous-name "${RENDEZVOUS_NAME}" \
      --rank "${global_rank}" \
      --world-size 8 \
      --device 0 \
      --wire-dtype "${WIRE_DTYPE}" \
      --mx-server "${MX_SERVER}" \
      --listen-port "$((7200 + local_rank))" \
      --ready-file "${ROOT}/pub.r${global_rank}.ready" \
      --stop-file "${ROOT}/pub.stop" \
      >"/tmp/pub.r${global_rank}.log" 2>&1 &
  echo "started global_rank=${global_rank} pid=$!"
done

wait
