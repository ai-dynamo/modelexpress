#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

run_id=vime-delta-refit
base_version_id=$run_id-v0
model_path=/models
mx_server_url=vime-delta-refit-mx:8101
s3_bucket=delta-weights
s3_uri_prefix=s3://$s3_bucket/$run_id
# This throwaway MinIO has no external endpoint.
s3_endpoint_url=http://vime-delta-refit-minio:9000
s3_region=us-east-1
generation_url=http://vime-delta-refit-generation:8000
discovery_url=http://vime-delta-refit-frontend-admin:8001
# CI overrides these while the standalone example keeps its original behavior.
num_rollout=${NUM_ROLLOUT:-100}
full_hf_checkpoint_interval=${FULL_HF_CHECKPOINT_INTERVAL:-None}
scratch=/mxdelta/$run_id
vime_root=/root/vime

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "$scratch/mx-prepare"
test -f "$model_path/config.json"
test -f "$model_path/model.safetensors"

# Create the bucket and register the local model seed as catalog-only v0.
python3 - <<PY
import boto3
from botocore.config import Config
from modelexpress_rl import (
    ModelExpressControlClient,
    ObjectStorageSource,
    ObjectStorageType,
    WeightPayloadFormat,
    WeightVersionState,
)

boto3.client(
    "s3",
    endpoint_url="$s3_endpoint_url",
    region_name="$s3_region",
    config=Config(s3={"addressing_style": "path"}),
).create_bucket(Bucket="$s3_bucket")

with ModelExpressControlClient.connect(server_url="$mx_server_url") as control:
    control.create_weight_version(
        uid="$base_version_id",
        model_name="$run_id",
        idempotency_key="$run_id:v0",
        payload_format=WeightPayloadFormat.FULL_TENSOR,
        object_storage=ObjectStorageSource(
            storage_type=ObjectStorageType.S3,
            uri="$s3_uri_prefix/v0/model.safetensors.index.json",
        ),
        state=WeightVersionState.READY,
    )
PY

ray stop --force >/dev/null 2>&1 || true
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus=2 --disable-usage-stats
until ray status 2>/dev/null | grep -qE '/2\.0 GPU'; do sleep 2; done

mx_config=$(python3 - <<PY
import json
print(json.dumps({
    "model_name": "$run_id",
    "server_url": "$mx_server_url",
    "initial_base_version_id": "$base_version_id",
    "initial_version": "0",
    "s3_uri_prefix": "$s3_uri_prefix",
    "s3_endpoint_url": "$s3_endpoint_url",
    "s3_region_name": "$s3_region",
    "preparation_cache_dir": "$scratch/mx-prepare",
    "rpc_timeout_seconds": 30.0,
    "max_transfer_attempts": 3,
    "full_hf_checkpoint_interval": $full_hf_checkpoint_interval,
}))
PY
)

cd "$vime_root"
source scripts/models/qwen3-0.6B.sh
ckpt_args=(--hf-checkpoint "$model_path" --ref-load "$model_path" --load "$model_path" --megatron-to-hf-mode bridge)
rollout_args=(--prompt-data /opt/delta-refit/prompts.jsonl --input-key prompt --label-key label --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking": false}' --rollout-shuffle --rm-type random --num-rollout "$num_rollout" --rollout-batch-size 1 --n-samples-per-prompt 2 --rollout-max-response-len 32 --rollout-temperature 1 --global-batch-size 2 --balance-data --update-weights-interval 1)
parallel_args=(--tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --use-dynamic-batch-size --max-tokens-per-gpu 256)
grpo_args=(--advantage-estimator grpo --entropy-coef 0.01 --eps-clip 0.2 --eps-clip-high 0.28 --use-tis)
optimizer_args=(--optimizer adam --lr 1e-4 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98)
dynamo_args=(--rollout-dynamo-generation-url "$generation_url" --rollout-dynamo-rl-discovery-url "$discovery_url")
mx_args=(--update-weight-transport modelexpress --modelexpress-config "$mx_config")
misc_args=(--no-gradient-accumulation-fusion --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend unfused)

python3 train_async.py --train-backend megatron --actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 1 \
  "${MODEL_ARGS[@]}" "${ckpt_args[@]}" "${rollout_args[@]}" "${optimizer_args[@]}" \
  "${grpo_args[@]}" "${parallel_args[@]}" "${dynamo_args[@]}" "${mx_args[@]}" "${misc_args[@]}"

echo "TRAINING COMPLETE"
