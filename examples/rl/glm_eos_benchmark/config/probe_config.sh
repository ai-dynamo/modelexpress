#!/bin/bash
set -euo pipefail
source /bundle/env.sh
source /bundle/fabric.sh
export ROLE=trainer ROLE_RANK=0
export OUT=/evidence UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=/app/.venv
export MX_HOST=127.0.0.1 ORCHESTRATOR_HOST=127.0.0.1 INFER_HEAD=127.0.0.1
export HOSTS_CSV
HOSTS_CSV=$(seq -f 'probe%03g' -s, 1 36)
cd /app
export CFG_BASE="/tmp/cfg-$RUN_ID-$ROLE-$ROLE_RANK"
export CONFIG_ROOT="$CFG_BASE/resolved"
test -f "$MODEL_PATH/config.json"
extra_args=()
dp=$((INFER_NODES*GPUS_PER_NODE/TP))
if [[ "$EXECUTOR" = ray ]]; then
    dp=1
    extra_args+=(--inference.vllm.distributed-executor-backend ray --slurm.template-path /bundle/config-only.sbatch.j2)
fi
uv run rl @ /bundle/vendor/prime.toml --model.name "$MODEL_PATH" \
    --run.name "$RUN_ID" --output-dir "$CFG_BASE" --dry-run \
    --deployment.type multi_node --deployment.num-train-nodes "$TRAIN_NODES" \
    --deployment.num-infer-nodes "$NODES_PER_REPLICA" --deployment.num-infer-replicas "$INFER_REPLICAS" --deployment.gpus-per-node "$GPUS_PER_NODE" \
    --trainer.model.ep "$TRAIN_EP" --trainer.model.attn flash_attention_2 \
    --inference.vllm.tensor-parallel-size "$TP" \
    --inference.vllm.data-parallel-size "$dp" \
    --inference.vllm.dtype bfloat16 --inference.vllm.enforce-eager \
    --seq-len "$SEQ_LEN" --inference.vllm.max-model-len "$SEQ_LEN" \
    --inference.vllm.gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --orchestrator.max-off-policy-steps 8 --orchestrator.renderer.name "$MODEL_RENDERER" \
    --orchestrator.batch-size "$BATCH_SIZE" \
    --max-steps "$MAX_STEPS" --log.level debug \
    --weight-broadcast.type mx_refit --weight-broadcast.host "$MX_HOST" \
    --weight-broadcast.port 8001 --weight-broadcast.run-uid "$RUN_ID" \
    --weight-broadcast.timeout "$BROADCAST_TIMEOUT" \
    --weight-broadcast.reclaim-memory "$RECLAIM_MEMORY" "${extra_args[@]}"
uv run python /bundle/configure.py
cp -r "$CONFIG_ROOT" /evidence/resolved
uv run python /bundle/verify_overlay.py --resolved

uv run --no-sync python - <<'CHECK'
import json
from pathlib import Path
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
for cls,name in [(TrainerConfig,'trainer'),(InferenceConfig,'inference-0'),(OrchestratorConfig,'orchestrator')]:
    cls.model_validate(json.loads((Path('/evidence/resolved')/(name+'.json')).read_text()))
print('REBASED_GLM_CONFIGURATION_PASS')
CHECK
