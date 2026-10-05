#!/bin/bash
set -euo pipefail
umask 077
source /bundle/env.sh
source /bundle/fabric.sh
ROLE=$1
ROLE_RANK=$2
if [[ "$ROLE" = preflight || "$ROLE" = transport ]]; then ROLE_RANK=${SLURM_PROCID:-0}; fi
export ROLE ROLE_RANK
if [[ "$ROLE" = ray || "$ROLE" = inference ]]; then
    export MX_RDMA_NIC_PIN=auto
else
    unset MX_RDMA_NIC_PIN
fi
export OUT="/data/outputs/$RUN_ID"
export UV_NO_SYNC=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export UV_PROJECT_ENVIRONMENT=/app/.venv RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export HF_HOME=/models WANDB_MODE=offline WANDB_DIR="$OUT/wandb"
export FLASHINFER_CACHE_DIR=/data/flashinfer FLASHINFER_CUBIN_DIR=/data/flashinfer/cubins
export MX_WORKER_HOST
MX_WORKER_HOST=$(hostname -I | awk '{print $1}')
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd /app
mkdir -p "$OUT"
CHILDREN=()
RAY_CAPTURE_ACTIVE=0
RAY_CAPTURE_SOURCE=
finish() {
    local code=$?
    trap - EXIT INT
    trap "" TERM
    for pid in "${CHILDREN[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for _ in {1..5}; do
        local alive=0
        for pid in "${CHILDREN[@]}"; do if kill -0 "$pid" 2>/dev/null; then alive=1; fi; done
        (( alive )) || break
        sleep 1
    done
    for pid in "${CHILDREN[@]}"; do kill -KILL "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; done
    # Terminal-only snapshot, after tracked children have stopped. A failed
    # capture never replaces the role's original exit status; the final checker
    # requires all four successful manifests before qualifying the workload.
    if [[ "$RAY_CAPTURE_ACTIVE" = 1 ]]; then
        local capture_code=0
        timeout --signal=TERM --kill-after=5s 90s \
            uv run --no-sync python /bundle/capture_ray_worker_logs.py \
            --run-id "$RUN_ID" --role "$ROLE" --role-rank "$ROLE_RANK" \
            --role-exit-code "$code" --out "$OUT" --source "$RAY_CAPTURE_SOURCE" \
            > "$OUT/ray-worker-capture-$ROLE-$ROLE_RANK.log" 2>&1 || capture_code=$?
        printf '%s\n' "$capture_code" > "$OUT/ray-worker-capture-$ROLE-$ROLE_RANK.exit"
    fi
    printf '%s\n' "$code" > "$OUT/exit-$ROLE-$ROLE_RANK.txt"
    exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# Signal only the role Bash from inside its container; srun stays alive for EXIT capture.
if [[ "$EXECUTOR" = ray && ( "$ROLE" = ray || "$ROLE" = inference ) ]]; then
    role_shell_pid=$BASHPID
    role_shell_start=$(awk '{print $22}' "/proc/$role_shell_pid/stat")
    (
        trap - EXIT TERM INT
        while kill -0 "$role_shell_pid" 2>/dev/null; do
            if [[ -f "$OUT/shutdown-request.txt" ]]; then
                requested_job=$(cat "$OUT/shutdown-request.txt")
                [[ "$requested_job" = "$SLURM_JOB_ID" ]] || exit 2
                current_start=$(awk '{print $22}' "/proc/$role_shell_pid/stat" 2>/dev/null) || exit 0
                [[ "$current_start" = "$role_shell_start" ]] || exit 0
                printf '%s\n' "$SLURM_JOB_ID" > "$OUT/shutdown-ack-$ROLE-$ROLE_RANK.txt.tmp"
                mv "$OUT/shutdown-ack-$ROLE-$ROLE_RANK.txt.tmp" "$OUT/shutdown-ack-$ROLE-$ROLE_RANK.txt"
                kill -TERM "$role_shell_pid"
                exit 0
            fi
            sleep 0.2
        done
    ) &
    CHILDREN+=("$!")
fi
uv run python /bundle/verify_overlay.py
uv run python /bundle/vendor/gate_prime.py
if [[ "$ROLE" = cache ]]; then
    uv run python -c 'import os; from huggingface_hub import snapshot_download; print(snapshot_download(repo_id=os.environ["MODEL_REPO"], revision=os.environ["MODEL_REVISION"], cache_dir="/models/hub"))'
    exit
fi
if [[ "$ROLE" = preflight ]]; then
    uv run python /bundle/preflight.py
    uv run python /bundle/verify_runtime.py
    exit
fi
if [[ "$ROLE" = transport ]]; then
    uv run torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$ROLE_RANK" \
        --rdzv-backend=c10d --rdzv-id="$RUN_ID" --rdzv-endpoint="$TRAIN_HEAD:29500" \
        /bundle/probe_nixl.py &
    CHILDREN+=("$!"); wait "${CHILDREN[-1]}"
    exit
fi
if [[ "$EXECUTOR" = ray && ( "$ROLE" = ray || "$ROLE" = inference ) ]]; then
    IFS=, read -ra hosts <<< "$HOSTS_CSV"
    replica_head=${hosts[$((ROLE_RANK/NODES_PER_REPLICA*NODES_PER_REPLICA))]}
    export VLLM_HOST_IP="$MX_WORKER_HOST" RAY_ADDRESS="$replica_head:6381"
    export MX_REFIT_REPLICA_ID="$((ROLE_RANK/NODES_PER_REPLICA))"
    export RAY_DEDUP_LOGS=0
    export VLLM_CONFIG_ROOT="/tmp/vllm-config-$RUN_ID-$ROLE_RANK"
    mkdir -p "$VLLM_CONFIG_ROOT"
    # Keep each actor's node-local NIXL address inherited from its Ray node.
    printf '["MX_WORKER_HOST"]\n' > "$VLLM_CONFIG_ROOT/ray_non_carry_over_env_vars.json"
    if [[ "$ROLE" = ray || "$ROLE" = inference ]]; then
        ray_args=(--node-ip-address "$VLLM_HOST_IP" --num-gpus "$GPUS_PER_NODE" --num-cpus 8 --disable-usage-stats --block)
        if (( ROLE_RANK % NODES_PER_REPLICA == 0 )); then
            ray_args+=(--head --port 6381 --include-dashboard=false --temp-dir "/tmp/mx-ray-$SLURM_JOB_ID-$MX_REFIT_REPLICA_ID")
        else
            deadline=$((SECONDS+300))
            until timeout 1 bash -c 'echo > /dev/tcp/'"$replica_head"'/6381' 2>/dev/null; do
                (( SECONDS < deadline )) || exit 1
                sleep 2
            done
            ray_args+=(--address "$RAY_ADDRESS")
        fi
        # Ray workers inherit the head's temp-dir/session metadata (confirmed
        # in all CPU32 ray-N.log startup paths), even without --temp-dir locally.
        RAY_CAPTURE_SOURCE="/tmp/mx-ray-$SLURM_JOB_ID-$MX_REFIT_REPLICA_ID/session_latest/logs"
        RAY_CAPTURE_ACTIVE=1
        uv run ray start "${ray_args[@]}" &
        CHILDREN+=("$!")
        if [[ "$ROLE" = ray ]]; then
            wait "${CHILDREN[-1]}"
            exit
        fi
    fi
    deadline=$((SECONDS+300))
    until timeout 1 bash -c 'echo > /dev/tcp/'"$replica_head"'/6381' 2>/dev/null; do
        (( SECONDS < deadline )) || exit 1
        sleep 2
    done
    uv run python /bundle/ray_ready.py
fi
if [[ "$ROLE" = router ]]; then
    IFS=, read -ra hosts <<< "$HOSTS_CSV"
    workers=()
    for ((n=0; n<INFER_NODES; n++)); do
        if [[ "$EXECUTOR" = ray ]]; then
            if (( n % NODES_PER_REPLICA == 0 )); then workers+=("http://${hosts[$n]}:8100"); fi
            continue
        fi
        for ((d=0; d<GPUS_PER_NODE/TP; d++)); do workers+=("http://${hosts[$n]}:$((8100+d))"); done
    done
    vllm-router --policy round_robin --host 0.0.0.0 --port 8000 \
        --request-id-headers x-session-id --prometheus-port 29000 \
        --worker-startup-timeout-secs 4200 --worker-urls "${workers[@]}" &
    CHILDREN+=("$!"); wait "${CHILDREN[0]}"
    exit
fi
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
uv run python /bundle/verify_overlay.py --resolved
cp -r "$CONFIG_ROOT" "$OUT/config-$ROLE-$ROLE_RANK"
if [[ "$ROLE" = trainer || "$ROLE" = inference ]]; then
    uv run python /bundle/vendor/mx_railwatch.py --selftest-only
    uv run python /bundle/vendor/mx_railwatch.py "$OUT/railwatch-$ROLE-$ROLE_RANK.jsonl" "$RUN_ID" \
        > "$OUT/railwatch-$ROLE-$ROLE_RANK.stderr.log" 2>&1 &
    CHILDREN+=("$!")
fi
case "$ROLE" in
trainer)
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 "$((GPUS_PER_NODE-1))")
    # torchrun overwrites ROLE_RANK with the global rank for this single role.
    # Preserve the launch-node index independently for entry-proof joins.
    export GLM_LAUNCH_NODE_RANK="$ROLE_RANK"
    uv run torchrun --nnodes="$TRAIN_NODES" --nproc-per-node="$GPUS_PER_NODE" \
        --node-rank="$ROLE_RANK" --rdzv-backend=c10d --rdzv-id="$RUN_ID" \
        --rdzv-endpoint="$TRAIN_HEAD:29500" --log-dir="$OUT/torchrun-$ROLE_RANK" --redirects=3 --tee=3 \
        /bundle/trainer_entry.py @ "$CONFIG_ROOT/trainer.json" &
    CHILDREN+=("$!"); wait "${CHILDREN[-1]}"
    ;;
inference)
    engines=()
    local_engines=$((GPUS_PER_NODE/TP))
    if [[ "$EXECUTOR" = ray ]]; then local_engines=1; fi
    for ((d=0; d<local_engines; d++)); do
        gpus=$(seq -s, "$((d*TP))" "$((d*TP+TP-1))")
        if [[ "$EXECUTOR" = ray ]]; then gpus=$(seq -s, 0 "$((GPUS_PER_NODE-1))"); fi
        rpc_path="/tmp/vllm-$RUN_ID-$ROLE_RANK-$d"
        mkdir -p "$rpc_path"
        CUDA_VISIBLE_DEVICES="$gpus" VLLM_RPC_BASE_PATH="$rpc_path" \
            uv run inference @ "$CONFIG_ROOT/inference-$d.json" \
            > "$OUT/inference-$ROLE_RANK-dp$d.log" 2>&1 &
        CHILDREN+=("$!"); engines+=("$!")
    done
    # Any engine exit is unexpected until the batch supervisor terminates this role.
    wait -n "${engines[@]}"
    echo 'Inference engine exited before coordinated shutdown' >&2
    exit 1
    ;;
orchestrator)
    deadline=$((SECONDS+BROADCAST_TIMEOUT))
    until curl --max-time 5 -fsS "http://$INFER_HEAD:8000/v1/models" >/dev/null; do
        (( SECONDS < deadline )) || { echo 'Inference readiness timed out' >&2; exit 1; }
        sleep 5
    done
    shopt -s nullglob
    envs=("$CONFIG_ROOT"/envs/train/*.json)
    (( ${#envs[@]} )) || { echo 'Missing environment configs' >&2; exit 1; }
    for file in "${envs[@]}"; do uv run env-server @ "$file" & CHILDREN+=("$!"); done
    uv run orchestrator @ "$CONFIG_ROOT/orchestrator.json" &
    CHILDREN+=("$!"); wait "${CHILDREN[-1]}"
    ;;
*) echo "Unknown role: $ROLE" >&2; exit 2;;
esac
