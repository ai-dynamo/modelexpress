#!/bin/bash
#SBATCH --job-name=coreai_tritoninference_triton3-mx.glm030-clean-v2
#SBATCH --account=coreai_tritoninference_triton3
#SBATCH --partition=batch
#SBATCH --time=01:15:00
#SBATCH --nodes=36
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the rendered directory}"
source ./env.sh
MODE=${1:-preflight}
case "$MODE" in preflight|cache|transport|benchmark) ;; *) echo 'Use preflight, cache, transport, or benchmark' >&2; exit 2;; esac
sha256sum -c SHA256SUMS
declare -A CHECKED_IMAGES=()
for kind in PRIME SERVER REDIS; do
    image_var="IMAGE_$kind"
    image=${!image_var}
    if [[ "$image" = /* ]]; then
        hash_var="IMAGE_${kind}_SHA256"
        image_spec="${!hash_var}:$image"
        if [[ -z ${CHECKED_IMAGES[$image_spec]+present} ]]; then
            printf '%s  %s\n' "${!hash_var}" "$image" | sha256sum -c -
            CHECKED_IMAGES[$image_spec]=1
        fi
    fi
done
test "$(pwd -P)" = "$RUN_HOST"
mkdir -p "$DATA_HOST/outputs/$RUN_ID" "$MODELS_HOST/hub"
OUT_HOST="$DATA_HOST/outputs/$RUN_ID"
# A fresh render is required for each attempt, including cache and preflight runs.
mkdir "$OUT_HOST/attempt.lock"
printf '%s\n' "$SLURM_JOB_ID" > job-id.txt
scontrol show job "$SLURM_JOB_ID" > "$OUT_HOST/slurm-job.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$OUT_HOST/hosts.txt"
cp "$OUT_HOST/hosts.txt" expected-hosts.txt
mapfile -t HOSTS < "$OUT_HOST/hosts.txt"
test "${#HOSTS[@]}" -eq "$((TRAIN_NODES + INFER_NODES))"
export MX_HOST=${HOSTS[$INFER_NODES]}
export ORCHESTRATOR_HOST=$MX_HOST
export TRAIN_HEAD=$MX_HOST
export INFER_HEAD=${HOSTS[0]}
export HOSTS_CSV
HOSTS_CSV=$(IFS=,; echo "${HOSTS[*]}")
MOUNTS="$RUN_HOST:/bundle:ro,$DATA_HOST:/data,$MODELS_HOST:/models,$RUN_HOST/prime-overlay/src/prime_rl:/app/src/prime_rl:ro"
PIDS=()
WORKLOAD_FINISHED=0
cleanup() {
    local code=$?
    trap - EXIT TERM INT
    # Preserve the workload code before any terminal verification can fail.
    printf '%s\n' "$code" > "$OUT_HOST/workload-exit.txt"
    local shutdown_code=0
    if [[ "$MODE" = benchmark && "$EXECUTOR" = ray ]]; then
        printf '%s\n' "$SLURM_JOB_ID" > "$OUT_HOST/shutdown-request.txt.tmp"
        mv "$OUT_HOST/shutdown-request.txt.tmp" "$OUT_HOST/shutdown-request.txt"
        # Keep every srun alive until remote role EXIT handlers acknowledge and capture.
        step "$TRAIN_HEAD" "$IMAGE_PRIME" /bin/bash -c '
            cd /app && UV_NO_SYNC=1 uv run --no-sync python /bundle/wait_for_ray_worker_capture.py /bundle "/data/outputs/$RUN_ID" \
                --timeout-seconds 120
        ' > "$OUT_HOST/ray-worker-capture-completion.log" 2>&1 || shutdown_code=$?
        if (( code == 0 && shutdown_code != 0 )); then code=$shutdown_code; fi
    fi
    for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    # Role EXIT handlers have 5s child cleanup + a bounded 90s log snapshot.
    for _ in {1..120}; do
        local alive=0
        for pid in "${PIDS[@]}"; do if kill -0 "$pid" 2>/dev/null; then alive=1; fi; done
        (( alive )) || break
        sleep 1
    done
    for pid in "${PIDS[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    # Only successful completion of every trainer and the orchestrator enables
    # post-run gates. All inputs are offline logs; Redis/MX need not stay live.
    if [[ "$WORKLOAD_FINISHED" = 1 && "$code" = 0 && ${CHECK_CACHE_ARM:-0} = 1 ]]; then
        local verification_code=0
        step "$TRAIN_HEAD" "$IMAGE_PRIME" /bin/bash -c '
            set -euo pipefail
            cd /app
            UV_NO_SYNC=1 uv run --no-sync python /bundle/verify_ray_worker_capture.py /bundle "/data/outputs/$RUN_ID" \
                > "/data/outputs/$RUN_ID/ray-worker-log-check.log" 2>&1
            UV_NO_SYNC=1 uv run --no-sync python /bundle/verify_runtime.py \
                --verify-all /bundle "/data/outputs/$RUN_ID" \
                > "/data/outputs/$RUN_ID/runtime-749-all-node-proof.log" 2>&1
            UV_NO_SYNC=1 uv run --no-sync python /bundle/recover_and_replay.py \
                --bundle /bundle --logs "/data/outputs/$RUN_ID" \
                --out "/data/outputs/$RUN_ID/recovered-verification"
        ' > "$OUT_HOST/recovered-verification.log" 2>&1 || verification_code=$?
        if (( verification_code != 0 )); then code=$verification_code; fi
    fi
    if [[ "$WORKLOAD_FINISHED" = 1 && "$code" = 0 ]]; then
        echo 'WORKLOAD_COMPLETE; installed-weight correctness and timing qualification remain separate gates.'
    fi
    printf '%s\n' "$code" > "$OUT_HOST/batch-exit.txt"
    exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
step() {
    local host=$1 image=$2
    shift 2
    local step_cpus="$ROLE_CPUS" step_role=auxiliary
    if [[ $# -ge 3 && "$1" = /bin/bash && "$2" = /bundle/role.sh ]]; then
        step_role=$3
        if [[ "$step_role" = trainer ]]; then step_cpus="$TRAINER_CPUS"; fi
    fi
    srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task="$step_cpus" --cpu-bind=none \
        --nodelist="$host" --container-image="$image" --container-mounts="$MOUNTS" \
        --container-workdir=/ --no-container-mount-home \
        uv run --project /app --no-sync python /bundle/cpu_grant.py \
        --role "$step_role" --expected-cpus "$step_cpus" -- "$@"
}
# Background-only: make the tracked PID own srun; foreground step must return.
background_step() {
    local host=$1 image=$2
    shift 2
    local step_cpus="$ROLE_CPUS" step_role=auxiliary
    if [[ $# -ge 3 && "$1" = /bin/bash && "$2" = /bundle/role.sh ]]; then
        step_role=$3
        if [[ "$step_role" = trainer ]]; then step_cpus="$TRAINER_CPUS"; fi
    fi
    exec srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task="$step_cpus" --cpu-bind=none \
        --nodelist="$host" --container-image="$image" --container-mounts="$MOUNTS" \
        --container-workdir=/ --no-container-mount-home \
        uv run --project /app --no-sync python /bundle/cpu_grant.py \
        --role "$step_role" --expected-cpus "$step_cpus" -- "$@"
}
if [[ "$MODE" = cache ]]; then
    step "$TRAIN_HEAD" "$IMAGE_PRIME" /bin/bash /bundle/role.sh cache 0
    exit
fi

# Verify every allocated node before starting any trainer or inference worker.
srun --nodes="${#HOSTS[@]}" --ntasks="${#HOSTS[@]}" --ntasks-per-node=1 \
    --cpus-per-task=8 --cpu-bind=none --kill-on-bad-exit=1 \
    --container-image="$IMAGE_PRIME" --container-mounts="$MOUNTS" \
    --container-workdir=/app --no-container-mount-home \
    /bin/bash /bundle/role.sh preflight 0
if [[ "$MODE" = preflight ]]; then exit; fi
if [[ "$MODE" = transport ]]; then
    test "${#HOSTS[@]}" -eq 2
    test "$GPUS_PER_NODE" -eq 8
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task="$ROLE_CPUS" \
        --cpu-bind=none --kill-on-bad-exit=1 \
        --container-image="$IMAGE_PRIME" --container-mounts="$MOUNTS" \
        --container-workdir=/app --no-container-mount-home \
        /bin/bash /bundle/role.sh transport 0
    exit
fi

# Dedicated per-job Redis; the server and Redis share only the trainer-head host.
background_step "$TRAIN_HEAD" "$IMAGE_REDIS" redis-server --port 6379 --bind 127.0.0.1 \
    --save '' --appendonly no > "$OUT_HOST/redis.log" 2>&1 &
PIDS+=("$!")
REDIS_PID=$!
# Redis is loopback-only. Wait from the colocated server container, not the batch node.
background_step "$TRAIN_HEAD" "$IMAGE_SERVER" /bin/bash -c '
    set -euo pipefail
    for _ in {1..120}; do
        if timeout 1 bash -c "echo > /dev/tcp/127.0.0.1/6379" 2>/dev/null; then
            exec env MX_METADATA_BACKEND=redis REDIS_URL=redis://127.0.0.1:6379 /app/modelexpress-server --port 8001
        fi
        sleep 1
    done
    echo "Local Redis startup timed out" >&2
    exit 1
' > "$OUT_HOST/server.log" 2>&1 &
PIDS+=("$!")
SERVER_PID=$!
ready=0
for _ in {1..300}; do
    kill -0 "$REDIS_PID" && kill -0 "$SERVER_PID" || exit 1
    if timeout 1 bash -c 'echo > /dev/tcp/'"$MX_HOST"'/8001' 2>/dev/null; then ready=1; break; fi
    sleep 1
done
(( ready )) || { echo 'MX startup timed out' >&2; exit 1; }

SERVICES=("$REDIS_PID" "$SERVER_PID")
if [[ "$EXECUTOR" = ray ]]; then
    for ((n=0; n<INFER_NODES; n++)); do
        # Replica heads run Ray and the API driver in the same container so
        # Ray's local Unix sockets share a filesystem.
        if (( n % NODES_PER_REPLICA == 0 )); then continue; fi
        background_step "${HOSTS[$n]}" "$IMAGE_PRIME" /bin/bash /bundle/role.sh ray "$n" \
            > "$OUT_HOST/ray-$n.log" 2>&1 &
        PIDS+=("$!"); SERVICES+=("$!")
    done
fi
for ((n=0; n<INFER_NODES; n++)); do
    if [[ "$EXECUTOR" = ray ]] && (( n % NODES_PER_REPLICA != 0 )); then continue; fi
    background_step "${HOSTS[$n]}" "$IMAGE_PRIME" /bin/bash /bundle/role.sh inference "$n" \
        > "$OUT_HOST/inference-$n.log" 2>&1 &
    PIDS+=("$!"); SERVICES+=("$!")
done
background_step "$INFER_HEAD" "$IMAGE_PRIME" /bin/bash /bundle/role.sh router 0 \
    > "$OUT_HOST/router.log" 2>&1 &
PIDS+=("$!"); SERVICES+=("$!")
background_step "$ORCHESTRATOR_HOST" "$IMAGE_PRIME" /bin/bash /bundle/role.sh orchestrator 0 \
    > "$OUT_HOST/orchestrator-0.log" 2>&1 &
PIDS+=("$!")
WORKERS=("$!")
for ((n=0; n<TRAIN_NODES; n++)); do
    background_step "${HOSTS[$((INFER_NODES+n))]}" "$IMAGE_PRIME" /bin/bash /bundle/role.sh trainer "$n" \
        > "$OUT_HOST/trainer-$n.log" 2>&1 &
    PIDS+=("$!"); WORKERS+=("$!")
done
# A failed service aborts immediately. Successful trainers and orchestrator must all finish.
while (( ${#WORKERS[@]} )); do
    for pid in "${SERVICES[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Required service exited early: $pid" >&2
            exit 1
        fi
    done
    remaining=()
    for pid in "${WORKERS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then remaining+=("$pid"); else wait "$pid"; fi
    done
    WORKERS=("${remaining[@]}")
    sleep 1
done
# EXIT performs coordinated service shutdown, raw worker capture, then offline gates.
WORKLOAD_FINISHED=1
