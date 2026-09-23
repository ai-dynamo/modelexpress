#!/usr/bin/env bash
# Run a previously rendered experiment. On failure retain resources for inspection.
set -euo pipefail
source "$(dirname -- "$0")/lib.sh" "${1:?usage: run.sh RESULTS_DIR}"
[[ ! -e "$RESULTS_DIR/started" ]] || { echo 'Render a fresh run; this directory was already started.' >&2; exit 1; }
k cluster-info
touch "$RESULTS_DIR/started"
pids=()
finish() {
  status=$?
  trap - EXIT
  bash "$SCRIPT_DIR/collect.sh" "$RESULTS_DIR" || status=1
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  echo "Resources retained. After reviewing evidence: bash $SCRIPT_DIR/cleanup.sh $RESULTS_DIR"
  exit "$status"
}
trap finish EXIT
for file in harness.json control.yaml worker-s3.yaml; do
  k apply --dry-run=server -f "$RESULTS_DIR/$file"
  k apply -f "$RESULTS_DIR/$file"
done
k wait --for=condition=Ready "pod/$CONTROL_POD" "pod/$SOURCE_POD" --timeout=10m
start_worker() {
  local role=$1 pod="$RESOURCE_PREFIX-$1"
  k exec "$pod" -c main -- python3 -u /opt/benchmark/native_server.py > "$RESULTS_DIR/$role-worker.log" 2>&1 &
  pids+=("$!")
  k exec "$pod" -c main -- python3 -u /opt/benchmark/monitor.py > "$RESULTS_DIR/$role-monitor.log" 2>&1 &
  pids+=("$!")
}
start_worker s3
k exec "$SOURCE_POD" -c main -- python3 -u /opt/benchmark/download_seed.py > "$RESULTS_DIR/seed.log" 2>&1 &
seed_pid=$!; pids+=("$seed_pid")
k exec "$CONTROL_POD" -c main -- python3 -u /opt/benchmark/publish_delta.py > "$RESULTS_DIR/publication.log" 2>&1 &
publish_pid=$!; pids+=("$publish_pid")
# All TP ranks must finish before the second worker requests peer weights.
deadline=$((SECONDS + 7200))
until grep -q '^BENCH_READY$' "$RESULTS_DIR/s3-worker.log"; do
  kill -0 "${pids[0]}" 2>/dev/null || { tail -n 30 "$RESULTS_DIR/s3-worker.log"; exit 1; }
  (( SECONDS < deadline )) || { echo 'S3 worker readiness timed out'; exit 1; }
  sleep 5
done
if [[ " $ROLES " == *' peer '* ]]; then
  k apply --dry-run=server -f "$RESULTS_DIR/worker-peer.yaml"
  k apply -f "$RESULTS_DIR/worker-peer.yaml"
  k wait --for=condition=Ready "pod/$PEER_POD" --timeout=10m
  start_worker peer
fi
wait "$seed_pid"
wait "$publish_pid"
k exec "$CONTROL_POD" -c main -- cat /tmp/mx-delta/report.json > "$RESULTS_DIR/publication.json"
k exec "$CONTROL_POD" -c main -- python3 -u /opt/benchmark/run_e2e.py > "$RESULTS_DIR/e2e-driver.log" 2>&1
grep -q '^E2E_PASS$' "$RESULTS_DIR/e2e-driver.log"
