#!/usr/bin/env bash
# Best-effort collection also works when a worker OOM-killed or the driver failed.
set -euo pipefail
source "$(dirname -- "$0")/lib.sh" "${1:?usage: collect.sh RESULTS_DIR}"
for role in $ROLES; do
  k get pod "$RESOURCE_PREFIX-$role" -o json > "$RESULTS_DIR/$role-pod.json.tmp" && mv "$RESULTS_DIR/$role-pod.json.tmp" "$RESULTS_DIR/$role-pod.json" || true
  k exec "$RESOURCE_PREFIX-$role" -c main -- tar -C /refit -czf - benchmark > "$RESULTS_DIR/$role-evidence.tar.gz.tmp" && mv "$RESULTS_DIR/$role-evidence.tar.gz.tmp" "$RESULTS_DIR/$role-evidence.tar.gz" || true
done
k exec "$CONTROL_POD" -c main -- cat /tmp/mx-delta/report.json > "$RESULTS_DIR/publication.json.tmp" && mv "$RESULTS_DIR/publication.json.tmp" "$RESULTS_DIR/publication.json" || true
k exec "$CONTROL_POD" -c main -- tar -C /tmp -czf - mx-e2e > "$RESULTS_DIR/e2e-evidence.tar.gz.tmp" && mv "$RESULTS_DIR/e2e-evidence.tar.gz.tmp" "$RESULTS_DIR/e2e-evidence.tar.gz" || true
k logs "$CONTROL_POD" -c server > "$RESULTS_DIR/server.log" || true
python3 "$RESULTS_DIR/report.py" "$RESULTS_DIR"
