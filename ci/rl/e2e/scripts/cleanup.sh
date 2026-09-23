#!/usr/bin/env bash
# Requires successful publication; otherwise inspect the exact run prefix manually.
set -euo pipefail
source "$(dirname -- "$0")/lib.sh" "${1:?usage: cleanup.sh RESULTS_DIR}"
bash "$SCRIPT_DIR/collect.sh" "$RESULTS_DIR" || true
k exec "$CONTROL_POD" -c main -- python3 -u /opt/benchmark/cleanup_objects.py > "$RESULTS_DIR/cleanup-objects.json"
python3 - "$RESULTS_DIR" <<'PY'
import json,sys
from pathlib import Path
assert json.loads((Path(sys.argv[1])/'cleanup-objects.json').read_text())['verified_absent']
PY
pods=("pod/$CONTROL_POD")
for role in $ROLES; do
  k delete -f "$RESULTS_DIR/worker-$role.yaml" --ignore-not-found --wait=false
  pods+=("pod/$RESOURCE_PREFIX-$role")
done
k delete -f "$RESULTS_DIR/control.yaml" -f "$RESULTS_DIR/harness.json" --ignore-not-found --wait=false
k wait --for=delete "${pods[@]}" --timeout=2m
k get pods,services,configmaps -l "mx-regression=$RUN_ID" -o json > "$RESULTS_DIR/cleanup-kubernetes.json"
python3 - "$RESULTS_DIR" <<'PY'
import json,sys
from pathlib import Path
assert not json.loads((Path(sys.argv[1])/'cleanup-kubernetes.json').read_text())['items']
print('Cleanup verified; evidence retained in',sys.argv[1])
PY
