#!/usr/bin/env bash
# CI namespace ownership and lifecycle; all inputs come from the trusted workflow.
set -euo pipefail
mode=${1:?usage: ci.sh setup|run|cleanup}
: "${MODEL_PROFILE:?}" "${NAMESPACE:?}" "${KUBE_CONTEXT:?}" "${RUN_ID:?}" "${RESULTS_DIR:?}"
: "${SERVER_IMAGE:?}" "${WORKER_IMAGE:?}" "${MX_E2E_S3_ROLE_ARN:?}"
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
k=(kubectl --kubeconfig="${KUBECONFIG:-/teleport/kubeconfig.yaml}" --context="$KUBE_CONTEXT" -n "$NAMESPACE")
owner="${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}"
render() {
  python3 "$root/scripts/prepare.py" "$MODEL_PROFILE" "$RESULTS_DIR" \
    --environment aws-ci --run-id "$RUN_ID" --paths s3 --service-account mx-e2e
}
owned() {
  actual=$("${k[@]}" get namespace "$NAMESPACE" --ignore-not-found \
    -o go-template='{{ index .metadata.labels "ci.modelexpress.nvidia.com/run-id" }}')
  [[ "$actual" == "$owner" ]]
}
case "$mode" in
  setup)
    "${k[@]}" create namespace "$NAMESPACE" --dry-run=client -o json \
      | "${k[@]}" label --local -f - "ci.modelexpress.nvidia.com/run-id=$owner" -o json \
      | "${k[@]}" create -f -
    render
    gpu_count=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tp"])' "$RESULTS_DIR/config.json")
    "${k[@]}" create quota e2e-gpu-budget --hard="requests.nvidia.com/gpu=$gpu_count,limits.nvidia.com/gpu=$gpu_count"
    "${k[@]}" create serviceaccount mx-e2e
    "${k[@]}" annotate serviceaccount mx-e2e "eks.amazonaws.com/role-arn=$MX_E2E_S3_ROLE_ARN"
    "${k[@]}" create secret docker-registry nvcr-imagepullsecret --docker-server=nvcr.io \
      --docker-username="\$oauthtoken" --docker-password="${NGC_API_KEY:?}"
    python3 - "$RESULTS_DIR" <<'PYCODE'
import sys
from pathlib import Path
import yaml
for path in Path(sys.argv[1]).glob('*.yaml'):
    manifest = yaml.safe_load(path.read_text())
    for item in manifest['items']:
        if item['kind'] == 'Pod':
            item['spec']['activeDeadlineSeconds'] = 3300
    path.write_text(yaml.safe_dump(manifest))
PYCODE
    "${k[@]}" apply -f "$RESULTS_DIR/harness.json" -f "$RESULTS_DIR/control.yaml"
    "${k[@]}" wait --for=condition=Ready "pod/mx-$RUN_ID-control" --timeout=5m
    "${k[@]}" exec "mx-$RUN_ID-control" -c main -- python3 -c '
import boto3
from config import CONFIG
s3 = boto3.client("s3", region_name=CONFIG["storage"]["region"])
s3.head_object(Bucket=CONFIG["bucket"], Key=CONFIG["seed_prefix"] + "snapshot-manifest.json")
print("S3 snapshot manifest accessible")
'
    ;;
  run)
    owned || { echo 'Namespace ownership mismatch' >&2; exit 1; }
    bash "$root/scripts/run.sh" "$RESULTS_DIR"
    ;;
  cleanup)
    actual=$("${k[@]}" get namespace "$NAMESPACE" --ignore-not-found -o name)
    [[ -n "$actual" ]] || exit 0
    owned || { echo 'Namespace ownership mismatch; refusing cleanup' >&2; exit 1; }
    trap 'status=$?; trap - EXIT; "${k[@]}" delete namespace "$NAMESPACE" --wait=true --timeout=2m || status=1; exit "$status"' EXIT
    "${k[@]}" delete pod "mx-$RUN_ID-control" "mx-$RUN_ID-s3" --ignore-not-found --wait=true --timeout=2m
    render
    # Reconstruct the trusted cleanup payload even if setup/publication was interrupted.
    "${k[@]}" apply -f "$RESULTS_DIR/harness.json"
    python3 - "$RESULTS_DIR" <<'PY'
import json, sys
from pathlib import Path
import yaml
root = Path(sys.argv[1])
items = yaml.safe_load((root / 'control.yaml').read_text())['items']
configmap = next(x for x in items if x['kind'] == 'ConfigMap')
pod = next(x for x in items if x['kind'] == 'Pod')
pod['metadata']['name'] += '-cleanup'
pod['spec']['activeDeadlineSeconds'] = 300
main = pod['spec']['containers'][0]
main['command'] = ['python3', '-u', '/opt/benchmark/cleanup_run.py']
main['resources'] = {'requests': {'cpu': '100m', 'memory': '256Mi'}, 'limits': {'cpu': '1', 'memory': '1Gi'}}
pod['spec']['containers'] = [main]
(root / 'cleanup.yaml').write_text(json.dumps({'apiVersion': 'v1', 'kind': 'List', 'items': [configmap, pod]}))
PY
    status=0
    "${k[@]}" apply -f "$RESULTS_DIR/cleanup.yaml" || status=1
    "${k[@]}" wait --for=jsonpath='{.status.phase}'=Succeeded "pod/mx-$RUN_ID-control-cleanup" --timeout=6m || status=1
    "${k[@]}" logs "mx-$RUN_ID-control-cleanup" > "$RESULTS_DIR/cleanup-objects.log" 2>&1 || status=1
    cat "$RESULTS_DIR/cleanup-objects.log"
    exit "$status"
    ;;
  *) echo "Unknown operation: $mode" >&2; exit 1 ;;
esac
