#!/usr/bin/env bash
# Collect evidence for a Dynamo TRT-LLM autoscale startup-latency demo.
#
# Required:
#   NAMESPACE=<namespace> ./collect_demo_metrics.sh
#
# Optional:
#   RUN_TYPE=mx|vanilla
#   APP_LABEL='app.kubernetes.io/part-of=kimi-agg-autoscale'
#   DGD_NAME=kimi-agg-autoscale
#   DGDSA_NAME=kimi-agg-autoscale-source
#   HPA_NAME=kimi-agg-autoscale
#   MODEL_NAME=nvidia/Kimi-K2.5-NVFP4
#   OUT_DIR=./metrics

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  NAMESPACE=<namespace> ./collect_demo_metrics.sh

Environment:
  NAMESPACE   Kubernetes namespace to inspect. Required.
  APP_LABEL   Pod selector for the autoscale deployment.
              Default: app.kubernetes.io/part-of=kimi-agg-autoscale
  RUN_TYPE    Run family: mx or vanilla. Default: mx
  DGD_NAME    DynamoGraphDeployment name. Default: kimi-agg-autoscale
  DGDSA_NAME  DynamoGraphDeploymentScalingAdapter name.
              Default: kimi-agg-autoscale-source
  HPA_NAME    Optional HPA name. Default: kimi-agg-autoscale
  MODEL_NAME  Served model id. Default: nvidia/Kimi-K2.5-NVFP4
  OUT_DIR     Directory for metrics artifacts.
              Default: ./metrics/<timestamp>
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

NAMESPACE="${NAMESPACE:-${1:-}}"
RUN_TYPE="${RUN_TYPE:-mx}"
APP_LABEL="${APP_LABEL:-app.kubernetes.io/part-of=kimi-agg-autoscale}"
DGD_NAME="${DGD_NAME:-kimi-agg-autoscale}"
DGDSA_NAME="${DGDSA_NAME:-${DGD_NAME}-source}"
HPA_NAME="${HPA_NAME:-${DGD_NAME}}"
MODEL_NAME="${MODEL_NAME:-nvidia/Kimi-K2.5-NVFP4}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-metrics/${STAMP}}"

if [[ -z "${NAMESPACE}" || "${NAMESPACE}" == "-h" || "${NAMESPACE}" == "--help" ]]; then
  usage
  exit 1
fi

require_cmd kubectl
require_cmd jq
require_cmd perl

mkdir -p "${OUT_DIR}/logs"

echo "Collecting Kubernetes state into ${OUT_DIR}"

kubectl get dgd,dgdsa,hpa,pods,svc,deploy,pvc -n "${NAMESPACE}" -o wide \
  > "${OUT_DIR}/resources.txt" 2>&1 || true
kubectl get events -n "${NAMESPACE}" --sort-by=.lastTimestamp \
  > "${OUT_DIR}/events.txt" 2>&1 || true
kubectl get dgd "${DGD_NAME}" -n "${NAMESPACE}" -o yaml \
  > "${OUT_DIR}/dgd.yaml" 2>&1 || true
kubectl get dgdsa "${DGDSA_NAME}" -n "${NAMESPACE}" -o yaml \
  > "${OUT_DIR}/dgdsa.yaml" 2>&1 || true
kubectl describe hpa "${HPA_NAME}" -n "${NAMESPACE}" \
  > "${OUT_DIR}/hpa.describe.txt" 2>&1 || true

kubectl get pods -n "${NAMESPACE}" -l "${APP_LABEL}" -o json \
  > "${OUT_DIR}/pods.json"

jq -r '
  def cond($name):
    (.status.conditions // [] | map(select(.type == $name)) | .[0] // {});
  ["pod","phase","created","scheduled","initialized","containers_ready","ready","node"],
  (.items[] |
    [
      .metadata.name,
      .status.phase,
      .metadata.creationTimestamp,
      (cond("PodScheduled").lastTransitionTime // ""),
      (cond("Initialized").lastTransitionTime // ""),
      (cond("ContainersReady").lastTransitionTime // ""),
      (cond("Ready").lastTransitionTime // ""),
      (.spec.nodeName // "")
    ]
  ) | @csv
' "${OUT_DIR}/pods.json" > "${OUT_DIR}/pod_timeline.csv"

kubectl get modelmetadata -n "${NAMESPACE}" -o json \
  > "${OUT_DIR}/modelmetadata.json" 2>&1 || true
jq -r '
  ["name","model","rank","status","backend","tensor_count","updated_at","created_at"],
  (.items[] |
    [
      .metadata.name,
      .spec.modelName,
      (.status.worker.workerRank // ""),
      (.status.worker.status // ""),
      (.status.worker.backendType // ""),
      (.status.worker.tensorCount // ""),
      (.status.worker.updatedAt // ""),
      (.metadata.creationTimestamp // "")
    ]
  ) | @csv
' "${OUT_DIR}/modelmetadata.json" > "${OUT_DIR}/modelmetadata_worker_status.csv" 2>/dev/null || true

PODS_FILE="${OUT_DIR}/pod_names.txt"
jq -r '.items[].metadata.name' "${OUT_DIR}/pods.json" > "${PODS_FILE}"
pod_count=0
while IFS= read -r pod; do
  [[ -z "${pod}" ]] && continue
  pod_count=$((pod_count + 1))
  echo "Collecting logs for ${pod}"
  kubectl logs -n "${NAMESPACE}" "${pod}" --all-containers --timestamps --tail=-1 \
    > "${OUT_DIR}/logs/${pod}.k8s.log" 2>&1 || true
  kubectl exec -n "${NAMESPACE}" "${pod}" -- sh -lc \
    'for f in /tmp/mx_logs/*.log; do [ -e "$f" ] || continue; echo "===== $f ====="; cat "$f"; done' \
    > "${OUT_DIR}/logs/${pod}.mx_logs.txt" 2>&1 || true
done < "${PODS_FILE}"

grep -hE 'transferred .*Gbps|Transfer complete:.*Gbps|MX P2P weight transfer succeeded|publish.*ModelExpress|published .*GB' \
  "${OUT_DIR}"/logs/* 2>/dev/null \
  > "${OUT_DIR}/mx_key_log_lines.txt" || true

{
  echo "pod,rank,tensors,gb,seconds,gbps"
  for file in "${OUT_DIR}"/logs/*.mx_logs.txt; do
    [[ -e "${file}" ]] || continue
    pod="$(basename "${file}" .mx_logs.txt)"
    POD_NAME="${pod}" perl -ne '
      if (/Rank\s+(\d+):\s+transferred\s+(\d+)\s+params\s+\(([\d.]+)\s+GB\)\s+in\s+([\d.]+)s\s+\(([\d.]+)\s+Gbps\)/) {
        print join(",", $ENV{POD_NAME}, $1, $2, $3, $4, $5), "\n";
      } elsif (/Transfer complete:\s+(\d+)\s+tensors,\s+([\d.]+)\s+GB\s+in\s+([\d.]+)s\s+\(([\d.]+)\s+Gbps\)/) {
        print join(",", $ENV{POD_NAME}, "", $1, $2, $3, $4), "\n";
      }
    ' "${file}"
  done
} > "${OUT_DIR}/rdma_transfers.csv"

{
  echo "pod,rank,detected_at,success_at,detected_to_success_seconds"
  for file in "${OUT_DIR}"/logs/*.k8s.log; do
    [[ -e "${file}" ]] || continue
    pod="$(basename "${file}" .k8s.log)"
    POD_NAME="${pod}" perl -MTime::Piece -ne '
      sub epoch {
        my ($ts) = @_;
        return "" unless $ts;
        $ts =~ s/\.\d+Z$/Z/;
        return Time::Piece->strptime($ts, "%Y-%m-%dT%H:%M:%SZ")->epoch;
      }
      if (/^(\S+).*?\[RANK\s+(\d+)\].*MX sources detected, attempting P2P transfer/) {
        $detected{$2} //= $1;
      } elsif (/^(\S+).*?\[RANK\s+(\d+)\].*MX P2P weight transfer succeeded/) {
        $success{$2} //= $1;
      }
      END {
        for my $rank (sort { $a <=> $b } keys %{{map { $_ => 1 } (keys %detected, keys %success)}}) {
          my $seconds = "";
          if ($detected{$rank} && $success{$rank}) {
            $seconds = epoch($success{$rank}) - epoch($detected{$rank});
          }
          print join(",", $ENV{POD_NAME}, $rank, $detected{$rank} // "", $success{$rank} // "", $seconds), "\n";
        }
      }
    ' "${file}"
  done
} > "${OUT_DIR}/p2p_weight_transfer_events.csv"

{
  modelmetadata_records="$(jq -r '.items | length' "${OUT_DIR}/modelmetadata.json" 2>/dev/null || echo 0)"
  modelmetadata_records="${modelmetadata_records:-0}"
  modelmetadata_published_workers="$(jq -r '[.items[] | select(.status.worker.backendType == "nixl" and (.status.worker.tensorCount // 0) > 0)] | length' "${OUT_DIR}/modelmetadata.json" 2>/dev/null || echo 0)"
  modelmetadata_published_workers="${modelmetadata_published_workers:-0}"
  echo "# Dynamo TRT-LLM Autoscale Demo Metrics"
  echo
  echo "- Run type: ${RUN_TYPE}"
  echo "- Namespace: ${NAMESPACE}"
  echo "- Selector: ${APP_LABEL}"
  echo "- DGD: ${DGD_NAME}"
  echo "- DGDSA: ${DGDSA_NAME}"
  echo "- Model: ${MODEL_NAME}"
  echo "- Captured: ${STAMP}"
  echo "- ModelMetadata records: ${modelmetadata_records}"
  echo "- ModelMetadata published NIXL workers: ${modelmetadata_published_workers}"
  echo "- Pods captured: ${pod_count}"
  echo
  echo "## RDMA Transfer Summary"
  awk -F, '
    NR > 1 && NF >= 6 {
      count++;
      gb += $4;
      seconds += $5;
      gbps += $6;
      if (min_gbps == "" || $6 < min_gbps) min_gbps = $6;
      if ($6 > max_gbps) max_gbps = $6;
      if (min_sec == "" || $5 < min_sec) min_sec = $5;
      if ($5 > max_sec) max_sec = $5;
    }
    END {
      if (count == 0) {
        print "No RDMA transfer lines parsed yet.";
      } else {
        printf("- Parsed ranks/transfers: %d\n", count);
        printf("- Total GB parsed: %.2f\n", gb);
        printf("- Avg transfer seconds: %.2f\n", seconds / count);
        printf("- Min/max transfer seconds: %.2f / %.2f\n", min_sec, max_sec);
        printf("- Avg bandwidth: %.1f Gbps\n", gbps / count);
        printf("- Min/max bandwidth: %.1f / %.1f Gbps\n", min_gbps, max_gbps);
      }
    }
  ' "${OUT_DIR}/rdma_transfers.csv"
  echo
  echo "## P2P Transfer Events"
  awk -F, '
    NR > 1 && $5 != "" {
      count++;
      seconds += $5;
      if (min_sec == "" || $5 < min_sec) min_sec = $5;
      if ($5 > max_sec) max_sec = $5;
    }
    END {
      if (count == 0) {
        print "No MX P2P success events parsed yet.";
      } else {
        printf("- Successful rank events: %d\n", count);
        printf("- Avg detected-to-success seconds: %.2f\n", seconds / count);
        printf("- Min/max detected-to-success seconds: %.2f / %.2f\n", min_sec, max_sec);
      }
    }
  ' "${OUT_DIR}/p2p_weight_transfer_events.csv"
  echo
  echo "## Artifacts"
  echo "- pod_timeline.csv: Kubernetes creation/scheduling/readiness timestamps"
  echo "- rdma_transfers.csv: per-rank transfer time and bandwidth"
  echo "- p2p_weight_transfer_events.csv: per-rank MX source detection and transfer success timestamps"
  echo "- mx_key_log_lines.txt: key MX publish/transfer log lines"
  echo "- resources.txt/events.txt/hpa.describe.txt: autoscaling evidence"
  echo "- modelmetadata_worker_status.csv: Kubernetes metadata worker lifecycle states"
  if [[ -f "${OUT_DIR}/scale_event.json" ]]; then
    echo
    echo "## Controlled Scale Event"
    jq -r '
      "- Scale from/to: \(.from) -> \(.to)",
      "- Scale timestamp: \(.timestamp)",
      "- Scale delay after traffic marker: \(.delay_after_traffic_marker_seconds)s",
      "- Scale marker: \(.aiperf_scale_marker)"
    ' "${OUT_DIR}/scale_event.json" 2>/dev/null || true
    if [[ -f "${OUT_DIR}/traffic_marker_time.txt" ]]; then
      echo "- Traffic marker timestamp: $(cat "${OUT_DIR}/traffic_marker_time.txt")"
    fi
    if [[ -d "${OUT_DIR}/aiperf_artifacts" ]]; then
      echo "- AIPerf artifacts: aiperf_artifacts/"
    fi
  fi
} > "${OUT_DIR}/summary.md"

cat "${OUT_DIR}/summary.md"
