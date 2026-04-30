#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Orchestrate the gRPC streaming benchmark on a Kubernetes cluster.
#
# Steps:
#   1. Apply server Deployment + Service.
#   2. Apply client Pod (same-node OR cross-node).
#   3. Wait for both to be Ready, capture the server pod IP.
#   4. Discover which network interface pod traffic actually traverses
#      (route + ethtool) so we know which fabric the result is sized against.
#   5. Run iperf3 from client -> server pod IP for the TCP-plane ceiling.
#   6. Sweep MX gRPC streaming chunk sizes and emit JSON-line results.
#   7. Tear down or leave running with --keep.
#
# Usage:
#   ./run.sh same-node [--keep]
#   ./run.sh cross-node [--keep]
#   ./run.sh both [--keep]
#
# Env:
#   NS               namespace, default: bench
#   IMAGE            image, default: <your-registry>/modelexpress-bench:latest
#   TOTAL_BYTES      per-run payload, default: 8G
#   CHUNK_SIZES      space-separated sweep, default: "32K 256K 1M 4M 16M"
#   WARMUP_BYTES     warmup payload, default: 512M
#   STRICT           "true" to enable production validation per chunk, default: false

set -euo pipefail

NS="${NS:-bench}"
IMAGE="${IMAGE:-<your-registry>/modelexpress-bench:latest}"
TOTAL_BYTES="${TOTAL_BYTES:-8G}"
CHUNK_SIZES="${CHUNK_SIZES:-32K 256K 1M 4M 16M}"
WARMUP_BYTES="${WARMUP_BYTES:-512M}"
STRICT="${STRICT:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEEP=0

usage() {
    cat <<EOF
Usage: $0 [same-node|cross-node|both] [--keep]

Env:
  NS=$NS
  IMAGE=$IMAGE
  TOTAL_BYTES=$TOTAL_BYTES
  CHUNK_SIZES=$CHUNK_SIZES
  WARMUP_BYTES=$WARMUP_BYTES
  STRICT=$STRICT
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi
TOPOLOGY="$1"
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep) KEEP=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

case "$TOPOLOGY" in
    same-node|cross-node|both) ;;
    *) echo "topology must be same-node|cross-node|both" >&2; exit 1 ;;
esac

apply_server() {
    echo "==> Applying server Deployment + Service in $NS"
    kubectl apply -n "$NS" -f "$SCRIPT_DIR/grpc-server.yaml"
    echo "==> Waiting for server Deployment Available"
    kubectl wait -n "$NS" --for=condition=Available --timeout=180s deployment/bench-grpc-server
}

apply_client() {
    local topo="$1"
    local manifest="$SCRIPT_DIR/grpc-client-${topo}.yaml"
    echo "==> Applying client Pod (${topo})"
    kubectl apply -n "$NS" -f "$manifest"
    echo "==> Waiting for client Pod Ready"
    kubectl wait -n "$NS" --for=condition=Ready --timeout=180s "pod/bench-grpc-client-${topo}"
}

server_pod_ip() {
    kubectl get pod -n "$NS" -l app=bench-grpc-server -o jsonpath='{.items[0].status.podIP}'
}

run_topology() {
    local topo="$1"
    local server_ip
    server_ip="$(server_pod_ip)"
    if [[ -z "$server_ip" ]]; then
        echo "could not resolve bench-grpc-server pod IP" >&2
        exit 1
    fi
    echo "==> Server pod IP: $server_ip"

    local client_pod="bench-grpc-client-${topo}"

    echo "==> Fabric discovery from $client_pod"
    kubectl exec -n "$NS" "$client_pod" -c bench-client -- \
        bash -c "ip route get $server_ip; echo ---; iface=\$(ip route get $server_ip | awk '/dev/ {for(i=1;i<=NF;i++) if (\$i==\"dev\") print \$(i+1); exit}'); echo iface=\$iface; ethtool \$iface 2>/dev/null | head -20 || echo 'ethtool unavailable'"

    echo "==> iperf3 baseline (10s)"
    kubectl exec -n "$NS" "$client_pod" -c bench-client -- \
        iperf3 -c "$server_ip" -p 5201 -t 10 -J | tee "/tmp/iperf3-${topo}.json" | tail -5

    echo "==> MX gRPC sweep: total=$TOTAL_BYTES, warmup=$WARMUP_BYTES, strict=$STRICT"
    local strict_flag=""
    if [[ "$STRICT" == "true" ]]; then strict_flag="--strict"; fi
    local out_file="/tmp/bench-grpc-${topo}.jsonl"
    : > "$out_file"
    for chunk in $CHUNK_SIZES; do
        echo "==> chunk=$chunk"
        kubectl exec -n "$NS" "$client_pod" -c bench-client -- \
            bench_grpc_streaming client \
                --server-addr "http://${server_ip}:8001" \
                --total-bytes "$TOTAL_BYTES" \
                --warmup-bytes "$WARMUP_BYTES" \
                --chunk-size "$chunk" \
                --label "${topo}-${chunk}" \
                $strict_flag | tee -a "$out_file"
    done
    echo "==> Results for $topo: $out_file"
}

apply_server
case "$TOPOLOGY" in
    same-node)
        apply_client same-node
        run_topology same-node
        ;;
    cross-node)
        apply_client cross-node
        run_topology cross-node
        ;;
    both)
        apply_client same-node
        run_topology same-node
        apply_client cross-node
        run_topology cross-node
        ;;
esac

if [[ "$KEEP" -eq 0 ]]; then
    echo "==> Tearing down"
    kubectl delete -n "$NS" --ignore-not-found=true \
        -f "$SCRIPT_DIR/grpc-server.yaml"
    for topo in same-node cross-node; do
        kubectl delete -n "$NS" --ignore-not-found=true \
            -f "$SCRIPT_DIR/grpc-client-${topo}.yaml" || true
    done
else
    echo "==> Leaving resources up (--keep)"
fi
