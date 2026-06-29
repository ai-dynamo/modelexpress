#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Efficacy benchmark for the P2P source-selection layer.
#
# Targets the single-source-convergence problem: pre-warm M>=2 source replicas,
# then start N>M target replicas concurrently and measure how the N targets
# distribute across the M sources -- random vs rendezvous_hash.
#
# Primary signal (efficacy): source-utilization balance, collected from each
# target's structured selection log (source_worker_id of the chosen source) and
# summarized by collect_distribution.py. Pair with fan-out makespan (wall-clock
# for all N targets to finish loading). Both policies emit the same logs, so no
# metrics backend is required; set MX_P2P_METRICS_ENABLED=1 for the Prometheus
# layer instead.
#
# Shape uses the primary sweep: TP=1 (one GPU per replica) packed
# onto fragmented free GPUs, so a high fan-out ratio is cheap. Each replica is a
# central-coordinator (redis-backed) MX client over the shared overlay configmap
# (the mx-client-src overlay configmap) -- no image rebuild.
#
# Prereqs (run against any GPU cluster with InfiniBand/RDMA, e.g. 8x B200 nodes):
#   - a kube context (set MX_KUBE_CONTEXT, defaults to the current context) and namespace
#   - an image pull secret + the mx-client-src overlay configmap present
#   - a model staged on a RWX PVC (MODEL_PVC / MODEL_PATH below)
#
# Usage:
#   ./run_bench.sh <policy> <M_sources> <N_targets>
#   ./run_bench.sh rendezvous_hash 4 20
#   ./run_bench.sh random 4 20
set -euo pipefail

POLICY="${1:?policy: random|rendezvous_hash}"
M="${2:?number of source replicas (>=2)}"
N="${3:?number of target replicas (>M)}"

CTX="${MX_KUBE_CONTEXT:-$(kubectl config current-context)}"
NS="${MX_NAMESPACE:-modelexpress}"
# A vLLM runtime image that bundles the modelexpress client plugin (no default --
# point this at your own image). The server image defaults to the official
# release on NGC; override as needed.
RUNTIME_IMAGE="${MX_RUNTIME_IMAGE:?set MX_RUNTIME_IMAGE to a vLLM runtime image with the modelexpress plugin}"
SERVER_IMAGE="${MX_SERVER_IMAGE:-nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.4.0}"
# Pull secret for the images above (create with: kubectl create secret
# docker-registry nvcr-imagepullsecret --docker-server=nvcr.io ...).
IMAGE_PULL_SECRET="${MX_IMAGE_PULL_SECRET:-nvcr-imagepullsecret}"
# ConfigMap holding the client-source overlay tarball (mxclient.tar.gz).
OVERLAY_CM="${MX_OVERLAY_CONFIGMAP:-mx-client-src}"
MODEL_PVC="${MX_MODEL_PVC:-shared-model-cache}"
MODEL_PATH="${MX_MODEL_PATH:-/model-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/main}"
SERVED_NAME="${MX_SERVED_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
K="${MX_KILL:-0}"   # set 1 to tear down sources+targets at end

kc() { kubectl --context "$CTX" -n "$NS" "$@"; }

echo "[bench] policy=$POLICY M=$M N=$N ns=$NS"

# --- control plane: redis + modelexpress-server (central coordinator) ---
kc apply -f - <<YAML
apiVersion: apps/v1
kind: Deployment
metadata: { name: mx-oss-redis, labels: { app: mx-oss-redis } }
spec:
  replicas: 1
  selector: { matchLabels: { app: mx-oss-redis } }
  template:
    metadata: { labels: { app: mx-oss-redis } }
    spec:
      imagePullSecrets: [{ name: ${IMAGE_PULL_SECRET} }]
      containers:
        - name: redis
          image: redis:7-alpine
          args: ["redis-server", "--save", "", "--appendonly", "no"]
          ports: [{ containerPort: 6379 }]
---
apiVersion: v1
kind: Service
metadata: { name: mx-oss-redis }
spec:
  selector: { app: mx-oss-redis }
  ports: [{ port: 6379, targetPort: 6379 }]
---
apiVersion: apps/v1
kind: Deployment
metadata: { name: mx-oss-server, labels: { app: mx-oss-server } }
spec:
  replicas: 1
  selector: { matchLabels: { app: mx-oss-server } }
  template:
    metadata: { labels: { app: mx-oss-server } }
    spec:
      imagePullSecrets: [{ name: ${IMAGE_PULL_SECRET} }]
      containers:
        - name: main
          image: ${SERVER_IMAGE}
          command: ["/bin/sh", "-c"]
          args: ["exec /app/modelexpress-server --port 8001"]
          env:
            - { name: MX_METADATA_BACKEND, value: redis }
            - { name: REDIS_URL, value: "redis://mx-oss-redis:6379" }
          ports: [{ containerPort: 8001 }]
---
apiVersion: v1
kind: Service
metadata: { name: mx-oss-server }
spec:
  selector: { app: mx-oss-server }
  ports: [{ port: 8001, targetPort: 8001 }]
YAML

echo "[bench] waiting for control plane..."
kc rollout status deploy/mx-oss-redis --timeout=120s
kc rollout status deploy/mx-oss-server --timeout=180s

# Flush stale metadata before a run (CLAUDE.md: stale metadata breaks P2P).
kc exec deploy/mx-oss-redis -- redis-cli FLUSHALL >/dev/null 2>&1 || true

# --- worker pod factory (role=source|target, index, optional policy) ---
make_worker() {
  local role="$1" idx="$2" policy_env="$3"
  cat <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: mx-oss-${role}-${idx}
  labels: { app: mx-oss-${role}, bench-role: "${role}", bench-policy: "${policy_env:-none}" }
spec:
  restartPolicy: Never
  imagePullSecrets: [{ name: ${IMAGE_PULL_SECRET} }]
  nodeSelector: { nvidia.com/gpu.product: NVIDIA-B200 }
  tolerations: [{ operator: Exists }]
  volumes:
    - { name: model-cache, persistentVolumeClaim: { claimName: ${MODEL_PVC} } }
    - { name: mx-client-src, configMap: { name: ${OVERLAY_CM} } }
  containers:
    - name: main
      image: ${RUNTIME_IMAGE}
      command: ["/bin/sh", "-c"]
      args:
        - |-
          set -eux
          export HF_HOME=/model-cache HF_HUB_CACHE=/model-cache/hub HF_HUB_OFFLINE=1
          export VLLM_PLUGINS=modelexpress
          export MX_SERVER_ADDRESS=mx-oss-server:8001
          export MODEL_EXPRESS_URL=mx-oss-server:8001
          export MX_P2P_METADATA=1
          export MX_RDMA_NIC_PIN=auto
          ${policy_env:+export MX_P2P_SOURCE_SELECTOR=${policy_env}}
          mkdir -p /tmp/mxclient
          tar -xzf /opt/mxclient-src/mxclient.tar.gz -C /tmp/mxclient
          python3 -m pip install --no-deps /tmp/mxclient
          exec python3 -m dynamo.vllm \
            --model ${MODEL_PATH} --served-model-name ${SERVED_NAME} \
            --load-format modelexpress --tensor-parallel-size 1 --trust-remote-code
      securityContext: { runAsUser: 0, capabilities: { add: [IPC_LOCK, SYS_PTRACE] } }
      resources:
        requests: { cpu: "8", memory: 48Gi, nvidia.com/gpu: "1", rdma/ib: "1" }
        limits: { cpu: "16", memory: 96Gi, nvidia.com/gpu: "1", rdma/ib: "1" }
      volumeMounts:
        - { name: model-cache, mountPath: /model-cache }
        - { name: mx-client-src, mountPath: /opt/mxclient-src, readOnly: true }
YAML
}

# --- pre-warm M sources (no selector needed; they publish, do not select) ---
for i in $(seq 0 $((M - 1))); do make_worker source "$i" "" | kc apply -f -; done
echo "[bench] waiting for $M sources to become READY..."
for i in $(seq 0 $((M - 1))); do kc wait --for=condition=Ready pod/mx-oss-source-"$i" --timeout=1200s; done
# Let sources publish READY to the server.
sleep 30

# --- start N targets concurrently with the chosen policy ---
START_EPOCH=$(date +%s)
for i in $(seq 0 $((N - 1))); do make_worker target "$i" "$POLICY" | kc apply -f -; done
echo "[bench] waiting for $N targets to finish loading..."
for i in $(seq 0 $((N - 1))); do
  kc wait --for=condition=Ready pod/mx-oss-target-"$i" --timeout=1800s || true
done
END_EPOCH=$(date +%s)
echo "[bench] fan-out makespan: $((END_EPOCH - START_EPOCH))s"

# --- collect each target's selected source from its log ---
mkdir -p "/tmp/bench-${POLICY}"
for i in $(seq 0 $((N - 1))); do
  kc logs mx-oss-target-"$i" > "/tmp/bench-${POLICY}/target-${i}.log" 2>&1 || true
done
python3 "$(dirname "$0")/collect_distribution.py" "/tmp/bench-${POLICY}" "$POLICY" "$M" "$N" "$((END_EPOCH - START_EPOCH))"

if [ "$K" = "1" ]; then
  kc delete pod -l app=mx-oss-source --wait=false
  kc delete pod -l app=mx-oss-target --wait=false
fi
