# Autoscaling TRT-LLM with ModelExpress P2P

End-to-end demo showing how to autoscale a TRT-LLM serving deployment
where new replicas receive their weights via ModelExpress/NIXL P2P
instead of loading from disk.

The first replica loads the model from disk and publishes via MX.
Every subsequent replica that HPA brings up auto-detects the existing
MX source and pulls weights over the network. The measured benefit is
visible at two levels: the weight path itself is much faster, and the
end-to-end scale-up readiness time drops even after TRT-LLM runtime and
readiness overheads are included.

## Validated Results

Latest validated run, 2026-05-01:

- Model: `nvidia/Kimi-K2.5-NVFP4`
- Topology: aggregated TRT-LLM, TP=8, 2 nodes per replica, 4 GPUs/node
- Cluster: GCP GB200, GPU node pool `customer-gpu-o7v`
- Model cache: `shared-model-cache` PVC backed by Lustre CSI PV
  `shared-model-cache-zheng`, RWX, 36000Gi
- Traffic: Mooncake inputs, continuous AIPerf concurrency ramp 1 -> 128,
  900s total, 30s reporting slices

| Metric | Disk / vanilla TRT-LLM | MX P2P | Delta |
|--------|-------------------------|--------|-------|
| Overall scale-up startup latency | 542s | 305s | **44% drop** |
| Startup phase: pre-weight init | 204s | 205s | same |
| Startup phase: weight load / P2P transfer | 296s | 13s | **283s saved** |
| Startup phase: post-weight to Ready | 42s | 88s | +46s MX overhead |

The startup phase breakdown reconciles the numbers: MX saves ~283s on
the weight path, but gives back ~46s in post-weight runtime/readiness
work, so the measured net startup saving is 237s.

## Files

| File | Purpose |
|------|---------|
| `kimi-agg-autoscale-dgd.yaml` | The aggregated worker DGD. Same spec for every replica — auto-detect handles source vs target |
| `kimi-agg-autoscale-hpa.yaml` | DGDSA + HPA wrapper that exposes the `scale` subresource and drives replica count |
| `kimi-agg-autoscale-aiperf.yaml` | AIPerf Job for synthetic load or controlled-rate Mooncake inputs against the frontend |
| `run_synthetic_traffic.sh` | Helper that waits for readiness, creates the AIPerf Job, and streams logs |
| `run_autoscale_collection.sh` | Orchestrates an apples-to-apples autoscale run: starts traffic, triggers scale-up, samples metrics, and collects artifacts |
| `build_time_series.py` | Builds dashboard-ready time-sliced request, replica, P2P, and GPU CSVs from a run directory |
| `collect_demo_metrics.sh` | Captures pod timelines, logs, ModelMetadata, Kubernetes events, HPA state, and summary artifacts |
| `kimi-agg-vanilla-autoscale-dgd.yaml` | Vanilla Dynamo TRT-LLM DGD used for disk-loading comparison |
| `kimi-trtllm-image-prewarm.yaml` | Optional image prewarm helper to avoid image-pull noise in startup measurements |
| `aiperf.Dockerfile` | Reproducible AIPerf image definition when the job startup path is too slow |

## Pre-built Container Image

You don't need to build anything. Use:

```
nvcr.io/nvidian/dynamo-dev/kavink:dynamo-trtllm-mx-v3.0.0
```

This image (ARM64, GB200) layers MX client, MXCheckpointLoader, and
Dynamo P2P hooks on top of `tensorrtllm-runtime:1.1.0-dev.3` (TRT-LLM
1.3.0rc11). It works out-of-the-box with the yamls in this folder.

If you need to rebuild (different base, different model, etc.), see the
parent directory's `README.md` for build instructions.

## Prerequisites

1. **Kubernetes cluster** with GB200 nodes (ARM64) and CPU nodes
2. **Dynamo platform** (etcd + NATS) reachable via FQDN from your namespace
3. **DGD operator** + **DGDSA controller** installed (both part of Dynamo platform)
4. **NVCR image pull secret** in your namespace
5. **HF token secret** in your namespace
6. **`shared-model-cache` PVC** with `nvidia/Kimi-K2.5-NVFP4` in HF cache layout
7. **ComputeDomain** sized for `maxReplicas × multinode.nodeCount` nodes
8. **ModelMetadata CRD** installed and **ModelExpress RBAC** applied in your namespace
   for the in-DGD ModelExpress service

The DGD sets `max_seq_len: 256000`, which is close to the model's
262144-token maximum and is sufficient for this demo.

If ModelExpress CRD/RBAC isn't set up yet:

```bash
kubectl apply -f ../../../server/kubernetes_backend/crd-modelmetadata.yaml
kubectl apply -n <namespace> -f ../../../server/kubernetes_backend/rbac-modelmetadata.yaml
kubectl patch rolebinding -n <namespace> modelexpress-metadata \
  --type=json \
  -p='[{"op":"replace","path":"/subjects/0/namespace","value":"<namespace>"}]'
```

## Step-by-Step Demo

### Step 1 — Update yamls for your cluster

Open `kimi-agg-autoscale-dgd.yaml` and update:
- Image tag: `nvcr.io/nvidian/dynamo-dev/<user>:dynamo-trtllm-mx-<tag>`
- Node pool name (`cloud.google.com/gke-nodepool` value)
- `NATS_SERVER` and `ETCD_ENDPOINTS` namespace FQDNs
- `resourceClaimTemplateName` (your compute domain channel)

The source worker has a delayed startup probe so kubelet does not spam
`503 Service Unavailable` while TRT-LLM is still loading the model.
The current value is `30s`; this is long enough to avoid immediate
probe noise, but short enough not to hide a fast ModelExpress P2P
scale-up behind an artificial five-minute delay.

DGD service topology is immutable after creation. If you already created
`kimi-agg-autoscale` from an older manifest, delete and recreate the DGD
so `ModelExpress` and `Frontend` are both present from the start.

### Step 2 — Deploy the DGD with replicas=1

```bash
kubectl apply -n <namespace> -f kimi-agg-autoscale-dgd.yaml
```

Watch the first replica come up. It will load weights from disk
(~20 minutes for Kimi K2.5):

```bash
kubectl get pods -n <namespace> -l app.kubernetes.io/part-of=kimi-agg-autoscale -w
```

### Step 3 — Wait for the source to publish

Verify all 8 ranks have published NIXL metadata records to the
ModelMetadata CRD:

```bash
MODEL_NAME=nvidia/Kimi-K2.5-NVFP4
kubectl get modelmetadata -n <namespace> -o json \
  | jq --arg model "$MODEL_NAME" \
      '[.items[] | select(.spec.modelName == $model and .status.worker.backendType == "nixl" and (.status.worker.tensorCount // 0) > 0)] | length'
# Expected: 8 published records
```

### Step 4 — Apply DGDSA + HPA

Once the first replica is `1/1 Running` and 8 workers are published in
ModelMetadata:

```bash
kubectl apply -n <namespace> -f kimi-agg-autoscale-hpa.yaml
```

Verify they're wired up:

```bash
kubectl get dgdsa,hpa -n <namespace>
```

### Step 5 — Trigger a scale-up

For a manual demo (no inference load required):

```bash
kubectl patch dgdsa -n <namespace> kimi-agg-autoscale-source \
  --type=merge -p '{"spec":{"replicas":2}}'
```

For a real autoscale demo with HPA, send inference traffic to push CPU
above 50% (the threshold in `kimi-agg-autoscale-hpa.yaml`). HPA will
update the DGDSA replicas automatically.

To generate Mooncake input traffic with AIPerf, use the companion Job.
It follows Dynamo's trace-replay benchmark pattern but rewrites trace
timestamps to a controlled step schedule: wait for `/v1/models`, run a
short synthetic warmup, then run Mooncake dataset inputs with
`--custom-dataset-type mooncake_trace --fixed-schedule`.
For Kimi, the Job installs `aiperf==0.7.0` plus `tiktoken` and passes
`--use-server-token-count --tokenizer-trust-remote-code`; AIPerf still
needs the tokenizer to synthesize prompts even though token metrics come
from the server.

```bash
NAMESPACE=<namespace> ./run_synthetic_traffic.sh
```

By default, the job targets:

| Setting | Default |
|---------|---------|
| Endpoint | `kimi-agg-autoscale-frontend:8000` |
| Model | `nvidia/Kimi-K2.5-NVFP4` |
| Dataset | Mooncake trace from `kvcache-ai/Mooncake` |
| Trace bounds | 200 requests, max ISL 256k, max OSL 200 |
| Schedule | `STEP_RATES=1,1,4,4,2,1`, `STEP_DURATIONS=60` |
| Synthetic fallback | 10k input tokens, 200 output tokens |
| Artifacts | `/model-cache/perf/<epoch>_<job>/` |

By default, the Job downloads the upstream Mooncake trace into the
`shared-model-cache` PVC at
`/model-cache/perf/traces/mooncake_trace_raw.jsonl`, prepares a bounded
Kimi subset at `/model-cache/perf/traces/mooncake_trace_kimi_subset.jsonl`,
rewrites its timestamps to the configured `STEP_RATES`/`STEP_DURATIONS`,
and writes AIPerf CSV/JSON plus server metrics under the artifact
directory. This keeps the Mooncake prompt/input distribution while
making MX vs vanilla runs rate-controlled and repeatable. To run a pure
synthetic profile instead, unset `TRACE_FILE` in
`kimi-agg-autoscale-aiperf.yaml`.

### Step 6 — Watch the new replica load via RDMA

```bash
kubectl get pods -n <namespace> -l app.kubernetes.io/part-of=kimi-agg-autoscale -w
```

You should see `kimi-agg-autoscale-0-source-1-source-{ldr,wkr}-*` pods
appear and reach `1/1 Ready` in ~5 minutes (vs ~22 min for the source).

Verify the new replica used RDMA:

```bash
NEW_LDR=$(kubectl get pods -n <namespace> \
  -l app.kubernetes.io/part-of=kimi-agg-autoscale \
  -o jsonpath='{.items[?(@.metadata.name contains "source-1.*ldr")].metadata.name}')

kubectl exec -n <namespace> $NEW_LDR -- cat /tmp/mx_logs/rank0.log | grep -E "transferred|Gbps"
# Expected output:
# "Rank 0: transferred 1815 params (90.75 GB) in 1.62s (447.7 Gbps) — DIRECT into model params"
```

## Metric Capture

Use the helper scripts in this directory when preparing demo evidence.

For customer-facing MX vs vanilla comparisons, prefer the orchestrated
collector. It owns the scale-up event so both runs use the same traffic
profile and scale timing:

```bash
NAMESPACE=<namespace> \
RUN_TYPE=mx \
TRAFFIC_PROFILE_ID=mooncake-kimi-200-speedup5 \
STEP_RATES=1,1,4,4,2,1 \
STEP_DURATIONS=60 \
TRACE_REQUEST_COUNT=200 \
SCALE_FROM=1 \
SCALE_TO=2 \
SCALE_DELAY_SECONDS=60 \
./run_autoscale_collection.sh
```

For a vanilla Dynamo TRT-LLM run, pass the vanilla deployment names but
keep the traffic and scale settings identical:

```bash
NAMESPACE=<namespace> \
RUN_TYPE=vanilla \
DGD_NAME=<vanilla-dgd-name> \
SERVICE_NAME=<scaled-worker-service-name> \
FRONTEND_SERVICE=<vanilla-frontend-service>:8000 \
AIPERF_MANIFEST=./kimi-agg-autoscale-aiperf.yaml \
TRAFFIC_PROFILE_ID=mooncake-kimi-200-speedup5 \
STEP_RATES=1,1,4,4,2,1 \
STEP_DURATIONS=60 \
TRACE_REQUEST_COUNT=200 \
SCALE_FROM=1 \
SCALE_TO=2 \
SCALE_DELAY_SECONDS=60 \
./run_autoscale_collection.sh
```

The script renders the AIPerf Job with the requested model/frontend and
step schedule, waits for the actual AIPerf Mooncake-input or synthetic
profile marker in the job logs, then patches the DGDSA after
`SCALE_DELAY_SECONDS`. This keeps the scale-up anchor consistent across
MX and vanilla runs.

To run the full README flow and collect artifacts at the end:

```bash
NAMESPACE=<namespace> ./reproduce_demo.sh
```

The script applies ModelMetadata CRD/RBAC, deploys the DGD with
ModelExpress + Frontend + source worker, waits for all `WORLD_SIZE`
ranks to publish NIXL ModelMetadata records, applies DGDSA/HPA, patches
the DGDSA to `SCALE_TO=2`, and then runs metric collection. Review and
customize the manifests first; the DGD still needs cluster-specific
node affinity, NATS/etcd FQDNs, and `resourceClaimTemplateName`.

To collect metrics from an already-running demo:

```bash
NAMESPACE=<namespace> ./collect_demo_metrics.sh
```

Artifacts are written under `metrics/<timestamp>/`:

| Artifact | Use |
|----------|-----|
| `summary.md` | Customer-demo headline metrics from the captured run |
| `time_series_<slice>s.csv` | Dashboard-ready time series: target RPS, started/completed requests, errors, latency, TTFT, ITL, replicas, ready pods, P2P events, GPU samples |
| `pod_timeline.csv` | Pod creation, scheduling, container-ready, and ready timestamps for cold start vs scale-up |
| `rdma_transfers.csv` | Parsed per-rank transfer GB, seconds, and Gbps |
| `mx_key_log_lines.txt` | Key publish/transfer log lines for screenshots or appendix evidence |
| `resources.txt`, `events.txt`, `hpa.describe.txt` | HPA/DGD state and Kubernetes event evidence |

Key demo metrics to report:

| Metric | How to capture |
|--------|----------------|
| Initial disk cold-start | Time from source DGD apply/pod creation to source pod `Ready` in `pod_timeline.csv` |
| Source publish readiness | Time until ModelMetadata has `WORLD_SIZE` NIXL records with nonzero `.status.worker.tensorCount` |
| HPA/manual scale-up latency | Time from DGDSA replica change to new replica pod `Ready` in `pod_timeline.csv` |
| RDMA weight-load time | Per-rank `seconds` in `rdma_transfers.csv` |
| RDMA throughput | Per-rank and average `gbps` in `rdma_transfers.csv` |
| Autoscaler behavior | `hpa.describe.txt`, `events.txt`, and DGDSA/DGD replica counts |

## Preparing the Next Customer Demo

Use this checklist when preparing a repeatable MX P2P vs vanilla TRT-LLM
demo. The goal is to keep all variables identical except the weight
loading path.

### 1. Cluster and storage prerequisites

Verify these before applying either DGD:

```bash
kubectl get nodes -L cloud.google.com/gke-nodepool,kubernetes.io/arch
kubectl -n <namespace> get pvc shared-model-cache -o wide
kubectl get pv <bound-pv-name> -o yaml | grep -E 'driver:|filesystem:|storage:|accessModes:|volumeHandle:'
kubectl -n <namespace> get secret nvcr-imagepullsecret hf-token-secret
kubectl get crd modelmetadatas.model-express.ai
kubectl -n <namespace> get role modelexpress-metadata
kubectl -n <namespace> get rolebinding modelexpress-metadata
kubectl -n <namespace> get serviceaccount modelexpress
```

For the latest validated run, the model cache was:

- PVC: `shared-model-cache`
- PV: `shared-model-cache-zheng`
- Driver: `lustre.csi.storage.gke.io`
- Filesystem: `model`
- Access: RWX
- Size: 36000Gi

Keep the MX and vanilla runs on the same GPU node pool and storage
backend. Image-pull and scheduling delays can easily dominate the story,
so prewarm the runtime images before the customer run if the nodes are
fresh:

```bash
kubectl apply -n <namespace> -f kimi-trtllm-image-prewarm.yaml
kubectl -n <namespace> rollout status ds/kimi-trtllm-image-prewarm --timeout=20m
kubectl delete -n <namespace> -f kimi-trtllm-image-prewarm.yaml
```

### 2. Reset cleanly between runs

Before each run, return to exactly one source replica and remove stale
metadata. Stale ModelMetadata can make a new first replica attempt P2P
against a deleted source and fail with `NIXL_ERR_REMOTE_DISCONNECT`.

```bash
kubectl patch dgdsa -n <namespace> <dgd-name>-source \
  --type=merge -p '{"spec":{"replicas":1}}'

# For MX runs only, clear old model metadata when doing a clean reset.
kubectl -n <namespace> delete modelmetadata \
  -l modelName=nvidia/Kimi-K2.5-NVFP4 --ignore-not-found
```

If labels are not present on the ModelMetadata CRs, inspect and delete by
model name:

```bash
kubectl -n <namespace> get modelmetadata -o json \
  | jq -r '.items[]
      | select(.spec.modelName == "nvidia/Kimi-K2.5-NVFP4")
      | .metadata.name' \
  | while read -r name; do
      [ -n "$name" ] && kubectl -n <namespace> delete modelmetadata "$name"
    done
```

Wait for the baseline source leader and worker to be ready before
starting traffic:

```bash
kubectl -n <namespace> get pods \
  -l app.kubernetes.io/part-of=<dgd-name> -w
```

### 3. Run MX and vanilla with the same collector profile

Use `run_autoscale_collection.sh` for both runs. It starts AIPerf,
samples Kubernetes/frontend/GPU state, triggers the controlled scale-up,
collects logs, and builds `time_series_30s.csv`.

Latest validated MX profile:

```bash
NAMESPACE=zheng \
RUN_TYPE=mx \
DGD_NAME=kimi-agg-autoscale \
FRONTEND_SERVICE=kimi-agg-autoscale-frontend:8000 \
SCALE_TRIGGER_MODE=log_marker \
SCALE_DELAY_SECONDS=0 \
BASELINE_CONCURRENCY=1 \
SURGE_CONCURRENCY=128 \
BASELINE_DURATION_SECONDS=90 \
SURGE_DURATION_SECONDS=810 \
CONCURRENCY_RAMP_DURATION_SECONDS=90 \
TRACE_REQUEST_COUNT=20000 \
TRACE_MAX_ISL=256000 \
TRACE_MAX_OSL=8000 \
./run_autoscale_collection.sh
```

Run vanilla immediately after with the same traffic knobs:

```bash
NAMESPACE=zheng \
RUN_TYPE=vanilla \
DGD_NAME=kimi-agg-vanilla-autoscale \
DGDSA_NAME=kimi-agg-vanilla-autoscale-source \
FRONTEND_SERVICE=kimi-agg-vanilla-autoscale-frontend:8000 \
SCALE_TRIGGER_MODE=log_marker \
SCALE_DELAY_SECONDS=0 \
BASELINE_CONCURRENCY=1 \
SURGE_CONCURRENCY=128 \
BASELINE_DURATION_SECONDS=90 \
SURGE_DURATION_SECONDS=810 \
CONCURRENCY_RAMP_DURATION_SECONDS=90 \
TRACE_REQUEST_COUNT=20000 \
TRACE_MAX_ISL=256000 \
TRACE_MAX_OSL=8000 \
./run_autoscale_collection.sh
```

Use the same model, PVC/PV, node pool, TP, node count, traffic duration,
concurrency, ISL cap, OSL cap, and scale trigger mode for both runs.

### 4. Interpret the startup metrics correctly

Do not compare MX P2P transfer time directly with pod Ready time. They
measure different scopes.

- **Overall startup latency**: controlled scale-up trigger -> new
  TRT-LLM leader pod Ready. This includes scheduling, pre-weight init,
  model load/transfer, CUDA graph/runtime work, and readiness.
- **Weight load / P2P phase**: wall-clock phase inside startup. Disk is
  first `Start to load safetensor file` -> last `Memory used after
  loading model weights`; MX is first `MX sources detected` -> last
  `MX P2P weight transfer succeeded`.

For the latest run:

```text
Disk:   204s pre-weight + 296s weight load + 42s post-weight = 542s
MX P2P: 205s pre-weight +  13s P2P       + 88s post-weight = 305s
```

The weight path saved 283s, but MX had 46s more post-weight overhead, so
the net overall startup saving was 237s, or a 44% drop.

### 5. Generate customer-facing data and graphs

The collector produces the raw data needed for plotting:

- `time_series_30s.csv`: latency, throughput, errors, concurrency,
  ready replicas, GPU samples, P2P events
- `pod_timeline.csv`: pod creation/scheduled/initialized/ready
- `p2p_weight_transfer_events.csv`: MX rank-level transfer timing
- `logs/*.k8s.log`: TRT-LLM phase boundaries for disk load and P2P
- `run_config.json`: model, traffic, scale, and collection settings

For the customer chart, use 30s time-series buckets and drop drain
buckets after traffic has stopped. In the latest run, buckets at
`seconds_from_traffic_start >= 900` were drain/shutdown buckets where
`started_requests=0` but completions were still arriving; including them
made request latency look artificially odd.

Recommended panels:

| Row | Panels |
|-----|--------|
| 1 | TTFT p50, TTFT p90 |
| 2 | Request latency p50, Request latency p90 |
| 3 | Output token throughput, Request throughput |
| 4 | Overall startup latency, Startup phase breakdown |

Include a short context subtitle at the top:

```text
Model: nvidia/Kimi-K2.5-NVFP4 | TP=8 | Nodes=2 | PV: Lustre CSI (RWX shared model cache)
```

### 6. Known pitfalls from the latest run

- **Image pull noise**: A new node without the runtime image adds
  minutes. Prewarm images or keep node affinity on warmed nodes.
- **SchedulingGated/Pending pods**: usually node resources, compute
  domain capacity, affinity, or ephemeral-storage requests. Check
  `kubectl describe pod` before changing model/runtime settings.
- **Frontend and ModelExpress placement**: these should run on CPU nodes
  when possible; source workers should consume the GPU nodes.
- **Readiness probe delay**: a 300s startup probe delay hides the real
  P2P readiness improvement. Use a short initial delay such as 30s with
  a generous failure threshold.
- **Grove init latency**: first pod startup can include Grove/container
  setup and image pull. Do not include it in weight-load-only metrics.
- **AIPerf drain buckets**: remove buckets after the traffic duration
  when plotting request latency.
- **Mooncake multi-turn traces**: make sampled rows independent
  requests; otherwise accumulated session context can exceed
  `max_seq_len` and produce negative `default_max_tokens` errors.
- **ITL diff was not compelling** in this profile, so the final chart
  focuses on TTFT and request latency quantiles.
- **CPU HPA target is only a demo trigger**. For production-like scaling,
  prefer Prometheus adapter metrics such as queue depth, request rate, or
  p95/p99 latency.

## Cleanup

```bash
kubectl delete -n <namespace> -f kimi-agg-autoscale-hpa.yaml
kubectl delete -n <namespace> -f kimi-agg-autoscale-dgd.yaml
```

The ModelMetadata CRD/RBAC can stay in the namespace for future deploys.

## How Auto-Detect Works

Both replicas use the **same** DGD spec. The difference comes from
`MXCheckpointLoader.load_weights()` in TRT-LLM:

```python
def load_weights(self, checkpoint_dir, mapping, **kwargs):
    if self._has_existing_sources():
        return self._try_p2p_transfer(model, mapping, checkpoint_dir)
    else:
        # Fall through to HfCheckpointLoader.load_weights()
        return super().load_weights(checkpoint_dir, mapping=mapping, **kwargs)
```

- **First replica**: probes MX server, no sources found → falls through
  to disk loading → publishes via `publish_as_source()` after load
- **Subsequent replicas**: probes MX server, sources detected → calls
  `MxLiveWeightLoader` for NIXL RDMA receive

Same image, same yaml, same env vars — different runtime behavior based
on what's already published in MX.

## Production Considerations

- **HPA metric**: CPU is a placeholder. For LLM workloads use a
  Prometheus adapter and scale on TRT-LLM queue depth, request rate,
  or P99 latency. Set `kubectl explain hpa.spec.metrics` for options.
- **Scale-down stabilization**: 300s default is conservative. Tune
  `behavior.scaleDown.stabilizationWindowSeconds` for your workload.
- **MX server HA**: The MX server is a single point of failure for
  the *registration* path, but transfers are direct GPU↔GPU. If the
  MX server goes down, in-flight pods continue serving; only new
  scale-ups would fall back to disk loading.
- **Compute domain sizing**: `maxReplicas × multinode.nodeCount` nodes
  must fit in your ComputeDomain.
- **Source pod lifetime**: Keep the source replica running for as long
  as you may want to scale up. Restarting the source rotates its NIXL
  agent ID and existing target replicas keep working but new scale-ups
  must wait for re-publish.

## Companion PRs

| PR | Repo | Status | What it provides |
|----|------|--------|------------------|
| [#13531](https://github.com/NVIDIA/TensorRT-LLM/pull/13531) | TRT-LLM | Ready | `MXCheckpointLoader` with `_has_existing_sources()` auto-detect |
| [#8037](https://github.com/ai-dynamo/dynamo/pull/8037) | Dynamo | Open | `--model-express-url` engine integration |
| [#218](https://github.com/ai-dynamo/modelexpress/pull/218) | ModelExpress | Open | Deployment yamls, Dockerfiles, patch scripts (this folder) |
| [#202](https://github.com/ai-dynamo/modelexpress/pull/202) | ModelExpress | **Merged** | `MxLiveWeightLoader`, `publish_model_params` |
