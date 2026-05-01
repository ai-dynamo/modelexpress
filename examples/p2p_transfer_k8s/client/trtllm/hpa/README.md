# Autoscaling TRT-LLM with ModelExpress P2P

End-to-end demo showing how to autoscale a TRT-LLM serving deployment
where new replicas receive their weights via NIXL RDMA instead of
loading from disk — turning a ~20 minute cold-start into ~5 minutes.

The first replica loads the model from disk and publishes via MX.
Every subsequent replica that HPA brings up auto-detects the existing
MX source and pulls weights over the network, **~750× faster than disk**
for the loading phase.

## Validated Results

Kimi K2.5 (TP=8, 2 nodes per replica), GCP GB200, 4× 400G RoCE:

| Replica | Weight loading | End-to-end ready |
|---------|---------------|------------------|
| Initial (disk) | ~20 minutes | ~22 minutes |
| HPA scale-up #1 (RDMA) | **~1.6 seconds per rank** | **~4.5 minutes** |

Per-rank RDMA throughput on the new replica: **361–583 Gbps**
(8 ranks × 90.75 GB = 726 GB transferred end-to-end).

AIPerf Mooncake trace replay, 2 replicas, 200 requests, 0 errors:
avg ISL **34,985** tokens, max ISL **169,993** tokens, avg OSL
**98** tokens, benchmark duration **243s**, throughput **0.82 req/s**.

## Files

| File | Purpose |
|------|---------|
| `kimi-agg-autoscale-dgd.yaml` | The aggregated worker DGD. Same spec for every replica — auto-detect handles source vs target |
| `kimi-agg-autoscale-hpa.yaml` | DGDSA + HPA wrapper that exposes the `scale` subresource and drives replica count |
| `kimi-agg-autoscale-aiperf.yaml` | AIPerf Job for synthetic load or controlled-rate Mooncake inputs against the frontend |
| `run_synthetic_traffic.sh` | Helper that waits for readiness, creates the AIPerf Job, and streams logs |
| `run_autoscale_collection.sh` | Orchestrates an apples-to-apples autoscale run: starts traffic, triggers scale-up, samples metrics, and collects artifacts |
| `build_time_series.py` | Builds dashboard-ready time-sliced request, replica, P2P, and GPU CSVs from a run directory |

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
