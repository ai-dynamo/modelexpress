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

## Files

| File | Purpose |
|------|---------|
| `kimi-agg-autoscale-dgd.yaml` | The aggregated worker DGD. Same spec for every replica — auto-detect handles source vs target |
| `kimi-agg-autoscale-hpa.yaml` | DGDSA + HPA wrapper that exposes the `scale` subresource and drives replica count |

## Container Image

This demo uses the same image as the rest of the parent examples — see
[`../README.md`](../README.md) for build instructions. In short:

```bash
cd <modelexpress-repo-root>

docker buildx build --platform linux/arm64 --no-cache \
    -f examples/p2p_transfer_k8s/client/trtllm/Dockerfile.dynamo-runtime \
    --build-context trtllm=../TensorRT-LLM \
    -t <YOUR_REGISTRY>/<YOUR_NAME>:<YOUR_TAG> \
    --push .
```

Substitute the resulting image URI into the `image:` fields of
`kimi-agg-autoscale-dgd.yaml` (search for `<REGISTRY>/<NAME>:<TAG>`).

The image layers the MX client, MXCheckpointLoader, and Dynamo P2P hooks
on top of `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.0-dev.3`
(TRT-LLM 1.3.0rc11). The `tensorrtllm-runtime` image is published for
both `arm64` and `amd64`.

## Prerequisites

1. **Kubernetes cluster** with GB200 nodes (ARM64) and CPU nodes
2. **Dynamo platform** (etcd + NATS) reachable via FQDN from your namespace
3. **DGD operator** + **DGDSA controller** installed (both part of Dynamo platform)
4. **NVCR image pull secret** in your namespace
5. **HF token secret** in your namespace
6. **`shared-model-cache` PVC** with the Kimi K2.5 model in HF cache layout
7. **ComputeDomain** sized for `maxReplicas × multinode.nodeCount` nodes
8. **MX infrastructure** running (`modelexpress-server-decode` + `redis-decode`)

If MX infra isn't set up yet, deploy it from the parent directory:

```bash
kubectl apply -n <namespace> -f ../mx-infra-decode.yaml
```

## Step-by-Step Demo

### Step 1 — Update yamls for your cluster

Open `kimi-agg-autoscale-dgd.yaml` and update:
- Image tag: `nvcr.io/nvidian/dynamo-dev/<user>:dynamo-trtllm-mx-<tag>`
- Node pool name (`cloud.google.com/gke-nodepool` value)
- `NATS_SERVER` and `ETCD_ENDPOINTS` namespace FQDNs
- `MODEL_EXPRESS_URL` (must match your MX server's FQDN)
- `resourceClaimTemplateName` (your compute domain channel)

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

Verify all 8 ranks have published to Redis:

```bash
kubectl exec -n <namespace> deploy/redis-decode -- redis-cli KEYS 'mx:source:*:*' | wc -l
# Expected: 8
```

### Step 4 — Apply DGDSA + HPA

Once the first replica is `1/1 Running` and 8 workers are in Redis:

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

## Cleanup

```bash
kubectl delete -n <namespace> -f kimi-agg-autoscale-hpa.yaml
kubectl delete -n <namespace> -f kimi-agg-autoscale-dgd.yaml
```

MX infra and Redis can stay running for future deploys.

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
