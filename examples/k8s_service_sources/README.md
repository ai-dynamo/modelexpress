# K8s-Service-Routed Sources Example

Concrete TP=2 manifests for deploying the `k8s-service` metadata backend - source pools sit behind a Kubernetes Service, kube-proxy load-balances, clients open a direct gRPC channel to the Service DNS name.

For design rationale, backend-selection guidance, and trade-offs, see [`docs/K8S_SERVICE_BACKEND.md`](../../docs/K8S_SERVICE_BACKEND.md). For the generic deployment how-to (env vars, kubectl operations), see [`docs/DEPLOYMENT.md`](../../docs/DEPLOYMENT.md#k8s-service-routed-backend).

## Limitations

This backend is for **stable-weight inference only**. Weights loaded at pod startup don't change for the lifetime of the pod. If your workload is RL rollouts, live fine-tune broadcasts, mixed-revision serving under load, or anything needing per-worker addressability, use the central-coordinator backends (`redis` or `kubernetes`) instead. Full details in [`K8S_SERVICE_BACKEND.md`](../../docs/K8S_SERVICE_BACKEND.md#limitations).

## Files

- [`sources-tp2-single-pod.yaml`](sources-tp2-single-pod.yaml) - **Multi-GPU-per-pod shape** (primary). One Deployment with 2-GPU pods running `--tensor-parallel-size=2`. ONE Service named `mx-sources` with two named ports (`rank-0: 6555`, `rank-1: 6556`). Client uses the default pattern `mx-sources`; port is auto-computed from rank. This is the topology for production TP inference (NVLink is intra-node-only).
- [`sources-tp2.yaml`](sources-tp2.yaml) - **1-GPU-per-pod shape**. Two Deployments (one per rank) with rank-labeled pods. Two Services selecting by `mx.rank`. Pattern: `MX_K8S_SERVICE_PATTERN=mx-sources-rank-{rank}:6555`. For per-rank autoscaling or cross-pod setups where TP isn't involved.
- [`target.yaml`](target.yaml) - Target Deployments that pull weights via the k8s-service backend. Paired with `sources-tp2.yaml` (1-GPU-per-pod). For multi-GPU-per-pod, scale `sources-tp2-single-pod.yaml` directly - new replicas join as both targets and sources.

## Architecture (multi-GPU-per-pod, the primary shape)

```mermaid
graph TD
    subgraph "Source Pods (replicas)"
        PA[Pod A (2 GPUs)<br/>rank 0 on :6555<br/>rank 1 on :6556]
        PB[Pod B (2 GPUs)<br/>rank 0 on :6555<br/>rank 1 on :6556]
    end
    SVC[Service: mx-sources<br/>port 6555 -> :6555<br/>port 6556 -> :6556]
    T[Target Pod (2 GPUs)]
    SVC --> PA
    SVC --> PB
    T -- "mx-sources:6555 (rank 0)" --> SVC
    T -- "mx-sources:6556 (rank 1)" --> SVC
    T -- "NIXL/RDMA pull" --> PA
```

The Service's Endpoints object is the source list, maintained by Kubernetes based on pod readiness. No `modelexpress-server` in this topology; `mx_source_id` is computed client-side and validated on the `GetTensorManifest` response.

## Prerequisites

1. Kubernetes cluster with GPU nodes. The YAMLs request `rdma/shared_ib` resources for InfiniBand/RoCE - the fast path and the configuration production should run on. Without RDMA, UCX/NIXL falls back to plain TCP at significant throughput cost; drop the `rdma/shared_ib` resource requests from the manifests to run without it.
2. A path for weights to reach the pods. Any of: pre-downloaded to a shared PVC, streamed from S3 (set `MX_S3_URI`), or downloaded from HuggingFace at pod startup. For the HuggingFace option, create the token secret with `kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=<token>`.
3. A model revision you trust. Pin it via `MX_MODEL_REVISION=<commit_sha>` (or set `model_config.revision` in vLLM) so `mx_source_id` is content-addressed.

## Deploying

**Multi-GPU-per-pod (primary):**

```bash
kubectl apply -f sources-tp2-single-pod.yaml
kubectl wait --for=condition=Ready pod -l app=mx-sources --timeout=15m
kubectl get svc mx-sources
# Scale to add more replicas (new ones pull via P2P from existing).
kubectl scale deployment mx-sources --replicas=4
```

**1-GPU-per-pod:**

```bash
kubectl apply -f sources-tp2.yaml
kubectl wait --for=condition=Ready pod -l app=mx-sources --timeout=15m
kubectl apply -f target.yaml
```

## Environment variables

| Variable                         | Default                         | Meaning                                                                       |
|----------------------------------|---------------------------------|-------------------------------------------------------------------------------|
| `MX_METADATA_BACKEND`            | `""` (central server)           | Set to `k8s-service` to enable this backend.                                   |
| `MX_K8S_SERVICE_PATTERN`         | `mx-sources`                    | DNS template. `{rank}` substitutes the worker's rank. If the pattern has no `:port`, the client auto-appends `:{MX_WORKER_GRPC_PORT + rank}`. |
| `MX_K8S_SOURCE_RETRIES`          | `5`                             | Max retries on `FAILED_PRECONDITION` before giving up.                         |
| `MX_K8S_SOURCE_BACKOFF_SECONDS`  | `0.5`                           | Sleep between retries (fresh channel per attempt).                             |
| `MX_MODEL_REVISION`              | unset                           | Override for `SourceIdentity.revision`. Useful for local / non-HF checkpoints. |
| `MX_WORKER_GRPC_PORT`            | `6555`                          | Base port for the WorkerGrpcServer (bound port is this + `device_id`).         |
