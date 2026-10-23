# K8s-Service-Routed Sources Example

Concrete TP=2 manifests for deploying the `k8s-service` metadata backend - source pools sit behind a Kubernetes Service, kube-proxy load-balances, clients open a direct gRPC channel to the Service DNS name.

For design rationale, backend-selection guidance, and trade-offs, see [`docs/K8S_SERVICE_BACKEND.md`](../../docs/K8S_SERVICE_BACKEND.md). For the generic deployment how-to (env vars, kubectl operations), see [`docs/DEPLOYMENT.md`](../../docs/DEPLOYMENT.md#k8s-service-routed-backend).

## Limitations

This backend is for **stable-weight inference only**. Weights loaded at pod startup don't change for the lifetime of the pod. If your workload is RL rollouts, live fine-tune broadcasts, mixed-revision serving under load, or anything needing per-worker addressability, use the central-coordinator backends (`redis` or `kubernetes`) instead. Full details in [`K8S_SERVICE_BACKEND.md`](../../docs/K8S_SERVICE_BACKEND.md#limitations).

## Files

- [`leader-discovery-service.yaml`](leader-discovery-service.yaml) - One discovery Service and one port for a whole multi-node engine. See [Leader discovery](#leader-discovery) for worker configuration and readiness behavior.
- [`artifact-transfer-tp1.yaml`](artifact-transfer-tp1.yaml) - Small TP=1 vLLM example with `MX_ARTIFACT_TRANSFER=1`. Start one healthy source, then scale the same Deployment to two replicas to exercise Service discovery followed by pod-pinned artifact transfer.
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

The Service's Endpoints object is the source list, maintained by Kubernetes based on pod readiness. No `modelexpress-server` in this topology; `mx_source_id` is computed client-side and validated on the `GetTensorManifest` response. When `MX_ARTIFACT_TRANSFER=1`, device 0 also publishes pod-scoped cache artifacts. The target discovers a bundle through `mx-sources:6555`, then uses the returned Pod IP endpoint for all stateful chunk-lease RPCs.

## Prerequisites

1. Kubernetes cluster with GPU nodes. The YAMLs request `rdma/shared_ib` resources for InfiniBand/RoCE - the fast path and the configuration production should run on. Without RDMA, UCX/NIXL falls back to plain TCP at significant throughput cost; drop the `rdma/shared_ib` resource requests from the manifests to run without it.
2. A path for weights to reach the pods. Any of: pre-downloaded to a shared PVC, streamed from S3 (set `MX_MODEL_URI`), or downloaded from HuggingFace at pod startup. For the HuggingFace option, create the token secret with `kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=<token>`.
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
| `MX_K8S_LEADER_SERVICE`          | unset                           | Set to `mx-sources:6555` to use a single leader discovery endpoint for weights and artifacts. Supersedes both Service patterns. |
| `MX_K8S_SERVICE_PATTERN`         | `mx-sources`                    | DNS template. `{rank}` substitutes the worker's rank. If the pattern has no `:port`, the client auto-appends `:{MX_WORKER_GRPC_PORT + rank}`. |
| `MX_K8S_SOURCE_RETRIES`          | `5`                             | Max retries on `FAILED_PRECONDITION` before giving up.                         |
| `MX_K8S_SOURCE_BACKOFF_SECONDS`  | `0.5`                           | Sleep between retries (fresh channel per attempt).                             |
| `MX_MODEL_REVISION`              | unset                           | Override for `SourceIdentity.revision`. Useful for local / non-HF checkpoints. |
| `MX_WORKER_GRPC_PORT`            | `6555`                          | Base port for the WorkerGrpcServer (bound port is this + `device_id`).         |
| `MX_K8S_ARTIFACT_SERVICE_PATTERN`| inherits weight Service pattern | Optional node-scoped artifact Service template; supports `{node_rank}` and `{rank}`. |
| `MX_ARTIFACT_OWNER_DEVICE_ID`    | `0`                             | Worker that publishes pod-scoped artifacts; expose its gRPC port through the artifact Service. |
| `MX_ARTIFACT_TRANSFER`           | `0`                             | Set to `1` on source and target Pods to transfer compatible JIT caches.        |

## Leader discovery

Set `MX_K8S_LEADER_SERVICE=mx-sources:6555` on every source and target worker to
enable instance metadata discovery. Without this setting the original rank/node
Service patterns continue to work. All workers must use the new protocol; this
mode does not fall back to rank Services when a leader lacks the metadata RPC.

The Service selects only healthy instance leaders (global rank 0, local device
0), with a single port targeting that worker's gRPC port. Apply
[`leader-discovery-service.yaml`](leader-discovery-service.yaml) alongside your
existing LWS/engine manifests. Keep the engine's normal rendezvous networking;
no extra per-rank or per-node **MX discovery** Services are needed. Changing from
two nodes with eight GPUs each to another supported TP/PP topology does not
change the discovery Service.

Use the same immutable model revision, runtime versions and compatible TP/PP
configuration on sources and targets. Pin the actual engine checkpoint revision
as well as `MX_MODEL_REVISION`. Set `MX_WORKER_HOST` from `status.podIP`. Workers
still listen on `MX_WORKER_GRPC_PORT + local_device`; target workers and source
leaders must reach these Pod-direct ports and the NIXL fabric.

After loading, workers register their global rank, shard rank, node rank,
generation ID and actual bound endpoint with their **own** engine head. This
address comes from the load context (`master_addr`, SGLang's `dist_init_addr`, or
`LWS_LEADER_ADDRESS`), never from the load-balanced discovery Service. For
single-node engines the loopback fallback is sufficient. Multi-node engines must
provide a reachable head address, and use a fixed, common worker base port.
Registration and withdrawal reuse the weight publisher's READY/STALE heartbeats.
Registrations expire after `max(15, 3 * MX_HEARTBEAT_INTERVAL_SECS)` seconds; a
live rank cannot be overwritten by a different generation. A graceful STALE
withdrawal permits immediate replacement.

On `GetInstanceMetadata`, the leader checks engine `/health`, requires every
rank to have a live READY registration, then fetches each worker's lightweight
inventory directly. Worker generations are checked; an unreachable, stale or
restarted worker makes the directory unavailable. The leader queries its own
inventory locally, avoiding self-RPC deadlocks under concurrent targets. Remote
inventory collection has a five-second total budget. The response contains
endpoints and currently sealed artifact IDs/types, not MB-scale tensor manifests.

A target caches one directory snapshot per client/load and uses it for every
artifact type and weight lookup. Missing cache types immediately produce a
best-effort miss, without probing six artifact headers or repeating backoff for
each absent type. Existing types go directly to the listed owner on the matching
node; manifests, chunk leases and NIXL bytes stay on that endpoint. Header checks
reject mismatched identity, worker generation, node, type or accelerator before
creating target files. Weight requests also carry the selected worker ID.

Engine health and artifact publication are separate: a cache still being sealed
is absent from that snapshot and is skipped. Later targets see new publications;
the current target does not wait for optional caches to appear. Warm the source
and observe the desired artifact publication before scaling when testing cache
reuse. Discovery failures use the existing bounded retry policy once per client,
then subsequent lookups reuse that failure rather than multiplying startup delay.

This mode supports stable-weight, one-world TP/PP engines with unique model-shard
ranks. Independent target worker clients can select different compatible source
instances behind the Service; there is no target-wide donor-affinity collective.
Live expert placement/EPLB updates are outside this protocol. Central backends
and legacy rank/node discovery retain their existing behavior.

Transfer-domain enforcement also applies to leader discovery. Required policies
reject unknown local domains and nonmatching worker locations; directory entries
carry topology into weight source selection. Artifact headers recheck the domain
before creating target files, so a changed source location cannot bypass the
policy through a cached directory. Legacy artifact Services validate the same
header fields and fail closed when required topology is absent.

## GPU validation

On October 23, 2026, the leader discovery implementation was tested on
`nebius-2` with NVIDIA H200 GPUs, vLLM 0.26.0, PyTorch 2.11.0+cu130,
ModelExpress 0.8.0 and NIXL 1.3.1 (UCX). The model was `Qwen/Qwen3-0.6B`,
pinned to revision `c1899de289a04d12100db370d81485cdf75e47ca`. The image was
`nvcr.io/nvidian/dynamo-dev/zhongdongmin@sha256:2ea256bced645fe5614ca28c7056a29ad2dd35f37797d67f6db51297e8bac252`.

| Case | Source / target layout | Verified result |
| --- | --- | --- |
| TP1, eager | One GPU per Pod, source and target on different hosts | 339 tensors / 1.20 GB transferred by RDMA; Triton and FlashInfer installed; one directory lookup; matching inference |
| TP2 | Two GPUs per Pod, source and target on different hosts | Both ranks transferred about 0.61 GB each; 47.97 MiB torch.compile cache installed; both ranks directly loaded AOT functions; matching inference |
| PP2 / TP1 | Each instance spans two hosts, one GPU per Pod | Ranks 0/1 transferred 170/171 tensors, about 0.76 GB each; both node-specific markers matched; both Pods reused their installed AOT cache; matching inference |

Every case used one MX discovery Service with one port. Target workers fetched
one directory each. TP1 skipped four absent optional artifact types within the
same logged second, without per-type backoff. Real worker RPC checks also
confirmed that withdrawn membership is rejected, READY restores discovery, and
old worker IDs are rejected. All three test namespaces were deleted afterwards.

The distributed PP2 run used four Pods interleaved across two physical hosts:
source node 0 shared a host with target node 1, and source node 1 with target
node 0. Each corresponding rank's weights and artifacts crossed hosts. Other
idle nodes carried reservation taints and were not used. This run does not
qualify a two-node-by-eight-GPU topology or provide a startup-latency benchmark.

The reproducible runner is
[`ci/k8s/client/leader_discovery_gpu.py`](../../ci/k8s/client/leader_discovery_gpu.py).
It saves directory responses, Pod placement, per-Pod logs, inference responses
and `result.json` before cleanup. For example, from the repository root:

```bash
python ci/k8s/client/leader_discovery_gpu.py \
  --context nebius-2 --namespace mx-leader-my-test \
  --image nvcr.io/nvidian/dynamo-dev/zhongdongmin@sha256:2ea256bced645fe5614ca28c7056a29ad2dd35f37797d67f6db51297e8bac252 \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --tp 1 --pp 2 --nodes 2 \
  --pull-secret-namespace YOUR_NAMESPACE --pull-secret nvcr-secret \
  --output /tmp/mx-leader-gpu-results
```

The namespace must be new. For an explicitly retained test namespace, `--resume`
requires the runner's ownership label and matching image. The two instance
groups need separate engine rendezvous Services in addition to the single MX
discovery Service. To rebuild the validation image, use
[`Dockerfile.leader-test`](Dockerfile.leader-test) from the repository root.
