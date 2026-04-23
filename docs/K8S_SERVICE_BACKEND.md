<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# K8s-Service Metadata Backend

A decentralized metadata backend that uses Kubernetes Services for source discovery and load balancing, with no central coordinator. This document covers the *why* and the *design*; deployment steps live in [`DEPLOYMENT.md`](DEPLOYMENT.md#k8s-service-routed-backend) and the low-level contract lives in [`ARCHITECTURE.md`](ARCHITECTURE.md#k8s-service-metadata-backend).

## Limitations

The first thing to know: **this backend is for stable-weight inference deployments**. The weights loaded at pod startup don't change for the lifetime of the pod. Reach for the central-coordinator backends (`redis` or `kubernetes`) for anything outside that box.

What does NOT work on this backend:

- **RL-style live weight refits.** Training loops that produce a new checkpoint per step and broadcast to all inference pods without a redeploy. `WorkerGrpcServer` is bound to one `mx_source_id` at pod construction; there is no in-place swap path.
- **Hot swap between different models.** Same reason: the source identity is fixed at pod startup.
- **Mixed-revision serving during anything longer than a brief rolling update.** Transitions beyond `MX_K8S_SOURCE_RETRIES` worth of pods exhaust the retry loop.
- **Per-worker addressability.** The K8s Service selector is the granularity; there is no `worker_id`-level addressing through the gRPC call.

## Choosing a Metadata Backend

Pick based on workload, not operational preference. The choice has structural consequences.

| Workload shape                                                                                   | Backend                 | Why                                                                                                                                                                                                             |
|--------------------------------------------------------------------------------------------------|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stable-weight inference. Weights fixed at pod startup, no mid-life refit. Simple K8s deployment. | `k8s-service`           | Lowest deployment footprint. No server, no Redis, no CRDs. Matches the homogeneous pool assumption that Service routing requires.                                                                               |
| RL rollouts. Training loop updates weights every step, all inference pods refit in-place.        | `redis` or `kubernetes` | Central store tracks each worker's state individually by `worker_id`. Targets can fetch "worker W as it exists right now" instead of random-sampling a pool. Live refits stay consistent at the per-worker level. |
| Live fine-tune broadcasts. New checkpoint pushed to all replicas, hot-swapped in place.           | `redis` or `kubernetes` | Same reason as RL. The k8s-service backend can't swap a live pod's source_id without restarting the pod.                                                                                                        |
| Mixed-version fleet. Multiple revisions serving concurrently.                                     | `redis` or `kubernetes` | Central store indexes by `mx_source_id`, multiple identities coexist cleanly. k8s-service requires one Service pool per identity.                                                                               |
| Heterogeneous hardware. Callers match on topology (H100 vs B200, different TP degrees).          | `redis` or `kubernetes` | Central store carries per-worker metadata including identity fields; k8s-service's pool assumption requires all pods to be interchangeable.                                                                     |
| Multiple checkpoints in parallel (base + LoRA, fp16 + nvfp4, etc.).                              | Either                  | Different `SourceIdentity` produces different `mx_source_id`. Each identity gets its own Service (k8s-service) or its own source records (central). Both work.                                                   |

## Why This Backend Exists

The central-coordinator backends (`redis`, `kubernetes`) have been the defaults since MX started shipping. They support the full range of workloads by design - per-worker addressability, live weight updates, heterogeneous fleets. The cost is operational: you deploy and maintain a `modelexpress-server` pod, wire up Redis or K8s CRDs, and treat it as a first-class component of your stack.

A customer ask through the Slack channel (summary: "just expose a Service we can load-balance against, we don't need the rest") motivated a simpler path for the subset of deployments that don't need the rich central-coordinator semantics. The `k8s-service` backend is that path: no central store, no substrate advertisement, routing delegated entirely to kube-proxy, with `mx_source_id` validated at the GetTensorManifest handshake rather than at the metadata-store level.

## Design

### The decoupling principle

**Service names are deliberately decoupled from `mx_source_id`.** The Service name is a deployer-chosen string that lives in Kubernetes's namespace; `mx_source_id` is an internal MX value derived cryptographically from `SourceIdentity` fields (model, dtype, quantization, TP, revision, `mx_version`, proto schema). These two namespaces never reconcile automatically, and have no mechanism to stay in sync beyond operator discipline.

**It is the operator's responsibility** to make sure the Service their pods sit behind actually serves the identity their client is asking for.

That decoupling is the whole reason the backend is robust to library-side changes: `mx_version` bumps, `SourceIdentity` proto additions, canonical-JSON tweaks all shift `mx_source_id` without touching Service names, so a Helm chart written today keeps resolving after every future MX release. The cost is that the operator owns the alignment between what the Service is named and what identity its pods serve, and any misalignment has to be caught downstream rather than prevented by the naming scheme itself.

### The handshake as safety net

Every `GetTensorManifest` call passes an `mx_source_id`. If the client resolves its pattern, connects to a pod, and that pod's `WorkerServiceServicer` is serving a different `mx_source_id`, the server returns `FAILED_PRECONDITION`. The client retries on a fresh channel up to `MX_K8S_SOURCE_RETRIES` times so kube-proxy can route to a potentially-matching backend. The client also validates `resp.mx_source_id` and `resp.worker_rank` against the requested values before accepting the manifest, as defense-in-depth against empty-ID requests or misconfigured Service selectors that could slip past the server's check. Content mismatches fail loudly and give the caller a retry budget; wrong weights are never silently transferred.

### Rank encoding: hostname vs port

`MxK8sServiceClient` supports two deployment shapes via `MX_K8S_SERVICE_PATTERN`:

- **Pattern with explicit `:port`** (e.g. `mx-sources-rank-{rank}:6555`) - used verbatim after `{rank}` substitution. Rank is encoded in the hostname; caller hits one Service per rank, each with a label selector scoped to pods holding that rank. Fits the 1-GPU-per-pod topology.
- **Pattern without a port** (e.g. `mx-sources`, the default) - client auto-appends `:{MX_WORKER_GRPC_PORT + rank}`. Rank is encoded in the port; caller hits one Service with N named ports, each targeting the matching in-pod port. Fits the multi-GPU-per-pod topology where every pod has every rank.

The multi-GPU-per-pod shape is the one that works for heavy TP inference (NVLink is intra-node-only, so TP ranks have to share a pod). The 1-GPU-per-pod shape is useful for per-rank autoscaling or cross-pod setups where TP isn't involved.

## Alternatives Considered and Rejected

### `{mx_source_id}` as Service name

Using the `mx_source_id` hash as the Service name isn't viable from a UX perspective, even though it would superficially offer "wrong-identity traffic can't even reach the wrong Service" as a DNS-level guarantee. The cost is silent brittleness under any change that shifts `mx_source_id`:

- `mx_version` bumps on every MX release. A routine container-image bump changes every pod's computed source_id without the deployer touching their Helm values; the declared Service names go stale; resolution fails cluster-wide.
- `SourceIdentity` proto gaining fields (like `revision`, added as part of this backend). All pre-computed hashes shift by one schema revision. Every Helm chart in the wild becomes stale on the next release unless the deployer re-runs the hashing CLI and re-applies.
- `revision` / `dtype` / `quantization` / TP reshape. Rolling updates across one of those boundaries put half the pool under one name and half under another.
- No graceful degradation to "any pod serving the right identity" because the DNS layer hard-fails before the source_id handshake even runs.

Deployer-chosen names used today are strictly more robust to library-side drift, at the cost of the operator-alignment responsibility described above.

### Client-side endpoint enumeration

An alternative that does use K8s for discovery but at a finer granularity: have the client query the Service's `Endpoints` object, enumerate individual pods, filter by pod labels (e.g. rank, source_id) and status, and connect to a specific backend. This gives per-backend addressability without a central server.

Rejected for this backend because it undoes the "minimal infrastructure" property the customer specifically asked for. The client would need K8s API access (ServiceAccount, RBAC for `endpoints` or `endpointslices`), coupling to K8s API semantics rather than DNS alone. It's a meaningful backend on its own merits - worth considering as a separate `k8s-endpoints` backend variant if a customer needs it - but it's not the shape the k8s-service backend is aiming for.

## Relationship to Other Backends

### Versus central-coordinator (`redis`, `kubernetes`)

Server-coordinated backends give per-worker addressability. The client-facing RPCs are organized around `(mx_source_id, worker_id)` tuples: `ListSources(identity)` returns each live worker individually; `GetMetadata(mx_source_id, worker_id)` fetches that specific worker's current state. This lets a target pull from "worker W as it exists right now" - which is what RL rollouts, live fine-tune broadcasts, and mixed-version deployments rely on.

The `k8s-service` backend gives pool-level addressability only. There is no way through the gRPC call to say "I want worker W specifically, not whichever of the N ready pods kube-proxy happened to pick." That's a deliberate simplification - it's what lets the backend work with zero infrastructure beyond the Service itself - but it has hard consequences enumerated in the [Limitations](#limitations) section.

### Versus a hypothetical DHT backend

A libp2p-style DHT backend would also be decentralized, but would bring back per-peer addressability: each peer registers under a key like `hash(mx_source_id) + worker_id`, callers look up specific keys. That would erase the k8s-service limitations (RL rollouts, live refits, mixed-version fleets would all work) at the cost of bringing in bootstrap, consistency, and security-model complexity. Different point on the simplicity/capability curve.

## See Also

- [Deployment how-to](DEPLOYMENT.md#k8s-service-routed-backend) - concrete `kubectl apply` steps and env var reference.
- [Example manifests](../examples/k8s_service_sources/) - complete TP=2 source and target Deployments for both 1-GPU-per-pod and multi-GPU-per-pod shapes.
- [Architecture reference](ARCHITECTURE.md#k8s-service-metadata-backend) - low-level contract (RPCs, protocol, data shapes).
