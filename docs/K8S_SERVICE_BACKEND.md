<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# K8s-Service Metadata Backend

A decentralized metadata backend that uses Kubernetes Services for source discovery and load balancing, with no central coordinator. This document covers the *why* and the *design*; deployment steps live in [`DEPLOYMENT.md`](DEPLOYMENT.md#k8s-service-routed-backend) and the low-level contract lives in [`ARCHITECTURE.md`](ARCHITECTURE.md#k8s-service-metadata-backend).

## Limitations

The first thing to know: **this backend is for stable-weight inference deployments**. The weights loaded at pod startup don't change for the lifetime of the pod. Workflows that require per-worker addressability need a central-coordinator backend (`redis` or `kubernetes`); receiver-driven RL refit and live fine-tune broadcast are still under development.

What does NOT work on this backend:

- **RL-style live weight refits.** Training loops that produce a new checkpoint per step and broadcast to all inference pods without a redeploy. `WorkerGrpcServer` is bound to one `mx_source_id` at pod construction; there is no in-place swap path.
- **Hot swap between different models.** Same reason: the source identity is fixed at pod startup.
- **Mixed-revision serving during anything longer than a brief rolling update.** Transitions beyond `MX_K8S_SOURCE_RETRIES` worth of pods exhaust the retry loop.
- **Per-worker addressability.** The K8s Service selector is the granularity; there is no `worker_id`-level addressing through the gRPC call.
- **Adaptive expert placement (EPLB-style MoE).** MoE deployments where the expert-to-rank assignment shifts at runtime based on traffic load. The rank-based addressing scheme assumes each rank holds a deterministic shard known at deploy time; live placement reshuffles break that. No MX backend supports this case today; static expert placement does work, see [Expert parallelism: static vs adaptive](#expert-parallelism-static-vs-adaptive).

## Choosing a Metadata Backend

Pick based on workload, not operational preference. The choice has structural consequences.

| Workload shape                                                                                   | Backend                 | Why                                                                                                                                                                                                             |
|--------------------------------------------------------------------------------------------------|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stable-weight inference. Weights fixed at pod startup, no mid-life refit. Simple K8s deployment. | `k8s-service`           | Lowest deployment footprint. No server, no Redis, no CRDs. Matches the homogeneous pool assumption that Service routing requires.                                                                               |
| MoE inference with static expert placement. Experts pinned to specific ranks at deploy time, no live rebalancing. | `k8s-service`           | Rank-based addressing extends to the `(TP, EP)` coordinate space; declare one Service per coordinate, or one Service with `TP * EP` named ports for the multi-GPU-per-pod shape. See [Expert parallelism: static vs adaptive](#expert-parallelism-static-vs-adaptive). |
| Future RL refit workflows (under development). Training updates weights every step and rollout workers need per-worker sources. | `redis` or `kubernetes` | The central store provides the per-worker addressability required by the planned receiver-driven refit workflow. Selecting this backend does not enable end-to-end live refit today. |
| Future live fine-tune broadcasts (under development). New checkpoints are pushed to running replicas. | `redis` or `kubernetes` | These workflows require the same per-worker addressability. The `k8s-service` backend cannot swap a live pod's `mx_source_id` without restarting the pod. |
| Mixed-version fleet. Multiple revisions serving concurrently.                                     | `redis` or `kubernetes` | Central store indexes by `mx_source_id`, multiple identities coexist cleanly. k8s-service requires one Service pool per identity.                                                                               |
| Heterogeneous hardware. Callers match on topology (H100 vs B200, different TP degrees).          | `redis` or `kubernetes` | Central store carries per-worker metadata including identity fields; k8s-service's pool assumption requires all pods to be interchangeable.                                                                     |
| Multiple checkpoints in parallel (base + LoRA, fp16 + nvfp4, etc.).                              | Either                  | Different `SourceIdentity` produces different `mx_source_id`. Each identity gets its own Service (k8s-service) or its own source records (central). Both work.                                                   |

## Why This Backend Exists

The central-coordinator backends (`redis`, `kubernetes`) have been the defaults since MX started shipping. They provide per-worker addressability and support heterogeneous and mixed-version fleets. They are also the required metadata foundation for receiver-driven live-weight workflows under development. The cost is operational: you deploy and maintain a `modelexpress-server` pod, wire up Redis or K8s CRDs, and treat it as a first-class component of your stack.

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

### Expert parallelism: static vs adaptive

The rank encoding scheme above addresses each rank deterministically through DNS or port number. For MoE inference with expert parallelism, this extends directly when the expert-to-rank assignment is fixed at deploy time:

- Each `(TP, EP)` coordinate maps to one rank using the standard `rank = ep_idx * tp_size + tp_idx` convention (or whatever convention the inference runtime uses).
- Each rank holds a known subset of experts. The operator declares one Service per `(TP, EP)` coordinate, or one Service with `TP * EP` named ports for the multi-GPU-per-pod shape.
- Clients resolve `MX_K8S_SERVICE_PATTERN` against their own rank to reach the pods serving their shard.

This works because each rank's content is stable for the lifetime of the deployment. Pods within a rank pool remain interchangeable replicas; the homogeneous-pool assumption holds at the per-rank level.

What does NOT extend cleanly is *adaptive* expert placement (EPLB-style). When the inference runtime monitors expert activation frequency and shifts hot experts across ranks at runtime, the `(rank -> expert subset)` mapping is no longer static. A request for "the rank holding expert N" can't be answered by a fixed rank-to-Service pattern, because which rank holds expert N changes over time. Solving this requires a content-aware backend that queries pods for what they currently hold. The closest existing shape is the rejected `k8s-endpoints` variant in [Alternatives Considered and Rejected](#alternatives-considered-and-rejected), with a `WorkerService.ListContents` RPC layered on top.

The k8s-service backend deliberately does not address adaptive placement; static expert placement is what's in scope.

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

Server-coordinated backends give per-worker addressability. The client-facing RPCs are organized around `(mx_source_id, worker_id)` tuples: `ListSources(identity)` returns each live worker individually; `GetMetadata(mx_source_id, worker_id)` fetches that specific worker's current state. This lets a target pull from "worker W as it exists right now," which mixed-version deployments use today and planned RL refit and live fine-tune workflows will require.

The `k8s-service` backend gives pool-level addressability only. There is no way through the gRPC call to say "I want worker W specifically, not whichever of the N ready pods kube-proxy happened to pick." That's a deliberate simplification - it's what lets the backend work with zero infrastructure beyond the Service itself - but it has hard consequences enumerated in the [Limitations](#limitations) section.

## See Also

- [Deployment how-to](DEPLOYMENT.md#k8s-service-routed-backend) - concrete `kubectl apply` steps and env var reference.
- [Example manifests](../examples/k8s_service_sources/) - complete TP=2 source and target Deployments for both 1-GPU-per-pod and multi-GPU-per-pod shapes.
- [Architecture reference](ARCHITECTURE.md#k8s-service-metadata-backend) - low-level contract (RPCs, protocol, data shapes).
