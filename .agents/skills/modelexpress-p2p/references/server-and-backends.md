# Server And Metadata Backends

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Authoritative sources: `docs/DEPLOYMENT.md` (Choosing a Metadata Backend, P2P
Kubernetes Deployment), `docs/K8S_SERVICE_BACKEND.md`, `helm/README.md`.

## Choosing A Backend

| Backend | Use when | Server | Extra infrastructure |
|---|---|---|---|
| `redis` | Central discovery backed by Redis | Yes | A Redis instance you provide |
| `kubernetes` | Central discovery backed by Kubernetes custom resources | Yes | `ModelMetadata` and `ModelCacheEntry` CRDs, plus RBAC |
| `k8s-service` | Weights fixed for each pod's lifetime and every pod behind a Service serves the same checkpoint | No | Kubernetes Services in front of the source pods |

## Install The Server With Helm

Install the chart from `helm/` in a ModelExpress checkout. Its image defaults to
`nvcr.io/nvidia/ai-dynamo/modelexpress-server` at the chart's `appVersion`.
`MX_METADATA_BACKEND` is required; the server fails to start without it.

Redis backend. The chart does not bundle Redis. If you do not already run one,
`examples/p2p_transfer_k8s/server/redis_backend/redis-standalone.yaml` deploys
a single Redis instance behind a Service named `redis`, which the `REDIS_URL`
below points at:

```bash
kubectl apply -n "${NAMESPACE}" -f examples/p2p_transfer_k8s/server/redis_backend/redis-standalone.yaml
helm install modelexpress ./helm -n "${NAMESPACE}" \
  --set env.MX_METADATA_BACKEND=redis \
  --set env.REDIS_URL=redis://redis:6379
```

Kubernetes backend. Helm installs the chart's CRDs only when they are missing
and never updates existing ones, so apply them first on any cluster that
already has ModelExpress CRDs:

```bash
kubectl apply -f helm/crds/modelexpress-crds.yaml
helm install modelexpress ./helm -n "${NAMESPACE}" \
  --set env.MX_METADATA_BACKEND=kubernetes \
  --set serviceAccount.rbac.enabled=true
```

`serviceAccount.rbac.enabled` creates a ClusterRole and ClusterRoleBinding for
the metadata resources. The server writes them into `MX_METADATA_NAMESPACE`,
which defaults to the release namespace.

`helm/deploy.sh` wraps the same install with preset values files
(`values-development.yaml`, `values-production.yaml`,
`values-local-storage.yaml`); see `helm/README.md`.

## Health Check

The server listens for gRPC on port `8001`. For the `modelexpress` release
installed above:

```bash
kubectl -n "${NAMESPACE}" rollout status deployment/modelexpress
helm test modelexpress -n "${NAMESPACE}"
kubectl -n "${NAMESPACE}" logs -l app.kubernetes.io/name=modelexpress --tail=50
```

The liveness and readiness probes only check that the gRPC port accepts
connections. `helm test` runs `grpc-health-probe` against the standard
`grpc.health.v1` service, which is the stronger check. A server that starts
without a valid backend setting exits; look for the reason in its logs.

## Point Workers At The Server

Workers use `MX_SERVER_ADDRESS=<service>:8001`. A worker in another namespace
must qualify the name: `<service>.<server-namespace>.svc.cluster.local:8001`.
All workers that use one server share one source pool, so workers in different
namespaces can be sources for each other; a separate server per namespace
splits the pool.

`MODEL_EXPRESS_URL` is a deprecated alias that takes precedence over
`MX_SERVER_ADDRESS` when both are set. Set it to the same value whenever
anything else may inject it.

Workers talk to the server over gRPC only; they need no Kubernetes RBAC for the
metadata resources, even with the `kubernetes` backend.

## `k8s-service` Backend

No server, Redis, or CRDs. Set `MX_METADATA_BACKEND=k8s-service` on the
workers and put the source pods behind a Service matched by
`MX_K8S_SERVICE_PATTERN`. Start from `examples/k8s_service_sources/` and read
`docs/K8S_SERVICE_BACKEND.md` for its limits.

## Stale Sources

A source goes `STALE` when its worker exits cleanly or stops heartbeating for
`MX_HEARTBEAT_TIMEOUT_SECS`, and the reaper deletes it later
(`docs/metadata.md`, Source Lifecycle). To inspect sources, list
`ModelMetadata` resources (`kubernetes` backend) or the scoped Redis keys
(`redis` backend). Do not run `FLUSHALL` on a shared Redis.
