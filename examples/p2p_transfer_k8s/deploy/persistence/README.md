# Persistence Backends for ModelExpress Server

By default, the ModelExpress server uses an **in-memory** metadata backend — fast, zero-dependency,
and sufficient for most deployments. If you need metadata to survive server restarts, you can
enable **write-through persistence** to Redis or Kubernetes CRDs.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                 ModelExpress Server                    │
│                                                       │
│   ┌─────────────────────────────────────────────┐    │
│   │          In-Memory Cache (always on)         │    │
│   │   Fast reads/writes via RwLock<HashMap>      │    │
│   └────────────────────┬────────────────────────┘    │
│                        │ write-through (optional)     │
│              ┌─────────┴──────────┐                  │
│              ▼                    ▼                   │
│   ┌──────────────────┐ ┌────────────────────┐        │
│   │   Redis Backend  │ │ Kubernetes CRD     │        │
│   │   (HA, fast)     │ │ Backend (native)   │        │
│   └──────────────────┘ └────────────────────┘        │
└──────────────────────────────────────────────────────┘
```

On startup, the server **hydrates** the in-memory cache from the persistent backend,
so it recovers state without sources needing to re-publish.

## When to Use Persistence

| Scenario | Recommended Backend |
|----------|-------------------|
| Development / testing | In-memory (default) |
| Single MX server, can tolerate re-publish on restart | In-memory (default) |
| Need HA — metadata survives MX server restarts | Redis write-through |
| Kubernetes-native, no Redis dependency | K8s CRD write-through |

## Files

| File | Purpose |
|------|---------|
| `modelexpress-server-redis.yaml` | MX server with Redis write-through backend |
| `modelexpress-server-kubernetes.yaml` | MX server with Kubernetes CRD backend |
| `redis-standalone.yaml` | Standalone Redis deployment with PVC persistence |
| `crd-modelmetadata.yaml` | Custom Resource Definition for model metadata |
| `rbac-modelmetadata.yaml` | RBAC roles for CRD access |
| `vllm-source-redis.yaml` | vLLM source configured for Redis backend |
| `vllm-target-redis.yaml` | vLLM target configured for Redis backend |

## Usage

### Redis Write-Through

```bash
# 1. Deploy Redis with persistent storage
kubectl apply -f persistence/redis-standalone.yaml

# 2. Deploy MX server with Redis write-through
kubectl apply -f persistence/modelexpress-server-redis.yaml

# 3. Deploy vLLM source/target (use Redis-aware YAMLs)
kubectl apply -f persistence/vllm-source-redis.yaml
kubectl apply -f persistence/vllm-target-redis.yaml
```

Set `MX_METADATA_BACKEND=redis` and `REDIS_URL=redis://redis:6379` on the server.

### Kubernetes CRD

```bash
# 1. Create CRD and RBAC
kubectl apply -f persistence/crd-modelmetadata.yaml
kubectl apply -f persistence/rbac-modelmetadata.yaml

# 2. Deploy MX server with K8s CRD backend
kubectl apply -f persistence/modelexpress-server-kubernetes.yaml
```

Set `MX_METADATA_BACKEND=kubernetes` on the server. Requires a ServiceAccount with
permissions to create/read/update ModelMetadata custom resources.
