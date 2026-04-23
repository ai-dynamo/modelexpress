# ModelExpress Server Deployment

The ModelExpress server requires a persistent metadata backend. Two options are supported:

## Backends

| Backend | Directory | When to Use |
|---------|-----------|-------------|
| **Redis** | [`redis_backend/`](redis_backend/) | General deployment, need HA, low-latency |
| **Kubernetes CRD** | [`kubernetes_backend/`](kubernetes_backend/) | K8s-native, no external dependency |

## Redis Backend

```bash
kubectl apply -f redis_backend/redis-standalone.yaml
kubectl apply -f redis_backend/modelexpress-server-redis.yaml
```

Set `MX_METADATA_BACKEND=redis` and `REDIS_URL=redis://redis:6379` on the server.

## Kubernetes CRD Backend

```bash
# Consolidated CRD bundle: installs both the ModelMetadata (P2P) and
# ModelCacheEntry (registry) CRDs in one apply.
kubectl apply -f ../../crds.yaml
kubectl apply -f kubernetes_backend/rbac-modelmetadata.yaml -n <namespace>
kubectl apply -f kubernetes_backend/modelexpress-server-kubernetes.yaml -n <namespace>
```

Set `MX_METADATA_BACKEND=kubernetes` on the server. Requires a ServiceAccount with
permissions to create/read/update both `ModelMetadata` (P2P metadata) and
`ModelCacheEntry` (model registry) custom resources; see `rbac-modelmetadata.yaml`
for the full rule set.
