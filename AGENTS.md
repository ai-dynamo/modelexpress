---
name: modelexpress-user-guide
description: Help users understand, deploy, operate, and debug public ModelExpress. Use for questions about MX concepts, quick starts, Kubernetes, Helm, metadata backends, engine integrations, ModelStreamer, P2P transfer, and troubleshooting.
license: Apache-2.0
compatibility: Applies to the public ai-dynamo/modelexpress repository.
metadata:
  author: ModelExpress maintainers
  version: "1.0"
---

# ModelExpress guide for user-facing agents

Use this file when helping a user get ModelExpress running or debug a
deployment. Prefer clear operational guidance over internal implementation
details.

## What ModelExpress does

ModelExpress manages model weights for LLM inference:

- downloads or resolves model artifacts from Hugging Face, object storage, or
  local/PVC-backed paths
- tracks model lifecycle and worker metadata with Redis or Kubernetes CRDs
- lets new inference replicas receive weights from an already-loaded replica
  through GPU-to-GPU P2P transfer
- can also run ModelStreamer paths that stream from object storage without an
  MX coordination server

ModelExpress can run standalone. Dynamo is one integration path, not a
requirement. vLLM uses `--load-format modelexpress`; SGLang uses
`remote_instance` with the `modelexpress` backend; TensorRT-LLM P2P is a beta
path using the documented patch/runtime flow.

## Route users by goal

| User goal | Send them to | Notes |
|-----------|--------------|-------|
| Try MX locally | [`README.md#quick-start`](README.md#quick-start), [`docs/CLI.md`](docs/CLI.md) | Good first path; basic health/cache commands do not require GPUs. |
| Deploy MX server on Kubernetes | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md), [`helm/README.md`](helm/README.md) | Choose Redis or Kubernetes CRD metadata before deploying. |
| Speed up vLLM scale-out with P2P | [`examples/p2p_transfer_k8s/README.md`](examples/p2p_transfer_k8s/README.md) | Requires MX server, metadata backend, GPU workers, and RDMA-capable networking for the fast path. |
| Use SGLang | [`docs/SGLANG.md`](docs/SGLANG.md) | Use an SGLang image with the ModelExpress delegation hook; install MX with `--no-deps`. |
| Evaluate TensorRT-LLM P2P | [`examples/p2p_transfer_k8s/client/trtllm/`](examples/p2p_transfer_k8s/client/trtllm/) | Treat as beta and keep the TRT-LLM/Dynamo runtime requirements visible. |
| Stream from object storage | [`examples/model_streamer_k8s/README.md`](examples/model_streamer_k8s/README.md) | ModelStreamer does not require MX server or RDMA by itself. |
| Use Dynamo | [`examples/dynamo_model_cache_k8s/README.md`](examples/dynamo_model_cache_k8s/README.md), [`examples/dynamo_p2p_transfer_k8s/README.md`](examples/dynamo_p2p_transfer_k8s/README.md) | Explain Dynamo as optional orchestration around MX. |
| Check support and tested versions | [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md), [`ci/TEST_PLAN.md`](ci/TEST_PLAN.md) | Do not invent version pins; derive them from these files and Dockerfiles. |

## Deployment questions to answer first

Before recommending a deployment, identify:

- runtime: vLLM, SGLang, TensorRT-LLM, Dynamo, or standalone CLI/server
- environment: local Docker, Kubernetes, Helm, DynamoGraphDeployment, or cloud
  managed Kubernetes
- weight source: Hugging Face, S3, GCS, Azure Blob Storage, local disk, or PVC
- metadata backend: Redis, Kubernetes CRD, or `k8s-service`
- networking: InfiniBand/RoCE, AWS EFA, or no RDMA
- scaling shape: single replica, replica scale-out, tensor parallelism,
  live-refit/RL, or stable-weight serving

### Metadata backend guidance

- Use `redis` or `kubernetes` for coordinated fleets, live refits, RL loops,
  mixed revisions, or heterogeneous workers.
- Use `k8s-service` only for stable-weight inference where all pods behind a
  Service serve the same checkpoint and avoiding a central MX server is the
  main simplification.
- For Redis, set `MX_METADATA_BACKEND=redis` and `REDIS_URL`.
- For Kubernetes CRD, apply `examples/crds.yaml`, configure RBAC, and set
  `MX_METADATA_BACKEND=kubernetes` plus namespace wiring.

See [`docs/DEPLOYMENT.md#choosing-a-metadata-backend`](docs/DEPLOYMENT.md#choosing-a-metadata-backend).

## Important environment variables

| Variable | When to mention |
|----------|-----------------|
| `MX_METADATA_BACKEND` | Required server-side for Redis/Kubernetes metadata. |
| `REDIS_URL` | Required when the metadata backend is Redis. |
| `MX_METADATA_NAMESPACE` | Kubernetes CRD namespace override. |
| `MX_SERVER_ADDRESS` | Recommended client gRPC endpoint for central-server P2P paths. |
| `MODEL_EXPRESS_URL` | Deprecated, but still needed by some legacy/client paths; set both during transition when docs say so. |
| `VLLM_PLUGINS=modelexpress` | vLLM plugin registration when needed by the launch path. |
| `MX_MODEL_URI` | Enables ModelStreamer storage loading. |
| `MX_NIXL_BACKEND` | Use `UCX` for InfiniBand/RoCE; use `LIBFABRIC` on AWS EFA examples. |
| `MX_RDMA_NIC_PIN` | Use for NIC pinning or `auto` topology selection on multi-NIC RDMA hosts. |
| `MODEL_EXPRESS_LOG_LEVEL` | Set to `DEBUG` for more detailed MX Python logs. |

## Debugging playbook

Start with the failure surface: server unreachable, model cache issue, metadata
issue, image/config issue, or P2P/RDMA issue.

### Local CLI and server

```bash
modelexpress-cli -vv health
nc -vz localhost 8001
modelexpress-cli model status
modelexpress-cli model validate
modelexpress-cli model stats --detailed
```

The MX server speaks gRPC, not REST; `curl http://localhost:8001/health` is not
a valid health check.

### Kubernetes server and metadata

```bash
kubectl -n $NAMESPACE logs -f deploy/modelexpress-server
kubectl -n $NAMESPACE describe pod -l app=modelexpress-server

# Redis backend
kubectl -n $NAMESPACE exec deploy/modelexpress-server -c redis -- redis-cli KEYS 'mx:source:*'
kubectl -n $NAMESPACE exec deploy/modelexpress-server -c redis -- redis-cli HGETALL 'mx:source:<source_id>'

# Kubernetes CRD backend
kubectl -n $NAMESPACE get modelmetadatas
kubectl -n $NAMESPACE get modelcacheentries
```

If stale Redis metadata is suspected after redeploy, flushing Redis is a valid
debug step, but call out that it clears MX metadata:

```bash
kubectl -n $NAMESPACE exec deploy/modelexpress-server -c redis -- redis-cli FLUSHALL
```

### Inference worker and P2P

```bash
kubectl -n $NAMESPACE logs -f deploy/mx-vllm
kubectl -n $NAMESPACE exec deploy/mx-vllm -- curl -s http://localhost:8000/v1/models
kubectl -n $NAMESPACE exec deploy/mx-vllm -c vllm -- ibstat
kubectl -n $NAMESPACE exec deploy/mx-vllm -c vllm -- ucx_info -d
```

Look for logs that indicate loader registration, native source load, metadata
publish, source discovery, transfer start, and transfer completion. If a target
falls back to storage, check whether a READY source exists, whether the
`mx_source_id` identity matches, and whether source metadata is stale.

### Storage and ModelStreamer

For ModelStreamer, confirm `MX_MODEL_URI` is set and the pod has the right
cloud identity or secret. Expected vLLM logs include the ModelExpress loader
registration, `Trying strategy: model_streamer`, and a storage streaming
completion message. See [`examples/model_streamer_k8s/README.md`](examples/model_streamer_k8s/README.md).

## Answering standards

- Give the shortest viable path first, then link to deeper docs.
- Include prerequisites before commands: GPU/RDMA requirements, secrets, CRDs,
  RBAC, image build/push, or cloud credentials.
- Be explicit about status: supported, beta, experimental, example-only, or
  not in CI.
- Keep public answers free of private repositories, internal registries,
  internal runner names, and secrets.
- When editing docs, keep README for orientation and put detailed support,
  compatibility, and version pins in `docs/COMPATIBILITY.md`.

## Sources to verify before changing docs

- [`README.md`](README.md) for overview and first-run paths.
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for server, Kubernetes, metadata,
  P2P, ModelStreamer, and debugging details.
- [`docs/CLI.md`](docs/CLI.md) for CLI commands and troubleshooting.
- [`docs/SGLANG.md`](docs/SGLANG.md) for SGLang-specific launch guidance.
- [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) and
  [`ci/TEST_PLAN.md`](ci/TEST_PLAN.md) for support status and tested versions.
- [`examples/`](examples/) for copy/paste deployment manifests.

For docs-only edits, run:

```bash
git diff --check
find AGENTS.md README.md docs CONTRIBUTING.md helm/README.md examples -name '*.md' -print0 \
  | xargs -0 awk 'BEGIN{bad=0} /^```/{count[FILENAME]++} END{for (f in count) if (count[f] % 2) {print f ": odd fence count " count[f]; bad=1} exit bad}'
helm lint helm
```
