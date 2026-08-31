<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Troubleshooting

Start with the worker's loader decision and the server/backend health. ModelExpress logs the eligible strategies before it tries them, so most startup problems can be classified without guessing at the network path.

## Fast triage

| Symptom | First check | Likely cause |
|---|---|---|
| The worker starts, but P2P is not used | Eligible loaders in the worker log | NIXL, metadata, adapter, or model-identity gate excluded rdma |
| The server will not start | MX_METADATA_BACKEND and its connection variables | Missing or invalid backend configuration |
| The worker cannot resolve the model offline | Runtime model path and non-weight files | P2P transfers weights, not repository metadata |
| model_streamer is absent | MX_MODEL_URI and runai_model_streamer | URI or package gate is missing |
| P2P falls back after a timeout | Source/target identity and NIXL/UCX logs | Fabric reachability, stale source, or incompatible manifest |
| S3 loading fails | MX_MODEL_URI, region, and worker credentials | Wrong prefix, credential chain, or object layout |
| The worker is ready but artifacts are not reused | MX_ARTIFACT_TRANSFER, metadata mode, and cache paths | Artifact transfer is opt-in and compatibility-scoped |

## Inspect the loading decision

Set the runtime log level before starting the worker:

~~~bash
export MODEL_EXPRESS_LOG_LEVEL=DEBUG
~~~

Look for:

~~~text
Eligible loaders: [...]
Trying strategy: rdma
Trying strategy: model_streamer
Weights loaded from disk
~~~

The strategy names are rdma, server-cache, instant_tensor, model_streamer, gds, and default. If the expected name is missing from Eligible loaders, changing timeout or retry settings will not enable it; fix the eligibility condition first. See [Configuration](CONFIGURATION.md#loading-strategy-selection).

## Server and metadata backend

The server speaks gRPC on port 8001 by default. Its Prometheus endpoint is separate, on port 9401 by default. A browser or curl http://localhost:8001/health is not a valid gRPC health check.

~~~bash
modelexpress-cli health --endpoint http://localhost:8001
nc -vz localhost 8001
curl -s http://localhost:9401/metrics | head
~~~

For Redis:

~~~bash
export MX_METADATA_BACKEND=redis
export REDIS_URL=redis://redis:6379
modelexpress-cli health --endpoint http://localhost:8001
~~~

For Kubernetes:

~~~bash
export MX_METADATA_BACKEND=kubernetes
export POD_NAMESPACE=modelexpress
kubectl get modelmetadatas -n modelexpress
kubectl get modelcacheentries -n modelexpress
~~~

If the server reports a missing backend variable, do not add a localhost fallback in production. Set REDIS_URL or the Kubernetes namespace explicitly and make sure the CRDs and RBAC are installed.

## P2P is not selected

Confirm these conditions on both source and target:

1. The worker image contains the ModelExpress Python client and the runtime integration.
2. NIXL is importable and initialized on a supported accelerator.
3. MX_SERVER_ADDRESS points to the central server, or MX_METADATA_BACKEND=k8s-service points to the correct Service pattern.
4. The source and target use matching model revision, parallelism, dtype, quantization, and accelerator identity.
5. The first replica reached application readiness and published metadata before the target started.
6. The worker has the RDMA or fabric resources and matching UCX/NIXL environment.

For a central backend, inspect source state in the server's backend. For the Kubernetes backend:

~~~bash
kubectl get modelmetadatas -A
kubectl describe modelmetadata -n <namespace> <name>
kubectl logs -n <namespace> deploy/modelexpress-server
~~~

For the Redis backend, inspect the Redis keys using the Redis deployment's own tooling. Do not run FLUSHALL on a shared production Redis instance; use the scoped cleanup operation for the affected model or namespace.

If P2P is eligible but transfer fails, ModelExpress can retry another source and fall back to a later strategy. Compare the SourceIdentity fields and mx_source_id before tuning timeouts. A target-side failure does not automatically mean the source is unhealthy.

## No shared storage

There are two different no-shared-storage cases:

- P2P no-shared-storage: targets receive weight tensors from a ready source. The target still needs model configuration and non-weight repository files.
- Server-backed no-shared-storage: set MODEL_EXPRESS_NO_SHARED_STORAGE=1 and MODEL_EXPRESS_URL or MX_SERVER_ADDRESS so the server installs repository files and supplies weights into a worker-local cache.

For server-backed loading, point MODEL_EXPRESS_CACHE_DIRECTORY and HF_HUB_CACHE at the same writable path. The server needs upstream access and a token for gated repositories; the worker does not. See [the deployment guide](DEPLOYMENT.md#server-backed-model-cache-no-shared-storage).

## ModelStreamer and S3

For direct S3 loading, verify the URI and credentials inside the worker:

~~~bash
echo "$MX_MODEL_URI"
echo "$AWS_DEFAULT_REGION"
export BUCKET=my-bucket
python -c 'import boto3, os; print(boto3.client("s3").head_bucket(Bucket=os.environ["BUCKET"]))'
~~~

The last command checks bucket access only; replace `my-bucket` with the actual bucket and use the provider's normal CLI or SDK check as appropriate. ModelStreamer expects safetensor files and their index/configuration in the configured prefix. A ModelExpress server is not part of direct streaming.

For SGLang, keep the model identity in --model-path and the storage URI in MX_MODEL_URI. Putting s3://... in --model-path bypasses ModelExpress and selects SGLang's native loader.

## SGLang

Confirm the upstream delegation flag exists:

~~~bash
python -m sglang.launch_server --help | grep modelexpress-config
~~~

Use --load-format remote_instance, --remote-instance-weight-loader-backend modelexpress, and a JSON --modelexpress-config with transport set to nixl or transfer_engine. The complete command and image requirements are in [SGLang](integrations/runtimes/sglang.md) and [docs/SGLANG.md](SGLANG.md).

## Kubernetes Service backend

The k8s-service backend does not use a ModelExpress server. Check the Service endpoints and rank mapping:

~~~bash
kubectl get svc mx-sources
kubectl get endpoints mx-sources
kubectl describe svc mx-sources
~~~

Use MX_K8S_SERVICE_PATTERN=mx-sources for the multi-GPU-per-pod shape, where the client adds MX_WORKER_GRPC_PORT + rank. Use an explicit pattern such as mx-sources-rank-{rank}:6555 for the one-GPU-per-pod shape. A wrong Service selector or port can produce FAILED_PRECONDITION; the client retries on a fresh channel so kube-proxy can choose another backend.

This backend assumes stable weights and interchangeable pods. Use Redis or Kubernetes CRDs for mixed revisions, live updates, live refit, or per-worker source selection.

## Artifact transfer

Artifact transfer is disabled unless MX_ARTIFACT_TRANSFER=1. It also requires MX_P2P_METADATA=1, a central coordinator, writable staging/cache directories, and a compatible artifact identity. Set MX_ARTIFACT_READY_URL when the runtime's readiness endpoint is not the default. Artifacts can contain executable code, so restrict the worker and manifest endpoints to trusted callers.

## Kubernetes rollout failures

Check the image, secret, ServiceAccount, GPU/fabric resources, and the first source pod before scaling:

~~~bash
kubectl describe pod -n <namespace> <pod>
kubectl logs -n <namespace> <pod> -c <runtime-container>
kubectl get events -n <namespace> --sort-by=.lastTimestamp
kubectl rollout status -n <namespace> deploy/<deployment>
~~~

The checked-in manifests are the most reliable starting point because they include the required downward-API identity and port settings. Compare a customized manifest with [the vLLM P2P example](../examples/p2p_transfer_k8s/client/vllm/vllm-single-node.yaml) or the [SGLang example](../examples/p2p_transfer_k8s/client/sglang/sglang-single-node-p2p.yaml).

## Validate configuration before a rollout

~~~bash
MX_METADATA_BACKEND=redis REDIS_URL=redis://localhost:6379 cargo run --bin modelexpress-server -- --config model-express.yaml --validate-config
helm lint ./helm
cmp examples/crds.yaml helm/crds/modelexpress-crds.yaml
~~~

The main CI workflow runs Rust, Python, generated-protobuf, Helm CRD-sync, and integration checks. The Kubernetes CI workflow runs the GPU and orchestrator scenarios listed on [Compatibility](COMPATIBILITY.md#ci-coverage-map).
