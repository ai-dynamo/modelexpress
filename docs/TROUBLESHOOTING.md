<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Troubleshooting

Start with the worker's loader decision, then check the dependency used by that path.

| Symptom | First check |
|---|---|
| P2P is not attempted | `Eligible loaders` in the worker log |
| P2P times out or falls back | Source identity, source readiness, and NIXL/UCX logs |
| ModelStreamer is absent | `MX_MODEL_URI` and the `runai_model_streamer` package |
| S3 loading fails | URI, region, credentials, and object layout inside the worker |
| Offline worker cannot resolve the model | Model configuration, tokenizer, and other non-weight files |
| Server does not start | `MX_METADATA_BACKEND` and its required connection settings |
| Weights load but artifacts are rebuilt | `MX_ARTIFACT_TRANSFER`, metadata mode, and writable cache paths |

## Loader selection

Set `MODEL_EXPRESS_LOG_LEVEL=DEBUG` and look for:

```text
Eligible loaders: [...]
Trying strategy: rdma
Trying strategy: model_streamer
```

The strategy names are `rdma`, `server-cache`, `instant_tensor`, `model_streamer`, `gds`, and `default`. If a path is absent from `Eligible loaders`, changing retries will not enable it; fix its package, adapter, device, URI, or metadata requirement first. See [Configuration](CONFIGURATION.md#loading-strategy-selection).

## Server and metadata

The server uses gRPC on port `8001` by default and exposes Prometheus metrics separately on `9401`. Do not use a browser request to `/health` as a gRPC health check.

```bash
modelexpress-cli health --endpoint http://localhost:8001
nc -vz localhost 8001
curl -s http://localhost:9401/metrics | head
```

Redis requires `REDIS_URL`, or both host and port settings. Kubernetes requires the CRDs, RBAC, and `POD_NAMESPACE` or `MX_METADATA_NAMESPACE`. Inspect `ModelMetadata` resources or the scoped Redis keys for source state; do not run `FLUSHALL` on shared production Redis.

## P2P

Confirm that source and target have the ModelExpress runtime integration, NIXL and fabric resources, a reachable metadata path, compatible model revision and parallelism, and matching dtype, quantization, and accelerator identity. The first replica must reach readiness and publish before the target starts.

```bash
kubectl get modelmetadatas -A
kubectl logs -n <namespace> deploy/modelexpress-server
kubectl logs -n <namespace> <target-pod> -c <runtime-container>
```

If P2P is eligible but fails, compare `SourceIdentity` and `mx_source_id` before tuning timeouts. ModelExpress can retry another source and then fall through to a later eligible strategy.

## No shared storage

P2P transfers weight tensors, not the whole model repository. Targets still need configuration, tokenizer, and other non-weight files. Server-backed mode instead uses `MODEL_EXPRESS_NO_SHARED_STORAGE=1` with `MODEL_EXPRESS_URL` or `MX_SERVER_ADDRESS` to install repository files and weights in a worker-local cache. Point `MODEL_EXPRESS_CACHE_DIRECTORY` and `HF_HUB_CACHE` at the same writable path. See [Deployment](DEPLOYMENT.md#server-backed-model-cache-no-shared-storage).

## ModelStreamer and object storage

Verify `MX_MODEL_URI` and storage access from inside the runtime container. The prefix must contain the expected safetensors and index/configuration files. Direct ModelStreamer does not use the ModelExpress server.

For SGLang, keep the model identity in `--model-path` and the object URI in `MX_MODEL_URI`. Passing the object URI as `--model-path` selects SGLang's native loader. Use the checked-in [vLLM](../examples/model_streamer_k8s/client/vllm/README.md) or [SGLang](../examples/model_streamer_k8s/client/sglang/README.md) examples as the baseline.

## Kubernetes rollout

Check the image, secrets, ServiceAccount, GPU/fabric resources, Service endpoints, and first source pod before scaling.

```bash
kubectl describe pod -n <namespace> <pod>
kubectl get events -n <namespace> --sort-by=.lastTimestamp
kubectl rollout status -n <namespace> deploy/<deployment>
```

For `k8s-service`, also check the source Service selectors, endpoints, and rank-to-port mapping. Use a central Redis or Kubernetes backend for mixed revisions, live updates, or per-worker source selection.

## Artifact transfer

Artifact transfer requires `MX_ARTIFACT_TRANSFER=1`, `MX_P2P_METADATA=1`, a central coordinator, compatible artifact identity, and writable staging/cache directories. Restrict worker and manifest endpoints to trusted callers because transferred caches may contain executable code.
