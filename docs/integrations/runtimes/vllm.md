<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM

Use ModelExpress to load the first vLLM replica from a checkpoint, then start compatible replicas by copying its GPU weights. The Kubernetes walkthrough below uses a small public model, `Qwen/Qwen3-0.6B`, with one GPU per replica.

## Start a source on Kubernetes

You need Kubernetes 1.27 or newer, two GPU nodes with one available NVIDIA GPU each, and RDMA connectivity between workers. The manifest requests `nvidia.com/gpu` and `rdma/ib`; adapt the RDMA resource name to your cluster's device plugin. It places replicas on separate nodes to test network transfer. The first replica downloads its checkpoint from Hugging Face. Both replicas need access to configuration and tokenizer files. For a fully offline deployment, use the [no-shared-storage setup](../../DEPLOYMENT.md#server-backed-model-cache-no-shared-storage) or package those files locally.

Build server and worker images from the same repository checkout, then push them to a registry your cluster can pull:

```bash
docker build -f docker/Dockerfile \
  -t registry.example.com/modelexpress-server:quickstart .
docker build -f examples/p2p_transfer_k8s/client/vllm/Dockerfile \
  -t registry.example.com/modelexpress-vllm:quickstart .
docker push registry.example.com/modelexpress-server:quickstart
docker push registry.example.com/modelexpress-vllm:quickstart
```

Replace `registry.example.com` with your registry. The [Dockerfile](../../../examples/p2p_transfer_k8s/client/vllm/Dockerfile) starts from `vllm/vllm-openai:v0.23.0` and installs the checked-out MX client. NIXL is supplied by the runtime image, not by the MX Python dependency list; confirm the image has a compatible NIXL/CUDA stack. vLLM 0.23.0 supports `--load-format modelexpress` natively. Older supported images use the MX plugin with `VLLM_PLUGINS=modelexpress`; `mx` remains a compatibility alias. Check [Compatibility](../../COMPATIBILITY.md) when changing runtime versions.

Open [`quickstart.yaml`](../../../examples/p2p_transfer_k8s/quickstart.yaml), replace both `your-registry/...:TAG` image references, and adapt GPU/fabric resources to your cluster. It includes the MX server, Redis, and one vLLM replica. This evaluation setup uses ephemeral storage and unauthenticated APIs; use a trusted cluster network. Compilation is disabled to keep the smoke test small. Source and target use the same model revision and runtime settings; `--revision` pins the checkpoint actually loaded. `MX_MODEL_REVISION` only labels MX source identity and does not download or pin that checkpoint.

```bash
kubectl create namespace mx-demo
kubectl -n mx-demo apply -f examples/p2p_transfer_k8s/quickstart.yaml
kubectl -n mx-demo rollout status deployment/modelexpress-server --timeout=5m
kubectl -n mx-demo rollout status deployment/mx-vllm --timeout=20m
SOURCE_POD=$(kubectl -n mx-demo get pods -l app=mx-vllm \
  -o jsonpath='{.items[0].metadata.name}')
kubectl -n mx-demo exec "$SOURCE_POD" -c vllm -- \
  curl -fsS http://localhost:8000/health
kubectl -n mx-demo logs "$SOURCE_POD" -c vllm
```

The first worker has no peer to copy, so a message such as `No RDMA source available` is expected. It loads through an eligible storage path and publishes its post-processed GPU tensors for later workers. Wait for `Source published successfully` in its logs, then keep it running. HTTP readiness can precede publication by a few seconds.

## Start a target and prove P2P worked

Scale only after the source is healthy and has published:

```bash
kubectl -n mx-demo scale deployment/mx-vllm --replicas=2
kubectl -n mx-demo rollout status deployment/mx-vllm --timeout=20m
TARGET_POD=$(kubectl -n mx-demo get pods -l app=mx-vllm \
  --field-selector "metadata.name!=$SOURCE_POD" \
  -o jsonpath='{.items[0].metadata.name}')
kubectl -n mx-demo logs "$TARGET_POD" -c vllm
```

In the **target's** logs, look for `Trying strategy: rdma` followed by `[TIMING] RDMA transfer complete: ... tensors, ... GB`. An eligible loader or a transfer attempt alone is not success. If the target falls back to `instant_tensor`, `model_streamer`, `gds`, or `default`, it may still serve correctly, but that run has not demonstrated P2P loading.

Send a request directly to the target so a Service cannot route it to the source:

```bash
kubectl -n mx-demo exec "$TARGET_POD" -c vllm -- \
  curl -fsS http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
```

The transfer completion marker and a valid completion response establish that this target received weights and can use them. For tensor-parallel deployments, check a transfer completion for every expected worker rank. This small model validates the path; use your production model and topology to measure startup savings.

If vLLM reports insufficient free GPU memory, lower `--gpu-memory-utilization` for this small model or use an available GPU. Check the underlying startup error before changing MX transport settings.

When finished with this example, release its GPUs and other resources:

```bash
kubectl delete namespace mx-demo
```

## Load directly from object storage

Use the [ModelStreamer examples](../../../examples/model_streamer_k8s/client/vllm/README.md) for S3, GCS, Azure Blob Storage, or local safetensors. Set `MX_MODEL_URI` and storage credentials on the **vLLM worker**. Weight bytes go from storage directly to that worker; this mode does not need the MX server or RDMA. A server address is needed only if you also want central P2P discovery.

Confirm `Model streamer weight loading complete` in the worker logs. A successful HTTP response alone can hide fallback to another loader. On current MX, object-storage URIs automatically skip InstantTensor; `MX_INSTANT_TENSOR=0` is not a blanket requirement for S3. Model configuration and tokenizer access remain separate from weight streaming.

For larger deployments, see the [single-node](../../../examples/p2p_transfer_k8s/client/vllm/vllm-single-node.yaml), [multi-node](../../../examples/p2p_transfer_k8s/client/vllm/vllm-multi-node.yaml), and [Dynamo](../orchestrators/dynamo.md) examples. Optional JIT cache reuse uses `MX_ARTIFACT_TRANSFER=1`; use the larger examples with compilation enabled after weight transfer works. For updating a running RL rollout worker, use the [RL integration](../../guides/rl.md).
