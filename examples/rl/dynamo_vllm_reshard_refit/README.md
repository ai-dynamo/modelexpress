# Dynamo + vLLM full-weight refit

Use this example for a first trainer-to-generator weight update. It runs one GPU publisher and one vLLM inference worker with `Qwen/Qwen3-0.6B`, using NIXL for transfer and Dynamo for worker discovery and control. See the [RL guide](../../../docs/guides/rl.md) for the broader workflow.

1. The vLLM worker starts from the model checkpoint and serves a baseline request.
2. The publisher loads the same checkpoint and publishes a full-weight `WeightVersion` through the FSDP adapter in `modelexpress_rl`. It keeps the GPU buffers available as the NIXL source.
3. The coordinator pauses generation and calls vLLM's `init_weight_transfer_engine`, `start_weight_update`, `update_weights`, and `finish_weight_update` routes through Dynamo.
4. It checks the installed version UID, resumes generation, and compares deterministic output before and after the update.

The publisher does not run an optimizer or modify the weights. The test checks full-weight transfer and generation parity; it does not test multi-rank resharding, S3 deltas, or replacement-worker recovery.

The coordinator resumes workers in its cleanup block even if an update fails. Use it as a smoke test; a production integration must keep a failed worker out of service until its installed version is recovered and verified.

## Requirements

- A Kubernetes cluster with the `nvidia.com/v1beta1` Dynamo operator and two available GPUs, each with an `rdma/ib` resource.
- A namespace you own and an `hf-token-secret` there; both GPU pods need access to download `Qwen/Qwen3-0.6B`.
- `docker`, `kubectl`, `envsubst`, and a registry the cluster can pull from.

Use the pinned images below together. The engine Dockerfile pins a vLLM image built from commit `a9a17e7095a66ef6c6685a1c7ddd657781a78d3c`; this sidecar flow requires its Control gRPC service. The Dynamo sidecar is built from a revision with weight-transfer route forwarding; the frontend is pinned in `dgd.yaml`.

## Build

From the ModelExpress repository root, build and push the server and engine images. Use immutable tags in a registry visible to the cluster.

```bash
export REGISTRY=registry.example.com/your-project
export MX_COMMIT=$(git rev-parse --short HEAD)

docker build -f ci/k8s/server/Dockerfile.server \
  -t "$REGISTRY/modelexpress-server:$MX_COMMIT-dynamo-refit" .
docker push "$REGISTRY/modelexpress-server:$MX_COMMIT-dynamo-refit"

docker build -f examples/rl/dynamo_vllm_reshard_refit/Dockerfile.vllm \
  -t "$REGISTRY/modelexpress-vllm:a9a17e7-$MX_COMMIT-dynamo-refit" .
docker push "$REGISTRY/modelexpress-vllm:a9a17e7-$MX_COMMIT-dynamo-refit"
```

Build Dynamo's sidecar from the pinned revision:

```bash
git clone https://github.com/ai-dynamo/dynamo.git /tmp/mx-dynamo
git -C /tmp/mx-dynamo checkout ff959852b740ee5981e58a5fcf18d0d4ca2d5079
docker build -f /tmp/mx-dynamo/lib/sidecar/vllm/Dockerfile \
  -t "$REGISTRY/dynamo-vllm-sidecar:ff95985" /tmp/mx-dynamo
docker push "$REGISTRY/dynamo-vllm-sidecar:ff95985"
```

## Deploy and test

Run these commands from the ModelExpress repository root. If the images are in a private registry, configure image-pull credentials on the namespace's ServiceAccount or add `imagePullSecrets` entries to the pod specs.

```bash
export NAMESPACE=your-namespace
export MODEL_NAME=Qwen/Qwen3-0.6B
export MX_SERVER_IMAGE="$REGISTRY/modelexpress-server:$MX_COMMIT-dynamo-refit"
export VLLM_ENGINE_IMAGE="$REGISTRY/modelexpress-vllm:a9a17e7-$MX_COMMIT-dynamo-refit"
export DYNAMO_SIDECAR_IMAGE="$REGISTRY/dynamo-vllm-sidecar:ff95985"

envsubst < examples/rl/dynamo_vllm_reshard_refit/server.yaml |
  kubectl apply -n "$NAMESPACE" -f -
envsubst < examples/rl/dynamo_vllm_reshard_refit/dgd.yaml |
  kubectl apply -n "$NAMESPACE" -f -

kubectl wait -n "$NAMESPACE" --for=condition=Ready \
  dgd/mx-vllm-refit --timeout=15m

kubectl create configmap mx-vllm-rl-coordinator -n "$NAMESPACE" \
  --from-file=rl_coordinator.py=examples/rl/dynamo_vllm_reshard_refit/rl_coordinator.py \
  --dry-run=client -o yaml | kubectl apply -n "$NAMESPACE" -f -
envsubst < examples/rl/dynamo_vllm_reshard_refit/rl-job.yaml |
  kubectl apply -n "$NAMESPACE" -f -
kubectl wait -n "$NAMESPACE" --for=condition=complete \
  job/mx-vllm-rl-job --timeout=15m
kubectl logs -n "$NAMESPACE" job/mx-vllm-rl-job
```

A successful run ends with `E2E PASS` and includes the installed version UID, worker count, and post-refit generation. Preserve the worker, server, and coordinator logs when evaluating the result. The script leaves the deployment running; remove the example resources when finished.

For full checkpoints, deltas, and restart recovery, continue with the [S3 lifecycle example](../dynamo_vllm_s3_delta_refit/README.md). For a training loop, use [Vime + Dynamo](../vime_dynamo_delta_refit/README.md).
