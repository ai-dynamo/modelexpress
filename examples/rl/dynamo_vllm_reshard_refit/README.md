# Dynamo + vLLM reshard refit

This example validates the complete inference-side lifecycle designed for
ModelExpress `WeightVersion` updates:

1. A GPU trainer job loads `Qwen/Qwen3-0.6B`, publishes one immutable full-weight
   version through `modelexpress_rl`, and waits as the NIXL source.
2. On cold start, all vLLM ranks agree on the desired UID and load it through
   desired-version P2P or canonical S3 replay. The vLLM init container's startup
   probe then writes and verifies that UID through vLLM's native Control gRPC
   service before Dynamo starts its sidecar.
3. Dynamo discovers the admitted vLLM worker and pauses generation.
4. vLLM invokes the `modelexpress` weight-transfer backend for initialization,
   `start_weight_update`, `update_weights`, and `finish_weight_update`.
5. The RL coordinator verifies the exact UID on every worker, resumes generation,
   and compares deterministic inference before and after the refit.

The DGD uses the current `nvidia.com/v1beta1` schema and Dynamo's native Rust
vLLM sidecar. Dynamo main is pinned because the weight-transfer route forwarding
is newer than the latest v1.4.1 release. The engine image pins the current vLLM
nightly digest built from commit `a9a17e7095a66ef6c6685a1c7ddd657781a78d3c`.
The latest vLLM v0.27.1 release predates the merged RL Control gRPC service, so
it cannot serve this Dynamo sidecar flow.

Global rank 0 broadcasts its `MX_REFIT_DESIRED_VERSION_UID`, and every vLLM rank
checks its local value against that pinned UID before loading. Local rank 0 on
each node reconstructs the canonical S3 checkpoint; followers reuse the
node-local result. A rank disagreement or partial distributed load fails
startup, and the checkpoint activation marker is committed only after every
rank succeeds.

## Build

From the ModelExpress repository root, build and push the server and engine
images. Use immutable tags in a registry visible to the cluster.

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

Build Dynamo's experimental sidecar from the pinned live-main revision:

```bash
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo
git checkout ff959852b740ee5981e58a5fcf18d0d4ca2d5079
docker build -f lib/sidecar/vllm/Dockerfile \
  -t "$REGISTRY/dynamo-vllm-sidecar:ff95985" .
docker push "$REGISTRY/dynamo-vllm-sidecar:ff95985"
```

## Deploy and test

Use a user-owned namespace. The commands require the Dynamo operator and an
`hf-token-secret` in that namespace. If the images are in a private registry,
configure image-pull credentials on the namespace's ServiceAccount or add your
own `imagePullSecrets` entries to the pod specs.

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

A successful run ends with `E2E PASS` and includes the installed WeightVersion
UID, worker count, and post-refit generation. Keep worker, server, and coordinator
logs as separate evidence; the pass line does not by itself qualify throughput
or delta/S3 behavior. The checked-in coordinator exercises the full-weight NIXL
active-refit lifecycle; it does not qualify canonical S3 delta cold start.
