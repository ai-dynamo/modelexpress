# Dynamo + vLLM S3 delta-refit lifecycle

This example updates two running vLLM generators through Dynamo using S3 checkpoints and ModelExpress. A deterministic CPU publisher stands in for training, so you can exercise the weight-update lifecycle without installing a training framework. For the direct GPU-transfer path or a real training loop, see the [RL guide](../../../docs/guides/rl.md).

Starting from the initial checkpoint, it installs one full snapshot (`v1`) and two XOR deltas (`d2` and `d3`). For each version, the first generator reconstructs the target from S3; the second must receive the installed weights from the first over NIXL. The test fails if both generators use S3.

The orchestrator advances the deployment's desired version only after every generator reports success. The script then deletes one generator pod and verifies that its replacement loads version `v3` from the surviving peer. It fails if replacement startup falls back to S3.

## What it checks

- `ModelExpressTrainerClient` publishes full HF checkpoints and XOR deltas to MinIO.
- The Redis-backed MX catalog tracks each `WeightVersion`.
- Dynamo discovers workers and forwards their RL control routes.
- vLLM installs updates through the MX weight-transfer backend and reports the exact version through its Control API.
- The first worker reads S3, the second uses peer transfer, and a replacement inherits the desired version.

The publisher mutates a fixed slice of the model's largest floating-point parameter. This is a synchronization test, not a training or model-quality benchmark.

## Requirements

- A Kubernetes cluster with the Dynamo v1beta1 operator and two GPUs with `rdma/ib` resources.
- The `shared-model-cache` PVC populated with `Qwen/Qwen3-0.6B`.
- `hf-token-secret`, `mx-minio-creds` (keys `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`), and `nvcr-imagepullsecret` in the target namespace.
- `envsubst`, `kubectl`, `docker buildx`, and images visible to the cluster.

The model PVC can be prepared with [`ci/rl/model-download.yaml`](../../../ci/rl/model-download.yaml); adapt its namespace and storage settings to your cluster. Use a namespace you own. Resource names are fixed, so run only one instance per namespace. MinIO and Redis in this example use ephemeral storage; worker-restart recovery assumes those services remain available.

## Build

Build one image containing ModelExpress, vLLM, and Dynamo's vLLM sidecar. The Dynamo build context must use the same revision as the frontend image:

```bash
export REGISTRY=registry.example.com/project
export MX_COMMIT=$(git rev-parse --short HEAD)
export DYNAMO_REF=c3e05f0244ae6264d7953f68e2499c6dc2f54723

git clone https://github.com/ai-dynamo/dynamo.git /tmp/dynamo
git -C /tmp/dynamo checkout "$DYNAMO_REF"

docker buildx build --platform linux/amd64 \
  -f examples/rl/dynamo_vllm_s3_delta_refit/Dockerfile \
  --build-context dynamo=/tmp/dynamo --push \
  -t "$REGISTRY/modelexpress-dynamo-vllm:$MX_COMMIT-s3-refit" .
```

The example pins Dynamo commit `c3e05f0244ae6264d7953f68e2499c6dc2f54723` with frontend image `nvcr.io/nvidia/ai-dynamo/dynamo-frontend-nightly:20260909-c3e05f0`. Use the vLLM base image pinned in the Dockerfile; this flow requires its Control gRPC service.

## Run

```bash
export NAMESPACE=your-namespace
export RUN_ID=mx-s3-$(date +%s)
export MODEL_NAME=Qwen/Qwen3-0.6B
export MODEL_SUBPATH=Qwen/Qwen3-0.6B
export WORKER_IMAGE="$REGISTRY/modelexpress-dynamo-vllm:$MX_COMMIT-s3-refit"
export DYNAMO_FRONTEND_IMAGE=nvcr.io/nvidia/ai-dynamo/dynamo-frontend-nightly:20260909-c3e05f0

examples/rl/dynamo_vllm_s3_delta_refit/run.sh
```

A successful run ends with both:

```text
E2E PASS: ...
RESTART PASS: ...
```

The script leaves the deployment running. Preserve the logs, then remove the example resources when finished.

## Coverage boundary

This smoke covers two deltas, S3-to-peer sharing, and one generator restart while its peer and metadata services remain healthy. It does not cover chain fast-forward, rollback, partial-update failure, missing S3 objects, Redis catalog loss, scale-up, HTTP 425 behavior, or checkpoint identity in inference responses.

The orchestrator resumes workers in its cleanup block even if an update fails. A production integration must keep a failed worker out of service until its installed version is recovered and verified.
