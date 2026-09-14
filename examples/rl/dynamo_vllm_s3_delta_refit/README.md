# Dynamo + vLLM S3 delta-refit lifecycle

This example exercises the CoreWeave-shaped weight-synchronization lifecycle
without a training framework. A deterministic CPU job represents the external
RL orchestrator while the real Dynamo, vLLM, ModelExpress, and S3 paths remain
in the loop.

The test starts the DGD at its configured initial model `W1`, publishes a new
full snapshot `v1` and two incremental snapshots `d2` and `d3`, and performs:

```text
W1 initial load -> v1 full hot-load -> d2 hot-load -> d3 hot-load
                                      |
                                      +-> desired UID advances after success
```

Two generators are updated sequentially. For each version, the first generator
has no peer serving the target and reconstructs it from S3. The second generator
must load the target from the first over post-PWAL P2P. The test checks both
source selections instead of accepting an all-S3 run.

It then deletes one generator pod. The replacement reads the DGD-scoped desired
UID and must come back serving `v3`, not `W1`; the surviving generator provides
the desired version over P2P. The run fails if replacement startup falls back
from P2P to S3.

## What is real

- `ModelExpressTrainerClient` creates the full HF checkpoint and XOR deltas.
- MinIO stores the immutable checkpoint objects.
- ModelExpress stores the `WeightVersion` catalog in Redis.
- Dynamo discovers the worker and forwards its native RL control routes.
- vLLM applies updates through the ModelExpress weight-transfer backend.
- vLLM's Control gRPC value is checked after every update and after restart.
- The orchestrator updates desired intent only after every worker reports the
  requested version.

The synthetic publisher mutates a fixed slice of the model's largest floating
point parameter. It is not a training or numerical-quality test.

## Requirements

- A Kubernetes cluster with the Dynamo v1beta1 operator and two GPUs with RDMA.
- The `shared-model-cache` PVC populated with `Qwen/Qwen3-0.6B`.
- `hf-token-secret`, `mx-minio-creds`, and `nvcr-imagepullsecret`.
- `envsubst`, `kubectl`, and images visible to the cluster.

The model PVC can be prepared with `ci/rl/model-download.yaml`. Use a unique
namespace or the user-owned `zheng` namespace; do not deploy this example to
`default`. Resource names are fixed, so run only one instance per namespace.

## Build

Build one image containing ModelExpress, vLLM, and Dynamo's vLLM sidecar. The
Dynamo build context must use the same revision as the frontend image:

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

The PR #736 environment used Dynamo commit
`c3e05f0244ae6264d7953f68e2499c6dc2f54723` and frontend image
`nvcr.io/nvidia/ai-dynamo/dynamo-frontend-nightly:20260909-c3e05f0`.

## Run

```bash
export NAMESPACE=zheng
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
RESTART ...
```

## Current coverage boundary

This first smoke covers the CoreWeave happy path through two deltas, S3-to-P2P
fanout, and a single-generator restart while another generator remains healthy.
It does not yet cover chain fast-forward, rollback, partial update failure,
missing S3 objects, Redis catalog loss, scale-up, HTTP 425 behavior, or
checkpoint identity in inference responses.
