# Vime + Dynamo S3 delta refit

This example runs 100 Vime training steps, publishes an XOR delta after each step
to MinIO through ModelExpress, and installs them in one live Dynamo vLLM worker.

```mermaid
flowchart LR
    V[Vime trainer<br/>Megatron TP2 / 2 GPUs<br/>vLLM library only]
    V -->|XOR delta| S[MinIO]
    V -->|version lifecycle| M[ModelExpress + Redis]
    V -->|discover and control| D[Dynamo]
    D --> L[vLLM rollout<br/>TP1 / 1 GPU]
    S --> L
    M --> L
```

## Requirements

- A target namespace in the current `kubectl` context with a compatible
  `nvidia.com/v1beta1` Dynamo operator, three available NVIDIA GPUs, and about
  70 GiB of node-local storage headroom for the 100 deltas.
- A `shared-model-cache` PVC containing `Qwen/Qwen3-0.6B`.
- An `nvcr-imagepullsecret` image-pull Secret and an `mx-minio-creds` Secret
  containing `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`.
- A sibling `dynamo` worktree checked out at the latest `origin/main`. The
  trainer uses the pinned Vime revision declared in the Dockerfile. Its base
  already contains Megatron-LM and Megatron Bridge; the recipe does not install
  another copy.
- `docker buildx` and `kubectl` on the client machine.

If your PVC or Secret names differ, edit the YAML directly.

## Run

From the ModelExpress repository root, build and push the two images:

```bash
docker buildx build \
  -f examples/rl/vime_dynamo_delta_refit/Dockerfile \
  --target trainer --push \
  -t registry.example.com/team/vime-dynamo-mx-trainer:delta-refit .

docker buildx build \
  -f examples/rl/vime_dynamo_delta_refit/Dockerfile \
  --build-context dynamo=../dynamo --target worker --push \
  -t registry.example.com/team/vime-dynamo-mx-worker:delta-refit .
```

Set the target namespace, images, and model path relative to the
`shared-model-cache` PVC root, then run the example:

```bash
export NAMESPACE="<namespace>"
export WORKER_IMAGE="registry.example.com/team/vime-dynamo-mx-worker:delta-refit"
export TRAINER_IMAGE="registry.example.com/team/vime-dynamo-mx-trainer:delta-refit"
export MODEL_SUBPATH="<path/to/models/Qwen3-0.6B>"

examples/rl/vime_dynamo_delta_refit/run.sh
```

The script prints the trainer log and ends with `TRAINING COMPLETE`. It intentionally
does not delete anything; discard the namespace after inspecting the run.
