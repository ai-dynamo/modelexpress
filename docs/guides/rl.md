<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Update rollout weights for RL

ModelExpress moves a new model version from training into running inference workers, also called **generators**. This weight update is called **refit**. In an RL loop, generators produce samples with the updated weights; rollout clients submit generation requests to them.

Your training framework still owns optimization, rewards, rollout scheduling, and when workers pause or resume. MX handles weight publication, version discovery, transfer, and engine installation. If your weights stay fixed and you only want faster inference startup, start with the [inference guide](choose-a-path.md).

## Start with a runnable example

For a first refit, use the [Dynamo + vLLM full-weight example](../../examples/rl/dynamo_vllm_reshard_refit/README.md). It publishes `Qwen/Qwen3-0.6B` from one GPU, transfers it to one running vLLM worker, checks the installed version, and compares deterministic generation before and after. It exercises synchronization without running an optimizer.

All three examples run on Kubernetes, include a Redis-backed MX server, and require the Dynamo operator. Use each example's pinned runtime images and namespace prerequisites.

| What you want to try | Example | Additional resources |
|---|---|---|
| Direct trainer-to-generator transfer | [Full-weight NIXL refit](../../examples/rl/dynamo_vllm_reshard_refit/README.md) | 2 GPUs with RDMA; model download access |
| Full checkpoints, deltas, peer sharing, and worker restart | [S3 delta-refit lifecycle](../../examples/rl/dynamo_vllm_s3_delta_refit/README.md) | 2 GPUs with RDMA; model PVC; MinIO credentials |
| An actual training loop that publishes updates | [Vime + Dynamo delta refit](../../examples/rl/vime_dynamo_delta_refit/README.md) | 3 SM90+ GPUs; model PVC; MinIO credentials and local checkpoint space |

The Vime example uses S3 transfers and does not request RDMA. Its random reward is a synchronization exercise, not a model-quality benchmark.

## What happens during an update

1. The framework creates a weight version and publishes its trainer shards or checkpoint artifacts. It keeps the source stable until the required readers finish.
2. The framework pauses generation and handles in-flight work through the engine's control API. MX transfers the selected version and the engine installs it while generation is paused.
3. The framework checks that every required generator reports the requested version, then resumes generation. An update failure needs recovery before that worker serves more requests.
4. Once readers finish, the framework retires the published NIXL version, releases its source buffers, and continues training. S3 checkpoints and their version records remain available for recovery according to the integration's retention policy.

Direct publication currently uses a synchronous lifetime: training must not overwrite published buffers while generators are reading them. Staging choices belong to the framework integration; copying a snapshot can separate it from live trainer storage, but does not create an asynchronous training protocol by itself.

## Choose how weights move

| Path | How it works | Main requirement |
|---|---|---|
| Direct NIXL | Generators pull registered trainer shards and reconstruct their inference layout. Compatible generators can also share installed runtime tensors. | GPU connectivity, matching adapters, and stable source buffers |
| S3 full checkpoints and XOR deltas | Trainers publish immutable artifacts; generators reconstruct a local checkpoint and reload the engine. A delta needs its exact base version. | S3 access, a seed checkpoint, and local disk for reconstruction |

The MX server carries metadata, not weight bytes. RL's `RefitService` currently requires `MX_METADATA_BACKEND=redis`; deploying it on Kubernetes does not make the Kubernetes metadata backend a replacement for Redis. Ordinary inference loading has [other backend options](../CONFIGURATION.md).

Direct pulls avoid a trainer-side full-model gather for supported shard layouts, at the cost of retaining readable source buffers. S3 decouples transfer from trainer GPU lifetime, at the cost of artifact upload, download, and checkpoint reconstruction. Neither path removes engine installation time.

## Integrate with your framework

The Python package exposes `ModelExpressControlClient`, `ModelExpressTrainerClient`, and `ModelExpressGeneratorClient` through `modelexpress_rl`. [Trainer publication](../../modelexpress_client/python/README.md#rl-trainer-publication) explains the adapter contract; the [S3 refit guide](../S3_DELTA_WEIGHT_REFIT.md) covers checkpoint publication and replay.

Megatron and FSDP/DTensor trainer adapters are present. vLLM has the weight-transfer backend used by the examples. SGLang has a checkpoint installer; it does not currently expose the same direct tensor-refit path. These interfaces still need framework lifecycle hooks and validation for your model, dtype, and parallel layout. The [refit internals](../../modelexpress_client/python/modelexpress/refit/README.md) describe geometry capture and its limits.
