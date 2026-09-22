<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MILES ModelExpress M2N proof

This example runs real MILES GRPO training with disaggregated SGLang and the
external ModelExpress collective protocol. One rollout performs the initial
weight load, one optimizer step, and the post-step update. The default topology
uses a two-rank PP2 trainer and two independent TP1 generator engines on one
four-GPU H100 node. This exercises two real M2N source lanes while fitting the
capacity currently available on `aws-dev-02`. The same template supports six
generators when an entire eight-GPU node is free.

The Kubernetes proof explicitly sets `MX_MILES_VERIFY_TENSOR_EQUALITY=1`.
This enables a bounded, exact SHA-256 pass over every trainer wire tensor and
every SGLang receive tensor before an update may report success. The setting is
forwarded into both trainer and SGLang subprocesses. It defaults off outside
this qualification recipe because exact verification copies every tensor byte
to host memory for hashing.

M2N device communicators require NCCL symmetric-memory support. The image,
pod, and Ray runtime therefore pin `NCCL_CUMEM_ENABLE=1`; the runtime verifier
fails before scheduling the GPU workload if that setting is absent.

The image is pinned to the CUDA 13 MILES base, overlays the local MILES checkout
at `/root/miles` and a local Megatron-Bridge checkout at
`/opt/megatron-bridge`, installs the current ModelExpress Python client and
`nccl-extensions[cu13]`, and fails its build unless the loaded NCCL is at least
2.30.7 and the Bridge export API exposes `yield_pp_local`. The Bridge source
directory is first on `PYTHONPATH`, so the image does not silently use a base
image copy. The ModelExpress server image is digest-pinned in the manifest.

## Build and push

Run from the ModelExpress worktree root. Docker named contexts deliberately
select the sibling MILES worktree containing the external protocol support.

```bash
export RUNTIME_IMAGE=nvcr.io/nvidian/dynamo-dev/modelexpress-miles-m2n:YOUR_TAG
export MILES_WORKTREE=../miles-mx-nccl-m2n
export MEGATRON_BRIDGE_WORKTREE=../megatron-bridge-miles-pp-local

docker buildx build --push \
  --build-context miles="$MILES_WORKTREE" \
  --build-context modelexpress=./modelexpress_client/python \
  --build-context megatron_bridge="$MEGATRON_BRIDGE_WORKTREE" \
  -f examples/rl/miles_m2n_collective/Dockerfile \
  -t "$RUNTIME_IMAGE" .
```

Use a clean checkout of the Megatron-Bridge revision containing the
`yield_pp_local` API for `MEGATRON_BRIDGE_WORKTREE`. A detached checkout or
directory produced from `git archive` is supported; no repository metadata or
absolute local paths are required inside the image.

## Deploy

`deploy.sh` uses the current Kubernetes context and the
`modelexpress-miles-m2n` namespace by default. Set `KUBE_CONTEXT`,
`NAMESPACE`, `GPU_PRODUCT`, and `RUNTIME_PULL_SECRET` for the target cluster.
The Qwen model and GSM8K dataset are downloaded into a pod-local `emptyDir`;
the public defaults do not require a Hugging Face token.

For the qualified `aws-dev-02` path:

```bash
export KUBE_CONTEXT=nv-prd-dgxc.teleport.sh-dynamo-aws-dev-02
export NAMESPACE="${USER}-mx-miles-m2n"
export RUNTIME_IMAGE=nvcr.io/nvidian/dynamo-dev/modelexpress-miles-m2n:YOUR_TAG
export MX_SERVER_IMAGE=nvcr.io/nvidian/dynamo-dev/modelexpress-server@sha256:YOUR_DIGEST
export RUN_ID="$(date -u +%Y%m%d-%H%M%S)"

examples/rl/miles_m2n_collective/deploy.sh preflight
examples/rl/miles_m2n_collective/deploy.sh apply
examples/rl/miles_m2n_collective/deploy.sh logs
examples/rl/miles_m2n_collective/deploy.sh wait
```

For the two-trainer/six-generator run, wait for one node to have eight free
GPUs and override:

```bash
export ACTOR_GPUS=2
export PIPELINE_PARALLEL_SIZE=2
export ROLLOUT_GPUS=6
export ROLLOUT_GPUS_PER_ENGINE=1
export TOTAL_GPUS=8
```

The job is deliberately not assigned a TTL. Preserve the pod logs as the proof
receipt, then delete only that run with
`examples/rl/miles_m2n_collective/deploy.sh delete`.
