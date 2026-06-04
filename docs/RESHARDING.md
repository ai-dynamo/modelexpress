<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cross-Parallelism Resharding

ModelExpress should treat cross-parallelism resharding as a separate capability
from same-rank fan-out. The target is elastic, fault-tolerant refit across
mismatched trainer and inference parallelism without forcing either of these
fallbacks by default:

- Trainer full all-gather before inference-side apply.
- Trainer-side conversion into one specific inference backend layout.

The intended MX path is:

1. Source workers publish precise slice ownership for the tensor ranges they
   actually hold.
2. Receiver workers publish slice requests for the tensor ranges they need in
   their own runtime layout.
3. A planner intersects requested ranges with source-owned ranges and emits
   contiguous segment reads.
4. The target runtime issues one-sided NIXL reads that land each segment at the
   correct target buffer offset.

This keeps current recovery and fan-out work as the control-plane foundation,
but moves the data-plane contract from same-rank tensor equality to explicit
range coverage.

## Minimal Metadata

`SliceOwnership` is source-published metadata:

- Model name and version.
- Tensor name, global shape, dtype, and source logical range per axis.
- Storage byte offset, strides, and contiguity.
- Worker identity, lease/version, and NIXL descriptor identity.
- Layout tags for FSDP, TP, PP, EP, MoE expert axis, and storage-sensitive
  details such as axis order or packing.
- Quantization metadata scope: absent, local, global-required, or
  generated-on-target.

`SliceRequest` is receiver-side metadata:

- Tensor name, requested global range, target shape, and dtype.
- Destination byte offset and strides.
- Runtime/framework identity.
- Target layout tags for TP, PP, EP, MoE placement, and storage-sensitive
  compatibility.

`SegmentPlan` is planner output:

- Source id, worker id, tensor name, source range, and target range.
- Source and target byte offsets.
- Byte count.
- Lease/version and NIXL descriptor used.
- Retry policy for segment-level replan or fallback.

The Python POC implementation lives in `modelexpress.resharding`. It is pure
planner/simulator code and does not change the current RDMA loader behavior.
The executable GPU proof lives in `modelexpress.refit_poc` and can be mounted
into an existing GPU image for nscale validation without rebuilding the image.

The first control-plane integration step lives in
`modelexpress.resharding_control_plane` and the P2P metadata proto. It adds
slice ownership descriptors to `WorkerMetadata`, preserves them through central
server records and K8s-service manifest retrieval, and lets targets plan from
metadata returned by MX-client-shaped APIs. The live central-server lifecycle is
covered by the nscale Redis-backed smoke in
`artifacts/resharding/nscale-live-control-plane.log`. The NIXL POC now has a
`--control-plane live-mx` path that publishes source-rank ownerships through MX
and plans target reads from returned READY metadata. GPU verification of that
path still needs a schedulable nscale GPU slot; the failed scheduling attempt is
recorded in `artifacts/resharding/nscale-live-mx-nixl-capacity.log`.

Qwen-style MoE manifest classification lives in
`modelexpress.resharding_manifest`. It emits tensor family,
quantization-scope, expert-axis, and layout-sensitive tags for synthetic
Qwen-style manifests and can read local or Hugging Face safetensors headers
without loading tensor payloads. The current BF16 real-model coverage artifact
is `artifacts/resharding/qwen3-30b-a3b-moe-manifest.json.gz`, generated from
`Qwen/Qwen3-30B-A3B` safetensors headers across all 16 shards. The current FP8
coverage artifact is
`artifacts/resharding/qwen3-30b-a3b-fp8-moe-manifest.json.gz`, generated from
`Qwen/Qwen3-30B-A3B-FP8` headers across all 7 shards; it includes real
`weight_scale_inv` tensors classified as `global-required` fallback metadata.

Receiver-side runtime tensor helpers live in
`modelexpress.resharding_receiver`. They convert framework-owned torch tensors
into `SliceRequest`s and install planned segment payloads into target tensor
slices. The current smoke covers vLLM-shaped and SGLang-shaped requests on CPU;
live vLLM/SGLang process integration remains future work.

## Planner Gate

The planner gate is local and should pass before cluster work:

- Non-aligned trainer shards cover inference slices with no gaps.
- Duplicate coverage, missing coverage, dtype mismatch, storage-layout mismatch,
  and global quantization metadata requirements are rejected.
- Segment plans serialize to stable JSON artifacts.
- The simulator reports direct bipartite P2P vs primary/replica fan-out,
  redundant cross-boundary byte factor, segment count, source/target balance,
  and predicted bottleneck.

## Private POC Ladder

Keep broad publication gated until the patent-sensitive review is complete.

1. Pure planner correctness: multiple trainer shards satisfy one inference slice
   spanning source boundaries.
2. Small CUDA/NIXL assembly: preallocate one target buffer and issue parallel
   one-sided reads from multiple source holders into the correct offsets.
3. Segment-level recovery: fail one source mid-read, replan the failed segment
   from an alternate holder, and validate the final tensor.
4. MoE metadata: validate Qwen-style expert tensors and classify tensor families
   that need special handling.
5. Cross-framework smoke: one source publication satisfies compatible vLLM and
   SGLang receiver requests and installs planned payloads into runtime-owned
   target tensors.

nscale is the right place for the cluster POCs once the local planner gate is
green, especially the CUDA/NIXL assembly, kill/recovery, Qwen MoE, vLLM, and
SGLang smoke checks.

The current POC artifacts are stored under `artifacts/resharding/`. The primary
Level 2 proof artifact is `nscale-nixl-distributed-refit.json`, with the full
pod log in `nscale-nixl-gpu-refit.log` and the machine-readable completion
audit in `completion-audit.json`. It proves same-node cross-GPU NIXL READs into
planned target offsets; multi-pod cross-node refit and live MX metadata driving
the NIXL data plane in a completed GPU run are still future gates. The partial
Level 3 control-plane evidence is in `nscale-control-plane-pytest.log`,
`docker-rust-p2p-tests.log`, `nscale-live-control-plane.log`, and
`nscale-live-mx-nixl-capacity.log`. Qwen BF16/FP8 safetensors-header extraction
and receiver-side runtime tensor install smokes are covered by
`nscale-python-full-pytest.log`.

## Metrics

Cluster benchmarks should capture:

- Trainer-to-inference bytes.
- Inference-side fan-out bytes.
- Redundant cross-boundary byte factor.
- Segment count and source count per target tensor.
- Raw NIXL read duration.
- Registration, publish, planner, activation, and install durations.
- Retries and rediscovery count.
- Checksum or sampled correctness.
