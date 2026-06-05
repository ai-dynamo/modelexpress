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
metadata returned by MX-client-shaped APIs. Refit endpoint publication also
dual-writes ownership into a prefixed legacy string sidecar so targets can
recover ownership from older live MX server images that preserve tensor/NIXL
fields but drop the newer `slice_ownerships` proto field. The sidecar is only a
compatibility bridge; the production schema remains `slice_ownerships`.
The live central-server lifecycle is
covered by the nscale Redis-backed smoke in
`artifacts/resharding/nscale-live-control-plane.log`. The NIXL POC now has a
`--control-plane live-mx` path that publishes source-rank ownerships through MX
and plans target reads from returned READY metadata. It also publishes source
NIXL endpoint metadata after registration so the target can get source agent
metadata and remote CUDA tensor descriptors from MX instead of torch distributed
object gather. The completed nscale same-node 4-B200 ownership proof is
`artifacts/resharding/nscale-live-mx-nixl-refit.json`, with the full torchrun
log in `artifacts/resharding/nscale-live-mx-nixl-refit.log` and live server log
in `artifacts/resharding/nscale-live-mx-nixl-server.log`. The endpoint proof is
`artifacts/resharding/nscale-live-mx-nixl-endpoint-refit.json`, with full
torchrun log in `artifacts/resharding/nscale-live-mx-nixl-endpoint-refit.log`
and compressed server log in
`artifacts/resharding/nscale-live-mx-nixl-endpoint-server.log.gz`. The earlier
capacity-blocked attempt remains recorded in
`artifacts/resharding/nscale-live-mx-nixl-capacity.log`.

The current cross-node proof is
`artifacts/resharding/nscale-crossnode-mx-nixl-refit.json`, with source
publication evidence in `artifacts/resharding/nscale-crossnode-mx-nixl-source.json`,
UCX/NIXL debug output in `artifacts/resharding/nscale-crossnode-mx-nixl-refit.log`,
and placement evidence in
`artifacts/resharding/nscale-crossnode-mx-nixl-capacity.log`. It runs two pods
on two GPU nodes: `mx-crossnode-refit-source` on
`cluster-0967a26d-pool-14bee067-prctr-g2j7h` and
`mx-crossnode-refit-target` on
`cluster-0967a26d-pool-14bee067-prctr-w4xnn`. The target discovers source
NIXL endpoint metadata from MX, constructs a target NIXL agent, adds the remote
source agent, forms UCX rc lanes over `mlx5_3:1` with
`UCX_TLS=rc_x,rc,tcp,cuda_copy`, excludes bonded NICs, performs planned
one-sided READs into the target buffer, and validates allclose/checksum.

The stricter one-pod-per-source-rank cross-node proof is
`artifacts/resharding/nscale-crossnode-one-pod-per-source-rank-target.json`,
with source publication artifacts in
`nscale-crossnode-one-pod-per-source-rank-source-rank0.json`,
`nscale-crossnode-one-pod-per-source-rank-source-rank1.json`, and
`nscale-crossnode-one-pod-per-source-rank-source-rank2-alt.json`. It runs three
independent source-rank pods on
`cluster-0967a26d-pool-14bee067-prctr-9c2x7` and one target pod on
`cluster-0967a26d-pool-14bee067-prctr-g2j7h`. Each source pod publishes exactly
one ownership and one distinct NIXL source agent; the target discovers all
three endpoints from MX, performs cross-node UCX/IB rc NIXL reads from the
needed source ranks, replans the failed primary segment to the alternate holder,
and validates allclose/checksum. This proves independent source-pod fan-in for
the synthetic MX/NIXL refit path. It does not yet prove real trainer pod churn
or real runtime-owned vLLM/SGLang refit.

The current runtime-owned vLLM trainer-loop cross-node proof is
`artifacts/resharding/nscale-live-vllm-mx-runtime-trainer-loop-crossnode-20260605.json`.
Two independent source pods on
`cluster-0967a26d-pool-14bee067-prctr-g2j7h` publish versioned trainer-loop
step-2 source ownership/NIXL endpoint metadata through MX; a target pod on
`cluster-0967a26d-pool-14bee067-prctr-9c2x7` loads live vLLM 0.17.1, discovers
both endpoints from MX, performs two cross-node UCX/NIXL reads into CUDA
staging, infers expected optimizer step 2 from source metadata, installs via
`LLM.apply_model`, and validates allclose/checksum. This proves tiny
single-tensor vLLM runtime-owned staging-copy refit with trainer-loop metadata;
direct NIXL landing into vLLM-owned storage, SGLang trainer-loop cross-node
rerun, full-model refit, and real RL trainer integration remain open.

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
The artifact `qwen3-30b-a3b-fp8-zero-copy-fallback-smoke.json` verifies that a
real `global-required` `weight_scale_inv` tensor is rejected by the zero-copy
planner with `QuantizationMetadataError`; the runtime fallback install path is
still future work.

Receiver-side runtime tensor helpers live in
`modelexpress.resharding_receiver`. They convert framework-owned torch tensors
into `SliceRequest`s and install planned segment payloads into target tensor
slices. The current smoke covers vLLM-shaped and SGLang-shaped requests on CPU;
live vLLM/SGLang process integration remains future work.

The competitive refit simulator also lives in `modelexpress.resharding`. It
compares MX direct bipartite P2P, MX primary/replica fanout, NCCL Reshard-style
fixed-membership full-tensor movement, and CheckpointEngine-style full
gather/apply. The committed artifact
`artifacts/resharding/competitive-refit-simulation.json` records a two-step RL
rollout case where four receiver requests span two trainer holders; the
simulator reports cross-boundary bytes, inference-side fanout bytes, trainer
collective bytes, checkpoint storage bytes, redundant byte factors, segment
count, source/target balance, source count per target tensor, and predicted
bottlenecks. This is a byte/cost simulator, not a real performance benchmark.

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
planned target offsets. The primary Level 3 same-node proof is
`nscale-live-mx-nixl-refit.json`: live MX-returned slice ownership metadata
drives the NIXL data plane in a completed GPU run. The stronger endpoint proof
is `nscale-live-mx-nixl-endpoint-refit.json`: source NIXL agent metadata and
remote CUDA tensor descriptors are discovered through MX worker metadata, and
the artifact records `torch_distributed_nixl_metadata_exchange_used=false`.
The current cross-node proof set includes `nscale-crossnode-mx-nixl-refit.json`
for the two-pod source/target case,
`nscale-crossnode-one-pod-per-source-rank-target.json` for independent
source-rank pods feeding one target across nodes, and
`nscale-live-vllm-mx-runtime-trainer-loop-crossnode-20260605.json` for tiny
live-vLLM runtime-owned staging-copy refit with trainer-loop source metadata.
Real source pod churn, direct runtime-owned zero-copy writes, full-model runtime
refit, and real RL trainer integration are still future gates. The first nscale
attempt to schedule independent source-rank pods is recorded as a superseded
capacity block in
`nscale-one-pod-per-source-capacity-block.log` and
`nscale-one-pod-per-source-capacity-block.json`; the later one-pod-per-source
artifact is the checksum-backed claim.
The Level 3 control-plane evidence is in `nscale-control-plane-pytest.log`,
`nscale-refit-endpoint-control-plane-pytest.log`, `docker-rust-p2p-tests.log`,
`nscale-live-control-plane.log`, `nscale-live-mx-nixl-refit.log`,
`nscale-live-mx-nixl-server.log`, and
`nscale-live-mx-nixl-endpoint-server.log.gz`, plus the cross-node source,
target, and placement logs. Qwen BF16/FP8
safetensors-header extraction, real FP8 zero-copy fallback, and receiver-side
runtime tensor install smokes are covered by `nscale-python-full-pytest.log`.
The partial Level 5 simulator evidence is `competitive-refit-simulation.json`.
The real Level 5 timing table against MX/NIXL, NCCL Reshard, and
CheckpointEngine remains unmeasured; the current capacity block is recorded in
`nscale-level5-timing-capacity-block.json`.

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
