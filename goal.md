# Goal: Make MX Multi-Source Refit Real

## Why This Goal Exists

The current nscale POC is useful, but it is not the full idea from the weight
refit discussion. It proves one narrow thing: planner-driven GPU segment
assembly and segment-level recovery can work. It does **not** yet prove
production MX/NIXL refit, real trainer integration, real vLLM/SGLang receiver
integration, or the full TorchStore/NCCL-Reshard-style algorithmic scope.

This goal file should prevent us from rewarding a hacked demo as if it finished
the strategic work.

## Strategic Product Goal

ModelExpress should become the K8s-native control plane for versioned
distributed tensor refit across mismatched trainer and inference layouts:

- Trainers publish exactly what tensor ranges they own.
- Inference runtimes request the tensor ranges their own layout needs.
- MX computes intersections, validates compatibility, chooses transfer/fanout
  strategy, and drives one-sided GPU reads into target buffers.
- The path avoids mandatory trainer all-gather and avoids trainer-side
  conversion into one inference backend's layout.
- The system is elastic, fault-tolerant, versioned, rollback-aware, and usable
  across vLLM, SGLang, TRT-LLM, and RL frameworks.

## Ideas We Must Explicitly Cover

These are the idea buckets from the weight-refit/PDF discussion and related
Tanushriya-style feedback. A future claim is not complete unless these are
addressed.

1. TorchStore-style overlap planning
   - Model every source shard as a `TensorSlice`.
   - Model every receiver need as a requested global slice.
   - Compute source/request intersections at runtime.
   - Fetch only overlap regions, with no full all-gather.

2. NCCL Reshard comparison
   - Treat NCCL Reshard as the strongest same-collective benchmark, not as an
     irrelevant alternative.
   - Compare direct P2P, hierarchical replication, and trainer-to-generator
     redistribution.
   - Track when NCCL wins: fixed membership, homogeneous GPUs, simple dtensor
     layouts, high fanout.
   - Track when MX/NIXL should win: elasticity, pod churn, mixed frameworks,
     versioning/rollback, K8s lifecycle, fault isolation.

3. NIXL/MX control-plane value
   - MX must broker metadata, liveness, leases, source version, target version,
     rediscovery, and rollback.
   - NIXL should provide the one-sided read data plane.
   - The POC must eventually issue actual NIXL reads, not only NCCL send/recv
     or local CUDA copies.

4. Two-step RL pattern
   - Step 1: trainer-to-first-rollout reshard.
   - Step 2: rollout replica fanout/broadcast.
   - MX must decide when direct bipartite P2P is better than primary/replica
     fanout.

5. Real runtime integration
   - Source path: FSDP/TP/PP/EP trainer publishes tensor ownership metadata.
   - Target path: vLLM and SGLang request and install their slices without
     forcing trainer-side backend conversion.
   - TRT-LLM should remain a future compatibility axis.

6. MoE and quantization correctness
   - Expert-axis tensors are first-class, not just row/column dense shards.
   - Hidden/layout-sensitive tensors must be classified.
   - Quantization metadata scope must be explicit: local, absent,
     generated-on-target, or global-required fallback.

7. Elasticity and recovery
   - Failed source segments should replan from alternate holders.
   - Leases/version checks should prevent stale reads.
   - Recovery should be segment-level, not whole-model restart.

## What Is Proven Now

Artifacts under `artifacts/resharding/` prove:

- CPU planner tests pass on nscale: `nscale-cpu-pytest.log`.
- A single-GPU assembly/recovery smoke passes:
  `nscale-single-gpu-refit.json`.
- A 4-rank B200 NCCL P2P/CUMEM distributed POC passes:
  `nscale-distributed-refit.json`.
- A 4-rank B200 NIXL/UCX same-node distributed POC passes:
  `nscale-nixl-distributed-refit.json`.
- The target slice spans two primary trainer holders.
- One failed source segment is replanned from an alternate holder.
- The NIXL POC registers source/target GPU buffers, exchanges metadata through
  a control-plane object gather, uses `add_remote_agent`, and performs one-sided
  READs into target offsets.
- The target buffer validates with allclose and checksum.
- The POC does not use trainer full all-gather, trainer-side inference-layout
  conversion, or host-side `torch.cat`.
- The NIXL POC does not use torch distributed tensor send/recv as the data
  plane.
- MX P2P metadata schema now has `SliceOwnershipDescriptor`,
  `SliceRequestDescriptor`, and `SegmentPlanDescriptor`.
- Server-side P2P metadata records can round-trip slice ownerships through
  proto conversion, Redis JSON, and `GetMetadata`.
- Python control-plane helpers can publish slice ownerships through an
  MX-client-shaped API, query READY sources, and plan receiver requests from
  returned metadata.
- K8s-service-routed manifest retrieval preserves slice ownership descriptors.
- A live Redis-backed central MX server on nscale accepts slice ownership
  `PublishMetadata`, serves `GetMetadata`, accepts `UpdateStatus`, returns READY
  workers from `ListSources`, and supports target-side planning from returned
  metadata.
- `modelexpress.refit_poc --mode nixl-distributed --control-plane live-mx`
  now publishes source-rank ownerships through MX and has the target rank plan
  NIXL reads from MX-returned metadata.
- A Qwen-style MoE manifest extractor now classifies expert-axis tensors,
  layout-sensitive shared-expert tensors, global quantization metadata, and
  generated-on-target tensors from tensor names/shapes.
- The extractor can read local safetensors headers and Hugging Face
  safetensors headers through HTTP range requests, preserving source shard
  filenames without loading tensor payloads.
- A real Qwen3 MoE coverage artifact was generated from
  `Qwen/Qwen3-30B-A3B` safetensors headers:
  `artifacts/resharding/qwen3-30b-a3b-moe-manifest.json.gz` plus
  `artifacts/resharding/qwen3-30b-a3b-moe-manifest.summary.json`. It covers
  18,867 tensors across 16 shards, including 18,432 MoE expert tensors and 435
  layout-sensitive tensors.
- A real Qwen3 FP8 MoE coverage artifact was generated from
  `Qwen/Qwen3-30B-A3B-FP8` safetensors headers:
  `artifacts/resharding/qwen3-30b-a3b-fp8-moe-manifest.json.gz` plus
  `artifacts/resharding/qwen3-30b-a3b-fp8-moe-manifest.summary.json`. It covers
  37,491 tensors across 7 shards, including 18,624 real
  `global-required` quantization metadata tensors classified as fallback
  required.
- A real Qwen3 FP8 fallback smoke proves that a `global-required`
  `weight_scale_inv` entry from the manifest is rejected by the zero-copy
  segment planner with `QuantizationMetadataError`, rather than silently
  producing a NIXL segment plan:
  `artifacts/resharding/qwen3-30b-a3b-fp8-zero-copy-fallback-smoke.json`.
- Qwen-style MoE metadata and compatible vLLM/SGLang request metadata are
  represented in planner smoke tests.
- Receiver-side helpers now build `SliceRequest`s from runtime-owned torch
  tensors and install planned segment payloads into those tensors. nscale CPU
  tests cover compatible vLLM-shaped and SGLang-shaped runtime requests whose
  target slice spans two trainer holders.

This is a **Level 2 same-node synthetic proof**, not a production refit
implementation. Level 3 control-plane metadata lifecycle support is now proven
through a live central MX server smoke, and the live-MX NIXL planning path is
implemented. The live-MX NIXL GPU run is not yet proven because nscale had no
schedulable GPU capacity during the attempt.

## What Is Not Proven Yet

The current POC does **not** prove:

- Completed GPU run where live MX-returned slice ownership metadata drives the
  NIXL refit POC.
- Real trainer process integration with FSDP, TP, PP, EP, or RL training loop.
- Live vLLM or SGLang process integration where the real engine owns the
  post-load/refit lifecycle.
- Multi-pod cross-node refit.
- Full-model or multi-layer refit.
- Runtime installation of real Qwen MoE model tensors.
- Runtime fallback installation of real quantized Qwen tensors after
  `global-required` metadata is detected.
- Hierarchical fanout to many rollout replicas.
- Versioned rollback using multiple GPU-resident training steps.
- Performance competitiveness against NCCL Reshard, CheckpointEngine, Mooncake,
  or TorchStore-style systems.

## Proof Levels

### Level 0: Planner Correctness

Status: implemented.

Evidence:

- `modelexpress.resharding`
- `modelexpress_client/python/tests/test_resharding.py`
- `artifacts/resharding/nscale-cpu-pytest.log`

Required capabilities:

- `SliceOwnership`
- `SliceRequest`
- `SegmentPlan`
- range intersection
- missing/duplicate/dtype/layout/quantization rejection
- JSON artifacts
- direct-vs-fanout simulator

### Level 1: Synthetic GPU Assembly

Status: implemented.

Evidence:

- `modelexpress.refit_poc`
- `artifacts/resharding/nscale-single-gpu-refit.json`
- `artifacts/resharding/nscale-distributed-refit.json`
- `artifacts/resharding/completion-audit.json`

Limit:

- Uses synthetic tensors and NCCL send/recv or CUDA copies, not actual NIXL
  one-sided reads.

### Level 2: Real NIXL Segment Reads

Status: implemented for same-node cross-GPU synthetic refit; cross-node remains
pending.

Goal:

- Two or more source ranks register disjoint GPU tensor ranges with NIXL.
- One target rank preallocates a target buffer.
- The target uses `SegmentPlan` to issue one-sided NIXL reads from multiple
  source holders into exact offsets.
- Source failure replans only failed segments from alternate holder.

Acceptance evidence:

- nscale same-node cross-GPU artifact:
  `artifacts/resharding/nscale-nixl-distributed-refit.json`.
- Captures NIXL registration duration, metadata blob size/identity,
  `add_remote_agent`, prep duration, raw read duration, and checksum.
- Cross-node/multi-pod remains the next Level 2 extension.

### Level 3: MX Control Plane Integration

Status: partially implemented; live metadata lifecycle proven; live-MX NIXL
planning path implemented but not GPU-verified.

Goal:

- Extend MX metadata publication beyond rank-matched full tensors to
  source-owned slices.
- Store leases, versions, tensor family, quantization scope, and layout tags.
- Targets query MX, create `SliceRequest`s, receive `SegmentPlan`s, and execute
  planned reads.

Acceptance evidence:

- MX server/client tests and nscale run using actual publish/list/get/status
  lifecycle rather than in-process synthetic metadata.

Current evidence:

- Proto/schema support in `modelexpress_common/proto/p2p.proto`.
- Server record/backend/service support in `modelexpress_server/src/p2p`.
- Python conversion and query/plan helpers in
  `modelexpress.resharding_control_plane`.
- nscale Python tests: `artifacts/resharding/nscale-control-plane-pytest.log`
  (`54 passed, 1 skipped`; the skipped test is the opt-in live-server smoke).
- Dockerized Rust server p2p tests:
  `artifacts/resharding/docker-rust-p2p-tests.log` (`79 passed`).
- nscale live central MX server smoke:
  `artifacts/resharding/nscale-live-control-plane.log` (`1 passed`).
- `modelexpress.refit_poc` live-MX NIXL path:
  `--mode nixl-distributed --control-plane live-mx`.
- GPU verification attempted on nscale but capacity blocked:
  `artifacts/resharding/nscale-live-mx-nixl-capacity.log`.
- Qwen MoE manifest extractor in `modelexpress.resharding_manifest`.
- Real Qwen3-30B-A3B safetensors-header coverage artifact:
  `artifacts/resharding/qwen3-30b-a3b-moe-manifest.json.gz`.
- Real Qwen3-30B-A3B-FP8 safetensors-header coverage artifact:
  `artifacts/resharding/qwen3-30b-a3b-fp8-moe-manifest.json.gz`.
- Real Qwen3-30B-A3B-FP8 zero-copy fallback smoke:
  `artifacts/resharding/qwen3-30b-a3b-fp8-zero-copy-fallback-smoke.json`.

Remaining gap:

- Rerun the live-MX NIXL path on nscale when GPU capacity is available and emit
  the completed JSON artifact. Prefer the normal 4-rank/4-GPU run; an explicit
  `MX_REFIT_ALLOW_GPU_REUSE=1` fallback exists only for capacity-constrained
  smoke testing.

### Level 4: Real Runtime Refit

Status: not implemented; CPU receiver-install smoke implemented.

Goal:

- Trainer-like process publishes FSDP/TP/EP/PP ownership.
- vLLM receiver and SGLang receiver each request compatible target slices.
- Receiver installs weights in runtime-owned target tensors.
- No trainer-side inference-layout conversion.

Acceptance evidence:

- vLLM smoke artifact.
- SGLang smoke artifact.
- tensor family coverage report for Qwen MoE.

Current partial evidence:

- `modelexpress.resharding_receiver` builds receiver requests from runtime-owned
  torch tensors and installs planned segment payloads into target tensor slices.
- `modelexpress_client/python/tests/test_resharding_receiver.py` covers
  vLLM-shaped and SGLang-shaped runtime tensor install smokes on nscale.
- Full nscale Python gate:
  `artifacts/resharding/nscale-python-full-pytest.log`
  (`272 passed, 19 skipped`).

### Level 5: Competitive Benchmark

Status: not implemented.

Goal:

- Compare MX/NIXL reshard to NCCL Reshard, CheckpointEngine/Mooncake style,
  direct all-gather, and primary/replica fanout.

Acceptance metrics:

- trainer-to-inference bytes
- fanout bytes
- redundant byte factor
- segment count
- source balance
- target balance
- registration/publish/planner/read/install timings
- retry/rediscovery count
- correctness checksum

## Near-Term Achievable Work

These are the next useful things to do, in order:

1. Rerun `refit_poc.py --mode nixl-distributed --control-plane live-mx` on
   nscale when GPU capacity is available and save the completed JSON artifact.
2. Add a multi-pod nscale test: two source pods, one target pod, one alternate
   source pod.
3. Promote the receiver tensor install smoke into live vLLM and SGLang process
   integration with each engine's post-load/refit lifecycle.
4. Add a real vLLM receiver artifact and a real SGLang receiver artifact after
   the cold-load path is stable.
5. Implement the actual quantized Qwen fallback install path after
   `GLOBAL_REQUIRED` metadata is detected.
6. Add a fanout simulator and nscale fanout microbenchmark for rollout replicas.
7. Run a first competitive timing table against a baseline all-gather/cat path
   and a direct NCCL P2P path.

## Current Claim We Can Safely Make

Safe claim:

> MX now has a planner and synthetic GPU POC showing that source-published slice
> ownership plus receiver-side slice requests can drive multi-source GPU
> assembly, same-node NIXL one-sided segment reads, and segment-level recovery
> without trainer full all-gather. The same slice ownership metadata now
> round-trips through a live Redis-backed central MX server on nscale and can be
> used by the target planner. The metadata side also covers real Qwen3 MoE and
> Qwen3 FP8 safetensors headers, including real global-required quantization
> metadata, plus CPU runtime-shaped vLLM/SGLang target tensor install smokes.

Unsafe claim:

> MX has implemented production on-the-fly trainer-to-inference resharding.

Unsafe claim:

> MX already beats NCCL Reshard, CheckpointEngine, Mooncake, TorchStore, RDT, or
> direct NIXL.
