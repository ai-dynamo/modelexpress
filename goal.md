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
- A completed 4-rank/4-B200 live-MX NIXL run on nscale proves that
  MX-returned slice ownership metadata drives the NIXL refit POC:
  `artifacts/resharding/nscale-live-mx-nixl-refit.json`. The run used source
  publications through a live Redis-backed central MX server, target-side
  planning from returned metadata, actual one-sided NIXL reads into planned
  target offsets, and segment-level replan from an alternate holder after a
  failed source segment.
- A second 4-rank/4-B200 live-MX NIXL endpoint run proves the target can fetch
  source NIXL agent metadata and remote CUDA tensor descriptors from MX worker
  metadata instead of torch distributed object gather:
  `artifacts/resharding/nscale-live-mx-nixl-endpoint-refit.json`. The artifact
  records `nixl_source_endpoints_from_mx=true`,
  `torch_distributed_nixl_metadata_exchange_used=false`, three MX-discovered
  source endpoints, actual one-sided NIXL reads, allclose/checksum validation,
  and no GPU reuse.
- A two-pod cross-node B200 live-MX NIXL endpoint run proves the same refit path
  can cross GPU nodes over UCX/IB rc:
  `artifacts/resharding/nscale-crossnode-mx-nixl-refit.json`. Source endpoint
  metadata was published by `mx-crossnode-refit-source` on
  `cluster-0967a26d-pool-14bee067-prctr-g2j7h`; the target ran in
  `mx-crossnode-refit-target` on
  `cluster-0967a26d-pool-14bee067-prctr-w4xnn`. The target discovered three
  source endpoints from MX, used `UCX_TLS=rc_x,rc,tcp,cuda_copy`, pinned
  `UCX_NET_DEVICES=mlx5_3:1`, excluded `mlx5_bond_0`, performed actual NIXL
  READs for 64 bytes across two source holders, and validated
  `allclose=true` with checksum `556224.0`.
- A stricter one-pod-per-source-rank cross-node B200 live-MX NIXL endpoint run
  proves independent source-rank pods can publish separate source endpoints and
  be consumed by one target on another GPU node:
  `artifacts/resharding/nscale-crossnode-one-pod-per-source-rank-target.json`.
  Three source pods (`mx-rankpod-src0`, `mx-rankpod-src1`, and
  `mx-rankpod-src2`) ran on
  `cluster-0967a26d-pool-14bee067-prctr-9c2x7`; the target pod
  `mx-rankpod-target` ran on
  `cluster-0967a26d-pool-14bee067-prctr-g2j7h`. The target discovered three
  MX source endpoints, added three distinct source NIXL agents, performed
  actual cross-node NIXL READs from `trainer-rank1` and
  `trainer-rank2-alt`, and validated `allclose=true` with checksum
  `556224.0`. The artifact records
  `mode=nixl-crossnode-one-pod-per-source-rank`, `gpu_count=4`,
  `one_nixl_agent_per_source_rank=true`, and no trainer all-gather or
  trainer-side inference-layout conversion.
- A source-churn/stale-source variant now proves the one-pod-per-source-rank
  path can recover when a source rank has published an MX/NIXL endpoint and
  then becomes STALE before the target reads:
  `artifacts/resharding/nscale-crossnode-stale-source-recovery-target.json`.
  Rank0 ran in `mx-stale-src0-stale-20260604-184226` and exited after publish;
  MX reported `trainer-rank0` as `SOURCE_STATUS_STALE`. The target pod
  `mx-stale-target-stale-20260604-184226` ran on
  `cluster-0967a26d-pool-14bee067-prctr-g2j7h`, discovered all three endpoints
  from MX, added/read only READY `trainer-rank1` and `trainer-rank2-alt`,
  replanned the stale rank0 segment from the alternate holder, and validated
  `allclose=true` with checksum `556224.0`. This is stale-before-read
  recovery; it does not prove a hard pod kill during an in-flight NIXL read.
- A synthetic one-pod-per-source-rank cross-node hard-kill run now proves
  segment-level recovery when a source pod dies after the target submits an
  in-flight NIXL READ:
  `artifacts/resharding/nscale-hard-kill-gpu-inflight-recovery-summary.json`,
  `artifacts/resharding/nscale-hard-kill-gpu-inflight-recovery-target.json`,
  and `artifacts/resharding/nscale-hard-kill-gpu-inflight-recovery-target.log`.
  Run `hardkillgpu2-20260604-190228` scheduled three independent source GPU
  pods on `cluster-0967a26d-pool-14bee067-prctr-9c2x7` and one target GPU pod
  on `cluster-0967a26d-pool-14bee067-prctr-g2j7h` with
  `UCX_TLS=rc_x,rc,tcp,cuda_copy` and `UCX_NET_DEVICES=mlx5_10:1`. The target
  emitted a post-submit marker for the 256 MiB `trainer-rank0` READ, rank0 was
  force-deleted, NIXL returned `NIXL_ERR_REMOTE_DISCONNECT`, the target
  replanned only the failed rank0 target range from `trainer-rank2-alt`, and the
  final 1 GiB target validated with `allclose=true` and matching checksum
  `1.3098182687131505e+24`. The raw target artifact was emitted before the
  proof-field fix, so the hard-kill claim is backed by
  `read_failure_recovery_used=true`, the read-failure/recovery segment records,
  and the derived summary artifact; future target artifacts set
  `failed_then_succeeded` and `replanned_only_failed_segments` directly.
- The live nscale server image used for that cross-node run drops the newer
  `slice_ownerships` proto field even though tensor/NIXL fields survive. The
  source now dual-writes ownership into the new proto field and a prefixed
  legacy string sidecar so older MX server deployments can still return
  ownership for endpoint planning. This is a compatibility bridge, not a
  replacement for the production schema.
- An earlier one-pod-per-source-rank cross-node attempt was capacity-blocked:
  `artifacts/resharding/nscale-one-pod-per-source-capacity-block.log` and
  `.json` record three independent source-rank GPU pods Pending with
  `0/29 nodes are available`, `10 Insufficient nvidia.com/gpu`, and
  autoscaler max node group size reached. That block is now superseded by the
  later checksum-backed one-pod-per-source-rank run above.
- A dedicated normalized Level-5 real timing table was also not run in that
  pass: `artifacts/resharding/nscale-level5-timing-capacity-block.json`
  records that comparable checksum-backed MX/NIXL, NCCL Reshard, and
  CheckpointEngine measured baselines remained unmeasured because the required
  multi-GPU nscale capacity was unavailable.
- Existing real Qwen3 BF16 timing jobs were audited in
  `artifacts/resharding/nscale-level5-existing-job-evidence-audit.json`. The
  audit records five-sample full-model/vLLM timing context for MX live refit,
  NCCL, and CheckpointEngine, including 61,064,245,248-byte Qwen3 payloads.
  Those jobs are useful competitive context, but they do **not** satisfy the
  Level-5 claim because they validate by live vLLM output-change rather than
  allclose/checksum and do not expose a common registration/publish/planner/
  read/install metric schema across all three rows.
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
- Same-node, one-pod live runtime bridge artifacts now connect actual UCX/NIXL
  segment reads into live SGLang and vLLM runtime-owned weights. The SGLang
  artifact reads 16,384 bytes from two trainer-like source ranks into CUDA
  staging, installs via `Engine.update_weights_from_tensor`, and validates
  allclose/checksum. The vLLM artifact reads 4,096 bytes from two trainer-like
  source ranks into CUDA staging, installs/restores through `LLM.apply_model`,
  and validates allclose/checksum. Those committed GPU artifacts still use the
  earlier deterministic source values and staging-copy runtime APIs; they do not
  prove direct NIXL writes into runtime-owned storage or a live trainer
  optimizer loop.
- Current branch runtime bridge source helpers now replace the static source
  formula with a source-rank `torch.optim.SGD` optimizer-step publisher over a
  small synthetic objective. `TrainerStepSourcePublication` now carries the
  post-step source tensor, annotated `SliceOwnership`, source lease/descriptor
  identity, and provenance together. The nscale CPU artifacts
  `artifacts/resharding/nscale-trainer-step-runtime-source-smoke-20260605.json`,
  `artifacts/resharding/nscale-trainer-step-source-publication-smoke-20260605.json`,
  and `artifacts/resharding/nscale-trainer-step-mx-publication-smoke-20260605.json`,
  plus pytest logs, prove source-owned ranges reconstruct the post-step target
  tensor, source-publication metadata can be published/listed/planned through
  the MX metadata client path, and vLLM/SGLang source-rank paths now expose
  optimizer-step publication metadata. Live same-node, one-pod vLLM+NIXL and
  SGLang+NIXL GPU reruns of this updated path are now banked in
  `artifacts/resharding/nscale-live-vllm-nixl-runtime-trainer-step-20260605.json`
  and
  `artifacts/resharding/nscale-live-sglang-nixl-runtime-trainer-step-20260605.json`,
  plus summary JSONs. Both use two source ranks publishing
  post-`torch.optim.SGD` source shards, actual UCX/NIXL reads into CUDA staging,
  runtime engine install/restore APIs, and allclose/checksum gates. The SGLang
  rerun explicitly pins `UCX_NET_DEVICES=mlx5_10:1` to avoid the known
  `mlx5_bond_0` UCX failure and validates runtime readback against the BF16
  storage round-trip while keeping the NIXL staging gate full precision. These
  supersede the earlier same-node trainer-step GPU capacity/runtime blocks for
  vLLM and SGLang scope, but remain same-node/one-pod GPU-reuse proofs over a
  synthetic objective. The vLLM and SGLang cross-node proofs below now cover
  one-pod-per-source-rank runtime placement for tiny runtime weights with
  staging-copy install; direct NIXL landing into runtime-owned storage and a
  real RL trainer loop remain unproven.
  A live `mx-server-rl` trainer-step publication pass is now banked in
  `artifacts/resharding/nscale-live-mx-trainer-step-publication-sidecar-pass-20260605.json`
  and `.log`: the deployed server still drops the new repeated ownership field,
  but preserves the legacy `metadata_endpoint` sidecar, and the client lists and
  plans trainer-step slices from that returned sidecar metadata.
- Source publication now has a versioned trainer-loop smoke on top of the
  single-step publisher:
  `artifacts/resharding/nscale-trainer-loop-source-publication-smoke-20260605.json`
  plus
  `artifacts/resharding/nscale-trainer-loop-source-publication-pytest-20260605.log`.
  It publishes coherent step-specific model versions, leases, descriptors, and
  provenance across two source-rank ownerships, reconstructs the step-2 target
  from source-owned ranges, proves step-2 differs from step-1, and round-trips
  loop ownership metadata through the MX metadata planning path in tests. This
  is still a synthetic trainer-loop smoke over a deterministic
  `torch.optim.SGD` objective; it is not a real distributed RL/FSDP trainer
  loop.
- The vLLM and SGLang MX runtime bridge source helpers now consume the
  versioned trainer-loop publication path instead of calling the single-step
  source publisher directly. Evidence:
  `artifacts/resharding/nscale-runtime-bridges-trainer-loop-publication-smoke-20260605.json`
  plus
  `artifacts/resharding/nscale-runtime-bridges-trainer-loop-publication-pytest-20260605.log`.
  The smoke proves both bridge source paths publish `trainer_loop_publisher`
  metadata, use step-specific source/request model versions accepted by the MX
  endpoint planner, and reconstruct the step-2 target for vLLM and SGLang
  bridge-shaped ownerships. This is CPU/control-plane bridge evidence only; it
  is not by itself a live GPU vLLM/SGLang runtime rerun and not real
  distributed trainer ownership.
- A live vLLM GPU rerun now closes the vLLM side of that bridge-to-runtime gap
  for trainer-loop metadata:
  `artifacts/resharding/nscale-live-vllm-mx-runtime-trainer-loop-crossnode-20260605.json`,
  source JSONs, target/source logs, and pod-placement log. In run
  `mxvllmloop-20260605-ib2`, source pods
  `mx-vllm-loop-src0-20260605` and `mx-vllm-loop-src1-20260605` ran on
  `cluster-0967a26d-pool-14bee067-prctr-g2j7h`, each published a versioned
  trainer-loop step-2 `SliceOwnership`, CUDA tensor descriptor, and NIXL
  metadata endpoint through live `mx-server-rl:8001`. The target pod
  `mx-vllm-loop-tgt-20260605` ran on
  `cluster-0967a26d-pool-14bee067-prctr-9c2x7`, loaded live vLLM 0.17.1,
  discovered both source endpoints from MX, issued two cross-node UCX/NIXL READs
  into CUDA staging, inferred the expected optimizer step from source ownership
  metadata, installed/restored through `LLM.apply_model`, and validated staging
  allclose, runtime allclose, and checksum. The artifact records
  `result=pass`, `cross_node=true`, `one_pod_per_source_rank=true`,
  `trainer_loop_source_publication_used=true`,
  `receiver_expected_update_from_source_metadata=true`,
  `expected_optimizer_step_count=2`, `trainer_to_inference_bytes=4096`,
  `raw_nixl_read_duration_ms=11.092181084677577`,
  `metadata_query_duration_ms=28.369619976729155`,
  `planner_duration_ms=0.1897109905257821`, and
  `activation_install_duration_ms=0.15898107085376978`. Scope boundary: vLLM
  only, tiny single tensor, staging-copy install, deterministic trainer-loop
  smoke over `torch.optim.SGD`, no direct NIXL landing into vLLM-owned storage,
  and no real RL trainer loop.
- A live SGLang GPU rerun now closes the SGLang side of the same
  bridge-to-runtime gap for trainer-loop metadata:
  `artifacts/resharding/nscale-live-sglang-mx-runtime-trainer-loop-crossnode-20260605.json`,
  source JSONs, target/source logs, pod placement/describe logs, and
  source/target IB/GPU snapshots. In run `mxsglangloop-20260605-ib1`, source
  pods `mx-sglang-loop-src0-20260605` and `mx-sglang-loop-src1-20260605` ran on
  `cluster-0967a26d-pool-14bee067-prctr-g2j7h`, each published a versioned
  trainer-loop step-2 `SliceOwnership`, CUDA tensor descriptor, and NIXL
  metadata endpoint through live `mx-server-rl:8001`. The target pod
  `mx-sglang-loop-tgt-20260605` ran on
  `cluster-0967a26d-pool-14bee067-prctr-9c2x7`, loaded live SGLang
  `0.0.0.dev1+g229cadec0`, discovered both source endpoints from MX, issued two
  cross-node UCX/NIXL READs into CUDA staging, inferred the expected optimizer
  step from source ownership metadata, installed/restored through
  `Engine.update_weights_from_tensor`, and validated staging allclose, runtime
  allclose, and checksum. The artifact records `result=pass`,
  `cross_node=true`, `one_pod_per_source_rank=true`,
  `trainer_loop_source_publication_used=true`,
  `receiver_expected_update_from_source_metadata=true`,
  `expected_optimizer_step_count=2`, `runtime_storage_dtype=bfloat16`,
  `trainer_to_inference_bytes=16384`,
  `raw_nixl_read_duration_ms=10.27112896554172`,
  `metadata_query_duration_ms=41.94057593122125`,
  `planner_duration_ms=0.1561130629852414`, and
  `activation_install_duration_ms=13.772701029665768`. Scope boundary: SGLang
  only, tiny single tensor, staging-copy install, deterministic trainer-loop
  smoke over `torch.optim.SGD`, no direct NIXL landing into SGLang-owned
  storage, and no real RL trainer loop.
- `modelexpress.refit_vllm_mx_runtime` now proves cross-node vLLM+NIXL runtime
  refit with the MX endpoint control plane and one pod per source rank. In run
  `mxvllmrt-20260605-ib3`, two independent source pods
  (`mx-vllm-mx-src0-20260605` and `mx-vllm-mx-src1-20260605`) ran on
  `cluster-0967a26d-pool-14bee067-prctr-g2j7h`, each published one
  optimizer-step `SliceOwnership`, CUDA tensor descriptor, and NIXL metadata
  endpoint through live `mx-server-rl:8001`. The target pod
  `mx-vllm-mx-tgt-20260605` ran on
  `cluster-0967a26d-pool-14bee067-prctr-9c2x7`, loaded live vLLM 0.17.1,
  built its receiver `SliceRequest` from the worker-owned `lm_head.weight`,
  discovered both source endpoints from MX, issued two cross-node UCX/NIXL READs
  over `UCX_TLS=rc_x,rc,tcp,cuda_copy`/`UCX_NET_DEVICES=mlx5_10:1`, installed
  through `LLM.apply_model`, and validated staging allclose, runtime allclose,
  and checksum. Evidence:
  `artifacts/resharding/nscale-live-vllm-mx-runtime-crossnode-20260605.json`,
  source JSONs, pod placement/describe logs, and source/target IB/GPU snapshots.
  The target artifact records `result=pass`, `cross_node=true`,
  `one_pod_per_source_rank=true`, `trainer_to_inference_bytes=4096`,
  `raw_nixl_read_duration_ms=9.514622972346842`,
  `metadata_query_duration_ms=31.619531917385757`,
  `planner_duration_ms=0.18348696175962687`, and checksum/allclose gates. Scope
  boundary: this is tiny single-tensor vLLM runtime evidence with staging-copy
  install and a synthetic optimizer objective; it is not direct NIXL landing
  into vLLM-owned storage, full-model refit, or a real RL trainer loop.
- `modelexpress.refit_sglang_mx_runtime` now proves cross-node SGLang+NIXL
  runtime refit with the MX endpoint control plane and one pod per source rank.
  In run `mxsglangrt-20260605-ib1`, two independent source pods
  (`mx-sglang-mx-src0-20260605` and `mx-sglang-mx-src1-20260605`) ran on
  `cluster-0967a26d-pool-14bee067-prctr-g2j7h`, each published one
  optimizer-step `SliceOwnership`, CUDA tensor descriptor, and NIXL metadata
  endpoint through live `mx-server-rl:8001`. The target pod
  `mx-sglang-mx-tgt-20260605` ran on
  `cluster-0967a26d-pool-14bee067-prctr-9c2x7`, loaded live SGLang
  `0.0.0.dev1+g229cadec0`, built its receiver `SliceRequest` from
  `lm_head.weight`, discovered both READY source endpoints from MX, issued two
  cross-node UCX/NIXL READs into CUDA staging, installed through
  `Engine.update_weights_from_tensor`, and validated staging allclose, runtime
  allclose, and checksum. Evidence:
  `artifacts/resharding/nscale-live-sglang-mx-runtime-crossnode-20260605.json`,
  source JSONs, pod placement/describe logs, and source/target IB/GPU snapshots.
  The target artifact records `result=pass`, `cross_node=true`,
  `one_pod_per_source_rank=true`, `trainer_to_inference_bytes=16384`,
  `raw_nixl_read_duration_ms=12.166149914264679`,
  `metadata_query_duration_ms=44.608733034692705`,
  `planner_duration_ms=0.14281098265200853`, and checksum/allclose gates. Scope
  boundary: this is tiny single-tensor SGLang runtime evidence with
  staging-copy install and a synthetic optimizer objective; it is not direct
  NIXL landing into SGLang-owned storage, full-model refit, or a real RL
  trainer loop.
- `artifacts/resharding/nscale-live-vllm-mx-runtime-crossnode-bw4bt-startup-block-20260605.json`
  and `.log` bank the first attempted source placement on
  `cluster-0967a26d-pool-14bee067-prctr-bw4bt`. IB was active on `mlx5_10`, but
  the source Python process and `nvidia-smi` entered uninterruptible sleep before
  any source-ready artifact was written. This is a node/GPU runtime startup
  block, not a proof; it is superseded by the successful `g2j7h` to `9c2x7`
  cross-node vLLM runtime run above.
- Target-side runtime read-failure recovery logic now exists in the cross-node
  harness. A nscale unit test simulates a READY primary source failing during
  its read group, verifies that only that failed source range is replanned from
  `trainer-rank2-alt`, and preserves existing stale-source behavior:
  `artifacts/resharding/nscale-runtime-read-failure-recovery-pytest.log`.
- Hard-kill proof harness support exposes scaled cross-node payloads and a
  NIXL post-submit marker/sleep hook so orchestration can delete a source pod
  after the target submits a READ:
  `artifacts/resharding/nscale-hard-kill-harness-support-smoke.json` and
  `artifacts/resharding/nscale-hard-kill-harness-pytest.log` (`4 passed`). A
  first GPU attempt scheduled the source/target pods and reached target NIXL
  setup, but was blocked before the kill marker by byte-valued NIXL remote agent
  metadata in the post-submit probe:
  `artifacts/resharding/nscale-hard-kill-gpu-attempt-json-serialization-block.json`.
  The probe now serializes byte metadata safely, and the later hard-kill run
  above supersedes the block for synthetic in-flight GPU source-pod kill proof.
- A competitive refit simulator now compares MX direct bipartite P2P,
  MX primary/replica fanout, NCCL Reshard-style fixed-membership full-tensor
  movement, and CheckpointEngine-style full gather/apply. The committed nscale
  artifact `artifacts/resharding/competitive-refit-simulation.json` shows a
  two-step RL rollout case where MX primary/replica fanout moves each unique
  requested slice once across the trainer/inference boundary and is preferred
  under the declared assumptions.

The cross-node endpoint and hard-kill artifacts are **Level 3 cross-node
synthetic proof**, not production refit implementation. Level 3 control-plane
metadata lifecycle support is proven through a live central MX server smoke, and
live MX-returned ownership plus source endpoint metadata now drives completed
same-node, two-pod cross-node, one-pod-per-source-rank cross-node, and synthetic
hard-kill recovery NIXL GPU data-plane runs. Level 4 now has same-node,
one-pod live vLLM and SGLang runtime bridge proofs where NIXL reads from
trainer-like source ranks feed runtime-owned weights through engine APIs, and
both runtime bridges have same-node GPU reruns whose source ranks publish
optimizer-step shards instead of deterministic replacement values. vLLM and
SGLang also have checksum-backed cross-node, one-pod-per-source-rank MX-endpoint
runtime bridges with staging-copy install, plus trainer-loop-metadata
cross-node GPU reruns for both runtimes. Full production real
trainer/inference refit remains unproven.

## What Is Not Proven Yet

The current POC does **not** prove:

- Real trainer process integration with FSDP, TP, PP, EP, or RL training loop.
- Full live vLLM or SGLang process integration with real trainer-process
  payloads, direct runtime-buffer NIXL landing, and the production
  post-load/refit lifecycle. Tiny live vLLM V1 and SGLang Engine-owned
  tensor/weight smokes are proven, same-node vLLM/SGLang NIXL-to-runtime
  bridges are proven, both runtimes have same-node GPU reruns using
  optimizer-step source publications, and both vLLM and SGLang now have
  checksum-backed cross-node one-pod-per-source-rank runtime bridges. Still
  unproven: direct NIXL landing into runtime-owned storage, full-model/runtime
  lifecycle integration, and a real RL trainer loop.
- A hard source pod kill during an in-flight NIXL read against real
  trainer-owned and runtime-owned tensors. The synthetic cross-node GPU harness
  now proves the segment-level recovery mechanism under a forced source pod
  deletion, but it is not real trainer/runtime integration.
- Full-model or multi-layer refit.
- Runtime installation of real Qwen MoE model tensors.
- End-to-end runtime fallback installation using real Qwen FP8 payload bytes
  and real runtime-owned model tensors after `global-required` metadata is
  detected. The helper path is implemented and tested with a real manifest
  entry, but payload bytes are still synthetic.
- Hierarchical fanout to many rollout replicas.
- Versioned rollback using multiple GPU-resident training steps. A
  receiver-side CPU/runtime tensor transaction helper now proves rollback
  semantics for multiple versions, but not GPU-resident rollback.
- Performance competitiveness against NCCL Reshard, CheckpointEngine, Mooncake,
  or TorchStore-style systems beyond the committed CPU byte/cost simulator.
- A measured Level-5 timing table against NCCL Reshard and CheckpointEngine.
  Existing Qwen3 timing jobs are partial evidence only; a Level-5 row still
  needs a real checksum/allclose gate and normalized byte/timing fields for
  MX/NIXL, NCCL Reshard, and CheckpointEngine.

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

Status: implemented for same-node cross-GPU synthetic refit and a two-pod
cross-node synthetic refit over UCX/IB rc.

Goal:

- Two or more source ranks register disjoint GPU tensor ranges with NIXL.
- One target rank preallocates a target buffer.
- The target uses `SegmentPlan` to issue one-sided NIXL reads from multiple
  source holders into exact offsets.
- Source failure replans only failed segments from alternate holder.

Acceptance evidence:

- nscale same-node cross-GPU artifact:
  `artifacts/resharding/nscale-nixl-distributed-refit.json`.
- nscale two-pod cross-node artifact:
  `artifacts/resharding/nscale-crossnode-mx-nixl-refit.json`.
- Captures NIXL registration duration, metadata blob size/identity,
  `add_remote_agent`, prep duration, raw read duration, and checksum.
- Independent source-pod fan-in and synthetic hard-kill recovery are now
  covered by Level 3 artifacts; real runtime-owned tensors remain future work.

### Level 3: MX Control Plane Integration

Status: implemented for same-node and two-pod cross-node synthetic live-MX NIXL
refit; production runtime integration remains pending.

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
- Completed live-MX NIXL nscale GPU run:
  `artifacts/resharding/nscale-live-mx-nixl-refit.json`.
- Full proof log:
  `artifacts/resharding/nscale-live-mx-nixl-refit.log`.
- Live central-server log from that proof:
  `artifacts/resharding/nscale-live-mx-nixl-server.log`.
- MX-discovered NIXL endpoint proof:
  `artifacts/resharding/nscale-live-mx-nixl-endpoint-refit.json`.
- Endpoint proof log:
  `artifacts/resharding/nscale-live-mx-nixl-endpoint-refit.log`.
- Endpoint proof server log:
  `artifacts/resharding/nscale-live-mx-nixl-endpoint-server.log.gz`.
- Endpoint control-plane nscale tests:
  `artifacts/resharding/nscale-refit-endpoint-control-plane-pytest.log`
  (`7 passed`).
- Two-pod cross-node MX endpoint NIXL proof:
  `artifacts/resharding/nscale-crossnode-mx-nixl-refit.json`.
- Cross-node source artifact and UCX debug log:
  `artifacts/resharding/nscale-crossnode-mx-nixl-source.json` and
  `artifacts/resharding/nscale-crossnode-mx-nixl-refit.log`.
- Cross-node placement/capacity log:
  `artifacts/resharding/nscale-crossnode-mx-nixl-capacity.log`.
- Earlier capacity-blocked attempt, superseded by the completed run:
  `artifacts/resharding/nscale-live-mx-nixl-capacity.log`.
- Qwen MoE manifest extractor in `modelexpress.resharding_manifest`.
- Real Qwen3-30B-A3B safetensors-header coverage artifact:
  `artifacts/resharding/qwen3-30b-a3b-moe-manifest.json.gz`.
- Real Qwen3-30B-A3B-FP8 safetensors-header coverage artifact:
  `artifacts/resharding/qwen3-30b-a3b-fp8-moe-manifest.json.gz`.
- Real Qwen3-30B-A3B-FP8 zero-copy fallback smoke:
  `artifacts/resharding/qwen3-30b-a3b-fp8-zero-copy-fallback-smoke.json`.

Remaining gap:

- Repeat stale-source and hard in-flight source-kill recovery against real
  trainer-owned and runtime-owned tensors. The current synthetic cross-node
  proof covers independent source pods, stale-before-read recovery, and forced
  source pod deletion after a target NIXL READ is submitted, but not real engine
  ownership or full-model install.

### Level 4: Real Runtime Refit

Status: partially implemented; CPU receiver-install, tiny live vLLM/SGLang
engine smokes, same-node SGLang+NIXL and vLLM+NIXL runtime bridges, and
cross-node one-pod-per-source-rank vLLM/SGLang+NIXL MX-endpoint runtime bridges
are implemented.

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
  torch tensors, installs planned segment payloads into target tensor slices,
  installs global-required quantization fallback payloads into runtime-owned
  metadata tensors, and can snapshot/rollback runtime tensors across
  model-version installs.
- `modelexpress_client/python/tests/test_resharding_receiver.py` covers
  vLLM-shaped and SGLang-shaped runtime tensor install smokes on nscale,
  including a real Qwen3 FP8 `weight_scale_inv` manifest entry for the
  fallback install path.
- Focused nscale fallback-install gate:
  `artifacts/resharding/nscale-qwen-fp8-runtime-fallback-install-pytest.log`
  (`17 passed`).
- Runtime fallback install artifact:
  `artifacts/resharding/qwen3-30b-a3b-fp8-runtime-fallback-install-smoke.json`.
- Runtime versioned rollback helper evidence:
  `artifacts/resharding/nscale-runtime-refit-versioned-rollback-smoke.json`
  and `artifacts/resharding/nscale-runtime-refit-versioned-rollback-pytest.log`
  (`11 passed`). The smoke installs two Qwen-style layer tensors for
  `step-8`, rolls back to `step-7`, then installs and commits `step-9`;
  it is CPU/runtime-tensor evidence, not GPU-resident rollback proof. It also
  rejects dtype drift before rollback.
- Full nscale Python gate:
  `artifacts/resharding/nscale-python-full-pytest.log`
  (`276 passed, 19 skipped`).
- `modelexpress.refit_vllm_receiver_smoke` now provides a live vLLM V1
  receiver-owned tensor smoke through `LLM.apply_model`, plus a
  framework-explicit module helper. The live entrypoint creates a tiny Qwen2
  checkpoint, starts a real vLLM 0.17.1 V1 engine, runs the receiver refit logic
  inside the worker-owned `Qwen2ForCausalLM`, builds a receiver-side
  `SliceRequest` from `lm_head.weight` on `cuda:0`, plans two synthetic
  trainer-held source ranges, installs planned payloads into that vLLM-owned
  tensor, validates checksum/allclose, and restores the original tensor.
  Evidence: `artifacts/resharding/nscale-live-vllm-receiver-applymodel-smoke-20260604.json`,
  `.log`, and `-placement.log`. This is live vLLM engine-owned tensor evidence,
  not real trainer/NIXL data-plane evidence. The earlier
  `nscale-live-vllm-receiver-v1-module-discovery-block-20260604.log` and
  `nscale-live-vllm-receiver-v0-env-unsupported-block-20260604.log` explain why
  the V1 path uses `LLM.apply_model` instead of parent-process module traversal
  or `VLLM_USE_V1=0`.
- `modelexpress.refit_vllm_nixl_runtime_smoke` now bridges the proven NIXL
  segment-read data plane into the live vLLM V1 worker update path. The nscale
  artifact
  `artifacts/resharding/nscale-live-vllm-nixl-runtime-refit-smoke-20260604.json`
  and `.log` prove a same-node, one-pod, 3-rank run on one GPU with explicit
  GPU reuse: source ranks 0/1 own CUDA trainer-like shard tensors, target rank
  2 starts a live vLLM `LLM`, builds a receiver-side `SliceRequest` from the
  worker-owned `lm_head.weight`, reads 4,096 bytes from the two source ranks
  over UCX/NIXL into a preallocated CUDA staging tensor, installs/restores the
  assembled tensor through `LLM.apply_model`, validates NIXL staging allclose,
  runtime allclose, checksum match, and restores the original weight. The
  artifact records `actual_nixl_reads_used=true`,
  `real_runtime_engine_used=true`,
  `source_rank_owned_trainer_tensors_used=true`,
  `nixl_reads_land_directly_in_runtime_tensor=false`,
  `runtime_update_payload_copied_through_apply_model=true`, and
  `real_training_loop_used=false`. This is real NIXL-to-live-vLLM runtime
  evidence, but it is not cross-node, not direct zero-copy into vLLM-owned
  storage, and not a live trainer/optimizer loop.
- `modelexpress.refit_trainer_step` now provides the shared optimizer-step
  source publisher used by the vLLM/SGLang NIXL runtime bridge source helpers.
  It materializes only the source-owned range from a source-rank
  `torch.optim.SGD` parameter step over a synthetic objective and records
  explicit provenance (`optimizer_step_publisher_used=true`,
  `static_replacement_formula_used=false`). `TrainerStepSourcePublication`
  carries the source tensor, annotated `SliceOwnership`, source lease/descriptor
  identity, and artifact metadata together so future MX publication can use one
  contract. The in-memory MX metadata path now publishes trainer-step
  source publications, lists them back, and plans receiver slices from the
  returned ownership metadata. nscale CPU/control-plane evidence:
  `artifacts/resharding/nscale-trainer-step-runtime-source-smoke-20260605.json`,
  `artifacts/resharding/nscale-trainer-step-source-publication-smoke-20260605.json`,
  `artifacts/resharding/nscale-trainer-step-mx-publication-smoke-20260605.json`,
  `artifacts/resharding/nscale-trainer-step-mx-publication-pytest-20260605.log`,
  and `artifacts/resharding/nscale-live-mx-trainer-step-publication-sidecar-pass-20260605.json`
  prove the live MX publication path as well. The focused nscale log records
  `12 passed` for in-memory control-plane coverage plus `2 passed` for live
  `mx-server-rl`; live discovery used the sidecar because the deployed server
  still dropped the repeated ownership field. A live vLLM GPU rerun of the
  updated path was attempted but blocked by 1-GPU nscale capacity:
  `artifacts/resharding/nscale-live-vllm-nixl-runtime-trainer-step-capacity-block-20260605.json`.
- `modelexpress.refit_trainer_step` also exposes
  `TrainerLoopStepPublication` for coherent versioned source publication across
  source ranks. It resets leases/descriptors per trainer-loop step, records
  `trainer_loop_publisher_used=true`, and keeps the no-overclaim flags explicit
  (`synthetic_trainer_loop_smoke_used=true`,
  `real_rl_training_loop_used=false`). Evidence:
  `artifacts/resharding/nscale-trainer-loop-source-publication-smoke-20260605.json`
  and
  `artifacts/resharding/nscale-trainer-loop-source-publication-pytest-20260605.log`
  (`29 passed`). This moves source publication closer to a trainer loop, but it
  is still a deterministic CPU/control-plane smoke, not a real RL trainer.
- `modelexpress.refit_vllm_mx_runtime` and
  `modelexpress.refit_sglang_mx_runtime` now derive a shared
  `*-trainer-loop-step-*` runtime model version and materialize source
  publications through `TrainerLoopStepPublication`. The CPU/control-plane
  bridge artifact
  `artifacts/resharding/nscale-runtime-bridges-trainer-loop-publication-smoke-20260605.json`
  records `runtime_bridge_count=2`, `source_publication_count=4`,
  `segment_count=4`, `trainer_to_inference_bytes=192`, allclose/checksum pass,
  and explicit no-overclaim flags (`gpu_runtime_rerun_used=false`,
  `real_distributed_trainer_loop_used=false`). Focused nscale pytest:
  `artifacts/resharding/nscale-runtime-bridges-trainer-loop-publication-pytest-20260605.log`
  (`29 passed`). This updates the bridge source code path; the prior cross-node
  GPU artifacts remain the latest live runtime proof until rerun.
- `modelexpress.refit_sglang_receiver_smoke` now has both the earlier
  SGLang-shaped module helper and a live `sglang.Engine` weight-update smoke.
  `artifacts/resharding/nscale-sglang-receiver-smoke.json` proves the
  module-shaped path with `real_runtime_engine_used=false`.
  `artifacts/resharding/nscale-live-sglang-engine-receiver-smoke-20260604.json`,
  `.log`, and `-placement.log` prove a 1-GPU live SGLang Engine path: the
  CLI creates a tiny Llama checkpoint, starts SGLang
  `0.0.0.dev1+g229cadec0`, fetches `lm_head.weight` via
  `Engine.get_weights_by_name`, builds a receiver-side `SliceRequest`, plans
  two synthetic trainer-held ranges, assembles the replacement from segment
  payloads, installs it through `Engine.update_weights_from_tensor`, validates
  checksum/allclose through SGLang, and restores the original weight. This is
  live SGLang engine-owned weight evidence, not real trainer/NIXL data-plane
  evidence.
- `modelexpress.refit_sglang_nixl_runtime_smoke` now bridges the proven
  NIXL segment-read data plane into the live SGLang Engine update path. The
  nscale artifact
  `artifacts/resharding/nscale-live-sglang-nixl-runtime-refit-smoke-20260604.json`
  and `.log` prove a same-node, one-pod, 3-rank run on one GPU with explicit
  GPU reuse: source ranks 0/1 own CUDA trainer-like shard tensors, target rank
  2 starts a live SGLang `Engine`, builds a receiver-side `SliceRequest` from
  `lm_head.weight`, reads 16,384 bytes from the two source ranks over UCX/NIXL
  into a preallocated CUDA staging tensor, installs that assembled tensor via
  `Engine.update_weights_from_tensor`, validates NIXL staging allclose, runtime
  allclose, checksum match, and restores the original weight. The artifact
  records `actual_nixl_reads_used=true`, `real_runtime_engine_used=true`,
  `source_rank_owned_trainer_tensors_used=true`,
  `nixl_reads_land_directly_in_runtime_tensor=false`, and
  `real_training_loop_used=false`. This is real NIXL-to-live-SGLang runtime
  evidence, but it is not cross-node, not direct zero-copy into SGLang-owned
  storage, and not a live trainer/optimizer loop. The block logs
  `nscale-live-sglang-nixl-runtime-refit-smoke-20260604-outer-dist-before-sglang-block.log`
  and `nscale-live-sglang-nixl-runtime-refit-smoke-20260604-old-runner-copy-block.log`
  bank the earlier SGLang startup-order hangs that were fixed by starting
  SGLang before the outer Gloo process group on the target rank. The matching
  nscale Python gate is
  `artifacts/resharding/nscale-python-full-pytest-sglang-nixl-runtime-20260604.log`
  (`311 passed, 19 skipped`).
- `artifacts/resharding/nscale-live-sglang-nixl-runtime-trainer-step-20260605.json`
  and summary JSON now prove the same SGLang bridge with optimizer-step source
  publications instead of deterministic source values. The run cloned commit
  `58b37adc7e3a3a5d8a1893ac6fcc34f9f5cdca92`, pinned
  `UCX_NET_DEVICES=mlx5_10:1`, performed two actual UCX/NIXL reads for
  16,384 trainer-to-inference bytes, installed/restored through
  `Engine.update_weights_from_tensor`, and validated NIXL staging allclose,
  runtime allclose, and checksum. It records
  `trainer_optimizer_step_publisher_used=true`,
  `synthetic_source_values_used=false`, `runtime_storage_dtype=bfloat16`, and
  a 0.03125 full-precision-to-BF16 readback delta. Scope remains same-node,
  one-pod, GPU-reuse evidence over a synthetic optimizer objective.
- SGLang runtime availability is now separately banked.
  `artifacts/resharding/nscale-sglang-runtime-import-probe.json` verifies
  `torch`, `sglang`, and `sglang.srt` import in the nscale SGLang runtime
  builder pod. `artifacts/resharding/nscale-sglang-gpu-import-smoke.json` uses
  the Docker-published SGLang image in a 1-GPU nscale pod, sees CUDA with one
  device, and imports SGLang. This is availability evidence only.
- The earlier current-branch 1-GPU live vLLM receiver smoke pod was attempted
  under a submitted spec that did not schedule:
  `artifacts/resharding/nscale-live-vllm-receiver-smoke-capacity-block-20260604.json`
  and `.log`. That block is now superseded by the checksum-backed
  `nscale-live-vllm-receiver-applymodel-smoke-20260604.json` run; keep it only
  as submitted-spec capacity history.

### Level 5: Competitive Benchmark

Status: synthetic same-node measured benchmark pass. The current branch now
has checksum/allclose-gated, real measured rows for MX/NIXL, NCCL
Reshard-style, and CheckpointEngine-style refit in the same 4-GPU B200
same-node/single-pod synthetic placement. This is not a full-model,
cross-node, or production competitiveness claim; those Level-5 variants remain
unproven.

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

Current partial evidence:

- `modelexpress.resharding.simulate_competitive_refit` compares MX direct P2P,
  MX primary/replica fanout, NCCL Reshard-style fixed-membership full-tensor
  movement, and CheckpointEngine-style full gather/apply.
- `artifacts/resharding/competitive-refit-simulation.json` records a two-step
  RL rollout case with four compatible receiver requests spanning two trainer
  holders. It reports trainer-to-inference bytes, inference-side fanout bytes,
  trainer collective bytes, checkpoint storage bytes, redundant byte factors,
  segment count, source/target balance, source count per target tensor, and
  predicted bottlenecks.
- nscale full Python gate:
  `artifacts/resharding/nscale-python-full-pytest.log`
  (`276 passed, 19 skipped`).
- `modelexpress.refit_level5` provides the checksum/allclose-gated
  normalized timing-table schema and synthetic same-node GPU baseline runners
  for NCCL Reshard-style full-tensor movement and CheckpointEngine-style
  full-gather/write/read/apply.
- `artifacts/resharding/nscale-level5-same-node-synthetic-table-20260605.json`
  is the first measured Level-5 table with all required strategies present in
  the same placement scope. It reports `result=pass`,
  `level5_synthetic_smoke_pass=true`, `production_competitive_claim_safe=false`,
  and `level5_full_model_claim_safe=false`. Rows are real measured and
  checksum/allclose gated: MX/NIXL reads move 64 trainer-to-inference bytes
  with redundant factor 1.0, NCCL Reshard-style moves 128 bytes with redundant
  factor 2.0, and CheckpointEngine-style records 256 checkpoint-storage bytes
  with redundant factor 4.0.
- `artifacts/resharding/nscale-level5-nccl-reshard-baseline-20260605.json` and
  `artifacts/resharding/nscale-level5-checkpoint-engine-baseline-20260605.json`
  are the 4-GPU B200 same-node baseline rows that replaced the earlier capacity
  block. `artifacts/resharding/nscale-level5-baseline-summary-20260605.json`
  records `nccl_return_code=0`, `checkpoint_engine_return_code=0`, and
  `table_return_code=0` for pod `mx-level5-baseline-20260605-1228`.
- Focused nscale Python gate for the Level-5 normalizer and existing refit
  control-plane paths: `artifacts/resharding/nscale-level5-normalizer-pytest-20260605.log`
  (`19 passed`).
- `artifacts/resharding/nscale-level5-same-node-synthetic-table-missing-baselines.json`
  remains the historical missing-baseline table and is superseded by
  `artifacts/resharding/nscale-level5-same-node-synthetic-table-20260605.json`.
- `artifacts/resharding/nscale-level5-baseline-capacity-block.json` and
  `artifacts/resharding/nscale-level5-baseline-capacity-block-20260605.json`
  remain historical scheduler blocks. They are superseded for the synthetic
  same-node baseline scope by the successful 4-GPU B200 run above.

## Near-Term Achievable Work

These are the next useful things to do, in order:

1. Continue replacing the synthetic optimizer-step publisher in the vLLM/SGLang
   NIXL runtime bridges with real trainer-loop integration. The current branch
   now has CPU tests, versioned trainer-loop publication smoke, vLLM/SGLang
   bridge source paths wired to trainer-loop publication metadata, same-node
   vLLM/SGLang GPU reruns, and cross-node one-pod-per-source-rank runtime
   evidence for both runtimes; the next source-side step is real FSDP/TP/PP/EP
   or RL trainer ownership publication and a GPU rerun of the loop-published
   runtime bridge path.
2. Extend the runtime bridge placement work beyond tiny single-tensor vLLM and
   SGLang targets while keeping the allclose/checksum gates.
3. Promote the runtime bridges into each engine's production post-load/refit
   lifecycle, using the same source-published ownership and receiver-side
   request path.
4. Repeat stale-before-read and hard in-flight source-kill recovery against
   real trainer-owned/runtime-owned tensors once live receiver ownership exists.
5. Scale the vLLM and SGLang receiver artifacts beyond tiny single-tensor
   smokes after the cold-load path is stable.
6. Extend the quantized Qwen fallback from helper-level runtime tensor install
   to real Qwen FP8 payload bytes and real engine-owned model tensors.
7. Extend versioned rollback from CPU/runtime-tensor transaction semantics
   to GPU-resident rollback across multiple training steps.
8. Add an nscale fanout microbenchmark for rollout replicas using the simulator
   scenario as the shape contract.
9. Re-run the synthetic same-node Level-5 baseline pod when 4 GPUs are
   schedulable, then generate a passing normalized table only if MX/NIXL,
   NCCL Reshard, and CheckpointEngine rows all have checksum/allclose gates.
10. After the synthetic table passes, repeat the same schema for real Qwen/full
    runtime rows before making any competitive Level-5 claim.

## Current Claim We Can Safely Make

Safe claim:

> MX now has a planner and synthetic GPU POC showing that source-published slice
> ownership plus receiver-side slice requests can drive multi-source GPU
> assembly, same-node and cross-node NIXL one-sided segment reads, and
> segment-level recovery without trainer full all-gather. The same slice
> ownership metadata now round-trips through a live Redis-backed central MX
> server on nscale and has driven completed 4-B200 live-MX NIXL refit POCs where
> the target plans from MX-returned metadata and gets source NIXL endpoint
> handles from MX worker metadata rather than torch object gather. The
> cross-node evidence includes two-pod, one-pod-per-source-rank, stale-source,
> and synthetic in-flight source-pod hard-kill recovery runs over UCX/IB rc, each
> gated by checksum/allclose. The metadata side also covers real Qwen3 MoE and
> Qwen3 FP8 safetensors headers, including real global-required quantization
> metadata, plus CPU runtime-shaped vLLM/SGLang target tensor install smokes. A
> CPU competitive simulator now records MX direct, MX fanout, NCCL Reshard, and
> CheckpointEngine-style byte/cost comparisons for a two-step RL rollout
> scenario. Tiny live vLLM and SGLang engine-owned smokes prove receiver-owned
> install/restore through `LLM.apply_model` and
> `Engine.update_weights_from_tensor`. Same-node, one-pod vLLM+NIXL and
> SGLang+NIXL runtime bridges now prove trainer-like source-rank CUDA shards can
> be NIXL-read into staging and installed through engine APIs with
> allclose/checksum, and both have same-node optimizer-step source reruns. vLLM
> and SGLang now also have cross-node, one-pod-per-source-rank MX-endpoint
> runtime bridges over UCX/IB rc with checksum/allclose. These runtime artifacts
> still use tiny single-tensor payloads, staging-copy runtime APIs, synthetic
> optimizer objectives, and no live RL trainer loop; direct runtime-buffer NIXL
> landing, full-model refit, and production real trainer/runtime refit remain
> unproven.

Unsafe claim:

> MX has implemented production on-the-fly trainer-to-inference resharding.

Unsafe claim:

> MX already beats NCCL Reshard, CheckpointEngine, Mooncake, TorchStore, RDT, or
> direct NIXL.
