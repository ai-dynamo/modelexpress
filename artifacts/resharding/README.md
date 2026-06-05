# Refit POC Artifacts

Generated during the nscale run for `goal.md`.

Large real-model manifests are banked as `.json.gz` plus `.summary.json` to
stay below the repository's 1000 KB added-file pre-commit limit. Expanded raw
JSON copies may exist locally but are ignored.

Files:

- `nscale-cpu-pytest.log`: CPU planner and metadata pytest gate.
- `nscale-single-gpu-refit.json`: one-GPU assembly and recovery smoke.
- `nscale-distributed-refit.json`: four-rank NCCL trainer-to-inference refit
  proof with two primary trainer holders, one alternate holder, and one target.
- `nscale-gpu-refit.log`: full GPU pod log, including CUDA inventory, NCCL
  setup, single-GPU JSON, distributed JSON, and artifact listing.
- `nscale-nixl-distributed-refit.json`: four-rank B200 NIXL/UCX same-node
  refit proof. Rank 3 preallocates the target buffer, reads one segment from
  rank 1, replans the failed rank 0 segment from alternate rank 2, and validates
  checksum/allclose.
- `nscale-nixl-gpu-refit.log`: full NIXL GPU pod log, including CUDA inventory,
  NIXL import/backend setup, auto NIC pinning, and the emitted JSON artifact.
- `nscale-control-plane-pytest.log`: nscale Python 3.12 test log for planner,
  slice ownership control-plane helpers, and K8s-service manifest preservation
  (`54 passed, 1 skipped`; the skipped test is the opt-in live-server smoke).
- `nscale-resharding-focused-pytest.log`: latest focused nscale Python 3.12
  gate covering planner, control-plane helpers, live-control-plane opt-in skip,
  and Qwen/safetensors manifest extraction (`24 passed, 1 skipped`).
- `nscale-python-full-pytest.log`: full nscale Python 3.12 package test tree
  after the safetensors/Qwen BF16+FP8 manifest, FP8 zero-copy fallback,
  receiver-install smoke, competitive simulator changes, and resharding/refit
  POC module split (`276 passed, 19 skipped`).
- `competitive-refit-simulation.json`: CPU byte/cost simulator artifact for a
  two-step RL rollout shape. It compares MX direct bipartite P2P, MX
  primary/replica fanout, NCCL Reshard-style fixed-membership full-tensor
  movement, and CheckpointEngine-style full gather/apply; under the declared
  assumptions it prefers MX primary/replica fanout.
- `nscale-qwen-fp8-fallback-pytest.log`: focused nscale Python 3.12 gate for
  Qwen manifest extraction plus the real FP8 zero-copy fallback smoke
  (`10 passed`).
- `docker-rust-p2p-tests.log`: Dockerized Rust server p2p unit-test summary
  covering proto/backend/service slice ownership round trips.
- `nscale-live-control-plane.log`: nscale live central MX server smoke. A
  Redis-backed Rust `modelexpress-server` receives slice ownership
  `PublishMetadata`, serves `GetMetadata`, accepts `UpdateStatus`, returns READY
  workers through `ListSources`, and lets the target planner build a multi-source
  `SegmentPlan`.
- `nscale-live-mx-nixl-refit.json`: completed 4-rank/4-B200 live-MX NIXL proof.
  Source ranks publish slice ownerships through a live Redis-backed central MX
  server, the target rank plans from MX-returned metadata, actual NIXL READs
  land at planned target offsets, and a failed source segment is replanned from
  an alternate holder.
- `nscale-live-mx-nixl-refit.log`: full torchrun log for the completed live-MX
  NIXL GPU proof, including CUDA inventory, NIXL setup, source publications,
  target metadata query, and emitted JSON artifact.
- `nscale-live-mx-nixl-server.log`: live Rust server log from the completed
  live-MX NIXL GPU proof.
- `nscale-refit-endpoint-control-plane-pytest.log`: focused nscale Python 3.12
  gate for MX-published refit NIXL endpoint metadata (`7 passed`).
- `nscale-live-mx-nixl-endpoint-refit.json`: completed 4-rank/4-B200 live-MX
  NIXL endpoint proof. Source ranks publish slice ownership, NIXL agent
  metadata, and remote CUDA tensor descriptors through MX; the target discovers
  three source endpoints from MX, plans from MX-returned ownerships, performs
  actual NIXL READs, and validates checksum/allclose. The proof records
  `torch_distributed_nixl_metadata_exchange_used=false`.
- `nscale-live-mx-nixl-endpoint-refit.log`: full torchrun log for the endpoint
  proof.
- `nscale-live-mx-nixl-endpoint-server.log.gz`: compressed live Rust server log
  for the endpoint proof. The raw log is compressed to stay below the
  repository's large-file hook limit.
- `nscale-live-mx-nixl-capacity.log`: earlier attempted live-MX NIXL GPU
  verification that was capacity-blocked (`Insufficient nvidia.com/gpu`;
  autoscaler max size reached). It is superseded by
  `nscale-live-mx-nixl-refit.json`.
- `nscale-crossnode-mx-nixl-capacity.log`: placement evidence for the completed
  two-pod cross-node proof. It records `mx-crossnode-refit-source` and
  `mx-crossnode-refit-target` running on distinct GPU nodes.
- `nscale-crossnode-mx-nixl-source.json`: source-side publication artifact for
  the completed cross-node proof. It records source endpoint publication
  through MX, UCX backend selection, automatic non-bond NIC pinning, and the
  source pod/node.
- `nscale-crossnode-mx-nixl-source.log`: source-side NIXL/UCX startup log for
  the completed cross-node proof.
- `nscale-crossnode-mx-nixl-refit.json`: completed two-pod cross-node live-MX
  NIXL endpoint proof. The target pod on a second GPU node discovers source
  NIXL endpoint metadata from MX, plans from MX-returned ownership metadata,
  performs actual UCX/IB rc NIXL READs into planned target offsets, and
  validates checksum/allclose. The artifact records `cross_node=true`,
  `nixl_source_endpoints_from_mx=true`,
  `torch_distributed_nixl_metadata_exchange_used=false`,
  `UCX_TLS=rc_x,rc,tcp,cuda_copy`, `UCX_NET_DEVICES=mlx5_3:1`, and
  `mlx5_bond_excluded_from_ucx_devices=true`.
- `nscale-crossnode-mx-nixl-refit.log`: target-side UCX/NIXL debug log for the
  completed cross-node proof, including endpoint wait, NIXL agent construction,
  remote-agent add, inter-node rc lane setup, segment READs, and validation.
- `nscale-crossnode-one-pod-per-source-rank-target.json`: completed
  one-pod-per-source-rank cross-node live-MX NIXL endpoint proof. Three
  independent source-rank pods publish one source endpoint and one distinct
  NIXL source agent each; a target pod on a different GPU node discovers those
  endpoints from MX, performs planned NIXL READs from the needed source ranks,
  and validates checksum/allclose. The artifact records
  `mode=nixl-crossnode-one-pod-per-source-rank`, `gpu_count=4`,
  `source_endpoint_count=3`, `distinct_source_agent_count=3`,
  `one_nixl_agent_per_source_rank=true`, `trainer_to_inference_bytes=64`, and
  `raw_nixl_read_duration_ms=13.304431922733784`.
- `nscale-crossnode-one-pod-per-source-rank-source-rank0.json`,
  `nscale-crossnode-one-pod-per-source-rank-source-rank1.json`, and
  `nscale-crossnode-one-pod-per-source-rank-source-rank2-alt.json`:
  source-side publication artifacts for the one-pod-per-source-rank proof.
- `nscale-crossnode-one-pod-per-source-rank-capacity.log`: placement evidence
  for the one-pod-per-source-rank proof. It records three source pods on
  `cluster-0967a26d-pool-14bee067-prctr-9c2x7` and the target pod on
  `cluster-0967a26d-pool-14bee067-prctr-g2j7h`.
- `nscale-crossnode-one-pod-per-source-rank.log`: nscale-generated summary log
  for the one-pod-per-source-rank target gate.
- `nscale-crossnode-stale-source-recovery-target.json`: completed
  one-pod-per-source-rank cross-node stale-source recovery proof. Rank0
  published a source endpoint and exited; MX then reported
  `trainer-rank0` as `SOURCE_STATUS_STALE`. The target discovered all three
  source endpoints through MX with `status_filter=None`, added/read only READY
  `trainer-rank1` and `trainer-rank2-alt`, replanned the stale rank0 segment
  from the alternate holder, and validated checksum/allclose. The artifact
  records `stale_source_recovery_used=true`,
  `stale_source_ids_excluded_from_nixl_reads=true`,
  `trainer_to_inference_bytes=64`, and
  `raw_nixl_read_duration_ms=15.527020092122257`. This proves
  stale-before-read recovery, not a mid-read pod kill.
- `nscale-crossnode-stale-source-recovery-source-rank0.json`,
  `nscale-crossnode-stale-source-recovery-source-rank1.json`, and
  `nscale-crossnode-stale-source-recovery-source-rank2-alt.json`:
  source-side publication artifacts for the stale-source recovery proof.
- `nscale-crossnode-stale-source-recovery-summary.json`: compact summary of
  the stale-source recovery target/source artifacts and key metrics.
- `nscale-runtime-read-failure-recovery-pytest.log`: nscale Python gate for
  target-side runtime read-failure recovery logic (`40 passed, 1 skipped`). It
  simulates a READY primary source failing during its read group, verifies the
  target replans only that failed source range from an alternate holder, and
  preserves the existing stale-source/control-plane tests. This is code-path
  evidence only; it is not a hard GPU in-flight pod-kill proof.
- `nscale-hard-kill-harness-support-smoke.json` and
  `nscale-hard-kill-harness-pytest.log`: nscale CPU evidence that the
  cross-node harness can widen synthetic payload columns and emit a NIXL
  post-submit marker/sleep hook for hard source-pod kill orchestration
  (`4 passed`). This records harness readiness; the later GPU hard-kill proof is
  banked separately below.
- `nscale-hard-kill-gpu-attempt-json-serialization-block.json` and
  `nscale-hard-kill-gpu-attempt-json-serialization-block.log`: honest GPU
  attempt artifact for `hardkillgpu-20260604-185555`. Four GPU pods scheduled
  across the intended source/target nodes and the target reached NIXL setup,
  but the run stopped before the kill marker because the post-submit probe
  tried to JSON-serialize byte-valued remote agent metadata. This is a
  software-block artifact, not a hard-kill proof.
- `nscale-hard-kill-gpu-inflight-recovery-summary.json`: compact proof summary
  for `hardkillgpu2-20260604-190228`. Three independent source-rank GPU pods ran
  on `cluster-0967a26d-pool-14bee067-prctr-9c2x7`, one target GPU pod ran on
  `cluster-0967a26d-pool-14bee067-prctr-g2j7h`, the target emitted a
  post-submit marker for the 256 MiB rank0 NIXL READ, rank0 was force-deleted,
  NIXL reported `NIXL_ERR_REMOTE_DISCONNECT`, recovery read from
  `trainer-rank2-alt`, and the final 1 GiB target validated allclose/checksum.
  The summary records `hard_pod_kill_inflight_nixl_read_proven=true`,
  `read_failure_recovery_used=true`, and `replanned_only_failed_segments=true`.
- `nscale-hard-kill-gpu-inflight-recovery-target.json` and
  `nscale-hard-kill-gpu-inflight-recovery-target.log`: raw target artifact/log
  for the hard-kill run. The raw target JSON was emitted before the proof-field
  fix, so its high-level `failed_then_succeeded` and
  `replanned_only_failed_segments` booleans are stale; use its
  `read_failure_recovery_used=true`, read-failure/recovery segment records, and
  the compact summary above for the hard-kill claim.
- `nscale-hard-kill-gpu-inflight-recovery-source-rank1.json` and
  `nscale-hard-kill-gpu-inflight-recovery-source-rank2-alt.json`: surviving
  source-side publication artifacts from the hard-kill run. There is no rank0
  source artifact because that pod was deliberately force-deleted after the
  target's post-submit marker.
- `nscale-hard-kill-gpu-inflight-recovery-summary.log`: orchestration summary
  recording the target marker, rank0 force-delete, target completion, and key
  checksum/timing fields.
- `nscale-hard-kill-inflight-recovery-final-verify.log`: final nscale focused
  verification after banking the hard-kill proof and fixing future target proof
  fields. It records Black checks, focused pytest (`47 passed`), JSON artifact
  validation, and `git diff --check`.
- `nscale-cursor-code-review-availability-hardkill.log`: nscale availability
  check for `cursor-code-review`. The command was not found in the pod PATH or
  searched nscale directories, so review was not run rather than using the local
  laptop.
- `nscale-crossnode-control-plane-pytest.log`: focused Python control-plane
  pytest run inside the nscale target pod after the cross-node patch
  (`8 passed`). It covers the MX refit endpoint helper path, including legacy
  sidecar recovery when a live server drops `slice_ownerships`.
- `nscale-cpu-crossnode-final-verify.log`: final nscale CPU verification for
  this change. It runs syntax checks on the changed Python modules, validates
  the cross-node and capacity-block JSON artifacts, and reruns the focused
  control-plane pytest (`8 passed`).
- `nscale-one-pod-per-source-capacity-block.log` and
  `nscale-one-pod-per-source-capacity-block.json`: earlier honest capacity
  block for the stricter one-pod-per-source-rank proof. Three independent
  source-rank GPU pods remained Pending with `0/29 nodes`,
  `10 Insufficient nvidia.com/gpu`, and autoscaler max node group size reached.
  This block is superseded by the later checksum-backed
  `nscale-crossnode-one-pod-per-source-rank-target.json` run.
- `nscale-level5-timing-capacity-block.json`: honest block for a dedicated
  normalized Level-5 timing table. Comparable checksum-backed MX/NIXL, NCCL
  Reshard, and CheckpointEngine timings were not run in that pass because the
  required multi-GPU nscale capacity was unavailable.
- `nscale-level5-normalizer-pytest-20260605.log`: focused nscale Python gate
  for the checksum-gated Level-5 timing normalizer and existing refit
  control-plane paths (`19 passed`).
- `nscale-level5-same-node-synthetic-table-missing-baselines.json`: historical
  normalized synthetic same-node table where NCCL Reshard and CheckpointEngine
  rows were explicitly unmeasured; superseded by the passing 2026-06-05 table.
- `nscale-level5-nccl-reshard-baseline-20260605.json` and `.log`: real
  measured 4-GPU B200 same-node NCCL Reshard-style baseline row. It is
  checksum/allclose gated, records `trainer_full_all_gather_used=true`,
  `trainer_to_inference_bytes=128`, `trainer_collective_bytes=160`,
  `redundant_cross_boundary_factor=2.0`, and passes.
- `nscale-level5-checkpoint-engine-baseline-20260605.json` and `.log`: real
  measured 4-GPU B200 same-node CheckpointEngine-style baseline row. It is
  checksum/allclose gated, records `checkpoint_storage_used=true`,
  `checkpoint_storage_bytes=256`, `trainer_collective_bytes=160`,
  `redundant_cross_boundary_factor=4.0`, and passes.
- `nscale-level5-same-node-synthetic-table-20260605.json` and `.log`: passing
  normalized Level-5 synthetic same-node table with measured MX/NIXL, NCCL
  Reshard-style, and CheckpointEngine-style rows in the same placement scope.
  It records `level5_synthetic_smoke_pass=true`,
  `production_competitive_claim_safe=false`, and
  `level5_full_model_claim_safe=false`.
- `nscale-level5-baseline-summary-20260605.json` and
  `nscale-level5-baseline-nvidia-smi-20260605.log`: pod summary and B200 GPU
  inventory for `mx-level5-baseline-20260605-1228`; all three return codes were
  zero.
- `nscale-level5-baseline-capacity-block.json`, `.log`,
  `nscale-level5-baseline-capacity-block-20260605.json`, and
  `nscale-level5-baseline-capacity-block-20260605.log`: historical 4-GPU
  scheduler blocks retained for provenance; superseded for synthetic same-node
  scope by the passing 2026-06-05 run.
- `nscale-level5-existing-job-evidence-audit.json`: audit of existing nscale
  timing jobs that looked close to Level-5 evidence. It banks real Qwen3 BF16
  full-model/vLLM timing context for MX live refit, NCCL, and
  CheckpointEngine, but marks the Level-5 claim as failed because those rows
  use live output-change validation instead of allclose/checksum and do not
  share a normalized registration/publish/planner/read/install timing schema.
- `qwen3-30b-a3b-moe-manifest.json.gz` and
  `qwen3-30b-a3b-moe-manifest.summary.json`: real Qwen/Qwen3-30B-A3B
  safetensors-header coverage artifact generated through HTTP range reads over
  all 16 checkpoint shards. It covers 18,867 tensors: 18,432 MoE expert tensors
  and 435 layout-sensitive tensors. No tensor payloads were downloaded.
- `qwen3-30b-a3b-fp8-moe-manifest.json.gz` and
  `qwen3-30b-a3b-fp8-moe-manifest.summary.json`: real
  Qwen/Qwen3-30B-A3B-FP8 safetensors-header coverage artifact generated through
  HTTP range reads over all 7 checkpoint shards. It covers 37,491 tensors,
  including 18,624 real `global-required` quantization metadata tensors.
- `qwen3-30b-a3b-fp8-zero-copy-fallback-smoke.json`: real FP8 manifest smoke
  showing a `global-required` `weight_scale_inv` entry raises
  `QuantizationMetadataError` and no zero-copy segment plan is created.
- `qwen3-30b-a3b-fp8-runtime-fallback-install-smoke.json`: receiver-side
  runtime fallback install smoke for a real Qwen3 FP8 global-required
  `weight_scale_inv` manifest entry. It installs a materialized synthetic
  payload into a runtime-owned vLLM-shaped target tensor, validates
  allclose/checksum, records `zero_copy_plan_created=false`, and does not
  claim real Qwen payload transfer.
- `nscale-qwen-fp8-runtime-fallback-install-pytest.log`: focused nscale
  Python gate for receiver fallback install and Qwen manifest handling
  (`17 passed`).
- `nscale-runtime-refit-versioned-rollback-smoke.json`: CPU/runtime-tensor
  rollback smoke for two Qwen-style layer tensors. It snapshots `step-7`,
  installs `step-8`, rolls back to `step-7`, then installs and commits
  `step-9`, with checksum validation. It records `gpu_resident=false`,
  rejects dtype drift before rollback, and is not GPU-resident rollback proof.
- `nscale-runtime-refit-versioned-rollback-pytest.log`: focused nscale
  receiver rollback gate (`11 passed`).
- `nscale-vllm-receiver-smoke-pytest.log`: focused nscale Python gate for the
  current-branch vLLM receiver-owned tensor smoke helper (`4 passed`). It covers
  receiver request creation from a module-owned tensor, two-source planning,
  install into that tensor, checksum/allclose validation, and original tensor
  restore. This is not a live GPU vLLM artifact.
- `nscale-vllm-receiver-smoke-capacity-block.json` and
  `nscale-vllm-receiver-smoke-capacity-block.log`: earlier honest block for the
  first 1-GPU live vLLM receiver smoke probe. It is superseded by the later
  checksum-backed `nscale-live-vllm-receiver-applymodel-smoke-20260604.json`
  run below.
- `nscale-vllm-receiver-smoke-final-verify.log`: final focused nscale gate for
  the first vLLM receiver-smoke change. It records Black checks, focused pytest
  (`52 passed`), JSON validation, and `git diff --check`.
- `nscale-live-vllm-receiver-smoke-capacity-block-20260604.json` and
  `nscale-live-vllm-receiver-smoke-capacity-block-20260604.log`: current-branch
  repro of the original live vLLM receiver smoke pod spec. The submitted pod did
  not schedule and produced no checksum/allclose artifact. Because later GPU
  probes used the `nvidia.com/gpu` toleration and did schedule, this artifact is
  scoped to that submitted pod spec, not a cluster-wide 1-GPU capacity claim.
- `nscale-live-vllm-receiver-v1-module-discovery-block-20260604.log`: scheduled
  1-GPU vLLM 0.17.1 attempt showing the original parent-process module traversal
  failed after the V1 engine loaded because the model lives in `EngineCore_DP0`.
- `nscale-live-vllm-receiver-v0-env-unsupported-block-20260604.log`: scheduled
  1-GPU retry showing `VLLM_USE_V1` is not a supported override in this vLLM
  build, so forcing V0 is not the right path.
- `nscale-live-vllm-receiver-applymodel-smoke-20260604.json`, `.log`, and
  `-placement.log`: completed 1-GPU live vLLM V1 receiver-owned tensor smoke.
  The CLI creates a tiny Qwen2 checkpoint, starts a real vLLM 0.17.1 V1 engine,
  uses `LLM.apply_model` to run inside the worker-owned `Qwen2ForCausalLM`,
  builds a vLLM `SliceRequest` from `lm_head.weight` on `cuda:0`, plans two
  synthetic trainer-held ranges, installs into that vLLM-owned tensor, validates
  checksum/allclose, and restores the original tensor. The artifact records
  `vllm_apply_model_used=true`, `vllm_worker_owned_target_tensor=true`,
  `real_runtime_engine_used=true`, `actual_nixl_reads_used=false`, and
  `synthetic_trainer_payloads_used=true`.
- `nscale-live-vllm-nixl-runtime-refit-smoke-20260604.json` and `.log`:
  completed same-node, one-pod, 3-rank vLLM+NIXL runtime bridge proof on
  nscale. Source ranks own CUDA trainer-like shard tensors, the target rank
  starts a live vLLM 0.17.1 V1 `LLM`, builds the receiver request from the
  worker-owned `lm_head.weight`, reads two UCX/NIXL source segments into a CUDA
  staging tensor, installs/restores the assembled tensor through
  `LLM.apply_model`, validates staging allclose, runtime allclose, checksum
  match, and restores the original weight. The artifact records
  `actual_nixl_reads_used=true`, `real_runtime_engine_used=true`,
  `source_rank_owned_trainer_tensors_used=true`, 4,096 trainer-to-inference
  bytes, about 0.94 ms raw NIXL read duration, about 0.18 ms in-worker
  activation install duration, and about 84.3 ms end-to-end `apply_model`
  install duration. Scope boundary: this committed GPU artifact is
  same-node/one-pod evidence with GPU reuse and the earlier deterministic
  source values; NIXL lands in staging and is copied through `apply_model`, so
  it is not cross-node, not direct NIXL landing into vLLM-owned storage, and not
  a live trainer optimizer loop.
- `nscale-trainer-step-runtime-source-smoke-20260605.json` and
  `nscale-trainer-step-runtime-source-pytest-20260605.log`: nscale CPU evidence
  for the new optimizer-step source publisher used by the vLLM/SGLang NIXL
  runtime bridge source helpers. It runs a source-rank `torch.optim.SGD` step
  over a synthetic objective, materializes only the source-owned range, proves
  that range reconstructs the post-step full target slice, and records
  `optimizer_step_publisher_used=true` plus
  `static_replacement_formula_used=false` (`13 passed`). This is code-path
  evidence only; it is not a live GPU runtime rerun.
- `nscale-trainer-step-source-publication-smoke-20260605.json` and
  `nscale-trainer-step-source-publication-pytest-20260605.log`: nscale CPU
  evidence for `TrainerStepSourcePublication`. The publication object carries
  the post-step source tensor, annotated `SliceOwnership`, source lease,
  NIXL descriptor identity, and source-update provenance together, and the
  vLLM/SGLang source-rank paths now emit `source_publication` metadata (`15
  passed`). This is code-path evidence only; it is not a live GPU runtime rerun.
- `nscale-trainer-loop-source-publication-smoke-20260605.json` and
  `nscale-trainer-loop-source-publication-pytest-20260605.log`: nscale
  CPU/control-plane evidence for versioned trainer-loop source publication.
  Two source-rank ownerships publish coherent step-specific model versions,
  leases, descriptors, and provenance; the smoke reconstructs the step-2 target
  from source-owned ranges, proves step-2 differs from step-1, and the pytest
  gate verifies MX metadata round-trip/planning for the loop-owned slices
  (`29 passed`). This is still a deterministic trainer-loop smoke over a
  synthetic `torch.optim.SGD` objective, not a real distributed RL/FSDP trainer
  loop.
- `nscale-runtime-bridges-trainer-loop-publication-smoke-20260605.json`,
  `nscale-runtime-bridges-trainer-loop-publication-pytest-20260605.log`, and
  `nscale-cursor-code-review-runtime-bridges-trainer-loop-publication-20260605.log`:
  nscale CPU/control-plane evidence that the vLLM and SGLang MX runtime bridge
  source helpers now materialize source publications through the trainer-loop path.
  The smoke records `runtime_bridge_count=2`, `source_publication_count=4`,
  `segment_count=4`, `trainer_to_inference_bytes=192`, allclose/checksum pass,
  step-specific source/request model-version compatibility, and MX endpoint
  planner acceptance. This is not a live GPU runtime rerun and not real
  distributed RL/FSDP trainer ownership.
- `nscale-trainer-step-mx-publication-smoke-20260605.json` and
  `nscale-trainer-step-mx-publication-pytest-20260605.log`: nscale CPU/control
  plane evidence that trainer-step source publications can be published through
  the MX metadata client path, listed back as `SliceOwnership`s, and used for
  receiver-side `SegmentPlan`s.
- `nscale-live-vllm-nixl-runtime-trainer-step-20260605.json` and
  `nscale-live-vllm-nixl-runtime-trainer-step-summary-20260605.json`:
  completed same-node, one-pod, 3-rank vLLM+NIXL runtime bridge rerun using
  the optimizer-step source publisher. Two source ranks publish post-step
  `torch.optim.SGD` CUDA shards plus source-publication provenance, the target
  starts live vLLM 0.17.1, reads two UCX/NIXL segments into CUDA staging,
  installs/restores through `LLM.apply_model`, and validates staging allclose,
  runtime allclose, and checksum. The artifact records
  `trainer_optimizer_step_publisher_used=true`,
  `synthetic_source_values_used=false`, `trainer_to_inference_bytes=4096`,
  `raw_nixl_read_duration_ms=14.29297006689012`, and `run_rc=0` in the
  summary. Scope boundary: this is same-node/one-pod evidence with GPU reuse
  and a synthetic optimizer objective; it is not cross-node runtime placement,
  not one-pod-per-source-rank runtime placement, not direct NIXL landing into
  vLLM-owned storage, and not a real RL trainer loop.
- `nscale-live-vllm-mx-runtime-crossnode-20260605.json`: completed cross-node,
  one-pod-per-source-rank vLLM+NIXL runtime bridge using MX endpoint discovery.
  Source pods `mx-vllm-mx-src0-20260605` and `mx-vllm-mx-src1-20260605` ran on
  `cluster-0967a26d-pool-14bee067-prctr-g2j7h`; target pod
  `mx-vllm-mx-tgt-20260605` ran on
  `cluster-0967a26d-pool-14bee067-prctr-9c2x7`. The target loaded live vLLM
  0.17.1, built the receiver request from worker-owned `lm_head.weight`,
  discovered two READY source endpoints from `mx-server-rl:8001`, added two
  distinct NIXL source agents, performed two cross-node UCX/NIXL READs into
  CUDA staging, installed/restored through `LLM.apply_model`, and validated
  staging allclose, runtime allclose, and checksum. It records
  `cross_node=true`, `one_pod_per_source_rank=true`,
  `trainer_to_inference_bytes=4096`, `raw_nixl_read_duration_ms=9.514622972346842`,
  `metadata_query_duration_ms=31.619531917385757`, and
  `planner_duration_ms=0.18348696175962687`. Scope boundary: vLLM only, tiny
  single tensor, staging-copy install, synthetic optimizer objective, no direct
  NIXL landing into vLLM-owned storage, and no real RL trainer loop.
- `nscale-live-vllm-mx-runtime-crossnode-source0-20260605.json` and
  `nscale-live-vllm-mx-runtime-crossnode-source1-20260605.json`: source-side
  publication artifacts for the two independent source-rank pods in the vLLM
  cross-node runtime proof.
- `nscale-live-vllm-mx-runtime-crossnode-pods-20260605.log`,
  `nscale-live-vllm-mx-runtime-crossnode-pod-describe-20260605.log`,
  `nscale-live-vllm-mx-runtime-crossnode-source-nvidia-smi-ib-20260605.log`, and
  `nscale-live-vllm-mx-runtime-crossnode-target-nvidia-smi-ib-20260605.log`:
  placement and GPU/IB evidence for the successful vLLM cross-node runtime run.
- `nscale-live-vllm-mx-runtime-trainer-loop-crossnode-20260605.json`: completed
  cross-node, one-pod-per-source-rank vLLM+NIXL runtime rerun using MX endpoint
  discovery plus versioned trainer-loop step-2 source publication. Source pods
  `mx-vllm-loop-src0-20260605` and `mx-vllm-loop-src1-20260605` ran on
  `cluster-0967a26d-pool-14bee067-prctr-g2j7h`; target pod
  `mx-vllm-loop-tgt-20260605` ran on
  `cluster-0967a26d-pool-14bee067-prctr-9c2x7`. The target loaded live vLLM
  0.17.1, discovered both READY source endpoints from `mx-server-rl:8001`, read
  two UCX/NIXL segments into CUDA staging, inferred the expected optimizer step
  from source ownership metadata, installed/restored through `LLM.apply_model`,
  and validated staging allclose, runtime allclose, and checksum. It records
  `cross_node=true`, `one_pod_per_source_rank=true`,
  `trainer_loop_source_publication_used=true`,
  `receiver_expected_update_from_source_metadata=true`,
  `expected_optimizer_step_count=2`, `trainer_to_inference_bytes=4096`,
  `raw_nixl_read_duration_ms=11.092181084677577`,
  `metadata_query_duration_ms=28.369619976729155`,
  `planner_duration_ms=0.1897109905257821`, and
  `activation_install_duration_ms=0.15898107085376978`. Scope boundary: vLLM
  only, tiny single tensor, staging-copy install, deterministic trainer-loop
  smoke, no direct NIXL landing into vLLM-owned storage, and no real RL trainer
  loop.
- `nscale-live-vllm-mx-runtime-trainer-loop-crossnode-source0-20260605.json`
  and `nscale-live-vllm-mx-runtime-trainer-loop-crossnode-source1-20260605.json`:
  source-side trainer-loop publication artifacts for the vLLM cross-node
  trainer-loop runtime proof.
- `nscale-live-vllm-mx-runtime-trainer-loop-crossnode-20260605.log`,
  `nscale-live-vllm-mx-runtime-trainer-loop-crossnode-source0-20260605.log`,
  `nscale-live-vllm-mx-runtime-trainer-loop-crossnode-source1-20260605.log`,
  and `nscale-live-vllm-mx-runtime-trainer-loop-crossnode-pods-20260605.log`:
  target/source logs and placement evidence for the successful vLLM
  trainer-loop cross-node runtime rerun.
- `nscale-cursor-code-review-trainer-loop-crossnode-vllm-20260605.log`:
  Cursor code-review rubric artifact for this rerun. It records the hard-coded
  optimizer-default finding, the fix to reuse the shared trainer-step constants,
  and the post-fix focused pytest gate (`42 passed`).
- `nscale-live-vllm-mx-runtime-crossnode-bw4bt-startup-block-20260605.json` and
  `.log`: honest first-placement block. Source pods on
  `cluster-0967a26d-pool-14bee067-prctr-bw4bt` saw active `mlx5_10` IB, but the
  source Python process and `nvidia-smi` entered uninterruptible sleep before
  source-ready artifacts were written. This is not a proof and is superseded by
  the successful `g2j7h` to `9c2x7` artifact above.
- `nscale-live-sglang-mx-runtime-crossnode-20260605.json`: completed
  cross-node, one-pod-per-source-rank SGLang+NIXL runtime bridge using MX
  endpoint discovery. Source pods `mx-sglang-mx-src0-20260605` and
  `mx-sglang-mx-src1-20260605` ran on
  `cluster-0967a26d-pool-14bee067-prctr-g2j7h`; target pod
  `mx-sglang-mx-tgt-20260605` ran on
  `cluster-0967a26d-pool-14bee067-prctr-9c2x7`. The target loaded live SGLang
  `0.0.0.dev1+g229cadec0`, built the receiver request from `lm_head.weight`,
  discovered two READY source endpoints from `mx-server-rl:8001`, added two
  distinct NIXL source agents, performed two cross-node UCX/NIXL READs into
  CUDA staging, installed/restored through `Engine.update_weights_from_tensor`,
  and validated staging allclose, runtime allclose, and checksum. It records
  `cross_node=true`, `one_pod_per_source_rank=true`,
  `runtime_storage_dtype=bfloat16`, `trainer_to_inference_bytes=16384`,
  `raw_nixl_read_duration_ms=12.166149914264679`,
  `metadata_query_duration_ms=44.608733034692705`, and
  `planner_duration_ms=0.14281098265200853`. Scope boundary: SGLang only, tiny
  single tensor, staging-copy install, synthetic optimizer objective, no direct
  NIXL landing into SGLang-owned storage, and no real RL trainer loop.
- `nscale-live-sglang-mx-runtime-crossnode-source0-20260605.json` and
  `nscale-live-sglang-mx-runtime-crossnode-source1-20260605.json`: source-side
  publication artifacts for the two independent source-rank pods in the SGLang
  cross-node runtime proof.
- `nscale-live-sglang-mx-runtime-crossnode-pods-20260605.log`,
  `nscale-live-sglang-mx-runtime-crossnode-pod-describe-20260605.log`,
  `nscale-live-sglang-mx-runtime-crossnode-source-nvidia-smi-ib-20260605.log`,
  and
  `nscale-live-sglang-mx-runtime-crossnode-target-nvidia-smi-ib-20260605.log`:
  placement and GPU/IB evidence for the successful SGLang cross-node runtime
  run.
- `nscale-cursor-code-review-sglang-mx-crossnode-runtime-20260605.log`:
  cursor-code-review artifact for the SGLang cross-node runtime change. It
  records the fixed vLLM/SGLang helper-boundary finding, the extraction of
  `refit_mx_runtime_common.py`, and the focused post-fix pytest gate.
- `nscale-live-sglang-nixl-runtime-trainer-step-20260605.json` and
  `nscale-live-sglang-nixl-runtime-trainer-step-summary-20260605.json`:
  completed same-node, one-pod, 3-rank SGLang+NIXL runtime bridge rerun using
  the optimizer-step source publisher. Two source ranks publish post-step
  `torch.optim.SGD` CUDA shards plus source-publication provenance, the target
  starts live SGLang `0.0.0.dev1+g229cadec0`, reads two UCX/NIXL segments into
  CUDA staging with `UCX_NET_DEVICES=mlx5_10:1`, installs/restores through
  `Engine.update_weights_from_tensor`, and validates staging allclose, runtime
  allclose, and checksum. The artifact records
  `trainer_optimizer_step_publisher_used=true`,
  `synthetic_source_values_used=false`, `runtime_storage_dtype=bfloat16`,
  `trainer_to_inference_bytes=16384`,
  `raw_nixl_read_duration_ms=5.667333956807852`, and `run_rc=0` in the
  summary. The BF16 runtime readback is modeled explicitly; the full-precision
  staging gate still passes before runtime install. Scope boundary: this is
  same-node/one-pod evidence with GPU reuse and a synthetic optimizer
  objective; it is not cross-node runtime placement, not one-pod-per-source-rank
  runtime placement, not direct NIXL landing into SGLang-owned storage, and not
  a real RL trainer loop.
- `nscale-live-mx-trainer-step-publication-sidecar-pass-20260605.json` and
  `.log`: live `mx-server-rl:8001` pass. The deployed server accepted
  `PublishMetadata`, returned `metadata_endpoint` sidecars while still dropping
  the new repeated `slice_ownerships` field, and the client discovered and
  planned trainer-step ownership from those returned sidecars. The banked pytest
  log records `12 passed` for in-memory control-plane coverage plus `2 passed`
  for live MX.
- `nscale-cursor-code-review-sidecar-20260605.log`: nscale review artifact
  applying the `cursor-code-review` rubric from the dotfiles-provenance skill;
  it records the single-ownership/endpoint-preservation finding and the list
  sidecar fix before final verification.
- `nscale-live-mx-trainer-step-publication-server-drop-block-20260605.json` and
  `.log`: historical pre-bridge block retained for provenance; superseded by
  the sidecar-pass artifact above.
- `nscale-live-vllm-nixl-runtime-trainer-step-capacity-block-20260605.json` and
  `.log`: historical honest block for the first attempted live vLLM+NIXL
  runtime GPU rerun using the optimizer-step source publisher patch. It is
  superseded for same-node vLLM trainer-step scope by the later checksum-backed
  `nscale-live-vllm-nixl-runtime-trainer-step-20260605.json` run above.
- `nscale-1gpu-capacity-probe-block-20260605.json` and `.log`: fresh 1-GPU
  nscale capacity probe for the next live runtime rerun. It hit the same
  scheduler/autoscaler block, so no GPU runtime rerun was attempted in that
  pass.
- `nscale-sglang-receiver-smoke.json` and
  `nscale-sglang-receiver-smoke.log`: SGLang-shaped module-owned receiver smoke
  using `modelexpress.refit_sglang_receiver_smoke`. It builds an SGLang
  `SliceRequest`, plans two synthetic trainer-held ranges, installs into the
  runtime-owned tensor, validates checksum/allclose, restores the original
  tensor, and records `real_runtime_engine_used=false`.
- `nscale-sglang-runtime-import-probe.json` and
  `nscale-sglang-runtime-import-probe.log`: import probe from the long-running
  nscale SGLang runtime builder pod. It verifies `torch`, `sglang`, and
  `sglang.srt` import in that image, but does not request a GPU or prove engine
  refit.
- `nscale-sglang-gpu-import-smoke.json` and
  `nscale-sglang-gpu-import-smoke.log`: one-GPU nscale SGLang runtime import
  smoke using the Docker-published SGLang image. The pod scheduled on a GPU
  node, imported SGLang, and saw CUDA with one device. This is runtime
  availability evidence, superseded for live refit by the Engine smoke below.
- `nscale-live-sglang-engine-receiver-smoke-20260604.json`, `.log`, and
  `-placement.log`: completed 1-GPU live SGLang Engine-owned weight smoke.
  The CLI creates a tiny Llama checkpoint, starts SGLang
  `0.0.0.dev1+g229cadec0`, uses `Engine.get_weights_by_name` to fetch
  `lm_head.weight`, builds a receiver-side request, plans two synthetic
  trainer-held ranges, assembles the replacement from segment payloads,
  installs it through `Engine.update_weights_from_tensor`, validates
  checksum/allclose through SGLang, and restores the original weight. The
  artifact records `real_runtime_engine_used=true`,
  `sglang_engine_update_weights_from_tensor_used=true`,
  `actual_nixl_reads_used=false`, and
  `synthetic_trainer_payloads_used=true`.
- `nscale-live-sglang-nixl-runtime-refit-smoke-20260604.json` and `.log`:
  completed same-node, one-pod, 3-rank SGLang+NIXL runtime bridge proof on
  nscale. Source ranks own CUDA trainer-like shard tensors, the target rank
  starts a live SGLang `Engine`, builds the receiver request from
  `lm_head.weight`, reads two UCX/NIXL source segments into a CUDA staging
  tensor, installs the assembled tensor through
  `Engine.update_weights_from_tensor`, validates staging allclose, runtime
  allclose, checksum match, and restores the original weight. The artifact
  records `actual_nixl_reads_used=true`, `real_runtime_engine_used=true`,
  `source_rank_owned_trainer_tensors_used=true`, 16,384 trainer-to-inference
  bytes, about 1.24 ms raw NIXL read duration, and about 28.0 ms activation
  install duration. Scope boundary: this is same-node/one-pod evidence with
  GPU reuse; it is not cross-node, not direct NIXL landing into SGLang-owned
  storage, and not a live trainer optimizer loop. The paired block logs
  `nscale-live-sglang-nixl-runtime-refit-smoke-20260604-outer-dist-before-sglang-block.log`
  and `nscale-live-sglang-nixl-runtime-refit-smoke-20260604-old-runner-copy-block.log`
  capture earlier SGLang startup-order hangs before the target-first runtime
  startup fix.
- `nscale-python-full-pytest-sglang-nixl-runtime-20260604.log`: full nscale
  Python verification after adding the SGLang+NIXL runtime bridge (`311 passed,
  19 skipped`).
- `nscale-python-full-pytest-vllm-nixl-runtime-20260604.log`: full nscale
  Python verification after adding the vLLM+NIXL runtime bridge (`316 passed,
  19 skipped`).
- `nscale-cursor-code-review-availability-vllm-smoke.log`: nscale availability
  check for `cursor-code-review`. The command was not found in the pod PATH or
  searched nscale directories, so review was not run rather than using the local
  laptop.
- Qwen MoE manifest extraction is covered by the Python gate. It classifies
  stacked expert-axis tensors, per-expert tensor-name layouts, global
  quantization metadata, generated-on-target tensors, and layout-sensitive
  shared-expert tensors.

The NIXL JSON is the primary Level 2 same-node proof artifact. The cross-node
live-MX NIXL endpoint JSONs are the strongest current Level 3 synthetic proof
artifacts, with the one-pod-per-source-rank, stale-source recovery, and
hard-kill in-flight recovery target/summary JSONs as the strictest cross-node
claims. Level 4 now has same-node SGLang and vLLM NIXL-to-runtime bridge
artifacts, checksum-backed same-node optimizer-step reruns for both runtimes,
and checksum-backed cross-node/one-pod-per-source-rank vLLM and SGLang
MX-endpoint runtime bridges. These remain tiny/staging-copy runtime proofs with
synthetic optimizer objectives; direct runtime-buffer NIXL landing, full-model
refit, and real RL trainer-loop claims remain unproven. The NCCL
distributed JSON remains the Level 1 comparison artifact. Existing Qwen3
timing jobs are now banked as partial competitive context. The Level-5
synthetic same-node table now has comparable checksum-backed MX/NIXL, NCCL
Reshard-style, and CheckpointEngine-style rows in one placement scope and passes
for `synthetic-same-node-smoke`; full-model, cross-node, and production
competitive Level-5 claims remain unproven.
