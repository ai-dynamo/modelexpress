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
  post-submit marker/sleep hook for a later hard source-pod kill proof
  (`4 passed`). This is harness-readiness evidence only; it records
  `hard_pod_kill_inflight_proven=false`.
- `nscale-hard-kill-gpu-attempt-json-serialization-block.json` and
  `nscale-hard-kill-gpu-attempt-json-serialization-block.log`: honest GPU
  attempt artifact for `hardkillgpu-20260604-185555`. Four GPU pods scheduled
  across the intended source/target nodes and the target reached NIXL setup,
  but the run stopped before the kill marker because the post-submit probe
  tried to JSON-serialize byte-valued remote agent metadata. This is a
  software-block artifact, not a hard-kill proof.
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
- `nscale-level5-normalizer-pytest.log`: focused nscale Python gate for the
  checksum-gated Level-5 timing normalizer and existing refit control-plane
  paths (`18 passed`).
- `nscale-level5-same-node-synthetic-table-missing-baselines.json`: normalized
  synthetic same-node Level-5 table generated from the existing same-node
  MX/NIXL checksum-backed artifact. The MX row passes, the NCCL Reshard and
  CheckpointEngine rows are explicitly unmeasured, and the table result is
  correctly `fail`.
- `nscale-level5-baseline-capacity-block.json` and
  `nscale-level5-baseline-capacity-block.log`: honest block for the new
  synthetic same-node NCCL Reshard and CheckpointEngine baseline runners. The
  4-GPU pod stayed Pending with `0/29 nodes`,
  `10 Insufficient nvidia.com/gpu`, `19` untolerated taints, and autoscaler
  max node group size reached.
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
- Qwen MoE manifest extraction is covered by the Python gate. It classifies
  stacked expert-axis tensors, per-expert tensor-name layouts, global
  quantization metadata, generated-on-target tensors, and layout-sensitive
  shared-expert tensors.

The NIXL JSON is the primary Level 2 same-node proof artifact. The cross-node
live-MX NIXL endpoint JSONs are the strongest current Level 3 synthetic proof
artifacts, with the one-pod-per-source-rank and stale-source recovery target JSONs as
the strictest cross-node claims. The NCCL distributed JSON remains the Level 1 comparison
artifact. Existing Qwen3 timing jobs are now banked as partial competitive
context. The Level-5 normalizer/baseline harness now exists, but real Level-5
timing remains unproven until comparable checksum-backed MX/NIXL, NCCL
Reshard, and CheckpointEngine rows are completed in the same placement scope.
