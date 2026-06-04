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
- `nscale-live-mx-nixl-capacity.log`: attempted live-MX NIXL GPU verification.
  The code path is implemented, but nscale could not schedule 4-, 2-, or 1-GPU
  pods during this run (`Insufficient nvidia.com/gpu`; autoscaler max size
  reached).
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
- Qwen MoE manifest extraction is covered by the Python gate. It classifies
  stacked expert-axis tensors, per-expert tensor-name layouts, global
  quantization metadata, generated-on-target tensors, and layout-sensitive
  shared-expert tensors.

The NIXL JSON is the primary Level 2 proof artifact. The NCCL distributed JSON
remains the Level 1 comparison artifact.
