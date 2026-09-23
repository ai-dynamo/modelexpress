<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Matched MILES Weight-Update Benchmark

This harness runs paired native MILES broadcast and ModelExpress NCCL M2N
workloads through the real Kubernetes launcher. It does not simulate either
transport and it does not modify MILES or ModelExpress production code.

## Metrics

The PR #3304 headline is the exact MILES
`update_weights_implementation` timer:

```text
pre-weights_getter after begin barrier
  -> source refresh/materialization
  -> routed M2N
  -> residual broadcast
  -> destination install
  -> transfer-side trainer barrier
```

The broader outer `update_weights` timer is retained separately as
`e2e_update_s`. It includes pause/begin and finalize/resume work and must not
replace the headline metric.

Each arm runs `--num-rollout 20`, which produces 21 update calls:

| Raw index | Treatment |
| ---: | --- |
| 0 | cold, retained and excluded from steady statistics |
| 1-19 | steady PR comparison window |
| 12 | preserved PR step-12 reference |
| 20 | final tail, retained and excluded from the PR window |

Paired blocks use deterministic seeded arm order. Summaries include mean,
median, p10, p90, p95, MAD, paired speedups, step-12 speedup, the broader
rollout-ready distribution, byte coverage, and residual fraction.
Acceptance requires all 19 exact steady indices and matching weight-version
receipts in every paired block.

## Run

Use `config.json.example` as the template for a persistent configuration,
replace every placeholder, and execute:

```bash
python3 benchmarks/miles_weight_update/benchmark.py /persistent/config.json
```

`output_root` must not be under `/tmp`, `/var/tmp`, `/dev/shm`, or `/run`.
Runtime and server images must be immutable digest references. The benchmark
deletes each Job after collecting receipts and deletes the ModelExpress
Deployment and Service after each external arm. External server resources are
uniquely named per run/block/repetition so cleanup cannot delete an unrelated
server. Cleanup records each applied object UID and ownership labels, verifies
both immediately before deletion, and sends the UID as a Kubernetes delete
precondition. Before apply, resource absence is accepted only from a successful
`kubectl get --ignore-not-found` with an empty response. RBAC, API, transport,
and timeout errors abort without applying anything.

`storage.model_claim` and `storage.workspace_claim` must name existing, bound
`ReadWriteOnce` filesystem claims in the benchmark namespace. The harness
rewrites `/models` and `/root/shared_data` to those claims and fails before
apply if either path remains `emptyDir`, `hostPath`, or an inline ephemeral
volume. The claims may be the same; this is the shortest AWS first-run path
when a single node-pinned workspace already contains enough capacity. The
workspace is also mounted at `/workspace`, allowing the pinned DAPO path in
`config.json.example` to resolve exactly as recorded.

The rendered Job mounts a run-scoped ConfigMap containing the benchmark-only
runtime driver. The host rejects a render unless the embedded driver and
bootstrap match the current frozen source byte-for-byte. Before MILES starts,
the driver verifies the exact DAPO Git revision and file SHA-256. It also
canonicalizes the persistent checkpoint's full-file SHA manifest, matches its
configured digest, validates every referenced path, streams every checkpoint
file through SHA-256 with bounded memory, and requires every digest to match
exactly. It then matches the repository and revision receipts. All workload
integrity checks finish before training starts and are outside the benchmark
timers. The driver passes DAPO directly to `--prompt-data`; there is no implicit
dataset download or GSM8K fallback.

The existing example launcher only allowlists its two Qwen demo pairs. The
harness uses the default Qwen pair solely while asking that launcher to render
the resource scaffold, then replaces the workload command and model fields
with the pinned benchmark driver and configured checkpoint. Final validation
requires the configured `MODEL_ID`, `MILES_MODEL_TYPE`, checkpoint receipt, and
dataset receipt; a scaffold value cannot reach an accepted Job.

Before an external run, the driver actively probes the installed MILES loader
with a normal-lifecycle `WeightTransferProtocol` and verifies that the
configured ModelExpress factory is a synchronous callable. A runtime image
built against the stale whole-round interface fails before training. The
external arm then uses the current PR #3304 hybrid protocol factory, with
routed NCCL M2N and residual broadcast bytes accounted separately.

The only permitted `emptyDir` is the RAM-backed `/dev/shm` volume. It carries
no models, workspaces, logs, or results and is intentionally recreated for each
Job.

Keep `require_symmetric_weight_checker=true` for accepted paired results. The
driver passes `--ci-test`, requires MILES'
`check_weight_update_equal` to be active, and records only actual completed
MILES compare calls against the corresponding weight version. The current PR
#3304 driver performs that compare after the initial update only. Later updates
therefore remain `receipt_complete=false` and cannot become acceptance evidence
until MILES exposes an observed per-update comparison; the harness never
promotes the enabled flag into equality. `verify_tensor_equality=false` is the
performance default because that setting controls an additional
ModelExpress-only transport digest, not the symmetric MILES correctness gate.
It may be enabled for a separate ModelExpress qualification run.

## Evidence Layout

```text
<output_root>/<run_id>/
  manifest.json
  schedule.jsonl
  trials.jsonl
  summary.json
  failure.json                    # only on failure
  preflight/
    node.json
    pods.json
    pvc-*.json
    pv-*.json
    storage.json
  jobs/block-*/
    manifest.yaml
    environment.json
    commands.jsonl
    raw.log
    trials.jsonl
    summary.json
    job.json
    pods.json
    nodes.json
    events.json
    pvcs.json
    pod-logs.json
    *.stdout.log
    *.stderr.log
```

Legacy rank-zero timer logs produce timing statistics but are marked
`receipt_complete=false`. Acceptance requires structured records with full
coverage, byte accounting, version/readiness receipts, tensor digest equality,
the symmetric MILES checker, and successful completion of the real training
command. The schema does not claim a separate per-version post-refit generation
check; `training_run_completed` is emitted only after `execute_train` returns
successfully.

All configured source repositories must be clean. The harness records observed
pod placement, GPU product/count, PVC claim names, writable mount paths, and
current RWO consumers; a claimed matched run fails if any differs from the
configured paired node. Preflight also sums effective GPU requests from
Running and Pending pods on the paired node and fails unless the exact
configured Job request remains schedulable. The observed workload pod must
request and limit exactly `topology.total_gpus`; zero-GPU and sidecar-GPU
manifests are rejected.

As of September 23, 2026, the configured AWS development node has all eight
H100 GPUs requested by the standing devbox. The read-only preflight therefore
rejects the four-GPU example until those GPUs are released or the persistent
claims are staged on another schedulable node.

## Claim Boundary

Use `benchmark_kind=topology_preserving_validation` for the AWS development
workload. A PR #3304 reproduction claim is accepted only with the exact
64-GB300 envelope recorded in the metrics contract. The example pins the
materialized DeepSeek V4 checkpoint receipt, but its single-node AWS topology
still cannot establish the published 16-node GB300 result.
