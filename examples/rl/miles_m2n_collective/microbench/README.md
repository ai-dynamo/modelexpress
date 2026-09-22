<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Standalone M2N microbench

This is a transport microbench for the MILES NCCL M2N hot path. It is not the
existing MILES GRPO benchmark and does not measure exporter, SGLang,
ModelExpress control-plane, or Kubernetes overhead.

The GPU worker calls the production
`modelexpress_rl.collective.backend._reshard` binding and drains through
`LaneCommunicator.synchronize(timeout_s=...)`. The CPU mode has no CUDA or NCCL
dependency; it validates deterministic scheduling, drain counts, and logical
destination equality only.

`--schedule miles-current` is the default and is the only shipping-shaped
schedule: it creates singleton parameter groups, has senders drain once after
all submissions, and has receivers drain after each group. `--schedule
synthetic` enables the `--grouping` and `--drain` comparison knobs; its output
is explicitly labelled as a synthetic scheduling experiment.

## CPU behavioral matrix

```bash
examples/rl/miles_m2n_collective/microbench/run_microbench.sh mock \
  --matrix \
  --profile smoke \
  --warmup 1 \
  --repetitions 3 \
  --json-out /tmp/m2n-mock.json
```

The JSON marks this mode as behavioral. Do not interpret its timing fields as
GPU performance numbers.

## Native GPU case

Use a CUDA environment with the installed production Python client,
`nccl-extensions`, a loaded NCCL version of at least 2.30.7, and one visible
GPU for each source partition plus destination. No ModelExpress server is
needed because the harness bootstraps lane IDs directly.

```bash
export M2N_MB_PARTITIONS=2
export M2N_MB_FANOUT=4
export M2N_MB_TIMEOUT_S=180
export M2N_MB_COHORT_TIMEOUT_S=1800
export M2N_MB_OUT=/tmp/m2n-p2-f4.json
export NCCL_CUMEM_ENABLE=1

examples/rl/miles_m2n_collective/microbench/run_microbench.sh gpu \
  --profile smoke \
  --streams 2 \
  --group-keys unique \
  --warmup 2 \
  --repetitions 8
```

`M2N_MB_TIMEOUT_S` is the deadline for each native round.
`M2N_MB_COHORT_TIMEOUT_S` is the independent process-cohort ceiling and must
cover bootstrap, warmups, measured rounds, and cleanup.

GPU mode rejects any value other than `NCCL_CUMEM_ENABLE=1` and rejects
`NCCL_COMM_ID` before Gloo or NCCL formation. The launcher uses `timeout`
around the complete cohort. Within the process, all drains share one full-round
deadline; a bootstrap, timeout, or exact-equality failure aborts every lane
before Gloo teardown and does not write a successful result.

## Profiles and comparison axes

`smoke` is intentionally small. `qwen-2.5-3b` is a synthetic representative
290-tensor BF16 manifest with a `model.embed_tokens.weight` tensor shaped
`(151936, 2048)` and exactly `622329856` bytes. It does not reproduce the
measured production plan's tensor count or PP-local ownership and must not be
used to project a production speedup. It is useful for controlled sensitivity
comparisons between schedules on one fixed payload. The large profile requires
an explicit native opt-in:

```bash
examples/rl/miles_m2n_collective/microbench/run_microbench.sh gpu \
  --profile qwen-2.5-3b \
  --allow-large-profile
```

The supported source partition counts are `1`, `2`, and `4`; fanout values are
`1`, `2`, `4`, and `8`; stream counts are `1`, `2`, and `4`. Grouping can be
`singleton`, `layer`, or `bucket`; drains can be `per-tensor`, `per-group`, or
`end`; group keys can be `unique` or `shared`. Grouping and drains are
synthetic-only controls.

Current production `_reshard` does not forward `ParamPlan.group_key` to the
native call. Shared keys remain in the plan identity and output, but the JSON
will report native fusion as unsupported rather than presenting a false fusion
comparison.

Every native measured round verifies byte equality on every destination tensor
by reducing the `uint8` minimum and maximum against a deterministic uniform
byte pattern, which avoids allocating a `full_like` verification tensor. Output
includes the case/manifest/plan fingerprints, imported-module provenance,
loaded NCCL, CUDA/driver, per-rank GPU identity, topology, immutable run UUID
and UTC start, VRAM envelope/preflight, high-resolution elapsed time, logical
throughput, delivered effective bandwidth, and per-rank issue/drain counts.
