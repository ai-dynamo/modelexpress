<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MILES external weight-transfer protocol (ModelExpress NCCL M2N)

This package carries the MILES-side adapter that streams base-model weights
from MILES trainer ranks to ModelExpress-managed inference engines over the
NCCL M2N collective path. `miles_protocol.py` implements the MILES
`WeightTransferProtocol` seam (duck-typed; MILES is never imported at module
scope) on top of the ModelExpress refit client in this repository.

## Enabling the protocol in MILES

Point MILES at the external protocol with two flags:

```text
--update-weight-transfer-mode=external
--update-weight-transfer-protocol=modelexpress_rl.collective.integrations.miles_protocol:build_protocol
```

`build_protocol(args)` lazily subclasses the MILES `WeightTransferProtocol`
ABC and returns a protocol instance. MILES calls, per weight version:
`connect(...)` once, then `begin_sync(version, iter_buckets)`, one
`send_bucket(bucket)` per HF weight bucket, and `finalize(version)` inside the
engine pause window (`use_weight_update_session=True`). The loader rejects
whole-round external protocols, so this adapter drives the standard bucket
stream: the first `send_bucket` prepares the ModelExpress session and opens
the round, each bucket publishes any completed publish groups in canonical
plan order, and `finalize` finishes the round and waits for generator-side
completion.

## Configuration

All knobs are mx-namespaced. Each can be set by a MILES argument or an
environment variable; the environment fallback is read only when the argument
is unset. **The environment variables are the supported path**: stock MILES
entry points parse their arguments strictly and reject the four
`--modelexpress-*` flags below, which are only reachable through a custom
entry hook.

| Environment variable | MILES argument | Default | Notes |
| --- | --- | --- | --- |
| `MX_SERVER_ADDRESS` | `--modelexpress-server-address` | none (required) | mx-server `host:port`. `build_protocol.validate_args` fails loudly when neither the env var nor the argument is set. Plain `host:port` or `grpc://`; secure schemes are rejected. |
| `MX_MILES_RUN_ID` | `--modelexpress-m2n-run-id` | unset | Optional run identity. When set it must be set identically on every trainer rank; a mix of set and unset ranks is rejected. |
| `MX_MILES_PUBLISH_GROUPS` | `--modelexpress-m2n-publish-groups` | `1` | Integer >= 1. Caps how many publish groups a round is chunked into along the canonical plan order; larger values give finer engine-side overlap granularity. Never splits a tensor. |
| `MX_MILES_CONNECT_TIMEOUT_S` | `--modelexpress-m2n-connect-timeout-s` | `10.0` | Seconds the first round waits for the mx-server channel to become ready before failing. |

Configuration errors are meant to surface before any weight moves:
`validate_args` runs during MILES argument validation and reports a missing
server address, an unsupported scheme, or a malformed knob with the exact
flag/env to fix. An unreachable mx-server fails at the first round's connect
probe with a `RuntimeError` naming the endpoint, the timeout, and the
remediation — not deep inside a round.

## Canonical plan order

Publish groups are contiguous byte-balanced chunks of the weight plan sorted
by the 795 canonical key (`ParamPlan.canonical()`), and every trainer rank
publishes every group id in that order. The ordering is the wire contract: the
795 receiver backend re-sorts `plan.bulk` canonically and executes each layer
group's active subset in canonical order regardless of caller order, so any
contiguous chunking of a canonically ordered plan stays wire-safe — the
receiver never has to reconcile two orderings of the same tensors.

## Supported geometry (fail-closed limits)

The adapter validates the topology at `connect`/`begin_sync` time and raises
outside the supported envelope:

- One source rank per PP partition: TP/EP must be fully gathered on the
  source rank (`required_placement` is `WeightUpdatePlacement(gather_pp=False)`
  with TP/EP gather); every trainer rank is a sender for its own partition.
- `--megatron-to-hf-mode raw` (the MILES default) is required: `bridge` mode
  forces `gather_pp=True`, and this adapter rejects that placement at
  `connect()` with "MILES NCCL M2N requires PP-local HF tensors".
- Explicit engine GPU topology: `engine_gpu_counts`/`engine_gpu_offsets` must
  be provided; ambiguous engine placement is rejected.
- Base-model rounds only: `supports_lora = False`; selector must be `"all"`
  or `"target"`. Speculative decoding is supported only with a frozen draft
  model and the `"target"` selector.
- BF16 weights only; tensor names, shapes, and dtypes are frozen after the
  first `begin_sync` and any change across rounds is an error.
- Contiguous, non-scalar tensors in stable storage: the bucket stream must
  reference the wire buffers materialized during `begin_sync`.

Any mid-round failure closes the protocol (channel, rendezvous, and session
torn down) and the next `begin_sync` raises rather than silently continuing a
degraded round. Failures raised inside `begin_sync` or `finalize` fan out to
every trainer rank through gloo all-gathers. A failure raised out of the
`send_bucket` path closes the local rank immediately and propagates out of
miles' `update_weights`, so that rank never reaches `finalize`: its peers
finish their bucket loop and then block in miles' gloo barrier until the
process group errors or its timeout, the receiver's transfer deadline
(`MX_NCCL_REFIT_TRANSFER_TIMEOUT_S`, default 600s), or job teardown releases
them. When the failing rank is rank 0 (the driver),
`end_weight_update`/`resume_engines` never run and the engines stay paused
until the job is torn down.

## Receiver-side deployment requirements

The derived `operation_id` (`miles-<group>-weight-version-<version>`) names a
transfer row this adapter never creates: `create_transfer` is not part of the
795 client surface, and the trainer intentionally never reports. The deployed
full-branch receiver reports with that `operation_id` at end of round, and
mx-server maps a missing transfer row to NOTFOUND, which fails the receiver's
round after the weights have landed. A deployment must therefore ensure a
transfer row exists per `operation_id` — seeded out-of-band, or created by a
server build carrying the idempotent transfer API — before the receiver's
report fires.

The adapter sends no tensor digests, so the receiver's
`MX_MILES_VERIFY_TENSOR_EQUALITY` digest check must stay at its default (off);
enabling it against this adapter fails loudly at round start.

The 795-base rendezvous has no bootstrap-fence RPCs, so this client cannot
answer a fence: groups must form with `requires_bootstrap_fence=0`. A group
formed with the fence required would deadlock at bootstrap, with receivers
waiting at the fence for trainers that have no RPC to arrive with.

## Performance caveat

This port targets the minimal 795-base client surface. The wide-lane and
drain defaults at this base are **untuned**: do not read throughput numbers
from this configuration. Certified performance numbers live on the full
feature branch (`modelexpress-miles-nccl-m2n`). The one tuning lever exposed
here is the publish-group count above, which trades per-group overhead
against engine-side overlap granularity.

Memory: the trainer pays two HF conversions per round — one materializing
the bucket stream inside `begin_sync`, one by the MILES updater's own
`send_bucket` pass (whose tensors this adapter ignores; the wire buffers
were already filled). The wire buffers themselves are a persistent extra
replica of this rank's PP-local BF16 weights, held for the protocol's
lifetime. Size trainer memory accordingly: steady state is one extra
partition replica, and `begin_sync` adds at most one bucket of transient on
top.
