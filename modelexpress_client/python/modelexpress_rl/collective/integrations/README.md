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

All knobs are mx-namespaced; every argument has an environment fallback read
only when the argument is unset.

| MILES argument | Environment variable | Default | Notes |
| --- | --- | --- | --- |
| `--modelexpress-server-address` | `MX_SERVER_ADDRESS` | `127.0.0.1:50051` | mx-server `host:port`. `build_protocol.validate_args` fails loudly when neither the argument nor the env var is set. Plain `host:port` or `grpc://`; secure schemes are rejected. |
| `--modelexpress-m2n-run-id` | `MX_MILES_RUN_ID` | unset | Optional run identity. When set it must be set identically on every trainer rank; a mix of set and unset ranks is rejected. |
| `--modelexpress-m2n-publish-groups` | `MX_MILES_PUBLISH_GROUPS` | `1` | Integer >= 1. Caps how many publish groups a round is chunked into along the canonical plan order; larger values give finer engine-side overlap granularity. Never splits a tensor. |
| `--modelexpress-m2n-connect-timeout-s` | `MX_MILES_CONNECT_TIMEOUT_S` | `10.0` | Seconds the first round waits for the mx-server channel to become ready before failing. |

Configuration errors are meant to surface before any weight moves:
`validate_args` runs during MILES argument validation and reports a missing
server address, an unsupported scheme, or a malformed knob with the exact
flag/env to fix. An unreachable mx-server fails at the first round's connect
probe with a `RuntimeError` naming the endpoint, the timeout, and the
remediation — not deep inside a round.

## Supported geometry (fail-closed limits)

The adapter validates the topology at `connect`/`begin_sync` time and raises
outside the supported envelope:

- One source rank per PP partition: TP/EP must be fully gathered on the
  source rank (`required_placement` is `WeightUpdatePlacement(gather_pp=False)`
  with TP/EP gather); every trainer rank is a sender for its own partition.
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
torn down, errors fanned out to every rank); the next `begin_sync` raises
rather than silently continuing a degraded round.

## Performance caveat

This port targets the minimal 795-base client surface. The wide-lane and
drain defaults at this base are **untuned**: do not read throughput numbers
from this configuration. Certified performance numbers live on the full
feature branch (`modelexpress-miles-nccl-m2n`). The one tuning lever exposed
here is the publish-group count above, which trades per-group overhead
against engine-side overlap granularity.
