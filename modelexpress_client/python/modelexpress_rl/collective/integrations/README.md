<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MILES over the ModelExpress NCCL M2N collective path

MILES support for the NCCL M2N collective refit path: a DP x TP MILES trainer
streams gathered (whole-tensor) BF16 base weights to SGLang rollout engines,
brokered by the ModelExpress server. Engines receive either whole tensors or,
in sharded mode, only the slice each engine TP rank keeps.

- Trainer side: `miles_protocol.py` implements the MILES
  `WeightTransferProtocol` seam. MILES is never imported at module scope:
  `build_protocol` subclasses the MILES ABC lazily inside the factory.
- Engine side: `sglang_receiver.py` is the receiver factory SGLang calls for
  groups that name it; `sglang.py` carries the loader and session machinery.
  SGLang needs the allowlisted external weight-update receiver hook (SG-1),
  currently only on the `sglang-miles` branch and not yet merged.
- `manifest.py` is the manifest codec both sides share (`m2n_manifest` /
  `receiver_init_payload`: plan plus topology under a single schema tag).
- `sglang_layout.py` is the one rule both sides use to decide which tensors
  an SGLang engine receives pre-split.

## Enabling it

Trainer (MILES; setting the protocol path implies external transfer mode):

```bash
export MX_SERVER_ADDRESS=<mx-server-host:port>
--custom-weight-transfer-protocol-path=modelexpress_rl.collective.integrations.miles_protocol.build_protocol
```

Install this package in the trainer environment too — MILES validates the
flag by importing the protocol path before launch. The endpoint resolves
through the shared client resolver
(`modelexpress.client._get_server_url`): a set `MODEL_EXPRESS_URL` takes
precedence, then `MX_SERVER_ADDRESS`. Unlike the other client paths there is
no localhost default — one of the two variables is required, and
constructing the protocol without one fails. Plain `host:port` is expected;
`grpc://` and `http://` prefixes are stripped, and the secure schemes
(`grpcs://`, `https://`) are rejected, so a configured MX auth token travels
unencrypted on this path. An unreachable server fails the first round's
connect probe within 10 s with an error naming the endpoint.

`MX_MILES_DST_LAYOUT` picks the destination layout:

- `replicate` (default): every generator receives every tensor whole and
  installs it through SGLang's own `load_weights`.
- `sharded`: each engine TP rank receives only the slice it keeps, written
  straight into SGLang's live (fused) parameter storage; tensors the shared
  rule leaves whole still go through `load_weights`. Every engine must have
  the same TP size, and the trainer reads the head counts, vocabulary size,
  and embedding tying from `--hf-checkpoint`'s `config.json`.

The mode is part of the receiver ABI identity and so of the plan digest; a
trainer and engines that disagree never form a group. An unknown layout fails
when MILES builds the protocol.

`MX_NCCL_REFIT_STACK_BYTES` (non-negative integer, bytes of one stacked
call's area) turns on equal-geometry stacking. `0` (the default) keeps one
M2N call per tensor, with the plan and digest unchanged. Above `0`, the
trainer stacks the tensors that share their plan geometry (shape, dtype,
meshes, placements; only plan facts) along a new leading dimension and labels
the members `group_key = "m2n-stack1-<id>"`. The keys are part
of the plan digest and the receiver ABI gains a `+stack1` suffix, so an
engine without the derivation refuses the topology at prepare. Each end
checks that every stack fits the largest `NCCL_RESHARD_PACK_BUFFSIZES` bucket
(2 GiB when unset). Stacks are submitted ungrouped, one call per publish
group. A sharded destination receives into one reusable scratch per lane
stream and copies each member into SGLang's live storage; a replicated one
reads each stack's members in place.

Engines (SGLang with the SG-1 hook, unmerged as of this writing): install
this package in the engine image and allowlist the factory on the engine
launch. The SGLang engine arg is `--weight-update-receivers`; in a MILES
launch the same value reaches the engines as:

```bash
--sglang-weight-update-receivers modelexpress_rl.collective.integrations.sglang_receiver.create_receiver
```

The trainer names that same path in the `receiver` field of its
`init_weights_update_group` request; SGLang refuses any receiver path that
is not on the allowlist. The engines get the mx-server endpoint as the
request's `master_address`/`master_port`, so they need no other endpoint
configuration. With MX auth enabled they also need the token provisioned:
the receiver wraps its mx-server channel in `auth.with_auth`, which reads
`MX_AUTH_TOKEN_PATH` (default `/var/run/secrets/tokens/modelexpress`) — the
same file the trainer side reads.

## How a round runs

MILES calls `connect(...)` once, then per weight version `begin_sync`, one
`send_bucket` per HF bucket, and `finalize`. `begin_sync` runs before MILES
pauses the engines; the `send_bucket`/`finalize` pass runs inside the pause
window. Every trainer rank is a sender in one reshard lane; its lane rank is
`dp.rank * tp.size + tp.rank`, its coordinate in the `(DP, TP)` source mesh.

- `begin_sync` validates every tensor and copies it into a persistent wire
  buffer. Round one agrees a run id (rank 0's), all-gathers every rank's
  manifest of names and shapes over the trainer's Gloo group, checks they
  are identical and cover every `(DP, TP)` coordinate once, and freezes the
  plan and topology. A failure on any rank fails `begin_sync` on all of them.
- The first `send_bucket` opens the mx-server channel, joins the collective
  group with this rank's trainer slot, and rank 0 sends
  `init_weights_update_group` with `receiver` and the manifest as
  `receiver_init_payload` to every engine. Each round, rank 0 creates the
  transfer operation bound to the joined group and epoch, sends
  `update_weights_from_distributed` with empty names/dtypes/shapes and a
  `receiver_payload` of exactly `{"operation_id", "version"}`, and every rank
  publishes the single publish group once all of its tensors have arrived.
- `finalize` finishes the round on every rank and rank 0 waits for every
  engine's response. A failure on any rank fails the round on all of them; it
  is reported to mx-server before the protocol closes.
- `close()` destroys the engine groups with `destroy_weights_update_group`
  and releases the session, rendezvous, and channel. Every resource is
  attempted even when an earlier close fails; the first failure propagates
  after the rest settle. Teardown is terminal for later rounds but retried
  per retained resource: a failed `close()` keeps the rendezvous, channel,
  and engine fan-out it could not release, and the retry re-runs those
  closes; the session's own close is one-shot, so it is not retained. The
  destroy fan-out tolerates engines
  that already forgot the group (SG-1 answers a repeat destroy with HTTP 400
  and a "does not exist" body), so a retry after a partial fan-out converges;
  with nothing left, `close()` returns silently.

No private SGLang patching or entry point is involved: the receiver group
creates no torch process group, and SGLang forwards the payload untouched.

The plan is sorted by `ParamPlan.canonical()`, and the receiver re-sorts
canonically too, so both sides post the same per-lane tensor sequence. The
source mesh is `(TP,)` when DP is one and `(DP, TP)` otherwise, all
`Replicate` (the trainer gathers TP). The destination mesh is `(generators,)`
in `replicate` mode; in `sharded` mode it is `(TP,)` for one engine and
`(engines, TP)` for several, with the engine axis replicated, because a split
is valid only inside one engine.

## Supported envelope (fail-closed)

- A DP x TP trainer gathering TP: PP, EP, ETP and CP must be one, DP must be
  intra-DP (`indep_dp` one), and the trainer world must be exactly DP x TP.
  The resolved placement must gather TP (all MILES iterators do), so every
  rank yields every tensor whole. Trainer-local sources — MILES yielding
  TP-local shards via the iterator's `tensor_layouts` — are not supported;
  they fail closed at connect and at the layout contract. That is phase-2
  work (MILES-P2 / MX-3b).
- `--megatron-to-hf-mode raw` (the MILES default): `bridge` mode forces
  `gather_pp=True`, which `connect()` rejects. Raw mode converts from a
  Megatron checkpoint, so the run needs a valid one at `--load` (or
  `--ref-load`).
- Explicit engine GPU topology: `engine_gpu_counts` and `engine_gpu_offsets`
  must cover every engine without overlap.
- Base-model rounds only: no LoRA; selector `"all"` or `"target"` (echoed on
  every round). Receiver refit requires engines without speculative
  decoding: SGLang rejects receiver rounds while a draft model exists, so
  the adapter refuses to connect when `sglang_speculative_algorithm` is
  set.
- BF16, contiguous, non-scalar tensors; names, local shapes and layouts are
  frozen after the first round, and any change is an error.
- Sharded destinations (`MX_MILES_DST_LAYOUT=sharded`): dense BF16
  Qwen2/Qwen3-style SGLang layers on the stock (unquantized, non-presharded)
  weight loaders, and the embedding on the engine's full TP group (no DP
  attention). q/k/v split only when the KV heads divide the engine TP size,
  and the embedding only when SGLang's vocabulary padding is a no-op;
  otherwise those tensors stay whole. A receiver that cannot prove a sharded
  entry's slice refuses `prepare`; it never stages a shard. The aliased
  storage is re-checked (address, shape, stride) at every round start. An
  upstream SGLang refactor that invalidates any of these proofs degrades
  to a hard refusal, never silent corruption.
  Sharded sources stay rejected: a plan that shards the source side fails
  closed on both ends.

Any mid-round failure closes the protocol (engines, session, rendezvous, and
channel) and later rounds raise. MILES never calls `close()` after successful
rounds, so those resources live until process exit. A failed round poisons
its engine-side receiver until the closing protocol fans out
`destroy_weights_update_group`.

A failed round is terminal for the trainer process: `connect()` refuses
re-entry, and MILES builds the protocol once per `WeightUpdater`, so after a
failed round the trainer cannot sync weights again in-process. Recovery is
restarting the trainer; there is no in-run rejoin.

A reconnect before any round has failed (MILES re-calling `connect` after an
engine heal) tears the live session down; the next round re-prepares against
the new engines under a fresh group name. A heal that changes the engine GPU
topology fails closed at the next `begin_sync`.

## Memory and performance

The wire buffers are a persistent extra BF16 copy of each trainer rank's
whole tensors, and each round runs two HF conversions: one in `begin_sync`,
one in the MILES updater's own `send_bucket` pass, whose tensors this adapter
does not read. Sharded destinations receive into the live model storage, so
sharded entries carry no receive buffer at all; replicated entries keep one
per canonical tensor. The lane topology and install defaults at this base are
untuned; do not read throughput numbers from this configuration.

## Testing

The CPU suite covers both sides:
`pytest tests/test_collective_miles*.py tests/test_collective_sglang_*.py`
from `modelexpress_client/python`. The `TestAgainstSglang` classes in
`tests/test_collective_sglang_public_receiver.py` skip unless
`MX_TEST_SGLANG_PYTHON` names the `python/` tree of an SGLang checkout with
the SG-1 receiver hook, which runs them against SG-1's real
`WeightUpdateReceiverContext` and `WeightUpdater`.
`tests/test_collective_sglang_engine_view.py` pins the sharded slice
arithmetic against the fork's real layer classes (same
`MX_TEST_SGLANG_PYTHON` gate); run it against every new fork base — it is
the canary that turns upstream drift into a visible test failure.
