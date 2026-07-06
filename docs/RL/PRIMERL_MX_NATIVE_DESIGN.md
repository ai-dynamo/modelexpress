# PRIME-RL × ModelExpress — Native API Design (Path B)

**Status**: Design proposal (no code yet)
**Last Updated**: April 2026
**Companion to**: `PRIMERL_MX_OVERVIEW.md` (the overlay-on-PI design, "Path A")

This document describes a **native MX-shaped weight broadcast backend** for PRIME-RL that uses [PI's NIXL transport](https://github.com/PrimeIntellect-ai/prime-rl/pull/2326) as the bytes-on-wire data plane, but exposes **ModelExpress's traditional API surface** to PRIME-RL — model-agnostic, server-mediated, scratch-buffer-default, cross-framework.

It's the alternative to "Path A" (strict overlay on PI's API). Path A is in flight as draft PR [PrimeIntellect-ai/prime-rl#2343](https://github.com/PrimeIntellect-ai/prime-rl/pull/2343); Path B is staged here as the design we'd advocate for if Path A hits friction or if the team wants the broader strategic value.

---

## 1. Why This Doc Exists

Path A's overlay strategy preserves PI's NIXL API surface end-to-end and replaces only the SPG rendezvous with MX Server discovery. It's small, cooperative, easy to merge — and it inherits every PI design constraint:

- **Per-model `conversion_specs()` requirement.** PI's `TransportPlan` only supports models that PI has authored a spec table for: today, just `glm_moe_dsa`. Plain Qwen3, Llama, Mixtral, anything else needs a spec table written. We just discovered this when scenario A's trainer crashed with `'FSDPQwen3ForCausalLM' object has no attribute 'conversion_specs'` and had to monkey-patch HF Qwen3 to unblock.
- **Direct refit only — KL drift class of bugs is in scope.** PI's PR is currently blocked by 27+ iterations of KL drift investigation. Their byte-exact transport is correct; the drift comes from concurrent UCX writes into live vLLM `param.data`. Inheriting their target-buffer model means inheriting that bug surface.
- **Static, startup-time tensor registration.** `Slot{Sharded,Gathered,Expert}` assume fixed tensor shapes registered with NIXL once at init. Elastic workloads (LoRA-RL with dynamic adapter add/remove, frozen-policy-adapts variants, growing context-len rollouts) don't fit this model cleanly.
- **prime-rl-only.** PI's `NIXLWeightBroadcast` lives in `prime_rl/trainer/rl/broadcast/nixl.py`; cross-framework portability would require us to copy the design into verl + future NeMo-RL.

Path B is the "what if we redesigned the prime-rl-side weight broadcast around MX's shape, but kept PI's UCX/NIXL setup as-is for the bytes" answer. The data plane is identical (same RDMA bytes on wire, same per-NIC bandwidth, same `rc_mlx5` transport, same per-rank NIC pin). Only the control plane and tensor ABI change.

---

## 2. What MX-Traditional Looks Like (Recap)

ModelExpress has a consistent API across the integrations we've shipped (verl `MxCheckpointEngine`, our existing PRIME-RL `ModelExpressWeightBroadcast` on `kavink/mx-weight-broadcast`):

```python
# Trainer side
publisher = MxTrainingPublisher(
    agent_name="trainer-rank-0",
    device_id=local_rank,
    server_url="modelexpress-server.kavin.svc.cluster.local:8001",
)
publisher.initialize()                           # NIXL agent up, gRPC connected
publisher.publish_weights(state_dict, step=N)    # per training step
# ...later, before optimizer.step() reuses buffers:
publisher.unpublish(version=N)                   # mutability contract


# Inference side
receiver = MxRefitReceiver(
    model_name="...",
    worker_rank=k,
    device_id=local_rank,
)
receiver.initialize(model_tensors=scratch_or_live_dict)   # NIXL register receive buffers
source = receiver.poll_for_source(min_version=N)          # discover trainer
receiver.receive_weights(source)                          # RDMA pull (or accept WRITE)
# scratch path: vllm_model.load_weights(scratch_iter)
# direct path: receive lands directly in vllm_model.param.data
```

Salient differences from PI's design:

| Concern | PI (Path A overlay inherits) | MX-traditional (Path B exposes) |
|---------|------------------------------|--------------------------------|
| Discovery | SPG static rendezvous | gRPC `MxClient` (publish, list_sources, get_metadata) |
| Source identity | Implicit rank pairing | Content-addressed `mx_source_id = sha256(SourceIdentity)` |
| Per-model contract | `model.conversion_specs(layer_idx)` | None — model-agnostic; publishes live `state_dict` tuples |
| Tensor ABI | Fixed `Slot{...}` registered at startup | Per-step `(name, shape, dtype, gpu_addr)` published; receiver re-registers if shape drifts |
| Receive target | Direct WRITE into live `param.data` | **Scratch buffer + `model.load_weights()`** by default; opt-in direct refit |
| Quantization | First-class `ConversionSpec`/`QuantizationSpec` | Trainer pre-quantizes if needed; MX is dtype-agnostic |
| Lifecycle | rendezvous → register × N → write × N per startup | publish/poll/receive/unpublish per step; no static startup registration |
| Versioning | Implicit step counter | First-class `extra_parameters.training_step` |
| Mutability contract | Implicit (root cause of KL drift?) | Explicit `unpublish()` with drain |
| Cross-framework | prime-rl only | Same client runs in verl, prime-rl, future NeMo-RL |
| vLLM integration | Worker extension only | Worker extension OR `WeightTransferEngine` plugin (Step 11) |

---

## 3. Path B Architecture

Same NIXL data plane as PI; different prime-rl-side abstractions on top of it.

```text
┌───────────────────────────────────────────────────────────────────┐
│  prime-rl trainer process                                         │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  MxWeightBroadcast (new, in prime_rl/trainer/rl/broadcast/) │  │
│  │  ─ implements PI's WeightBroadcast ABC                      │  │
│  │  ─ delegates the data path to MxTrainingPublisher           │  │
│  │  ─ delegates the control path to MxClient gRPC              │  │
│  └─────────────────────────────────────────────────────────────┘  │
│        │                                  │                       │
│        ▼ data                             ▼ metadata              │
│   MxTrainingPublisher              MxClient(server_url=...)       │
│   (modelexpress)                   (modelexpress)                 │
│        │                                  │                       │
│        ▼                                  ▼                       │
│   NixlAgentWrapper            ┌─────────────────────────────┐    │
│   (PI's existing class)       │  MX Server (gRPC + Redis)   │    │
│        │                      │  ─ source registry          │    │
│        ▼ post_write           │  ─ poll_for_source          │    │
│   UCX rc_mlx5 RDMA            │  ─ pipeline replication     │    │
│   (same wire as PI)           │  ─ retention + versioning   │    │
│        │                      └─────────────────────────────┘    │
│        ▼                                                          │
│   ConnectX-7 NIC ─── RoCE ───► rollout NIC                        │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│  prime-rl inference (vLLM) worker                                 │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  MxWeightUpdateWorker (new)                                 │  │
│  │  ─ vLLM worker extension                                    │  │
│  │  ─ MxRefitReceiver delegate                                 │  │
│  │  ─ default: scratch buffer + model.load_weights()           │  │
│  │  ─ opt-in: direct refit into live param.data                │  │
│  └─────────────────────────────────────────────────────────────┘  │
│        │                                                          │
│        ▼                                                          │
│   MxRefitReceiver (modelexpress)                                  │
│        │                                                          │
│        ▼                                                          │
│   NixlAgentWrapper (PI's existing class)                          │
│        │                                                          │
│        ▼                                                          │
│   accepts WRITE from trainer (same as PI)                         │
└───────────────────────────────────────────────────────────────────┘
```

### What we adopt unchanged from PI's PR

These are real engineering wins; we'd be foolish to redo them:

- `NixlAgentWrapper` (UCX agent setup, register_tensor, prep_local/prep_remote, post_write, wait, drain)
- `pin_ucx_rail` (per-rank NIC pinning that's the difference between 4.8 GB/s and 7.5 GB/s)
- `classic_cuda_pool` (allocator workaround for `expandable_segments` + `ibv_reg_mr` "local protection" bug)
- The runtime image stack (UCX 1.19, NIXL 0.10.1, ARM64 quirks)
- Pre-write SPG barrier / quiescence pattern (the iter15 fix; we'd implement equivalent via MX Server fence RPC)
- HSDP primary-replica gate (only `dp_replicate == 0` runs the protocol)

These come into MX as either direct re-export or via a `prime_rl.utils` import; no need to fork.

### What we replace with MX-shape

| PI component | Our replacement |
|--------------|-----------------|
| `NIXLWeightBroadcast` | `MxWeightBroadcast` — same prime-rl ABC (`WeightBroadcast`), different internals |
| `NIXLWeightUpdateWorker` | `MxWeightUpdateWorker` — same vLLM worker_extension_cls slot |
| `TransportPlan` | Replaced. Per-step iteration over `state_dict` tuples instead of slot table |
| `model.conversion_specs(layer_idx)` | Removed entirely. Trainer publishes whatever's in `state_dict`; vLLM's `model.load_weights()` does the HF→kernel format on inference (its tested code path) |
| `Slot{Sharded,Gathered,Expert}` | Removed. The per-rank publishing semantics come from `MxTrainingPublisher`'s `worker_rank` field; tiny-tensor coalescing is a NIXL register optimization not a slot type |
| SPG rendezvous (`StatelessProcessGroup`) | Removed. Discovery is `MxClient.poll_for_source`; per-step barrier is `MxClient.fence` (new) or fall back to a small SPG over MX-discovered endpoints |
| `ConversionSpec`/`QuantizationSpec` (FP8 quantize on trainer) | Optional on trainer side. Adopted as `MxQuantizer` if user wants FP8 fast path; default is BF16 passthrough. Inference-side decompresses via vLLM's existing FP8 loader, not via NIXL post-processing. |

### What we add that PI doesn't have

These are MX traditional features that translate naturally into prime-rl now that we're not constrained by PI's slot abstraction:

- **Pipeline replication** (TensorHub-style DAG): rollout publishes itself as a secondary source after receive; MX Server load-balances new pollers across trainer + rollouts.
- **Peer recovery**: a restarting rollout pod pulls from any surviving peer (via `poll_for_sources` ranked by health/locality), not always from trainer.
- **Versioning + retention**: keep-latest-N versions on MX Server; rollouts can request a specific version or "latest stable."
- **Mutability contract**: explicit `unpublish(version)` before trainer reuses slot buffers; server blocks until in-flight pulls drain. Directly addresses PI's KL drift hypothesis (write ordering / live param visibility).
- **Cross-framework**: same `MxTrainingPublisher`/`MxRefitReceiver` already proven in verl and our existing PRIME-RL POC. Path B is the alignment: prime-rl + verl + future NeMo-RL share the same MX abstractions.
- **Scratch-buffer default**: receive lands in isolated GPU tensors; vLLM `model.load_weights()` applies them via its tested NCCL-equivalent path. KL-drift class of bugs falls away. Direct refit becomes opt-in for users who measure and accept the correctness risk.
- **Elastic shapes**: per-step `(name, shape, dtype)` publishing means LoRA-RL, dynamic adapters, growing context lengths all work without re-init.

---

## 4. Concrete File Footprint

What changes in `KavinKrishnan/prime-rl:kavink/mx-on-nixl` to ship Path B (relative to current Path A overlay):

### New files

| File | Purpose | Est. LOC |
|------|---------|----------|
| `src/prime_rl/trainer/rl/broadcast/mx.py` | `MxWeightBroadcast` — new prime-rl `WeightBroadcast` impl. ~3 methods: `__init__` (publisher + agent setup), `broadcast_weights(model, step)` (publish state_dict tuples, signal orchestrator), `shutdown()` | ~300 |
| `src/prime_rl/inference/vllm/worker/mx.py` | `MxWeightUpdateWorker` — vLLM worker_extension_cls. Delegates to `MxRefitReceiver` for receive, then `model.load_weights(scratch_iter)` for apply. Direct-refit opt-in via env. | ~250 |
| `src/prime_rl/configs/...` | New `MxWeightBroadcastConfig` discriminator on `WeightBroadcastConfig` union. `type: "mx"` | ~60 |
| `docs/weight-transfer-modelexpress.md` | Already requested by `@mikasenghaas` in #2326 review. Documents all four backends (filesystem, nccl, nixl, mx) with selection guidance | ~250 |

### Modified files

| File | Change | Est. LOC |
|------|--------|----------|
| `src/prime_rl/configs/trainer.py`, `orchestrator.py`, `rl.py` | Add `MxWeightBroadcastConfig` to discriminated union; thread through unify-mode for the `rl` entrypoint | +40 |
| `src/prime_rl/trainer/rl/broadcast/__init__.py` | Add `mx` dispatch in `setup_weight_broadcast()` | +15 |
| `src/prime_rl/inference/vllm/server.py` (or where worker_extension_cls is wired) | `mx` value selects `MxWeightUpdateWorker` | +10 |
| `src/prime_rl/utils/client.py` (TRANSFER_READY marker) | Touch for `mx` backend too (already protocol-agnostic per the existing review feedback) | +5 |

### Files we don't touch

PI's NIXL backend stays in place. Users who want PI's path get `type: "nixl"`; users who want ours get `type: "mx"`. No conflict between the two backends — they coexist as discriminator options.

### What MX-side code we add (in `ai-dynamo/modelexpress`)

Mostly already exists from the verl + existing PRIME-RL POCs. Net new:

- `MxTrainingPublisher.publish_weights_via_nixl(state_dict, step, agent)` — adapt our existing publisher to use PI's `NixlAgentWrapper` directly (same NIXL bytes as PI, just driven from our publisher class).
- `MxRefitReceiver.receive_weights_via_nixl(source, agent, target)` — same.
- `MxClient.fence(model, version, world_size)` — server-mediated barrier (replaces SPG barrier per step). Optional; can fall back to SPG over MX-discovered endpoints if we want to minimize server changes.

Plus the server-side capabilities tracked in `PRIMERL_MX_OVERVIEW.md` §3 (pipeline replication index, retention, peer recovery preference ordering).

---

## 5. Migration Path for Users

Users currently on PI's `type: "nixl"`:

```toml
# Before (PI's path)
[weight_broadcast]
type = "nixl"
host = "..."
port = 29502
inference_world_size = N
backends = ["UCX"]

# After (MX-native path), no rebuild required
[weight_broadcast]
type = "mx"
mx_server_url = "modelexpress-server.kavin.svc.cluster.local:8001"
model_name = "..."
inference_world_size = N
# transfer_mode = "scratch"  # default; opt-in "direct" for the speed/correctness tradeoff
# pipeline_replication = false  # default; opt-in for fan-out scale
```

Same image, same UCX setup, same NIXL transport — just a different config discriminator and ~600 LOC of new code in prime-rl. PI's existing `nixl` backend stays available.

For ourselves: we delete the monkey-patch on Qwen3 (`qwen3_specs_patch.py`) — it's no longer needed because Path B doesn't require per-model conversion specs. Path A's overlay survives as-is for users who want to stick with PI's transport API exactly.

---

## 6. Pitch Sequence for the Design Conversation

If we propose Path B to PI on the existing draft PR (or in a new sibling PR):

1. **Acknowledge PI's transport win directly.** "Your NIXL transport is excellent — UCX setup, classic_cuda_pool, pin_ucx_rail, FP8 ConversionSpec are all correct. Path B keeps every byte of that."

2. **Frame the divergence as scope.** "We've been running ModelExpress as a metadata + elasticity layer across verl and our internal PRIME-RL POC for several months. The MX-shape API is model-agnostic, server-mediated, scratch-buffer-default — it solves problems your `Slot`/`ConversionSpec` design doesn't address (cross-framework, elastic shapes, KL drift via scratch path, retention, pipeline replication)."

3. **Show the demo evidence.** "Path A overlay (already up as #2343) proved the metadata layer works on top of your transport. Path B extends that with a native MX API — here's the design doc, here's a draft diff."

4. **Make the cohabitation explicit.** "Path B doesn't replace `type: nixl`. It adds `type: mx` as a sibling discriminator. Users pick. PI's GLM-5 production keeps using `nixl`; users who want a model-agnostic / cross-framework / scratch-default path use `mx`. Same UCX runtime image, same NIXL bytes, same `pin_ucx_rail` discipline — different control plane."

5. **Address the KL drift directly.** "The drift you're chasing in iter22-27 is consistent with concurrent live-param writes. The MX scratch-buffer path isn't subject to that bug surface because writes land in isolated tensors, then `model.load_weights()` applies them via vLLM's NCCL-equivalent code path. Happy to A/B this on your iter26/27 config — same NIXL bytes, just a different target buffer. If your drift disappears in scratch mode, that's diagnostic data even if you keep `nixl` as default."

---

## 7. Inflection Points to Pivot A → B

- **PI's PR stalls on KL drift > 2 weeks** without a root-cause fix landing. Path B's scratch-buffer default ships independent of that investigation.
- **Reviewer pushback on Path A's overlay shape** — env-var-as-config or monkey-patching Qwen3 specs gets pushback that suggests a cleaner design is wanted.
- **Need for elastic / LoRA-RL features** that PI's slot system can't accommodate. Path B's per-step publish handles dynamic shapes naturally.
- **Cross-framework alignment becomes a strategic priority** — leadership wants "one weight broadcast story across prime-rl + verl + future frameworks" rather than per-framework redesigns.
- **Scenario A's first NIXL run reveals the same KL drift on Qwen3** as PI saw on GLM-5. That's a strong empirical signal that direct-refit-into-live-params is the bug surface, not GLM-specific. Path B's scratch-default becomes the obvious fix.

---

## 8. Decisions Pending

| Decision | Default if not made |
|----------|---------------------|
| Ship Path B alongside Path A or as a follow-up PR? | Follow-up PR; Path A goes first |
| MX-mediated barrier (`MxClient.fence`) or SPG-over-MX-discovered endpoints? | SPG-over-MX-discovered for v0.1 (smaller server change) |
| `MxWeightBroadcast` lives in `prime_rl/trainer/rl/broadcast/mx.py` (in-tree) or as a plugin from `modelexpress` package? | In-tree mirroring PI's nixl.py pattern |
| Scratch vs direct refit default | Scratch (correctness-safe). Direct opt-in via `transfer_mode: direct` |
| Pipeline replication default | Off. Opt-in via `pipeline_replication: true` |

---

## 9. Status

- Path A draft PR: [PrimeIntellect-ai/prime-rl#2343](https://github.com/PrimeIntellect-ai/prime-rl/pull/2343) — Qwen3 conversion_spec patch in flight, image rebuild done, deploy pending tsh refresh.
- Path B: design only (this doc). No code yet. Ready to author when an inflection point above triggers.
- Tracking: `recovery/reinforcement learning/PRIME_INTELLECT_PR2326_Analysis.md` for the strategic comparison; `PRIMERL_MX_OVERVIEW.md` for the Path A overlay design.

This doc is the artifact we'd reference in the conversation with PI if Path A doesn't carry the day on its own.
