<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<h1 align="center">ModelExpress for RL Weight Refit</h1>

<p align="center">
  <strong>Move each trainer version into rollout-worker layouts without first assembling a full model on one rank.</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#the-design">Design</a> •
  <a href="#implementation-status">Status</a> •
  <a href="#integration-contract">Integration</a> •
  <a href="#validation">Validation</a>
</p>

> [!IMPORTANT]
> The refit package is experimental. It contains framework-neutral resharding primitives, a NIXL transport, normalized timing, and vLLM receiver/install code. It does not yet provide a turnkey integration for an RL framework, and several safety gates listed in [Implementation status](#implementation-status) remain open.

## Executive summary

Reinforcement Learning (RL) post-training repeatedly moves a new model version from distributed trainer ranks to rollout workers. ModelExpress (MX) extends its peer-to-peer loading model to this **refit** path: trainer ranks publish the shards they already own, and each rollout rank pulls the ranges needed for its own Tensor Parallelism (TP), Pipeline Parallelism (PP), and Expert Parallelism (EP) layout. The MX server coordinates discovery but never carries weight bytes; NVIDIA Interconnect eXchange Library (NIXL) reads move data directly between registered Graphics Processing Unit (GPU) buffers. The current code establishes this shared design and a vLLM receiver, while framework orchestration, trainer publication adapters, version-atomic discovery, and broader engine coverage remain integration work.

## Overview

An RL training step changes the model. Rollout workers must install that version before they generate samples against it. The time from “trainer version ready” to “required rollout workers ready” is refit latency, and it sits on the training loop's critical path.

| Refit cost | Conventional path | ModelExpress design |
|---|---|---|
| Trainer layout | Gather or checkpoint distributed shards | Keep each rank's native shard registered |
| Topology change | Central process reconstructs, then receivers reshard | Each receiver plans from source ownership into its own layout |
| Data path | Trainer gather, storage, or full-payload broadcast | Direct NIXL reads from owning ranks |
| Worker membership | Usually tied to a collective or checkpoint barrier | Receiver discovers sources and starts independently |
| Installation | Re-run the engine's general loader | Use an engine adapter; optionally cache destination mappings |

![Conventional RL refit centralizes the model before redistribution, while ModelExpress publishes existing trainer shards and lets rollout ranks pull their needed ranges](images/rl-refit-critical-path.svg)

### A plain-English model

Think of the trainer as a library whose book is already split across several desks. A rollout worker does not ask one desk to photocopy and bind the whole book. It brings a page list for its local edition, looks up which desk owns each page, and reads those pages directly into the right sections.

That analogy maps to four MX concepts:

| Term | Meaning |
|---|---|
| **Shard ownership** | A trainer rank owns a range of a global tensor at a registered address. |
| **Target geometry** | The ranges and destination layout one rollout rank's real model loader expects. |
| **Transfer plan** | Source-to-destination byte runs that intersect ownership with target geometry. |
| **Install** | Engine-specific work that commits receive buffers into the live inference model. |

The difficult part is not the network copy. Training and inference often store the same logical tensor in different layouts, with fused projections, different parallelism, quantization, or grouped Mixture-of-Experts (MoE) weights. MX separates the framework-independent range planning from the engine-specific interpretation and installation of those tensors.

## The design

The going-forward pattern is:

> **Per-rank publish, receiver-side pull, receiver-side transform and install.**

The RL framework decides when a version is ready and which rollout workers must finish. Trainer adapters describe the buffers and global ranges each rank owns. MX discovers those sources, plans direct reads, and executes the transfer. An inference adapter captures the destination layout and installs the result into the live engine.

![Six-step receiver-driven refit lifecycle from trainer publication through ModelExpress discovery, planning, NIXL reads, and rollout installation](images/receiver-driven-refit.svg)

### End-to-end lifecycle

1. **Publish trainer ownership.** Each trainer rank registers its existing buffers with NIXL and publishes tensor shape, dtype, global shard offset, local shape, address, and endpoint metadata.
2. **Discover a source set.** A rollout rank waits for the expected READY trainer ranks and fetches their shard tables through [`MxReshardRendezvous`](reshard/rendezvous.py).
3. **Capture target geometry.** The inference adapter dry-runs the engine's real weight loader with zero-storage [`LazyWeight`](reshard/geometry.py) tensors. It records which source view is read and where that view lands in a destination parameter; no weight bytes move.
4. **Build a receiver-local plan.** [`plan_transfer`](reshard/transfer_plan.py) intersects each recorded target range with the published trainer shards and emits contiguous byte runs.
5. **Pull into registered buffers.** [`NixlReshardTransport`](reshard/transport/nixl.py) groups reads by source session and issues batched one-sided Remote Direct Memory Access (RDMA) reads.
6. **Transform and install.** The receiver casts dtype-mismatched staging buffers when supported, then calls the engine adapter to update live model storage and derived state.

The first update performs discovery, geometry capture, plan construction, allocation, and memory registration. Later updates reuse the plan and buffers while source topology, shard boundaries, and addresses remain unchanged.

### Resharding during transfer

Source and target layouts are expressed in one global tensor coordinate system. A trainer rank might publish “rows 0–3,” while a rollout rank requests “columns 3–4 across every row.” The planner intersects the request with each owning shard and emits reads that reconstruct the rollout parameter directly.

![A rollout rank requests a column slice crossing two row-sharded trainer ranks; ModelExpress plans two reads that reconstruct the local destination without a full-model gather](images/reshard-while-moving.svg)

Geometry capture supports pure views that can be represented as a rank-preserving, axis-aligned box, including narrowing, unit-step slicing, and dimension permutations. [`paired_runs`](reshard/slice_plan.py) preserves actual destination strides, so non-contiguous destinations become multiple correct byte runs instead of one incorrect contiguous copy.

This design removes the trainer-side full-model gather. It does **not** guarantee that every tensor becomes one large network operation: a strided view can produce many short runs, and unsupported loader operations require a separate fallback strategy.

### Control plane and data plane

The MX server is a directory. It stores source identity and rendezvous metadata, while trainer-to-rollout weight bytes stay on the NIXL data plane.

| Layer | Responsibility |
|---|---|
| RL framework | Select version, trigger publish/refit, wait for required workers, enforce rollout staleness policy |
| Trainer adapter | Translate native FSDP, DTensor, or Megatron ownership into global tensor ranges |
| MX rendezvous | Publish and discover READY ranks and their registered shard metadata |
| MX receiver | Capture target geometry, build/reuse the plan, allocate buffers, execute reads |
| Inference adapter | Interpret names/fusions, install parameters, refresh quantized or derived state |
| Inference engine | Own live parameters, caches, compiled graphs, and final readiness |

Fully Sharded Data Parallel (FSDP), DTensor, and Megatron integrations belong in trainer adapters; they are not hard-coded into the planner. The current rendezvous format carries shard geometry in a JSON side table because the public protobuf does not yet have typed multidimensional ownership fields.

### Transport and installation are separate

Refit has two independent optimization surfaces:

1. **Move fewer and better-shaped bytes.** The reshard planner selects source ranges and NIXL reads them into receiver buffers.
2. **Commit those bytes with less loader overhead.** The inference adapter updates live model storage, quantization scales, fused parameters, and derived tensors.

The RL path keeps transfer and engine concerns separate. [`nixl_staged_transfer.py`](../../modelexpress_rl/inference/nixl_staged_transfer.py) owns exact-manifest planning, registered staging, transfer, and verification. The private vLLM [`installer.py`](../../modelexpress_rl/inference/engines/vllm/installer.py) captures load-time geometry on an unquantized meta-model twin and uses vLLM's layerwise reload path to preserve storage referenced by compiled Compute Unified Device Architecture (CUDA) graphs.

[`MdlLoader`](../engines/vllm/refit/installer.py) is a separate experimental vLLM installer called Mapped Direct Load (MDL). It caches direct, fused, and expert destination views so warm updates can copy into known slots instead of repeating general loader dispatch. MDL can consume partial input batches, but the reshard transport in this package does not yet expose a selector that reduces wire bytes for partial updates. The two features must not be treated as one end-to-end partial-refit path until that selector is wired and validated.

Each receiver keeps one load-time receive buffer per captured destination, plus a source-dtype conversion staging buffer for any parameter whose served dtype differs from its load-time dtype. Both come from classic CUDA allocations and stay registered for the receiver's lifetime, because re-registering per refit is what the cached plan exists to avoid.

That buffer shape is why the vLLM receiver installs through `process_weights_after_loading` (PWAL) rather than MDL. The receiver reconstructs *load-time* tensors, and a quantized model still needs the engine's post-load processing to derive its runtime representation from them. MDL is appropriate only when the incoming tensors already match the validated runtime representation, which is why it is a separate opt-in path rather than the default.

## Integration contract

The shared receiver has two engine-specific hooks:

```python
from modelexpress.refit.reshard import ReshardReceiver


class RuntimeReceiver(ReshardReceiver):
    def _capture(self, manifest):
        # Dry-run the runtime's loader and return:
        # (CaptureResult, {parameter_name: (load_time_shape, load_time_dtype)})
        ...

    def _install(self, receive_buffers):
        # Commit buffers into live model storage and refresh derived state.
        ...
```

The framework constructs one receiver per rollout rank and calls `update_weights(step)` when its version barrier permits the update. A trainer-side adapter must build [`PublishedTensor`](reshard/rendezvous.py) records, wrap them with NIXL endpoint metadata, and publish one READY record per trainer rank.

This is an adapter contract, not a complete quick start. The repository does not currently include the RL framework lifecycle hooks or a general trainer publisher that derives ownership from arbitrary training backends.

### Stable-plan assumption

The receiver caches its first discovery and transfer plan. Every later call reuses the original trainer addresses and shard boundaries.

This is valid only when:

- trainer rank membership is stable;
- registered buffer addresses stay stable;
- tensor names, shapes, dtypes, and ownership do not change;
- the receiver's load-time layout stays stable.

A trainer restart, reshard, scale event, or buffer replacement requires rediscovery and replanning. The current receiver does not detect those changes.

## Implementation status

### Implemented in this repository

| Capability | Status | Evidence |
|---|---|---|
| Loader-driven geometry capture | Implemented | [`geometry.py`](reshard/geometry.py), [`test_reshard_refit_geometry.py`](../../tests/test_reshard_refit_geometry.py) |
| Multidimensional shard intersection | Implemented | [`slice_plan.py`](reshard/slice_plan.py), [`test_reshard_refit_slice_plan.py`](../../tests/test_reshard_refit_slice_plan.py) |
| Strided destination reconstruction | Implemented in reference tests | [`test_reshard_refit_transfer.py`](../../tests/test_reshard_refit_transfer.py) |
| Multi-rank shard rendezvous | Implemented with temporary JSON metadata | [`rendezvous.py`](reshard/rendezvous.py) |
| Per-source batched NIXL reads | Implemented | [`transport/nixl.py`](reshard/transport/nixl.py), [`test_reshard_refit_nixl_transport.py`](../../tests/test_reshard_refit_nixl_transport.py) |
| Same-shape dtype conversion | Implemented through staging buffers | [`receiver.py`](reshard/receiver.py) |
| Stable-topology plan and buffer reuse | Implemented | [`ReshardReceiver`](reshard/receiver.py) |
| RL exact-version staged NIXL transfer | Implemented | [`modelexpress_rl/inference/nixl_staged_transfer.py`](../../modelexpress_rl/inference/nixl_staged_transfer.py) |
| vLLM geometry capture and layerwise install | Implemented adapter code | [`modelexpress_rl/inference/engines/vllm/installer.py`](../../modelexpress_rl/inference/engines/vllm/installer.py) |
| vLLM mapped direct install | Implemented as a separate opt-in installer | [`engines/vllm/refit/installer.py`](../engines/vllm/refit/installer.py) |
| Normalized refit timing schema | Implemented | [`timing.py`](timing.py), [`test_refit_timing.py`](../../tests/test_refit_timing.py) |
| Descriptor bound for strided slices | Implemented for gap-free dim-0 partitions | [`transfer_plan.py`](reshard/transfer_plan.py), [`test_reshard_refit_transfer.py`](../../tests/test_reshard_refit_transfer.py) |

“Implemented” means the code and focused tests are present. It does not by itself mean a framework/model/topology combination has passed distributed end-to-end validation.

### Not yet provided as a general guarantee

| Gap | Current behavior |
|---|---|
| Full-pull fallback for unsupported operations | The planner identifies unsupported tensors, but `ReshardReceiver` fails closed because its full-pull/install fallback is not implemented. This is distinct from the descriptor bound above, which pulls whole source shards for *supported* but descriptor-heavy slices. |
| Complete coverage gate | The planner does not yet prove that published overlaps cover every requested element before transfer. |
| Version-atomic multi-rank manifest | Discovery waits for a rank count but does not commit and pin one atomic version across all trainer records. |
| Topology-change handling | The cached plan is not invalidated after trainer restart, reshard, scaling, or address change. |
| Partial/subset wire filtering | MDL accepts subset batches, but the reshard receiver currently executes its full cached plan on each update. |
| Expert-aware wire filtering | Expert destination mapping exists in MDL; the reshard planner has no receiver-owned expert selector. |
| Parameter digest verification | Publishers can stamp each shard with a position-sensitive digest (`MX_RESHARD_PUBLISH_DIGEST`, see `refit/reshard/verify.py`), and it is carried through discovery into the planning inputs, but the live receiver does not yet recompute and compare. The comparison needs a fresh-discovery refresh of the expectation, or ordinary training updates between prepare and a later step read as corruption. |
| Inference-to-inference fan-out | Rollout workers do not republish installed refit buffers through this package. |
| General engine support | The shared core is engine-neutral, but only a vLLM receiver adapter is present. |
| General trainer support | Framework-specific FSDP, DTensor, and Megatron publisher adapters are not present here. |
| Transport-neutral receiver | A transport protocol exists for planning tests, but `ReshardReceiver` setup and handshake are currently NIXL-bound. |

## Timing and configuration

[`RefitTimingRecorder`](timing.py) defines a shared stage vocabulary so transport and installer changes can be compared without conflating wire time with end-to-end readiness:

1. control discovery;
2. source preparation;
3. setup and registration;
4. transfer planning;
5. wire transfer;
6. receive synchronization;
7. transformation;
8. installation;
9. post-install work;
10. rollout readiness.

Set `MX_REFIT_TIMING_STDOUT=1` when a benchmark harness must collect the normalized `MX_REFIT_TIMING` JSON record from worker stdout. Lower layers add spans only when a recorder is active.

The reshard planner uses this control:

| Variable | Default | Purpose |
|---|---|---|
| `MX_RESHARD_MAX_SEGMENTS_PER_COPY` | `64` | Descriptor budget per captured copy. Above it, the planner pulls whole gap-free dim-0 source shards into contiguous staging instead of issuing one descriptor per strided run. |

vLLM's separate MDL path uses these controls:

| Variable | Default | Purpose |
|---|---|---|
| `MX_LOAD_MODE` | `stock` | Set `direct` to enable mapped direct installation. |
| `MX_FP8_LOADERLESS` | automatic | `1` forces loaderless 8-bit floating-point (FP8) installation; `0` disables the guard. |
| `MX_LOAD_LAYOUT_VERSION` | empty | Explicitly invalidates cached destination mappings after a layout change. |

## Validation

### Focused tests

Run the framework-neutral refit tests from `modelexpress_client/python`:

```bash
pytest \
  tests/test_reshard_refit_geometry.py \
  tests/test_reshard_refit_slice_plan.py \
  tests/test_reshard_refit_transfer.py \
  tests/test_reshard_refit_rendezvous.py \
  tests/test_reshard_refit_nixl_transport.py \
  tests/test_refit_timing.py
```

The strongest local test reconstructs destination parameters from sharded source buffers through capture, planning, and the in-memory reference transport, then compares them byte-for-byte with the engine loader's ground truth. NIXL dispatch tests verify descriptor grouping and address/device mapping without requiring a GPU.

### Distributed acceptance criteria

A distributed integration should not claim a passing refit based on transfer completion alone. At minimum it should verify:

- one pinned model version across every trainer rank;
- complete requested-byte coverage with no silent fallback;
- parameter equality after installation;
- generation parity after the refit;
- correct TP, PP, EP, and replica placement;
- explicit source, wire, transform, install, and rollout-ready timings;
- bounded behavior after worker restart, late join, and timeout.

Performance claims must identify the exact implementation path. Reference transport, sliced NIXL resharding, full-tensor NIXL transfer, MDL installation, and framework collective paths measure different work and should be reported separately.

## Tradeoffs and failure modes

- **Receiver-driven pull vs. trainer-driven push:** pull lets workers join independently and avoids maintaining a receiver list on trainers. It requires source buffers and registrations to remain alive until receivers finish.
- **Exact slices vs. descriptor count:** exact slicing reduces bytes but can create many short reads. The planner bounds this: when a captured copy exceeds `MX_RESHARD_MAX_SEGMENTS_PER_COPY` (default 64) it pulls each gap-free dim-0 source shard once into contiguous staging and replays the captured views locally, trading extra wire bytes for a descriptor count bounded by source shard count. When the published layout is not a complete dim-0 partition it keeps the exact descriptors instead, so the bound never changes correctness behavior.
- **Cached plan vs. elasticity:** plan reuse removes repeated setup from warm updates. It is unsafe after source membership, ownership, or addresses change unless the receiver detects and rebuilds.
- **Generic capture vs. explicit adapters:** dry-running the real loader avoids a hand-written reshard specification for every model pair. Unsupported arithmetic, materializing reshapes, and model-specific derived state still require engine adapter work.
- **Fail closed vs. fallback:** failing on unsupported tensors prevents silently serving mixed model versions. A production fallback must materialize and install those tensors without weakening version and coverage checks.

## Open questions

- **Version commit:** Which component publishes the atomic manifest that pins all trainer ranks to one version?
- **Plan invalidation:** What stable topology and address digest should trigger rediscovery?
- **Partial refit:** Should the selector be expressed as layers, parameter names, expert IDs, or a framework-supplied predicate?
- **Source lifetime:** How does the orchestrator acknowledge completion before trainers release old buffers?
- **Contention:** How should several rollout ranks distribute reads across replicated trainer sources?
- **Installation contract:** Which inference-engine API owns derived tensors, quantization, and compiled-graph storage stability?

## Roadmap

### Near term

- Add complete coverage and version-consistency gates.
- Implement the full-pull fallback for unsupported loader operations; the descriptor-heavy case is already bounded.
- Rebuild plans when source topology or registered addresses change.
- Add a public trainer publisher contract with typed shard geometry.

### Mid term

- Wire partial parameter and expert selectors through planning and transport.
- Validate the shared receiver contract with additional inference engines.
- Add load-aware source selection and rollout-to-rollout fan-out.
- Unify reshard transport and MDL installation under explicit, measured integration paths.

### Longer term

- Support planner decisions across several receivers instead of independent local plans.
- Retain and release versions through an orchestrator-visible completion contract.
- Feed rollout-readiness and fabric contention into topology and source selection.

## Package map

| Path | Role |
|---|---|
| [`timing.py`](timing.py) | Normalized timing stages and context propagation |
| [`reshard/geometry.py`](reshard/geometry.py) | Record the engine loader's source views and destination writes |
| [`reshard/slice_plan.py`](reshard/slice_plan.py) | Resolve views, intersect shard boxes, emit contiguous runs |
| [`reshard/transfer_plan.py`](reshard/transfer_plan.py) | Build and execute receiver-local plans |
| [`reshard/rendezvous.py`](reshard/rendezvous.py) | Publish/discover shard ownership and NIXL endpoints |
| [`reshard/receiver.py`](reshard/receiver.py) | Shared receiver lifecycle, buffers, staging, and install hooks |
| [`reshard/transport/`](reshard/transport/) | Reference and NIXL transport adapters |
| [`modelexpress_rl/inference/nixl_staged_transfer.py`](../../modelexpress_rl/inference/nixl_staged_transfer.py) | RL exact-manifest planning, staged NIXL transfer, and verification |
| [`modelexpress_rl/inference/engines/vllm/installer.py`](../../modelexpress_rl/inference/engines/vllm/installer.py) | vLLM geometry capture and graph-safe layerwise install |
| [`engines/vllm/refit/installer.py`](../engines/vllm/refit/installer.py) | Optional vLLM mapped direct installer |

## Related documentation

- [ModelExpress overview](../../../../README.md)
- [Python client](../../README.md)
- [ModelExpress architecture](../../../../docs/ARCHITECTURE.md)
- [Deployment and NIXL configuration](../../../../docs/DEPLOYMENT.md)
