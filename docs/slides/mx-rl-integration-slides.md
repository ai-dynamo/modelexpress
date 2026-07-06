# ModelExpress for RL Weight Updates

> **April 2026 — Integration Design Overview**
>
> `NVIDIA NIXL` · `ModelExpress` · `RL Post-Training`

Extending GPU-to-GPU RDMA transfers from **inference scaling** to the **training→inference refit boundary** in reinforcement learning post-training.

Target frameworks: **NeMo RL** · **verl** · **PRIME-RL**

---

## Slide 1 — Title

**ModelExpress for RL Weight Updates**

Extending GPU-to-GPU RDMA transfers from **inference scaling** to the **training→inference refit boundary** in reinforcement learning post-training.

Target frameworks: **NeMo RL** · **verl** · **PRIME-RL**

---

## Slide 2 — The Problem: The Weight Sync Bottleneck

On-policy RL (GRPO, PPO, DAPO) alternates between rollout generation on inference GPUs and gradient updates on training GPUs. After every training step, updated weights must reach inference before the next rollout — this **refit phase** stalls both sides.

### Wall-clock time breakdown (illustrative)

```text
|  Rollout (40%)  | Rew | Train (20%) | ██ REFIT (30%) ██ |
                                        ▲ BOTTLENECK ▲
```

> Up to 30–40% of wall-clock for 70B+ models

### Current refit latency (70B-class model, multi-node)

| Method | Latency |
|--------|---------|
| Filesystem (PRIME-RL) | ~20s+ |
| NCCL Broadcast (NeMo RL) | ~10s |
| ZMQ IPC (NeMo RL, co-located) | ~3-5s |
| **MX RDMA P2P (target)** | **~5s** |

---

## Slide 3 — The Solution: ModelExpress for Training→Inference Refit

Extend MX from inference-to-inference P2P to the training→inference boundary. Training workers register updated weights with NIXL, publish metadata to the MX Server, and RDMA-WRITE directly into inference GPU memory — bypassing CPU, disk, and collective overheads.

### High-level data flow

```text
Training Workers          MX Server              Inference Workers
(FSDP2 / Megatron)       (gRPC + Redis/CRD)     (vLLM / SGLang)
                                                 
  WeightExtractor    ──gRPC──►  Metadata Coord  ◄──gRPC──  MxRefitReceiver
  MxTrainingPublisher           Version Tracking            NIXL Agent
  NIXL Agent                                               
       │                                                    ▲
       └══════════════ RDMA WRITE (GPU→GPU) ════════════════┘
                   bypasses CPU & disk
```

> *See: [diagram-architecture.svg](diagram-architecture.svg)*

### Performance

| Metric | Value |
|--------|-------|
| Llama-3.3-70B (140 GB) | **~5s** |
| DeepSeek-V3 MoE (681 GB) | **~15s** |
| CPU staging required | **0** |
| Transport fallback | **Auto**: IPC → RDMA → TCP |

---

## Slide 4 — Architecture: Component Deep-Dive

> *See: [diagram-component-stack.svg](diagram-component-stack.svg)*

### Training Workers

| Component | Status | Description |
|-----------|--------|-------------|
| RL Framework | Existing | NeMo RL / verl / PRIME-RL |
| Training Backend | Existing | FSDP2 / Megatron-LM |
| **WeightExtractor** | **NEW** | Gather params per bucket (FSDP2 / Megatron) |
| **MxTrainingPublisher** | **NEW** | Register tensors with NIXL, publish metadata (gRPC), version tag |
| **ResharderPlugin** | **NEW** | Gather-then-shard / Direct-match / Auto |
| NIXL Agent | Existing | UCX backend, per-GPU, RDMA WRITE |

### ModelExpress Server (Rust)

| Component | Status | Description |
|-----------|--------|-------------|
| gRPC P2P Service | Existing | PublishMetadata, ListSources, GetMetadata, UpdateStatus |
| Redis / K8s CRD Backend | Existing | Metadata persistence and HA |
| **Refit Version Tracking** | **NEW** | training_step in SourceIdentity; version-filtered ListSources |
| **Bucket Coordination** | **NEW** | RefitCoordination message for bucket-level progress |
| Heartbeat / Reaper | Existing | Stale source detection and GC |
| p2p.proto | **MODIFIED** | New enums, fields, messages for RL refit |

### Inference Workers

| Component | Status | Description |
|-----------|--------|-------------|
| Inference Engine | Existing | vLLM / SGLang |
| **MxRefitReceiver** | **NEW** | Poll for new weight versions, coordinate with MX Server |
| NIXL Agent | Existing | UCX backend, pre-registered receive buffers |
| Apply Weights In-Place | Existing | FP8 quantization on receiver side |
| **SyncPolicyController** | **NEW** | Sync / One-step-off / Fully async modes |
| Resume Rollout | Existing | Continue generation after refit |

### Data paths

- **Metadata (gRPC)**: Training → MX Server → Inference (dashed, control plane)
- **Data plane (RDMA WRITE)**: Training NIXL Agent → Inference NIXL Agent (per bucket)

---

## Slide 5 — Integration Map: Three Frameworks, One Abstraction

A framework-agnostic **WeightSyncBackend** decouples the transfer mechanism from each framework's orchestration and sync policy.

> *See: [diagram-framework-comparison.svg](diagram-framework-comparison.svg)*

### Comparison

| Dimension | NeMo RL | verl | PRIME-RL |
|-----------|---------|------|----------|
| Training Backend | DTensor / Megatron | FSDP / FSDP2 / Megatron | FSDP2 (EP/CP) |
| Inference Backend | vLLM, SGLang | vLLM, SGLang, HF | vLLM |
| Current Sync | ZMQ IPC / HTTP / NCCL | NCCL / CheckpointEngine | Filesystem + HTTP |
| **MX Insertion Point** | `refit_policy_generation()` | `CheckpointEngine` ABC | Orchestrator `relay_weights()` |
| Primary Benefit | RDMA replaces ZMQ/NCCL | Multi-node sans filesystem | Eliminates disk I/O |

### Per-framework summary

**NeMo RL** — New branch alongside ZMQ IPC and NCCL in refit function. Bucket-streamed transfer maps to MX publish. Ray actor integration.

**verl** — `ModelExpressCheckpointEngine` implements v0.7 `CheckpointEngine` ABC. Targets async server mode. Engine mode stays NCCL (already optimal co-located).

**PRIME-RL** — Replaces filesystem relay in orchestrator. For cross-DC (Intellect-2), MX acts as fast intra-cluster delivery under SHARDCAST.

---

## Slide 6 — Delivery: Phased Integration Plan

### Phase 1 — Weeks 1-4: Foundation

- WeightExtractor abstraction (FSDP2 + Megatron)
- MxTrainingPublisher with NIXL + gRPC
- Proto extensions (`RL_REFIT`, `training_step`)
- Single-node RDMA validation
- Benchmark vs. ZMQ IPC baseline

### Phase 2 — Weeks 5-10: Framework Integrations

- NeMo RL: MX branch in refit function
- verl: `ModelExpressCheckpointEngine`
- PRIME-RL: Orchestrator MX relay
- Fallback to existing mechanisms
- E2E GRPO/PPO correctness tests

### Phase 3 — Weeks 11-13: Hardening

- ResharderPlugin (gather-then-shard)
- Error handling + fallback paths
- MoE bucket completion tracking
- FP8 quantization validation
- Multi-node benchmarks (70B, MoE)

### Target Performance

| Model | Nodes | Current | MX Target |
|-------|-------|---------|-----------|
| Llama-3.1-8B | 2 | NCCL ~3s | **MX ~1s** |
| Llama-3.3-70B | 4 | ~10-20s | **MX ~5s** |
| DeepSeek-V3 MoE | 8+ | ~30s+ | **MX ~15s** |

### Key risks

- **Parallelism layout mismatch** — mitigated by ResharderPlugin and config alignment guidance
- **InfiniBand dependency** — clean fallback to NCCL/TCP/filesystem when unavailable

---

## Diagrams

All diagrams are available as standalone SVG files for embedding in external slide tools:

| File | Description |
|------|-------------|
| [diagram-rl-loop-bottleneck.svg](diagram-rl-loop-bottleneck.svg) | RL training loop with refit bottleneck highlighted |
| [diagram-architecture.svg](diagram-architecture.svg) | Three-column architecture: Training → MX Server → Inference |
| [diagram-component-stack.svg](diagram-component-stack.svg) | Full component stack with NEW/MODIFIED/EXISTING tags |
| [diagram-transfer-flow.svg](diagram-transfer-flow.svg) | Sequence diagram: one refit step (publish → poll → RDMA WRITE → apply) |
| [diagram-framework-comparison.svg](diagram-framework-comparison.svg) | NeMo RL / verl / PRIME-RL comparison grid |
