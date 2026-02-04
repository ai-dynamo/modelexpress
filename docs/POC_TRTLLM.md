# TensorRT-LLM P2P Transfer POC Documentation

This document describes the Proof of Concept (POC) implementation for P2P GPU-to-GPU weight transfers with TensorRT-LLM (TRT-LLM) using NIXL/RDMA, including the current design, validated results, and future optimization paths.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [POC Results](#2-poc-results)
3. [Current Design Overview](#3-current-design-overview)
4. [TRT-LLM Stack Integration Points](#4-trt-llm-stack-integration-points)
5. [Step-by-Step P2P Transfer Workflow](#5-step-by-step-p2p-transfer-workflow)
6. [Ideal Long-Term Design](#6-ideal-long-term-design)
7. [Potential Optimizations](#7-potential-optimizations)
8. [Key Code Components](#8-key-code-components)
9. [Lessons Learned](#9-lessons-learned)

---

## 1. Executive Summary

The POC demonstrates successful **P2P GPU-to-GPU model weight transfer** for TensorRT-LLM using:
- **NIXL (NVIDIA Interconnect eXchange Library)** for zero-copy RDMA transfers
- **ModelExpress server** for metadata coordination via gRPC
- **TRT-LLM's PyTorch backend** for direct HuggingFace model loading

### Key Achievement

| Model | Transfer Size | Bandwidth | Inference |
|-------|--------------|-----------|-----------|
| Llama 70B (TP=8) | 170.54 GB | 112.3 Gbps | 15.9 TPS |

This is a **10-20x speedup** over traditional NVMe-based model loading for large models.

---

## 2. POC Results

### 2.1 Validated Test Configuration

```
Model: meta-llama/Llama-3.1-70B-Instruct
Tensor Parallelism: 8 GPUs
Dtype: bfloat16
Infrastructure: 2x DGX nodes with InfiniBand (HDR 200Gbps)
TRT-LLM Version: 0.21.0
NIXL Version: 0.8.0
```

### 2.2 Performance Metrics

| Phase | Metric | Value |
|-------|--------|-------|
| **P2P Transfer** | Total size | 170.54 GB |
| | Per-rank transfer | ~21 GB |
| | Per-rank time | 1.5s |
| | Bandwidth | 112.3 Gbps |
| **Reconstruction** | Tensors | 723 |
| | Reconstructed size | 141.11 GB |
| | Time | ~5s |
| **TRT-LLM Load** | HF model load | 4.7s |
| | Engine build | ~113s |
| | Total | 117.8s |
| **Inference** | Tokens/sec | 15.9 TPS |
| | Generated text | Coherent |

### 2.3 Sample Output

```
Prompt: "Hello, I am a large language model trained by"
Generated: "Meta AI. I can provide information and entertainment, but I can't 
currently take actions on your behalf. For example, I can plan a custom travel 
itinerary, but I can't buy tickets or book hotels..."
```

---

## 3. Current Design Overview

### 3.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        ModelExpress P2P for TRT-LLM                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   Source Node                                    Target Node                  │
│   ┌─────────────────────────────┐               ┌─────────────────────────┐  │
│   │                             │               │                         │  │
│   │  HuggingFace Model (Disk)   │               │  P2P Receiver           │  │
│   │         │                   │               │  (Init Container)       │  │
│   │         ▼                   │               │         │               │  │
│   │  MxTrtllmSourcePublisher    │               │         ▼               │  │
│   │  ┌─────────────────────┐   │               │  MxTrtllmTargetLoader   │  │
│   │  │ Load HF weights     │   │               │  ┌───────────────────┐  │  │
│   │  │ Shard for TP=8      │   │               │  │ Query MX server   │  │  │
│   │  │ Register NIXL       │   │   RDMA        │  │ Allocate tensors  │  │  │
│   │  │ Publish metadata    │   │ ═══════════►  │  │ Receive weights   │  │  │
│   │  └─────────────────────┘   │  112 Gbps     │  │ Reconstruct full  │  │  │
│   │         │                   │               │  │ Save safetensors  │  │  │
│   │         ▼                   │               │  └───────────────────┘  │  │
│   │  Serve weights (keep alive) │               │         │               │  │
│   │                             │               │         ▼               │  │
│   │                             │               │  TRT-LLM Inference      │  │
│   │                             │               │  (Main Container)       │  │
│   └─────────────────────────────┘               └─────────────────────────┘  │
│                                                                               │
│                        ┌──────────────────────┐                              │
│                        │   ModelExpress       │                              │
│                        │   Server (gRPC)      │                              │
│                        │   + Redis            │                              │
│                        └──────────────────────┘                              │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **HuggingFace mode (not checkpoint)** | TRT-LLM PyTorch backend loads HF directly; no conversion needed |
| **Init container for P2P** | Separates NIXL environment from TRT-LLM runtime |
| **Full weight reconstruction** | TRT-LLM expects full HF format, not TP-sharded weights |
| **CPU copy after receive** | Prevents CUDA context issues when switching devices |
| **Shape in proto** | Required for proper tensor allocation on target |

### 3.3 Component Responsibilities

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `MxTrtllmSourcePublisher` | `trtllm_loader.py` | Load HF model, shard, register NIXL, publish |
| `MxTrtllmTargetLoader` | `trtllm_loader.py` | Query, receive, reconstruct, save |
| `NixlTransferManager` | `nixl_transfer.py` | NIXL agent lifecycle, RDMA operations |
| `P2pService` | `p2p_service.rs` | gRPC metadata coordination |
| `TensorRecord` | `state.rs` | Tensor metadata with shapes |

---

## 4. TRT-LLM Stack Integration Points

### 4.1 TRT-LLM Components Used

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRT-LLM Architecture                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        LLM Class (High-Level API)                     │   │
│  │                     tensorrt_llm/llm.py                               │   │
│  │  ════════════════════════════════════════════════════════════════    │   │
│  │  ▲ POC USES: LLM(model=path, backend="pytorch")                       │   │
│  │  - Loads model from local HuggingFace format directory               │   │
│  │  - Uses PyTorch backend (no engine build required)                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     PyTorch Backend (v0.17+)                          │   │
│  │                tensorrt_llm/models/                                   │   │
│  │  ════════════════════════════════════════════════════════════════    │   │
│  │  ▲ POC USES: Direct HuggingFace model loading                        │   │
│  │  - Loads config.json + model.safetensors                              │   │
│  │  - No checkpoint conversion needed                                    │   │
│  │  - Applies TP sharding internally                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Inference Runtime                                 │   │
│  │  ════════════════════════════════════════════════════════════════    │   │
│  │  - KV Cache management                                                │   │
│  │  - MPI-based multi-GPU coordination                                   │   │
│  │  - Sampling (top-k, top-p, temperature)                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Specific TRT-LLM APIs Used

| API | Location | Purpose |
|-----|----------|---------|
| `tensorrt_llm.LLM` | `tensorrt_llm/llm.py` | High-level inference API |
| `LLM(backend="pytorch")` | PyTorch backend mode | Skip TensorRT engine build |
| `SamplingParams` | `tensorrt_llm/sampling_params.py` | Generation configuration |
| `llm.generate()` | Main inference API | Text generation |

### 4.3 Files We Interact With (HuggingFace Format)

```
/shared/mx_trtllm/checkpoint/
├── config.json           # HF model config (copied from source PVC)
├── model.safetensors     # Reconstructed full weights (141 GB)
├── tokenizer.json        # Tokenizer (copied from source PVC)
└── tokenizer_config.json # Tokenizer config (copied from source PVC)
```

### 4.4 What We Don't Use (Bypassed)

| Component | Reason |
|-----------|--------|
| `trtllm-build` | PyTorch backend doesn't need TensorRT engine |
| Checkpoint conversion | Direct HF loading eliminates conversion step |
| `IRefitter` API | No engine to refit |
| `rank*.safetensors` | We reconstruct full model, not per-rank shards |

---

## 5. Step-by-Step P2P Transfer Workflow

### Phase 1: Source Initialization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: SOURCE INITIALIZATION (MxTrtllmSourcePublisher.initialize)         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 1.1: Load HuggingFace Config                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ with open(hf_model_path / "config.json") as f:                         │ │
│  │     hf_config = json.load(f)                                           │ │
│  │ # Extract: architecture, hidden_size, num_layers, vocab_size           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Step 1.2: For each TP rank (0 to tp_size-1)                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ a) Load safetensors to GPU:                                            │ │
│  │    weights = safetensors.torch.load_file(path, device=f"cuda:{rank}")  │ │
│  │                                                                         │ │
│  │ b) Shard tensors for this rank:                                        │ │
│  │    - q_proj, k_proj, v_proj: shard dim=-1 (column parallel)            │ │
│  │    - o_proj, down_proj: shard dim=0 (row parallel)                     │ │
│  │    - embed_tokens, layer_norm: no sharding (replicated)                │ │
│  │                                                                         │ │
│  │ c) Initialize NIXL agent for this rank:                                │ │
│  │    nixl_manager = NixlTransferManager(                                 │ │
│  │        agent_name=f"trtllm-hf-source-rank{rank}-{uuid}",              │ │
│  │        device_id=rank                                                  │ │
│  │    )                                                                   │ │
│  │    nixl_manager.initialize()  # Creates NIXL agent with UCX backend    │ │
│  │                                                                         │ │
│  │ d) Register tensors with NIXL:                                         │ │
│  │    nixl_manager.register_tensors(weights)  # VRAM memory registration  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Step 1.3: Publish Metadata to MX Server                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ For each tensor:                                                       │ │
│  │   TensorDescriptor(                                                    │ │
│  │     name="model.layers.0.self_attn.q_proj.weight",                     │ │
│  │     addr=tensor.data_ptr(),      # GPU memory address                  │ │
│  │     size=tensor.numel() * elem_size,  # Size in bytes                  │ │
│  │     device_id=rank,                                                    │ │
│  │     dtype="torch.bfloat16",                                            │ │
│  │     shape=[1024, 8192],          # ← CRITICAL: Shape for reconstruction│ │
│  │   )                                                                    │ │
│  │                                                                         │ │
│  │ gRPC: stub.PublishMetadata(model_name, workers=[...])                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Step 1.4: Keep Running (serve weights)                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ while True:                                                            │ │
│  │     time.sleep(3600)  # Weights remain in GPU memory for transfers     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Target P2P Transfer (Init Container)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: TARGET P2P TRANSFER (MxTrtllmTargetLoader.load)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 2.1: Query MX Server for Source                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ while not found:                                                       │ │
│  │     response = stub.GetMetadata(model_name)                            │ │
│  │     if response.found and len(response.workers) > 0:                   │ │
│  │         break                                                          │ │
│  │     time.sleep(30)                                                     │ │
│  │                                                                         │ │
│  │ # Receive: WorkerMetadata for each rank (nixl_metadata, tensors)       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Step 2.2: For each rank in source workers                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ a) Initialize target NIXL agent:                                       │ │
│  │    nixl_manager = NixlTransferManager(                                 │ │
│  │        agent_name=f"trtllm-target-rank{rank}-{uuid}",                 │ │
│  │        device_id=rank                                                  │ │
│  │    )                                                                   │ │
│  │                                                                         │ │
│  │ b) Allocate GPU tensors with correct shapes:                           │ │
│  │    for t in worker.tensors:                                            │ │
│  │        shape = tuple(t.shape)  # Use shape from proto!                 │ │
│  │        weights[t.name] = torch.empty(shape, dtype, device=f"cuda:{r}") │ │
│  │                                                                         │ │
│  │ c) Register allocated tensors with NIXL:                               │ │
│  │    nixl_manager.register_tensors(weights)                              │ │
│  │                                                                         │ │
│  │ d) Add remote source agent:                                            │ │
│  │    nixl_manager.add_remote_agent(worker.nixl_metadata)                 │ │
│  │                                                                         │ │
│  │ e) Initiate RDMA transfer:                                             │ │
│  │    # NIXL pulls data from source GPU to target GPU                     │ │
│  │    # Zero-copy over InfiniBand (no CPU involvement)                    │ │
│  │    nixl_manager.receive_from_source(source_tensors)                    │ │
│  │                                                                         │ │
│  │ f) Copy to CPU immediately (avoid CUDA context issues):                │ │
│  │    cpu_weights = {k: v.cpu() for k, v in weights.items()}              │ │
│  │    all_rank_weights[rank] = cpu_weights                                │ │
│  │                                                                         │ │
│  │ g) Shutdown NIXL for this rank:                                        │ │
│  │    nixl_manager.shutdown()                                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Step 2.3: Reconstruct Full HuggingFace Weights                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ # all_rank_weights = {rank: {name: tensor, ...}, ...}                  │ │
│  │                                                                         │ │
│  │ for tensor_name in all_rank_weights[0].keys():                         │ │
│  │     shards = [all_rank_weights[r][tensor_name] for r in ranks]         │ │
│  │                                                                         │ │
│  │     # Determine concat dimension based on naming pattern               │ │
│  │     if "q_proj" or "k_proj" or "v_proj" or "up_proj" in name:          │ │
│  │         concat_dim = -1  # Column parallel                             │ │
│  │     elif "o_proj" or "down_proj" in name:                              │ │
│  │         concat_dim = 0   # Row parallel                                │ │
│  │     else:                                                              │ │
│  │         # Not sharded (embed_tokens, layer_norm)                       │ │
│  │         full_weights[name] = shards[0]                                 │ │
│  │         continue                                                       │ │
│  │                                                                         │ │
│  │     full_weights[name] = torch.cat(shards, dim=concat_dim)             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Step 2.4: Save HuggingFace Format                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ from safetensors.torch import save_file                                │ │
│  │ save_file(full_weights, "/shared/mx_trtllm/checkpoint/model.safetensors")│
│  │ # Result: 141.11 GB single file                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Step 2.5: Copy Config Files from Source PVC                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ cp /models/config.json /shared/mx_trtllm/checkpoint/                   │ │
│  │ cp /models/tokenizer.json /shared/mx_trtllm/checkpoint/                │ │
│  │ cp /models/tokenizer_config.json /shared/mx_trtllm/checkpoint/         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 3: TRT-LLM Inference (Main Container)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: TRT-LLM INFERENCE                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 3.1: Load Model with PyTorch Backend                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ from tensorrt_llm import LLM, SamplingParams                           │ │
│  │                                                                         │ │
│  │ llm = LLM(                                                             │ │
│  │     model="/shared/mx_trtllm/checkpoint",  # P2P transferred weights   │ │
│  │     tensor_parallel_size=8,                                            │ │
│  │     # backend="pytorch" is default when loading HF format              │ │
│  │ )                                                                      │ │
│  │ # [1/2] Loading HF model to memory: 4.7s                               │ │
│  │ # [2/2] Building TRT-LLM engine: ~113s                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Step 3.2: Run Inference                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ outputs = llm.generate(                                                │ │
│  │     ["Hello, I am a large language model trained by"],                 │ │
│  │     SamplingParams(max_tokens=50, temperature=0.7)                     │ │
│  │ )                                                                      │ │
│  │                                                                         │ │
│  │ # Output: "Meta AI. I can provide information and entertainment..."    │ │
│  │ # TPS: 15.9 tokens/sec                                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Ideal Long-Term Design

### 6.1 Target State Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    IDEAL LONG-TERM DESIGN                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     TRT-LLM with Native P2P Support                    │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Custom ModelLoader: MxP2PModelLoader                              │ │ │
│  │  │ - Registered as: --load-format mx-p2p                             │ │ │
│  │  │ - Integrated with TRT-LLM's weight loading pipeline               │ │ │
│  │  │ - Handles NIXL transfer + TRT-LLM weight injection               │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │                                │                                       │ │
│  │                                ▼                                       │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Weight Streaming (In-Flight Transfer)                            │ │ │
│  │  │ - Stream weights directly to GPU during model initialization     │ │ │
│  │  │ - No intermediate disk storage                                    │ │ │
│  │  │ - Overlapped transfer + weight processing                         │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │                                │                                       │ │
│  │                                ▼                                       │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Multi-Source Fan-Out                                              │ │ │
│  │  │ - Multiple warm instances serve weights                           │ │ │
│  │  │ - Load balancing across sources                                   │ │ │
│  │  │ - Fault tolerance with automatic failover                         │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     IRefitter-Based Hot Reload                         │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ For pre-built weight-stripped engines:                            │ │ │
│  │  │ 1. Distribute .engine file (small, ~100MB)                        │ │ │
│  │  │ 2. P2P transfer weights to GPU                                    │ │ │
│  │  │ 3. IRefitter.set_weights() directly from GPU memory               │ │ │
│  │  │ 4. No disk I/O, no safetensors serialization                      │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     GDS/GMS Integration                                │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ GPU Direct Storage (GDS) + GPU Memory Server (GMS):               │ │ │
│  │  │ - Direct storage-to-GPU loading (bypass CPU)                      │ │ │
│  │  │ - Network-attached GPU memory pools                               │ │ │
│  │  │ - Sub-100ms model loading for cached models                       │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Comparison: Current vs Ideal

| Aspect | Current POC | Ideal Long-Term |
|--------|-------------|-----------------|
| **Integration** | External (init container) | Native TRT-LLM loader |
| **Disk I/O** | Save safetensors, TRT-LLM reloads | Zero disk I/O, direct to GPU |
| **Weight format** | Reconstruct full HF | Stream TP shards directly |
| **Coordination** | External gRPC server | Built-in service mesh |
| **Multi-source** | Single source only | Fan-out from multiple sources |
| **Engine reload** | Full model reload | IRefitter hot reload |

### 6.3 TRT-LLM Upstream Changes Needed

| Change | Priority | Complexity |
|--------|----------|------------|
| Custom ModelLoader registration | High | Medium |
| Direct weight injection API | High | High |
| NIXL integration in weight pipeline | Medium | Medium |
| IRefitter GPU-to-GPU support | Medium | Medium |
| Built-in metadata coordination | Low | Low |

---

## 7. Potential Optimizations

### 7.1 Short-Term Optimizations

#### A. Eliminate Disk Round-Trip
```python
# Current: Save to disk, TRT-LLM reloads
safetensors.save_file(weights, "model.safetensors")  # Disk write
llm = LLM(model="/path/to/checkpoint")                # Disk read

# Optimized: Direct weight injection
# Requires TRT-LLM modification to accept pre-loaded tensors
llm = LLM(preloaded_weights=weights)  # GPU memory only
```

**Expected Gain**: Save 141 GB disk write + read = ~60s saved

#### B. Parallel Reconstruction
```python
# Current: Sequential reconstruction
for name in tensor_names:
    full_weights[name] = torch.cat(shards, dim=concat_dim)

# Optimized: Parallel with thread pool
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(reconstruct_tensor, name, shards) 
               for name, shards in tensor_groups]
```

**Expected Gain**: Reconstruction 5s → 1s

#### C. Async Weight Streaming
```python
# Current: Transfer all, then reconstruct
for rank in ranks:
    receive_rank(rank)
reconstruct_all()

# Optimized: Pipeline transfer + reconstruction
async def stream_and_reconstruct():
    async for rank, weights in receive_ranks_async():
        await queue.put((rank, weights))
    
    async for name in reconstruction_order:
        shards = await collect_shards(name)
        yield name, torch.cat(shards)
```

**Expected Gain**: Overlap transfer + reconstruction = ~3s saved

### 7.2 Medium-Term Optimizations

#### D. Engine Cache with P2P Update
```
┌──────────────────────────────────────────────────────────────────┐
│ First Load: Build engine once, cache on shared storage           │
│ Subsequent Loads: P2P transfer + IRefitter weight injection      │
├──────────────────────────────────────────────────────────────────┤
│ First Load:                                                      │
│   P2P Transfer (12s) + Engine Build (113s) = 125s                │
│                                                                  │
│ Subsequent Loads:                                                │
│   P2P Transfer (12s) + IRefitter (5s) = 17s                      │
├──────────────────────────────────────────────────────────────────┤
│ Speedup: 7x for repeat loads                                     │
└──────────────────────────────────────────────────────────────────┘
```

#### E. Multi-Source Transfer
```
         Source A (GPU 0-3)        Source B (GPU 4-7)
                │                        │
                ▼                        ▼
    ┌───────────────────────────────────────────────┐
    │              Target                            │
    │   GPU 0-3 receive from A                       │
    │   GPU 4-7 receive from B                       │
    │   Each pair at 112 Gbps = 224 Gbps total      │
    └───────────────────────────────────────────────┘

Speedup: 170 GB / 224 Gbps = 6s (vs 12s from single source)
```

### 7.3 Long-Term Optimizations

#### F. Zero-Copy Weight Injection
```cpp
// Bypass safetensors serialization entirely
// Requires TRT-LLM native NIXL support

// After P2P transfer, weights are in GPU memory at known addresses
// Directly construct TRT-LLM weight objects from GPU pointers
tensorrt_llm::Weights createWeightsFromGpuPtr(void* gpu_ptr, size_t size);

// Feed directly to model initialization
model.loadWeightsFromGpu(weight_map);
```

#### G. GMS (GPU Memory Server) Integration
```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU Memory Server                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Persistent GPU memory pool (network-attached)            │    │
│  │ - Pre-loaded model weights                               │    │
│  │ - Shared across cluster                                  │    │
│  │ - Sub-100ms access                                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Target Instance                                          │    │
│  │ - Maps remote GPU memory                                 │    │
│  │ - Zero-copy model loading                                │    │
│  │ - Instant model availability                             │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Key Code Components

### 8.1 Proto Definition (Shape Preservation)

```protobuf
// modelexpress_common/proto/p2p.proto
message TensorDescriptor {
  string name = 1;
  uint64 addr = 2;           // GPU memory address
  uint64 size = 3;           // Size in bytes
  uint32 device_id = 4;      // GPU device ID
  string dtype = 5;          // Data type (e.g., "torch.bfloat16")
  repeated int64 shape = 6;  // ← CRITICAL: Tensor shape for reconstruction
}
```

### 8.2 Server-Side Shape Handling

```rust
// modelexpress_server/src/state.rs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorRecord {
    pub name: String,
    pub addr: u64,
    pub size: u64,
    pub device_id: u32,
    pub dtype: String,
    #[serde(default)]  // Backward compatibility
    pub shape: Vec<i64>,
}
```

### 8.3 Source Publisher (Shape Inclusion)

```python
# trtllm_loader.py - MxTrtllmSourcePublisher._publish_to_mx_server()
tensor_protos.append(p2p_pb2.TensorDescriptor(
    name=name,
    addr=tensor.data_ptr(),
    size=tensor.numel() * tensor.element_size(),
    device_id=rank,
    dtype=str(tensor.dtype),
    shape=list(tensor.shape),  # ← Preserve shape!
))
```

### 8.4 Target Loader (Shape Usage)

```python
# trtllm_loader.py - MxTrtllmTargetLoader._allocate_tensors()
for t in tensor_protos:
    if t.shape and len(t.shape) > 0:
        shape = tuple(t.shape)  # ← Use shape from proto
    else:
        # Legacy fallback
        numel = t.size // dtype_size
        shape = (numel,)
    
    weights[t.name] = torch.empty(shape, dtype=dtype, device=f"cuda:{device_id}")
```

### 8.5 Weight Reconstruction

```python
# trtllm_loader.py - MxTrtllmTargetLoader._reconstruct_full_weights()
col_parallel_patterns = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
row_parallel_patterns = ["o_proj", "down_proj"]

for name in rank0_weights.keys():
    shards = [all_rank_weights[r][name] for r in sorted(ranks)]
    
    concat_dim = None
    for pattern in col_parallel_patterns:
        if pattern in name.lower():
            concat_dim = -1
            break
    if concat_dim is None:
        for pattern in row_parallel_patterns:
            if pattern in name.lower():
                concat_dim = 0
                break
    
    if concat_dim is not None:
        full_weights[name] = torch.cat(shards, dim=concat_dim)
    else:
        full_weights[name] = shards[0]  # Not sharded
```

---

## 9. Lessons Learned

### 9.1 Critical Bug: CUDA Context Switching

**Problem**: Received GPU tensors became invalid when switching CUDA contexts between ranks.

**Symptom**: `Reconstructed 0 full tensors (0.00 GB)`

**Root Cause**: GPU tensors stored in `all_rank_weights[rank]` were invalidated when `nixl_manager.shutdown()` released the CUDA context.

**Fix**: Copy to CPU immediately after receive:
```python
# WRONG
all_rank_weights[rank] = weights  # GPU tensors

# CORRECT
cpu_weights = {k: v.cpu() for k, v in weights.items()}
all_rank_weights[rank] = cpu_weights  # CPU tensors persist
```

### 9.2 Proto Shape Field

**Problem**: Target allocated flat 1D tensors because shape information was lost.

**Symptom**: `ValueError: Parameter has invalid shape torch.Size([131334144]) compared with expected shape (16032, 8192)`

**Fix**: Added `repeated int64 shape = 6;` to `TensorDescriptor` proto, updated Rust server and Python client.

### 9.3 TRT-LLM Model Compatibility

**Problem**: DeepSeek-V3 architecture not supported by TRT-LLM LLM API.

**Symptom**: `The given huggingface model architecture DeepseekV3ForCausalLM is not supported`

**Fix**: Use supported models (Llama 70B) for POC validation. DeepSeek support requires custom model implementation.

### 9.4 Gated Models (HuggingFace)

**Problem**: Cannot download config files for gated models like Llama without authentication.

**Fix**: Copy config files from source PVC instead of downloading from HuggingFace Hub.

### 9.5 Init Container Strategy

**Benefit**: Separating P2P transfer (init container) from inference (main container) provides:
- Clean environment isolation
- Different base images (NIXL-enabled vs TRT-LLM official)
- Clear debugging (logs separated)
- Restart semantics (init must complete before main starts)

---

## References

1. [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
2. [TRT-LLM DeepSeek-R1 Blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html)
3. [NIXL GitHub](https://github.com/NVIDIA/nixl)
4. [ModelExpress P2P vLLM Implementation](../modelexpress_client/python/modelexpress/vllm_loader.py)
5. [TRT-LLM IRefitter API](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_refitter.html)
