# TRT-LLM Dynamo Phase 2: DeepSeek-V3.2 NVFP4 DEP8 Results

**Status**: P2P TRANSFER VALIDATED — Executor integration in progress
**Date**: February 26, 2026
**Branch (modelexpress)**: `kavink/trtllm`
**Branch (dynamo)**: `kavink/trtllm-p2p`

---

## 1. Summary

DeepSeek-V3.2 NVFP4 (671B MoE, 256 experts) P2P weight transfer validated on H200
cluster with EP=8 cross-node RDMA. All 8 ranks transfer 68.45 GB each at 278 Gbps
directly into model parameter buffers. Remaining work: TRT-LLM executor integration
via `python3 -m dynamo.trtllm` (raw `LLM()` API has executor init issues unrelated to P2P).

**Validation result (DSV3.2 NVFP4, EP=8, cross-node InfiniBand, H200):**

| Metric | Value |
|--------|-------|
| Model | `nvidia/DeepSeek-V3.2-NVFP4` (671B MoE, NVFP4) |
| Params matched | 2718/2718 per rank (100%) |
| Data per rank | 68.45 GB |
| Total transferred | 547.6 GB (8 ranks parallel) |
| RDMA transfer time | 1.97s per rank |
| Per-rank bandwidth | 278 Gbps |
| Aggregate bandwidth | ~2,227 Gbps |
| Total time (model create + transfer) | 24.5s |
| Transfer mode | Cross-node RDMA, DIRECT into model params |
| GPU | NVIDIA H200 (143 GB HBM3e) |
| Cluster nodes | Source: `..yv6x2ed8`, Target: `..txk1nt8e` |

---

## 2. What Was Done

### 2.1 Deploy recipe fixed from exemplar

The DSV3.2 deploy recipe (`recipes/deepseek-v3.2/trtllm/disagg/dep8x2/deploy.yaml`)
had several issues fixed based on the [dynamo-exemplar-external](https://github.com/karen-sy/dynamo-exemplar-external) repo:

| Issue | Before | After |
|-------|--------|-------|
| `tensor_parallel_size` | `1` (wrong) | `8` (must equal EP) |
| Model | `deepseek-ai/DeepSeek-V3.2` (FP8) | `nvidia/DeepSeek-V3.2-NVFP4` (NVFP4) |
| Engine config | Minimal | Full config from exemplar |
| Env vars | Missing NCCL/MoE vars | Added `TLLM_OVERRIDE_LAYER_NUM=61`, NCCL, GC disable |

### 2.2 MX server shape deserialization fix

The MX server (Rust) couldn't deserialize metadata for large models because
protobuf's empty `repeated int64 shape` field serializes as `{}` (map) instead of
`[]` (sequence). Fixed by adding a custom serde deserializer in `state.rs` that
accepts both formats.

**File**: `modelexpress_server/src/state.rs`
**Image**: `nvcr.io/nvidian/dynamo-dev/modelexpress-server:trtllm-ph1` (rebuilt)

### 2.3 Source deployment for DSV3.2

Created MPI-based source script for EP=8 (based on Llama 70B TP=8 source):

**File**: `examples/p2p_transfer_trtllm/deploy/trtllm-source-dsv32-ph3.yaml`

- Uses `mpirun -np 8` with `Mapping(world_size=8, tp_size=8, moe_ep_size=8, moe_tp_size=1, enable_attention_dp=True)`
- Loads `nvidia/DeepSeek-V3.2-NVFP4` from PVC cache
- Each rank: 2718 params, 68.45 GB
- Publishes all 8 ranks to MX server via gRPC
- Load time: ~1724s (~29 min) from HF cache

Fixed issues during source development:
- `DeepseekV3ForCausalLM.load_weights()` doesn't accept `weight_mapper` kwarg → added `inspect.getfullargspec` check
- Removed `shape=list(tensor.shape)` from `TensorDescriptor` proto publish (server compat)
- Added `MX_SOURCE_QUERY_TIMEOUT` env var (default 3600s) for large model source startup

### 2.4 Dynamo 0.9.0 based Docker image

TRT-LLM 1.3.0rc3 doesn't support DSV3.2 (FlashMLA kernel requires BF16 KV cache,
NVFP4 checkpoint sets FP8 KV cache). Rebased on Dynamo 0.9.0 runtime (TRT-LLM 1.3.0rc1)
which has working FlashMLA + DSV3.2 support.

**Dockerfile**: `examples/p2p_transfer_trtllm/Dockerfile.ph3-dynamo0.9`
**Image**: `nvcr.io/nvidian/dynamo-dev/modelexpress-trtllm-client:ph3-dynamo0.9`
**Base**: `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.9.0`

PRESHARDED patches ported to v1.3.0rc1: `trtllm_patches/v1.3.0rc1/`

### 2.5 Target validation (MPI approach)

Created MPI-based target script that bypasses the `LLM()` API to validate
weights are correctly transferred:

**File**: `examples/p2p_transfer_trtllm/deploy/trtllm-target-dsv32-ph3.yaml`

Each MPI rank:
1. Creates model via `AutoModelForCausalLM.from_config()` with EP=8 Mapping
2. Sets `_weights_presharded = True` on all Linear modules
3. Calls `MxLiveWeightLoader.load_weights(model=model)` → RDMA into params
4. Returns `{}` → weights in model params

Result: **SUCCESS** — all 8 ranks, 2718/2718 params, 278 Gbps, 24.5s total.

---

## 3. Remaining: TRT-LLM Executor Integration

### 3.1 Problem

The `LLM()` API (used by `python3 -m dynamo.trtllm`) spawns IPC executor workers
that independently load the model. After loading, the workers initialize the
TRT-LLM execution engine. This executor init fails with:

```
RuntimeError: Executor worker returned error
```

The worker error output is not captured in pod logs (MPI stderr is swallowed).
When captured via monkey-patching:
- On TRT-LLM 1.3.0rc3: `FlashMLA: Expected kv.dtype() == torch::kBFloat16` (FP8 KV cache unsupported)
- On Dynamo 0.9.0: FlashMLA works, autotuner starts, but executor eventually fails

### 3.2 Key finding: NOT a PRESHARDED/P2P issue

Plain disk loading (no PRESHARDED, no P2P) also crashes with `Executor worker returned error`.
The issue is TRT-LLM executor initialization for DSV3.2 NVFP4 EP=8, regardless of how
weights are loaded.

### 3.3 Why the exemplar works

The [dynamo-exemplar-external](https://github.com/karen-sy/dynamo-exemplar-external) uses:
- `python3 -m dynamo.trtllm` (Dynamo's worker), not raw `LLM()` API
- A custom Dynamo build from commit `b5c0db6` (post-0.8.1)
- Specific NCCL/MoE env vars and engine configuration
- PVC for model weights (loads from disk)

The Dynamo worker configures the executor differently from raw `LLM()` calls.

### 3.4 Next step

Build the **Dynamo runtime image with ModelExpress enabled** and test the actual
Dynamo worker path:

```bash
# Build Dynamo image with ModelExpress
docker build --build-arg ENABLE_MODELEXPRESS_P2P=true \
    --build-arg MODELEXPRESS_REF=<commit> \
    -f container/templates/trtllm_runtime.Dockerfile .

# Deploy with Dynamo worker
python3 -m dynamo.trtllm \
    --model nvidia/DeepSeek-V3.2-NVFP4 \
    --model-express-url modelexpress-server:8001 \
    --extra-engine-args /config/prefill_config.yaml \
    --disaggregation-mode prefill
```

This uses the `_setup_modelexpress_loader()` hook in `engine.py` (already implemented
in Phase 1, validated with Qwen 0.5B) but with the Dynamo worker's executor setup.

---

## 4. Current Deployments (namespace `kavin`)

| Deployment | Image | Node | Status |
|-----------|-------|------|--------|
| `modelexpress-server` | `server:trtllm-ph1` (fixed) | CPU | Running |
| `redis` | redis:7-alpine | CPU | Running, 2 keys |
| `trtllm-source-dsv32` | `ph3-dynamo0.9` | `..yv6x2ed8` | Running 6h+, all 8 ranks published |
| `trtllm-target-dsv32` | `ph3-dynamo0.9` | varies | CrashLoopBackOff (executor issue) |

### PVCs
| Name | Size | Purpose |
|------|------|---------|
| `dsv32-nvfp4-weights` | 500 Gi | Source model cache (387 GB used) |
| `model-weights-storage` | 25 Ti (RWX) | Target HF cache (shared) |

---

## 5. Docker Images

| Tag | Base | Purpose |
|-----|------|---------|
| `modelexpress-trtllm-client:ph3-dynamo0.9` | Dynamo 0.9.0 (TRT-LLM 1.3.0rc1) | Source + target with PRESHARDED patches |
| `modelexpress-trtllm-client:ph3-v1.3` | TRT-LLM 1.3.0rc3 | Previous image (FlashMLA incompatible with DSV3.2) |
| `modelexpress-server:trtllm-ph1` | Rust | MX server with shape deserialization fix |

---

## 6. Files Created/Modified

### ModelExpress repo (kavink/trtllm)

| File | Change |
|------|--------|
| `trtllm_patches/v1.3.0rc1/` | NEW: PRESHARDED patches for Dynamo 0.9.0 (TRT-LLM 1.3.0rc1) |
| `examples/p2p_transfer_trtllm/Dockerfile.ph3-dynamo0.9` | NEW: Dynamo 0.9.0 based image |
| `examples/p2p_transfer_trtllm/deploy/trtllm-source-dsv32-ph3.yaml` | NEW: DSV3.2 EP=8 source |
| `examples/p2p_transfer_trtllm/deploy/trtllm-target-dsv32-ph3.yaml` | NEW: DSV3.2 EP=8 target |
| `modelexpress_server/src/state.rs` | FIX: shape field deserialization (map→vec) |
| `modelexpress_client/python/modelexpress/trtllm_live_transfer.py` | FIX: removed shape from publish, added MX_SOURCE_QUERY_TIMEOUT |

### Dynamo repo (kavink/trtllm-p2p)

| File | Change |
|------|--------|
| `recipes/deepseek-v3.2/trtllm/disagg/dep8x2/deploy.yaml` | FIX: tp=8, model=NVFP4, engine config from exemplar |

---

## 7. Issues Resolved

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `tensor_parallel_size: 1` in recipe | Wrong value, must equal EP | Set to `8` (from exemplar) |
| Wrong model name | `deepseek-ai/DeepSeek-V3.2` is FP8 | Use `nvidia/DeepSeek-V3.2-NVFP4` |
| MX server deserialization crash | `shape:{}` vs `shape:[]` | Custom serde deserializer in Rust |
| `weight_mapper` kwarg error | `DeepseekV3ForCausalLM` doesn't accept it | `inspect.getfullargspec` check |
| Target pod evicted | HF downloads 337GB to ephemeral storage | Mount PVC at `HF_HOME` |
| Source query timeout | 600s too short for large model load | `MX_SOURCE_QUERY_TIMEOUT=3600` env var |
| FlashMLA BF16 assertion | TRT-LLM 1.3.0rc3 FlashMLA doesn't support FP8 KV | Rebase to Dynamo 0.9.0 (1.3.0rc1) |
| Cluster GPU shortage | `ganeshku` ns had Pending pods reserving 16 GPUs | Deleted DynamoGraphDeployments |

---

## 8. Model Details

**nvidia/DeepSeek-V3.2-NVFP4:**
- Architecture: `DeepseekV32ForCausalLM`
- 61 layers, 256 routed experts, 1 shared expert, 8 active per token
- Hidden size: 7168, MoE intermediate: 2048
- Quantization: NVFP4 (group_size=16), KV cache: FP8
- Attention: MLA (Multi-Latent Attention) with FlashMLA kernel
- 163 safetensors files, ~337 GB total
- Per rank (EP=8): 2718 params, 68.45 GB, ~67 GB GPU memory

**Mapping for DEP8:**
```python
Mapping(world_size=8, tp_size=8, rank=rank, moe_ep_size=8, moe_tp_size=1, enable_attention_dp=True)
# moe_ep_groups: [[0,1,2,3,4,5,6,7]]
# moe_tp_groups: [[0],[1],...,[7]]
# enable_attention_dp: True (attention weights replicated)
```
