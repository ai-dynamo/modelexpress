# TRT-LLM Dynamo Worker: P2P Weight Transfer Integration Plan

## Target Workflow
**FP4 - DEP8x2 Prefill DEP8x2 Decode, DeepSeek-V3.2**

- Model: `deepseek-ai/DeepSeek-V3.2` (671B MoE, 37B active)
- Quantization: FP4 (W4A8 via TRT-LLM)
- Prefill: DEP8x2 (8 expert parallel x 2 nodes = 16 GPUs)
- Decode: DEP8x2 (8 expert parallel x 2 nodes = 16 GPUs)
- Transfer: GPU-to-GPU via NIXL/RDMA (no PVC required for targets)

## Background

### What exists today

**vLLM in Dynamo (PR #6186):**
- `--load-format mx-source` / `--load-format mx-target` in `dynamo.vllm`
- `--model-express-url` CLI arg / `MODEL_EXPRESS_URL` env var
- Registers ModelExpress loaders via `register_modelexpress_loaders()`
- Sets worker class to `ModelExpressWorker`
- Dockerfile build arg `ENABLE_MODELEXPRESS_P2P`

**ModelExpress TRT-LLM POC (branch kavink/trtllm):**
- `trtllm_loader.py`: Source loads HF weights, shards for TP, registers with NIXL
- `trtllm_checkpoint_loader.py`: Integrated checkpoint loader (`checkpoint_format="mx-p2p"`)
- `trtllm_live_transfer.py`: Live param-to-param transfer from running model
- Validated with Llama 70B: ~12s transfer, ~112 Gbps

**Dynamo TRT-LLM Worker (current):**
- Entry: `python3 -m dynamo.trtllm`
- Weight loading: `TensorRTLLMEngine.initialize()` calls `LLM(**engine_args)` which loads from disk
- Disagg: `--disaggregation-mode prefill|decode` with `--extra-engine-args` YAML
- DeepSeek-R1 recipe exists: DEP4 prefill + DEP32 decode (GB200)
- No P2P hooks, no ModelExpress integration

### Key Differences: vLLM vs TRT-LLM Integration

| Aspect | vLLM | TRT-LLM (Phase 3) |
|--------|------|---------|
| Loader plugin | `--load-format mx-source/mx-target` | `checkpoint_loader=MxLiveCheckpointLoader()` |
| Weight injection | Custom `ModelLoader` class | `LoadFormat.PRESHARDED` + empty dict return |
| Worker override | `engine_args.worker_cls` | Not needed (PRESHARDED handles it) |
| Transfer format | Raw GPU tensors (pre-FP8) | Live GPU params (fused, TP-sharded, TRT-LLM format) |
| Disk round-trip | None (direct GPU injection) | **None** (direct GPU param-to-param RDMA) |
| TRT-LLM patches needed | None | 3 files (~30 lines): `llm_args.py`, `model_loader.py`, `linear.py` |

## Architecture

```
                    ModelExpress Server (CPU, gRPC)
                    Stores metadata + ready signals + config files
                           |
              +------------+------------+
              |                         |
    Source Instance              Target Instance(s)
    (1 node, 8 GPUs)           (N nodes, 8 GPUs each)
              |                         |
    MxLiveSource (MPI)         Dynamo TRT-LLM Worker
    - Load model via TRT-LLM   - --model-express-url flag
    - Params already fused      - MxLiveCheckpointLoader
      and TP-sharded           - Creates empty model on GPU
    - Register GPU params       - RDMA directly into params
      with NIXL                 - return {} (skip load_weights)
    - Publish metadata          - Engine compile + KV cache
              |                         |
              +--- NIXL/RDMA -----------+
              GPU params -> GPU params (zero-copy)
```

## Integration Plan

### Phase 1: CLI + Engine Hook (Minimal, mirrors vLLM pattern)

**Goal:** Add `--model-express-url` to `dynamo.trtllm` and integrate the checkpoint loader.

#### 1.1 Add CLI argument

File: `components/src/dynamo/trtllm/backend_args.py`

```python
parser.add_argument(
    "--model-express-url",
    type=str,
    default=os.environ.get("MODEL_EXPRESS_URL"),
    help="ModelExpress P2P server URL (e.g., modelexpress-server:8001). "
         "When set, weights are received via RDMA instead of loaded from disk.",
)
```

#### 1.2 Hook into engine initialization

File: `components/src/dynamo/trtllm/engine.py`

Before `self._llm = self._llm_cls(**self.engine_args)`, check if P2P is enabled:

```python
if model_express_url:
    from modelexpress.trtllm_live_transfer import MxLiveCheckpointLoader
    from tensorrt_llm.llmapi import LoadFormat
    loader = MxLiveCheckpointLoader()
    self.engine_args["checkpoint_loader"] = loader
    self.engine_args["load_format"] = LoadFormat.PRESHARDED
```

This uses Phase 3's `MxLiveCheckpointLoader` which:
- Receives weights via NIXL RDMA directly into target model parameter buffers
- Returns `{}` (empty dict) to skip `model.load_weights()` entirely
- Zero disk I/O, zero format conversion, zero weight mapper fusing
- Requires `LoadFormat.PRESHARDED` and 3 TRT-LLM patches (see `trtllm_patches/`)
- Validated at 141 GB in ~3s (368 Gbps aggregate) for Llama 70B TP=8

#### 1.3 Source deployment

The source is a standalone ModelExpress process (not a dynamo worker):
- Loads HF weights, shards for the target's TP/EP config
- Registers with NIXL, publishes metadata to MX server
- Stays running to serve multiple targets

#### 1.4 Files to modify

| File | Change |
|------|--------|
| `components/src/dynamo/trtllm/backend_args.py` | Add `--model-express-url` arg |
| `components/src/dynamo/trtllm/args.py` | Pass `model_express_url` through config |
| `components/src/dynamo/trtllm/engine.py` | Hook checkpoint loader before `LLM()` init |
| `components/src/dynamo/trtllm/workers/llm_worker.py` | Pass MX URL to engine |
| `container/templates/trtllm_runtime.Dockerfile` | Optional ModelExpress install |
| `container/context.yaml` | Add `enable_modelexpress_p2p` for TRT-LLM |

### Phase 2: DeepSeek-V3.2 DEP8x2 Support

**Goal:** Support the specific target workflow with FP4 quantization.

#### 2.1 DeepSeek-V3.2 Architecture Considerations

- 671B params, MoE with 256 experts (8 active)
- DEP8x2 = expert parallel 8 across 2 nodes (16 GPUs per role)
- FP4 quantization: weights are W4A8 format
- TRT-LLM PyTorch backend (not engine-compiled)

#### 2.2 Source Sharding for DEP8x2

The source must shard weights to match the target's parallelism:
- **Expert-parallel layers**: Each of 256 experts assigned to EP ranks
- **Attention layers**: Replicated or TP-sharded depending on config
- **Embeddings/norms**: Replicated across all ranks

Key: `moe_expert_parallel_size: 8` means each rank gets 256/8 = 32 experts.

Source sharding logic in `trtllm_loader.py` needs to handle:
1. MoE gate weights (replicated)
2. Expert weights (partitioned by EP rank)
3. Attention Q/K/V (column-sharded if TP > 1, replicated if TP=1 with EP)
4. FP4 quantized weights (transfer in quantized form, not raw BF16)

#### 2.3 FP4 Weight Transfer

Two approaches:

**A. Transfer pre-quantized FP4 weights** (preferred):
- Source loads FP4 checkpoint directly
- Transfers quantized weights (4x smaller than BF16)
- Target receives and passes to TRT-LLM as-is
- Requires source to have FP4 checkpoint available

**B. Transfer BF16 and quantize on target:**
- Source loads BF16, transfers raw
- Target receives, TRT-LLM quantizes to FP4
- 4x more data transferred, but simpler source

#### 2.4 Deployment Recipe

File: `recipes/deepseek-v3.2/trtllm/disagg/dep8x2/deploy.yaml`

```yaml
# ModelExpress Server (CPU-only)
modelexpress-server:
  replicas: 1
  image: modelexpress-server:latest
  resources:
    cpu: 4
    memory: 8Gi

# Source (1 node, 8 GPUs)
# Loads weights, serves via NIXL to prefill + decode targets
mx-source:
  replicas: 1
  image: modelexpress-trtllm-client:latest
  resources:
    gpu: 8
    rdma/ib: 8
  env:
    MODEL_NAME: "deepseek-ai/DeepSeek-V3.2"
    MODEL_EXPRESS_URL: "modelexpress-server:8001"
    MX_EXPECTED_WORKERS: "8"

# Prefill workers (2 nodes, 8 GPUs each = DEP8x2)
prefill:
  replicas: 2
  disaggregation-mode: prefill
  model-express-url: "modelexpress-server:8001"
  extra-engine-args:
    tensor_parallel_size: 1
    moe_expert_parallel_size: 8
    enable_attention_dp: true
    max_batch_size: 4
    max_num_tokens: 4608

# Decode workers (2 nodes, 8 GPUs each = DEP8x2)
decode:
  replicas: 2
  disaggregation-mode: decode
  model-express-url: "modelexpress-server:8001"
  extra-engine-args:
    tensor_parallel_size: 1
    moe_expert_parallel_size: 8
    enable_attention_dp: true
    max_batch_size: 256
    max_num_tokens: 256
```

### Phase 3: TRT-LLM Upstream Patches

**Goal:** Get the 3 TRT-LLM patches upstreamed.

The live transfer (Phase 3) is already validated and used by the dynamo integration.
It requires 3 small patches to TRT-LLM (~30 lines total), currently applied as Docker
overlay copies. These should be submitted as an upstream PR.

#### 3.1 Patches (see `trtllm_patches/` and `docs/TRTLLM_UPSTREAM_DIFF.md`)

1. **`llm_args.py`** (+4 lines): Add `LoadFormat.PRESHARDED = 3` enum value
2. **`model_loader.py`** (+20 lines): PRESHARDED branch that passes `model` ref to
   `load_weights()` and handles empty dict return (skip `model.load_weights()`)
3. **`linear.py`** (+6 lines x3): Skip TP slicing when `_weights_presharded = True`

#### 3.2 For Dynamo runtime image

Until upstreamed, the TRT-LLM Dockerfile needs to overlay these patches:
```dockerfile
COPY trtllm_patches/llm_args.py    /path/to/tensorrt_llm/llmapi/llm_args.py
COPY trtllm_patches/model_loader.py /path/to/tensorrt_llm/_torch/pyexecutor/model_loader.py
COPY trtllm_patches/linear.py      /path/to/tensorrt_llm/_torch/modules/linear.py
```

## Key Risks

| Risk | Mitigation |
|------|------------|
| DeepSeek-V3.2 not supported by TRT-LLM LLM API | Check TRT-LLM version, may need custom model impl |
| FP4 sharding complexity for MoE | Start with BF16 transfer + target-side quantization |
| Multi-node EP source sharding | Source can run DEP8 on single node, targets do cross-node EP |
| NIXL across different node topologies | UCX handles this, but test cross-rack RDMA bandwidth |
| Disk round-trip latency | Acceptable for Phase 1-2, eliminated in Phase 3 |

## Progress

### Phase 1: DONE
Committed on `kavink/trtllm-p2p` branch in dynamo repo (commit `14090fe1a`).

Files modified:
- `components/src/dynamo/trtllm/backend_args.py` - `--model-express-url` arg + config field
- `components/src/dynamo/trtllm/engine.py` - `_setup_modelexpress_loader()` hook
- `components/src/dynamo/trtllm/workers/llm_worker.py` - passes URL to engine
- `container/templates/trtllm_runtime.Dockerfile` - optional ModelExpress install
- `container/templates/args.Dockerfile` - build args for trtllm
- `container/context.yaml` - trtllm defaults for ModelExpress

### Phase 2: IN PROGRESS
Current `MxCheckpointLoader` supports:
- Single-rank and multi-rank TP transfers
- Dense TP sharding (q_proj, k_proj, v_proj column-parallel; o_proj, down_proj row-parallel)
- Full HF weight reconstruction from TP shards

Needs for DeepSeek-V3.2 DEP8x2:
- MoE expert-parallel sharding in source (`trtllm_loader.py`)
- MoE expert-parallel reconstruction in target (`trtllm_checkpoint_loader.py`)
- FP4 quantized weight handling (either transfer pre-quantized or BF16+requantize)
- DeepSeek-specific layer patterns (MLA attention, MoE gate routing, shared experts)
- Deployment recipe YAML

### Phase 3: NOT STARTED

## Timeline Estimate

| Phase | Scope | Effort | Status |
|-------|-------|--------|--------|
| Phase 1 | CLI + checkpoint loader hook | 2-3 days | DONE |
| Phase 2 | DeepSeek-V3.2 DEP8x2 recipe | 3-5 days | IN PROGRESS |
| Phase 3 | Zero-disk (upstream dependent) | TBD | NOT STARTED |

## References

- Dynamo vLLM P2P PR: https://github.com/ai-dynamo/dynamo/pull/6186
- ModelExpress vLLM PR: https://github.com/ai-dynamo/modelexpress/pull/140
- ModelExpress TRT-LLM POC: `docs/POC_TRTLLM.md`
- DeepSeek-R1 TRT-LLM recipe: `recipes/deepseek-r1/trtllm/disagg/`
- TRT-LLM disagg configs: `examples/backends/trtllm/engine_configs/deepseek-r1/`
- Dynamo trtllm-p2p branch: `kavink/trtllm-p2p`
