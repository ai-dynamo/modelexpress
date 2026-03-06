# TRT-LLM Zero-Copy P2P Integration Plan

This document is the engineering plan for achieving **true zero-copy GPU-to-GPU weight transfers** between TensorRT-LLM instances via ModelExpress + NIXL. It is based on deep analysis of both the [ModelExpress](https://github.com/ai-dynamo/modelexpress) and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) codebases.

---

## Current Status (Feb 17, 2026)

| Capability | Status |
|-----------|--------|
| TP=1 zero-disk loading (Qwen 0.5B) | **Working** — 151 Gbps, single container, no PVC |
| TP=8 RDMA transfer (Llama 70B) | **Working** — 85 Gbps/rank, 170 GB total |
| TP>1 zero-disk loading | **Blocked** — needs upstream pre-sharded weight loading |
| UCX/MPI conflict | **Resolved** — use system NIXL from NGC image |
| Config transfer (no PVC) | **Working** — `model_files` via gRPC |

**Next step**: TRT-LLM upstream PR for pre-sharded weight loading (Phase 2).

---

## Table of Contents

1. [Key Discovery: TRT-LLM Already Has a Plugin System](#1-key-discovery-trt-llm-already-has-a-plugin-system)
2. [Current POC vs Target Architecture](#2-current-poc-vs-target-architecture)
3. [Phase 1: Custom Checkpoint Loader — COMPLETE](#3-phase-1-custom-checkpoint-loader-no-upstream-changes)
4. [Phase 2: True Zero-Copy via Deferred NIXL Transfer](#4-phase-2-true-zero-copy-via-deferred-nixl-transfer-upstream)
5. [Phase 3: Direct Per-Rank Shard Injection — CRITICAL for TP>1](#5-phase-3-direct-per-rank-shard-injection-critical-for-tp1)
6. [Phase 4: Reuse TRT-LLM's NIXL Infrastructure](#6-phase-4-reuse-trt-llms-nixl-infrastructure)
7. [Performance Impact Analysis](#7-performance-impact-analysis)
8. [Upstream Change Proposals](#8-upstream-change-proposals)
9. [Implementation Roadmap (Revised)](#9-implementation-roadmap-revised)

---

## 1. Key Discovery: TRT-LLM Already Has a Plugin System

**This changes everything from our POC assumptions.** TRT-LLM's PyTorch backend (v0.17+) already has a checkpoint loader plugin system nearly identical to vLLM's `--load-format`:

### 1.1 The Registration Decorator

```python
# tensorrt_llm/_torch/models/modeling_utils.py
CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING = {}

def register_checkpoint_loader(name: str):
    def decorator(cls):
        CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING[name] = cls
        return cls
    return decorator
```

### 1.2 The LLM API Already Accepts Custom Formats

```python
# tensorrt_llm/llmapi/llm_args.py (TorchLlmArgs)
checkpoint_format: Optional[str] = Field(
    default=None,
    description="The format of the provided checkpoint. "
    "You may use a custom checkpoint format by subclassing "
    "`BaseCheckpointLoader` and registering it with `register_checkpoint_loader`."
)

checkpoint_loader: Optional[object] = Field(
    default=None,
    description="The checkpoint loader to use for this LLM instance. "
    "You may use a custom checkpoint loader by subclassing "
    "`BaseCheckpointLoader` and providing an instance of the subclass here."
)
```

### 1.3 What This Means

| POC Assumption | Reality |
|----------------|---------|
| "TRT-LLM needs a custom model loader plugin (P0 upstream)" | **Already exists** via `@register_checkpoint_loader` |
| "Need `LLM(preloaded_weights=...)` API (P0 upstream)" | **Already exists** via `checkpoint_loader=` parameter |
| "No way to inject custom loading logic" | `BaseCheckpointLoader` has clean abstract interface |
| "Must use init container + disk round-trip" | Can integrate directly into TRT-LLM loading pipeline |

### 1.4 Existing Extension Points

```
TRT-LLM Loading Pipeline (PyTorch Backend)
═══════════════════════════════════════════

LLM(model=path, checkpoint_format="mx-p2p")     ← 1. User specifies format
    │
    ▼
_construct_checkpoint_loader("mx-p2p")           ← 2. Looks up registered loader
    │
    ▼
MxCheckpointLoader.load_config(path)             ← 3. We return config from MX server
    │
    ▼
MxCheckpointLoader.load_weights(path, mapping)   ← 4. We return weights via NIXL RDMA
    │
    ▼
MxCheckpointLoader.get_initialized_weight_mapper()  ← 5. We reuse HfWeightMapper
    │
    ▼
model.load_weights(weights, weight_mapper)        ← 6. TRT-LLM handles mapping
```

---

## 2. Current POC vs Target Architecture

### 2.1 Current POC Data Flow (with disk round-trip)

```
Source GPU ──RDMA──► Target GPU ──cpu()──► CPU RAM ──save──► Disk ──load──► CPU ──cuda()──► GPU
     │                                                                                │
     │                 ~1.5s                ~5s (reconstruct)    ~30s (disk I/O)       │
     └── NIXL transfer: 170 GB @ 112 Gbps ──────────────────────────────────────────┘
                                                Total overhead: ~35s
```

### 2.2 Phase 1 Target (Custom Loader, no disk)

```
Source GPU ──RDMA──► Target GPU ──cpu()──► CPU dict ──load_weights()──► GPU
     │                                                              │
     │                 ~1.5s           ~5s (reconstruct + copy)     │
     └── NIXL transfer ────────────────────────────────────────────┘
                                  Total overhead: ~5s
                                  Saves: ~30s disk I/O
```

### 2.3 Phase 2 Target (Zero-copy, no CPU)

```
Source GPU ──RDMA──► Target GPU ──inject──► TRT-LLM model (in-place)
     │                                    │
     │                 ~1.5s              │
     └── NIXL transfer ──────────────────┘
                    Total overhead: ~0s
                    Saves: ~35s disk I/O + CPU copy
```

### 2.4 Phase 3 Target (Direct shard injection, no reconstruction)

```
Source GPU[0..7] ──RDMA──► Target GPU[0..7] ──inject──► TRT-LLM model[0..7]
     │                                                │
     │              8 × ~1.5s parallel                │
     └── 8 NIXL streams ─────────────────────────────┘
                    Total overhead: ~0s
                    No reconstruction, no re-sharding
```

---

## 3. Phase 1: Custom Checkpoint Loader (No Upstream Changes)

This phase eliminates the init container and disk round-trip by integrating directly into TRT-LLM's loading pipeline. **No upstream TRT-LLM changes required.**

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Single Container: TRT-LLM + ModelExpress Client                         │
│                                                                          │
│   from tensorrt_llm import LLM                                          │
│   import modelexpress  # Triggers @register_checkpoint_loader("mx-p2p") │
│                                                                          │
│   llm = LLM(                                                            │
│       model="/models/llama-70b",      # For config + tokenizer          │
│       checkpoint_format="mx-p2p",     # ← Uses our custom loader       │
│       tensor_parallel_size=8,                                           │
│   )                                                                      │
│                                                                          │
│   Inside MxCheckpointLoader.load_weights():                              │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │ 1. Query MX server for source metadata                          │  │
│   │ 2. For each TP rank:                                            │  │
│   │    a. Initialize NIXL agent                                      │  │
│   │    b. Allocate GPU tensors with correct shapes                   │  │
│   │    c. Receive weights via RDMA (GPU-to-GPU)                      │  │
│   │    d. Copy to CPU (for weight mapper compatibility)              │  │
│   │ 3. Reconstruct full HF weights from TP shards                   │  │
│   │ 4. Return HF-format weight dict                                 │  │
│   │    (TRT-LLM's HfWeightMapper handles the rest)                  │  │
│   └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Implementation: MxWeightLoader

```python
# modelexpress/trtllm_checkpoint_loader.py

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
    BaseWeightLoader, ConsumableWeightsDict)
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_weight_loader
from tensorrt_llm.mapping import Mapping

@register_checkpoint_weight_loader("mx-p2p")
class MxWeightLoader(BaseWeightLoader):
    """Loads weights via NIXL RDMA from a ModelExpress source."""

    def load_weights(self, checkpoint_dir: str,
                     mapping: Mapping) -> dict[str, Any]:
        """
        Instead of loading from disk, receives weights via P2P RDMA.

        Args:
            checkpoint_dir: Used to derive model_name for MX server query.
                           Config files (config.json, tokenizer) still read from here.
            mapping: TRT-LLM distributed mapping (tp_rank, tp_size, etc.)

        Returns:
            ConsumableWeightsDict with HF-format weight names.
            The HfWeightMapper will handle name conversion to TRT-LLM format.
        """
        from .nixl_transfer import NixlTransferManager
        from .client import MxClient

        mx_server = os.environ.get("MODEL_EXPRESS_URL", "localhost:8001")
        model_name = os.environ.get("MODEL_NAME", os.path.basename(checkpoint_dir))

        client = MxClient(mx_server)

        # 1. Query source metadata
        source_meta = client.get_metadata(model_name)

        # 2. Receive weights from all source ranks
        all_rank_weights = {}
        for worker in source_meta.workers:
            rank = worker.worker_rank
            nixl_mgr = NixlTransferManager(
                agent_name=f"trtllm-target-r{rank}",
                device_id=rank
            )
            nixl_mgr.initialize()

            # Allocate and receive
            weights = self._allocate_and_receive(nixl_mgr, worker)

            # Copy to CPU immediately (CUDA context safety)
            all_rank_weights[rank] = {k: v.cpu() for k, v in weights.items()}
            nixl_mgr.shutdown()

        # 3. Reconstruct full HF weights from TP shards
        full_weights = self._reconstruct_full_weights(all_rank_weights)

        return ConsumableWeightsDict(full_weights)

    def _allocate_and_receive(self, nixl_mgr, worker):
        """Allocate GPU tensors and receive via RDMA."""
        weights = {}
        for t in worker.tensors:
            shape = tuple(t.shape) if t.shape else (t.size // dtype_size,)
            weights[t.name] = torch.empty(
                shape, dtype=parse_dtype(t.dtype),
                device=f"cuda:{worker.worker_rank}"
            )
        nixl_mgr.register_tensors(weights)
        nixl_mgr.add_remote_agent(worker.nixl_metadata)
        nixl_mgr.receive_from_source(worker)
        return weights

    def _reconstruct_full_weights(self, all_rank_weights):
        """Reconstruct full HF weights from TP-sharded transfers."""
        # Same logic as current trtllm_loader.py
        # Concat shards based on naming patterns (q_proj → dim=-1, etc.)
        ...
```

### 3.3 Implementation: MxCheckpointLoader

```python
# modelexpress/trtllm_checkpoint_loader.py

from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.config_loader import HfConfigLoader
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_loader

@register_checkpoint_loader("mx-p2p")
class MxCheckpointLoader(BaseCheckpointLoader):
    """
    ModelExpress P2P checkpoint loader for TRT-LLM.

    Loads model config from local HF directory (or MX server),
    loads weights via NIXL RDMA P2P transfer.
    Reuses HfWeightMapper for name conversion.
    """

    def __init__(self, *, weight_loader=None, weight_mapper=None, config_loader=None):
        self._weight_loader = weight_loader or MxWeightLoader()
        self._config_loader = config_loader or HfConfigLoader()  # Reuse HF config loader
        self._weight_mapper = weight_mapper  # Will be auto-resolved to HfWeightMapper
        self._checkpoint_format = "mx-p2p"

    def get_default_weight_loader(self):
        return MxWeightLoader()

    def get_default_config_loader(self):
        return HfConfigLoader()

    def cleanup(self):
        if self._weight_mapper:
            self._weight_mapper.cleanup()
            self._weight_mapper = None
        if self._weight_loader:
            self._weight_loader.cleanup()
            self._weight_loader = None

    @property
    def weight_loader(self): return self._weight_loader

    @property
    def weight_mapper(self): return self._weight_mapper

    @weight_mapper.setter
    def weight_mapper(self, value): self._weight_mapper = value

    @property
    def config_loader(self): return self._config_loader

    @property
    def checkpoint_format(self): return self._checkpoint_format
```

### 3.4 Key Design Decision: Return HF-Format Weights

Our `load_weights()` returns weights in **HuggingFace format** (e.g., `model.layers.0.self_attn.q_proj.weight`). This is critical because:

1. **We reuse `HfWeightMapper`**: TRT-LLM's existing mapper handles:
   - `q_proj`, `k_proj`, `v_proj` → fused `qkv_proj`
   - `gate_proj`, `up_proj` → fused `gate_up_proj`
   - TP sharding per rank
   - KV weight duplication

2. **No custom weight mapper needed**: The `get_initialized_weight_mapper()` in `BaseCheckpointLoader` auto-resolves to `HfWeightMapper` via `AutoCheckpointMapper.get("mx-p2p", model_arch)` when we don't provide a custom mapper.

3. **Note**: For auto-resolution to work, we may need to register our format with the `AutoCheckpointMapper` or fall back to HF format explicitly:

```python
def get_initialized_weight_mapper(self, model, config):
    # Override to explicitly use HF mapper since our weights are HF-format
    from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
    mapper = HfWeightMapper()
    mapper.init_model_and_config(model, config)
    self._weight_mapper = mapper
    return mapper
```

### 3.5 Usage

```python
# Option A: Via checkpoint_format string
import modelexpress  # Side-effect: registers "mx-p2p" loader
from tensorrt_llm import LLM, SamplingParams

llm = LLM(
    model="/models/llama-70b",       # Config + tokenizer from local path
    checkpoint_format="mx-p2p",      # Weights via NIXL RDMA
    tensor_parallel_size=8,
)

# Option B: Via checkpoint_loader instance (more explicit)
from modelexpress.trtllm_checkpoint_loader import MxCheckpointLoader

llm = LLM(
    model="/models/llama-70b",
    checkpoint_loader=MxCheckpointLoader(),
    tensor_parallel_size=8,
)
```

### 3.6 Kubernetes Deployment (Single Container)

```yaml
# No more init container! Single container with both TRT-LLM and ModelExpress.
containers:
  - name: trtllm
    image: nvcr.io/nvidia/tensorrt-llm-with-mx:0.21.0  # Custom image
    env:
      - name: MODEL_NAME
        value: "meta-llama/Llama-3.1-70B-Instruct"
      - name: MODEL_EXPRESS_URL
        value: "modelexpress-server:8001"
    command: ["python3", "-c"]
    args:
      - |
        import modelexpress  # Register mx-p2p loader
        from tensorrt_llm import LLM, SamplingParams

        llm = LLM(
            model="/models/llama-70b",
            checkpoint_format="mx-p2p",
            tensor_parallel_size=8,
        )
        # Serve with OpenAI-compatible API...
```

### 3.7 What Changes vs POC

| Aspect | POC (Init Container) | Phase 1 (Custom Loader) | Phase 1 Validated |
|--------|---------------------|------------------------|-------------------|
| **Containers** | 2 (init + main) | 1 (single) | ✓ Single container |
| **PVC on target** | Required (config files) | Not needed | ✓ Config from MX server |
| **Disk I/O** | 141 GB write + read (~30s) | None | ✓ No disk (RDMA → CPU dict → model) |
| **Integration** | External to TRT-LLM | Inside TRT-LLM pipeline | ✓ `checkpoint_loader=` |
| **CPU copy** | Yes (CUDA context safety) | Yes (weight mapper needs CPU dict) | ✓ GPU→CPU→GPU |
| **Reconstruction** | Yes (TP shards → full HF) | Yes (same for TP>1) | ✓ (not needed for TP=1) |
| **NIXL + MPI** | Separate processes | Same process (system NIXL) | ✓ No UCX conflict |
| **Transfer speed** | 112 Gbps (Llama 70B) | 127.7 Gbps (Qwen 0.5B) | ✓ Validated |

### 3.8 UCX Library Conflict in MPI Workers (RESOLVED)

**Discovered and resolved during Phase 1 testing.**

`pip install nixl[cu13]` bundles its own UCX libraries which conflict with MPI's HPC-X UCX
when both are loaded in the same TRT-LLM worker process (segfault in `uct_md_query_tl_resources`).

**Resolution**: Use the **system NIXL** from the TRT-LLM NGC image (`/opt/nvidia/nvda_nixl/`).
It links against the same HPC-X UCX as MPI — no conflict. Validated end-to-end: MPI init first,
then NIXL agent creation, then RDMA transfer, all in the same process at 127.7 Gbps.

```dockerfile
# Use system NIXL (same UCX as MPI) — NOT pip nixl (bundles conflicting UCX)
RUN pip install --no-deps nixl && \
    ln -sf /opt/nvidia/nvda_nixl/lib/python3/dist-packages/nixl_cu13 \
           /usr/local/lib/python3.12/dist-packages/nixl_cu13 && \
    echo "/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/nixl-system.conf && \
    ldconfig
```

**Caveat**: Descriptor coalescing (`_coalesce_transfers`) fails with system NIXL's
`prepXferDlist` (`NIXL_ERR_NOT_FOUND`). Disabled by default; individual tensor transfers
work at full bandwidth (127.7 Gbps). This is a minor optimization gap, not a blocker.

### 3.9 Critical Constraint: No Access to Model Parameter Addresses

The custom loader's `load_weights()` method does **not** receive the model object. The exact call sequence in `ModelLoader.load()` is:

```python
# tensorrt_llm/_torch/pyexecutor/model_loader.py — ModelLoader.load()

# Step 1: Model created and allocated on GPU (parameter buffers exist)
with MetaInitMode():
    model = AutoModelForCausalLM.from_config(config)
model._apply(init_meta_tensor)    # Meta → real GPU tensors
model.to("cuda")                  # Parameters allocated on GPU with known data_ptr()

# Step 2: OUR load_weights() called — receives (checkpoint_dir, mapping) only, NOT model
weights = checkpoint_loader.load_weights(checkpoint_dir, mapping=self.mapping)

# Step 3: Mapper receives model (too late for load_weights)
self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(model, config)

# Step 4: Weights COPIED into model parameters via copy_weight()
self._call_load_weights(model.load_weights, weights, self.weight_mapper)
```

The final assignment at Step 4 is always a memcpy, never a pointer swap:

```python
# tensorrt_llm/_torch/modules/linear.py
def copy_weight(dst: Parameter, src: torch.Tensor):
    if dst.dtype != src.dtype:
        src = src.to(dst.dtype)
    dst.data.copy_(src)  # ← Always a copy, never param.data = src
```

**Implications for zero-copy**:

| What we want | Can we do it? | Why |
|-------------|--------------|-----|
| NIXL writes into our allocated GPU buffers → return as dict | **Yes (Phase 1)** | We allocate, NIXL writes, we return. But then `copy_weight()` copies again into model params. |
| NIXL writes directly into model parameter buffers | **No** | `load_weights()` is called without the model, so we don't know `param.data_ptr()` addresses. |
| Return GPU tensors and have `param.data = tensor` (pointer swap) | **No (without upstream)** | `copy_weight()` always does `.copy_()`, not `param.data =`. |

This means Phase 1 has an unavoidable **GPU → GPU copy** at the end (fast, but not zero-copy). True zero-copy requires upstream changes (Phase 2).

---

## 4. Phase 2: True Zero-Copy via Deferred NIXL Transfer (Upstream)

Phase 1 still has a final GPU→GPU `copy_weight()` because the loader can't access model parameter addresses. Phase 2 eliminates this via two possible approaches.

### 4.1 The Remaining Copy in Phase 1

```
Phase 1 data flow:
Source GPU ──RDMA──► Our GPU buffer ──copy_weight()──► Model param buffer
                         │                                    │
                    we allocate this              TRT-LLM allocates this
                    (known addr)                  (unknown at load_weights time)

Cost: ~2-3s GPU-to-GPU memcpy for 141 GB (much better than 30s disk, but not zero)
```

### 4.2 Approach A: Deferred NIXL with Lazy Weight Dict (No Upstream Changes)

The `BaseCheckpointLoader` instance persists across all calls. We can exploit this:

```python
@register_checkpoint_loader("mx-p2p")
class MxCheckpointLoader(BaseCheckpointLoader):
    def __init__(self, ...):
        self._model_ref = None
        self._source_meta = None
        self._nixl_managers = {}

    def load_weights(self, checkpoint_dir, mapping):
        """
        Called BEFORE we have the model. Don't transfer yet.
        Return a lazy dict that defers NIXL transfer until first access.
        """
        # Query MX server for source metadata (lightweight)
        self._source_meta = self._query_source(mapping)
        self._mapping = mapping

        # Return a lazy dict — weights will be populated later
        return LazyMxWeightDict(self)

    def get_initialized_weight_mapper(self, model, config):
        """
        Called AFTER load_weights, WITH the model.
        NOW we have access to model parameter addresses.
        Trigger NIXL to write directly into model params.
        """
        self._model_ref = model

        # Build map: HF weight name → model param data_ptr()
        # (after mapper resolves name mapping)
        mapper = HfWeightMapper()
        mapper.init_model_and_config(model, config)
        self._weight_mapper = mapper

        # NOW trigger NIXL: write directly into model parameter buffers
        self._transfer_into_model_params(model, mapper)

        return mapper

    def _transfer_into_model_params(self, model, mapper):
        """
        The key innovation: NIXL writes directly into model parameter GPU memory.
        """
        for worker in self._source_meta.workers:
            rank = worker.worker_rank
            nixl_mgr = NixlTransferManager(
                agent_name=f"trtllm-target-r{rank}",
                device_id=rank
            )
            nixl_mgr.initialize()

            # Build NIXL destination descriptors pointing at MODEL's param buffers
            dst_descriptors = {}
            for name, param in model.named_parameters():
                # Map TRT-LLM name back to HF name using mapper
                hf_name = mapper.reverse_map(name)
                dst_descriptors[hf_name] = {
                    'addr': param.data_ptr(),     # ← Model's actual GPU address
                    'size': param.numel() * param.element_size(),
                    'device_id': rank,
                }

            # NIXL pulls from source directly into model param buffers
            nixl_mgr.receive_into_addresses(
                source_metadata=worker.nixl_metadata,
                source_tensors=worker.tensors,
                dst_descriptors=dst_descriptors,
            )
            nixl_mgr.shutdown()


class LazyMxWeightDict(dict):
    """
    A dict that reports weights as 'already loaded' to skip copy_weight().
    Used when NIXL writes directly into model parameter buffers.
    """
    def __init__(self, loader):
        self._loader = loader

    def __contains__(self, key):
        return True  # Pretend all weights exist

    def __getitem__(self, key):
        # Return a sentinel that copy_weight() recognizes as "skip"
        # OR: return the actual param tensor (copy to self = no-op)
        ...
```

**Challenge**: This approach is creative but fragile:
- Requires a reverse name mapping (TRT-LLM → HF) which is non-trivial due to fused weights (qkv_proj ← q+k+v)
- The `LazyMxWeightDict` needs to fool the weight loading pipeline into thinking weights are loaded
- Fused weights (q_proj + k_proj + v_proj → single qkv buffer) mean NIXL must write to sub-regions of a single parameter

### 4.3 Approach B: Upstream `copy_weight()` Change (Clean, Small PR)

Change `copy_weight()` in TRT-LLM to support pointer swap when safe:

```python
# Proposed change to tensorrt_llm/_torch/modules/linear.py
def copy_weight(dst: Parameter, src: torch.Tensor):
    if dst.dtype != src.dtype:
        src = src.to(dst.dtype)
    if (src.device == dst.device
            and src.shape == dst.shape
            and src.is_contiguous()
            and dst.is_contiguous()):
        dst.data = src  # Zero-copy pointer swap (same device, same layout)
    else:
        dst.data.copy_(src)  # Fallback to copy
```

With this change, Phase 1's `MxWeightLoader` can return GPU tensors directly. When TRT-LLM assigns them, the pointer swap avoids the copy.

**But**: This alone isn't enough because the weight mapper transforms weights (sharding, fusing q/k/v). The returned tensors won't have the same shape as the model parameters. This approach only works with Phase 3 (pre-sharded weights).

### 4.4 Approach C: Upstream `load_weights()` Receives Model (Clean, Medium PR)

Pass the model to the checkpoint loader so it can do direct injection:

```python
# Proposed change to tensorrt_llm/_torch/pyexecutor/model_loader.py
# In ModelLoader.load():

# Current:
weights = checkpoint_loader.load_weights(checkpoint_dir, mapping=self.mapping)

# Proposed:
weights = checkpoint_loader.load_weights(
    checkpoint_dir, mapping=self.mapping, model=model)  # ← Pass model
```

And update `BaseCheckpointLoader`:

```python
# base_checkpoint_loader.py
def load_weights(self, checkpoint_dir, mapping, model=None, **kwargs):
    if model is not None:
        # Subclass can use model.named_parameters() for direct injection
        pass
    return self.weight_loader.load_weights(checkpoint_dir, mapping=mapping, **kwargs)
```

**Impact**: Backward compatible (model defaults to None). Existing loaders ignore it. Our loader uses it for zero-copy.

### 4.5 Recommended Approach for Phase 2

**Start with Approach C** (pass model to load_weights). It's the cleanest upstream change:
- Small diff (add `model=model` to one call site, add `model=None` to base class)
- Backward compatible
- Enables direct NIXL-into-params for our loader
- Other custom loaders can benefit too

**Fallback**: If the upstream PR is slow, use Approach A (deferred NIXL) as a temporary workaround.

### 4.6 Data Flow After Phase 2

```
Approach C (with upstream):
Source GPU ──RDMA──► Model param buffer (directly)
                    (true zero-copy: NIXL dst = param.data_ptr())

Approach A (workaround):
Source GPU ──RDMA──► Model param buffer (via deferred transfer in get_initialized_weight_mapper)
                    (true zero-copy but fragile implementation)
```

**Savings**: Eliminates ~2-3s GPU→GPU copy for 141 GB model. Combined with Phase 1, total overhead drops from ~35s to under 2s.

### 4.7 Critical Finding: TP>1 Requires Per-Worker Coordination

**Discovered during Llama 70B (TP=8) testing.** The custom checkpoint loader works for
TP=1 but breaks for TP>1 due to TRT-LLM's per-worker architecture.

**The conflict**: TRT-LLM spawns 8 MPI workers. Each independently calls `load_weights()`.
The `HfWeightMapper` expects **full** HF weights (141 GB) and shards per rank internally.
But with RDMA, each worker can only practically receive from one source rank (21 GB),
and the mapper will re-shard already-sharded weights incorrectly.

**Attempted solutions and failures**:
- All 8 workers receive from all 8 sources → 64 NIXL connections → `NIXL_ERR_REMOTE_DISCONNECT`
- Each worker receives only its rank's shard → mapper re-shards → wrong shapes
- All 8 ranks on one GPU → OOM (141 GB per GPU, but each GPU only has ~80 GB)

**The RDMA transfer itself works perfectly**: 8 ranks × 21.32 GB @ 85 Gbps per rank.
The bottleneck is entirely in TRT-LLM's weight loading architecture.

**This elevates "pre-sharded weight loading" from P1 to P0** — it's required for any
TP>1 deployment to work without disk I/O. See Section 5 for the detailed proposal.

---

## 5. Phase 3: Direct Per-Rank Shard Injection (CRITICAL for TP>1)

This is now a **P0 requirement** — without this, TP>1 deployments must fall back to
disk-based loading. Phase 3 eliminates the double-sharding AND solves the per-worker
coordination problem.

### 5.1 The Double-Sharding Problem (Now Also a TP>1 Blocker)

```
Current (TP=1 only — works):
Source full HF → RDMA → Worker 0 gets full weights → Mapper shards (no-op for TP=1)

Broken (TP>1):
Source shards → RDMA → Worker N gets shard N → Mapper tries to re-shard → WRONG

Ideal (TP>1):
Source shards → RDMA → Worker N gets shard N → Mapper fuses (q+k+v → qkv) but skips TP slice
```

### 5.2 How TRT-LLM Currently Shards

In the PyTorch backend, sharding happens inside the weight mapper:

```python
# tensorrt_llm/_torch/models/checkpoints/hf/weight_mapper.py (HfWeightMapper)
# During apply_callbacks():
#   - q_proj weight (full) → split by tp_size along last dim → rank's slice
#   - gate_proj weight (full) → split by tp_size along first dim → rank's slice
```

And in the model's `load_weights()`:

```python
# tensorrt_llm/_torch/modules/linear.py (load_weight_shard)
# Slices the full weight tensor to get this rank's shard
tp_slice = full_weight[..., start:end]  # For column parallel
```

### 5.3 Upstream Change: Accept Pre-Sharded Weights

**What we need**: A `load_format` or flag that tells TRT-LLM "these weights are already sharded for my rank, skip re-sharding":

```python
# Proposed API addition to TorchLlmArgs
load_format: LoadFormat = Field(default=LoadFormat.AUTO)

class LoadFormat(str, Enum):
    AUTO = "AUTO"
    DUMMY = "DUMMY"
    PRESHARDED = "PRESHARDED"  # ← NEW: Weights already sharded per rank
```

When `PRESHARDED`, the weight mapper would skip the sharding step and directly assign weights to the model parameters.

**Alternative**: A simpler approach is to have `MxWeightLoader.load_weights()` receive only the current rank's shard and return it with TRT-LLM-format names (bypassing HfWeightMapper):

```python
@register_checkpoint_weight_loader("mx-p2p-sharded")
class MxShardedWeightLoader(BaseWeightLoader):
    def load_weights(self, checkpoint_dir, mapping):
        # Only receive weights for mapping.tp_rank
        rank = mapping.tp_rank
        weights = receive_rank_weights_via_nixl(rank)
        # Return with TRT-LLM names (not HF names)
        return ConsumableWeightsDict(weights)
```

This would require a custom `MxWeightMapper` that maps source names to TRT-LLM names without sharding.

### 5.4 Data Flow After Phase 3

```
Source GPU[rank] ──RDMA──► Target GPU[rank] ──assign──► Model[rank]
                   ~1.5s                       ~0s

Total: 1.5s per rank (all 8 in parallel) = 1.5s wall clock
```

**Savings**: Eliminates ~5s reconstruction + ~3s re-sharding.

---

## 6. Phase 4: Reuse TRT-LLM's NIXL Infrastructure

TRT-LLM already has NIXL support for KV cache disaggregated serving. Phase 4 extends this infrastructure for weight transfers.

### 6.1 Existing NIXL in TRT-LLM

```
tensorrt_llm/
├── _torch/disaggregation/
│   ├── nixl/
│   │   ├── _agent_cpp.py       # C++ NIXL agent wrapper
│   │   ├── _agent_py.py        # Pure Python NIXL agent
│   │   └── ...
│   └── base/
│       └── agent.py            # Base transfer agent classes
│
cpp/tensorrt_llm/executor/cache_transmission/
├── nixl_utils/
│   ├── transferAgent.cpp       # C++ NIXL implementation
│   └── transferAgent.h         # Interface
└── ucx_utils/
    └── ucxCacheCommunicator.cpp  # UCX alternative
```

### 6.2 Key Patterns We Can Reuse

| TRT-LLM NIXL Pattern | Reuse for Weights |
|----------------------|-------------------|
| `BaseAgentConfig` with UCX backend | Same config, tune for large transfers |
| `registerMemory()` for VRAM regions | Register weight tensors |
| `submitTransferRequests()` for RDMA | Transfer weight blocks |
| `loadRemoteAgent()` for peer discovery | Discover source weight servers |
| Port management via file locks | Same approach |
| `TRTLLM_NIXL_INTERFACE` env var | Same network interface |

### 6.3 Integration Approach

Instead of using ModelExpress's separate `NixlTransferManager`, we could use TRT-LLM's native NIXL agent:

```python
from tensorrt_llm._torch.disaggregation.nixl import NixlTransferAgent, BaseAgentConfig

class MxWeightLoader(BaseWeightLoader):
    def load_weights(self, checkpoint_dir, mapping):
        config = BaseAgentConfig(
            name=f"mx-weight-target-{mapping.tp_rank}",
            use_prog_thread=True,
            use_listen_thread=True,
            backend_params={"num_threads": "4"}
        )
        agent = NixlTransferAgent(config)

        # Register local weight memory
        weight_descs = self._create_weight_descriptors(mapping)
        agent.register_memory(weight_descs)

        # Load remote source agent
        source_desc = self._get_source_agent_desc(mapping.tp_rank)
        agent.load_remote_agent(source_desc)

        # Submit transfer
        status = agent.submit_transfer_requests(transfer_req)
        status.wait()

        return weights
```

**Benefit**: Single NIXL stack, consistent configuration, shared backend pool.

---

## 7. Performance Impact Analysis

### 7.1 Llama 70B (TP=8) Breakdown

| Phase | Component | Current POC | Phase 1 | Phase 2 | Phase 3 |
|-------|-----------|-------------|---------|---------|---------|
| Transfer | NIXL RDMA (8 ranks) | 1.5s | 1.5s | 1.5s | 1.5s |
| Copy | GPU → CPU → GPU | 5s | 0s (GPU dict) | 0s | 0s |
| Reconstruct | TP shards → full HF | 5s | 5s | 5s | 0s |
| Disk I/O | Save + Load safetensors | 30s | 0s | 0s | 0s |
| Re-shard | TRT-LLM TP sharding | 3s | 3s | 3s | 0s |
| Final copy | copy_weight() GPU→GPU | 0s | 2s | 0s (zero-copy) | 0s |
| **Total** | | **~44s** | **~12s** | **~6.5s** | **~1.5s** |

Note on Phase 1 "Final copy": The custom loader returns GPU tensors, but `copy_weight()` in
TRT-LLM does `dst.data.copy_(src)` — a GPU-to-GPU memcpy (~2s for 141 GB). Phase 2 eliminates
this by having NIXL write directly into model parameter buffers (requires upstream change or
deferred transfer workaround — see Section 4.2-4.5).

### 7.2 End-to-End Time Comparison

| Scenario | NVMe Load | Current POC | Phase 1 | Phase 2 | Phase 3 |
|----------|-----------|-------------|---------|---------|---------|
| **Weight loading** | ~180s | ~44s | ~12s | ~6.5s | ~1.5s |
| **TRT-LLM init** | ~120s | ~120s | ~120s | ~120s | ~120s |
| **Total to first token** | ~300s | ~164s | ~132s | ~126.5s | ~121.5s |
| **Speedup (loading only)** | baseline | 4x | 15x | 28x | **120x** |

Note: TRT-LLM PyTorch backend initialization (~120s for engine build) dominates in all cases. The weight loading optimization matters most when:
- Using cached/pre-built engines (IRefitter path)
- Scaling to many target instances (amortized source cost)
- Comparing against NVMe/network storage

### 7.3 At Scale: DeepSeek-V3 (681 GB, TP=8)

| Phase | Weight Loading Time |
|-------|-------------------|
| NVMe (baseline) | ~12 min |
| Current POC | ~2 min |
| Phase 1 | ~45s |
| Phase 2 | ~30s |
| Phase 3 | ~6s |

---

## 8. Upstream Change Proposals

### 8.1 Changes That Already Exist (No Action Needed)

| Feature | TRT-LLM Support | Location |
|---------|-----------------|----------|
| Custom checkpoint loader plugin | `@register_checkpoint_loader` | `modeling_utils.py` |
| Custom weight loader | `@register_checkpoint_weight_loader` | `modeling_utils.py` |
| Checkpoint format parameter | `checkpoint_format` in `TorchLlmArgs` | `llm_args.py` |
| Direct loader instance | `checkpoint_loader` in `TorchLlmArgs` | `llm_args.py` |
| Weight mapper auto-resolution | `AutoCheckpointMapper.get()` | `auto_mapper.py` |

### 8.2 Proposed Upstream Changes (Based on Code Analysis)

All changes target TRT-LLM's PyTorch backend weight loading pipeline. Exact files and
functions identified from codebase analysis.

#### Change 1: Pre-Sharded Weight Loading via `load_weight_shard()` (P0)

**Why P0**: Without this, TP>1 cannot bypass disk. Each MPI worker independently calls
`load_weights()`, but the `HfWeightMapper` assumes full HF weights and re-shards.

**Core insight**: The sharding happens in `load_weight_shard()` (linear.py line 151):
```python
slice_width = math.ceil(width / tensor_parallel_size)
slice_start = tensor_parallel_rank * slice_width
```
This function is called from `load_weights_vanilla_helper()`, `load_weights_fused_qkv_helper()`,
and `load_weights_fused_gate_up_helper()`. All pass `module.tp_size` and `module.tp_rank`.

**Proposed change**: Add a `presharded` flag that skips the TP slicing step.

**Files to modify (5 files, ~20 lines each)**:

1. `tensorrt_llm/llmapi/llm_args.py` — add `PRESHARDED` to `LoadFormat`:
```python
class LoadFormat(Enum):
    AUTO = 0
    DUMMY = 1
    VISION_ONLY = 2
    PRESHARDED = 3  # NEW: weights already sharded per rank
```

2. `tensorrt_llm/_torch/modules/linear.py` — add `presharded` param to `load_weight_shard()`:
```python
def load_weight_shard(weight, tensor_parallel_size=1, tensor_parallel_rank=0,
                      tensor_parallel_mode=None, device=...,
                      presharded=False):  # NEW
    ...
    if presharded or tensor_parallel_mode is None or tensor_parallel_size <= 1:
        return maybe_convert_to_torch_tensor(weight)
    # ... existing slicing code unchanged
```

3. `tensorrt_llm/_torch/modules/linear.py` — thread `presharded` through all three helpers:
```python
def load_weights_vanilla_helper(module, weights, ..., presharded=False):
    weight = load_weight_shard(weights[0]['weight'], module.tp_size,
                               module.tp_rank, module.tp_mode, device,
                               presharded=presharded)
    # ... rest unchanged

def load_weights_fused_qkv_helper(module, weights, ..., presharded=False):
    q_weight = load_weight_shard(weights[0]['weight'], ..., presharded=presharded)
    k_weight = load_weight_shard(weights[1]['weight'], ..., presharded=presharded)
    v_weight = load_weight_shard(weights[2]['weight'], ..., presharded=presharded)
    # Fusing (cat) still happens — we NEED fusing, just not slicing

def load_weights_fused_gate_up_helper(module, weights, ..., presharded=False):
    # Same pattern
```

4. `tensorrt_llm/_torch/pyexecutor/model_loader.py` — add `PRESHARDED` branch:
```python
if load_format == LoadFormat.PRESHARDED:
    weights = checkpoint_loader.load_weights(
        checkpoint_dir, mapping=self.mapping)
    self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(
        model, config)
    # Pass presharded=True so load_weight_shard skips TP slicing
    self._call_load_weights(model.load_weights, weights,
                            self.weight_mapper,
                            presharded=True)  # NEW param
```

5. Thread `presharded` from `_call_load_weights` → `model.load_weights()` → `_load_weights_impl_v2()`
   → `module.load_weights()` → `LinearMethodBase.load_weights()` → helpers.

**Impact**: Low-medium risk. The slicing is cleanly isolated in `load_weight_shard()`.
The fusing logic (q+k+v → qkv, gate+up → gate_up) is **not affected** — fusing still
works because it concatenates whatever shards are provided. Only the slicing is skipped.

**How ModelExpress uses it**:
```python
# Each MPI worker receives only its rank's shard (21 GB for 70B/TP=8)
def load_weights(self, checkpoint_dir, mapping):
    my_rank = mapping.tp_rank
    weights = nixl_receive_from_rank(my_rank)  # 21 GB, HF names, per-rank sizes
    return weights  # mapper fuses but doesn't re-shard
```

#### Change 2: Pass Model to `load_weights()` (P1)

**Why**: Enables NIXL to write directly into model parameter buffers (zero-copy).
Also solves the TP>1 problem differently — each worker's model has params on the right GPU.

**Files to modify (3 files, ~5 lines each)**:

1. `tensorrt_llm/_torch/pyexecutor/model_loader.py` line 273:
```python
# Current:
weights = checkpoint_loader.load_weights(checkpoint_dir, mapping=self.mapping)
# Proposed (backward compatible):
weights = checkpoint_loader.load_weights(
    checkpoint_dir, mapping=self.mapping, model=model)
```

2. `tensorrt_llm/_torch/models/checkpoints/base_checkpoint_loader.py`:
```python
def load_weights(self, checkpoint_dir, mapping, model=None, **kwargs):
    return self.weight_loader.load_weights(
        checkpoint_dir, mapping=mapping, model=model, **kwargs)
```

3. `tensorrt_llm/_torch/models/checkpoints/base_weight_loader.py`:
```python
def load_weights(self, checkpoint_dir, mapping, model=None) -> dict:
    # Existing loaders ignore model=None (backward compatible)
```

**Impact**: Minimal — `model=None` default makes it backward compatible.

#### Change 3: Zero-Copy `copy_weight()` (P1)

**File**: `tensorrt_llm/_torch/modules/linear.py` line 159

```python
def copy_weight(dst: Parameter, src: torch.Tensor):
    if dst.dtype != src.dtype:
        src = src.to(dst.dtype)
    if src.data_ptr() == dst.data_ptr():
        return  # Already in place (e.g., NIXL wrote directly into param)
    if (src.device == dst.device and src.shape == dst.shape
            and src.is_contiguous() and dst.is_contiguous()):
        dst.data = src  # Zero-copy pointer swap
    else:
        dst.data.copy_(src)
```

### 8.3 Updated Priority Matrix

| Change | Priority | Upstream? | Effort | Impact | Enables |
|--------|----------|-----------|--------|--------|---------|
| Custom checkpoint loader | - | **Already exists** | - | - | TP=1 zero-disk ✓ |
| System NIXL for MPI compat | - | **Already exists** (NGC image) | - | - | NIXL in MPI workers ✓ |
| **Pre-sharded loading** | **P0** | Yes (5 files, ~20 lines each) | **Medium** | **TP>1 zero-disk** | **Unblocks production** |
| Pass model to loader | P1 | Yes (3 files, ~5 lines) | Low | True zero-copy RDMA | Eliminates GPU→GPU copy |
| Zero-copy copy_weight | P1 | Yes (1 file, ~10 lines) | Low | Skip memcpy | Pairs with model ref |

---

## 9. Implementation Roadmap (Revised)

### Phase 1: Custom Checkpoint Loader — COMPLETE

**Status**: Done. See TRTLLM_PLAN_PH1.md.

- [x] `MxCheckpointLoader`, `MxWeightLoader`, `MxConfigLoader` implemented
- [x] Config transfer via gRPC `model_files` (no PVC on target)
- [x] System NIXL from NGC image (MPI compatible)
- [x] Docker image: `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6` + ModelExpress
- [x] TP=1 validated: Qwen 0.5B, 151 Gbps, zero disk, inference OK
- [x] TP=8 RDMA validated: Llama 70B, 85 Gbps/rank, 170 GB total
- [ ] ~~TP=8 zero-disk~~ **BLOCKED** — requires Phase 2

### Phase 2: Pre-Sharded Weight Loading — TRT-LLM Upstream PR (1-2 weeks)

**Goal**: Enable TP>1 zero-disk loading. This is the critical path.

- [ ] Submit TRT-LLM PR: `LoadFormat.PRESHARDED` + `presharded` flag in `load_weight_shard()`
- [ ] Thread `presharded` through helpers (vanilla, fused_qkv, fused_gate_up)
- [ ] Add `PRESHARDED` branch in `ModelLoader.load()`
- [ ] Update `MxWeightLoader`: each worker receives only matching source rank
- [ ] Test: Llama 70B (TP=8) zero-disk end-to-end
- [ ] Test: Verify fusing still works (q+k+v → qkv with pre-sharded inputs)

### Phase 3: True Zero-Copy RDMA — TRT-LLM Upstream PR (1-2 weeks)

**Goal**: NIXL writes directly into model parameter buffers. No intermediate copies.

- [ ] Submit TRT-LLM PR: Pass `model` to `load_weights()` (backward compatible)
- [ ] Submit TRT-LLM PR: Zero-copy `copy_weight()` pointer swap
- [ ] Update `MxWeightLoader`: use `model.named_parameters()` for RDMA destinations
- [ ] Build TRT-LLM param name → HF source name mapping
- [ ] Test: Verify zero-copy path with Qwen 0.5B (TP=1) and Llama 70B (TP=8)

### Phase 4: Optimization and Scale (2-3 weeks)

- [ ] Investigate coalesced descriptor fix for system NIXL
- [ ] Multi-source fan-out (receive from multiple warm instances)
- [ ] Benchmark: Compare all phases against NVMe baseline and POC init-container
- [ ] Production hardening: error handling, timeouts, retry logic

---

## Appendix A: TRT-LLM Code References

### Checkpoint Loader Registration

```
tensorrt_llm/_torch/models/modeling_utils.py:625    CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING
tensorrt_llm/_torch/models/modeling_utils.py:692    @register_checkpoint_loader(name)
tensorrt_llm/_torch/models/modeling_utils.py:699    @register_checkpoint_weight_loader(name)
```

### Base Classes

```
tensorrt_llm/_torch/models/checkpoints/base_checkpoint_loader.py     BaseCheckpointLoader
tensorrt_llm/_torch/models/checkpoints/base_weight_loader.py         BaseWeightLoader
tensorrt_llm/_torch/models/checkpoints/base_weight_mapper.py         BaseWeightMapper
tensorrt_llm/_torch/models/checkpoints/base_config_loader.py         BaseConfigLoader
```

### HF Reference Implementation

```
tensorrt_llm/_torch/models/checkpoints/hf/checkpoint_loader.py      HfCheckpointLoader
tensorrt_llm/_torch/models/checkpoints/hf/weight_loader.py          HfWeightLoader
tensorrt_llm/_torch/models/checkpoints/hf/weight_mapper.py          HfWeightMapper
tensorrt_llm/_torch/models/checkpoints/hf/config_loader.py          HfConfigLoader
```

### Loading Pipeline

```
tensorrt_llm/llmapi/llm.py:1139                 _TorchLLM._build_model()
tensorrt_llm/_torch/pyexecutor/model_engine.py:198   PyTorchModelEngine → ModelLoader
tensorrt_llm/_torch/pyexecutor/model_loader.py:157   _construct_checkpoint_loader()
tensorrt_llm/_torch/pyexecutor/model_loader.py:215   ModelLoader.load()
tensorrt_llm/_torch/pyexecutor/model_loader.py:270   checkpoint_loader.load_weights()
```

### NIXL Infrastructure

```
cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.cpp
tensorrt_llm/_torch/disaggregation/nixl/_agent_py.py
tensorrt_llm/_torch/disaggregation/base/agent.py
```

### LLM API Parameters

```
tensorrt_llm/llmapi/llm_args.py:2995    checkpoint_loader: Optional[object]
tensorrt_llm/llmapi/llm_args.py:3011    checkpoint_format: Optional[str]
tensorrt_llm/llmapi/llm_args.py:3200    validate_checkpoint_format()
```

## Appendix B: Quick Reference — How vLLM vs TRT-LLM Compare

| Capability | vLLM | TRT-LLM | Status |
|-----------|------|---------|--------|
| Custom loader registration | `@register_model_loader` | `@register_checkpoint_loader` | **Both work** ✓ |
| CLI parameter | `--load-format mx-target` | `checkpoint_loader=instance` | **Both work** ✓ |
| NIXL inside inference worker | Yes (no MPI) | Yes (system NIXL from NGC) | **Both work** ✓ |
| TP=1 zero-disk | Yes | Yes | **Both work** ✓ |
| TP>1 zero-disk | Yes (loader gets full model) | **No** (mapper re-shards) | **TRT-LLM needs upstream** |
| Loader receives model object | Yes (`load_model()`) | No (`load_weights(dir, mapping)`) | **TRT-LLM needs upstream** |
| GPU tensor injection | `param.data = tensor` | `copy_weight()` does `.copy_()` | **TRT-LLM needs upstream** |

### Key Architectural Differences

**vLLM**: `load_model()` gives the loader the **full model object**. The loader controls
the entire flow — allocate, RDMA, assign to params. TP sharding is handled by the loader.

**TRT-LLM**: Weight loading is split across three independent steps:
1. `load_weights()` — returns a dict (no model access)
2. `HfWeightMapper` — fuses (q+k+v → qkv) AND shards (TP slicing)
3. `copy_weight()` — copies dict tensors into model params

For TP>1 RDMA, steps 2 and 3 are the blockers. Step 2 re-shards already-sharded weights.
Step 3 copies when it could pointer-swap.

### Upstream Changes Needed (3 PRs)

| PR | Change | Impact | Effort |
|----|--------|--------|--------|
| **PR 1 (P0)** | `LoadFormat.PRESHARDED` + `presharded` flag in `load_weight_shard()` | TP>1 zero-disk | ~100 lines / 5 files |
| PR 2 (P1) | `model=None` param on `load_weights()` | True zero-copy RDMA | ~15 lines / 3 files |
| PR 3 (P1) | `param.data = src` in `copy_weight()` | Skip GPU→GPU memcpy | ~10 lines / 1 file |
