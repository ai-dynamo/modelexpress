# TensorRT-LLM P2P Transfer Support for ModelExpress

This document provides comprehensive research, design, and implementation guidance for adding TensorRT-LLM (TRT-LLM) support to ModelExpress's P2P GPU weight transfer system.

---

## Executive Summary

ModelExpress currently enables sub-second model weight replication between vLLM instances using NIXL over RDMA. This document explores how to extend this capability to TensorRT-LLM, enabling fast weight transfers between TRT-LLM inference instances.

**Key Finding**: TRT-LLM's architecture differs significantly from vLLM in that weights are typically embedded into compiled TensorRT engines at build time. However, several integration paths exist:

| Approach | Complexity | Performance | Use Case |
|----------|------------|-------------|----------|
| **Checkpoint-Level Transfer** | Low | Good | Pre-build weight distribution |
| **IRefitter Runtime Injection** | Medium | Good | Post-build weight updates |
| **Custom Model Loader** | High | Best | Deep integration like vLLM |
| **Disaggregated Serving Extension** | Medium | Good | Leverage existing UCX/NIXL support |

---

## Table of Contents

1. [TensorRT-LLM Architecture Overview](#1-tensorrt-llm-architecture-overview)
2. [Weight Handling in TRT-LLM](#2-weight-handling-in-trt-llm)
3. [Comparison with vLLM Architecture](#3-comparison-with-vllm-architecture)
4. [Integration Approaches](#4-integration-approaches)
5. [Recommended Design: Checkpoint-Level P2P](#5-recommended-design-checkpoint-level-p2p)
6. [Alternative Design: IRefitter-Based Transfer](#6-alternative-design-irefitter-based-transfer)
7. [Implementation Plan](#7-implementation-plan)
8. [Code Changes Required](#8-code-changes-required)
9. [Kubernetes Deployment](#9-kubernetes-deployment)
10. [Performance Considerations](#10-performance-considerations)
11. [Open Questions and Risks](#11-open-questions-and-risks)

---

## 1. TensorRT-LLM Architecture Overview

### 1.1 Three-Phase Workflow

TRT-LLM follows a strict three-phase workflow:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRT-LLM Workflow                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: CONVERT              Phase 2: BUILD              Phase 3: RUN │
│  ┌─────────────────┐          ┌─────────────────┐        ┌────────────┐ │
│  │ HuggingFace     │          │ TRT-LLM         │        │ TensorRT   │ │
│  │ NeMo            │ ──────►  │ Checkpoint      │ ────►  │ Engine     │ │
│  │ DeepSpeed       │          │ (config.json +  │        │ (.plan)    │ │
│  │ JAX             │          │  rank*.safetensors)      │            │ │
│  └─────────────────┘          └─────────────────┘        └────────────┘ │
│                                                                          │
│  convert_checkpoint.py         trtllm-build              LLM/Executor   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **LLM Class** | High-level API for inference | `tensorrt_llm/llmapi/llm.py` |
| **Executor** | C++ request management, batching | `cpp/tensorrt_llm/executor/` |
| **PretrainedModel** | Base class for all models | `tensorrt_llm/models/modeling_utils.py` |
| **PyExecutor** | Python worker orchestration | Runtime layer |
| **ModelRunnerCpp** | Engine loading and generation | `tensorrt_llm/runtime/model_runner_cpp.py` |

### 1.3 Runtime Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Class (Entry Point)                       │
│                       generate() API                             │
├─────────────────────────────────────────────────────────────────┤
│                    PyExecutor (Worker)                           │
│  ┌──────────┐ ┌─────────────┐ ┌───────────┐ ┌──────────┐       │
│  │Scheduler │ │KVCacheManager│ │ModelEngine │ │ Sampler  │       │
│  └──────────┘ └─────────────┘ └───────────┘ └──────────┘       │
├─────────────────────────────────────────────────────────────────┤
│              C++ Executor API (bindings.cpp)                     │
├─────────────────────────────────────────────────────────────────┤
│                   TensorRT Engine (.plan)                        │
│            Compiled network graph + embedded weights             │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 Multi-GPU Support

TRT-LLM supports three parallelism strategies:

- **Tensor Parallelism (TP)**: Splits weight matrices across GPUs
- **Pipeline Parallelism (PP)**: Splits layers across GPUs  
- **Expert Parallelism (EP)**: Distributes MoE experts across GPUs

Configuration is stored in the checkpoint's `config.json`:

```json
{
    "mapping": {
        "world_size": 8,
        "tp_size": 4,
        "pp_size": 2
    }
}
```

---

## 2. Weight Handling in TRT-LLM

### 2.1 Checkpoint Format

TRT-LLM uses its own checkpoint format:

```
checkpoint_dir/
├── config.json           # Model hyperparameters + mapping
├── rank0.safetensors     # Weights for GPU rank 0
├── rank1.safetensors     # Weights for GPU rank 1
└── ...                   # One file per TP/PP rank
```

**config.json structure**:

```json
{
    "architecture": "LlamaForCausalLM",
    "dtype": "float16",
    "vocab_size": 32000,
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "mapping": {
        "world_size": 2,
        "tp_size": 2,
        "pp_size": 1
    },
    "quantization": {
        "quant_algo": "FP8",
        "kv_cache_quant_algo": "FP8"
    }
}
```

### 2.2 Weight Naming Convention

TRT-LLM uses hierarchical naming:

```python
# Attention weights
"transformer.layers.0.attention.qkv.weight"
"transformer.layers.0.attention.qkv.bias"
"transformer.layers.0.attention.dense.weight"

# MLP weights
"transformer.layers.0.mlp.fc.weight"       # Gate projection
"transformer.layers.0.mlp.proj.weight"     # Down projection

# LayerNorm
"transformer.layers.0.input_layernorm.weight"
"transformer.layers.0.post_layernorm.weight"

# Embeddings
"transformer.vocab_embedding.weight"
"lm_head.weight"

# FP8 scaling factors
"transformer.layers.0.attention.qkv.weights_scaling_factor"
"transformer.layers.0.attention.qkv.activation_scaling_factor"
```

### 2.3 Standard Engine (Embedded Weights)

By default, TRT-LLM embeds weights directly into the compiled engine:

```
┌────────────────────────────────────────┐
│           TensorRT Engine (.plan)       │
│  ┌────────────────────────────────┐    │
│  │    Compiled Network Graph      │    │
│  ├────────────────────────────────┤    │
│  │    Embedded Weights (Large!)   │    │  ← Weights baked in at build time
│  │    - All layer weights         │    │
│  │    - Embedding tables          │    │
│  │    - LayerNorm parameters      │    │
│  └────────────────────────────────┘    │
└────────────────────────────────────────┘
```

**Implication**: Standard engines cannot have their weights modified after build.

### 2.4 Weight-Stripped Engine (External Weights)

TRT-LLM supports weight-stripped engines via `--strip_plan`:

```bash
trtllm-build \
    --checkpoint_dir ./checkpoint \
    --output_dir ./engine \
    --strip_plan \           # ← Creates weight-stripped engine
    --gemm_plugin float16 \
    --max_batch_size 8
```

This creates a smaller engine that loads weights at runtime via TensorRT's `IRefitter` API.

```
┌────────────────────────────────────────┐
│     Weight-Stripped Engine (.plan)      │
│  ┌────────────────────────────────┐    │     ┌─────────────────┐
│  │    Compiled Network Graph      │    │     │ External Weights│
│  │    (Small - No Weights!)       │    │  +  │ (.safetensors)  │
│  │    - Weight placeholders       │    │     │                 │
│  └────────────────────────────────┘    │     └─────────────────┘
└────────────────────────────────────────┘
```

---

## 3. Comparison with vLLM Architecture

### 3.1 vLLM Weight Loading

vLLM uses a plugin-based model loader system:

```python
# vLLM's loader registration
@register_model_loader("mx-source")
class MxSourceModelLoader(DefaultModelLoader):
    def load_model(self, vllm_config, model_config):
        model = initialize_model(...)
        self.load_weights(model, model_config)  # Load from disk
        # HOOK: Register with NIXL before FP8 processing
        self._register_raw_tensors(model, device)
        process_weights_after_loading(...)       # FP8 transform
        return model
```

**Key vLLM characteristics**:
- Weights loaded dynamically at runtime
- Model structure separate from weights
- Custom loaders can intercept weight loading
- FP8 processing happens AFTER loading

### 3.2 TRT-LLM vs vLLM

| Aspect | vLLM | TRT-LLM |
|--------|------|---------|
| **Weight storage** | Loaded at runtime | Embedded in engine (default) |
| **Custom loaders** | Plugin system (`--load-format`) | `from_hugging_face()` method |
| **Weight modification** | Any time before inference | Build time only (standard) or via IRefitter |
| **FP8 handling** | Post-load processing | Pre-computed during conversion |
| **Model format** | PyTorch modules | TensorRT engine graph |

### 3.3 Key Challenge

The fundamental challenge is that TRT-LLM's compilation step (Phase 2) "bakes" weights into the engine. This means:

1. **Standard engines**: No runtime weight modification possible
2. **Weight-stripped engines**: Weights can be loaded via IRefitter, but this is a full replacement, not a P2P transfer

---

## 4. Integration Approaches

### 4.1 Approach A: Checkpoint-Level Transfer (Recommended)

**Concept**: Transfer TRT-LLM checkpoint files (`.safetensors`) via P2P RDMA, then build engines locally.

```
Source Node                                    Target Node
┌─────────────────────┐                       ┌─────────────────────┐
│ checkpoint/         │                       │ checkpoint/         │
│ ├── config.json     │ ═══ P2P RDMA ══════► │ ├── config.json     │
│ ├── rank0.safetensors                       │ ├── rank0.safetensors
│ ├── rank1.safetensors                       │ ├── rank1.safetensors
│ └── ...             │                       │ └── ...             │
└─────────────────────┘                       └─────────────────────┘
         │                                             │
         ▼                                             ▼
   trtllm-build                                  trtllm-build
         │                                             │
         ▼                                             ▼
┌─────────────────────┐                       ┌─────────────────────┐
│ TensorRT Engine     │                       │ TensorRT Engine     │
└─────────────────────┘                       └─────────────────────┘
```

**Pros**:
- Works with existing TRT-LLM workflow
- No changes to TRT-LLM internals
- Checkpoint files are standard safetensors (easy to handle)

**Cons**:
- Requires build step on target (slow for large models)
- Two-phase process (transfer + build)

### 4.2 Approach B: IRefitter Runtime Injection

**Concept**: Build weight-stripped engines, then inject weights via RDMA + IRefitter.

```python
# Target-side weight injection
import tensorrt as trt

# 1. Deserialize weight-stripped engine
with open("engine_stripped.plan", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

# 2. Create refitter
refitter = trt.Refitter(engine, logger)

# 3. Receive weights via NIXL into GPU memory
weights_gpu = receive_weights_via_nixl(source_metadata)

# 4. Inject weights via IRefitter
for name, weight_tensor in weights_gpu.items():
    refitter.set_weights(name, trt.WeightsRole.KERNEL, weight_tensor)

# 5. Apply refit
refitter.refit_cuda_engine()
```

**Pros**:
- Fast weight updates without rebuild
- Weights stay on GPU (no CPU roundtrip)
- Similar to vLLM's approach

**Cons**:
- Requires weight-stripped engine (separate build)
- IRefitter API complexity
- Not all layers are refittable

### 4.3 Approach C: Custom Model Loader

**Concept**: Create custom `from_modelexpress()` method that loads weights via P2P.

```python
class LlamaForCausalLM(DecoderModelForCausalLM):
    @classmethod
    def from_modelexpress(cls, model_name, mx_server_url, device_id):
        """Load model weights via ModelExpress P2P transfer."""
        # 1. Create model structure
        config = cls._get_config_from_mx(model_name, mx_server_url)
        model = cls(config)
        
        # 2. Initialize NIXL
        nixl_manager = NixlTransferManager(f"trtllm-target-{device_id}", device_id)
        nixl_manager.initialize()
        
        # 3. Get source metadata
        source_meta = mx_client.get_metadata(model_name)
        
        # 4. Receive weights via RDMA
        weights = nixl_manager.receive_weights(source_meta)
        
        # 5. Load into model
        model.load(weights)
        return model
```

**Pros**:
- Deep integration (like vLLM)
- Full control over weight handling
- Can handle FP8 scaling properly

**Cons**:
- Requires TRT-LLM modifications
- Complex implementation
- Maintenance burden

### 4.4 Approach D: Disaggregated Serving Extension

**Concept**: Leverage TRT-LLM's existing UCX/NIXL support for KV cache transfer.

TRT-LLM already has experimental NIXL support for disaggregated serving (context/generation separation). We could extend this to support weight transfer.

```bash
# Enable NIXL in TRT-LLM build
./scripts/build_wheel.py --trtllm-use-nixl-kvcache-experimental

# Runtime
export TRTLLM_USE_NIXL_KVCACHE=1
```

**Pros**:
- Builds on existing infrastructure
- NVIDIA-supported approach

**Cons**:
- Experimental feature
- Primarily designed for KV cache, not weights
- May require TRT-LLM modifications

---

## 5. Recommended Design: Checkpoint-Level P2P

For the initial implementation, we recommend **Approach A: Checkpoint-Level Transfer** as it provides the best balance of functionality, complexity, and maintainability.

### 5.1 Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ModelExpress P2P for TRT-LLM                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Source Node                              Target Node                     │
│  ┌─────────────────────────┐              ┌─────────────────────────┐    │
│  │ TRT-LLM (Running)       │              │ MxTrtllmTargetLoader    │    │
│  │ - Engine loaded         │              │ - Initialize NIXL       │    │
│  │ - Checkpoint available  │              │ - Query MX server       │    │
│  │                         │              │ - Receive checkpoint    │    │
│  │ MxTrtllmSourcePublisher │              │ - Build engine locally  │    │
│  │ - Register checkpoint   │    RDMA      │ - Start inference       │    │
│  │ - Publish to MX server  │ ════════════►│                         │    │
│  └───────────┬─────────────┘              └───────────┬─────────────┘    │
│              │                                        │                   │
│              │         ┌──────────────────┐           │                   │
│              └────────►│  MX Server       │◄──────────┘                   │
│                        │  (gRPC + Redis)  │                               │
│                        └──────────────────┘                               │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Source Side: MxTrtllmSourcePublisher

```python
class MxTrtllmSourcePublisher:
    """
    Publishes TRT-LLM checkpoint for P2P transfer.
    
    Runs alongside TRT-LLM inference to serve weights to targets.
    """
    
    def __init__(self, checkpoint_dir: str, model_name: str, mx_server: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.mx_server = mx_server
        self.nixl_managers: dict[int, NixlTransferManager] = {}
        
    def initialize(self):
        """Load checkpoint files and register with NIXL."""
        # 1. Parse config.json
        config_path = self.checkpoint_dir / "config.json"
        with open(config_path) as f:
            self.config = json.load(f)
        
        world_size = self.config["mapping"]["world_size"]
        
        # 2. Load and register each rank's weights
        for rank in range(world_size):
            weights_path = self.checkpoint_dir / f"rank{rank}.safetensors"
            weights = self._load_safetensors_to_gpu(weights_path, rank)
            
            # Initialize NIXL manager for this rank
            nixl_manager = NixlTransferManager(
                agent_name=f"trtllm-source-rank{rank}",
                device_id=rank
            )
            nixl_manager.initialize()
            nixl_manager.register_tensors(weights)
            self.nixl_managers[rank] = nixl_manager
        
        # 3. Publish metadata to MX server
        self._publish_to_mx_server()
    
    def _load_safetensors_to_gpu(self, path: Path, device_id: int) -> dict:
        """Load safetensors file directly to GPU memory."""
        import safetensors.torch
        
        with torch.cuda.device(device_id):
            weights = safetensors.torch.load_file(str(path), device=f"cuda:{device_id}")
        return weights
    
    def _publish_to_mx_server(self):
        """Publish tensor metadata to ModelExpress server."""
        import grpc
        from modelexpress import p2p_pb2, p2p_pb2_grpc
        
        channel = grpc.insecure_channel(self.mx_server)
        stub = p2p_pb2_grpc.P2pServiceStub(channel)
        
        workers = []
        for rank, nixl_manager in self.nixl_managers.items():
            # Build tensor descriptors
            tensor_protos = []
            for name, tensor in nixl_manager.registered_tensors.items():
                tensor_protos.append(p2p_pb2.TensorDescriptor(
                    name=name,
                    addr=tensor.data_ptr(),
                    size=tensor.numel() * tensor.element_size(),
                    device_id=rank,
                    dtype=str(tensor.dtype),
                ))
            
            workers.append(p2p_pb2.WorkerMetadata(
                worker_rank=rank,
                nixl_metadata=nixl_manager.nixl_metadata,
                tensors=tensor_protos,
            ))
        
        request = p2p_pb2.PublishMetadataRequest(
            model_name=self.model_name,
            workers=workers,
        )
        stub.PublishMetadata(request)
```

### 5.3 Target Side: MxTrtllmTargetLoader

```python
class MxTrtllmTargetLoader:
    """
    Receives TRT-LLM checkpoint via P2P and builds engine.
    """
    
    def __init__(self, model_name: str, mx_server: str, output_dir: str):
        self.model_name = model_name
        self.mx_server = mx_server
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoint"
        
    def load(self) -> str:
        """
        Receive weights via P2P and build TRT-LLM engine.
        
        Returns:
            Path to built engine directory
        """
        # 1. Query MX server for source
        source_meta = self._query_source()
        
        # 2. Receive weights via NIXL for each rank
        self._receive_checkpoint(source_meta)
        
        # 3. Build TRT-LLM engine
        engine_dir = self._build_engine()
        
        return str(engine_dir)
    
    def _query_source(self):
        """Query ModelExpress server for source metadata."""
        import grpc
        from modelexpress import p2p_pb2, p2p_pb2_grpc
        
        channel = grpc.insecure_channel(self.mx_server)
        stub = p2p_pb2_grpc.P2pServiceStub(channel)
        
        # Wait for source to be available
        max_wait = 3600  # 1 hour
        waited = 0
        while waited < max_wait:
            request = p2p_pb2.GetMetadataRequest(model_name=self.model_name)
            response = stub.GetMetadata(request)
            
            if response.found and len(response.workers) > 0:
                return response
            
            time.sleep(30)
            waited += 30
        
        raise TimeoutError(f"Source not found after {max_wait}s")
    
    def _receive_checkpoint(self, source_meta):
        """Receive checkpoint files via NIXL."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy config from source (or receive separately)
        # For now, assume config is available via MX server
        
        for worker in source_meta.workers:
            rank = worker.worker_rank
            
            # Initialize target NIXL manager
            nixl_manager = NixlTransferManager(
                agent_name=f"trtllm-target-rank{rank}",
                device_id=rank
            )
            nixl_manager.initialize()
            
            # Allocate GPU memory for weights
            weights = self._allocate_weight_tensors(worker.tensors, rank)
            nixl_manager.register_tensors(weights)
            
            # Receive via RDMA
            nixl_manager.receive_from_source(
                source_metadata=worker.nixl_metadata,
                source_tensors=worker.tensors,
            )
            
            # Save to safetensors
            import safetensors.torch
            weights_path = self.checkpoint_dir / f"rank{rank}.safetensors"
            
            # Move to CPU for saving
            cpu_weights = {k: v.cpu() for k, v in weights.items()}
            safetensors.torch.save_file(cpu_weights, str(weights_path))
    
    def _allocate_weight_tensors(self, tensor_protos, device_id):
        """Allocate GPU tensors matching source layout."""
        weights = {}
        with torch.cuda.device(device_id):
            for t in tensor_protos:
                dtype = self._parse_dtype(t.dtype)
                shape = self._infer_shape(t.size, dtype)
                weights[t.name] = torch.empty(shape, dtype=dtype, device=f"cuda:{device_id}")
        return weights
    
    def _build_engine(self) -> Path:
        """Build TRT-LLM engine from received checkpoint."""
        import subprocess
        
        engine_dir = self.output_dir / "engine"
        engine_dir.mkdir(parents=True, exist_ok=True)
        
        # Call trtllm-build
        cmd = [
            "trtllm-build",
            "--checkpoint_dir", str(self.checkpoint_dir),
            "--output_dir", str(engine_dir),
            "--gemm_plugin", "float16",
            "--max_batch_size", "8",
            "--max_input_len", "2048",
            "--max_seq_len", "4096",
        ]
        
        subprocess.run(cmd, check=True)
        return engine_dir
```

### 5.4 Integration with TRT-LLM High-Level API

```python
from tensorrt_llm import LLM

def create_llm_from_mx(model_name: str, mx_server: str, local_dir: str) -> LLM:
    """
    Create TRT-LLM LLM instance using ModelExpress P2P transfer.
    
    Args:
        model_name: Model identifier in ModelExpress
        mx_server: ModelExpress server address (host:port)
        local_dir: Local directory for checkpoint and engine
        
    Returns:
        Ready-to-use LLM instance
    """
    loader = MxTrtllmTargetLoader(model_name, mx_server, local_dir)
    engine_dir = loader.load()
    
    # Create LLM from engine
    llm = LLM(model=engine_dir)
    return llm
```

---

## 6. Alternative Design: IRefitter-Based Transfer

For scenarios requiring faster weight updates without rebuild, we can use TensorRT's IRefitter API.

### 6.1 Architecture

```
Source Node                                    Target Node
┌─────────────────────────┐                   ┌─────────────────────────┐
│ TRT-LLM + NIXL          │                   │ Weight-Stripped Engine  │
│ - Weights on GPU        │                   │ - Empty weight slots    │
│ - Registered with NIXL  │  RDMA Transfer    │                         │
│                         │ ════════════════► │ IRefitter               │
│                         │                   │ - set_weights()         │
│                         │                   │ - refit_cuda_engine()   │
└─────────────────────────┘                   └─────────────────────────┘
```

### 6.2 Weight-Stripped Engine Creation

```bash
# Build weight-stripped engine
trtllm-build \
    --checkpoint_dir ./checkpoint \
    --output_dir ./engine_stripped \
    --strip_plan \
    --gemm_plugin float16 \
    --max_batch_size 8
```

### 6.3 IRefitter-Based Target Loader

```python
import tensorrt as trt

class MxTrtllmIRefitterLoader:
    """
    Loads weights into weight-stripped TRT-LLM engine via IRefitter.
    """
    
    def __init__(self, engine_path: str, model_name: str, mx_server: str):
        self.engine_path = engine_path
        self.model_name = model_name
        self.mx_server = mx_server
        self.logger = trt.Logger(trt.Logger.INFO)
        
    def load(self):
        """Load weights via P2P and inject into engine."""
        # 1. Deserialize weight-stripped engine
        runtime = trt.Runtime(self.logger)
        with open(self.engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        # 2. Create refitter
        refitter = trt.Refitter(engine, self.logger)
        
        # 3. Get refittable weights
        refittable_weights = refitter.get_all()
        print(f"Refittable weights: {len(refittable_weights)}")
        
        # 4. Receive weights via NIXL
        source_meta = self._query_source()
        weights = self._receive_weights(source_meta, refittable_weights)
        
        # 5. Set weights via IRefitter
        for (name, role) in refittable_weights:
            if name in weights:
                weight_tensor = weights[name]
                trt_weights = self._tensor_to_trt_weights(weight_tensor)
                success = refitter.set_weights(name, role, trt_weights)
                if not success:
                    print(f"Failed to set weights for {name}")
        
        # 6. Check for missing weights
        missing = refitter.get_missing()
        if missing:
            raise RuntimeError(f"Missing weights: {missing}")
        
        # 7. Apply refit
        success = refitter.refit_cuda_engine()
        if not success:
            raise RuntimeError("Failed to refit engine")
        
        return engine
    
    def _tensor_to_trt_weights(self, tensor: torch.Tensor) -> trt.Weights:
        """Convert PyTorch tensor to TensorRT Weights."""
        # Ensure tensor is contiguous and on CPU
        tensor = tensor.contiguous().cpu()
        return trt.Weights(tensor.numpy())
```

### 6.4 Considerations

**IRefitter Limitations**:
- Not all layers are refittable (depends on TensorRT version)
- Weight shapes must match exactly
- Requires weight-stripped engine (separate build artifact)
- Some quantization configurations may not support refitting

---

## 7. Implementation Plan

### Phase 1: Checkpoint-Level Transfer (MVP)

1. **MxTrtllmSourcePublisher**
   - Load TRT-LLM checkpoint to GPU
   - Register with NIXL
   - Publish to MX server

2. **MxTrtllmTargetLoader**
   - Query MX server for source
   - Receive checkpoint via NIXL
   - Save to local storage
   - Build engine

3. **Integration Tests**
   - Single-node transfer
   - Multi-GPU (TP=4) transfer
   - DeepSeek-V3 scale test

### Phase 2: IRefitter Integration (Optional)

1. **Weight-stripped engine support**
2. **IRefitter-based loader**
3. **GPU-to-GPU weight injection** (avoid CPU roundtrip)

### Phase 3: Triton Integration

1. **Triton backend extension**
2. **Model repository integration**
3. **Dynamic weight updates**

---

## 8. Code Changes Required

### 8.1 New Files

```
modelexpress_client/python/modelexpress/
├── trtllm_loader.py        # MxTrtllmSourcePublisher, MxTrtllmTargetLoader
├── trtllm_refitter.py      # IRefitter-based loader (Phase 2)
└── trtllm_utils.py         # Shared utilities
```

### 8.2 trtllm_loader.py

```python
# modelexpress_client/python/modelexpress/trtllm_loader.py

"""
ModelExpress TensorRT-LLM Integration.

Enables P2P weight transfer for TRT-LLM checkpoints.

Usage:
    # Source side
    publisher = MxTrtllmSourcePublisher(
        checkpoint_dir="/path/to/checkpoint",
        model_name="llama-70b",
        mx_server="modelexpress-server:8001"
    )
    publisher.initialize()
    
    # Target side
    loader = MxTrtllmTargetLoader(
        model_name="llama-70b",
        mx_server="modelexpress-server:8001",
        output_dir="/path/to/output"
    )
    engine_dir = loader.load()
    
    # Use with TRT-LLM
    from tensorrt_llm import LLM
    llm = LLM(model=engine_dir)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.trtllm_loader")


class MxTrtllmSourcePublisher:
    """
    Publishes TRT-LLM checkpoint for P2P transfer.
    
    This class loads checkpoint files into GPU memory, registers them
    with NIXL for RDMA access, and publishes metadata to the ModelExpress
    server for targets to discover.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str,
        mx_server: str = "modelexpress-server:8001"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.mx_server = mx_server
        self.config: dict = {}
        self.nixl_managers: dict[int, NixlTransferManager] = {}
        self.weights: dict[int, dict[str, torch.Tensor]] = {}
        
    def initialize(self):
        """Load checkpoint and register with NIXL."""
        from .nixl_transfer import NixlTransferManager
        
        logger.info(f"Initializing TRT-LLM source publisher for {self.model_name}")
        
        # 1. Parse config.json
        config_path = self.checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path) as f:
            self.config = json.load(f)
        
        world_size = self.config.get("mapping", {}).get("world_size", 1)
        logger.info(f"Checkpoint has {world_size} ranks")
        
        # 2. Load and register each rank's weights
        for rank in range(world_size):
            weights_path = self.checkpoint_dir / f"rank{rank}.safetensors"
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights not found: {weights_path}")
            
            logger.info(f"Loading rank {rank} weights from {weights_path}")
            weights = self._load_safetensors_to_gpu(weights_path, rank)
            self.weights[rank] = weights
            
            total_size = sum(t.numel() * t.element_size() for t in weights.values())
            logger.info(f"Rank {rank}: {len(weights)} tensors, {total_size / 1e9:.2f} GB")
            
            # Initialize NIXL manager
            nixl_manager = NixlTransferManager(
                agent_name=f"trtllm-source-rank{rank}",
                device_id=rank
            )
            nixl_manager.initialize()
            nixl_manager.register_tensors(weights)
            self.nixl_managers[rank] = nixl_manager
            logger.info(f"Rank {rank}: Registered with NIXL")
        
        # 3. Publish to MX server
        self._publish_to_mx_server()
        logger.info("Published metadata to ModelExpress server")
    
    def _load_safetensors_to_gpu(
        self, path: Path, device_id: int
    ) -> dict[str, torch.Tensor]:
        """Load safetensors file directly to GPU memory."""
        import safetensors.torch
        
        with torch.cuda.device(device_id):
            weights = safetensors.torch.load_file(
                str(path), device=f"cuda:{device_id}"
            )
        return weights
    
    def _publish_to_mx_server(self):
        """Publish tensor metadata to ModelExpress server."""
        import grpc
        from . import p2p_pb2, p2p_pb2_grpc
        
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
        channel = grpc.insecure_channel(self.mx_server, options=options)
        stub = p2p_pb2_grpc.P2pServiceStub(channel)
        
        workers = []
        for rank, nixl_manager in self.nixl_managers.items():
            weights = self.weights[rank]
            
            tensor_protos = []
            for name, tensor in weights.items():
                tensor_protos.append(p2p_pb2.TensorDescriptor(
                    name=name,
                    addr=tensor.data_ptr(),
                    size=tensor.numel() * tensor.element_size(),
                    device_id=rank,
                    dtype=str(tensor.dtype),
                ))
            
            workers.append(p2p_pb2.WorkerMetadata(
                worker_rank=rank,
                nixl_metadata=nixl_manager.nixl_metadata,
                tensors=tensor_protos,
            ))
        
        request = p2p_pb2.PublishMetadataRequest(
            model_name=self.model_name,
            workers=workers,
        )
        response = stub.PublishMetadata(request)
        
        if not response.success:
            raise RuntimeError(f"Failed to publish: {response.message}")
        
        channel.close()
    
    def shutdown(self):
        """Clean up resources."""
        for nixl_manager in self.nixl_managers.values():
            nixl_manager.destroy()
        self.nixl_managers.clear()
        self.weights.clear()


class MxTrtllmTargetLoader:
    """
    Receives TRT-LLM checkpoint via P2P and builds engine.
    
    This class queries the ModelExpress server for source metadata,
    receives checkpoint weights via NIXL RDMA, saves them locally,
    and builds a TRT-LLM engine.
    """
    
    def __init__(
        self,
        model_name: str,
        mx_server: str = "modelexpress-server:8001",
        output_dir: str = "/tmp/mx_trtllm",
        build_config: dict | None = None,
    ):
        self.model_name = model_name
        self.mx_server = mx_server
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoint"
        self.engine_dir = self.output_dir / "engine"
        self.build_config = build_config or {
            "gemm_plugin": "float16",
            "max_batch_size": 8,
            "max_input_len": 2048,
            "max_seq_len": 4096,
        }
        
    def load(self, skip_build: bool = False) -> str:
        """
        Receive weights via P2P and optionally build engine.
        
        Args:
            skip_build: If True, only receive checkpoint without building
            
        Returns:
            Path to checkpoint (if skip_build) or engine directory
        """
        logger.info(f"Loading {self.model_name} via ModelExpress P2P")
        
        # 1. Query MX server for source
        source_meta, config = self._query_source()
        
        # 2. Save config
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # 3. Receive weights via NIXL
        self._receive_checkpoint(source_meta)
        
        if skip_build:
            return str(self.checkpoint_dir)
        
        # 4. Build engine
        return self._build_engine()
    
    def _query_source(self, timeout: int = 3600):
        """Query ModelExpress server for source metadata."""
        import grpc
        from . import p2p_pb2, p2p_pb2_grpc
        
        options = [
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
        channel = grpc.insecure_channel(self.mx_server, options=options)
        stub = p2p_pb2_grpc.P2pServiceStub(channel)
        
        logger.info(f"Querying for source: {self.model_name}")
        
        start = time.time()
        while time.time() - start < timeout:
            request = p2p_pb2.GetMetadataRequest(model_name=self.model_name)
            response = stub.GetMetadata(request)
            
            if response.found and len(response.workers) > 0:
                logger.info(f"Found source with {len(response.workers)} workers")
                channel.close()
                
                # Parse config from response (stored in model_config field)
                config = json.loads(response.model_config) if response.model_config else {}
                return response, config
            
            logger.info("Source not ready, waiting...")
            time.sleep(30)
        
        channel.close()
        raise TimeoutError(f"Source not found after {timeout}s")
    
    def _receive_checkpoint(self, source_meta):
        """Receive checkpoint weights via NIXL."""
        from .nixl_transfer import NixlTransferManager
        from .types import TensorDescriptor
        
        for worker in source_meta.workers:
            rank = worker.worker_rank
            logger.info(f"Receiving rank {rank} ({len(worker.tensors)} tensors)")
            
            # Initialize NIXL manager
            nixl_manager = NixlTransferManager(
                agent_name=f"trtllm-target-rank{rank}",
                device_id=rank
            )
            nixl_manager.initialize()
            
            # Allocate GPU tensors
            weights = self._allocate_tensors(worker.tensors, rank)
            nixl_manager.register_tensors(weights)
            
            # Build source descriptors
            source_tensors = [
                TensorDescriptor(
                    name=t.name,
                    addr=t.addr,
                    size=t.size,
                    device_id=t.device_id,
                    dtype=t.dtype,
                )
                for t in worker.tensors
            ]
            
            # Receive via RDMA
            t0 = time.perf_counter()
            bytes_transferred, _, _ = nixl_manager.receive_from_source(
                source_metadata=worker.nixl_metadata,
                source_tensors=source_tensors,
                timeout_seconds=300,
            )
            transfer_time = time.perf_counter() - t0
            
            bandwidth = (bytes_transferred * 8) / (transfer_time * 1e9)
            logger.info(
                f"Rank {rank}: Received {bytes_transferred / 1e9:.2f} GB "
                f"in {transfer_time:.2f}s ({bandwidth:.1f} Gbps)"
            )
            
            # Save to safetensors
            self._save_weights(weights, rank)
            
            # Cleanup
            nixl_manager.destroy()
    
    def _allocate_tensors(
        self, tensor_protos, device_id: int
    ) -> dict[str, torch.Tensor]:
        """Allocate GPU tensors matching source layout."""
        weights = {}
        
        with torch.cuda.device(device_id):
            for t in tensor_protos:
                dtype = self._parse_dtype(t.dtype)
                numel = t.size // self._dtype_size(dtype)
                # Create 1D tensor (will be reshaped by TRT-LLM)
                weights[t.name] = torch.empty(
                    numel, dtype=dtype, device=f"cuda:{device_id}"
                )
        
        return weights
    
    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        """Parse dtype string to torch.dtype."""
        dtype_map = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.bfloat16": torch.bfloat16,
            "torch.int8": torch.int8,
            "torch.int32": torch.int32,
            "torch.float8_e4m3fn": torch.float8_e4m3fn,
        }
        return dtype_map.get(dtype_str, torch.float16)
    
    def _dtype_size(self, dtype: torch.dtype) -> int:
        """Get size in bytes for dtype."""
        return torch.tensor([], dtype=dtype).element_size()
    
    def _save_weights(self, weights: dict[str, torch.Tensor], rank: int):
        """Save weights to safetensors file."""
        import safetensors.torch
        
        # Move to CPU for saving
        cpu_weights = {k: v.cpu() for k, v in weights.items()}
        
        weights_path = self.checkpoint_dir / f"rank{rank}.safetensors"
        safetensors.torch.save_file(cpu_weights, str(weights_path))
        logger.info(f"Saved rank {rank} to {weights_path}")
    
    def _build_engine(self) -> str:
        """Build TRT-LLM engine from checkpoint."""
        logger.info("Building TRT-LLM engine...")
        
        self.engine_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "trtllm-build",
            "--checkpoint_dir", str(self.checkpoint_dir),
            "--output_dir", str(self.engine_dir),
        ]
        
        for key, value in self.build_config.items():
            cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Build failed: {result.stderr}")
            raise RuntimeError(f"trtllm-build failed: {result.stderr}")
        
        logger.info(f"Engine built at {self.engine_dir}")
        return str(self.engine_dir)


def create_llm_from_mx(
    model_name: str,
    mx_server: str = "modelexpress-server:8001",
    output_dir: str = "/tmp/mx_trtllm",
    build_config: dict | None = None,
):
    """
    Create TRT-LLM LLM instance using ModelExpress P2P transfer.
    
    Args:
        model_name: Model identifier in ModelExpress
        mx_server: ModelExpress server address
        output_dir: Local directory for checkpoint and engine
        build_config: Optional trtllm-build configuration
        
    Returns:
        Ready-to-use tensorrt_llm.LLM instance
    """
    from tensorrt_llm import LLM
    
    loader = MxTrtllmTargetLoader(
        model_name=model_name,
        mx_server=mx_server,
        output_dir=output_dir,
        build_config=build_config,
    )
    
    engine_dir = loader.load()
    return LLM(model=engine_dir)
```

### 8.3 Updates to __init__.py

```python
# modelexpress_client/python/modelexpress/__init__.py

from .trtllm_loader import (
    MxTrtllmSourcePublisher,
    MxTrtllmTargetLoader,
    create_llm_from_mx,
)

__all__ = [
    # Existing exports...
    "MxTrtllmSourcePublisher",
    "MxTrtllmTargetLoader",
    "create_llm_from_mx",
]
```

---

## 9. Kubernetes Deployment

### 9.1 Source Deployment (trtllm-source.yaml)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trtllm-source
  labels:
    app: trtllm-source
spec:
  replicas: 1
  selector:
    matchLabels:
      app: trtllm-source
  template:
    metadata:
      labels:
        app: trtllm-source
    spec:
      containers:
        - name: trtllm
          image: nvcr.io/nvidia/tritonserver:24.01-trtllm-python-py3
          env:
            - name: MODEL_NAME
              value: "llama-70b"
            - name: MODEL_EXPRESS_URL
              value: "modelexpress-server:8001"
            - name: CHECKPOINT_DIR
              value: "/models/llama-70b/checkpoint"
          command: ["python3"]
          args:
            - -c
            - |
              from modelexpress.trtllm_loader import MxTrtllmSourcePublisher
              import time
              
              publisher = MxTrtllmSourcePublisher(
                  checkpoint_dir="/models/llama-70b/checkpoint",
                  model_name="llama-70b",
                  mx_server="modelexpress-server:8001"
              )
              publisher.initialize()
              
              # Keep running to serve weights
              print("Source ready, serving weights...")
              while True:
                  time.sleep(3600)
          resources:
            limits:
              nvidia.com/gpu: "8"
              rdma/ib: "8"
          volumeMounts:
            - name: model-cache
              mountPath: /models
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache
```

### 9.2 Target Deployment (trtllm-target.yaml)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trtllm-target
  labels:
    app: trtllm-target
spec:
  replicas: 1
  selector:
    matchLabels:
      app: trtllm-target
  template:
    metadata:
      labels:
        app: trtllm-target
    spec:
      containers:
        - name: trtllm
          image: nvcr.io/nvidia/tritonserver:24.01-trtllm-python-py3
          env:
            - name: MODEL_NAME
              value: "llama-70b"
            - name: MODEL_EXPRESS_URL
              value: "modelexpress-server:8001"
          command: ["python3"]
          args:
            - -c
            - |
              from modelexpress.trtllm_loader import create_llm_from_mx
              from tensorrt_llm import SamplingParams
              
              # Receive weights via P2P and build engine
              llm = create_llm_from_mx(
                  model_name="llama-70b",
                  mx_server="modelexpress-server:8001",
                  output_dir="/tmp/mx_trtllm"
              )
              
              # Test inference
              output = llm.generate("Hello, my name is", SamplingParams(max_tokens=50))
              print(f"Output: {output}")
              
              # Start serving...
          resources:
            limits:
              nvidia.com/gpu: "8"
              rdma/ib: "8"
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchLabels:
                  app: trtllm-source
              topologyKey: kubernetes.io/hostname
```

---

## 10. Performance Considerations

### 10.1 Transfer Performance

Based on vLLM benchmarks:

| Model | Size | Transfer Time (RDMA) | Transfer Time (NVMe) |
|-------|------|---------------------|---------------------|
| Llama-70B | 140 GB | ~5s | ~3 min |
| DeepSeek-V3 | 681 GB | 40-80s | ~25 min |

### 10.2 Build Time Impact

TRT-LLM engine build times vary significantly:

| Model | TP Size | Build Time |
|-------|---------|------------|
| Llama-7B | 1 | ~2-5 min |
| Llama-70B | 4 | ~10-20 min |
| DeepSeek-V3 | 8 | ~30-60 min |

**Total time = Transfer + Build**

For DeepSeek-V3 with TP=8:
- P2P Transfer: ~60s
- Engine Build: ~45 min
- **Total: ~46 min** (vs ~25 min load + ~45 min build = ~70 min from NVMe)

### 10.3 Optimization Opportunities

1. **Parallel Build**: Start build while transfer is in progress (requires streaming checkpoint)
2. **Engine Caching**: Cache built engines to avoid rebuild
3. **IRefitter Path**: For scenarios where engine is pre-built, skip build step entirely
4. **Multi-Source**: Fan-out transfer from multiple sources for higher bandwidth

---

## 11. Open Questions and Risks

### 11.1 Open Questions

1. **FP8 Handling**: How does TRT-LLM handle FP8 scaling factors in checkpoints?
   - vLLM transforms `weight_scale_inv` → `weight_scale` post-load
   - TRT-LLM may pre-compute during conversion

2. **IRefitter Compatibility**: Which TRT-LLM layers support IRefitter?
   - Need to test with specific model architectures
   - Quantized layers may have restrictions

3. **Triton Integration**: Best way to integrate with Triton backend?
   - Custom backend extension?
   - Pre-load hook?

4. **Config Transfer**: How to efficiently transfer `config.json`?
   - Include in MX server metadata?
   - Separate transfer channel?

### 11.2 Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Build time dominates | High | Engine caching, IRefitter path |
| IRefitter limitations | Medium | Fall back to checkpoint transfer |
| TRT-LLM version changes | Medium | Version pinning, abstraction layer |
| NIXL availability | Low | Already supported in TRT-LLM (experimental) |

### 11.3 Future Work

1. **Triton Backend Integration**: Native support in tensorrtllm_backend
2. **Engine Streaming**: Stream engine build during transfer
3. **Multi-Model Support**: Concurrent transfers for multiple models
4. **LoRA Integration**: P2P transfer for LoRA adapters

---

## References

1. [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
2. [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
3. [TensorRT IRefitter API](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_refitter.html)
4. [NIXL Documentation](https://github.com/NVIDIA/nixl)
5. [Triton TensorRT-LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend)
