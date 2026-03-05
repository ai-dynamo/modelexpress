# GDS Model Loader

ModelExpress provides a GPUDirect Storage (GDS) model loader that reads safetensors weights directly from NVMe storage into GPU memory, bypassing CPU bounce buffers entirely. It is built on top of [NIXL](https://github.com/ai-dynamo/nixl)'s GDS backend.

## How It Works

Traditional model loading copies data through multiple hops:
```
NVMe → Page Cache → CPU Memory → GPU Memory
```

With GDS, the path is shortened to a single DMA transfer:

```
NVMe → GPU Memory
```

The loader parses safetensors file headers to determine tensor offsets, then issues GDS reads directly into pre-allocated GPU tensors. It handles 4KB alignment constraints and automatic chunking required by the cuFile API transparently.

## Requirements

- NVIDIA GPU with CUDA driver
- GDS kernel module (`nvidia-gds`) installed and `cufile` driver loaded
- `nixl` with GDS support: `pip install nixl[cu12]`

## Framework Integration

The loader has a framework-agnostic core (`MxGdsLoader`) with thin wrappers for vLLM and sglang.

### vLLM

```bash
vllm serve <model> \
  --load-format mx-gds \
  --worker-cls modelexpress.vllm_worker.ModelExpressWorker
```

The `--worker-cls` flag is required to register the `mx-gds` loader before vLLM's engine starts.

**Example** (Qwen2.5-7B):

```bash
$ vllm serve Qwen/Qwen2.5-7B \
    --load-format mx-gds \
    --worker-cls modelexpress.vllm_worker.ModelExpressWorker

Initialized NIXL agent: mx-gds-0-943c39d1
GDS agent 'mx-gds-0-943c39d1' created on device 0 (max_chunk=1024KB)
GDS manager initialized for device 0
Found sharded model: 4 files, 339 tensors
Loaded model-00004-of-00004.safetensors
Loaded model-00001-of-00004.safetensors
Loaded model-00002-of-00004.safetensors
Loaded model-00003-of-00004.safetensors
GDS load complete: 15.23 GB in 11.51s
GDS weight loading complete
Model loading took 14.27 GiB memory and 13.070087 seconds
```

### SGLang

```bash
python -m sglang.launch_server --model <model> --load-format mx_gds
```

**Example** (Qwen2.5-7B):

```bash
$ python -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B \
    --load-format mx_gds

Initialized NIXL agent: mx-gds-0-5b63bc0c
GDS agent 'mx-gds-0-5b63bc0c' created on device 0 (max_chunk=1024KB)
GDS manager initialized for device 0
Found sharded model: 4 files, 339 tensors
Loaded model-00004-of-00004.safetensors
Loaded model-00001-of-00004.safetensors
Loaded model-00002-of-00004.safetensors
Loaded model-00003-of-00004.safetensors
GDS load complete: 15.23 GB in 12.73s
GDS weight loading complete
Load weight end. elapsed=14.13 s, type=Qwen2ForCausalLM, dtype=torch.bfloat16, avail mem=32.29 GB, mem usage=14.35 GB
```

### Python API (Framework-Agnostic)

The core loader can be used directly from any Python code:

```python
from modelexpress import MxGdsLoader

loader = MxGdsLoader()

# Load all tensors at once
tensors = loader.load("/path/to/model")

# Or iterate per-file (lower peak memory)
for name, tensor in loader.load_iter("/path/to/model"):
    print(f"{name}: {tensor.shape} on {tensor.device}")

loader.shutdown()
```

## Architecture

```
modelexpress/
├── gds_loader.py          # MxGdsLoader — framework-agnostic core
├── gds_transfer.py        # GdsTransferManager — NIXL GDS agent wrapper
├── vllm_gds_loader.py     # vLLM wrapper (--load-format mx-gds)
└── sglang_gds_loader.py   # sglang wrapper (--load-format mx_gds)
```

- **`MxGdsLoader`** — Parses safetensors headers, discovers sharded/single file layouts, yields `(name, tensor)` pairs loaded via GDS.
- **`GdsTransferManager`** — Manages the NIXL agent lifecycle, handles 4KB alignment, chunking, and buffer reuse.
- **Framework wrappers** — Thin adapters that bridge `MxGdsLoader` into vLLM's or SGLang's model loader interface.

## Troubleshooting

**"NIXL is not available" error:**
Ensure `nixl` is installed with GDS support: `pip install nixl[cu12]`. Verify with:
```bash
python -c "from nixl._api import nixl_agent; print('OK')"
```

**Ensure GDS is available on your machine:**



