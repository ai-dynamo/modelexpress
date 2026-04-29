# NIXL Compression Study — Capture Scripts

Two scripts for producing the data described in [`../NIXL_COMPRESSION_STUDY.md`](../NIXL_COMPRESSION_STUDY.md).

| Script | When to use |
|--------|-------------|
| `capture_weights_and_kv.py` | **Standalone** — capture from any HuggingFace model on any host. Doesn't require a running RL deployment. Just downloads the model and dumps weights + KV cache. CLI flags for model, dtype, device, output dir. |
| `capture_on_pod.py` | **Inside a running RL pod** — exec into a trainer pod and capture pre-step weights, simulate one AdamW step, capture post-step weights + delta + KV cache in one pass. Produces the four-directory layout we shipped to the NIXL team for Qwen2.5-1.5B. |

## Standalone capture (any model, no cluster needed)

```bash
pip install torch transformers safetensors

# Smallest model — ~3 GB output, ~5 minutes total
python capture_weights_and_kv.py \
    --model Qwen/Qwen2.5-1.5B \
    --output-dir ./nixl_data \
    --dtype bfloat16 \
    --device cpu \
    --kv-seq-len 512

# Larger model
python capture_weights_and_kv.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output-dir ./nixl_data \
    --dtype bfloat16 \
    --device cuda:0 \
    --kv-seq-len 2048

# Weights only / KV only
python capture_weights_and_kv.py --model <name> --output-dir <path> --weights-only
python capture_weights_and_kv.py --model <name> --output-dir <path> --kv-only
```

Output layout:

```text
<output-dir>/
├── weights/<model>/
│   ├── tensors/*.bin       # one file per parameter, raw bytes (BF16)
│   └── manifest.json       # name, shape, dtype, size, layer index, classification, stats
└── kvcache/<model>/
    ├── layer_N_key.bin     # one file per (layer, key/value)
    ├── layer_N_value.bin
    └── manifest.json
```

## Capture from a running pod (with RL-step simulation)

This script is what produced the `RL_Qwen25/` package referenced in the NIXL request. It captures weights pre- and post- a simulated AdamW step, then computes the delta:

```bash
# Copy the script into the pod
kubectl cp capture_on_pod.py <namespace>/<trainer-pod>:/tmp/capture.py

# Run it (uses /app/.venv/bin/python in our overlay image; adjust if different)
kubectl exec <namespace>/<trainer-pod> -- /app/.venv/bin/python /tmp/capture.py \
    --model Qwen/Qwen2.5-1.5B \
    --out /tmp/nixl_capture \
    --kv-seq-len 512 \
    --lr 5e-6

# Copy results back
kubectl cp <namespace>/<trainer-pod>:/tmp/nixl_capture ./RL_capture
```

Output layout (matches what we shipped to the NIXL team):

```text
nixl_capture/
├── weights_pre_rl/         # pre-step weight tensors + manifest.json
├── weights_post_rl/        # post-step weight tensors + manifest.json
├── weight_deltas/          # post - pre (BF16; mostly zero — see note below)
└── kv_cache/               # one prefill pass output + manifest.json
```

### Note on BF16 deltas

A single AdamW step at `lr=5e-6` produces parameter updates of magnitude ~1e-8 to 1e-6, which is **below BF16's representable precision** at typical weight magnitudes (0.01–0.1). The `weight_deltas/` files will therefore be mostly zero in BF16.

For meaningful delta-compression analysis, compute the diff in FP32 from the pre/post safetensors:

```python
import torch
from safetensors import safe_open

with safe_open("weights_pre_rl/...", framework="pt") as pre, \
     safe_open("weights_post_rl/...", framework="pt") as post:
    for k in pre.keys():
        delta_fp32 = post.get_tensor(k).float() - pre.get_tensor(k).float()
        if delta_fp32.abs().max() > 0:
            print(f"{k}: max_abs_delta={delta_fp32.abs().max():.2e}")
```

The pre/post tensors are saved as raw BF16 (the on-the-wire dtype). The FP32 delta is the meaningful analysis target — this is the signal nvCOMP would compress in a delta-transfer scheme.

## What gets captured

See [`../NIXL_COMPRESSION_STUDY.md`](../NIXL_COMPRESSION_STUDY.md) for the full breakdown of:

- Per-tensor layout (Qwen3-0.6B and Qwen2.5-1.5B examples)
- KV cache shape + scaling table
- Compression-relevant properties
- Where these tensors fit in the NIXL transfer path
