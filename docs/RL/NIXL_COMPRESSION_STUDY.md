# NIXL nvCOMP Compression Study — Reproducing with ModelExpress RL Workflows

**Last Updated**: April 29, 2026
**Audience**: NIXL compression team (`eschmidt@nvidia.com`)
**Purpose**: Guide the NIXL team to capture and study real RL weight-transfer payloads using our validated PRIME-RL and verl workflows with ModelExpress (MX).

---

## Background

The NIXL team is evaluating nvCOMP GPU compression on the tensors that flow through NIXL during RL post-training. There are two transfer types:

1. **RL refit** (training → inference): full model weights, every RL step.
2. **KV cache** (prefill → decode): per-request KV tensors in disaggregated inference.

We have **two validated end-to-end RL workflows** that produce these payloads over NIXL on GB200:

| Workflow | Framework | Status | PR | What it exercises |
|----------|-----------|--------|-----|-------------------|
| **PRIME-RL overlay** | PRIME-RL + vLLM | Scenarios A/B/C green on GB200 (20/20 steps each) | [PrimeIntellect-ai/prime-rl#2343](https://github.com/PrimeIntellect-ai/prime-rl/pull/2343) | NIXL RDMA weight push via PI's `NIXLWeightBroadcast` + `TransportPlan`, MX-mediated discovery |
| **verl MxCheckpointEngine** | verl + vLLM | 10 steps green on GB200 | [ai-dynamo/modelexpress#252](https://github.com/ai-dynamo/modelexpress/pull/252) | NIXL RDMA weight transfer via `MxCheckpointEngine` (`CheckpointEngine` plugin) |

Both produce the **exact same kind of data** the NIXL team requested: raw BF16 weight tensors flowing GPU-to-GPU over NIXL, plus pre/post RL-step weight deltas for delta-compression analysis.

---

## Option 1: Request the pre-captured data package (fastest)

We have a ready-made data package captured from a live PRIME-RL deployment on GB200. **It's not in this repo** (binary tensors at GB scale aren't appropriate to commit) — request access from `kavink@nvidia.com` and we'll share via the appropriate channel (NV S3 bucket, internal share, or direct upload to your `eschmidt@nvidia.com` inbox per the original request).

Package contents:

```text
RL_Qwen25/
├── model.safetensors              # 2.9 GB — all 338 weight tensors (BF16)
├── weights_pre_rl.safetensors     # 3.4 GB — weights before optimizer.step()
├── weights_post_rl.safetensors    # 3.4 GB — weights after 1 AdamW step (lr=5e-6)
├── weight_deltas.safetensors      # 3.4 GB — elementwise diff (post - pre), BF16
├── kv_cache/                      # 14 MB  — 56 KV tensors from a 501-token prefill
│   ├── layer_0_key.bin            #          shape [1, 2, 501, 128], BF16
│   ├── layer_0_value.bin
│   ├── ...
│   └── manifest.json              #          per-tensor metadata
├── manifest.json                  # 66 KB  — per-weight-tensor metadata
└── README.md                      #          full layout + compression properties
```

**Model**: Qwen2.5-1.5B BF16, 28 layers, 1.54B parameters. ~14 GB total package size.

**How to read**:

```python
from safetensors import safe_open
import torch

# Weights (the exact tensors NIXL transfers during RL refit)
with safe_open("model.safetensors", framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)  # torch.bfloat16
        raw_bytes = tensor.contiguous().untyped_storage()  # raw bytes as on the wire
        print(f"{key}: {tensor.shape}, {len(raw_bytes)} bytes")

# Weight delta (for delta-compression analysis — compute in FP32 for precision)
pre = safe_open("weights_pre_rl.safetensors", framework="pt")
post = safe_open("weights_post_rl.safetensors", framework="pt")
for key in pre.keys():
    delta = post.get_tensor(key).float() - pre.get_tensor(key).float()
    print(f"{key}: max_abs_delta={delta.abs().max():.2e}")

# KV cache (the exact tensors transferred prefill → decode via NIXL)
raw = open("kv_cache/layer_0_key.bin", "rb").read()
kv = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(1, 2, 501, 128)
```

**Key finding on deltas**: At BF16 precision, single-step RL deltas are mostly zero (AdamW updates at lr=5e-6 are below BF16's representable precision). For meaningful delta analysis, compute diffs in FP32. This suggests delta-compression should operate in FP32 and quantize back after.

---

## Option 2: Reproduce end-to-end on GB200 (PRIME-RL overlay)

Run our validated PRIME-RL overlay workflow and capture weights mid-flight using the published [`scripts/`](./scripts/) directory.

### Prerequisites

- GKE cluster with GB200 nodes (ARM64, `customer-gpu-o7v` pool or equivalent)
- `kavin` namespace (or your own) with:
  - MX Server running: `modelexpress-server.<ns>.svc.cluster.local:8001`
  - Redis backing the MX Server
  - `shared-model-cache` PVC for HF model cache
  - `nvcr-imagepullsecret` for pulling the overlay image
- `tsh` auth for `nvcr.io/nvidian/dynamo-dev/`

### Step 1: Deploy the PRIME-RL overlay

```bash
git clone git@github.com:KavinKrishnan/prime-rl.git
cd prime-rl
git checkout kavink/mx-on-nixl

# Build the ARM64 image (or use the pre-built one)
# Pre-built: nvcr.io/nvidian/dynamo-dev/prime-rl-mx-on-nixl:v0.2
docker buildx build --platform linux/arm64 \
  -f docker/Dockerfile.mx-on-nixl \
  -t nvcr.io/nvidian/dynamo-dev/prime-rl-mx-on-nixl:v0.2 \
  --push .

# Deploy scenario A (baseline — PI's NIXL transport, no MX env vars)
cd k8s/prime-rl-mx-on-nixl
./run.sh deploy A
./run.sh status   # wait until all 3 pods are Running
```

### Step 2: Verify the RL loop is running

```bash
kubectl -n kavin logs prime-rl-mx-on-nixl-trainer-0 --tail=20 | grep "SUCCESS.*Step"
kubectl -n kavin logs prime-rl-mx-on-nixl-inference-0 | grep "update_weights.*200"
```

### Step 3: Capture using the published script

We ship `capture_on_pod.py` in [`scripts/`](./scripts/) — same script that produced our pre-captured Qwen2.5-1.5B package. It captures pre/post RL weights, simulates one AdamW step, computes deltas, and dumps a KV cache prefill, all in one pass.

```bash
# Copy the script into the trainer pod
kubectl cp docs/RL/scripts/capture_on_pod.py \
  kavin/prime-rl-mx-on-nixl-trainer-0:/tmp/capture.py

# Run it inside the pod (overlay image's interpreter is /app/.venv/bin/python)
kubectl exec kavin/prime-rl-mx-on-nixl-trainer-0 -- /app/.venv/bin/python /tmp/capture.py \
  --model Qwen/Qwen2.5-1.5B \
  --out /tmp/nixl_capture \
  --kv-seq-len 512 \
  --lr 5e-6

# Copy the results back
kubectl cp kavin/prime-rl-mx-on-nixl-trainer-0:/tmp/nixl_capture ./RL_capture
```

Output `RL_capture/` contains four sub-directories (`weights_pre_rl/`, `weights_post_rl/`, `weight_deltas/`, `kv_cache/`) each with raw `.bin` files plus a `manifest.json`. See [`scripts/README.md`](./scripts/README.md) for the full layout + flag reference.

### Step 4 (optional): Capture without a running RL deployment

If reproducing the overlay is more cluster work than the data is worth, [`scripts/capture_weights_and_kv.py`](./scripts/capture_weights_and_kv.py) is the **standalone** variant — works on any host (CPU or single GPU), no Kubernetes / RL framework required:

```bash
pip install torch transformers safetensors

python docs/RL/scripts/capture_weights_and_kv.py \
  --model Qwen/Qwen2.5-1.5B \
  --output-dir ./nixl_data \
  --dtype bfloat16 \
  --device cpu
```

Doesn't simulate an RL step (no pre/post/delta), but produces the same weight + KV cache layout the NIXL team can compress against.

### Step 5: Tear down

```bash
./run.sh clean
```

---

## Option 3: Reproduce with verl MxCheckpointEngine

The verl integration uses the same MX client but through verl's `CheckpointEngine` plugin. This path captures the weights as they flow through `MxCheckpointEngine.send_weights()` / `receive_weights()`.

Deployment docs: `docs/RL/VERL_MX_OVERVIEW.md` §6 in the modelexpress repo.

The capture approach is the same as Option 2 (exec into the trainer pod, save state dict pre/post step) since the weight tensors are identical — both frameworks produce `model.named_parameters()` in BF16. The difference is the transport path (verl's bucket+ZMQ metadata vs prime-rl's TransportPlan+slot system), which doesn't affect the tensor content.

---

## What to capture for the compression study

| Artifact | File | Size (Qwen3-0.6B) | What it represents |
|----------|------|-------|---------------------|
| **Current weights** | `weights_current.safetensors` | ~1.2 GB | Exact tensors registered with NIXL and RDMA-written to inference GPU every RL step |
| **Post-step weights** | `weights_post_step.safetensors` | ~1.2 GB | After one AdamW step (lr=5e-6) |
| **Weight deltas** | `weight_deltas.safetensors` | ~1.2 GB | `post - pre` in BF16 (mostly zero — compute in FP32 for real deltas) |
| **KV cache** | `kv_cache/*.bin` | ~14 MB | Prefill output transferred to decode workers via NIXL |
| **Manifest** | `manifest.json` | ~30 KB | Per-tensor: name, shape, dtype, size_bytes |

### Larger models for more representative data

The steps above use Qwen3-0.6B (our scenario A model). For larger models closer to production:

| Model | Params | Weight payload | Notes |
|-------|--------|----------------|-------|
| Qwen3-0.6B (above) | 0.6B | ~1.2 GB | Validated in PR #2343 scenarios A/B/C |
| Qwen2.5-1.5B | 1.5B | ~3 GB | Pre-captured package available on request (see Option 1) |
| Qwen2.5-7B | 7.6B | ~15 GB | T1 model in our overlay plan |
| Qwen3-MoE (PI offered spec) | MoE | varies | Would exercise `ExpertSlot` + per-expert tensors — most representative for MoE compression |

For models requiring multiple GPUs, the weights are FSDP-sharded — each rank's shard is `total / num_ranks` in size. The bytes on the wire per-rank are the shard size, not the full model.

---

## Compression-relevant properties

| Property | Weights | KV Cache | Delta (FP32) |
|----------|---------|----------|--------------|
| **Dtype on wire** | BF16 (2 B/elem) | BF16 (2 B/elem) | BF16 stored, but FP32 is the meaningful analysis dtype |
| **Value distribution** | Normal, centered ~0, std 0.01–0.1 | Wider, context-dependent | Very small magnitude (~1e-8 to 1e-6 per element) |
| **Sparsity** | Dense (no zeros) | Dense | ~100% zero at BF16 precision; structured-sparse at FP32 |
| **Best compression angle** | Entropy coding on mantissa bits | Temporal locality across layers | FP32 delta + entropy coding — high compressibility expected |
| **Transfer frequency** | Every RL step (~5–60 s) | Every request | Once for analysis |
| **Bucket size on wire** | 596 MB (measured in scenario A/B/C) | per-request, scales with seq_len | N/A |

### NIXL integration point for nvCOMP

If nvCOMP compression is added at the NIXL layer, the integration is transparent to both MX and the RL frameworks:

```text
Current:
  Training GPU → NIXL register → RDMA WRITE (raw bytes) → Inference GPU

With NIXL-layer nvCOMP:
  Training GPU → NIXL register → nvCOMP compress (GPU) → RDMA WRITE (compressed) → nvCOMP decompress (GPU) → Inference GPU
```

No changes to `MxTrainingPublisher`, `MxRefitReceiver`, `NIXLWeightBroadcast`, `TransportPlan`, or the MX Server protocol. Compression is internal to NIXL's transfer path. Our bucket-streaming pattern is preserved — compression happens per-bucket.

---

## Questions?

Reach out to Kavin Krishnan (`kavink@nvidia.com`) for access to the pre-captured data or help reproducing on a cluster. The PRIME-RL overlay branch (`KavinKrishnan/prime-rl:kavink/mx-on-nixl`) and the modelexpress RL branch (`ai-dynamo/modelexpress:kavink/RL`) are the entry points.
