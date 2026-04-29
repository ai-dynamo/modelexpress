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

Run our validated PRIME-RL overlay workflow and capture weights mid-flight.

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
# Clone and check out the overlay branch
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

# Watch until all 3 pods are Running
./run.sh status
```

### Step 2: Verify the RL loop is running

```bash
# Trainer should show "Step N | Time: Xs" lines
kubectl -n kavin logs prime-rl-mx-on-nixl-trainer-0 --tail=20 | grep "SUCCESS.*Step"

# Inference should show /update_weights 200 OK
kubectl -n kavin logs prime-rl-mx-on-nixl-inference-0 | grep "update_weights.*200"
```

### Step 3: Capture weights from the running trainer

```bash
# Exec into the trainer pod
kubectl -n kavin exec -it prime-rl-mx-on-nixl-trainer-0 -- bash

# Inside the pod — capture pre/post RL weights + KV cache
cd /tmp
/app/.venv/bin/python - << 'PYEOF'
import torch, json, os, time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

model_name = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
out = Path("/tmp/nixl_compression_capture")
out.mkdir(exist_ok=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 1. Capture current weights (= what NIXL transfers during refit)
print("Saving current weights...")
sd = {k: v.clone() for k, v in model.state_dict().items()}
save_file(sd, str(out / "weights_current.safetensors"))

# 2. Simulate one RL step for delta capture
print("Simulating one RL step...")
model.to("cuda:0")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
inputs = tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt").to("cuda:0")
loss = model(**inputs, labels=inputs["input_ids"]).loss
loss.backward()
optimizer.step()

sd_post = {k: v.cpu().clone() for k, v in model.state_dict().items()}
save_file(sd_post, str(out / "weights_post_step.safetensors"))

# 3. Compute delta
deltas = {}
for k in sd:
    d = sd_post[k].float() - sd[k].float()
    deltas[k] = d.to(torch.bfloat16)
save_file(deltas, str(out / "weight_deltas.safetensors"))

# 4. KV cache from a prefill pass
print("Capturing KV cache...")
model.eval()
kv_out = out / "kv_cache"
kv_out.mkdir(exist_ok=True)
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
manifest = {"tensors": []}
for i, layer_kv in enumerate(outputs.past_key_values):
    for j, name in enumerate(["key", "value"]):
        t = layer_kv[j].cpu().contiguous()
        fname = f"layer_{i}_{name}.bin"
        (kv_out / fname).write_bytes(t.numpy().tobytes())
        manifest["tensors"].append({
            "name": f"layer_{i}_{name}", "shape": list(t.shape),
            "dtype": "bfloat16", "size_bytes": t.numel() * 2, "file": fname
        })
json.dump(manifest, open(kv_out / "manifest.json", "w"), indent=2)

# 5. Write weight manifest
w_manifest = {"model": model_name, "tensors": []}
for k, v in sd.items():
    w_manifest["tensors"].append({
        "name": k, "shape": list(v.shape), "dtype": str(v.dtype),
        "size_bytes": v.numel() * v.element_size()
    })
json.dump(w_manifest, open(out / "manifest.json", "w"), indent=2)

print(f"Done. Files in {out}")
PYEOF

# Copy out of the pod
exit
kubectl -n kavin cp prime-rl-mx-on-nixl-trainer-0:/tmp/nixl_compression_capture ./nixl_capture
```

### Step 4: Tear down

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
