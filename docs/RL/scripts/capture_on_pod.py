#!/usr/bin/env python3
"""Capture weight and KV cache data from a running PRIME-RL / verl deployment.

Designed to be exec'd inside a trainer pod. Captures four artifacts in the
same shape we shipped to the NIXL nvCOMP compression team for Qwen2.5-1.5B:

  weights_pre_rl/   raw .bin tensors + manifest.json (pre-step state dict)
  weights_post_rl/  raw .bin tensors + manifest.json (after one AdamW step)
  weight_deltas/    raw .bin tensors + manifest.json (post - pre)
  kv_cache/         raw .bin tensors + manifest.json (one prefill pass)

Then a final summary line tells you how to `kubectl cp` it out.

Usage (inside pod):
  python3 capture_on_pod.py
  python3 capture_on_pod.py --model Qwen/Qwen2.5-7B --out /tmp/nixl_capture --kv-seq-len 1024

Usage (from host, no pod):
  kubectl cp capture_on_pod.py <ns>/<trainer-pod>:/tmp/capture.py
  kubectl exec <ns>/<trainer-pod> -- python3 /tmp/capture.py --model <name>
  kubectl cp <ns>/<trainer-pod>:/tmp/nixl_capture ./RL_capture
"""
import argparse, json, os, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B",
                    help="HuggingFace model name (must match the running RL deployment)")
parser.add_argument("--out", default="/tmp/nixl_capture",
                    help="Output directory inside the pod")
parser.add_argument("--kv-seq-len", type=int, default=512,
                    help="Sequence length for the KV cache prefill pass")
parser.add_argument("--lr", type=float, default=5e-6,
                    help="Learning rate for the simulated AdamW step (matches PRIME-RL default)")
args = parser.parse_args()

MODEL = args.model
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

def tensor_stats(t):
    ft = t.float()
    return {"min": float(ft.min()), "max": float(ft.max()), "mean": float(ft.mean()),
            "std": float(ft.std()), "abs_mean": float(ft.abs().mean()),
            "zero_frac": float((t == 0).float().mean())}

def classify(name):
    for k, v in [("embed", "embedding"), ("lm_head", "lm_head"), ("norm", "norm"),
                 ("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"),
                 ("o_proj", "attn_o"), ("gate_proj", "mlp_gate"), ("up_proj", "mlp_up"),
                 ("down_proj", "mlp_down")]:
        if k in name: return v
    return "other"

def layer_idx(name):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i+1].isdigit():
            return int(parts[i+1])
    return -1

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()

# --- 1. Dump weights (pre-RL step) ---
print("\n=== Capturing pre-RL weights ===")
wdir = OUT / "weights_pre_rl"
wdir.mkdir(exist_ok=True)
manifest = {"model": MODEL, "dtype": "bfloat16", "capture": "pre_rl_weights",
            "description": "Exact weight tensors transferred during RL refit (training->inference via NIXL RDMA)",
            "tensors": []}
total = 0
for name, param in model.named_parameters():
    t = param.data.contiguous()
    raw = bytes(t.untyped_storage())[:t.numel() * t.element_size()]
    fname = name.replace(".", "_") + ".bin"
    (wdir / fname).write_bytes(raw)
    manifest["tensors"].append({"name": name, "file": fname, "shape": list(t.shape),
        "dtype": str(t.dtype), "size_bytes": len(raw), "numel": t.numel(),
        "layer": layer_idx(name), "type": classify(name), "stats": tensor_stats(t)})
    total += len(raw)
manifest["total_bytes"] = total
manifest["total_gb"] = round(total / 1e9, 3)
manifest["num_tensors"] = len(manifest["tensors"])
cfg = model.config
manifest["model_config"] = {"num_hidden_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
    "intermediate_size": cfg.intermediate_size, "num_attention_heads": cfg.num_attention_heads,
    "num_key_value_heads": cfg.num_key_value_heads, "vocab_size": cfg.vocab_size}
(wdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"  {manifest['num_tensors']} tensors, {manifest['total_gb']} GB -> {wdir}")

# --- 2. KV cache ---
print(f"\n=== Capturing KV cache (seq_len={args.kv_seq_len}) ===")
kvdir = OUT / "kv_cache"
kvdir.mkdir(exist_ok=True)
prompt = "The quick brown fox jumps over the lazy dog. " * (args.kv_seq_len // 10 + 1)
inputs = tokenizer(prompt, return_tensors="pt", max_length=args.kv_seq_len, truncation=True)
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
kv = outputs.past_key_values
kv_manifest = {"model": MODEL, "capture": "kv_cache_prefill", "seq_len": int(inputs["input_ids"].shape[1]),
               "description": "KV cache from prefill pass - transferred prefill->decode via NIXL in disagg inference",
               "tensors": []}
kv_total = 0
for li, layer_kv in enumerate(kv):
    for ki, kn in enumerate(["key", "value"]):
        t = layer_kv[ki].contiguous()
        raw = bytes(t.untyped_storage())[:t.numel() * t.element_size()]
        fname = f"layer_{li}_{kn}.bin"
        (kvdir / fname).write_bytes(raw)
        kv_manifest["tensors"].append({"name": f"layer_{li}.{kn}", "file": fname,
            "shape": list(t.shape), "dtype": str(t.dtype), "size_bytes": len(raw),
            "layer": li, "kv_type": kn, "stats": tensor_stats(t)})
        kv_total += len(raw)
kv_manifest["total_bytes"] = kv_total
kv_manifest["total_mb"] = round(kv_total / 1e6, 3)
kv_manifest["kv_config"] = {"num_layers": len(kv),
    "num_kv_heads": cfg.num_key_value_heads,
    "head_dim": cfg.hidden_size // cfg.num_attention_heads}
(kvdir / "manifest.json").write_text(json.dumps(kv_manifest, indent=2))
print(f"  {len(kv_manifest['tensors'])} tensors, {kv_manifest['total_mb']} MB -> {kvdir}")

# --- 3. Simulate one RL step and capture post-RL weights + delta ---
print(f"\n=== Simulating RL step (AdamW, lr={args.lr}, dummy loss) ===")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
dummy_input = tokenizer("Hello world", return_tensors="pt")
output = model(**dummy_input, labels=dummy_input["input_ids"])
output.loss.backward()
optimizer.step()
optimizer.zero_grad()
model.eval()

print("\n=== Capturing post-RL weights ===")
wdir2 = OUT / "weights_post_rl"
wdir2.mkdir(exist_ok=True)
manifest2 = {"model": MODEL, "dtype": "bfloat16", "capture": "post_rl_weights",
             "description": "Weights after 1 RL optimizer step (lr=5e-6, AdamW)", "tensors": []}
total2 = 0
for name, param in model.named_parameters():
    t = param.data.contiguous()
    raw = bytes(t.untyped_storage())[:t.numel() * t.element_size()]
    fname = name.replace(".", "_") + ".bin"
    (wdir2 / fname).write_bytes(raw)
    manifest2["tensors"].append({"name": name, "file": fname, "shape": list(t.shape),
        "dtype": str(t.dtype), "size_bytes": len(raw), "numel": t.numel(),
        "layer": layer_idx(name), "type": classify(name), "stats": tensor_stats(t)})
    total2 += len(raw)
manifest2["total_bytes"] = total2
manifest2["total_gb"] = round(total2 / 1e9, 3)
(wdir2 / "manifest.json").write_text(json.dumps(manifest2, indent=2))
print(f"  {len(manifest2['tensors'])} tensors, {manifest2['total_gb']} GB -> {wdir2}")

# --- 4. Compute and save deltas ---
print("\n=== Computing weight deltas (post - pre) ===")
ddir = OUT / "weight_deltas"
ddir.mkdir(exist_ok=True)
delta_manifest = {"model": MODEL, "capture": "weight_delta_1_step",
    "description": (
        f"Difference between weights after 1 RL step vs before. "
        f"RL uses lr={args.lr} so deltas are tiny — at BF16 most are exactly zero "
        f"(below mantissa precision). For meaningful delta-compression analysis, "
        f"compute diffs in FP32 from the pre/post safetensors instead of using "
        f"this BF16-stored delta directly."
    ),
    "tensors": []}
dtotal = 0
pre_files = {m["name"]: m["file"] for m in manifest["tensors"]}
for info in manifest2["tensors"]:
    pre_raw = (OUT / "weights_pre_rl" / pre_files[info["name"]]).read_bytes()
    post_raw = (wdir2 / info["file"]).read_bytes()
    pre_t = torch.frombuffer(bytearray(pre_raw), dtype=torch.bfloat16).reshape(info["shape"])
    post_t = torch.frombuffer(bytearray(post_raw), dtype=torch.bfloat16).reshape(info["shape"])
    delta = post_t - pre_t
    delta_raw = bytes(delta.contiguous().untyped_storage())[:delta.numel() * delta.element_size()]
    fname = "delta_" + info["file"]
    (ddir / fname).write_bytes(delta_raw)
    delta_manifest["tensors"].append({"name": info["name"], "file": fname,
        "shape": info["shape"], "dtype": "bfloat16", "size_bytes": len(delta_raw),
        "stats": tensor_stats(delta)})
    dtotal += len(delta_raw)
delta_manifest["total_bytes"] = dtotal
delta_manifest["total_gb"] = round(dtotal / 1e9, 3)
(ddir / "manifest.json").write_text(json.dumps(delta_manifest, indent=2))
print(f"  {len(delta_manifest['tensors'])} delta tensors, {delta_manifest['total_gb']} GB -> {ddir}")

# --- Summary ---
print(f"\n=== DONE ===")
print(f"Output: {OUT}")
print(f"  weights_pre_rl/  : {manifest['total_gb']} GB ({manifest['num_tensors']} tensors)")
print(f"  weights_post_rl/ : {manifest2['total_gb']} GB")
print(f"  weight_deltas/   : {delta_manifest['total_gb']} GB")
print(f"  kv_cache/        : {kv_manifest['total_mb']} MB ({len(kv_manifest['tensors'])} tensors)")
print(f"\nTo copy out: kubectl cp <namespace>/<trainer-pod>:{OUT} ./RL_capture")
