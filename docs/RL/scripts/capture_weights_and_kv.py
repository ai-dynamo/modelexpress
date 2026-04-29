#!/usr/bin/env python3
"""Capture raw model weights and KV cache data for NIXL nvCOMP compression study.

Outputs:
  weights/{model_name}/tensors/*.bin     — raw weight tensors (as transferred during RL refit)
  weights/{model_name}/manifest.json     — per-tensor metadata
  kvcache/{model_name}/*.bin             — KV cache tensors from a sample forward pass
  kvcache/{model_name}/manifest.json     — per-KV-tensor metadata

Usage:
  python capture_weights_and_kv.py --model Qwen/Qwen2.5-1.5B --output-dir ./nixl_data
  python capture_weights_and_kv.py --model Qwen/Qwen2.5-1.5B --output-dir ./nixl_data --kv-only
  python capture_weights_and_kv.py --model Qwen/Qwen2.5-1.5B --output-dir ./nixl_data --weights-only
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def tensor_stats(t: torch.Tensor) -> dict:
    with torch.no_grad():
        ft = t.float()
        return {
            "min": float(ft.min()),
            "max": float(ft.max()),
            "mean": float(ft.mean()),
            "std": float(ft.std()),
            "abs_mean": float(ft.abs().mean()),
            "zero_fraction": float((t == 0).float().mean()),
        }


def classify_tensor(name: str) -> str:
    if "embed" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "layernorm" in name or "norm" in name:
        return "norm"
    if "q_proj" in name:
        return "attention_q"
    if "k_proj" in name:
        return "attention_k"
    if "v_proj" in name:
        return "attention_v"
    if "o_proj" in name:
        return "attention_o"
    if "gate_proj" in name or "w1" in name:
        return "mlp_gate"
    if "up_proj" in name or "w3" in name:
        return "mlp_up"
    if "down_proj" in name or "w2" in name:
        return "mlp_down"
    return "other"


def get_layer_index(name: str) -> int:
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return -1


def capture_weights(model, model_name: str, output_dir: Path, dtype_name: str):
    """Dump all model weight tensors as raw binary files + manifest."""
    safe_name = sanitize_name(model_name)
    weight_dir = output_dir / "weights" / safe_name / "tensors"
    weight_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_name": model_name,
        "dtype": dtype_name,
        "capture_type": "model_weights_for_rl_refit",
        "description": (
            "These are the exact weight tensors transferred from training GPUs to "
            "inference GPUs during the RL refit (weight sync) phase. In RL post-training, "
            "after each optimizer step, the full model state dict is gathered and sent to "
            "the inference engine (vLLM). These tensors represent that payload."
        ),
        "tensors": [],
    }

    total_bytes = 0
    for name, param in model.named_parameters():
        t = param.data.contiguous().cpu()
        raw_bytes = bytes(t.untyped_storage())[:t.numel() * t.element_size()]
        fname = sanitize_name(name) + ".bin"
        (weight_dir / fname).write_bytes(raw_bytes)

        info = {
            "name": name,
            "file": f"tensors/{fname}",
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "size_bytes": len(raw_bytes),
            "numel": t.numel(),
            "layer_index": get_layer_index(name),
            "tensor_type": classify_tensor(name),
            "stats": tensor_stats(t),
        }
        manifest["tensors"].append(info)
        total_bytes += len(raw_bytes)

    manifest["total_tensors"] = len(manifest["tensors"])
    manifest["total_bytes"] = total_bytes
    manifest["total_gb"] = round(total_bytes / 1e9, 3)

    config = model.config
    manifest["model_config"] = {
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "architecture": config.architectures[0] if hasattr(config, "architectures") and config.architectures else None,
    }

    manifest_path = output_dir / "weights" / safe_name / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Weights captured: {len(manifest['tensors'])} tensors, {manifest['total_gb']} GB → {weight_dir}")
    return manifest


def capture_kvcache(model, tokenizer, model_name: str, output_dir: Path, seq_len: int = 512):
    """Run a forward pass and capture the KV cache tensors."""
    safe_name = sanitize_name(model_name)
    kv_dir = output_dir / "kvcache" / safe_name
    kv_dir.mkdir(parents=True, exist_ok=True)

    prompt = "The quick brown fox jumps over the lazy dog. " * (seq_len // 10)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=seq_len, truncation=True)

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past_kv = outputs.past_key_values

    manifest = {
        "model_name": model_name,
        "capture_type": "kv_cache_prefill_to_decode",
        "description": (
            "These are the KV cache tensors produced during the prefill phase and "
            "sent to decode workers in disaggregated inference (prefill/decode split). "
            "In NIXL-based KV transfer, these are the exact tensors transferred "
            "GPU-to-GPU between prefill and decode nodes."
        ),
        "sequence_length": int(inputs["input_ids"].shape[1]),
        "batch_size": 1,
        "tensors": [],
    }

    total_bytes = 0
    for layer_idx, layer_kv in enumerate(past_kv):
        for kv_idx, kv_name in enumerate(["key", "value"]):
            t = layer_kv[kv_idx].contiguous().cpu()
            raw_bytes = bytes(t.untyped_storage())[:t.numel() * t.element_size()]
            fname = f"layer_{layer_idx}_{kv_name}.bin"
            (kv_dir / fname).write_bytes(raw_bytes)

            info = {
                "name": f"layer_{layer_idx}.{kv_name}",
                "file": fname,
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "size_bytes": len(raw_bytes),
                "layer_index": layer_idx,
                "kv_type": kv_name,
                "stats": tensor_stats(t),
            }
            manifest["tensors"].append(info)
            total_bytes += len(raw_bytes)

    manifest["total_tensors"] = len(manifest["tensors"])
    manifest["total_bytes"] = total_bytes
    manifest["total_mb"] = round(total_bytes / 1e6, 3)

    config = model.config
    manifest["kv_config"] = {
        "num_layers": len(past_kv),
        "num_kv_heads": getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", None)),
        "head_dim": getattr(config, "hidden_size", 0) // getattr(config, "num_attention_heads", 1),
        "kv_dtype": str(past_kv[0][0].dtype),
    }

    manifest_path = kv_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"KV cache captured: {len(manifest['tensors'])} tensors, {manifest['total_mb']} MB → {kv_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Capture weight and KV cache data for NIXL compression study")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B", help="HuggingFace model name")
    parser.add_argument("--output-dir", type=str, default="./nixl_compression_data", help="Output directory")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str, default="cpu", help="Device to load model on (cpu or cuda:0)")
    parser.add_argument("--weights-only", action="store_true", help="Only capture weights, skip KV cache")
    parser.add_argument("--kv-only", action="store_true", help="Only capture KV cache, skip weights")
    parser.add_argument("--kv-seq-len", type=int, default=512, help="Sequence length for KV cache capture")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model} (dtype={args.dtype}, device={args.device})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True, device_map=args.device,
    )
    model.eval()

    if not args.kv_only:
        print("\n=== Capturing model weights (RL refit payload) ===")
        capture_weights(model, args.model, output_dir, args.dtype)

    if not args.weights_only:
        print(f"\n=== Capturing KV cache (prefill→decode, seq_len={args.kv_seq_len}) ===")
        capture_kvcache(model, tokenizer, args.model, output_dir, seq_len=args.kv_seq_len)

    print(f"\nDone. Data written to {output_dir}/")
    print("Send the output directory to the NIXL compression team.")


if __name__ == "__main__":
    main()
