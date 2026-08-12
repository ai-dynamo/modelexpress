#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inference-side runner for the reshard-refit E2E benchmark.

Loads Qwen3-30B in vLLM (TP=k), builds a VllmReshardReceiver per rank, and
measures a real inter-node RDMA refit: discover -> P2P handshake -> no-gather
RDMA pull -> install. Splits per-cycle transfer vs install, and times BOTH
installers (PWAL, meaning reshard-refit without MDL, and MDL) on the same pulled
buffers so the only difference is the install seam.

Correctness: corrupt the live model, run one refit, confirm greedy generation
recovers vs the pre-corruption baseline.

Usage (receiver pod, k GPUs), after the publisher is READY:
  MX_POOL_REG=1 python3 reshard_receiver_run.py \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 2 \
      --mx-server modelexpress-server.kavin.svc.cluster.local:8001 \
      --warm-cycles 10 --out /mnt/rl-workspace/kavink/reshard_e2e/tp2.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

PROMPTS = ["The capital of France is", "def add(a, b):\n    return"]
_BASE_PORT = 7300


def w_build(
    worker,
    rendezvous_name,
    mx_server,
    num_trainers,
    capture_layout,
    capture_cache_path,
):
    import torch

    from modelexpress.engines.vllm.reshard import VllmReshardReceiver

    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        rank = int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001
        rank = 0
    dev = next(worker.model_runner.model.parameters()).device
    vc = getattr(worker, "vllm_config", None) or getattr(
        worker.model_runner, "vllm_config", None
    )
    recv = VllmReshardReceiver(
        model=worker.model_runner.model,
        vllm_config=vc,
        model_config=vc.model_config,
        installer="pwal",
        capture_layout=capture_layout,
        capture_cache_path=capture_cache_path,
        model_name=rendezvous_name,
        mx_server=mx_server,
        agent_name=f"infer-{rank}",
        local_rank=dev.index,
        global_rank=rank,
        num_trainer_sources=num_trainers,
        device=dev,
        listen_port=_BASE_PORT + rank,
    )
    worker._recv = recv
    # snapshot reference (post-load correct params) on CPU
    worker._ref = {
        n: p.detach().to("cpu", copy=True)
        for n, p in worker.model_runner.model.named_parameters()
    }
    return {"rank": rank, "device": str(dev)}


def w_refit_timed(worker, n, installer_name):
    """Run one installer arm and split wire/assembly, install, quantize, E2E."""
    import torch

    from modelexpress.engines.vllm.installers import build_installer

    recv = worker._recv
    recv._installer = build_installer(
        installer_name,
        model=worker.model_runner.model,
        vllm_config=recv._vllm_config,
        model_config=recv._model_config,
        device=recv._device,
    )
    install_ms = []
    quantization_ms = []
    selected_modes = []

    orig_install = recv._install

    def timed_install(buffers):
        t = time.perf_counter()
        orig_install(buffers)
        torch.cuda.synchronize()
        install_ms.append((time.perf_counter() - t) * 1e3)
        quantization_ms.append(
            float(getattr(recv._installer, "last_quantization_s", 0.0)) * 1e3
        )
        selected_modes.append(
            getattr(recv._installer, "last_mode", installer_name)
        )

    recv._install = timed_install
    step = int(getattr(worker, "_refit_step", 0))
    if recv._plan is None:
        recv.update_weights(step)  # cold prepare + first transfer/install
        step += 1
    recv.update_weights(step)  # unreported warm-up for this installer arm
    step += 1
    totals = []
    install_ms.clear()
    quantization_ms.clear()
    selected_modes.clear()
    for _ in range(n):
        t0 = time.perf_counter()
        recv.update_weights(step)
        torch.cuda.synchronize()
        totals.append((time.perf_counter() - t0) * 1e3)
        step += 1
    worker._refit_step = step
    recv._install = orig_install
    transfer_ms = [tot - ins for tot, ins in zip(totals, install_ms)]
    bytes_planned = recv._plan.bytes_planned() if recv._plan is not None else 0
    return {
        "e2e_ms": totals,
        "install_ms": install_ms,
        "quantization_ms": quantization_ms,
        "transfer_ms": transfer_ms,
        "bytes_planned": bytes_planned,
        "selected_modes": selected_modes,
    }


def w_corrupt(worker):
    import torch

    with torch.no_grad():
        for p in worker.model_runner.model.parameters():
            try:
                p.data.view(torch.uint8).random_(0, 256)
            except Exception:  # noqa: BLE001
                p.data.normal_(0.0, 0.5)
    torch.cuda.synchronize()
    return {"ok": True}


def w_refit_once(worker):
    worker._recv.update_weights(9999)
    return {"ok": True}


def _gen(llm):
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=24)
    return [tuple(o.outputs[0].token_ids) for o in llm.generate(PROMPTS, sp, use_tqdm=False)]


def _agree(a, b):
    tot = ok = 0
    for x, y in zip(a, b):
        m = min(len(x), len(y))
        tot += max(len(x), len(y))
        ok += sum(1 for i in range(m) if x[i] == y[i])
    return ok / tot if tot else 0.0


def _crit(rank_lists, key):
    per = [r.get(key) or [] for r in rank_lists]
    if not any(per):
        return None
    n = min(len(x) for x in per)
    crit = [max(per[r][i] for r in range(len(per))) for i in range(n)]
    s = sorted(crit)
    return {
        "n": len(s),
        "min_ms": min(s),
        "median_ms": statistics.median(s),
        "p95_ms": s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))],
        "max_ms": max(s),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="vLLM model to load (bf16 id or fp8 dir)")
    ap.add_argument("--rendezvous-name", required=True, help="shared model_name matching the publisher")
    ap.add_argument("--dtype", choices=["bf16", "fp8"], default="bf16")
    ap.add_argument("--source-dtype", choices=["bf16", "fp8"], default="bf16")
    ap.add_argument(
        "--quantization",
        default=None,
        help="vLLM online quantization method (for example 'fp8')",
    )
    ap.add_argument(
        "--capture-layout",
        choices=["load_time", "runtime", "live", "cache"],
        default="load_time",
    )
    ap.add_argument(
        "--capture-cache-path",
        default=None,
        help="per-rank pickle path template; may contain {rank}",
    )
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--enable-expert-parallel", action="store_true")
    ap.add_argument("--num-trainers", type=int, default=8)
    ap.add_argument(
        "--installers",
        default="pwal,quantizing_mdl",
        help="comma-separated installer arms",
    )
    ap.add_argument("--mx-server", required=True)
    ap.add_argument("--warm-cycles", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    from vllm import LLM

    is_moe = "A3B" in args.model or "moe" in args.model.lower()
    kw = dict(
        model=args.model,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=float(os.environ.get("MX_GPU_MEM_UTIL", "0.45")),
        max_model_len=1024,
        trust_remote_code=True,
    )
    if args.dtype == "bf16":
        kw["dtype"] = "bfloat16"
    if args.quantization:
        kw["quantization"] = args.quantization
    if is_moe:
        kw["moe_backend"] = "triton"
        if args.enable_expert_parallel:
            kw["enable_expert_parallel"] = True
    print(f"[recv] loading {args.model} tp={args.tp}", flush=True)
    llm = LLM(**kw)

    rec = {
        "run_id": f"reshard-e2e-tp{args.tp}-{int(time.time())}",
        "model": args.model,
        "rendezvous_name": args.rendezvous_name,
        "source_dtype": args.source_dtype,
        "target_dtype": args.dtype,
        "quantization_method": args.quantization,
        "trainer_topology": f"FSDP{args.num_trainers}+EP{args.num_trainers}",
        "trainer_gpus": args.num_trainers,
        "inference_topology": (
            f"TP{args.tp}+EP{args.tp}"
            if args.enable_expert_parallel
            else f"TP{args.tp}+EP1"
        ),
        "generator_gpus": args.tp,
        "tp": args.tp,
        "warm_cycles": args.warm_cycles,
        "transport": "NIXL/RDMA inter-node",
    }
    base = _gen(llm)
    llm.collective_rpc(
        w_build,
        args=(
            args.rendezvous_name,
            args.mx_server,
            args.num_trainers,
            args.capture_layout,
            args.capture_cache_path,
        ),
    )
    rec["arms"] = {}
    last_arm = None
    for installer_name in [
        item.strip() for item in args.installers.split(",") if item.strip()
    ]:
        refit = llm.collective_rpc(
            w_refit_timed, args=(args.warm_cycles, installer_name)
        )
        transfer = _crit(refit, "transfer_ms")
        install = _crit(refit, "install_ms")
        quantization = _crit(refit, "quantization_ms")
        e2e = _crit(refit, "e2e_ms")
        bytes_per_rank = [r["bytes_planned"] for r in refit]
        total_bytes = sum(bytes_per_rank)
        aggregate_gbps = None
        if transfer and transfer["median_ms"] > 0:
            aggregate_gbps = (
                total_bytes * 8.0 / (transfer["median_ms"] / 1e3) / 1e9
            )
        modes = sorted(
            {
                mode
                for rank_result in refit
                for mode in rank_result.get("selected_modes", [])
            }
        )
        rec["arms"][installer_name] = {
            "selected_modes": modes,
            "bytes_planned_per_rank": bytes_per_rank,
            "aggregate_wire_gbps": aggregate_gbps,
            "transfer": transfer,
            "install": install,
            "quantization": quantization,
            "e2e": e2e,
        }
        last_arm = installer_name

    # Correctness gates the final selected arm after an actual transfer/install.
    llm.collective_rpc(w_corrupt, args=())
    corrupt_tok = _gen(llm)
    llm.collective_rpc(w_refit_once, args=())
    post = _gen(llm)
    rec["correctness"] = {
        "installer": last_arm,
        "corruption_detected": _agree(base, corrupt_tok) < 0.999,
        "recovery_tokens": _agree(base, post),
    }
    rec["result"] = (
        "PASS"
        if rec["correctness"]["corruption_detected"]
        and rec["correctness"]["recovery_tokens"] >= 0.999
        else "FAIL"
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2), flush=True)
    print(f"[out] {args.out} RESULT={rec['result']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
