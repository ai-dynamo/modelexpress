#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reshard-refit install comparison: PWAL (no MDL) vs MDL, install-only.

Answers "what is the load/install time with just reshard-refit and no MDL, and
what does MDL add?" It times the two swappable installers from the
reshard-refit receiver seam on identical post-transfer buffers:

  * PWAL (``MX_RESHARD_INSTALLER=pwal``, the default) — vLLM layerwise reload +
    process_weights_after_loading per layer;
  * MDL (``mdl``) — destination-mapped in-place copy.

``recv_buffers`` is what the receiver hands the installer: one load-time
(bf16, TP-local) tensor per live param. We reconstruct it from the loaded
model's own params (the correct values), so this isolates INSTALL cost only —
no discovery/transfer/translation (those are reported separately).

Usage (inside the reshard+MDL client pod):
  python3 reshard_install_bench.py --model <id|dir> --tp 2 --warm-cycles 10 \
      --out /mnt/.../reshard_install_tp2.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time

PROMPTS = ["The capital of France is", "def add(a, b):\n    return"]


def w_setup(worker):
    """Snapshot per-param load-time buffers (device) + reference (cpu)."""
    import torch

    model = worker.model_runner.model
    recv = {n: p.detach().clone() for n, p in model.named_parameters()}
    worker._recv = recv
    worker._ref = {n: t.detach().to("cpu", copy=True) for n, t in recv.items()}
    worker._bytes = int(sum(t.numel() * t.element_size() for t in recv.values()))
    # config/device for building the PWAL installer
    vc = getattr(worker, "vllm_config", None) or getattr(
        worker.model_runner, "vllm_config", None
    )
    worker._vllm_config = vc
    worker._model_config = vc.model_config if vc is not None else None
    worker._device = next(model.parameters()).device
    return {"params": len(recv), "bytes": worker._bytes, "device": str(worker._device)}


def _build(worker, name):
    from modelexpress.engines.vllm.installers import build_installer

    return build_installer(
        name,
        model=worker.model_runner.model,
        vllm_config=worker._vllm_config,
        model_config=worker._model_config,
        device=worker._device,
    )


def w_install_timed(worker, name, n):
    import torch

    inst = _build(worker, name)
    inst.install(worker._recv)  # cold/warm-up
    torch.cuda.synchronize()
    durs = []
    for _ in range(n):
        t0 = time.perf_counter()
        inst.install(worker._recv)
        torch.cuda.synchronize()
        durs.append(time.perf_counter() - t0)
    return {"durations_s": durs}


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


def w_apply(worker, name):
    import torch

    inst = _build(worker, name)
    inst.install(worker._recv)
    torch.cuda.synchronize()
    return {"ok": True}


def w_compare(worker):
    import torch

    model = worker.model_runner.model
    ref = worker._ref
    mism = 0
    mx = 0.0
    for n, p in model.named_parameters():
        r = ref.get(n)
        if r is None:
            continue
        cur = p.detach().to("cpu")
        if cur.shape != r.shape:
            mism += 1
            continue
        d = (cur.float() - r.float()).abs().max().item()
        mx = max(mx, d)
        if not torch.equal(cur, r):
            mism += 1
    return {"byte_mismatch": mism, "max_abs_diff": mx}


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


def _stats(durs):
    if not durs:
        return None
    s = sorted(durs)
    return {
        "n": len(s),
        "min_ms": min(s) * 1e3,
        "median_ms": statistics.median(s) * 1e3,
        "p95_ms": s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))] * 1e3,
        "max_ms": max(s) * 1e3,
    }


def _crit(rank_lists):
    per = [r.get("durations_s") or [] for r in rank_lists]
    if not any(per):
        return None
    n = min(len(x) for x in per)
    return _stats([max(per[r][i] for r in range(len(per))) for i in range(n)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, required=True)
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
        dtype="bfloat16",
        trust_remote_code=True,
    )
    if is_moe:
        kw["moe_backend"] = "triton"
    print(f"[load] {args.model} tp={args.tp} moe={is_moe}", flush=True)
    llm = LLM(**kw)

    rec = {
        "run_id": f"reshard-install-tp{args.tp}-{int(time.time())}",
        "compares": "PWAL (reshard-refit, no MDL) vs MDL install-only",
        "model": args.model,
        "tp": args.tp,
        "warm_cycles": args.warm_cycles,
    }
    base = _gen(llm)
    setup = llm.collective_rpc(w_setup, args=())
    rec["params"] = setup[0]["params"]
    rec["install_bytes"] = setup[0]["bytes"]

    pwal = llm.collective_rpc(w_install_timed, args=("pwal", args.warm_cycles))
    rec["pwal_install"] = _crit(pwal)
    post_pwal = _gen(llm)

    mdl = llm.collective_rpc(w_install_timed, args=("mdl", args.warm_cycles))
    rec["mdl_install"] = _crit(mdl)
    if rec["pwal_install"] and rec["mdl_install"]:
        rec["mdl_speedup_vs_pwal"] = (
            rec["pwal_install"]["median_ms"] / rec["mdl_install"]["median_ms"]
        )

    # correctness: corrupt then reinstall via each installer
    results = {}
    for name in ("pwal", "mdl"):
        llm.collective_rpc(w_corrupt, args=())
        corrupt_tok = _gen(llm)
        llm.collective_rpc(w_apply, args=(name,))
        cmp = llm.collective_rpc(w_compare, args=())
        post = _gen(llm)
        results[name] = {
            "byte_mismatch": sum(c["byte_mismatch"] for c in cmp),
            "max_abs_diff": max(c["max_abs_diff"] for c in cmp),
            "corruption_detected": _agree(base, corrupt_tok) < 0.999,
            "recovery_tokens": _agree(base, post),
        }
    rec["correctness"] = results
    rec["result"] = (
        "PASS"
        if all(
            r["recovery_tokens"] >= 0.999 and r["corruption_detected"]
            for r in results.values()
        )
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
