# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator side: a live vLLM engine, refit through the collective path.

The correctness argument is generation, not a tensor comparison. vLLM loads the
real checkpoint and answers a fixed prompt set greedily; its weights are then
overwritten with noise, which makes it answer differently; the trainer refits
it over the collective path; and the answers must come back token for token.

That ordering matters. A refit that moves nothing is fast and would compare
equal against a model that still held the right weights, so the corruption step
is what gives the final comparison any power at all.
"""

from __future__ import annotations

import argparse
import json
import os
import time


def token_ids(outputs) -> list[list[int]]:
    return [list(out.outputs[0].token_ids) for out in outputs]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trainers", type=int, required=True)
    parser.add_argument("--generators", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-fraction", type=float, default=0.4)
    parser.add_argument(
        "--dst-layout", default="replicate", choices=["replicate", "sharded"]
    )
    parser.add_argument("--out", default="/work/out")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    prompts = [
        "The capital of France is",
        "Water boils at a temperature of",
        "The first three prime numbers are",
        "In one sentence, a transformer is",
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    engine_start = time.perf_counter()
    llm = LLM(
        model=args.model_dir,
        tensor_parallel_size=args.generators,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_fraction,
        enforce_eager=True,
        worker_extension_cls="mx_m2n_e2e.vllm_loader.MxRefitWorker",
    )
    engine_s = time.perf_counter() - engine_start
    print(f"[gen] vLLM up in {engine_s:.1f}s", flush=True)

    described = llm.collective_rpc("mx_describe")
    print(f"[gen] workers: {described}", flush=True)

    reference = token_ids(llm.generate(prompts, sampling))
    print(f"[gen] reference tokens: {reference[0][:12]}", flush=True)

    llm.collective_rpc("mx_corrupt", kwargs={"seed": 1234})
    corrupted = token_ids(llm.generate(prompts, sampling))
    print(f"[gen] corrupted tokens: {corrupted[0][:12]}", flush=True)
    if corrupted == reference:
        print(
            "[gen] FAILED: corruption did not change generation, so a later match "
            "would prove nothing",
            flush=True,
        )
        return 1

    joined = llm.collective_rpc(
        "mx_join",
        kwargs={
            "model_dir": args.model_dir,
            "endpoint": args.endpoint,
            "trainers": args.trainers,
            "generators": args.generators,
            "model_name": args.model_name,
            "run_id": args.run_id,
            "dst_layout": args.dst_layout,
        },
    )
    print(f"[gen] joined: {joined}", flush=True)

    rounds = []
    restored: list[list[int]] | None = None
    for index in range(args.rounds):
        version = f"v{index}"
        timings = llm.collective_rpc("mx_refit", kwargs={"version": version})
        rounds.append({"round": index, "per_rank": timings})
        slowest = max(t["total_s"] for t in timings)
        print(f"[gen] refit {version}: slowest rank {slowest * 1e3:.1f}ms", flush=True)
        if index == 0:
            restored = token_ids(llm.generate(prompts, sampling))
            print(f"[gen] restored tokens: {restored[0][:12]}", flush=True)

    ok = restored == reference
    result = {
        "run_id": args.run_id,
        "model_dir": args.model_dir,
        "role": "generator",
        "dst_layout": args.dst_layout,
        "engine_start_s": engine_s,
        "tensor_parallel_size": args.generators,
        "trainers": args.trainers,
        "join": joined,
        "rounds": rounds,
        "generation": {
            "prompts": prompts,
            "reference": reference,
            "corrupted": corrupted,
            "restored": restored,
            "match": ok,
        },
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "generator.json"), "w") as handle:
        json.dump(result, handle, indent=2)

    llm.collective_rpc("mx_cleanup")
    print("E2E " + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
