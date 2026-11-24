# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Live AsyncLLM streaming pause A/B; no HTTP, S3 staging or trainer timing.

Uses production-example pause mode keep/clear_cache=False. In-flight requests
span weight versions, so their output is NOT a cold-target correctness oracle.
Fresh requests after draining/cache reset are verified separately, outside timing.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

PROMPTS = [
    "The capital of France is",
    "Compute 17 + 25. Answer:",
    "Explain gravity in one sentence.",
    "List the days of the week:",
]


async def main(args):
    os.environ["MX_REFIT_CHECKPOINT_SOURCE_REPLAY"] = "false"
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    output = Path(args.output)
    if output.exists():
        raise RuntimeError("Refuse to append to an existing run")

    def emit(row):
        with output.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print("PAUSE_RESULT", row["kind"], row.get("pair"), row.get("arm"), flush=True)

    llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            tensor_parallel_size=4,
            dtype="bfloat16",
            max_model_len=512,
            max_num_seqs=4,
            gpu_memory_utilization=0.5,
            enable_sleep_mode=True,
            weight_transfer_config={"backend": "modelexpress"},
            kernel_config={"moe_backend": "triton"},
            enable_prefix_caching=False,
            worker_extension_cls="receiver_ab.WinterWorkerExtension",
            seed=123,
        )
    )
    counter = 0

    async def rpc(action, path=None):
        path = args.model if path is None else path
        return await llm.collective_rpc(
            "winter_call", args=(action, path), timeout=1200
        )

    async def generate_short():
        nonlocal counter
        results = []
        for prompt in PROMPTS[:3]:
            counter += 1
            last = None
            async for item in llm.generate(
                prompt,
                SamplingParams(temperature=0, max_tokens=24, seed=123, logprobs=1),
                f"check-{counter}",
            ):
                last = item.outputs[0]
            results.append(
                {
                    "tokens": last.token_ids,
                    "text": last.text,
                    "logprobs": [
                        p[t].logprob for p, t in zip(last.logprobs, last.token_ids)
                    ],
                }
            )
        return results

    async def reset_seed():
        await llm.pause_generation(mode="wait", clear_cache=True)
        await rpc("ordinary")
        await llm.resume_generation()

    async def verify(ref, expected):
        # No requests are active. Verification is outside the timed pause.
        await llm.pause_generation(mode="wait", clear_cache=True)
        actual = await rpc("hash")
        if actual != ref:
            raise RuntimeError("registered bytes/storage changed")
        await llm.resume_generation()
        if await generate_short() != expected:
            raise RuntimeError("fresh post-update inference mismatch")

    async def burst(pair, arm):
        nonlocal counter
        events = [[] for _ in PROMPTS]
        ready = [asyncio.Event() for _ in PROMPTS]
        request_rows = [None for _ in PROMPTS]

        async def request(i):
            nonlocal counter
            counter += 1
            rid = f"stream-{counter}"
            start = time.perf_counter()
            last_count = 0
            async for item in llm.generate(
                PROMPTS[i],
                SamplingParams(
                    temperature=0, max_tokens=384, ignore_eos=True, seed=123
                ),
                rid,
            ):
                count = len(item.outputs[0].token_ids)
                if count > last_count:
                    events[i].append(
                        {
                            "t": time.perf_counter(),
                            "tokens": count,
                            "new_tokens": count - last_count,
                        }
                    )
                    last_count = count
                    if count >= 16:
                        ready[i].set()
            request_rows[i] = {
                "id": rid,
                "start": start,
                "end": time.perf_counter(),
                "tokens": last_count,
                "ttft_s": events[i][0]["t"] - start,
            }

        tasks = [asyncio.create_task(request(i)) for i in range(len(PROMPTS))]
        await asyncio.wait_for(asyncio.gather(*(e.wait() for e in ready)), timeout=120)
        if any(t.done() for t in tasks):
            raise RuntimeError("stream finished before pause")
        pause_start = time.perf_counter()
        await llm.pause_generation(mode="keep", clear_cache=False)
        pause_ack = time.perf_counter()
        if not await llm.is_paused():
            raise RuntimeError("scheduler not paused")
        install_start = time.perf_counter()
        workers = await rpc(arm, args.target) if arm != "control" else []
        install_end = time.perf_counter()
        resume_start = time.perf_counter()
        # On install failure, fail closed: no finally-resume into uncertain weights.
        await llm.resume_generation()
        resume_ack = time.perf_counter()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=180)
        stream_gaps = []
        for stream in events:
            before = [x["t"] for x in stream if x["t"] <= pause_ack]
            after = [x["t"] for x in stream if x["t"] >= resume_start]
            if not before or not after:
                raise RuntimeError("missing pause-spanning stream samples")
            stream_gaps.append(min(after) - max(before))
        return {
            "kind": "measurement",
            "pair": pair,
            "arm": arm,
            "pause_start": pause_start,
            "pause_ack": pause_ack,
            "install_start": install_start,
            "install_end": install_end,
            "resume_start": resume_start,
            "resume_ack": resume_ack,
            "pause_rpc_s": resume_ack - pause_start,
            "scheduler_hold_s": resume_start - pause_ack,
            "install_rpc_s": install_end - install_start,
            "stream_gaps_s": stream_gaps,
            "max_stream_gap_s": max(stream_gaps),
            "events_during_hold": sum(
                pause_ack < x["t"] < resume_start for ev in events for x in ev
            ),
            "requests": request_rows,
            "stream_events": events,
            "workers": workers,
        }

    try:
        seed = await rpc("hash")
        await llm.pause_generation(mode="wait", clear_cache=True)
        await rpc("ordinary", args.target)
        reference = await rpc("hash")
        if reference == seed:
            raise RuntimeError("target did not change runtime")
        await llm.resume_generation()
        expected = await generate_short()
        await reset_seed()
        await llm.pause_generation(mode="wait", clear_cache=True)
        t = time.perf_counter()
        capture = await rpc("capture")
        capture_s = time.perf_counter() - t
        if await rpc("hash") != seed:
            raise RuntimeError("capture changed seed state/storage")
        await rpc("mapped", args.target)
        await llm.resume_generation()
        await verify(reference, expected)
        emit(
            {
                "kind": "qualification",
                "result": "PASS",
                "capture_s": capture_s,
                "capture": capture,
                "reference_hashes": [
                    {k: v for k, v in h.items() if k != "pointers"} for h in reference
                ],
                "reference_outputs": expected,
                "cold_start_caveat": "Previous independent exact cold-start output gate failed; not resolved by this experiment.",
            }
        )
        await reset_seed()
        emit(await burst(-1, "control"))
        # Both paths receive one unreported warmup; correctness still checked.
        for arm in ("ordinary", "mapped"):
            await reset_seed()
            row = await burst(-1, arm)
            await verify(reference, expected)
            row["kind"] = "warmup"
            emit(row)
        for pair in range(args.pairs):
            order = ("ordinary", "mapped") if pair % 2 == 0 else ("mapped", "ordinary")
            for arm in order:
                await reset_seed()
                row = await burst(pair, arm)
                await verify(reference, expected)
                row["correctness"] = "PASS"
                emit(row)
        emit(
            {
                "kind": "complete",
                "pairs": args.pairs,
                "scope": "live engine pause and in-process streaming; no HTTP/S3/trainer",
                "pause_mode": "keep",
                "clear_cache": False,
                "concurrency": 4,
                "output_tokens": 384,
            }
        )
    finally:
        llm.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--model", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.pairs < 1:
        p.error("--pairs must be positive")
    a.model = str(Path(a.model).resolve())
    a.target = str(Path(a.target).resolve())
    if a.model == a.target:
        p.error("seed and target must differ")
    asyncio.run(main(a))
