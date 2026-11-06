# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL OBJ (S3-over-RDMA) load benchmark; loader only, no inference engine.

Loads a safetensors checkpoint from an S3-over-RDMA endpoint into GPU memory
with MxS3RdmaLoader, one process per GPU, and reports per-process and aggregate
throughput. With --metrics it also reports the server's RDMA byte counters so a
silent non-RDMA path cannot pass as a result.

Run from modelexpress_client/python with NIXL (OBJ + cuObject) installed and
the AWS_* endpoint and credential variables set:
    PYTHONPATH=. python benchmarks/bench_s3_rdma_load.py \\
        --uri s3://models/llama-3.1-8b-instruct --devices 0,1,2,3 \\
        --window-mb 256 --inflight 8 --repeats 3 \\
        --metrics http://aistor1:9100,http://aistor2:9100
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import re
import statistics
import time
import urllib.request

RDMA_METRICS_PATH = "/minio/metrics/v3/api/rdma"
RDMA_COUNTERS = ("minio_api_rdma_read_bytes_total", "minio_api_rdma_write_bytes_total")
_METRIC_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>\S+)")


def scrape_rdma(endpoints):
    totals = dict.fromkeys(RDMA_COUNTERS, 0.0)
    for endpoint in endpoints:
        with urllib.request.urlopen(endpoint.rstrip("/") + RDMA_METRICS_PATH, timeout=10) as resp:
            for line in resp.read().decode("utf-8", errors="replace").splitlines():
                match = _METRIC_RE.match(line)
                if match and match.group("name") in totals:
                    totals[match.group("name")] += float(match.group("value"))
    return totals


def worker(device, args, barrier, results):
    try:
        results.put(run_loads(device, args, barrier))
    except BaseException as exc:
        barrier.abort()
        results.put({"device": device, "error": repr(exc)})
        raise


def run_loads(device, args, barrier):
    logging.basicConfig(level=args.log_level, format=f"[gpu{device}] %(asctime)s %(name)s %(message)s")
    if args.cufile_template:
        os.environ["CUFILE_ENV_PATH_JSON"] = args.cufile_template.format(device=device)
    import torch

    from modelexpress.accelerators import CudaAcceleratorBackend
    from modelexpress.s3_rdma_loader import MxS3RdmaLoader, S3RdmaLoadConfig

    torch.cuda.set_device(device)
    config = S3RdmaLoadConfig(
        window_bytes=args.window_mb * 1024 * 1024,
        inflight=args.inflight,
        threads=args.threads,
        timeout_s=args.timeout,
        verify=args.verify,
        staging=args.staging,
    )
    runs = []
    for _ in range(args.repeats):
        barrier.wait()
        loader = MxS3RdmaLoader(CudaAcceleratorBackend(), device, config)
        start = time.perf_counter()
        tensors = 0
        try:
            plan = loader.prepare(args.uri)
            prepared = time.perf_counter()
            for _name, _tensor in loader.iter_tensors():
                tensors += 1
            torch.cuda.synchronize(device)
        finally:
            loader.release()
        end = time.perf_counter()
        runs.append(
            {
                "device": device,
                "bytes": plan.total_bytes,
                "tensors": tensors,
                "windows": len(plan.windows),
                "prepare_s": prepared - start,
                "stream_s": end - prepared,
                "total_s": end - start,
                "start": start,
                "end": end,
            }
        )
    return runs


def collect(procs, barrier, results):
    """Gather one outcome per worker without hanging if a worker dies unreported."""
    outcomes = []
    while len(outcomes) < len(procs):
        exited = all(proc.exitcode is not None for proc in procs)
        try:
            outcomes.append(results.get(timeout=1.0))
        except queue.Empty:
            if exited:
                break
            if any(proc.exitcode for proc in procs):
                barrier.abort()
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uri", required=True, help="s3://bucket/prefix of a safetensors checkpoint")
    parser.add_argument("--devices", default="0", help="comma-separated CUDA devices, one process each")
    parser.add_argument("--window-mb", type=int, default=256)
    parser.add_argument("--inflight", type=int, default=8)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--verify", choices=("tail", "full", "off"), default="tail")
    parser.add_argument("--staging", choices=("auto", "device", "host"), default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--metrics", default="", help="comma-separated AIStor base URLs for RDMA counters")
    parser.add_argument("--json-out", default="", help="write the summary JSON to this path")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--cufile-template",
        default="",
        help="per-process CUFILE_ENV_PATH_JSON, with {device} replaced by the GPU index",
    )
    args = parser.parse_args()

    devices = [int(device) for device in args.devices.split(",")]
    endpoints = [url for url in args.metrics.split(",") if url]
    before = scrape_rdma(endpoints) if endpoints else None

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(devices))
    results = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(device, args, barrier, results)) for device in devices]
    for proc in procs:
        proc.start()
    outcomes = collect(procs, barrier, results)
    for proc in procs:
        proc.join()
    errors = [f"gpu{outcome['device']}: {outcome['error']}" for outcome in outcomes if isinstance(outcome, dict)]
    if errors or any(proc.exitcode for proc in procs):
        codes = ", ".join(str(proc.exitcode) for proc in procs)
        raise SystemExit(f"benchmark failed (worker exit codes {codes}): " + "; ".join(errors))
    runs = [run for outcome in outcomes for run in outcome]

    rounds = []
    for index in range(args.repeats):
        batch = [runs[i] for i in range(len(runs)) if i % args.repeats == index]
        wall = max(run["end"] for run in batch) - min(run["start"] for run in batch)
        total_bytes = sum(run["bytes"] for run in batch)
        rounds.append({"wall_s": wall, "aggregate_gbps": total_bytes / wall / 1e9})
    per_process = [run["bytes"] / run["total_s"] / 1e9 for run in runs]
    summary = {
        "uri": args.uri,
        "devices": devices,
        "window_mb": args.window_mb,
        "inflight": args.inflight,
        "threads": args.threads or args.inflight,
        "verify": args.verify,
        "bytes_per_process": runs[0]["bytes"],
        "tensors": runs[0]["tensors"],
        "windows": runs[0]["windows"],
        "per_process_gbps_median": statistics.median(per_process),
        "prepare_s_median": statistics.median(run["prepare_s"] for run in runs),
        "aggregate_gbps_median": statistics.median(r["aggregate_gbps"] for r in rounds),
        "wall_s_median": statistics.median(r["wall_s"] for r in rounds),
        "runs": runs,
    }
    if before is not None:
        after = scrape_rdma(endpoints)
        delta = {name: after[name] - before[name] for name in RDMA_COUNTERS}
        loaded = sum(run["bytes"] for run in runs)
        summary["rdma_counter_delta"] = delta
        summary["rdma_ratio"] = max(delta.values()) / loaded if loaded else 0.0
    text = json.dumps(summary, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
