#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot vLLM MX P2P vs vanilla disk-loading comparison."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)")


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def read_scale_time(run_dir: Path) -> datetime:
    return parse_dt(json.loads((run_dir / "scale_event.json").read_text())["timestamp"])


def read_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def read_leader_ready(run_dir: Path, pod_prefix: str) -> datetime:
    with (run_dir / "pod_timeline.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    candidates = [
        row
        for row in rows
        if f"{pod_prefix}-0-vllmworker-1-vllmworker-ldr" in row["pod"]
    ]
    if len(candidates) != 1:
        raise SystemExit(f"expected one scaled leader pod in {run_dir}, found {len(candidates)}")
    return parse_dt(candidates[0]["ready"])


def as_float(value: str | float | int | None) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


def read_log_lines(path: Path) -> list[str]:
    return path.read_bytes().decode(errors="ignore").split("\n")


def read_time_series(run_dir: Path, x_max: int) -> list[dict[str, float | str]]:
    with (run_dir / "time_series_30s.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    parsed: list[dict[str, float | str]] = []
    for row in rows:
        seconds = as_float(row["seconds_from_traffic_start"])
        if seconds > x_max:
            continue
        active = (
            as_float(row["started_requests"]) > 0
            or as_float(row["completed_success"]) > 0
            or as_float(row["completed_errors"]) > 0
        )
        if as_float(row["target_concurrency"]) <= 0:
            active = False
        if not active:
            continue
        converted: dict[str, float | str] = {"slice_start": row["slice_start"]}
        for key, value in row.items():
            if key == "slice_start":
                continue
            converted[key] = as_float(value)
        parsed.append(converted)
    return parsed


def extract_max_float(run_dir: Path, pattern: str, pod_fragment: str) -> float:
    regex = re.compile(pattern)
    values: list[float] = []
    for path in (run_dir / "logs").glob("*.k8s.log"):
        if pod_fragment not in path.name:
            continue
        for line in read_log_lines(path):
            match = regex.search(line)
            if match:
                values.append(float(match.group(1)))
    if not values:
        raise SystemExit(f"no values matching {pattern!r} in {run_dir}")
    return max(values)


def parse_log_ts(line: str) -> datetime | None:
    match = LOG_TS_RE.search(line)
    if not match:
        return None
    return parse_dt(match.group(1))


def find_log_time(run_dir: Path, pod_fragment: str, pattern: str, *, last: bool = False) -> datetime:
    regex = re.compile(pattern)
    matches: list[datetime] = []
    for path in sorted((run_dir / "logs").glob("*.k8s.log")):
        if pod_fragment not in path.name:
            continue
        for line in read_log_lines(path):
            if regex.search(line):
                ts = parse_log_ts(line)
                if ts is not None:
                    matches.append(ts)
    if not matches:
        raise SystemExit(f"no timestamp matching {pattern!r} in {run_dir}")
    return matches[-1] if last else matches[0]


def startup_breakdown(
    run_dir: Path,
    pod_prefix: str,
    scale_time: datetime,
    ready_time: datetime,
    load_start_pattern: str,
    load_end_pattern: str,
) -> tuple[int, int, int]:
    pod_fragment = f"{pod_prefix}-0-vllmworker-1-vllmworker-ldr"
    load_start = find_log_time(run_dir, pod_fragment, load_start_pattern)
    load_end = find_log_time(run_dir, pod_fragment, load_end_pattern, last=True)
    total = round((ready_time - scale_time).total_seconds())
    pre = max(0, round((load_start - scale_time).total_seconds()))
    load = max(0, round((load_end - load_start).total_seconds()))
    post = max(0, total - pre - load)
    return pre, load, post


def add_markers(ax, scale_x: float, mx_ready_x: float, disk_ready_x: float, x_limit: float) -> None:
    ax.axvline(scale_x, color="#ff4d4d", linestyle="--", linewidth=1.4, label="Scale-up trigger")
    ax.axvline(mx_ready_x, color="#1f77b4", linestyle=":", linewidth=1.6, label="MX replica ready")
    ax.axvline(disk_ready_x, color="#ff7f0e", linestyle=":", linewidth=1.6, label="Disk replica ready")
    ax.set_xlim(0, x_limit)


def plot_series(ax, mx, disk, col, title, ylabel, scale_x, mx_ready_x, disk_ready_x, x_limit, legend=True):
    ax.plot([r["seconds_from_traffic_start"] for r in mx], [r[col] for r in mx], marker="o", markersize=3.5, linewidth=1.5, label="MX P2P")
    ax.plot([r["seconds_from_traffic_start"] for r in disk], [r[col] for r in disk], marker="s", markersize=3.5, linewidth=1.5, label="Disk")
    add_markers(ax, scale_x, mx_ready_x, disk_ready_x, x_limit)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc="best", fontsize=8)


def label_bar(ax, x, y, text, color="black") -> None:
    ax.text(x, y, text, ha="center", va="bottom", fontsize=10, fontweight="bold", color=color)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mx-run", required=True, type=Path)
    parser.add_argument("--disk-run", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--x-max", type=int, default=1800)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    x_max = args.x_max
    mx = read_time_series(args.mx_run, x_max)
    disk = read_time_series(args.disk_run, x_max)
    mx_config = read_run_config(args.mx_run)
    disk_config = read_run_config(args.disk_run)

    mx_start = parse_dt(str(mx[0]["slice_start"]))
    disk_start = parse_dt(str(disk[0]["slice_start"]))
    mx_scale_x = (read_scale_time(args.mx_run) - mx_start).total_seconds()
    disk_scale_x = (read_scale_time(args.disk_run) - disk_start).total_seconds()
    scale_x = (mx_scale_x + disk_scale_x) / 2
    mx_ready_time = read_leader_ready(args.mx_run, "mx-vllm")
    disk_ready_time = read_leader_ready(args.disk_run, "vanilla-vllm")
    mx_ready_x = (mx_ready_time - mx_start).total_seconds()
    disk_ready_x = (disk_ready_time - disk_start).total_seconds()
    max_series_x = max(
        [as_float(r["seconds_from_traffic_start"]) for r in mx + disk],
        default=0,
    )
    x_limit = max(1020, disk_ready_x + 30, mx_ready_x + 30, max_series_x + 30)

    mx_scale_time = read_scale_time(args.mx_run)
    disk_scale_time = read_scale_time(args.disk_run)
    mx_startup = mx_ready_time - mx_scale_time
    disk_startup = disk_ready_time - disk_scale_time
    mx_startup_s = round(mx_startup.total_seconds())
    disk_startup_s = round(disk_startup.total_seconds())

    mx_pre, mx_load_s, mx_post = startup_breakdown(
        args.mx_run,
        "mx-vllm",
        mx_scale_time,
        mx_ready_time,
        r"MxModelLoader starting",
        r"Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds",
    )
    disk_pre, disk_load_s, disk_post = startup_breakdown(
        args.disk_run,
        "vanilla-vllm",
        disk_scale_time,
        disk_ready_time,
        r"Loading safetensors checkpoint shards:.*0% Completed",
        r"Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds",
    )
    mx_rdmabw = extract_max_float(
        args.mx_run,
        r"GB, [0-9.]+s, ([0-9.]+) Gbps",
        "vllmworker-1-vllmworker-ldr",
    )

    drop = round((disk_startup_s - mx_startup_s) * 100 / disk_startup_s)
    faster = disk_startup_s - mx_startup_s
    load_saving = disk_load_s - mx_load_s

    fig, axes = plt.subplots(4, 2, figsize=(22, 22))
    fig.suptitle("MX P2P vs Disk Loading: Kimi K2.5 Autoscale Demo Metrics", fontsize=22, fontweight="bold", y=0.985)
    mx_conc = f"{mx_config.get('baseline_concurrency', '?')} -> {mx_config.get('surge_concurrency', '?')}"
    disk_conc = f"{disk_config.get('baseline_concurrency', '?')} -> {disk_config.get('surge_concurrency', '?')}"
    mx_isl = mx_config.get("trace_max_isl", "?")
    disk_isl = disk_config.get("trace_max_isl", "?")
    isl_note = f"TRACE_MAX_ISL: MX {mx_isl}, Disk {disk_isl}" if mx_isl != disk_isl else f"TRACE_MAX_ISL: {mx_isl}"
    fig.text(
        0.5,
        0.956,
        f"Model: nvidia/Kimi-K2.5-NVFP4 | TP=4, PP=2 | Nodes=2 | PV: Lustre CSI (RWX shared model cache) | AIPerf Mooncake trace | concurrency MX {mx_conc}, Disk {disk_conc} | {isl_note}",
        ha="center",
        fontsize=11,
    )

    series = [
        ("ttft_p50_ms", "TTFT p50", "TTFT (ms)"),
        ("ttft_p90_ms", "TTFT p90", "TTFT (ms)"),
        ("request_latency_p50_ms", "Request Latency p50", "Request Latency (ms)"),
        ("request_latency_p90_ms", "Request Latency p90", "Request Latency (ms)"),
        ("output_token_throughput_avg_tokens_per_sec", "Output Token Throughput (avg)", "Output Token Throughput (tokens/sec)"),
        ("request_throughput_avg_rps", "Request Throughput (avg)", "Request Throughput (req/sec)"),
    ]

    for ax, (col, title, ylabel) in zip(axes[:3].ravel(), series):
        plot_series(ax, mx, disk, col, title, ylabel, scale_x, mx_ready_x, disk_ready_x, x_limit)
    for ax in axes[2]:
        ax.set_xlabel("Time from first request (s)")

    ax = axes[3, 0]
    ax.bar(["Disk", "MX P2P"], [disk_startup_s, mx_startup_s], color=["#ff7f0e", "#1f77b4"], width=0.52)
    ax.set_title("Overall Startup Latency", fontweight="bold")
    ax.set_ylabel("Scale trigger to replica ready (s)")
    ax.grid(True, axis="y", alpha=0.3)
    startup_ylim = max(disk_startup_s, mx_startup_s) * 1.18
    ax.set_ylim(0, startup_ylim)
    label_bar(ax, 0, disk_startup_s + startup_ylim * 0.02, f"{disk_startup_s}s\n({disk_startup_s/60:.1f}m)")
    label_bar(ax, 1, mx_startup_s + startup_ylim * 0.02, f"{mx_startup_s}s\n({mx_startup_s/60:.1f}m)")
    ax.annotate(
        f"{drop}% drop\n{faster}s faster",
        xy=(1, mx_startup_s),
        xytext=(0.45, disk_startup_s * 0.72),
        arrowprops={"arrowstyle": "->", "color": "#2f5d50", "linewidth": 1.8},
        color="#2f5d50",
        fontsize=12,
        fontweight="bold",
        ha="center",
    )

    ax = axes[3, 1]
    labels = ["Disk", "MX P2P"]
    pre = [disk_pre, mx_pre]
    load = [disk_load_s, mx_load_s]
    post = [disk_post, mx_post]
    ax.bar(labels, pre, color="#a7adb5", label="Pre-weight init")
    ax.bar(labels, load, bottom=pre, color="#f28e2b", label="Weight load / P2P")
    ax.bar(labels, post, bottom=[pre[i] + load[i] for i in range(2)], color="#6f9973", label="Post-weight to Ready")
    ax.set_title("Startup Phase Breakdown", fontweight="bold")
    ax.set_ylabel("Seconds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    max_startup = max(disk_startup_s, mx_startup_s)
    breakdown_ylim = max_startup * 1.18
    ax.set_ylim(0, breakdown_ylim)
    for i, total in enumerate([disk_startup_s, mx_startup_s]):
        label_bar(ax, i, total + breakdown_ylim * 0.02, f"total {total}s")
        for bottom, value in [(0, pre[i]), (pre[i], load[i]), (pre[i] + load[i], post[i])]:
            if value >= 25:
                ax.text(i, bottom + value / 2, f"{value}s", ha="center", va="center", fontsize=10, fontweight="bold", color="white" if value != pre[i] else "#1f2933")
            else:
                ax.text(i + 0.14, bottom + value + max_startup * 0.01, f"{value}s", ha="left", va="bottom", fontsize=9, fontweight="bold", color="#1f2933")
    ax.text(0.5, breakdown_ylim * 0.96, f"Weight path saves {load_saving}s; net saves {faster}s", ha="center", color="#2f5d50", fontsize=12, fontweight="bold")
    ax.text(0.5, breakdown_ylim * 0.91, f"MX RDMA peak {mx_rdmabw:.1f} Gbps/rank", ha="center", color="#555", fontsize=10)

    definition = (
        "Definitions: Overall startup latency = controlled scale-up trigger to new vLLM leader pod Ready; this includes pre-weight init, "
        "weight load/transfer, graph/runtime work, and readiness. Startup phase breakdown is timestamp-based from leader logs: pre-weight init "
        "= scale trigger to first model-load/P2P log; weight load / P2P = first model-load/P2P log to vLLM Model loading complete; "
        "post-weight to Ready = model loading complete to Kubernetes Ready."
    )
    fig.text(0.05, 0.025, definition, ha="left", va="bottom", fontsize=10, wrap=True)
    fig.tight_layout(rect=[0, 0.055, 1, 0.94])

    png = args.out_dir / "vllm_mx_p2p_vs_vanilla_inference_metrics_30s_final.png"
    pdf = args.out_dir / "vllm_mx_p2p_vs_vanilla_inference_metrics_30s_final.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
