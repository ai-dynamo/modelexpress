#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build dashboard-ready time series from one autoscale collection run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.strip().strip('"').replace("Z", "+00:00"))


def ns_to_dt(value: int | float) -> datetime:
    return datetime.fromtimestamp(value / 1e9, tz=timezone.utc)


def metric_value(value):
    if isinstance(value, dict):
        return value.get("value")
    return value


def percentile(values, pct: float):
    series = sorted(float(v) for v in values if v is not None)
    if not series:
        return ""
    if len(series) == 1:
        return series[0]
    pos = (len(series) - 1) * pct / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return series[int(pos)]
    return series[lo] + (series[hi] - series[lo]) * (pos - lo)


def mean(values):
    series = [float(v) for v in values if v not in ("", None)]
    if not series:
        return ""
    return sum(series) / len(series)


def format_value(value):
    if value == "" or value is None:
        return ""
    if isinstance(value, (int, float)):
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.3f}"
    return str(value)


def load_run_config(out_dir: Path) -> dict:
    path = out_dir / "run_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def clean_log_line(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_log_timestamp(line: str) -> datetime | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)", line)
    if not match:
        return None
    return parse_dt(match.group(1))


def parse_kv_fields(line: str) -> dict[str, str]:
    fields = {}
    for key, quoted, bare in re.findall(r"(\w+)=(?:\"([^\"]*)\"|([^,\s]+))", line):
        fields[key] = quoted or bare
    return fields


def target_rps_for_second(run_config: dict, second: int):
    rates = [float(x) for x in str(run_config.get("step_rates", "")).split(",") if x.strip()]
    durations = [
        float(x) for x in str(run_config.get("step_durations", "")).split(",") if x.strip()
    ]
    if len(durations) == 1 and len(rates) > 1:
        durations = durations * len(rates)
    if not rates or len(rates) != len(durations):
        return ""
    elapsed = 0.0
    for rate, duration in zip(rates, durations):
        if elapsed <= second < elapsed + duration:
            return rate
        elapsed += duration
    return ""


def target_concurrency_for_second(run_config: dict, second: int):
    if run_config.get("traffic_mode") == "additive_concurrency_step":
        baseline = run_config.get("baseline_concurrency")
        surge = run_config.get("surge_concurrency")
        baseline_duration = run_config.get("baseline_duration_seconds")
        surge_duration = run_config.get("surge_duration_seconds")
        try:
            baseline = float(baseline)
            surge = float(surge)
            baseline_duration = float(baseline_duration)
            surge_duration = float(surge_duration or 0)
        except (TypeError, ValueError):
            return ""
        if 0 <= second < baseline_duration:
            return baseline
        if baseline_duration <= second < baseline_duration + surge_duration:
            return surge
        return ""

    if run_config.get("traffic_mode") == "continuous_concurrency_ramp":
        target = run_config.get("surge_concurrency")
        ramp_duration = run_config.get("concurrency_ramp_duration_seconds")
        total_duration = run_config.get("total_duration_seconds")
        try:
            target = float(target)
            ramp_duration = float(ramp_duration or 0)
            total_duration = float(total_duration or 0)
        except (TypeError, ValueError):
            return ""
        if total_duration and second >= total_duration:
            return ""
        if ramp_duration <= 0:
            return target
        if second < ramp_duration:
            return max(1.0, target * (second / ramp_duration))
        return target

    baseline = run_config.get("baseline_concurrency")
    surge = run_config.get("surge_concurrency")
    baseline_duration = run_config.get("baseline_duration_seconds")
    surge_duration = run_config.get("surge_duration_seconds")
    if baseline in (None, "") or surge in (None, "") or baseline_duration in (None, ""):
        return ""
    try:
        baseline = int(baseline)
        surge = int(surge)
        baseline_duration = float(baseline_duration)
        surge_duration = float(surge_duration or 0)
    except (TypeError, ValueError):
        return ""
    if 0 <= second < baseline_duration:
        return baseline
    if baseline_duration <= second < baseline_duration + surge_duration:
        return surge
    return ""


def find_profile_jsonls(out_dir: Path) -> list[Path]:
    artifacts = out_dir / "aiperf_artifacts"
    if not artifacts.exists():
        return []
    candidates = sorted(artifacts.glob("**/profile_export.jsonl"))
    return candidates


def parse_prom_metric_line(line: str):
    if not line or line.startswith("#"):
        return None, None
    match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$", line)
    if not match:
        return None, None
    try:
        return match.group(1), float(match.group(2))
    except ValueError:
        return None, None


def parse_prom_sample_time(path: Path) -> datetime | None:
    for pattern in ("%Y-%m-%dT%H-%M-%SZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(path.stem, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def append_frontend_prom_samples(out_dir: Path, bins: list[dict], bin_idx) -> None:
    metrics_dir = out_dir / "samples" / "frontend_metrics"
    if not metrics_dir.exists():
        return

    for path in sorted(metrics_dir.glob("*.prom")):
        timestamp = parse_prom_sample_time(path)
        if not timestamp:
            continue
        idx = bin_idx(timestamp)
        if not (0 <= idx < len(bins)):
            continue

        gauges = {
            "dynamo_frontend_queued_requests": [],
            "dynamo_frontend_router_queue_pending_requests": [],
            "dynamo_frontend_inflight_requests": [],
            "dynamo_request_plane_inflight_requests": [],
            "dynamo_component_inflight_requests": [],
        }
        for line in path.read_text(errors="ignore").splitlines():
            metric, value = parse_prom_metric_line(line)
            if metric in gauges:
                gauges[metric].append(value)

        if gauges["dynamo_frontend_queued_requests"]:
            bins[idx]["queue_depth"].append(sum(gauges["dynamo_frontend_queued_requests"]))
        elif gauges["dynamo_frontend_router_queue_pending_requests"]:
            bins[idx]["queue_depth"].append(
                sum(gauges["dynamo_frontend_router_queue_pending_requests"])
            )

        if gauges["dynamo_frontend_inflight_requests"]:
            bins[idx]["inflight_requests_metric"].append(
                sum(gauges["dynamo_frontend_inflight_requests"])
            )
        elif gauges["dynamo_request_plane_inflight_requests"]:
            bins[idx]["inflight_requests_metric"].append(
                sum(gauges["dynamo_request_plane_inflight_requests"])
            )
        elif gauges["dynamo_component_inflight_requests"]:
            bins[idx]["inflight_requests_metric"].append(
                sum(gauges["dynamo_component_inflight_requests"])
            )


def find_frontend_logs(out_dir: Path) -> list[Path]:
    logs_dir = out_dir / "logs"
    if not logs_dir.exists():
        return []
    return sorted(logs_dir.glob("*frontend*.k8s.log"))


def build_worker_routing_series(
    out_dir: Path, traffic_start: datetime, slice_seconds: int, bin_count: int
) -> tuple[Path, Path]:
    completions: dict[tuple[int, str], dict] = defaultdict(
        lambda: {
            "completed_success": 0,
            "prefill_success": 0,
            "decode_success": 0,
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "ttft": [],
            "itl": [],
            "elapsed": [],
        }
    )
    selections: dict[tuple[int, str, str], dict] = defaultdict(
        lambda: {"selected_count": 0, "logit": []}
    )

    def bin_idx(timestamp: datetime) -> int:
        return int((timestamp - traffic_start).total_seconds() // slice_seconds)

    for path in find_frontend_logs(out_dir):
        with path.open(errors="ignore") as file:
            for raw_line in file:
                line = clean_log_line(raw_line)
                timestamp = parse_log_timestamp(line)
                if not timestamp:
                    continue
                idx = bin_idx(timestamp)
                if not (0 <= idx < bin_count):
                    continue

                selected = re.search(
                    r"Selected worker: worker_type=(\w+), worker_id=(\d+).*?logit: ([\d.]+)",
                    line,
                )
                if selected:
                    key = (idx, selected.group(1), selected.group(2))
                    selections[key]["selected_count"] += 1
                    selections[key]["logit"].append(float(selected.group(3)))
                    continue

                if "request completed" not in line or "status=success" not in line:
                    continue

                fields = parse_kv_fields(line)
                prefill_worker_id = fields.get("prefill_worker_id")
                decode_worker_id = fields.get("decode_worker_id")
                if not prefill_worker_id and not decode_worker_id:
                    continue

                for role, worker_id in (
                    ("prefill", prefill_worker_id),
                    ("decode", decode_worker_id),
                ):
                    if not worker_id:
                        continue
                    bucket = completions[(idx, worker_id)]
                    if role == "prefill":
                        bucket["prefill_success"] += 1
                    else:
                        bucket["decode_success"] += 1
                    if prefill_worker_id == decode_worker_id or role == "decode":
                        bucket["completed_success"] += 1
                        for field, target in (
                            ("input_tokens", "input_tokens"),
                            ("output_tokens", "output_tokens"),
                        ):
                            try:
                                bucket[target] += float(fields.get(field, 0) or 0)
                            except ValueError:
                                pass
                        for field, target in (
                            ("ttft_ms", "ttft"),
                            ("avg_itl_ms", "itl"),
                            ("elapsed_ms", "elapsed"),
                        ):
                            try:
                                bucket[target].append(float(fields[field]))
                            except (KeyError, ValueError):
                                pass

    routing_path = out_dir / f"worker_routing_{slice_seconds}s.csv"
    with routing_path.open("w", newline="") as file:
        fields = [
            "slice_start",
            "seconds_from_traffic_start",
            "worker_id",
            "completed_success",
            "prefill_success",
            "decode_success",
            "input_tokens_sum",
            "output_tokens_sum",
            "ttft_avg_ms",
            "itl_avg_ms",
            "request_latency_avg_ms",
        ]
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for (idx, worker_id), bucket in sorted(completions.items()):
            start = traffic_start + timedelta(seconds=idx * slice_seconds)
            row = {
                "slice_start": start.isoformat().replace("+00:00", "Z"),
                "seconds_from_traffic_start": idx * slice_seconds,
                "worker_id": worker_id,
                "completed_success": bucket["completed_success"],
                "prefill_success": bucket["prefill_success"],
                "decode_success": bucket["decode_success"],
                "input_tokens_sum": bucket["input_tokens"],
                "output_tokens_sum": bucket["output_tokens"],
                "ttft_avg_ms": mean(bucket["ttft"]),
                "itl_avg_ms": mean(bucket["itl"]),
                "request_latency_avg_ms": mean(bucket["elapsed"]),
            }
            writer.writerow({key: format_value(value) for key, value in row.items()})

    selection_path = out_dir / f"worker_selection_{slice_seconds}s.csv"
    with selection_path.open("w", newline="") as file:
        fields = [
            "slice_start",
            "seconds_from_traffic_start",
            "worker_type",
            "worker_id",
            "selected_count",
            "logit_avg",
        ]
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for (idx, worker_type, worker_id), bucket in sorted(selections.items()):
            start = traffic_start + timedelta(seconds=idx * slice_seconds)
            row = {
                "slice_start": start.isoformat().replace("+00:00", "Z"),
                "seconds_from_traffic_start": idx * slice_seconds,
                "worker_type": worker_type,
                "worker_id": worker_id,
                "selected_count": bucket["selected_count"],
                "logit_avg": mean(bucket["logit"]),
            }
            writer.writerow({key: format_value(value) for key, value in row.items()})

    return routing_path, selection_path


def build_time_series(out_dir: Path, slice_seconds: int) -> Path:
    run_config = load_run_config(out_dir)
    profile_jsonls = find_profile_jsonls(out_dir)
    if not profile_jsonls:
        raise SystemExit(f"No profile_export.jsonl found under {out_dir / 'aiperf_artifacts'}")

    marker_path = out_dir / "traffic_marker_time.txt"
    traffic_start = parse_dt(marker_path.read_text().strip()) if marker_path.exists() else None

    scale_event = {}
    scale_path = out_dir / "scale_event.json"
    if scale_path.exists():
        scale_event = json.loads(scale_path.read_text())
    scale_dt = parse_dt(scale_event.get("timestamp"))

    records = []
    for profile_jsonl in profile_jsonls:
        for line in profile_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata", {})
            start_ns = metadata.get("request_start_ns") or metadata.get("credit_issued_ns")
            end_ns = metadata.get("request_end_ns") or start_ns
            if not start_ns:
                continue
            metrics = row.get("metrics") or {}
            error = row.get("error") or None
            records.append(
                {
                    "start": ns_to_dt(start_ns),
                    "end": ns_to_dt(end_ns),
                    "is_error": bool(error),
                    "error_code": error.get("code") if error else "",
                    "request_latency_ms": metric_value(metrics.get("request_latency")),
                    "ttft_ms": metric_value(metrics.get("time_to_first_token")),
                    "itl_ms": metric_value(metrics.get("inter_token_latency")),
                    "isl_tokens": metric_value(metrics.get("input_sequence_length")),
                    "osl_tokens": metric_value(metrics.get("output_sequence_length")),
                    "output_token_count": metric_value(
                        metrics.get("output_token_count") or metrics.get("output_sequence_length")
                    ),
                    "good_request_count": metric_value(metrics.get("good_request_count")),
                }
            )

    if not records:
        raise SystemExit(f"No request records found in {out_dir / 'aiperf_artifacts'}")

    first_record_start = min(record["start"] for record in records)
    if run_config.get("traffic_start_source", "first_request") == "first_request":
        traffic_start = first_record_start
    elif traffic_start is None:
        traffic_start = first_record_start

    last = max([traffic_start] + [record["start"] for record in records] + [record["end"] for record in records])
    for path in (
        out_dir / "samples" / "gpu.csv",
        out_dir / "samples" / "pod_samples.csv",
        out_dir / "samples" / "replica_timeline.csv",
    ):
        if not path.exists():
            continue
        with path.open() as file:
            for row in csv.DictReader(file):
                timestamp = parse_dt(row.get("timestamp"))
                if timestamp and timestamp > last:
                    last = timestamp

    bin_count = max(1, math.ceil((last - traffic_start).total_seconds() / slice_seconds) + 1)
    bins = []
    for idx in range(bin_count):
        start = traffic_start + timedelta(seconds=idx * slice_seconds)
        bins.append(
            {
                "idx": idx,
                "start": start,
                "end": start + timedelta(seconds=slice_seconds),
                "started": [],
                "completed": [],
                "gpu_util": [],
                "gpu_mem": [],
                "replicas": [],
                "ready_source_pods": [],
                "queue_depth": [],
                "inflight_requests_metric": [],
                "inflight_overlap_seconds": 0.0,
                "p2p_events": 0,
            }
        )

    def bin_idx(timestamp: datetime) -> int:
        return int((timestamp - traffic_start).total_seconds() // slice_seconds)

    for record in records:
        idx = bin_idx(record["start"])
        if 0 <= idx < len(bins):
            bins[idx]["started"].append(record)
        idx = bin_idx(record["end"])
        if 0 <= idx < len(bins):
            bins[idx]["completed"].append(record)
        first_overlap_idx = max(0, bin_idx(record["start"]))
        last_overlap_idx = min(len(bins) - 1, bin_idx(record["end"]))
        for overlap_idx in range(first_overlap_idx, last_overlap_idx + 1):
            overlap = (
                min(record["end"], bins[overlap_idx]["end"])
                - max(record["start"], bins[overlap_idx]["start"])
            ).total_seconds()
            if overlap > 0:
                bins[overlap_idx]["inflight_overlap_seconds"] += overlap

    append_frontend_prom_samples(out_dir, bins, bin_idx)

    replica_path = out_dir / "samples" / "replica_timeline.csv"
    if replica_path.exists():
        with replica_path.open() as file:
            for row in csv.DictReader(file):
                timestamp = parse_dt(row.get("timestamp"))
                if not timestamp:
                    continue
                idx = bin_idx(timestamp)
                if 0 <= idx < len(bins):
                    try:
                        bins[idx]["replicas"].append(
                            int(row.get("status_replicas") or row.get("spec_replicas") or 0)
                        )
                    except ValueError:
                        pass

    pod_path = out_dir / "samples" / "pod_samples.csv"
    if pod_path.exists():
        with pod_path.open() as file:
            for row in csv.DictReader(file):
                timestamp = parse_dt(row.get("timestamp"))
                if not timestamp:
                    continue
                idx = bin_idx(timestamp)
                if (
                    0 <= idx < len(bins)
                    and "source-" in row.get("pod", "")
                    and str(row.get("container_ready", "")).lower() == "true"
                ):
                    bins[idx]["ready_source_pods"].append(row.get("pod", ""))

    gpu_path = out_dir / "samples" / "gpu.csv"
    if gpu_path.exists():
        with gpu_path.open() as file:
            for row in csv.DictReader(file):
                timestamp = parse_dt(row.get("timestamp"))
                if not timestamp:
                    continue
                idx = bin_idx(timestamp)
                if 0 <= idx < len(bins):
                    try:
                        bins[idx]["gpu_util"].append(float(str(row["gpu_util_percent"]).strip()))
                        bins[idx]["gpu_mem"].append(float(str(row["memory_used_mib"]).strip()))
                    except (KeyError, ValueError):
                        pass

    p2p_path = out_dir / "p2p_weight_transfer_events.csv"
    if p2p_path.exists():
        with p2p_path.open() as file:
            for row in csv.DictReader(file):
                timestamp = parse_dt(row.get("success_at"))
                if not timestamp:
                    continue
                idx = bin_idx(timestamp)
                if 0 <= idx < len(bins):
                    bins[idx]["p2p_events"] += 1

    out = out_dir / f"time_series_{slice_seconds}s.csv"
    fields = [
        "slice_start",
        "seconds_from_traffic_start",
        "target_rps",
        "target_concurrency",
        "started_requests",
        "completed_success",
        "completed_errors",
        "completed_503_errors",
        "started_rps",
        "completed_success_rps",
        "request_throughput_avg_rps",
        "goodput_avg_rps",
        "output_token_throughput_avg_tokens_per_sec",
        "request_latency_avg_ms",
        "request_latency_p50_ms",
        "request_latency_p90_ms",
        "request_latency_p95_ms",
        "ttft_avg_ms",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "ttft_p95_ms",
        "itl_avg_ms",
        "itl_p50_ms",
        "itl_p90_ms",
        "itl_p95_ms",
        "queue_depth_avg_requests",
        "inflight_requests_avg",
        "inflight_requests_estimated_avg",
        "isl_p50_tokens",
        "osl_p50_tokens",
        "status_replicas_max",
        "ready_source_pods_max",
        "gpu_util_max_percent",
        "gpu_mem_max_mib",
        "p2p_success_events",
        "scale_event",
    ]
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for bucket in bins:
            completed = bucket["completed"]
            successes = [record for record in completed if not record["is_error"]]
            errors = [record for record in completed if record["is_error"]]
            good_requests = [
                record
                for record in successes
                if float(record.get("good_request_count") or 0) > 0
            ]
            output_tokens = sum(float(record.get("output_token_count") or 0) for record in successes)
            inflight_estimate = bucket["inflight_overlap_seconds"] / slice_seconds
            seconds_from_start = bucket["idx"] * slice_seconds
            row = {
                "slice_start": bucket["start"].isoformat().replace("+00:00", "Z"),
                "seconds_from_traffic_start": seconds_from_start,
                "target_rps": target_rps_for_second(run_config, seconds_from_start),
                "target_concurrency": target_concurrency_for_second(
                    run_config, seconds_from_start
                ),
                "started_requests": len(bucket["started"]),
                "completed_success": len(successes),
                "completed_errors": len(errors),
                "completed_503_errors": sum(
                    1 for record in errors if str(record["error_code"]) == "503"
                ),
                "started_rps": len(bucket["started"]) / slice_seconds,
                "completed_success_rps": len(successes) / slice_seconds,
                "request_throughput_avg_rps": len(successes) / slice_seconds,
                "goodput_avg_rps": len(good_requests) / slice_seconds,
                "output_token_throughput_avg_tokens_per_sec": output_tokens / slice_seconds,
                "request_latency_avg_ms": mean(
                    [record["request_latency_ms"] for record in successes]
                ),
                "request_latency_p50_ms": percentile(
                    [record["request_latency_ms"] for record in successes], 50
                ),
                "request_latency_p90_ms": percentile(
                    [record["request_latency_ms"] for record in successes], 90
                ),
                "request_latency_p95_ms": percentile(
                    [record["request_latency_ms"] for record in successes], 95
                ),
                "ttft_avg_ms": mean([record["ttft_ms"] for record in successes]),
                "ttft_p50_ms": percentile([record["ttft_ms"] for record in successes], 50),
                "ttft_p90_ms": percentile([record["ttft_ms"] for record in successes], 90),
                "ttft_p95_ms": percentile([record["ttft_ms"] for record in successes], 95),
                "itl_avg_ms": mean([record["itl_ms"] for record in successes]),
                "itl_p50_ms": percentile([record["itl_ms"] for record in successes], 50),
                "itl_p90_ms": percentile([record["itl_ms"] for record in successes], 90),
                "itl_p95_ms": percentile([record["itl_ms"] for record in successes], 95),
                "queue_depth_avg_requests": mean(bucket["queue_depth"]),
                "inflight_requests_avg": mean(bucket["inflight_requests_metric"])
                if bucket["inflight_requests_metric"]
                else inflight_estimate,
                "inflight_requests_estimated_avg": inflight_estimate,
                "isl_p50_tokens": percentile(
                    [record["isl_tokens"] for record in successes], 50
                ),
                "osl_p50_tokens": percentile(
                    [record["osl_tokens"] for record in successes], 50
                ),
                "status_replicas_max": max(bucket["replicas"]) if bucket["replicas"] else "",
                "ready_source_pods_max": len(set(bucket["ready_source_pods"]))
                if bucket["ready_source_pods"]
                else 0,
                "gpu_util_max_percent": max(bucket["gpu_util"]) if bucket["gpu_util"] else "",
                "gpu_mem_max_mib": max(bucket["gpu_mem"]) if bucket["gpu_mem"] else "",
                "p2p_success_events": bucket["p2p_events"],
                "scale_event": int(scale_dt is not None and bucket["start"] <= scale_dt < bucket["end"]),
            }
            writer.writerow({key: format_value(value) for key, value in row.items()})
    build_worker_routing_series(out_dir, traffic_start, slice_seconds, bin_count)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--slice-seconds", type=int, default=30)
    args = parser.parse_args()
    print(build_time_series(args.out_dir, args.slice_seconds))


if __name__ == "__main__":
    main()
