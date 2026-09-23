# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent matched benchmark for native MILES broadcast and ModelExpress M2N."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in incomplete runtimes.
    yaml = None

SCHEMA = "miles-m2n-benchmark-v1"
HARNESS_SCHEMA = "miles-m2n-matched-harness-v1"
ARMS = ("broadcast", "modelexpress")
MODE_NAMES = {
    "broadcast": "native_broadcast",
    "modelexpress": "modelexpress_m2n",
}
TRANSFER_MODES = {
    "broadcast": "broadcast",
    "modelexpress": "external",
}
HEADLINE_START_BOUNDARY = "pre_weights_getter_post_begin_barrier"
HEADLINE_END_BOUNDARY = "post_transfer_trainer_barrier"
E2E_START_BOUNDARY = "post_pause_pre_update"
E2E_END_BOUNDARY = "all_rollout_ready"
DEFAULT_NUM_ROLLOUTS = 20
EXPECTED_UPDATE_CALLS = DEFAULT_NUM_ROLLOUTS + 1
PR3304_STEADY_INDICES = range(1, 20)
PR3304_REFERENCE_INDEX = 12
PR3304_TAIL_INDEX = 20
LAUNCHER_SCAFFOLD_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LAUNCHER_SCAFFOLD_MODEL_TYPE = "qwen2.5-0.5B"
_DIGEST_IMAGE = re.compile(r"^[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_KUBE_NAME = re.compile(r"^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$")
_TIMESTAMP = re.compile(r"^(?P<timestamp>\S+)\s+")
_TIMER = re.compile(
    r"Timer "
    r"(?P<name>update_weights_implementation|finalize_and_resume_engines|update_weights) "
    r"(?P<event>start|end)"
    r"(?: \(elapsed: (?P<elapsed>[0-9]+(?:\.[0-9]+)?)s\))?"
)
_FORBIDDEN_OUTPUT_ROOTS = (
    Path("/tmp"),
    Path("/var/tmp"),
    Path("/dev/shm"),
    Path("/run"),
)


def validate_persistent_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_absolute():
        raise ValueError(f"output path must be absolute, got {path}")
    for root in _FORBIDDEN_OUTPUT_ROOTS:
        if resolved == root or root in resolved.parents:
            raise ValueError(
                f"output path must be persistent and cannot be under {root}: {resolved}"
            )
    return resolved


def validate_image(image: str, *, field: str) -> str:
    if not _DIGEST_IMAGE.fullmatch(image):
        raise ValueError(
            f"{field} must be digest-pinned as IMAGE@sha256:<64 lowercase hex>"
        )
    return image


def validate_storage_config(storage: Mapping[str, Any]) -> dict[str, str]:
    result = {}
    for field in ("model_claim", "workspace_claim"):
        value = str(storage.get(field, ""))
        if not value or len(value) > 253 or _KUBE_NAME.fullmatch(value) is None:
            raise ValueError(
                f"storage.{field} must name an explicit existing Kubernetes PVC"
            )
        result[field] = value
    return result


def build_schedule(*, blocks: int, seed: int = 3304) -> list[dict[str, Any]]:
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    randomizer = random.Random(seed)
    schedule: list[dict[str, Any]] = []
    for block in range(blocks):
        order = ARMS if randomizer.getrandbits(1) == 0 else tuple(reversed(ARMS))
        for position, arm in enumerate(order):
            schedule.append(
                {
                    "arm": arm,
                    "block": block,
                    "order": position,
                    "repetition": block,
                }
            )
    return schedule


def _parse_timestamp(value: str) -> dt.datetime:
    normalized = value
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    if "." in normalized:
        prefix, rest = normalized.split(".", 1)
        fraction, offset = rest.split("+", 1)
        normalized = f"{prefix}.{fraction[:6]}+{offset}"
    parsed = dt.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _format_timestamp(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat()


def parse_timer_log(text: str, *, expected_updates: int) -> list[dict[str, Any]]:
    """Parse legacy MILES timers from actor rank zero only.

    The exact implementation timer is the PR3304 headline. The outer actor
    timer is retained as the broader rollout-ready diagnostic.
    """

    records: list[dict[str, Any]] = []
    pending: dict[str, Any] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if "actor_cell0_rank0" not in line:
            continue
        match = _TIMER.search(line)
        if match is None:
            continue
        timestamp_match = _TIMESTAMP.match(line)
        timestamp = (
            _parse_timestamp(timestamp_match.group("timestamp"))
            if timestamp_match is not None
            else None
        )
        name = match.group("name")
        event = match.group("event")
        elapsed_text = match.group("elapsed")
        elapsed = float(elapsed_text) if elapsed_text is not None else None

        if name == "update_weights" and event == "start":
            if pending:
                raise ValueError(
                    f"new update started before the prior timer triplet ended at line {line_number}"
                )
            pending["start_timestamp"] = timestamp
            continue
        if event != "end":
            continue
        if elapsed is None:
            raise ValueError(f"timer end lacks elapsed time at line {line_number}")
        if name in pending:
            raise ValueError(
                f"duplicate Timer {name} before update completion at line {line_number}"
            )
        if name == "update_weights_implementation":
            pending[name] = elapsed
            continue
        if name == "finalize_and_resume_engines":
            if "update_weights_implementation" not in pending:
                raise ValueError(
                    "timer triplet is out of order: finalize must follow implementation"
                )
            pending[name] = elapsed
            continue
        missing = {
            "update_weights_implementation",
            "finalize_and_resume_engines",
        }.difference(pending)
        if missing:
            raise ValueError(
                "Timer update_weights ended without a complete timer triplet; "
                f"missing {', '.join(sorted(missing))}"
            )
        implementation = float(pending["update_weights_implementation"])
        finalize = float(pending["finalize_and_resume_engines"])
        primary = round(implementation + finalize, 6)
        residual_raw = round(elapsed - primary, 6)
        if residual_raw < -0.21:
            raise ValueError(
                "outer update timer is shorter than the common acceptance boundary"
            )
        records.append(
            {
                "end_timestamp_utc": _format_timestamp(timestamp),
                "e2e_update_s": elapsed,
                "finalize_and_resume_s": finalize,
                "index": len(records),
                "setup_residual_raw_s": residual_raw,
                "setup_residual_s": max(0.0, residual_raw),
                "start_timestamp_utc": _format_timestamp(
                    pending.get("start_timestamp")
                ),
                "update_weights_implementation_s": implementation,
            }
        )
        pending.clear()

    if pending:
        raise ValueError("log ended without a complete timer triplet")
    if len(records) != expected_updates:
        raise ValueError(
            f"expected {expected_updates} complete updates, observed {len(records)}"
        )
    return records


def extract_structured_trials(text: str) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for line in text.splitlines():
        if SCHEMA not in line:
            continue
        start = line.find("{")
        if start < 0:
            continue
        try:
            payload = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if payload.get("schema") == SCHEMA:
            trials.append(payload)
    return trials


def classify_samples(
    records: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    block: int,
    run_id: str,
) -> list[dict[str, Any]]:
    if len(records) != EXPECTED_UPDATE_CALLS:
        raise ValueError(
            f"expected {EXPECTED_UPDATE_CALLS} update records, observed {len(records)}"
        )
    result: list[dict[str, Any]] = []
    for index, source in enumerate(records):
        if index == 0:
            classification = "cold"
        elif index == PR3304_TAIL_INDEX:
            classification = "tail_excluded"
        else:
            classification = "steady"
        item = dict(source)
        item.update(
            {
                "arm": arm,
                "block": block,
                "classification": classification,
                "reference_step": index == PR3304_REFERENCE_INDEX,
                "run_id": run_id,
            }
        )
        result.append(item)
    return result


def classify_pr3304_reference(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return classify_samples(records, arm="reference", block=0, run_id="pr3304")


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def describe(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "n_valid": 0,
            "mad": None,
            "maximum": None,
            "mean": None,
            "median": None,
            "minimum": None,
            "p10": None,
            "p90": None,
            "p95": None,
            "spread_ok": False,
        }
    numbers = [float(value) for value in values]
    median = statistics.median(numbers)
    mad = statistics.median(abs(value - median) for value in numbers)
    p95 = _quantile(numbers, 0.95)
    return {
        "n_valid": len(numbers),
        "mad": mad,
        "maximum": max(numbers),
        "mean": statistics.fmean(numbers),
        "median": median,
        "minimum": min(numbers),
        "p10": _quantile(numbers, 0.10),
        "p90": _quantile(numbers, 0.90),
        "p95": p95,
        "spread_ok": (median > 0 and p95 / median <= 1.25 and mad / median <= 0.10),
    }


def summarize_job(
    samples: Sequence[Mapping[str, Any]],
    *,
    pod_start_timestamp: str | None,
) -> dict[str, Any]:
    first = next(item for item in samples if item["classification"] == "cold")
    steady = [item for item in samples if item["classification"] == "steady"]
    setup_before_first: float | None = None
    if pod_start_timestamp and first.get("start_timestamp_utc"):
        setup_before_first = (
            _parse_timestamp(str(first["start_timestamp_utc"]))
            - _parse_timestamp(pod_start_timestamp)
        ).total_seconds()
    return {
        "first_update": {
            "e2e_update_s": first.get("e2e_update_s"),
            "setup_residual_s": first.get("setup_residual_s"),
            "update_weights_implementation_s": first.get(
                "update_weights_implementation_s"
            ),
        },
        "setup_before_first_update_s": setup_before_first,
        "steady_state": {
            "e2e_update_s": describe([float(item["e2e_update_s"]) for item in steady]),
            "update_weights_implementation_s": describe(
                [float(item["update_weights_implementation_s"]) for item in steady]
            ),
        },
        "steady_count": len(steady),
        "tail_excluded_count": sum(
            item["classification"] == "tail_excluded" for item in samples
        ),
    }


def _bootstrap_interval(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [rng.choice(values) for _ in values]
        estimates.append(statistics.median(draw))
    return [_quantile(estimates, 0.025), _quantile(estimates, 0.975)]


def summarize_pairs(
    samples: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 3304,
) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    failure_counts = {arm: 0 for arm in ARMS}
    receipt_complete = {arm: True for arm in ARMS}
    for item in samples:
        if item.get("classification") != "steady":
            continue
        arm = str(item["arm"])
        if item.get("status", "ok") != "ok" or not item.get("timing_valid", True):
            failure_counts[arm] += 1
            continue
        if not item.get("receipt_complete", False):
            receipt_complete[arm] = False
        grouped[(int(item["block"]), arm)].append(item)

    block_results = []
    paired_ratios: list[float] = []
    paired_e2e_ratios: list[float] = []
    for block in sorted({key[0] for key in grouped}):
        broadcast = grouped.get((block, "broadcast"), [])
        modelexpress = grouped.get((block, "modelexpress"), [])
        if not broadcast or len(broadcast) != len(modelexpress):
            continue
        broadcast = sorted(broadcast, key=lambda item: int(item["index"]))
        modelexpress = sorted(modelexpress, key=lambda item: int(item["index"]))
        if [item["index"] for item in broadcast] != [
            item["index"] for item in modelexpress
        ]:
            continue
        observed_indices = [int(item["index"]) for item in broadcast]
        index_contract_complete = observed_indices == list(PR3304_STEADY_INDICES)
        version_contract_complete = all(
            left.get("weight_version") is not None
            and left.get("weight_version") == right.get("weight_version")
            for left, right in zip(broadcast, modelexpress, strict=True)
        )
        broadcast_values = [
            float(item["update_weights_implementation_s"]) for item in broadcast
        ]
        mx_values = [
            float(item["update_weights_implementation_s"]) for item in modelexpress
        ]
        broadcast_e2e = [float(item["e2e_update_s"]) for item in broadcast]
        mx_e2e = [float(item["e2e_update_s"]) for item in modelexpress]
        broadcast_median = statistics.median(broadcast_values)
        mx_median = statistics.median(mx_values)
        ratios = [
            left / right
            for left, right in zip(broadcast_values, mx_values, strict=True)
        ]
        e2e_ratios = [
            left / right for left, right in zip(broadcast_e2e, mx_e2e, strict=True)
        ]
        paired_ratios.extend(ratios)
        paired_e2e_ratios.extend(e2e_ratios)
        broadcast_reference = next(
            (
                item
                for item in broadcast
                if int(item["index"]) == PR3304_REFERENCE_INDEX
            ),
            None,
        )
        mx_reference = next(
            (
                item
                for item in modelexpress
                if int(item["index"]) == PR3304_REFERENCE_INDEX
            ),
            None,
        )
        block_results.append(
            {
                "block": block,
                "index_contract_complete": index_contract_complete,
                "version_contract_complete": version_contract_complete,
                "implementation": {
                    "broadcast_median_s": broadcast_median,
                    "broadcast_mean_s": statistics.fmean(broadcast_values),
                    "modelexpress_median_s": mx_median,
                    "modelexpress_mean_s": statistics.fmean(mx_values),
                    "speedup": broadcast_median / mx_median,
                    "improvement_pct": 100
                    * (broadcast_median - mx_median)
                    / broadcast_median,
                },
                "paired_trial_speedups": ratios,
                "step12": (
                    {
                        "broadcast_s": broadcast_reference[
                            "update_weights_implementation_s"
                        ],
                        "modelexpress_s": mx_reference[
                            "update_weights_implementation_s"
                        ],
                        "speedup": float(
                            broadcast_reference["update_weights_implementation_s"]
                        )
                        / float(mx_reference["update_weights_implementation_s"]),
                    }
                    if broadcast_reference is not None and mx_reference is not None
                    else None
                ),
            }
        )

    broadcast_all = [
        float(item["update_weights_implementation_s"])
        for item in samples
        if item.get("classification") == "steady"
        and item.get("arm") == "broadcast"
        and item.get("status", "ok") == "ok"
        and item.get("timing_valid", True)
    ]
    mx_all = [
        float(item["update_weights_implementation_s"])
        for item in samples
        if item.get("classification") == "steady"
        and item.get("arm") == "modelexpress"
        and item.get("status", "ok") == "ok"
        and item.get("timing_valid", True)
    ]
    broadcast_e2e = [
        float(item["e2e_update_s"])
        for item in samples
        if item.get("classification") == "steady"
        and item.get("arm") == "broadcast"
        and item.get("status", "ok") == "ok"
        and item.get("timing_valid", True)
    ]
    mx_e2e = [
        float(item["e2e_update_s"])
        for item in samples
        if item.get("classification") == "steady"
        and item.get("arm") == "modelexpress"
        and item.get("status", "ok") == "ok"
        and item.get("timing_valid", True)
    ]
    broadcast_stats = describe(broadcast_all)
    mx_stats = describe(mx_all)
    median_speedup = (
        broadcast_stats["median"] / mx_stats["median"]
        if broadcast_stats["median"] is not None and mx_stats["median"] not in (None, 0)
        else None
    )
    observed_blocks = {
        int(item["block"]) for item in samples if item.get("classification") == "steady"
    }
    exact_index_contract = (
        len(block_results) == len(observed_blocks)
        and all(item["index_contract_complete"] for item in block_results)
        and len(broadcast_all) == len(PR3304_STEADY_INDICES) * len(observed_blocks)
        and len(mx_all) == len(broadcast_all)
    )
    exact_version_contract = len(block_results) == len(observed_blocks) and all(
        item["version_contract_complete"] for item in block_results
    )
    mode_stats: dict[str, Any] = {}
    for arm, stats in (
        ("broadcast", broadcast_stats),
        ("modelexpress", mx_stats),
    ):
        valid = [
            item
            for item in samples
            if item.get("classification") == "steady"
            and item.get("arm") == arm
            and item.get("status", "ok") == "ok"
            and item.get("timing_valid", True)
        ]
        totals = [
            int(item["bytes_total"])
            for item in valid
            if item.get("bytes_total") is not None
        ]
        m2n = [
            int(item["bytes_m2n"])
            for item in valid
            if item.get("bytes_m2n") is not None
        ]
        residual = [
            int(item["bytes_residual"])
            for item in valid
            if item.get("bytes_residual") is not None
        ]
        coverage = [
            float(item["coverage_fraction"])
            for item in valid
            if item.get("coverage_fraction") is not None
        ]
        total_bytes = sum(totals) if len(totals) == len(valid) else None
        residual_bytes = sum(residual) if len(residual) == len(valid) else None
        mode_stats[arm] = stats | {
            "bytes_m2n_total": sum(m2n) if len(m2n) == len(valid) else None,
            "bytes_residual_total": residual_bytes,
            "bytes_total": total_bytes,
            "coverage_fraction": describe(coverage),
            "n_failed": failure_counts[arm],
            "receipt_complete": receipt_complete[arm] and bool(valid),
            "residual_byte_fraction": (
                residual_bytes / total_bytes
                if residual_bytes is not None and total_bytes
                else None
            ),
        }
    return {
        "blocks": block_results,
        "diagnostics": {
            "median_rollout_ready_e2e_speedup": (
                statistics.median(broadcast_e2e) / statistics.median(mx_e2e)
                if broadcast_e2e and mx_e2e
                else None
            ),
            "paired_e2e_speedup_median": (
                statistics.median(paired_e2e_ratios) if paired_e2e_ratios else None
            ),
            "rollout_ready_e2e": {
                "broadcast": describe(broadcast_e2e),
                "modelexpress": describe(mx_e2e),
            },
        },
        "headline": {
            "acceptance_ready": (
                len(broadcast_all) >= 10
                and len(broadcast_all) == len(mx_all)
                and len(block_results) > 0
                and not any(failure_counts.values())
                and all(receipt_complete.values())
                and exact_index_contract
                and exact_version_contract
            ),
            "exact_index_contract": exact_index_contract,
            "exact_version_contract": exact_version_contract,
            "median_implementation_speedup": median_speedup,
            "median_paired_trial_speedup": (
                statistics.median(paired_ratios) if paired_ratios else None
            ),
            "paired_blocks": len(block_results),
            "paired_speedup_bootstrap_95": _bootstrap_interval(
                paired_ratios,
                samples=bootstrap_samples,
                seed=seed,
            ),
        },
        "modes": mode_stats,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(dict(item), sort_keys=True, separators=(",", ":"))
        for item in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(
            json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n"
        )


def job_environment(
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    run_id: str,
) -> dict[str, str]:
    cluster = config["cluster"]
    experiment = config["experiment"]
    images = config["images"]
    topology = config.get("topology", {})
    workload = config.get("workload", {})
    return {
        "ACTOR_GPUS": str(topology.get("actor_gpus", 2)),
        "BENCH_NODE_NAME": str(cluster.get("node_name", "")),
        "GPU_PRODUCT": str(cluster.get("gpu_product", "")),
        "KUBE_CONTEXT": str(cluster["context"]),
        "MILES_BENCH_BLOCK": str(item["block"]),
        "MILES_BENCH_REPETITION": str(item.get("repetition", 0)),
        "MILES_BENCH_REQUIRE_WEIGHT_CHECKER": (
            "1" if experiment.get("require_symmetric_weight_checker", True) else "0"
        ),
        "MILES_BENCH_ROLLOUTS": str(experiment["num_rollouts"]),
        "MILES_BENCH_RUN_LABEL": run_id,
        "MILES_CHECKPOINT_MANIFEST_PATH": str(
            workload.get("checkpoint_manifest_path", "")
        ),
        "MILES_CHECKPOINT_REVISION": str(workload.get("checkpoint_revision", "")),
        "MILES_CHECKPOINT_SHA256": str(workload.get("checkpoint_sha256", "")),
        "MILES_DATASET_PATH": str(workload.get("dataset_path", "")),
        "MILES_DATASET_REVISION": str(workload.get("dataset_revision", "")),
        "MILES_DATASET_SHA256": str(workload.get("dataset_sha256", "")),
        "MILES_MODEL_PATH": str(workload.get("model_path", "")),
        "MILES_MODEL_TYPE": str(workload.get("miles_model_type", "")),
        "MILES_WEIGHT_TRANSFER_MODE": TRANSFER_MODES[str(item["arm"])],
        "MODEL_ID": str(workload.get("model_id", "")),
        "MX_MILES_VERIFY_TENSOR_EQUALITY": (
            "1" if experiment.get("verify_tensor_equality", False) else "0"
        ),
        "MX_NCCL_REFIT_NUM_STREAMS": str(topology.get("m2n_pp_concurrency", 2)),
        "MX_RESHARD_REQUIRE_FULL_COVERAGE": "1",
        "MX_SERVER_IMAGE": str(images["server"]),
        "NCCL_DEBUG": str(experiment.get("nccl_debug", "WARN")),
        "NAMESPACE": str(cluster["namespace"]),
        "PIPELINE_PARALLEL_SIZE": str(topology.get("pipeline_parallel_size", 2)),
        "ROLLOUT_GPUS": str(topology.get("rollout_gpus", 2)),
        "ROLLOUT_GPUS_PER_ENGINE": str(topology.get("rollout_gpus_per_engine", 1)),
        "RUN_ID": run_id,
        "MILES_BENCH_RAW_RECEIPTS": (
            f"/root/shared_data/miles-benchmark/{run_id}/"
            f"block-{item['block']}-{item.get('repetition', 0)}-"
            f"{TRANSFER_MODES[str(item['arm'])]}.raw.jsonl"
        ),
        "MILES_BENCH_E2E_RECEIPTS": (
            f"/root/shared_data/miles-benchmark/{run_id}/"
            f"block-{item['block']}-{item.get('repetition', 0)}-"
            f"{TRANSFER_MODES[str(item['arm'])]}.e2e.jsonl"
        ),
        "RUNTIME_PULL_SECRET": str(
            cluster.get("runtime_pull_secret", "nvcr-imagepullsecret")
        ),
        "RUNTIME_IMAGE": str(images["runtime"]),
        "TOTAL_GPUS": str(topology.get("total_gpus", 4)),
    }


def job_name(item: Mapping[str, Any], *, run_id: str) -> str:
    transfer_mode = TRANSFER_MODES[str(item["arm"])]
    return (
        f"miles-m2n-{transfer_mode}-{run_id}-b{item['block']}"
        f"-r{item.get('repetition', 0)}"
    )


def server_resource_name(item: Mapping[str, Any], *, run_id: str) -> str:
    return f"mx-{run_id}-b{item['block']}-r{item.get('repetition', 0)}"


def runtime_configmap_name(item: Mapping[str, Any], *, run_id: str) -> str:
    return f"miles-bench-{run_id}-b{item['block']}-r{item.get('repetition', 0)}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_receipt(repo: Path) -> dict[str, Any]:
    resolved = repo.expanduser().resolve()
    return {
        "branch": _git(resolved, "branch", "--show-current") or None,
        "commit": _git(resolved, "rev-parse", "HEAD"),
        "dirty": bool(_git(resolved, "status", "--porcelain")),
        "path": str(resolved),
        "remote": _git(resolved, "remote", "get-url", "origin"),
    }


def normalize_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    config = json.loads(json.dumps(payload))
    output_root = validate_persistent_output(Path(config["output_root"]))
    config["output_root"] = str(output_root)
    run_id = str(config["run_id"])
    if not _RUN_ID.fullmatch(run_id) or len(run_id) > 24:
        raise ValueError("run_id must be a lowercase Kubernetes component <= 24 chars")
    config["images"]["runtime"] = validate_image(
        str(config["images"]["runtime"]), field="images.runtime"
    )
    config["images"]["server"] = validate_image(
        str(config["images"]["server"]), field="images.server"
    )
    config["storage"] = validate_storage_config(config["storage"])
    node_name = str(config["cluster"].get("node_name", ""))
    if not node_name:
        raise ValueError(
            "cluster.node_name is required to pin paired RWO benchmark claims"
        )
    if len(node_name) > 253 or _KUBE_NAME.fullmatch(node_name) is None:
        raise ValueError("cluster.node_name must be a valid Kubernetes node name")
    experiment = config["experiment"]
    if int(experiment["num_rollouts"]) != DEFAULT_NUM_ROLLOUTS:
        raise ValueError("PR3304 metrics contract requires exactly 20 rollouts")
    if int(experiment["blocks"]) <= 0:
        raise ValueError("experiment.blocks must be positive")
    experiment.setdefault("require_symmetric_weight_checker", True)
    if not isinstance(experiment["require_symmetric_weight_checker"], bool):
        raise ValueError(
            "experiment.require_symmetric_weight_checker must be a boolean"
        )
    if not experiment["require_symmetric_weight_checker"]:
        raise ValueError(
            "accepted paired runs require the symmetric MILES weight checker"
        )
    if not isinstance(experiment.get("verify_tensor_equality", False), bool):
        raise ValueError("experiment.verify_tensor_equality must be a boolean")
    if experiment.get("nccl_debug", "WARN") not in {"WARN", "INFO"}:
        raise ValueError("experiment.nccl_debug must be WARN or INFO")
    launcher = Path(config["launcher"]).expanduser().resolve()
    if not launcher.is_file():
        raise ValueError(f"launcher does not exist: {launcher}")
    config["launcher"] = str(launcher)
    for field in ("checkpoint_sha256", "dataset_sha256"):
        value = str(config["workload"][field])
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"workload.{field} must be a sha256 digest")
        if len(set(value)) == 1:
            raise ValueError(f"workload.{field} cannot be a placeholder digest")
    dataset_path = Path(str(config["workload"]["dataset_path"]))
    if not dataset_path.is_absolute():
        raise ValueError("workload.dataset_path must be an absolute persistent path")
    for root in _FORBIDDEN_OUTPUT_ROOTS:
        if dataset_path == root or root in dataset_path.parents:
            raise ValueError("workload.dataset_path cannot use ephemeral storage")
    if not str(config["workload"].get("dataset_revision", "")):
        raise ValueError("workload.dataset_revision must be immutable and non-empty")
    model_path = Path(str(config["workload"].get("model_path", "")))
    if not model_path.is_absolute():
        raise ValueError("workload.model_path must be an absolute persistent path")
    for root in _FORBIDDEN_OUTPUT_ROOTS:
        if model_path == root or root in model_path.parents:
            raise ValueError("workload.model_path cannot use ephemeral storage")
    checkpoint_manifest_path = Path(
        str(config["workload"].get("checkpoint_manifest_path", ""))
    )
    if not checkpoint_manifest_path.is_absolute():
        raise ValueError(
            "workload.checkpoint_manifest_path must be an absolute persistent path"
        )
    if model_path not in checkpoint_manifest_path.parents:
        raise ValueError(
            "workload.checkpoint_manifest_path must be inside workload.model_path"
        )
    for root in _FORBIDDEN_OUTPUT_ROOTS:
        if checkpoint_manifest_path == root or root in checkpoint_manifest_path.parents:
            raise ValueError(
                "workload.checkpoint_manifest_path cannot use ephemeral storage"
            )
    checkpoint_revision = str(config["workload"].get("checkpoint_revision", ""))
    if re.fullmatch(r"[0-9a-f]{40}", checkpoint_revision) is None:
        raise ValueError(
            "workload.checkpoint_revision must be an immutable 40-hex revision"
        )
    for field in ("model_id", "miles_model_type"):
        if not str(config["workload"].get(field, "")):
            raise ValueError(f"workload.{field} must be non-empty")
    topology = config["topology"]
    actor_gpus = int(topology["actor_gpus"])
    rollout_gpus = int(topology["rollout_gpus"])
    total_gpus = int(topology["total_gpus"])
    pipeline_parallel = int(topology["pipeline_parallel_size"])
    rollout_per_engine = int(topology["rollout_gpus_per_engine"])
    if actor_gpus + rollout_gpus != total_gpus:
        raise ValueError("topology actor_gpus + rollout_gpus must equal total_gpus")
    if actor_gpus != pipeline_parallel:
        raise ValueError(
            "topology actor_gpus must equal pipeline_parallel_size for this launcher"
        )
    if rollout_gpus % rollout_per_engine:
        raise ValueError(
            "topology rollout_gpus must be divisible by rollout_gpus_per_engine"
        )
    benchmark_kind = str(config["benchmark_kind"])
    if benchmark_kind not in {
        "pr3304_reference_reproduction",
        "topology_preserving_validation",
    }:
        raise ValueError("benchmark_kind is not recognized")
    if benchmark_kind == "pr3304_reference_reproduction":
        reference_fields = {
            "actor_gpus": 32,
            "m2n_pp_concurrency": 2,
            "node_count": 16,
            "pipeline_parallel_size": 8,
            "rollout_gpus": 32,
            "rollout_gpus_per_engine": 8,
            "total_gpus": 64,
        }
        differences = {
            field: (topology.get(field), expected)
            for field, expected in reference_fields.items()
            if int(topology.get(field, -1)) != expected
        }
        if differences or "GB300" not in str(config["cluster"]["gpu_product"]):
            raise ValueError(
                "PR3304 reproduction requires the exact 64-GB300 topology; "
                f"differences={differences}"
            )
    return config


def source_receipts(config: Mapping[str, Any]) -> dict[str, Any]:
    receipts = {
        name: git_receipt(Path(path))
        for name, path in sorted(config["source_repositories"].items())
    }
    dirty = [name for name, receipt in receipts.items() if receipt["dirty"]]
    if dirty:
        raise ValueError(
            "benchmark source provenance must be clean; dirty repositories: "
            + ", ".join(dirty)
        )
    return receipts


def _run_command(
    args: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: Path,
    output_dir: Path,
    name: str,
    timeout_s: float,
    check: bool = True,
    input_text: str | None = None,
) -> dict[str, Any]:
    started = dt.datetime.now(dt.timezone.utc)
    start_monotonic = time.monotonic()
    stdout_path = output_dir / f"{name}.stdout.log"
    stderr_path = output_dir / f"{name}.stderr.log"
    timed_out = False
    returncode: int | None
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            completed = subprocess.run(
                list(args),
                cwd=cwd,
                env=os.environ | dict(environment),
                input=input_text,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = None
    receipt = {
        "args": list(args),
        "duration_s": time.monotonic() - start_monotonic,
        "ended_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "returncode": returncode,
        "started_at_utc": started.isoformat(),
        "stderr": str(stderr_path),
        "stdout": str(stdout_path),
        "timed_out": timed_out,
    }
    append_jsonl(output_dir / "commands.jsonl", receipt)
    if check and timed_out:
        raise RuntimeError(f"{name} timed out after {timeout_s}s")
    if check and returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {returncode}")
    return receipt


def _benchmark_job(
    documents: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    job = next(
        (item for item in documents if item.get("kind") == "Job"),
        None,
    )
    if job is None:
        raise ValueError("render lacks the benchmark Job")
    return job


def configure_persistent_storage(
    documents: Sequence[dict[str, Any]],
    storage: Mapping[str, Any],
) -> None:
    claims = validate_storage_config(storage)
    job = _benchmark_job(documents)
    pod_spec = job["spec"]["template"]["spec"]
    volumes = {str(volume["name"]): volume for volume in pod_spec.get("volumes", [])}
    required = {
        "model-cache": claims["model_claim"],
        "run-output": claims["workspace_claim"],
        "benchmark-workspace": claims["workspace_claim"],
    }
    for volume_name, claim_name in required.items():
        if volume_name not in volumes and volume_name == "benchmark-workspace":
            volume = {"name": volume_name}
            pod_spec.setdefault("volumes", []).append(volume)
            volumes[volume_name] = volume
        if volume_name not in volumes:
            raise ValueError(f"render lacks required volume {volume_name}")
        volumes[volume_name].clear()
        volumes[volume_name].update(
            {
                "name": volume_name,
                "persistentVolumeClaim": {"claimName": claim_name},
            }
        )
    container = pod_spec["containers"][0]
    mounts = {
        str(mount["name"]): mount for mount in container.setdefault("volumeMounts", [])
    }
    if "benchmark-workspace" not in mounts:
        container["volumeMounts"].append(
            {
                "name": "benchmark-workspace",
                "mountPath": "/workspace",
            }
        )


def configure_scoped_server_resources(
    documents: Sequence[dict[str, Any]],
    *,
    server_name: str,
    run_id: str,
    block: str,
    repetition: str,
) -> None:
    ownership_labels = {
        "modelexpress.nvidia.com/benchmark-run": run_id,
        "modelexpress.nvidia.com/benchmark-block": block,
        "modelexpress.nvidia.com/benchmark-repetition": repetition,
        "modelexpress.nvidia.com/server-id": server_name,
    }
    for document in documents:
        kind = document.get("kind")
        if kind not in {"Deployment", "Service"}:
            continue
        if document.get("metadata", {}).get("name") != "mx-server":
            continue
        document["metadata"]["name"] = server_name
        document["metadata"].setdefault("labels", {}).update(ownership_labels)
        if kind == "Service":
            document["spec"].setdefault("selector", {}).update(
                {"modelexpress.nvidia.com/server-id": server_name}
            )
            continue
        selector = document["spec"]["selector"].setdefault("matchLabels", {})
        selector["modelexpress.nvidia.com/server-id"] = server_name
        template_labels = document["spec"]["template"]["metadata"].setdefault(
            "labels", {}
        )
        template_labels.update(ownership_labels)
    job = _benchmark_job(documents)
    container = job["spec"]["template"]["spec"]["containers"][0]
    for entry in container.get("env", []):
        if entry.get("name") == "MX_SERVER_ADDRESS":
            entry["value"] = f"{server_name}:8000"


def _ownership_labels(
    *,
    run_id: str,
    block: str,
    repetition: str,
) -> dict[str, str]:
    return {
        "modelexpress.nvidia.com/benchmark-run": run_id,
        "modelexpress.nvidia.com/benchmark-block": block,
        "modelexpress.nvidia.com/benchmark-repetition": repetition,
    }


def configure_runtime_harness(
    documents: list[dict[str, Any]],
    *,
    environment: Mapping[str, str],
) -> None:
    job = _benchmark_job(documents)
    configmap_name = runtime_configmap_name(
        {
            "block": environment["MILES_BENCH_BLOCK"],
            "repetition": environment["MILES_BENCH_REPETITION"],
        },
        run_id=environment["RUN_ID"],
    )
    labels = _ownership_labels(
        run_id=environment["MILES_BENCH_RUN_LABEL"],
        block=environment["MILES_BENCH_BLOCK"],
        repetition=environment["MILES_BENCH_REPETITION"],
    )
    job_labels = labels | {
        "modelexpress.nvidia.com/benchmark-mode": environment[
            "MILES_WEIGHT_TRANSFER_MODE"
        ],
    }
    job.setdefault("metadata", {}).setdefault("labels", {}).update(job_labels)
    job["spec"]["template"].setdefault("metadata", {}).setdefault(
        "labels",
        {},
    ).update(job_labels)
    source_dir = Path(__file__).resolve().parent
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": configmap_name,
            "namespace": environment["NAMESPACE"],
            "labels": labels,
        },
        "data": {
            "runtime_harness.py": (source_dir / "runtime_harness.py").read_text(),
            "sitecustomize.py": (source_dir / "sitecustomize.py").read_text(),
        },
    }
    job_index = documents.index(job)
    documents.insert(job_index, configmap)
    pod_spec = job["spec"]["template"]["spec"]
    pod_spec.setdefault("volumes", []).append(
        {
            "name": "benchmark-runtime",
            "configMap": {"name": configmap_name},
        }
    )
    container = pod_spec["containers"][0]
    container["command"] = [
        "python3",
        "/opt/miles-benchmark/runtime_harness.py",
    ]
    container.setdefault("volumeMounts", []).append(
        {
            "name": "benchmark-runtime",
            "mountPath": "/opt/miles-benchmark",
            "readOnly": True,
        }
    )
    environment_entries = container.setdefault("env", [])
    by_name = {item["name"]: item for item in environment_entries}
    for name in (
        "MILES_BENCH_ENABLE_RUNTIME_RECEIPTS",
        "MILES_BENCH_E2E_RECEIPTS",
        "MILES_BENCH_RAW_RECEIPTS",
        "MILES_BENCH_REQUIRE_WEIGHT_CHECKER",
        "MILES_CHECKPOINT_MANIFEST_PATH",
        "MILES_CHECKPOINT_REVISION",
        "MILES_CHECKPOINT_SHA256",
        "MILES_DATASET_PATH",
        "MILES_DATASET_REVISION",
        "MILES_DATASET_SHA256",
        "MILES_MODEL_PATH",
        "MILES_MODEL_TYPE",
        "MODEL_ID",
    ):
        by_name[name] = {
            "name": name,
            "value": (
                "1"
                if name == "MILES_BENCH_ENABLE_RUNTIME_RECEIPTS"
                else environment[name]
            ),
        }
    existing_pythonpath = str(by_name.get("PYTHONPATH", {}).get("value", ""))
    by_name["PYTHONPATH"] = {
        "name": "PYTHONPATH",
        "value": (
            "/opt/miles-benchmark"
            + (f":{existing_pythonpath}" if existing_pythonpath else "")
        ),
    }
    container["env"] = list(by_name.values())


def launcher_render_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    rendered = dict(environment)
    rendered["MODEL_ID"] = LAUNCHER_SCAFFOLD_MODEL_ID
    rendered["MILES_MODEL_TYPE"] = LAUNCHER_SCAFFOLD_MODEL_TYPE
    return rendered


def _render_manifest(
    launcher: Path,
    *,
    environment: Mapping[str, str],
    output_dir: Path,
    storage: Mapping[str, Any],
    timeout_s: float,
) -> tuple[Path, list[dict[str, Any]]]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to render benchmark manifests")
    result = subprocess.run(
        [str(launcher), "render"],
        env=os.environ | launcher_render_environment(environment),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    (output_dir / "render.stderr.log").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"launcher render failed: {result.stderr.strip()}")
    documents = [item for item in yaml.safe_load_all(result.stdout) if item is not None]
    job = next(item for item in documents if item.get("kind") == "Job")
    container = job["spec"]["template"]["spec"]["containers"][0]
    environment_entries = container.setdefault("env", [])
    by_name = {item["name"]: item for item in environment_entries}
    by_name["MX_RESHARD_REQUIRE_FULL_COVERAGE"] = {
        "name": "MX_RESHARD_REQUIRE_FULL_COVERAGE",
        "value": "1",
    }
    container["env"] = list(by_name.values())
    configure_persistent_storage(documents, storage)
    configure_runtime_harness(documents, environment=environment)
    if environment["MILES_WEIGHT_TRANSFER_MODE"] == "external":
        configure_scoped_server_resources(
            documents,
            server_name=server_resource_name(
                {
                    "block": environment["MILES_BENCH_BLOCK"],
                    "repetition": environment["MILES_BENCH_REPETITION"],
                },
                run_id=environment["RUN_ID"],
            ),
            run_id=environment["MILES_BENCH_RUN_LABEL"],
            block=environment["MILES_BENCH_BLOCK"],
            repetition=environment["MILES_BENCH_REPETITION"],
        )
    manifest_path = output_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump_all(documents, sort_keys=False))
    return manifest_path, documents


def validate_rendered_documents(
    documents: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    runtime_image: str,
    server_image: str,
    storage: Mapping[str, Any],
    server_name: str | None = None,
) -> None:
    resources = {
        (str(item.get("kind")), str(item.get("metadata", {}).get("name")))
        for item in documents
    }
    mx_resources = {
        resource
        for resource in resources
        if resource[0] in {"Deployment", "Service"}
        and any(
            item.get("kind") == resource[0]
            and item.get("metadata", {}).get("name") == resource[1]
            and item.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/name")
            == "modelexpress-server"
            for item in documents
        )
    }
    if arm == "broadcast" and mx_resources:
        raise ValueError("broadcast render must omit ModelExpress server resources")
    expected_server_resources = (
        {
            ("Deployment", str(server_name)),
            ("Service", str(server_name)),
        }
        if server_name is not None
        else set()
    )
    if arm == "modelexpress" and mx_resources != expected_server_resources:
        raise ValueError(
            "modelexpress render must include only its scoped server resources"
        )

    claims = validate_storage_config(storage)
    job = _benchmark_job(documents)
    configmaps = [
        item
        for item in documents
        if item.get("kind") == "ConfigMap"
        and item.get("metadata", {}).get("name", "").startswith("miles-bench-")
    ]
    if len(configmaps) != 1:
        raise ValueError("render must contain exactly one benchmark runtime ConfigMap")
    containers = job["spec"]["template"]["spec"].get("containers", [])
    if not containers or containers[0].get("image") != runtime_image:
        raise ValueError("benchmark Job does not use the configured runtime digest")
    for entry in containers[0].get("env", []):
        if "value" in entry and not isinstance(entry["value"], str):
            raise ValueError(
                f"rendered environment value for {entry.get('name')} is not a string"
            )
    rendered_environment = {
        str(entry["name"]): entry.get("value")
        for entry in containers[0].get("env", [])
        if "value" in entry
    }
    for name in (
        "MILES_BENCH_REQUIRE_WEIGHT_CHECKER",
        "MILES_CHECKPOINT_MANIFEST_PATH",
        "MILES_CHECKPOINT_REVISION",
        "MILES_CHECKPOINT_SHA256",
        "MILES_DATASET_PATH",
        "MILES_DATASET_REVISION",
        "MILES_DATASET_SHA256",
        "MILES_MODEL_PATH",
        "MILES_MODEL_TYPE",
        "MODEL_ID",
    ):
        if not rendered_environment.get(name):
            raise ValueError(f"benchmark Job lacks runtime workload field {name}")
    if containers[0].get("command") != [
        "python3",
        "/opt/miles-benchmark/runtime_harness.py",
    ]:
        raise ValueError("benchmark Job does not execute the runtime receipt driver")
    runtime_source = configmaps[0].get("data", {}).get("runtime_harness.py", "")
    if "gsm8k" in runtime_source.lower():
        raise ValueError("benchmark runtime retains a GSM8K workload path")
    source_dir = Path(__file__).resolve().parent
    expected_runtime_source = (source_dir / "runtime_harness.py").read_text()
    expected_sitecustomize_source = (source_dir / "sitecustomize.py").read_text()
    if (
        runtime_source != expected_runtime_source
        or configmaps[0].get("data", {}).get("sitecustomize.py")
        != expected_sitecustomize_source
    ):
        raise ValueError(
            "benchmark runtime ConfigMap does not match the current frozen source"
        )
    pod_spec = job["spec"]["template"]["spec"]
    volumes = {str(volume["name"]): volume for volume in pod_spec.get("volumes", [])}
    mounts = {
        str(mount["name"]): mount for mount in containers[0].get("volumeMounts", [])
    }
    required = {
        "model-cache": (claims["model_claim"], "/models"),
        "run-output": (claims["workspace_claim"], "/root/shared_data"),
        "benchmark-workspace": (claims["workspace_claim"], "/workspace"),
    }
    for volume_name, (claim_name, mount_path) in required.items():
        expected = {"claimName": claim_name}
        actual = volumes.get(volume_name, {}).get("persistentVolumeClaim")
        if actual != expected:
            raise ValueError(
                f"{volume_name} must use persistentVolumeClaim {claim_name!r}"
            )
        if mounts.get(volume_name, {}).get("mountPath") != mount_path:
            raise ValueError(
                f"{volume_name} must be mounted at persistent path {mount_path}"
            )
    for volume_name, volume in volumes.items():
        if "ephemeral" in volume or "hostPath" in volume:
            raise ValueError(f"volume {volume_name} uses ephemeral storage")
        if "emptyDir" in volume and volume_name != "shared-memory":
            raise ValueError(
                f"volume {volume_name} uses emptyDir instead of persistent storage"
            )
    runtime_volume = volumes.get("benchmark-runtime", {}).get("configMap", {})
    if runtime_volume.get("name") != configmaps[0]["metadata"]["name"]:
        raise ValueError("benchmark runtime ConfigMap is not mounted")
    runtime_mount = mounts.get("benchmark-runtime", {})
    if (
        runtime_mount.get("mountPath") != "/opt/miles-benchmark"
        or runtime_mount.get("readOnly") is not True
    ):
        raise ValueError("benchmark runtime ConfigMap must be mounted read-only")
    shared_memory = volumes.get("shared-memory", {}).get("emptyDir")
    if (
        not isinstance(shared_memory, Mapping)
        or shared_memory.get("medium") != "Memory"
        or mounts.get("shared-memory", {}).get("mountPath") != "/dev/shm"
    ):
        raise ValueError(
            "shared-memory must be the sole RAM-backed emptyDir mounted at /dev/shm"
        )

    if arm == "modelexpress":
        deployment = next(
            item
            for item in documents
            if item.get("kind") == "Deployment"
            and item.get("metadata", {}).get("name") == server_name
        )
        server_containers = deployment["spec"]["template"]["spec"].get("containers", [])
        if not server_containers or server_containers[0].get("image") != server_image:
            raise ValueError(
                "mx-server Deployment does not use the configured server digest"
            )
        address = next(
            (
                entry.get("value")
                for entry in containers[0].get("env", [])
                if entry.get("name") == "MX_SERVER_ADDRESS"
            ),
            None,
        )
        if address != f"{server_name}:8000":
            raise ValueError("benchmark Job does not target its scoped server")


def validate_pvc_receipt(
    pvc: Mapping[str, Any],
    *,
    claim_name: str,
) -> None:
    if pvc.get("metadata", {}).get("name") != claim_name:
        raise ValueError(f"PVC receipt does not match requested claim {claim_name}")
    if pvc.get("metadata", {}).get("deletionTimestamp"):
        raise ValueError(f"PVC {claim_name} is being deleted")
    if pvc.get("status", {}).get("phase") != "Bound":
        raise ValueError(f"PVC {claim_name} must already be Bound")
    access_modes = set(pvc.get("spec", {}).get("accessModes", []))
    if "ReadWriteOnce" not in access_modes:
        raise ValueError(f"PVC {claim_name} must support ReadWriteOnce")
    if pvc.get("spec", {}).get("volumeMode", "Filesystem") != "Filesystem":
        raise ValueError(f"PVC {claim_name} must use Filesystem volume mode")
    if not pvc.get("spec", {}).get("volumeName"):
        raise ValueError(f"PVC {claim_name} lacks a bound persistent volume")


def _kubectl_json(
    config: Mapping[str, Any],
    *,
    args: Sequence[str],
    output_path: Path,
    output_dir: Path,
    command_name: str,
    environment: Mapping[str, str],
    all_namespaces: bool = False,
) -> dict[str, Any]:
    command = [
        str(config.get("kubectl", "kubectl")),
        "--context",
        str(config["cluster"]["context"]),
    ]
    if not all_namespaces:
        command.extend(("-n", str(config["cluster"]["namespace"])))
    command.extend(args)
    if all_namespaces:
        command.append("--all-namespaces")
    command.extend(("-o", "json"))
    receipt = _run_command(
        command,
        environment=environment,
        cwd=Path(config["launcher"]).parent,
        output_dir=output_dir,
        name=command_name,
        timeout_s=float(config["experiment"]["command_timeout_s"]),
    )
    output_path.write_text(Path(receipt["stdout"]).read_text())
    return json.loads(output_path.read_text())


def _node_selector_expression_matches(
    labels: Mapping[str, str],
    expression: Mapping[str, Any],
) -> bool:
    key = str(expression["key"])
    operator = str(expression["operator"])
    values = {str(value) for value in expression.get("values", [])}
    if operator == "In":
        return labels.get(key) in values
    if operator == "NotIn":
        return key in labels and labels[key] not in values
    if operator == "Exists":
        return key in labels
    if operator == "DoesNotExist":
        return key not in labels
    raise ValueError(f"unsupported persistent-volume node selector operator {operator}")


def validate_pv_node_compatibility(
    pv: Mapping[str, Any],
    node: Mapping[str, Any],
) -> None:
    required = (
        pv.get("spec", {})
        .get("nodeAffinity", {})
        .get("required", {})
        .get("nodeSelectorTerms", [])
    )
    if not required:
        return
    labels = {
        str(key): str(value)
        for key, value in node.get("metadata", {}).get("labels", {}).items()
    }
    if not any(
        all(
            _node_selector_expression_matches(labels, expression)
            for expression in term.get("matchExpressions", [])
        )
        for term in required
    ):
        raise ValueError(
            f"persistent volume {pv.get('metadata', {}).get('name')} is not "
            f"attachable to node {node.get('metadata', {}).get('name')}"
        )


def _gpu_quantity(value: Any, *, field: str) -> int:
    if value in (None, ""):
        return 0
    text = str(value)
    if re.fullmatch(r"[0-9]+", text) is None:
        raise ValueError(f"{field} must be a whole non-negative GPU quantity")
    return int(text)


def _container_gpu_request(container: Mapping[str, Any]) -> int:
    resources = container.get("resources", {})
    requests = resources.get("requests", {})
    limits = resources.get("limits", {})
    value = requests.get("nvidia.com/gpu", limits.get("nvidia.com/gpu", 0))
    return _gpu_quantity(
        value,
        field=f"container {container.get('name', '<unknown>')} GPU request",
    )


def pod_effective_gpu_request(pod: Mapping[str, Any]) -> dict[str, Any]:
    spec = pod.get("spec", {})
    application = [
        {
            "name": str(container.get("name", "")),
            "requested": _container_gpu_request(container),
            "type": "container",
        }
        for container in spec.get("containers", [])
    ]
    initializers = [
        {
            "name": str(container.get("name", "")),
            "requested": _container_gpu_request(container),
            "type": "initContainer",
        }
        for container in spec.get("initContainers", [])
    ]
    application_total = sum(item["requested"] for item in application)
    initializer_max = max(
        (item["requested"] for item in initializers),
        default=0,
    )
    return {
        "containers": application + initializers,
        "effective_request": max(application_total, initializer_max),
    }


def validate_node_gpu_headroom(
    node: Mapping[str, Any],
    pods: Mapping[str, Any],
    *,
    node_name: str,
    required_gpus: int,
) -> dict[str, Any]:
    if required_gpus <= 0:
        raise ValueError("benchmark GPU request must be positive")
    allocatable = _gpu_quantity(
        node.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu", 0),
        field=f"node {node_name} allocatable GPUs",
    )
    consumers = []
    competing = []
    consumed = 0
    for pod in pods.get("items", []):
        phase = pod.get("status", {}).get("phase")
        if phase not in {"Pending", "Running"}:
            continue
        pod_name = str(pod.get("metadata", {}).get("name", ""))
        pod_node = str(pod.get("spec", {}).get("nodeName", ""))
        request = pod_effective_gpu_request(pod)
        item = {
            "name": pod_name,
            "node": pod_node or None,
            "phase": phase,
            **request,
        }
        if pod_node == node_name:
            consumers.append(item)
            consumed += int(request["effective_request"])
        elif (
            not pod_node
            and phase == "Pending"
            and pod.get("spec", {})
            .get("nodeSelector", {})
            .get("kubernetes.io/hostname")
            == node_name
        ):
            competing.append(item)
    available = allocatable - consumed
    receipt = {
        "allocatable": allocatable,
        "available": available,
        "benchmark_required": required_gpus,
        "competing_pinned_pending": competing,
        "consumers": consumers,
        "existing_requested": consumed,
        "node": node_name,
        "schedulable": available >= required_gpus,
    }
    if not receipt["schedulable"]:
        raise ValueError(
            f"paired node {node_name} has {available} free GPUs after "
            f"{consumed}/{allocatable} are requested; benchmark requires "
            f"{required_gpus}"
        )
    return receipt


def preflight_persistent_storage(
    config: Mapping[str, Any],
    *,
    run_dir: Path,
) -> None:
    output_dir = run_dir / "preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    environment: dict[str, str] = {}
    node_name = str(config["cluster"]["node_name"])
    node = _kubectl_json(
        config,
        args=["get", f"node/{node_name}"],
        output_path=output_dir / "node.json",
        output_dir=output_dir,
        command_name="node-json",
        environment=environment,
    )
    pods = _kubectl_json(
        config,
        args=["get", "pods"],
        output_path=output_dir / "pods.json",
        output_dir=output_dir,
        command_name="pods-json",
        environment=environment,
        all_namespaces=True,
    )
    gpu_receipt_path = output_dir / "gpu-headroom.json"
    try:
        gpu_receipt = validate_node_gpu_headroom(
            node,
            pods,
            node_name=node_name,
            required_gpus=int(config["topology"]["total_gpus"]),
        )
    except ValueError:
        allocatable = _gpu_quantity(
            node.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu", 0),
            field=f"node {node_name} allocatable GPUs",
        )
        consumed = sum(
            pod_effective_gpu_request(pod)["effective_request"]
            for pod in pods.get("items", [])
            if pod.get("status", {}).get("phase") in {"Pending", "Running"}
            and pod.get("spec", {}).get("nodeName") == node_name
        )
        write_json(
            gpu_receipt_path,
            {
                "allocatable": allocatable,
                "available": allocatable - consumed,
                "benchmark_required": int(config["topology"]["total_gpus"]),
                "existing_requested": consumed,
                "node": node_name,
                "schedulable": False,
            },
        )
        raise
    write_json(gpu_receipt_path, gpu_receipt)
    claims_by_role = validate_storage_config(config["storage"])
    receipts = {}
    for index, claim_name in enumerate(sorted(set(claims_by_role.values()))):
        pvc = _kubectl_json(
            config,
            args=["get", f"pvc/{claim_name}"],
            output_path=output_dir / f"pvc-{index}.json",
            output_dir=output_dir,
            command_name=f"pvc-{index}-json",
            environment=environment,
        )
        validate_pvc_receipt(pvc, claim_name=claim_name)
        volume_name = str(pvc["spec"]["volumeName"])
        pv = _kubectl_json(
            config,
            args=["get", f"pv/{volume_name}"],
            output_path=output_dir / f"pv-{index}.json",
            output_dir=output_dir,
            command_name=f"pv-{index}-json",
            environment=environment,
        )
        validate_pv_node_compatibility(pv, node)
        consumers = []
        for pod in pods.get("items", []):
            if pod.get("status", {}).get("phase") not in {"Pending", "Running"}:
                continue
            if (
                pod.get("metadata", {}).get("namespace")
                != config["cluster"]["namespace"]
            ):
                continue
            pod_claims = {
                volume.get("persistentVolumeClaim", {}).get("claimName")
                for volume in pod.get("spec", {}).get("volumes", [])
            }
            if claim_name not in pod_claims:
                continue
            consumer_node = pod.get("spec", {}).get("nodeName")
            if consumer_node and consumer_node != node_name:
                raise ValueError(
                    f"RWO PVC {claim_name} is active on node {consumer_node}, "
                    f"not paired node {node_name}"
                )
            consumers.append(
                {
                    "name": pod.get("metadata", {}).get("name"),
                    "node": consumer_node,
                    "phase": pod.get("status", {}).get("phase"),
                }
            )
        receipts[claim_name] = {
            "consumers": consumers,
            "persistent_volume": volume_name,
            "roles": sorted(
                role
                for role, configured_claim in claims_by_role.items()
                if configured_claim == claim_name
            ),
        }
    write_json(
        output_dir / "storage.json",
        {
            "claims": receipts,
            "node": node_name,
            "persistent_only": True,
            "gpu_headroom": gpu_receipt,
        },
    )


def _pod_start_timestamp(pods: Mapping[str, Any]) -> str | None:
    starts = [
        item.get("status", {}).get("startTime")
        for item in pods.get("items", [])
        if item.get("status", {}).get("startTime")
    ]
    return min(starts) if starts else None


def _capture_runtime_receipts(
    config: Mapping[str, Any],
    *,
    base: Sequence[str],
    environment: Mapping[str, str],
    job_dir: Path,
    job_name_value: str,
    launcher: Path,
    pods_json: Mapping[str, Any],
    timeout_s: float,
) -> None:
    _kubectl_json(
        config,
        args=[
            "get",
            "events",
            "--field-selector",
            f"involvedObject.name={job_name_value}",
        ],
        output_path=job_dir / "events.json",
        output_dir=job_dir,
        command_name="events-json",
        environment=environment,
    )
    pvc_names = sorted(
        {
            volume["persistentVolumeClaim"]["claimName"]
            for pod in pods_json.get("items", [])
            for volume in pod.get("spec", {}).get("volumes", [])
            if "persistentVolumeClaim" in volume
        }
    )
    pvc_receipts = {}
    for index, pvc_name in enumerate(pvc_names):
        pvc_receipts[pvc_name] = _kubectl_json(
            config,
            args=["get", f"pvc/{pvc_name}"],
            output_path=job_dir / f"pvc-{index}.json",
            output_dir=job_dir,
            command_name=f"pvc-{index}-json",
            environment=environment,
        )
    write_json(job_dir / "pvcs.json", pvc_receipts)

    pod_log_receipts = []
    for pod in sorted(
        pods_json.get("items", []),
        key=lambda item: item.get("metadata", {}).get("name", ""),
    ):
        pod_name = str(pod["metadata"]["name"])
        for container in pod.get("spec", {}).get("containers", []):
            container_name = str(container["name"])
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{pod_name}-{container_name}")
            receipt = _run_command(
                [
                    *base,
                    "logs",
                    "--timestamps",
                    f"pod/{pod_name}",
                    "-c",
                    container_name,
                ],
                environment=environment,
                cwd=launcher.parent,
                output_dir=job_dir,
                name=f"pod-log-{safe_name}",
                timeout_s=timeout_s,
            )
            pod_log_receipts.append(
                {
                    "container": container_name,
                    "pod": pod_name,
                    "stdout": receipt["stdout"],
                }
            )
    write_json(job_dir / "pod-logs.json", pod_log_receipts)


def validate_observed_topology(
    config: Mapping[str, Any],
    *,
    pods: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_node = str(config["cluster"]["node_name"])
    pod_items = pods.get("items", [])
    if not pod_items:
        raise ValueError("benchmark Job has no observed Pod")
    observed_nodes = {str(pod.get("spec", {}).get("nodeName", "")) for pod in pod_items}
    if observed_nodes != {expected_node}:
        raise ValueError(
            f"benchmark Pods landed on {sorted(observed_nodes)}, "
            f"expected only {expected_node}"
        )
    if set(nodes) != {expected_node}:
        raise ValueError("node receipts do not match the paired benchmark node")
    node = nodes[expected_node]
    labels = node.get("metadata", {}).get("labels", {})
    observed_gpu = labels.get("nvidia.com/gpu.product")
    expected_gpu = str(config["cluster"]["gpu_product"])
    if observed_gpu != expected_gpu:
        raise ValueError(
            f"observed GPU product {observed_gpu!r} does not match {expected_gpu!r}"
        )
    allocatable_gpus = int(
        node.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu", 0)
    )
    required_gpus = int(config["topology"]["total_gpus"])
    if required_gpus <= 0:
        raise ValueError("configured benchmark GPU request must be positive")
    if allocatable_gpus < required_gpus:
        raise ValueError(
            f"paired node exposes {allocatable_gpus} GPUs, requires {required_gpus}"
        )

    claims = validate_storage_config(config["storage"])
    expected_volumes = {
        "model-cache": claims["model_claim"],
        "run-output": claims["workspace_claim"],
        "benchmark-workspace": claims["workspace_claim"],
    }
    writers = []
    gpu_allocations = []
    for pod in pod_items:
        volumes = {
            str(volume["name"]): volume
            for volume in pod.get("spec", {}).get("volumes", [])
        }
        containers = pod.get("spec", {}).get("containers", [])
        if not containers:
            raise ValueError("benchmark Pod has no workload container")
        workload = next(
            (container for container in containers if container.get("name") == "miles"),
            None,
        )
        if workload is None:
            raise ValueError("benchmark Pod lacks the miles workload container")
        resources = workload.get("resources", {})
        requested = _gpu_quantity(
            resources.get("requests", {}).get("nvidia.com/gpu", 0),
            field="miles workload GPU request",
        )
        limited = _gpu_quantity(
            resources.get("limits", {}).get("nvidia.com/gpu", 0),
            field="miles workload GPU limit",
        )
        if requested != required_gpus or limited != required_gpus:
            raise ValueError(
                "miles workload GPU request and limit must exactly match "
                f"topology.total_gpus={required_gpus}; observed "
                f"request={requested}, limit={limited}"
            )
        other_gpu_users = []
        for container_type, candidates in (
            ("container", containers),
            ("initContainer", pod.get("spec", {}).get("initContainers", [])),
        ):
            for container in candidates:
                if container is workload:
                    continue
                value = _container_gpu_request(container)
                if value:
                    other_gpu_users.append(
                        {
                            "name": str(container.get("name", "")),
                            "requested": value,
                            "type": container_type,
                        }
                    )
        if other_gpu_users:
            raise ValueError(
                "only the miles workload container may request GPUs; "
                f"observed {other_gpu_users}"
            )
        gpu_allocations.append(
            {
                "container": "miles",
                "limit": limited,
                "pod": pod["metadata"]["name"],
                "requested": requested,
            }
        )
        mounts = {
            str(mount["name"]): mount for mount in workload.get("volumeMounts", [])
        }
        for volume_name, claim_name in expected_volumes.items():
            actual_claim = (
                volumes.get(volume_name, {})
                .get("persistentVolumeClaim", {})
                .get("claimName")
            )
            if actual_claim != claim_name:
                raise ValueError(
                    f"observed {volume_name} claim {actual_claim!r}, "
                    f"expected {claim_name!r}"
                )
            if mounts.get(volume_name, {}).get("readOnly", False):
                raise ValueError(f"observed {volume_name} is mounted read-only")
            writers.append(
                {
                    "claim": claim_name,
                    "container": workload["name"],
                    "mount_path": mounts[volume_name]["mountPath"],
                    "pod": pod["metadata"]["name"],
                    "volume": volume_name,
                    "writable": True,
                }
            )
    return {
        "gpu_count_allocatable": allocatable_gpus,
        "gpu_count_limit": required_gpus,
        "gpu_count_requested": required_gpus,
        "gpu_allocations": gpu_allocations,
        "gpu_product": observed_gpu,
        "node": expected_node,
        "pods": sorted(pod["metadata"]["name"] for pod in pod_items),
        "same_node": True,
        "writers": writers,
    }


def expected_owned_resources(
    config: Mapping[str, Any],
    item: Mapping[str, Any],
) -> list[dict[str, Any]]:
    run_id = str(config["run_id"])
    block = str(item["block"])
    repetition = str(item.get("repetition", 0))
    common_labels = _ownership_labels(
        run_id=run_id,
        block=block,
        repetition=repetition,
    )
    resources = [
        {
            "kind": "ConfigMap",
            "name": runtime_configmap_name(item, run_id=run_id),
            "labels": common_labels,
        },
        {
            "kind": "Job",
            "name": job_name(item, run_id=run_id),
            "labels": common_labels
            | {
                "modelexpress.nvidia.com/benchmark-mode": TRANSFER_MODES[
                    str(item["arm"])
                ],
            },
        },
    ]
    if item["arm"] == "modelexpress":
        server_name = server_resource_name(item, run_id=run_id)
        server_labels = common_labels | {
            "modelexpress.nvidia.com/server-id": server_name,
        }
        resources.extend(
            (
                {
                    "kind": "Deployment",
                    "name": server_name,
                    "labels": server_labels,
                },
                {
                    "kind": "Service",
                    "name": server_name,
                    "labels": server_labels,
                },
            )
        )
    return resources


def _kubectl_get_optional(
    config: Mapping[str, Any],
    *,
    resource: Mapping[str, Any],
    environment: Mapping[str, str],
    output_dir: Path,
    command_name: str,
    fail_on_error: bool = False,
) -> dict[str, Any] | None:
    command = [
        str(config.get("kubectl", "kubectl")),
        "--context",
        str(config["cluster"]["context"]),
        "-n",
        str(config["cluster"]["namespace"]),
        "get",
        f"{str(resource['kind']).lower()}/{resource['name']}",
        "-o",
        "json",
        "--ignore-not-found=true",
    ]
    receipt = _run_command(
        command,
        environment=environment,
        cwd=Path(config["launcher"]).parent,
        output_dir=output_dir,
        name=command_name,
        timeout_s=float(config["experiment"]["command_timeout_s"]),
        check=False,
    )
    if receipt["returncode"] != 0:
        if not fail_on_error:
            return None
        detail = (
            "timed out"
            if receipt.get("timed_out")
            else Path(receipt["stderr"]).read_text().strip()
        )
        raise RuntimeError(
            "pre-apply resource lookup failed for "
            f"{resource['kind']}/{resource['name']}: "
            f"{detail or f'exit code {receipt['returncode']}'}"
        )
    stdout = Path(receipt["stdout"]).read_text()
    if not stdout.strip():
        return None
    return json.loads(stdout)


def assert_owned_resources_absent(
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
    output_dir: Path,
) -> None:
    collisions = []
    for index, resource in enumerate(expected_owned_resources(config, item)):
        observed = _kubectl_get_optional(
            config,
            resource=resource,
            environment=environment,
            output_dir=output_dir,
            command_name=f"preapply-resource-{index}",
            fail_on_error=True,
        )
        if observed is not None:
            collisions.append(
                f"{resource['kind']}/{resource['name']} "
                f"uid={observed.get('metadata', {}).get('uid')}"
            )
    if collisions:
        raise ValueError(
            "refusing to overwrite existing benchmark resources: "
            + ", ".join(collisions)
        )


def capture_owned_resources(
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
    output_dir: Path,
    require_complete: bool,
) -> dict[str, Any]:
    captured = []
    missing = []
    for index, expected in enumerate(expected_owned_resources(config, item)):
        observed = _kubectl_get_optional(
            config,
            resource=expected,
            environment=environment,
            output_dir=output_dir,
            command_name=f"ownership-resource-{index}",
        )
        if observed is None:
            missing.append(f"{expected['kind']}/{expected['name']}")
            continue
        metadata = observed.get("metadata", {})
        uid = str(metadata.get("uid", ""))
        if not uid:
            raise ValueError(
                f"applied {expected['kind']}/{expected['name']} has no UID"
            )
        labels = metadata.get("labels", {})
        mismatched = {
            key: (labels.get(key), value)
            for key, value in expected["labels"].items()
            if labels.get(key) != value
        }
        if mismatched:
            raise ValueError(
                f"applied {expected['kind']}/{expected['name']} ownership "
                f"labels mismatch: {mismatched}"
            )
        captured.append(
            {
                "apiVersion": str(observed.get("apiVersion", "")),
                "kind": expected["kind"],
                "labels": expected["labels"],
                "name": expected["name"],
                "namespace": str(
                    metadata.get("namespace", config["cluster"]["namespace"])
                ),
                "uid": uid,
            }
        )
    payload = {
        "complete": not missing,
        "missing": missing,
        "resources": captured,
    }
    write_json(output_dir / "ownership.json", payload)
    if require_complete and missing:
        raise ValueError(f"applied resource inventory is incomplete: {missing}")
    return payload


def cleanup_job(
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    run_dir: Path,
) -> None:
    arm = str(item["arm"])
    job_dir = (
        run_dir
        / "jobs"
        / (f"block-{int(item['block']):02d}-order-{int(item['order'])}-{arm}")
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    environment = job_environment(config, item, run_id=str(config["run_id"]))
    ownership_path = job_dir / "ownership.json"
    if not ownership_path.is_file():
        write_json(
            job_dir / "cleanup.json",
            {
                "deleted": [],
                "refused": ["ownership receipt is absent"],
            },
        )
        return
    ownership = json.loads(ownership_path.read_text())
    deleted = []
    refused = []
    delete_paths = {
        "ConfigMap": ("api/v1", "configmaps"),
        "Deployment": ("apis/apps/v1", "deployments"),
        "Job": ("apis/batch/v1", "jobs"),
        "Service": ("api/v1", "services"),
    }
    for index, expected in enumerate(ownership.get("resources", [])):
        current = _kubectl_get_optional(
            config,
            resource=expected,
            environment=environment,
            output_dir=job_dir,
            command_name=f"cleanup-verify-{index}",
        )
        resource_name = f"{str(expected['kind']).lower()}/{expected['name']}"
        if current is None:
            continue
        metadata = current.get("metadata", {})
        labels = metadata.get("labels", {})
        if str(metadata.get("uid", "")) != str(expected["uid"]):
            refused.append(f"{resource_name}: UID changed")
            continue
        mismatched = {
            key: (labels.get(key), value)
            for key, value in expected["labels"].items()
            if labels.get(key) != value
        }
        if mismatched:
            refused.append(f"{resource_name}: labels changed {mismatched}")
            continue
        kind = str(expected["kind"])
        delete_path = delete_paths.get(kind)
        if delete_path is None:
            refused.append(f"{resource_name}: unsupported cleanup kind")
            continue
        api_prefix, resource_plural = delete_path
        delete_receipt = _run_command(
            [
                str(config.get("kubectl", "kubectl")),
                "--context",
                str(config["cluster"]["context"]),
                "delete",
                (
                    f"--raw=/{api_prefix}/namespaces/"
                    f"{config['cluster']['namespace']}/{resource_plural}/"
                    f"{expected['name']}"
                ),
                "-f",
                "-",
            ],
            environment=environment,
            cwd=Path(config["launcher"]).parent,
            output_dir=job_dir,
            name=f"cleanup-delete-{index}",
            timeout_s=float(config["experiment"]["command_timeout_s"]),
            check=False,
            input_text=json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "DeleteOptions",
                    "preconditions": {"uid": str(expected["uid"])},
                    "propagationPolicy": "Background",
                }
            ),
        )
        if delete_receipt["returncode"] != 0:
            refused.append(f"{resource_name}: UID-preconditioned delete failed")
            continue
        deleted.append(resource_name)
    write_json(
        job_dir / "cleanup.json",
        {
            "deleted": deleted,
            "refused": refused,
        },
    )


def _capture_failure_evidence(
    *,
    base: Sequence[str],
    environment: Mapping[str, str],
    job_dir: Path,
    job_name_value: str,
    launcher: Path,
    server_name: str | None,
    timeout_s: float,
) -> None:
    commands: list[tuple[str, list[str]]] = [
        ("failed-job", [*base, "get", f"job/{job_name_value}", "-o", "yaml"]),
        ("failed-job-json", [*base, "get", f"job/{job_name_value}", "-o", "json"]),
        (
            "failed-pods",
            [*base, "get", "pods", "-l", f"job-name={job_name_value}", "-o", "wide"],
        ),
        (
            "failed-events",
            [
                *base,
                "get",
                "events",
                "--sort-by=.lastTimestamp",
                "-o",
                "yaml",
            ],
        ),
        (
            "failed-logs",
            [*base, "logs", "--timestamps", f"job/{job_name_value}"],
        ),
        (
            "failed-node",
            [
                *base,
                "get",
                f"node/{environment['BENCH_NODE_NAME']}",
                "-o",
                "yaml",
            ],
        ),
        ("failed-pvcs", [*base, "get", "pvc", "-o", "yaml"]),
    ]
    if server_name is not None:
        commands.extend(
            (
                (
                    "failed-server-deployment",
                    [*base, "get", f"deployment/{server_name}", "-o", "yaml"],
                ),
                (
                    "failed-server-service",
                    [*base, "get", f"service/{server_name}", "-o", "yaml"],
                ),
                (
                    "failed-server-endpoints",
                    [*base, "get", f"endpoints/{server_name}", "-o", "yaml"],
                ),
                (
                    "failed-server-replicasets",
                    [
                        *base,
                        "get",
                        "replicasets",
                        "-l",
                        f"modelexpress.nvidia.com/server-id={server_name}",
                        "-o",
                        "yaml",
                    ],
                ),
                (
                    "failed-server-pods",
                    [
                        *base,
                        "get",
                        "pods",
                        "-l",
                        f"modelexpress.nvidia.com/server-id={server_name}",
                        "-o",
                        "wide",
                    ],
                ),
            )
        )
    for command_name, command in commands:
        _run_command(
            command,
            environment=environment,
            cwd=launcher.parent,
            output_dir=job_dir,
            name=command_name,
            timeout_s=min(timeout_s, 300),
            check=False,
        )
    pod_selectors = [("job", f"job-name={job_name_value}")]
    if server_name is not None:
        pod_selectors.append(
            (
                "server",
                f"modelexpress.nvidia.com/server-id={server_name}",
            )
        )
    for selector_name, selector in pod_selectors:
        pod_query = _run_command(
            [*base, "get", "pods", "-l", selector, "-o", "json"],
            environment=environment,
            cwd=launcher.parent,
            output_dir=job_dir,
            name=f"failed-{selector_name}-pods-json",
            timeout_s=min(timeout_s, 300),
            check=False,
        )
        if pod_query["returncode"] != 0:
            continue
        payload = json.loads(Path(pod_query["stdout"]).read_text())
        for pod in payload.get("items", []):
            pod_name = str(pod.get("metadata", {}).get("name", ""))
            if not pod_name:
                continue
            _run_command(
                [*base, "describe", f"pod/{pod_name}"],
                environment=environment,
                cwd=launcher.parent,
                output_dir=job_dir,
                name=f"failed-describe-{pod_name}",
                timeout_s=min(timeout_s, 300),
                check=False,
            )
            for container_type, candidates in (
                ("container", pod.get("spec", {}).get("containers", [])),
                (
                    "init",
                    pod.get("spec", {}).get("initContainers", []),
                ),
            ):
                for container in candidates:
                    container_name = str(container.get("name", ""))
                    for previous in (False, True):
                        suffix = "-previous" if previous else ""
                        command = [
                            *base,
                            "logs",
                            "--timestamps",
                            f"pod/{pod_name}",
                            "-c",
                            container_name,
                        ]
                        if previous:
                            command.append("--previous")
                        _run_command(
                            command,
                            environment=environment,
                            cwd=launcher.parent,
                            output_dir=job_dir,
                            name=(
                                f"failed-log-{pod_name}-{container_type}-"
                                f"{container_name}{suffix}"
                            ),
                            timeout_s=min(timeout_s, 300),
                            check=False,
                        )


def _legacy_trials(
    text: str,
    *,
    expected_updates: int,
    mode: str,
) -> list[dict[str, Any]]:
    records = parse_timer_log(text, expected_updates=expected_updates)
    trials = []
    for record in records:
        trials.append(
            {
                "schema": SCHEMA,
                "mode": mode,
                "trial": record["index"],
                "step": record["index"],
                "cold": record["index"] == 0,
                "headline_start_boundary": HEADLINE_START_BOUNDARY,
                "headline_end_boundary": HEADLINE_END_BOUNDARY,
                "update_weights_implementation_s": record[
                    "update_weights_implementation_s"
                ],
                "e2e_start_boundary": E2E_START_BOUNDARY,
                "e2e_end_boundary": E2E_END_BOUNDARY,
                "e2e_update_s": record["e2e_update_s"],
                "e2e_boundary_exact": False,
                "transport_s": None,
                "residual_s": None,
                "install_s": record["finalize_and_resume_s"],
                "cold_setup_s": record["setup_residual_s"],
                "bytes_total": None,
                "bytes_m2n": None,
                "bytes_residual": None,
                "coverage_fraction": None,
                "correctness": None,
                "excluded_names": None,
                "receipt_complete": False,
                "status": "ok",
                "timing_source": "actor_cell0_rank0_timer",
                "timing_valid": True,
                "weight_version": None,
                "start_timestamp_utc": record["start_timestamp_utc"],
            }
        )
    return trials


def validate_structured_trial(
    trial: Mapping[str, Any],
    *,
    mode: str,
) -> None:
    required = {
        "bytes_m2n",
        "bytes_residual",
        "bytes_total",
        "cold",
        "coverage_fraction",
        "e2e_update_s",
        "e2e_end_boundary",
        "e2e_start_boundary",
        "excluded_names",
        "headline_end_boundary",
        "headline_start_boundary",
        "install_s",
        "mode",
        "residual_s",
        "receipt_complete",
        "schema",
        "status",
        "step",
        "transport_s",
        "trial",
        "update_weights_implementation_s",
        "weight_version",
    }
    missing = required.difference(trial)
    if missing:
        raise ValueError(f"structured trial lacks fields: {sorted(missing)}")
    if trial["mode"] != mode:
        raise ValueError(
            f"structured trial mode {trial['mode']!r} does not match {mode!r}"
        )
    for field in ("trial", "step"):
        if isinstance(trial[field], bool) or not isinstance(trial[field], int):
            raise ValueError(f"structured trial {field} must be an integer")
    if trial["step"] != trial["trial"]:
        raise ValueError("structured trial step must equal its trial index")
    if trial["headline_start_boundary"] != HEADLINE_START_BOUNDARY:
        raise ValueError("structured trial has the wrong headline start boundary")
    if trial["headline_end_boundary"] != HEADLINE_END_BOUNDARY:
        raise ValueError("structured trial has the wrong headline end boundary")
    if trial["e2e_start_boundary"] != E2E_START_BOUNDARY:
        raise ValueError("structured trial has the wrong e2e start boundary")
    if trial["e2e_end_boundary"] != E2E_END_BOUNDARY:
        raise ValueError("structured trial has the wrong e2e end boundary")
    if trial["status"] != "ok":
        return
    if not str(trial["weight_version"]):
        raise ValueError("successful structured trial lacks a weight version")
    if float(trial["coverage_fraction"]) != 1.0:
        raise ValueError("successful structured trial must have full coverage")
    if int(trial["bytes_m2n"]) + int(trial["bytes_residual"]) != int(
        trial["bytes_total"]
    ):
        raise ValueError("structured trial byte accounting is inconsistent")
    correctness = trial.get("correctness")
    required_correctness = {
        "all_names_installed",
        "all_rollout_ready",
        "digest_equal",
        "training_run_completed",
        "version_agreement",
        "weight_checker_enabled",
    }
    if not isinstance(correctness, Mapping):
        raise ValueError("successful structured trial lacks correctness receipts")
    correctness_values = [correctness.get(field) for field in required_correctness]
    if any(value is False for value in correctness_values):
        raise ValueError("successful structured trial failed a correctness gate")
    all_correctness_observed = all(value is True for value in correctness_values)
    if bool(trial["receipt_complete"]) != all_correctness_observed:
        raise ValueError("structured trial receipt_complete is inconsistent")
    if trial.get("excluded_names"):
        raise ValueError("successful structured trial has unaccounted excluded names")


def _classify_trials(
    trials: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    item: Mapping[str, Any],
) -> list[dict[str, Any]]:
    experiment = config["experiment"]
    expected = int(experiment["num_rollouts"]) + 1
    if len(trials) != expected:
        raise ValueError(f"expected {expected} trials, observed {len(trials)}")
    result = []
    for index, source in enumerate(trials):
        if (
            isinstance(source["trial"], bool)
            or not isinstance(source["trial"], int)
            or isinstance(source["step"], bool)
            or not isinstance(source["step"], int)
            or source["trial"] != index
            or source["step"] != index
        ):
            raise ValueError(
                "trial and step indices must be the exact contiguous PR3304 raw "
                f"sequence 0-{EXPECTED_UPDATE_CALLS - 1}"
            )
        item_record = dict(source)
        if index == 0:
            classification = "cold"
        elif index == PR3304_TAIL_INDEX:
            classification = "tail_excluded"
        else:
            classification = "steady"
        item_record.update(
            {
                "arm": item["arm"],
                "block": item["block"],
                "classification": classification,
                "index": index,
                "receipt_complete": bool(item_record.get("receipt_complete", True)),
                "reference_step": index == PR3304_REFERENCE_INDEX,
                "run_id": config["run_id"],
                "timing_valid": bool(item_record.get("timing_valid", True)),
            }
        )
        result.append(item_record)
    return result


def run_job(
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    run_dir: Path,
) -> list[dict[str, Any]]:
    arm = str(item["arm"])
    job_dir = (
        run_dir
        / "jobs"
        / (f"block-{int(item['block']):02d}-order-{int(item['order'])}-{arm}")
    )
    job_dir.mkdir(parents=True)
    environment = job_environment(config, item, run_id=str(config["run_id"]))
    write_json(job_dir / "environment.json", environment)
    launcher = Path(config["launcher"])
    timeout_s = float(config["experiment"]["command_timeout_s"])
    manifest_path, documents = _render_manifest(
        launcher,
        environment=environment,
        output_dir=job_dir,
        storage=config["storage"],
        timeout_s=timeout_s,
    )
    validate_rendered_documents(
        documents,
        arm=arm,
        runtime_image=str(config["images"]["runtime"]),
        server_image=str(config["images"]["server"]),
        storage=config["storage"],
        server_name=(
            server_resource_name(item, run_id=str(config["run_id"]))
            if arm == "modelexpress"
            else None
        ),
    )
    kubectl = str(config.get("kubectl", "kubectl"))
    base = [
        kubectl,
        "--context",
        str(config["cluster"]["context"]),
        "-n",
        str(config["cluster"]["namespace"]),
    ]
    name = job_name(item, run_id=str(config["run_id"]))
    scoped_server = (
        server_resource_name(item, run_id=str(config["run_id"]))
        if arm == "modelexpress"
        else None
    )
    try:
        assert_owned_resources_absent(
            config,
            item,
            environment=environment,
            output_dir=job_dir,
        )
        for command_name, extra in (
            (
                "apply-dry-run",
                ["apply", "--dry-run=client", "-f", str(manifest_path)],
            ),
            ("apply", ["apply", "-f", str(manifest_path)]),
        ):
            _run_command(
                [*base, *extra],
                environment=environment,
                cwd=launcher.parent,
                output_dir=job_dir,
                name=command_name,
                timeout_s=timeout_s,
            )
        capture_owned_resources(
            config,
            item,
            environment=environment,
            output_dir=job_dir,
            require_complete=True,
        )
        if scoped_server is not None:
            _run_command(
                [
                    *base,
                    "rollout",
                    "status",
                    f"deployment/{scoped_server}",
                    "--timeout=5m",
                ],
                environment=environment,
                cwd=launcher.parent,
                output_dir=job_dir,
                name="server-rollout",
                timeout_s=timeout_s,
            )
        _run_command(
            [str(launcher), "wait"],
            environment=environment,
            cwd=launcher.parent,
            output_dir=job_dir,
            name="job-wait",
            timeout_s=timeout_s,
        )
    except BaseException:
        try:
            capture_owned_resources(
                config,
                item,
                environment=environment,
                output_dir=job_dir,
                require_complete=False,
            )
        except BaseException as ownership_error:
            write_json(
                job_dir / "ownership-capture-failure.json",
                {"error": repr(ownership_error)},
            )
        _capture_failure_evidence(
            base=base,
            environment=environment,
            job_dir=job_dir,
            job_name_value=name,
            launcher=launcher,
            server_name=scoped_server,
            timeout_s=timeout_s,
        )
        raise
    job_json = _kubectl_json(
        config,
        args=["get", f"job/{name}"],
        output_path=job_dir / "job.json",
        output_dir=job_dir,
        command_name="job-json",
        environment=environment,
    )
    pods_json = _kubectl_json(
        config,
        args=["get", "pods", "-l", f"job-name={name}"],
        output_path=job_dir / "pods.json",
        output_dir=job_dir,
        command_name="pods-json",
        environment=environment,
    )
    nodes = sorted(
        {
            item["spec"].get("nodeName")
            for item in pods_json.get("items", [])
            if item.get("spec", {}).get("nodeName")
        }
    )
    node_receipts = {}
    for index, node in enumerate(nodes):
        command = [
            kubectl,
            "--context",
            str(config["cluster"]["context"]),
            "get",
            f"node/{node}",
            "-o",
            "json",
        ]
        receipt = _run_command(
            command,
            environment=environment,
            cwd=launcher.parent,
            output_dir=job_dir,
            name=f"node-{index}-json",
            timeout_s=timeout_s,
        )
        node_receipts[node] = json.loads(Path(receipt["stdout"]).read_text())
    write_json(job_dir / "nodes.json", node_receipts)
    write_json(
        job_dir / "observed-topology.json",
        validate_observed_topology(
            config,
            pods=pods_json,
            nodes=node_receipts,
        ),
    )
    _capture_runtime_receipts(
        config,
        base=base,
        environment=environment,
        job_dir=job_dir,
        job_name_value=name,
        launcher=launcher,
        pods_json=pods_json,
        timeout_s=timeout_s,
    )

    logs_receipt = _run_command(
        [*base, "logs", "--timestamps", f"job/{name}"],
        environment=environment,
        cwd=launcher.parent,
        output_dir=job_dir,
        name="raw-pod",
        timeout_s=timeout_s,
    )
    raw_log_path = job_dir / "raw.log"
    raw_log_path.write_text(Path(logs_receipt["stdout"]).read_text())
    text = raw_log_path.read_text(errors="replace")
    trials = extract_structured_trials(text)
    mode_name = MODE_NAMES[arm]
    if trials:
        for trial in trials:
            validate_structured_trial(trial, mode=mode_name)
    else:
        trials = _legacy_trials(
            text,
            expected_updates=int(config["experiment"]["num_rollouts"]) + 1,
            mode=mode_name,
        )
    classified = _classify_trials(trials, config=config, item=item)
    write_jsonl(job_dir / "trials.jsonl", classified)
    write_json(
        job_dir / "summary.json",
        {
            "job_status": job_json.get("status", {}),
            "pod_start_timestamp": _pod_start_timestamp(pods_json),
            "timing": summarize_job(
                classified,
                pod_start_timestamp=_pod_start_timestamp(pods_json),
            ),
            "status_counts": dict(
                (status, sum(item["status"] == status for item in classified))
                for status in sorted({str(item["status"]) for item in classified})
            ),
        },
    )
    return classified


def run(config_path: Path) -> Path:
    config = normalize_config(json.loads(config_path.read_text()))
    run_dir = Path(config["output_root"]) / str(config["run_id"])
    if run_dir.exists():
        raise ValueError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    schedule = build_schedule(
        blocks=int(config["experiment"]["blocks"]),
        seed=int(config["experiment"].get("seed", 3304)),
    )
    harness_source = Path(__file__).resolve()
    launcher = Path(config["launcher"])
    manifest = {
        "benchmark_kind": config["benchmark_kind"],
        "config": {
            "path": str(config_path.resolve()),
            "sha256": _sha256(config_path),
        },
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "harness": {
            "path": str(harness_source),
            "sha256": _sha256(harness_source),
        },
        "launcher": {
            "path": str(launcher),
            "sha256": _sha256(launcher),
        },
        "metrics_contract": {
            "e2e_end_boundary": E2E_END_BOUNDARY,
            "e2e_start_boundary": E2E_START_BOUNDARY,
            "headline_end_boundary": HEADLINE_END_BOUNDARY,
            "headline_metric": "update_weights_implementation_s",
            "headline_start_boundary": HEADLINE_START_BOUNDARY,
            "num_rollouts": DEFAULT_NUM_ROLLOUTS,
            "raw_update_calls": EXPECTED_UPDATE_CALLS,
            "reference_index": PR3304_REFERENCE_INDEX,
            "steady_indices": list(PR3304_STEADY_INDICES),
            "tail_excluded_index": PR3304_TAIL_INDEX,
        },
        "normalized_config": config,
        "schema": HARNESS_SCHEMA,
        "sources": source_receipts(config),
    }
    write_json(run_dir / "manifest.json", manifest)
    write_jsonl(run_dir / "schedule.jsonl", schedule)
    all_trials: list[dict[str, Any]] = []
    try:
        preflight_persistent_storage(config, run_dir=run_dir)
        for item in schedule:
            try:
                trials = run_job(config, item, run_dir=run_dir)
                all_trials.extend(trials)
                for trial in trials:
                    append_jsonl(run_dir / "trials.jsonl", trial)
            finally:
                cleanup_job(config, item, run_dir=run_dir)
    except BaseException as error:
        write_json(
            run_dir / "failure.json",
            {
                "error": repr(error),
                "failed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        raise
    summary = summarize_pairs(
        all_trials,
        bootstrap_samples=int(config["experiment"].get("bootstrap_samples", 10_000)),
        seed=int(config["experiment"].get("seed", 3304)),
    )
    write_json(run_dir / "summary.json", summary)
    return run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="JSON configuration describing the persistent matched run.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_dir = run(args.config)
    print(run_dir)


if __name__ == "__main__":
    main()
