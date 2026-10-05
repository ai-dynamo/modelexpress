"""Export native MX stages from unforwarded Ray logs, preserving their scope."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

MARKER = "MX_REFIT_TIMING "
WORKER = re.compile(r"\(Worker_TP(?P<rank>\d+)_EP\d+ pid=\d+\)")
STAGES = {
    "control_discovery",
    "source_preparation",
    "setup_registration",
    "transfer_planning",
    "wire_transfer",
    "receive_sync",
    "transformation",
    "installation",
    "post_install",
    "rollout_readiness",
}
METRICS = {
    "materialization_s": "Host duration materializing vLLM-owned load-time tensors; nested in install_commit_s. GPU work can complete at the later drain.",
    "receive_copy_s": "Host duration submitting receive-to-engine copies; nested in install_commit_s. Not an independently synchronized GPU copy measurement.",
    "post_load_processing_s": "Host duration of native vLLM complete-module processing, including kernel destination copy/restoration; nested in install_commit_s. Deferred attention finalization belongs to reload_s instead.",
}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value, name, *, integer=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (integer and not isinstance(value, int))
    ):
        raise ValueError(f"Invalid native {name}: {value}")
    return value


def validate_payload(payload):
    if not isinstance(payload, dict) or payload.get("backend") != "rl_generator":
        raise ValueError("Native timing payload has an invalid backend")
    if not isinstance(payload.get("version"), str) or not payload["version"]:
        raise ValueError("Native timing payload has no version")
    # vLLM does not always populate LOCAL_RANK. Global rank is independently
    # proven by the worker prefix and the complete receiver ledger.
    local_rank = payload.get("rank")
    if local_rank is not None:
        number(local_rank, "local rank", integer=True)
        if local_rank >= 8:
            raise ValueError("Native local rank is outside the eight-GPU host")
    for name in ("bytes", "e2e_ms", "unattributed_ms"):
        number(payload.get(name), name, integer=name == "bytes")
    stages = payload.get("stages")
    if not isinstance(stages, dict) or set(stages) != STAGES:
        raise ValueError("Native timing stage schema is incomplete or unexpected")
    for name, stage in stages.items():
        if not isinstance(stage, dict):
            raise ValueError(f"Invalid native stage {name}")
        count = number(stage.get("count"), f"{name}.count", integer=True)
        duration = number(stage.get("duration_ms"), f"{name}.duration_ms")
        status = stage.get("status")
        if status not in {"ok", "mixed", "combined", "not_applicable", "not_recorded"}:
            raise ValueError(f"Failed or unknown native stage status: {name}={status}")
        if status in {"combined", "not_applicable", "not_recorded"} and (
            count or duration
        ):
            raise ValueError(f"Unmeasured native stage has a duration: {name}")
        if status == "ok" and count == 0:
            raise ValueError(f"Measured native stage has no spans: {name}")
        statuses = stage.get("statuses", [])
        if not isinstance(statuses, list) or any(
            item not in {"ok", "combined", "not_applicable"} for item in statuses
        ):
            raise ValueError(f"Failed or unknown native span statuses: {name}")
        if status == "mixed" and not statuses:
            raise ValueError(f"Mixed native stage has no statuses: {name}")
        metadata = stage.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid native stage metadata: {name}")
        for key, value in metadata.items():
            if key.endswith("_s"):
                number(value, f"{name}.{key}")
    for name in METRICS:
        number(stages["installation"].get("metadata", {}).get(name), name)


def verify_coverage(records, receiver_rows):
    expected = {}
    for row in receiver_rows:
        key = (row["version_uid"], int(row["rank"]))
        if key in expected:
            raise ValueError(f"Duplicate authoritative receiver row: {key}")
        expected[key] = int(row["step"])
    versions = {row["version_uid"] for row in receiver_rows}
    if len(versions) != 12 or len(expected) != 32 * 12:
        raise ValueError("Expected a complete 32-rank, twelve-update receiver ledger")
    for version in versions:
        if {rank for current, rank in expected if current == version} != set(range(32)):
            raise ValueError(f"Incomplete receiver rank coverage: {version}")
    groups = {}
    for record in records:
        validate_payload(record["payload"])
        rank = number(record["global_tp_rank"], "global TP rank", integer=True)
        key = (record["payload"]["version"], rank)
        groups.setdefault(key, []).append(record)
    if set(groups) != set(expected):
        missing, extra = set(expected) - set(groups), set(groups) - set(expected)
        raise ValueError(
            f"Native timing coverage differs from receiver ledger: {len(missing)} missing, {len(extra)} extra"
        )
    for key, values in groups.items():
        if expected[key] >= 2 and len(values) != 1:
            raise ValueError(f"Warm native cycle is ambiguous: {key}, {len(values)}")
    return groups


def extract(logs):
    records = []
    for path in sorted((logs / "ray-worker-logs").glob("*/files/worker-*.out")):
        text = path.read_text(errors="strict")
        rank_set = {int(match["rank"]) for match in WORKER.finditer(text)}
        if MARKER not in text:
            continue
        if len(rank_set) != 1:
            raise ValueError(f"Native log does not identify one global TP rank: {path}")
        rank = rank_set.pop()
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        seen_payloads = set()
        for line_number, line in enumerate(text.splitlines(), 1):
            if MARKER not in line:
                continue
            raw = line.split(MARKER, 1)[1]
            payload, end = json.JSONDecoder().raw_decode(raw)
            if raw[end:].strip():
                raise ValueError(
                    f"Trailing native record content: {path}:{line_number}"
                )
            if payload.get("backend") != "rl_generator":
                continue
            # Logger and explicit stdout can both target stdout; discard only
            # byte-identical duplicates from the same producer file.
            if raw in seen_payloads:
                continue
            seen_payloads.add(raw)
            records.append(
                {
                    "global_tp_rank": rank,
                    "source_file": str(path),
                    "source_line": line_number,
                    "source_sha256": source_sha,
                    "payload": payload,
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    records = extract(args.logs)
    if not records:
        raise ValueError("No native MX timing records in captured Ray worker stdout")
    with gzip.open(
        args.results / "receiver-per-rank.csv.gz", "rt", newline=""
    ) as stream:
        receiver_rows = list(csv.DictReader(stream))
    groups = verify_coverage(records, receiver_rows)
    stages, observed_cycles = [], {}
    for record in records:
        payload = record["payload"]
        key = (payload["version"], record["global_tp_rank"])
        cycle = observed_cycles.get(key, 0)
        observed_cycles[key] = cycle + 1
        for stage_name, stage in payload["stages"].items():
            stages.append(
                {
                    "version_uid": payload["version"],
                    "global_tp_rank": record["global_tp_rank"],
                    "native_local_rank": payload["rank"],
                    "cycle_index": cycle,
                    "stage": stage_name,
                    "duration_s": stage["duration_ms"] / 1000,
                    "status": stage["status"],
                    "count": stage["count"],
                    "metadata_json": json.dumps(
                        stage.get("metadata", {}), sort_keys=True
                    ),
                    "source_file": record["source_file"],
                    "source_line": record["source_line"],
                    "source_sha256": record["source_sha256"],
                }
            )
    with (args.results / "per-update.csv").open(newline="") as stream:
        updates = list(csv.DictReader(stream))
    rows = []
    for update in updates:
        step = int(update["step"])
        key = (update["version_uid"], int(update["critical_receiver_rank"]))
        cycles = groups.get(key, [])
        if step >= 2 and len(cycles) != 1:
            raise ValueError(
                f"Warm native cycle is missing or ambiguous: {key}, {len(cycles)}"
            )
        if len(cycles) != 1:
            continue
        record = cycles[0]
        metadata = record["payload"]["stages"]["installation"].get("metadata", {})
        row = {
            key: update[key]
            for key in (
                "run_id",
                "job_id",
                "step",
                "version_uid",
                "critical_receiver_rank",
                "diagnostic_replay",
            )
        }
        for name in METRICS:
            value = metadata.get(name)
            number(value, name)
            row[name] = value
        row["source_file"] = record["source_file"]
        row["source_line"] = record["source_line"]
        rows.append(row)
    warm = [row for row in rows if 2 <= int(row["step"]) <= 11]
    if len(warm) != 10:
        raise ValueError("Native warm summary requires ten critical-rank updates")
    summaries = []
    for name, definition in METRICS.items():
        values = sorted(row[name] for row in warm)
        summaries.append(
            {
                "metric": name,
                "unit": "seconds",
                "n": len(values),
                "min": values[0],
                "median": statistics.median(values),
                "p95": values[math.ceil(0.95 * len(values)) - 1],
                "max": values[-1],
                **{f"update_{int(row['step']):02d}": row[name] for row in rows},
                "definition": definition,
                "sampling": "Same critical receiver rank as main.csv. Host intervals nested inside installation; do not add again to parent times. Diagnostic replay designation follows the selected arm.",
            }
        )
    write_csv(args.results / "mx-native-stages.csv", stages)
    write_csv(args.results / "mx-native-install-per-update.csv", rows)
    write_csv(args.results / "mx-native-install-summary.csv", summaries)
    (args.results / "mx-native-records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    print(
        json.dumps(
            {
                "native_records": len(records),
                "native_stages": len(stages),
                "critical_updates": len(rows),
            }
        )
    )


if __name__ == "__main__":
    main()
