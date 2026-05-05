# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pull Criterion benchmark timings from GitHub Actions runs and emit a time-series.

Defaults target the post-merge benchmark job on `main`. Override with flags to
focus on specific bench names, narrower time windows, or other workflows/jobs.

Requirements: an authenticated `gh` CLI on PATH and Python 3.10+.

Examples:
    # Default: last 50 post-merge runs on main, all benches, CSV
    python3 scripts/bench_perf_history.py

    # Only status_* benches, last 20 runs, table
    python3 scripts/bench_perf_history.py --filter '^status_' --limit 20 --format table

    # Future: track transfer-throughput benches once they exist
    python3 scripts/bench_perf_history.py --filter 'transfer'
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime

DEFAULT_WORKFLOW = "ci.yml"
DEFAULT_JOB_NAME = "Performance Benchmarks"
DEFAULT_BRANCH = "main"
DEFAULT_EVENT = "push"
DEFAULT_LIMIT = 50
DEFAULT_FILTER = r".*"
DEFAULT_FORMAT = "csv"
DEFAULT_WORKERS = 8

UNIT_TO_NS: dict[str, float] = {
    "ps": 1e-3,
    "ns": 1.0,
    "us": 1e3,
    "µs": 1e3,
    "ms": 1e6,
    "s": 1e9,
}

GH_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[^Z]+Z\s+")

TIME_LINE_RE = re.compile(
    r"^(?P<name>\S+)\s+time:\s+"
    r"\[(?P<low>[\d.]+)\s+(?P<low_u>\S+)\s+"
    r"(?P<mid>[\d.]+)\s+(?P<mid_u>\S+)\s+"
    r"(?P<high>[\d.]+)\s+(?P<high_u>\S+)\]\s*$"
)


@dataclass
class BenchSample:
    run_id: int
    run_started_at: str
    head_sha: str
    bench_name: str
    low_ns: float
    mid_ns: float
    high_ns: float
    raw_unit: str
    raw_low: float
    raw_mid: float
    raw_high: float


def gh(*args: str, check: bool = True) -> str:
    cmd = ["gh", *args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
    except FileNotFoundError as exc:
        raise SystemExit("gh CLI not found on PATH; install from https://cli.github.com") from exc
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise SystemExit(f"gh command failed ({' '.join(cmd)}): {msg}") from exc
    return result.stdout


def detect_repo() -> str:
    out = gh("repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner").strip()
    if not out:
        raise SystemExit("Could not detect repo; pass --repo owner/name")
    return out


def list_runs(
    repo: str,
    workflow: str,
    branch: str,
    event: str,
    limit: int,
    since: datetime | None,
) -> list[dict]:
    out = gh(
        "run",
        "list",
        "--repo",
        repo,
        "--workflow",
        workflow,
        "--branch",
        branch,
        "--event",
        event,
        "--limit",
        str(limit),
        "--json",
        "databaseId,headSha,createdAt,conclusion,workflowName",
    )
    runs = json.loads(out)
    runs = [r for r in runs if r.get("conclusion") == "success"]
    if since is not None:
        runs = [r for r in runs if datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00")) >= since]
    return runs


def find_bench_job_id(repo: str, run_id: int, job_name: str) -> int | None:
    out = gh(
        "api",
        f"/repos/{repo}/actions/runs/{run_id}/jobs",
        "--paginate",
        "-q",
        f'.jobs[] | select(.name == "{job_name}") | .id',
    ).strip()
    if not out:
        return None
    return int(out.splitlines()[0])


def fetch_job_log(repo: str, job_id: int) -> str:
    return gh("api", f"/repos/{repo}/actions/jobs/{job_id}/logs")


def parse_timing_lines(log: str) -> list[tuple[str, str, float, float, float]]:
    samples: list[tuple[str, str, float, float, float]] = []
    for raw in log.splitlines():
        line = GH_TS_RE.sub("", raw).rstrip()
        m = TIME_LINE_RE.match(line)
        if not m:
            continue
        unit = m.group("mid_u")
        scale = UNIT_TO_NS.get(unit)
        if scale is None:
            continue
        samples.append(
            (
                m.group("name"),
                unit,
                float(m.group("low")),
                float(m.group("mid")),
                float(m.group("high")),
            )
        )
    return samples


def collect_samples(
    repo: str,
    runs: list[dict],
    job_name: str,
    name_filter: re.Pattern[str],
    workers: int,
) -> list[BenchSample]:
    def work(run: dict) -> list[BenchSample]:
        run_id = run["databaseId"]
        job_id = find_bench_job_id(repo, run_id, job_name)
        if job_id is None:
            return []
        log = fetch_job_log(repo, job_id)
        out: list[BenchSample] = []
        for name, unit, low, mid, high in parse_timing_lines(log):
            if not name_filter.search(name):
                continue
            scale = UNIT_TO_NS[unit]
            out.append(
                BenchSample(
                    run_id=run_id,
                    run_started_at=run["createdAt"],
                    head_sha=run["headSha"],
                    bench_name=name,
                    low_ns=low * scale,
                    mid_ns=mid * scale,
                    high_ns=high * scale,
                    raw_unit=unit,
                    raw_low=low,
                    raw_mid=mid,
                    raw_high=high,
                )
            )
        return out

    samples: list[BenchSample] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, r) for r in runs]
        for fut in as_completed(futures):
            samples.extend(fut.result())
    samples.sort(key=lambda s: (s.run_started_at, s.bench_name))
    return samples


def emit_csv(samples: list[BenchSample]) -> None:
    if not samples:
        return
    writer = csv.DictWriter(sys.stdout, fieldnames=list(asdict(samples[0]).keys()))
    writer.writeheader()
    for s in samples:
        writer.writerow(asdict(s))


def emit_json(samples: list[BenchSample]) -> None:
    json.dump([asdict(s) for s in samples], sys.stdout, indent=2)
    sys.stdout.write("\n")


def emit_table(samples: list[BenchSample]) -> None:
    if not samples:
        return
    headers = ["run_started_at", "head_sha", "bench_name", "mid", "unit"]
    rows = [
        [
            s.run_started_at,
            s.head_sha[:8],
            s.bench_name,
            f"{s.raw_mid:.3f}",
            s.raw_unit,
        ]
        for s in samples
    ]
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--workflow", default=DEFAULT_WORKFLOW, help="workflow file name")
    p.add_argument("--job", default=DEFAULT_JOB_NAME, help="job display name within workflow")
    p.add_argument("--branch", default=DEFAULT_BRANCH, help="branch filter")
    p.add_argument("--event", default=DEFAULT_EVENT, help="event filter (push = post-merge)")
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="max runs to fetch")
    p.add_argument("--since", default=None, help="ISO date floor, e.g. 2026-01-01")
    p.add_argument("--filter", default=DEFAULT_FILTER, help="regex on bench name")
    p.add_argument(
        "--format",
        choices=("csv", "json", "table"),
        default=DEFAULT_FORMAT,
        help="output format",
    )
    p.add_argument("--repo", default=None, help="owner/name (autodetected from git remote)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel log fetches")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    try:
        name_filter = re.compile(args.filter)
    except re.error as exc:
        print(f"Invalid --filter regex: {exc}", file=sys.stderr)
        return 2

    since = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since).astimezone()
        except ValueError as exc:
            print(f"Invalid --since: {exc}", file=sys.stderr)
            return 2

    repo = args.repo or detect_repo()

    runs = list_runs(repo, args.workflow, args.branch, args.event, args.limit, since)
    if not runs:
        print("No matching successful runs found.", file=sys.stderr)
        return 0

    samples = collect_samples(repo, runs, args.job, name_filter, args.workers)
    if not samples:
        print(
            f"No benchmark timings parsed from {len(runs)} runs "
            f"(job '{args.job}' may not exist or had no timings).",
            file=sys.stderr,
        )
        return 0

    if args.format == "csv":
        emit_csv(samples)
    elif args.format == "json":
        emit_json(samples)
    else:
        emit_table(samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
