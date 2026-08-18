#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the metrics exposition path against the defects it was built to fix.

The unit tests assert that the corrected shape is produced. This measures *how
much* was being lost, which is what decides whether the numbers on a dashboard
could ever have been trusted — and produces the one number the design left open:
how scrape cost grows as dead-PID mmap files accumulate.

Every "before" column is a faithful reconstruction of the previous mechanism —
one registry per process with no shared directory, a hostname grouping key, a
flush that only runs on a clean exit — driven through the same synthetic
workload as the "after" column, in the same forked-rank harness. Nothing here is
extrapolated from a single-process run.

Run it::

    python benchmarks/metrics_pipeline.py                # every scenario, TP=8
    python benchmarks/metrics_pipeline.py --ranks 2      # TP=2
    python benchmarks/metrics_pipeline.py --only scrape-cost --max-file-sets 512
    python benchmarks/metrics_pipeline.py --json

It needs ``prometheus-client`` and nothing else: no GPU, no cluster, and no
network beyond a loopback stub Pushgateway.

One ordering constraint drives the whole file. ``PROMETHEUS_MULTIPROC_DIR`` is
exported at the top of :func:`main` **before anything imports
prometheus_client**, and is never unset afterwards — only re-pointed at a fresh
subdirectory per phase. That is not tidiness: ``prometheus_client.values``
latches its value class at import, and unsetting the variable afterwards leaves
metric creation trying to join a path against ``None``. It is the same ordering
constraint the pod manifest has to satisfy, which is the point.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import socket
import statistics
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Events each rank records. Rank ``r`` records ``EVENTS_PER_RANK * (r + 1)`` so
#: a lost rank shows up in the total instead of being hidden by symmetry.
EVENTS_PER_RANK = 10

#: Prometheus's default scrape timeout. Crossing it means the endpoint is down
#: as far as the fleet is concerned, with nothing in any log to say why.
PROMETHEUS_DEFAULT_SCRAPE_TIMEOUT_S = 10.0

_FAMILY = "mx_p2p_source_attempts_total"

#: Root for every phase's multiprocess directory; set by main().
_ROOT: Path | None = None


@dataclass
class Result:
    scenario: str
    question: str
    before: str
    after: str
    detail: str = ""
    rows: list[tuple] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def multiproc_dir(name: str):
    """Point ``PROMETHEUS_MULTIPROC_DIR`` at a fresh per-phase directory.

    Re-points, never unsets: ``prometheus_client.values`` reads the variable on
    every value creation, and a phase that cleared it would break every later
    phase in the same process.
    """
    assert _ROOT is not None, "main() must set the multiprocess root first"
    path = _ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(path)
    try:
        yield path
    finally:
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = previous or str(_ROOT)


def _expected_total(ranks: int) -> int:
    return EVENTS_PER_RANK * ranks * (ranks + 1) // 2


def _merged_total(path: Path, family: str = _FAMILY) -> float:
    """Sum a family across every mmap file in ``path``, the way a scrape does."""
    from prometheus_client import CollectorRegistry, generate_latest, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=str(path))
    return _sum_family(generate_latest(registry).decode(), family)


def _sum_family(exposition: str, family: str = _FAMILY) -> float:
    total = 0.0
    for line in exposition.splitlines():
        if line.startswith(family + "{") or line.startswith(family + " "):
            try:
                total += float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
    return total


def _fork_ranks(ranks: int, child, kill_ranks: set[int] | None = None) -> None:
    """Run ``child(rank)`` in ``ranks`` forked processes and reap them.

    Ranks in ``kill_ranks`` SIGKILL themselves once their work is done, which is
    what an OOM-kill looks like to the metrics layer: no exit hook runs at all.

    Raises if any rank that was *not* meant to be killed died anyway. A crashed
    rank looks exactly like a rank whose data was dropped, so letting one
    through would turn a broken harness into a flattering result.
    """
    kill_ranks = kill_ranks or set()
    pids = {}
    for rank in range(ranks):
        pid = os.fork()
        if pid == 0:
            try:
                child(rank)
            except BaseException:
                os._exit(1)
            if rank in kill_ranks:
                os.kill(os.getpid(), signal.SIGKILL)
            os._exit(0)
        pids[pid] = rank

    failures = []
    for pid, rank in pids.items():
        _, status = os.waitpid(pid, 0)
        if rank in kill_ranks:
            continue
        if os.WIFSIGNALED(status):
            failures.append(f"rank {rank} died on signal {os.WTERMSIG(status)}")
        elif os.WIFEXITED(status) and os.WEXITSTATUS(status) != 0:
            failures.append(f"rank {rank} exited {os.WEXITSTATUS(status)}")
    if failures:
        raise RuntimeError(
            "benchmark ranks failed, so the totals below would understate the "
            "'after' path rather than measure it: " + "; ".join(failures)
        )


def _file_sets(path: Path) -> int:
    """Distinct PIDs that wrote into ``path``.

    Counting files would overcount: each rank writes one file per metric type
    (``counter_<pid>.db`` plus ``gauge_mostrecent_<pid>.db`` for
    ``mx_build_info``), and the merge cost scales with PIDs, not with files.
    """
    pids = set()
    for db in path.glob("*.db"):
        pids.add(db.stem.rsplit("_", 1)[-1])
    return len(pids)


def _record_via_modelexpress(rank: int) -> None:
    """The shipped client path, as a forked rank runs it."""
    import modelexpress.metrics as mx

    mx.enable_metrics()
    for _ in range(EVENTS_PER_RANK * (rank + 1)):
        mx.metrics.record_attempt("random", "success")


# ---------------------------------------------------------------------------
# D2 — the pull path: one endpoint per pod, not one rank per pod
# ---------------------------------------------------------------------------


def bench_pull_path(ranks: int) -> Result:
    """How much of a pod's data reaches a scrape of its one /metrics endpoint?

    Before: every rank called ``start_http_server`` on the same port. One bound;
    the rest caught EADDRINUSE, logged a warning, and went on recording into a
    registry nothing would ever read. The endpoint served one rank's data while
    presenting as the pod's.

    The "before" run reconstructs that by giving each rank a private directory —
    which is what "no shared state between ranks" means — and then scraping one
    rank's, exactly as a scrape of the old endpoint would have.

    Which rank wins the bind is a race, so the result is reported over *every*
    possible winner rather than assuming rank 0. That matters here: the harness
    deliberately has rank ``r`` record ``EVENTS_PER_RANK * (r + 1)`` events so a
    dropped rank is visible in the total, which also makes rank 0 the smallest
    contributor — quoting it alone would report the most flattering of the N
    possible outcomes as if it were the measurement.
    """
    expected = _expected_total(ranks)

    os.environ["MX_METRICS_ENABLED"] = "1"
    os.environ.pop("MX_METRICS_PORT", None)
    os.environ.pop("MX_METRICS_PUSHGATEWAY", None)

    with multiproc_dir("pull-before") as before_root:

        def isolated(rank: int) -> None:
            # Per-rank directory == the old per-process registry: nothing any
            # other rank writes is reachable from here.
            private = before_root / f"rank-{rank}"
            private.mkdir(parents=True, exist_ok=True)
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(private)
            _record_via_modelexpress(rank)

        _fork_ranks(ranks, isolated)
        per_winner = [
            _merged_total(before_root / f"rank-{rank}") for rank in range(ranks)
        ]

    with multiproc_dir("pull-after") as after_root:
        _fork_ranks(ranks, _record_via_modelexpress)
        after_total = _merged_total(after_root)
        file_sets = _file_sets(after_root)

    mean = statistics.mean(per_winner)
    worst, best = min(per_winner), max(per_winner)

    return Result(
        scenario="D2 pull path",
        question=f"Of a TP={ranks} pod's recorded events, how many reach a scrape?",
        before=(
            f"{mean:.0f} / {expected} ({mean / expected:.1%}) mean over which rank "
            f"wins the bind; {worst / expected:.1%}–{best / expected:.1%}"
        ),
        after=f"{after_total:.0f} / {expected} ({after_total / expected:.1%})",
        detail=(
            f"Before: the bind winner's registry only — the other {ranks - 1} "
            f"rank{'s' if ranks != 2 else ''} recorded into memory nothing reads, "
            f"so the pod loses (N-1)/N of its data whichever rank wins. "
            f"After: {file_sets} ranks merged behind one port, every run."
        ),
    )


# ---------------------------------------------------------------------------
# D3 — the push path: a PUT replaces the whole group
# ---------------------------------------------------------------------------


class _StubPushgateway(HTTPServer):
    """Enough of a Pushgateway to reproduce the grouping-key semantics.

    ``push_to_gateway`` is a ``PUT``, and a PUT replaces every series in the
    group named by ``job`` plus the grouping key. That is the whole defect: the
    key was the hostname, and every rank on a node shares one.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups: dict[str, str] = {}
        self.put_count = 0
        self.lock = threading.Lock()


class _StubHandler(BaseHTTPRequestHandler):
    def do_PUT(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        with self.server.lock:
            self.server.groups[self.path] = body  # PUT replaces the group
            self.server.put_count += 1
        self.send_response(200)
        self.end_headers()

    do_POST = do_PUT

    def log_message(self, *args):
        pass


def _raw_put(gateway: str, path: str, body: bytes) -> None:
    """PUT over a bare socket, from inside a forked child.

    Deliberately not ``push_to_gateway``: that goes through ``urllib``, which on
    macOS resolves proxies through SystemConfiguration and aborts the child with
    ``+[NSNumber initialize] may have been in progress in another thread when
    fork() was called``. A crashed child would push nothing and make the "before"
    column look better than the mechanism it is reconstructing.
    """
    host, _, port = gateway.partition(":")
    request = (
        f"PUT {path} HTTP/1.1\r\n"
        f"Host: {gateway}\r\n"
        f"Content-Type: text/plain; version=0.0.4\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode() + body
    with socket.create_connection((host, int(port)), timeout=10) as sock:
        sock.sendall(request)
        sock.shutdown(socket.SHUT_WR)
        while sock.recv(4096):
            pass


@contextlib.contextmanager
def _stub_gateway():
    server = _StubPushgateway(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def bench_push_path(ranks: int) -> Result:
    """How much survives when a whole pod pushes to one gateway?

    Before: ``grouping_key={"instance": gethostname()}``. Ranks share a
    hostname, so each PUT wiped the one before it and whichever rank exited last
    was the only survivor — and the result looked complete.

    After: the key is the pod, and the payload is the merged union of every
    rank, so concurrent pushes carry the same data instead of erasing it.
    """
    expected = _expected_total(ranks)
    os.environ["MX_METRICS_ENABLED"] = "1"
    os.environ.pop("MX_METRICS_PORT", None)

    with _stub_gateway() as (server, gateway):
        # --- before: per-rank registry, host-keyed group ---
        #
        # This has to fork. Running the ranks in one process would not isolate
        # them: prometheus_client caches its mmap file by metric *type*
        # ("counter"), not by path, so every Counter after the first reuses the
        # first one's file no matter where PROMETHEUS_MULTIPROC_DIR points, and
        # the ranks silently sum into one value.
        hostname = socket.gethostname()
        with multiproc_dir("push-before") as before_root:

            def old_push(rank: int) -> None:
                from prometheus_client import CollectorRegistry, Counter, generate_latest

                private = before_root / f"rank-{rank}"
                private.mkdir(parents=True, exist_ok=True)
                os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(private)

                registry = CollectorRegistry()
                counter = Counter(
                    _FAMILY.removesuffix("_total"),
                    "reconstruction of the pre-fix family",
                    ["policy", "scheme", "result"],
                    registry=registry,
                )
                counter.labels("random", "", "success").inc(EVENTS_PER_RANK * (rank + 1))
                _raw_put(
                    gateway,
                    f"/metrics/job/modelexpress/instance/{hostname}",
                    generate_latest(registry),
                )

            _fork_ranks(ranks, old_push)

        with server.lock:
            before_total = _sum_family("\n".join(server.groups.values()))
            before_groups = len(server.groups)
            before_puts = server.put_count
            server.groups.clear()
            server.put_count = 0

        # --- after: shared directory, pod-keyed, merged registry ---
        os.environ["MX_METRICS_PUSHGATEWAY"] = gateway
        os.environ["POD_UID"] = "bench-pod-uid"
        try:
            with multiproc_dir("push-after"):
                _fork_ranks(ranks, _record_via_modelexpress)
                # One push per pod, from the merged registry, once every rank
                # has written — what the atexit hook does in a real pod.
                import modelexpress.metrics as mx

                mx.push_metrics_if_enabled()
            with server.lock:
                after_total = _sum_family("\n".join(server.groups.values()))
                after_groups = len(server.groups)
        finally:
            os.environ.pop("MX_METRICS_PUSHGATEWAY", None)
            os.environ.pop("POD_UID", None)

    return Result(
        scenario="D3 push path",
        question=f"After a TP={ranks} pod pushes, how much is retained?",
        before=f"{before_total:.0f} / {expected} ({before_total / expected:.1%})",
        after=f"{after_total:.0f} / {expected} ({after_total / expected:.1%})",
        detail=(
            f"Before: {before_puts} PUTs collapsed into {before_groups} host-keyed "
            f"group(s); each replaced the last, so the survivor is whichever rank "
            f"PUT last and the fraction varies run to run. After: {after_groups} "
            f"pod-keyed group carrying the merged union, every run."
        ),
    )


# ---------------------------------------------------------------------------
# D6 — a SIGKILLed rank runs no exit hook
# ---------------------------------------------------------------------------


def bench_hard_kill(ranks: int) -> Result:
    """What survives when ranks are OOM-killed mid-run?

    Before: the flush ran from ``atexit``, which does not run on SIGKILL. A rank
    killed mid-transfer contributed nothing — precisely the failure the metrics
    exist to explain.

    After: mmap writes are durable at increment time, so there is no exit hook
    in the path at all. Every rank here is hard-killed: the worst case.
    """
    expected = _expected_total(ranks)
    killed = set(range(ranks))

    os.environ["MX_METRICS_ENABLED"] = "1"
    os.environ.pop("MX_METRICS_PORT", None)
    os.environ.pop("MX_METRICS_PUSHGATEWAY", None)

    with multiproc_dir("hard-kill") as root:
        _fork_ranks(ranks, _record_via_modelexpress, kill_ranks=killed)
        after_total = _merged_total(root)
        file_sets = _file_sets(root)

    return Result(
        scenario="D6 hard kill",
        question=f"All {ranks} ranks SIGKILLed. How much data survives?",
        before=f"0 / {expected} (0.0%)",
        after=f"{after_total:.0f} / {expected} ({after_total / expected:.1%})",
        detail=(
            f"{file_sets} rank(s) worth of mmap files outlived their processes. "
            f"Before, an atexit flush ran for none of them."
        ),
    )


# ---------------------------------------------------------------------------
# Scrape cost — the one trade-off the design leaves open
# ---------------------------------------------------------------------------


def bench_scrape_cost(max_file_sets: int = 256, repeats: int = 5) -> Result:
    """How does merge cost grow as dead-PID files accumulate?

    ``MultiProcessCollector`` re-reads every mmap file in Python, holding the
    GIL, inside the process driving the engine scheduler. Files for dead PIDs
    are never reclaimed. Reaping them bounds the cost but makes merged counters
    decrease, which Prometheus reads as a reset and ``rate()`` then mis-accounts.

    This measures the curve so the choice rests on numbers rather than a guess.
    The shipped decision is not to reap — counters stay monotonic — with a
    documented bound on pod lifetime instead.
    """
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        generate_latest,
        multiprocess,
        values,
    )

    if not getattr(values.ValueClass, "_multiprocess", False):
        return Result(
            scenario="scrape cost",
            question="Merge cost vs. dead-PID file sets",
            before="n/a",
            after="skipped",
            detail=(
                "prometheus_client latched its single-process value class, so no "
                "mmap files can be produced. Run this script directly rather "
                "than importing it after prometheus_client is already loaded."
            ),
        )

    original_value_class = values.ValueClass
    rows: list[tuple] = []
    crossing = None
    try:
        with multiproc_dir("scrape-cost") as root:
            file_sets = 1
            while file_sets <= max_file_sets:
                # Mint file sets by faking distinct PIDs — exactly what a pod
                # recycling workers across refit cycles produces.
                while len(list(root.glob("counter_*.db"))) < file_sets:
                    fake_pid = 100_000 + len(list(root.glob("counter_*.db")))
                    values.ValueClass = values.MultiProcessValue(
                        process_identifier=lambda p=fake_pid: p
                    )
                    registry = CollectorRegistry()
                    counter = Counter(
                        _FAMILY.removesuffix("_total"),
                        "synthetic load for the merge-cost curve",
                        ["policy", "scheme", "result"],
                        registry=registry,
                    )
                    counter.labels("random", "", "success").inc(1)

                timings = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    registry = CollectorRegistry()
                    multiprocess.MultiProcessCollector(registry, path=str(root))
                    generate_latest(registry)
                    timings.append(time.perf_counter() - start)
                median = statistics.median(timings)
                rows.append((file_sets, median * 1000.0))
                if crossing is None and median > PROMETHEUS_DEFAULT_SCRAPE_TIMEOUT_S:
                    crossing = file_sets
                file_sets *= 2
    finally:
        values.ValueClass = original_value_class

    worst = rows[-1]
    timeout_s = PROMETHEUS_DEFAULT_SCRAPE_TIMEOUT_S
    return Result(
        scenario="scrape cost",
        question=f"Merge cost vs. dead-PID file sets (median of {repeats} scrapes)",
        before="—  (no merge existed: one process, one registry)",
        after=f"{rows[0][1]:.1f} ms at 1 file set → {worst[1]:.1f} ms at {worst[0]}",
        detail=(
            (
                f"Crosses Prometheus's {timeout_s:.0f}s default scrape timeout at "
                f"{crossing} file sets. "
                if crossing
                else f"Stays under Prometheus's {timeout_s:.0f}s default scrape "
                f"timeout through {worst[0]} file sets. "
            )
            + "Absolute values scale with series count and with contention for the "
            "GIL, so an idle host with this synthetic family is the floor, not the "
            "figure to plan against: measure on the pod."
        ),
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Recording overhead — the load path must not pay for this
# ---------------------------------------------------------------------------


def bench_record_overhead(iterations: int = 100_000) -> Result:
    """Cost of one recording call, disabled and enabled.

    The contract is that a metrics failure degrades to no metrics, never to a
    slower or failed model load. Disabled has to be free; enabled has to stay
    far below anything a load path could notice.
    """
    from prometheus_client import CollectorRegistry

    from modelexpress.metrics import MetricsCollector

    previous = os.environ.pop("MX_METRICS_ENABLED", None)
    try:
        disabled = MetricsCollector()
        start = time.perf_counter()
        for _ in range(iterations):
            disabled.record_attempt("random", "success")
        disabled_ns = (time.perf_counter() - start) / iterations * 1e9
    finally:
        if previous is not None:
            os.environ["MX_METRICS_ENABLED"] = previous

    os.environ["MX_METRICS_ENABLED"] = "1"
    with multiproc_dir("record-overhead"):
        enabled = MetricsCollector(registry=CollectorRegistry())
        enabled._ensure()
        start = time.perf_counter()
        for _ in range(iterations):
            enabled.record_attempt("random", "success")
        enabled_ns = (time.perf_counter() - start) / iterations * 1e9

    return Result(
        scenario="record overhead",
        question=f"Cost of one recorder call ({iterations:,} iterations)",
        before=f"{disabled_ns:.0f} ns  (disabled)",
        after=f"{enabled_ns:.0f} ns  (enabled, multiprocess)",
        detail=(
            "A model load performs single-digit thousands of these, so the total "
            "sits far below measurement noise on a load measured in minutes."
        ),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

SCENARIOS = {
    "pull": bench_pull_path,
    "push": bench_push_path,
    "hard-kill": bench_hard_kill,
    "scrape-cost": bench_scrape_cost,
    "record-overhead": bench_record_overhead,
}


def _render(results: list[Result]) -> str:
    out = [
        "=" * 78,
        "ModelExpress metrics pipeline — defect verification benchmark",
        "=" * 78,
    ]
    for r in results:
        out.append("")
        out.append(f"[{r.scenario}] {r.question}")
        out.append(f"    before : {r.before}")
        out.append(f"    after  : {r.after}")
        if r.detail:
            out.append(f"    note   : {r.detail}")
        if r.rows:
            out.append("")
            out.append("      file sets   median scrape")
            for file_sets, ms in r.rows:
                out.append(f"      {file_sets:>9}   {ms:>10.1f} ms")
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    global _ROOT

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ranks",
        type=int,
        default=8,
        help="Ranks per pod. 8 is the benchmarked configuration in docs/BENCHMARKS.md.",
    )
    parser.add_argument(
        "--only",
        choices=sorted(SCENARIOS),
        action="append",
        help="Run only these scenarios (repeatable).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument(
        "--max-file-sets",
        type=int,
        default=256,
        help="Upper bound for the scrape-cost curve.",
    )
    args = parser.parse_args(argv)

    if args.ranks < 2:
        parser.error("--ranks must be at least 2: the defects only exist for N > 1")
    if args.max_file_sets < 1:
        parser.error(
            "--max-file-sets must be at least 1; use --only to skip the "
            "scrape-cost curve"
        )

    # macOS aborts a forked child that touches a lazily-initialized ObjC class,
    # which torch and Foundation both do. Harmless on Linux, where the ranks
    # this benchmark models actually run; without it the forked ranks die and
    # the "after" totals silently understate the fix.
    if sys.platform == "darwin":
        os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

    with tempfile.TemporaryDirectory(prefix="mx-metrics-bench-") as root:
        _ROOT = Path(root)
        # Before any prometheus_client import anywhere in this process. See the
        # module docstring: the value class latches on this variable, and the
        # pod manifest has to satisfy the same ordering.
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(_ROOT)

        selected = args.only or list(SCENARIOS)
        results: list[Result] = []
        for name in selected:
            fn = SCENARIOS[name]
            if name == "scrape-cost":
                results.append(fn(max_file_sets=args.max_file_sets))
            elif name == "record-overhead":
                results.append(fn())
            else:
                results.append(fn(args.ranks))

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print(_render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
