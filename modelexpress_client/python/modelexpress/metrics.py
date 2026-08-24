# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Prometheus metrics for the ModelExpress client.

Disabled by default; enable with ``MX_METRICS_ENABLED=1``. The client is a
library inside someone else's engine process, so it stays opt-in — server
metrics default on, these do not.

Two properties are load-bearing and must survive any change here:

* **Recording never raises into the load path.** Every public recorder swallows
  everything. A metrics failure degrades to no metrics, never to a failed model
  load.
* **Structured logs remain the source of truth.** Disabling metrics loses
  comparability, not correctness.

Exposition is one endpoint per **pod**, not per rank. A ModelExpress pod runs one
worker process per GPU, so this module uses ``prometheus_client`` multiprocess
mode: each rank mmaps per-PID files into a pod-local directory, and one process
binds the port and serves the ``MultiProcessCollector`` union of every rank,
including ranks that have already exited. Losing the bind is the *normal* case
rather than a failure, and mmap writes are durable at increment time, so no exit
hook is in the pull path.

``docs/METRICS.md`` carries the operator-facing half: the deployment shape
(memory-backed ``emptyDir``, the two pod-manifest environment variables, the
entrypoint ``--reset``), the four failure modes this replaced, and the
scrape-cost and shared-directory trade-offs. Every rule that fails *silently*
when broken is also stated at the code site that depends on it — those comments
are the point, not decoration.
"""

from __future__ import annotations

import atexit
import errno
import logging
import os
import threading

from . import envs

logger = logging.getLogger("modelexpress.metrics")

# Env-var name kept for callers/tests; values are read via ``envs``.
ENV_ENABLED = "MX_METRICS_ENABLED"

#: Bucket boundaries for the candidate-count funnel. Starts at 0 because zero is
#: a meaningful observation: it is what distinguishes "no peers published" from
#: "peers listed but every one filtered out".
_CANDIDATE_BUCKETS = (0, 1, 2, 4, 8, 16, 32, 64, 128)

#: ``mx_p2p_list_sources_total`` outcomes. A closed three-value enum, so the
#: family is bounded no matter how the backend misbehaves.
LIST_SOURCES_RESULTS = ("ok", "empty", "error")

# Classified at the assignment sites in nixl_transfer, never at the consumer: the
# underlying field is a formatted free-text string, so classifying it later would
# mean parsing prose into a label and the domain would grow with the wording.
NIXL_ERROR_KINDS = ("timeout", "status_error")

# What a receive actually moved. `partial` is the dangerous one: the transfer
# reports success while some locally registered tensors keep their dummy values.
# `rejected` is the strict-mode refusal, which raises rather than returning; it is
# counted so the family is a complete partition of receives rather than of
# successful ones.
NIXL_RECEIVE_RESULTS = ("complete", "partial", "empty", "rejected")


def _enabled() -> bool:
    return envs.MX_METRICS_ENABLED


def _multiproc_dir() -> str | None:
    """The configured multiprocess directory, or ``None`` for single-process mode."""
    value = envs.PROMETHEUS_MULTIPROC_DIR
    return value.strip() or None if value else None


def _package_version() -> str:
    """Installed ``modelexpress`` version, or ``unknown`` if it cannot be read."""
    try:
        from importlib.metadata import version

        return version("modelexpress")
    except Exception:
        return "unknown"


def reset_multiproc_dir(path: str | None = None) -> None:
    """Delete stale ``.db`` files from the multiprocess directory.

    **Container entrypoint only.** An ``emptyDir`` survives container restarts
    within a pod, so files from the previous run would otherwise be merged into
    this run's exposition. Calling this from worker code instead unlinks the
    mmapped files of ranks that started earlier; those ranks keep writing to an
    unlinked inode and disappear from every subsequent scrape with no error
    anywhere.

    Also reachable as ``python -m modelexpress.metrics --reset``.
    """
    directory = path or _multiproc_dir()
    if not directory:
        return
    try:
        os.makedirs(directory, exist_ok=True)
        removed = 0
        for name in os.listdir(directory):
            if not name.endswith(".db"):
                continue
            try:
                os.unlink(os.path.join(directory, name))
                removed += 1
            except OSError as e:
                logger.warning("Failed to remove stale metrics file %s: %s", name, e)
        logger.info("Reset metrics directory %s (%d stale file(s))", directory, removed)
    except OSError as e:
        logger.warning("Failed to reset metrics directory %s: %s", directory, e)


class MetricsCollector:
    """Lazy holder for prometheus_client collectors.

    Construction is attempted once. Any import or registration failure disables
    the layer permanently with a warning. Metric families are grouped by feature
    so a new group slots in alongside the P2P source-selection group without
    touching the exposition plumbing.
    """

    def __init__(self, registry=None) -> None:
        #: Registry the families are constructed in. ``None`` means
        #: prometheus_client's default. Injectable so a test can build the real
        #: families without polluting the process-global registry — under
        #: multiprocess mode the choice is cosmetic anyway, because values live
        #: in the mmap files and exposition goes through MultiProcessCollector.
        self._registry = registry
        self._ready = False
        self._init_attempted = False
        self._server_started = False
        self._push_registered = False
        self._bind_owner = False
        # Real value is latched in _build_families, which runs lazily -- this
        # singleton is constructed at import, before any deployment env is read.
        self._source_id_label = False
        self._retry_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.scheme = envs.MX_METRICS_SCHEME

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _ensure(self) -> bool:
        """Build the families and start exposition, at most once.

        Still reachable from every recorder, so a caller that never invoked
        :func:`enable_metrics` is not silently unrecorded. :func:`enable_metrics` is the
        preferred entry point because it also runs on loads that record nothing.
        """
        with self._lock:
            if self._ready:
                return True
            if self._init_attempted:
                return False
            self._init_attempted = True
            if not _enabled():
                return False
            try:
                self._build_families()
                # Now that a metric has been written, whether multiprocess mode
                # is really active is observable on disk.
                multiproc_dir = _multiproc_dir()
                if multiproc_dir:
                    _warn_if_multiprocess_inactive(multiproc_dir)
                self._ready = True
                logger.info(
                    "ModelExpress metrics enabled (scheme=%r, multiprocess=%s)",
                    self.scheme,
                    bool(_multiproc_dir()),
                )
            except Exception as e:
                logger.warning("Failed to initialize metrics, disabling: %s", e)
                self._ready = False
                return False

        # Exposition is started outside the lock: binding a socket and spawning
        # the retry thread must not be able to deadlock a recording call.
        #
        # And it is guarded, because this runs on the recorder path. _try_bind's
        # own try covers only start_http_server; the code that runs *inside* its
        # except OSError handler -- _start_bind_retry's Thread.start(), which can
        # raise RuntimeError under thread or FD exhaustion on a large-TP worker
        # -- is not covered by a sibling except. Without this, a recorder called
        # before enable_metrics() would propagate that into the model load.
        try:
            self._start_exposition()
        except Exception as e:
            logger.warning("Failed to start metrics exposition: %s", e)
        return True

    def _build_families(self) -> None:
        from prometheus_client import REGISTRY, Counter, Histogram

        registry = REGISTRY if self._registry is None else self._registry

        # --- Process constants ---
        # A Gauge set to 1, never an Info: under multiprocess mode an Info
        # exports nothing and raises nothing. This is also the exporter's proof
        # of life -- a scrape that returns it proves the endpoint came up, even
        # on a run that recorded no load events at all.
        self.build_info = _make_gauge(
            registry,
            "mx_build_info",
            "Build and deployment constants for this process; always 1.",
            ["component", "version", "scheme"],
        )
        self.build_info.labels("client", _package_version(), self.scheme).set(1)

        # --- P2P source-selection group ---
        # ``source_worker_id`` is absent by default. It is uuid4().hex[:8],
        # minted fresh per process, so its domain is bounded by process count
        # over time rather than by cluster size: unbounded series growth on any
        # long-lived Prometheus watching a fleet with pod churn.
        #
        # MX_METRICS_SOURCE_ID_LABEL=1 restores it for benchmark runs, where the
        # Prometheus is ephemeral and the question being asked -- is one peer
        # picked disproportionately under this selection policy? -- needs the
        # per-peer breakdown. Latched here at construction rather than read per
        # call, because a family's label set is fixed once it exists and a
        # mid-process env change would make every later `.labels()` call raise.
        self._source_id_label = envs.MX_METRICS_SOURCE_ID_LABEL
        selection_labels = ["policy", "scheme"]
        if self._source_id_label:
            selection_labels.append("source_worker_id")
            logger.warning(
                "MX_METRICS_SOURCE_ID_LABEL=1: mx_p2p_source_selections_total "
                "carries source_worker_id, whose domain grows with process "
                "count over time rather than with cluster size. Benchmark runs "
                "only -- never a long-lived production Prometheus."
            )
        self.selections = Counter(
            "mx_p2p_source_selections_total",
            "Source workers chosen, by selection policy.",
            selection_labels,
            registry=registry,
        )
        self.attempts = Counter(
            "mx_p2p_source_attempts_total",
            "Source attempts by result.",
            # success|metadata_miss|transfer_retry|transfer_fallback
            ["policy", "scheme", "result"],
            registry=registry,
        )
        self.metadata_failures = Counter(
            "mx_p2p_metadata_lookup_failures_total",
            "Metadata lookup failures during source selection.",
            ["policy", "scheme"],
            registry=registry,
        )
        self.list_sources = Counter(
            "mx_p2p_list_sources_total",
            "ListSources outcomes: ok|empty|error. Separates a backend outage "
            "from a healthy cluster that has published no peers yet.",
            ["policy", "scheme", "result"],
            registry=registry,
        )
        self.nixl_errors = Counter(
            "mx_nixl_data_plane_errors_total",
            "NIXL data-plane failures by classified kind: timeout|status_error. "
            "A non-zero rate is what demotes an agent from READY.",
            ["scheme", "kind"],
            registry=registry,
        )
        self.nixl_receives = Counter(
            "mx_nixl_receive_total",
            "Receive outcomes: complete|partial|empty|rejected. `partial` and "
            "`empty` both return success today, so they are otherwise "
            "indistinguishable from a healthy transfer; `rejected` is the "
            "strict-mode refusal, which raises instead of returning.",
            ["scheme", "result"],
            registry=registry,
        )
        self.candidates = Histogram(
            "mx_p2p_candidates",
            "Candidate count at a selection stage.",
            ["policy", "scheme", "stage"],  # listed|rank_matched|accelerator_matched
            buckets=_CANDIDATE_BUCKETS,
            registry=registry,
        )
        self.selection_seconds = Histogram(
            "mx_p2p_source_selection_seconds",
            "Selection (ordering) overhead in seconds.",
            ["policy", "scheme"],
            buckets=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
            registry=registry,
        )
        self.transfer_seconds = Histogram(
            "mx_p2p_transfer_seconds",
            "End-to-end transfer time in seconds.",
            ["policy", "scheme", "outcome"],  # success|retry|fallback
            buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300),
            registry=registry,
        )

    # ------------------------------------------------------------------
    # Exposition
    # ------------------------------------------------------------------

    def _exposition_registry(self):
        """Registry to serve or push: the merged multiprocess union when active."""
        multiproc_dir = _multiproc_dir()
        if multiproc_dir:
            from prometheus_client import CollectorRegistry, multiprocess

            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry, path=multiproc_dir)
            return registry
        from prometheus_client import REGISTRY

        return REGISTRY if self._registry is None else self._registry

    def _start_exposition(self) -> None:
        """Start the pull endpoint and/or arm the push, exactly once."""
        with self._lock:
            if self._server_started:
                return
            self._server_started = True

        # Parse once, then branch on the parsed value. Branching on the raw
        # string would make "0" -- this module's own documented disable value --
        # and any unparseable junk count as "a scrape endpoint is configured",
        # so a batch pod that turns the endpoint off with 0 and relies on the
        # push would get neither: the exclusion below would drop the push, and
        # the bind below would be skipped.
        port = _configured_port()
        gateway = envs.MX_METRICS_PUSHGATEWAY

        # Push and scrape are mutually exclusive, enforced here rather than
        # documented: running both double-counts every series.
        if port and gateway:
            logger.error(
                "MX_METRICS_PORT and MX_METRICS_PUSHGATEWAY are both set; "
                "running both double-counts every series. Serving the scrape "
                "endpoint and disabling the push."
            )
            gateway = None

        if port:
            self._try_bind(port)
        if gateway and not self._push_registered:
            self._push_registered = True
            # Best-effort only: atexit does not run on SIGKILL or an OOM-kill.
            # That is why the push is an opt-in escape hatch and the pull path
            # carries no exit hook at all.
            atexit.register(push_metrics_if_enabled)

    def _try_bind(self, port: int | None) -> None:
        """Attempt the /metrics bind; arm a retry thread if another rank holds it."""
        if port is None:
            return
        try:
            from prometheus_client import start_http_server

            start_http_server(port, registry=self._exposition_registry())
        except OSError as e:
            # The errno matters, not just whether a shared directory is set.
            # "Another rank got here first" is EADDRINUSE and nothing else: a
            # port the container may not bind (EACCES on <1024 without
            # CAP_NET_BIND_SERVICE) or an address that does not exist fails the
            # same way on *every* rank, so no rank ever serves and a retry loop
            # never succeeds. Reporting those as "another rank owns it" is the
            # reassuring-but-false log that this whole module exists to remove.
            if e.errno == errno.EADDRINUSE and _multiproc_dir():
                # The normal case: another rank in this pod owns the endpoint and
                # its MultiProcessCollector already serves this rank's files.
                logger.info(
                    "Metrics port %s is held by another rank in this pod (%s); "
                    "its endpoint serves this rank's metrics too.",
                    port,
                    e,
                )
                self._start_bind_retry(port)
            elif e.errno == errno.EADDRINUSE:
                logger.warning(
                    "Failed to bind metrics port %s (%s). Without "
                    "PROMETHEUS_MULTIPROC_DIR this process's metrics are not "
                    "exposed anywhere.",
                    port,
                    e,
                )
            else:
                logger.error(
                    "Metrics port %s cannot be bound (%s). This is not a lost "
                    "race with another rank -- it will fail identically on every "
                    "rank and no retry can fix it, so this pod will never be "
                    "scrapeable. Check the port number and the container's "
                    "capabilities.",
                    port,
                    e,
                )
            return
        except Exception as e:
            logger.warning("Failed to start metrics server on port %s: %s", port, e)
            return
        self._bind_owner = True
        logger.info("Metrics /metrics endpoint listening on :%s", port)

    def _start_bind_retry(self, port: int) -> None:
        """Re-attempt the bind periodically so endpoint ownership can migrate.

        Without this, a pod whose bind-winning rank exits first keeps running
        with no endpoint at all, and the scrape target goes down while the pod
        is healthy.
        """
        if self._retry_thread is not None:
            return
        interval = max(1.0, float(envs.MX_METRICS_BIND_RETRY_SECS))

        def _retry() -> None:
            while not self._bind_owner:
                try:
                    threading.Event().wait(interval)
                    from prometheus_client import start_http_server

                    start_http_server(port, registry=self._exposition_registry())
                except OSError:
                    continue  # Still held by another rank; keep waiting.
                except Exception as e:
                    logger.debug("Metrics bind retry gave up: %s", e)
                    return
                else:
                    self._bind_owner = True
                    logger.info(
                        "Took ownership of the metrics endpoint on :%s "
                        "(previous owner exited)",
                        port,
                    )
                    return

        thread = threading.Thread(
            target=_retry, name="mx-metrics-bind-retry", daemon=True
        )
        self._retry_thread = thread
        thread.start()

    # ------------------------------------------------------------------
    # P2P source-selection recording API (all no-op when disabled)
    # ------------------------------------------------------------------

    def record_selection(self, policy: str, source_worker_id: str | None = None) -> None:
        """Record that a source peer was chosen.

        ``source_worker_id`` is ignored unless ``MX_METRICS_SOURCE_ID_LABEL=1``,
        so the call site stays unconditional and the cardinality decision lives
        in one place.
        """
        if self._ensure():
            try:
                if self._source_id_label:
                    self.selections.labels(
                        policy, self.scheme, source_worker_id or "unknown"
                    ).inc()
                else:
                    self.selections.labels(policy, self.scheme).inc()
            except Exception:
                pass

    def record_attempt(self, policy: str, result: str) -> None:
        if self._ensure():
            try:
                self.attempts.labels(policy, self.scheme, result).inc()
            except Exception:
                pass

    def record_metadata_failure(self, policy: str) -> None:
        if self._ensure():
            try:
                self.metadata_failures.labels(policy, self.scheme).inc()
            except Exception:
                pass

    def record_list_sources(self, policy: str, result: str) -> None:
        """Record a ListSources outcome: ``ok``, ``empty`` or ``error``.

        Unrecognized values are clamped to ``error`` so the label domain stays a
        closed enum.
        """
        if self._ensure():
            try:
                if result not in LIST_SOURCES_RESULTS:
                    result = "error"
                self.list_sources.labels(policy, self.scheme, result).inc()
            except Exception:
                pass

    def record_nixl_error(self, kind: str) -> None:
        """Record a NIXL data-plane failure by classified kind.

        Unrecognized values clamp to ``status_error`` so the label stays a closed
        enum; the free-text message stays in the log, never in a label.
        """
        if self._ensure():
            try:
                if kind not in NIXL_ERROR_KINDS:
                    kind = "status_error"
                self.nixl_errors.labels(self.scheme, kind).inc()
            except Exception:
                pass

    def record_nixl_receive(self, result: str) -> None:
        """Record the outcome of one receive.

        ``complete``, ``partial`` and ``empty`` all return success to the caller,
        so without this a transfer that moved nothing and one that moved
        everything are the same observation. ``rejected`` is the strict-mode
        refusal, which raises instead of returning; counting it makes the family
        a partition of *every* receive rather than only the ones that returned.
        """
        if self._ensure():
            try:
                if result not in NIXL_RECEIVE_RESULTS:
                    result = "empty"
                self.nixl_receives.labels(self.scheme, result).inc()
            except Exception:
                pass

    def observe_candidates(self, policy: str, stage: str, count: int) -> None:
        if self._ensure():
            try:
                self.candidates.labels(policy, self.scheme, stage).observe(count)
            except Exception:
                pass

    def observe_selection_seconds(self, policy: str, seconds: float) -> None:
        if self._ensure():
            try:
                self.selection_seconds.labels(policy, self.scheme).observe(seconds)
            except Exception:
                pass

    def observe_transfer_seconds(self, policy: str, outcome: str, seconds: float) -> None:
        if self._ensure():
            try:
                self.transfer_seconds.labels(policy, self.scheme, outcome).observe(seconds)
            except Exception:
                pass


metrics = MetricsCollector()


def _configured_port() -> int | None:
    """The scrape port, parsed — ``None`` when there is no usable endpoint.

    Every decision that depends on "is a scrape endpoint configured?" must go
    through this, never through the raw ``MX_METRICS_PORT`` string. ``"0"`` and
    junk are both truthy strings but mean *no endpoint*, so a raw-string test
    reads them as "yes" and then binds nothing. ``0`` is the documented disable
    value, matching the server's ``--metrics-port 0``; passing it through would
    have the OS hand out an ephemeral port that nothing could ever be configured
    to scrape — an endpoint that exists and is unreachable, which is the failure
    mode this whole module exists to eliminate.
    """
    value = envs.MX_METRICS_PORT
    if value is None:
        return None
    try:
        port = int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid MX_METRICS_PORT=%r; not serving /metrics", value)
        return None
    if port <= 0:
        logger.info("MX_METRICS_PORT=%s disables the /metrics endpoint", port)
        return None
    return port


def _make_gauge(registry, name: str, documentation: str, labels: list[str]):
    """Build a Gauge with a multiprocess mode that is safe under hard kills.

    ``mostrecent`` is correct for process constants and carries no ``pid`` label.
    The default, ``all``, appends ``pid`` and is unbounded; ``livesum`` and
    ``liveall`` depend on ``mark_process_dead``, which only unlinks gauge files
    on a clean exit and so wedges a stale value forever when a rank is
    OOM-killed. ``max`` is the fallback for a prometheus_client too old to know
    ``mostrecent``; every process writes the same 1, so the max is also 1.
    """
    from prometheus_client import Gauge

    for mode in ("mostrecent", "max"):
        try:
            return Gauge(
                name, documentation, labels, registry=registry, multiprocess_mode=mode
            )
        except (ValueError, TypeError):
            continue
    return Gauge(name, documentation, labels, registry=registry)


def enable_metrics() -> bool:
    """Initialize the collector and start exposition. Idempotent; never raises.

    Call this **unconditionally** from every engine loader's constructor. That
    is the whole point: initialization used
    to be reachable only from a recording call, so a run that skipped P2P and
    fell back to a local or HuggingFace path started no endpoint, pushed nothing,
    and produced output byte-identical to ``MX_METRICS_ENABLED=0``. A scrape must
    prove the exporter came up even when nothing was recorded.

    Returns whether metrics are active, for logging only. Callers must not branch
    on it: every recorder is already a no-op when this returns ``False``.
    """
    if not _enabled():
        return False
    try:
        multiproc_dir = _multiproc_dir()
        if multiproc_dir:
            _check_multiproc_dir(multiproc_dir)
        return metrics._ensure()
    except Exception as e:  # noqa: BLE001 - enable_metrics() may never raise into a load
        logger.warning("Failed to enable metrics: %s", e)
        return False


def _check_multiproc_dir(multiproc_dir: str) -> None:
    """Make sure the multiprocess directory exists and is usable.

    Pre-flight only. Whether multiprocess mode is *actually active* cannot be
    known until a metric has been written; that is
    :func:`_warn_if_multiprocess_inactive`, called after the families are built.
    """
    try:
        # makedirs only, never unlink: wiping from worker code unlinks an
        # earlier rank's mmapped files and that rank vanishes from every later
        # scrape. Entrypoint only -- see reset_multiproc_dir.
        os.makedirs(multiproc_dir, exist_ok=True)
    except OSError as e:
        logger.error(
            "PROMETHEUS_MULTIPROC_DIR=%s is not usable (%s); metrics will be "
            "empty. Mount it as an emptyDir in the pod manifest.",
            multiproc_dir,
            e,
        )


def _warn_if_multiprocess_inactive(multiproc_dir: str) -> None:
    """Detect the latched-single-process case, from what is on disk.

    ``prometheus_client.values.get_value_class()`` latches at module import. If
    ``PROMETHEUS_MULTIPROC_DIR`` was assigned after the engine imported
    prometheus_client, metrics are created in single-process mode: no ``.db``
    files, a merged endpoint that returns 200 with nothing in it, and no error
    anywhere.

    Checked by looking for the files rather than by reading
    ``values.ValueClass._multiprocess``. That attribute is private and the
    dependency has no upper version bound, so a rename upstream would turn this
    into a false alarm on healthy pods -- and a warning that cries wolf is worse
    than none, because this is the message that explains an empty endpoint. The
    files are the thing that actually matters, and they are observable.

    Called after the families are built, since that is the write that creates
    them.
    """
    try:
        if any(os.scandir(multiproc_dir)):
            return
    except OSError:
        # Already reported by _check_multiproc_dir; nothing to add.
        return
    logger.error(
        "PROMETHEUS_MULTIPROC_DIR=%s is set but no .db files were written, so "
        "prometheus_client is not in multiprocess mode and the merged endpoint "
        "will be empty. get_value_class() latches at import, so the variable "
        "must be set in the pod manifest before the process starts -- assigning "
        "it from Python lands after the engine has already imported "
        "prometheus_client.",
        multiproc_dir,
    )


def push_metrics_if_enabled(job: str = "modelexpress") -> None:
    """Push to ``MX_METRICS_PUSHGATEWAY``, if configured. One push per **pod**.

    An opt-in escape hatch for pure-batch pods where every process exits before
    a scrape. Prefer the scrape endpoint: the Pushgateway has no TTL, so series
    are orphaned forever after a pod dies, and it hides counter resets from
    ``rate()``.

    The grouping key is the pod, not the host. ``push_to_gateway`` is a ``PUT``
    that replaces the entire group, and every rank on a node shares a hostname,
    so the previous host-keyed implementation had each rank wipe the one before
    it — whichever exited last was the only survivor, and the result looked
    complete. Under multiprocess mode the pushed registry is the merged union of
    every rank, so one push per pod is the complete picture and concurrent
    pushes from sibling ranks carry the same payload instead of erasing data.
    """
    if not _enabled():
        return
    gateway = envs.MX_METRICS_PUSHGATEWAY
    if not gateway:
        return
    # Parsed, not the raw string: "0" and junk mean there is no endpoint, so
    # they must not suppress the push. See _configured_port.
    if _configured_port():
        logger.debug("Scrape endpoint is configured; skipping the push.")
        return
    try:
        import socket

        from prometheus_client import push_to_gateway

        # Pod-scoped identity from the downward API, falling back to the
        # hostname (which is the pod name inside a Kubernetes pod anyway).
        pod = envs.POD_UID or envs.POD_NAME or socket.gethostname()
        grouping_key = {"pod": pod}
        push_to_gateway(
            gateway,
            job=job,
            grouping_key=grouping_key,
            registry=metrics._exposition_registry(),
        )
        logger.info("Pushed metrics to %s (job=%s, %s)", gateway, job, grouping_key)
    except Exception as e:
        logger.warning("Failed to push metrics to %s: %s", gateway, e)


def _main(argv: list[str] | None = None) -> int:
    """``python -m modelexpress.metrics --reset`` — container entrypoint helper."""
    import argparse

    parser = argparse.ArgumentParser(prog="modelexpress.metrics")
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete stale .db files from PROMETHEUS_MULTIPROC_DIR. Run this from "
            "the container entrypoint, never from worker code: an emptyDir "
            "survives container restarts within a pod, but wiping it after "
            "ranks have started unlinks their mmapped files."
        ),
    )
    parser.add_argument("--dir", default=None, help="Override PROMETHEUS_MULTIPROC_DIR.")
    args = parser.parse_args(argv)
    if args.reset:
        logging.basicConfig(level=logging.INFO)
        reset_multiproc_dir(args.dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
