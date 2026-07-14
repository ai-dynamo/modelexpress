# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Prometheus metrics for the ModelExpress client.

A single, generic collector. Disabled by default; enable with
``MX_METRICS_ENABLED=1``. It is built to grow: the P2P source-selection
counters/histograms below are the first group, and other client signals (model
load latency, P2P transfer latency, etc.) can be added to the same collector as
additional families without changing the exposition or enable/disable plumbing.

Exposition:
  - Pull: a worker that keeps running (e.g. a target that became a source) can
    serve a ``/metrics`` endpoint on ``MX_METRICS_PORT``.
  - Push: short-lived load-only processes can push to a Pushgateway via
    ``MX_METRICS_PUSHGATEWAY``.

Every metric carries a ``scheme`` run label from ``MX_METRICS_SCHEME`` (plus
group-specific labels) so multiple runs compare on one dashboard. All public
functions are no-ops when metrics are disabled or prometheus_client is
unavailable; nothing here may raise into the load path.
"""

from __future__ import annotations

import atexit
import logging

from . import envs

logger = logging.getLogger("modelexpress.metrics")

# Env-var name kept for callers/tests; values are read via ``envs``.
ENV_ENABLED = "MX_METRICS_ENABLED"


def _enabled() -> bool:
    return envs.MX_METRICS_ENABLED


class MetricsCollector:
    """Lazy holder for prometheus_client collectors.

    Construction is attempted once, on first use, only when enabled. Any import
    or registration failure disables the layer permanently with a warning. The
    metric families are grouped by feature so new groups slot in alongside the
    P2P source-selection group without touching the rest.
    """

    def __init__(self) -> None:
        self._ready = False
        self._init_attempted = False
        self._server_started = False
        self._atexit_registered = False
        self.scheme = envs.MX_METRICS_SCHEME

    def _ensure(self) -> bool:
        if self._ready:
            return True
        if self._init_attempted:
            return False
        self._init_attempted = True
        if not _enabled():
            return False
        try:
            from prometheus_client import Counter, Gauge, Histogram

            # --- P2P source-selection group ---
            self.selections = Counter(
                "mx_p2p_source_selections_total",
                "How often each source worker is chosen (utilization balance).",
                ["policy", "scheme", "source_worker_id"],
            )
            self.attempts = Counter(
                "mx_p2p_source_attempts_total",
                "Source attempts by result.",
                # success|metadata_miss|transfer_retry|transfer_fallback
                ["policy", "scheme", "result"],
            )
            self.metadata_failures = Counter(
                "mx_p2p_metadata_lookup_failures_total",
                "Metadata lookup failures during source selection.",
                ["policy", "scheme"],
            )
            self.candidates = Histogram(
                "mx_p2p_candidates",
                "Candidate count at a selection stage.",
                ["policy", "scheme", "stage"],  # listed|rank_matched|accelerator_matched
                buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128),
            )
            self.selection_seconds = Histogram(
                "mx_p2p_source_selection_seconds",
                "Selection (ordering) overhead in seconds.",
                ["policy", "scheme"],
                buckets=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
            )
            self.transfer_seconds = Histogram(
                "mx_p2p_transfer_seconds",
                "End-to-end transfer time in seconds.",
                ["policy", "scheme", "outcome"],  # success|retry|fallback
                buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300),
            )
            self.active_transfers = Gauge(
                "mx_p2p_active_transfers",
                "Server-estimated concurrent transfers per source, as observed by "
                "clients during selection (the signal the load_aware policy ranks on).",
                ["scheme", "source_worker_id"],
            )
            self._ready = True
            logger.info("ModelExpress metrics enabled (scheme=%r)", self.scheme)
        except Exception as e:
            logger.warning("Failed to initialize metrics, disabling: %s", e)
            self._ready = False
        if self._ready and not self._atexit_registered:
            self._atexit_registered = True
            # Auto-expose the pull endpoint so enabling metrics is enough; the
            # worker keeps running after load and serves /metrics for scraping.
            self._start_pull_server_once()
            # Short-lived load-only processes exit before a scrape; flush to the
            # Pushgateway on exit if one is configured.
            atexit.register(push_metrics_if_enabled)
        return self._ready

    def _start_pull_server_once(self) -> None:
        if self._server_started:
            return
        self._server_started = True
        port = envs.MX_METRICS_PORT
        if not port:
            return
        try:
            from prometheus_client import start_http_server

            start_http_server(int(port))
            logger.info("Metrics /metrics endpoint listening on :%s", port)
        except Exception as e:
            logger.warning("Failed to start metrics server on port %s: %s", port, e)

    # -- P2P source-selection recording API (all no-op when disabled) --

    def record_selection(self, policy: str, source_worker_id: str) -> None:
        if self._ensure():
            try:
                self.selections.labels(policy, self.scheme, source_worker_id).inc()
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

    def observe_transfer_seconds(
        self, policy: str, outcome: str, seconds: float
    ) -> None:
        if self._ensure():
            try:
                self.transfer_seconds.labels(policy, self.scheme, outcome).observe(
                    seconds
                )
            except Exception:
                pass

    def set_active_transfers(self, source_worker_id: str, value: float) -> None:
        if self._ensure():
            try:
                self.active_transfers.labels(self.scheme, source_worker_id).set(value)
            except Exception:
                pass


metrics = MetricsCollector()


def push_metrics_if_enabled(job: str = "modelexpress") -> None:
    """Push current metrics to MX_METRICS_PUSHGATEWAY, if configured.

    For short-lived load-only processes that exit before a scrape. No-op when
    disabled or no pushgateway is set.
    """
    if not _enabled():
        return
    gateway = envs.MX_METRICS_PUSHGATEWAY
    if not gateway:
        return
    try:
        import socket

        from prometheus_client import REGISTRY, push_to_gateway

        # push_to_gateway (PUT) replaces the whole group for a job+grouping_key.
        # Key by host so concurrent workers don't overwrite each other's metrics.
        grouping_key = {"instance": socket.gethostname()}
        push_to_gateway(gateway, job=job, grouping_key=grouping_key, registry=REGISTRY)
        logger.info("Pushed metrics to %s (job=%s, %s)", gateway, job, grouping_key)
    except Exception as e:
        logger.warning("Failed to push metrics to %s: %s", gateway, e)
