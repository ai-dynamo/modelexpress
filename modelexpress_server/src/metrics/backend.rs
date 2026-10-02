// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metrics for the storage backends behind the server.
//!
//! Three unrelated traits sit between the server and its storage --
//! `MetadataBackend` (P2P source metadata), `RegistryBackend` (model registry)
//! and `RefitBackend` (weight versions) -- each with its own error type and no
//! common supertrait. They are instrumented by decorators that implement the
//! same trait and delegate, so the concrete Redis, Kubernetes and in-memory
//! implementations are untouched and a new backend is instrumented by
//! construction rather than by remembering to add timing code.
//!
//! `store` names the **subsystem**, not the storage engine. Only one engine is
//! live per pod, so `store="redis"` would be a constant; the engine is already
//! carried by `mx_build_info{backend=...}` and joins on from there. Naming the
//! subsystem also keeps `op="connect"` unambiguous, which it would not be if two
//! subsystems both reported it against the same engine.

use std::future::Future;
use std::time::Instant;

use prometheus_client::encoding::{EncodeLabelSet, EncodeLabelValue, LabelValueEncoder};
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

use super::buckets;

/// Which storage subsystem performed the operation.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Store {
    /// P2P source metadata (`MetadataBackend`).
    P2p,
    /// Model registry (`RegistryBackend`).
    Registry,
    /// Weight-version store for RL refit (`RefitBackend`).
    Refit,
}

impl Store {
    const fn as_str(self) -> &'static str {
        match self {
            Self::P2p => "p2p",
            Self::Registry => "registry",
            Self::Refit => "refit",
        }
    }
}

// Hand-written for the same reason as `RpcOutcome`: the derive would encode the
// variant identifier, giving `P2p` rather than `p2p`.
impl EncodeLabelValue for Store {
    fn encode(&self, encoder: &mut LabelValueEncoder<'_>) -> Result<(), std::fmt::Error> {
        EncodeLabelValue::encode(&self.as_str(), encoder)
    }
}

/// Whether the operation succeeded.
///
/// The error *kind* is deliberately not a label: the three
/// backends have three unrelated error types, two of them boxed
/// `dyn Error`, so any faithful mapping would be an unbounded string.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum OpResult {
    /// The operation returned `Ok`.
    Ok,
    /// The operation returned `Err`.
    Error,
    /// The caller was dropped before the operation finished. Recorded from a
    /// drop guard, so a store that hangs is visible rather than absent.
    Cancelled,
}

impl OpResult {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Error => "error",
            Self::Cancelled => "cancelled",
        }
    }
}

impl EncodeLabelValue for OpResult {
    fn encode(&self, encoder: &mut LabelValueEncoder<'_>) -> Result<(), std::fmt::Error> {
        EncodeLabelValue::encode(&self.as_str(), encoder)
    }
}

/// Label set for the backend families.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct BackendLabels {
    /// Storage subsystem.
    pub store: Store,
    /// Trait method name. Bounded because every call site passes a literal.
    pub op: &'static str,
    /// Success or failure.
    pub result: OpResult,
}

/// Label set for in-flight tracking. No `result`: an operation still running
/// does not have one yet.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct BackendOpLabels {
    /// Storage subsystem.
    pub store: Store,
    /// Trait method name.
    pub op: &'static str,
}

fn fast_histogram() -> Histogram {
    Histogram::new(buckets::FAST)
}

type LatencyFamily = Family<BackendLabels, Histogram, fn() -> Histogram>;

/// Handles for the backend families. Cloning shares the underlying storage.
#[derive(Clone)]
pub struct BackendMetrics {
    ops: Family<BackendLabels, Counter>,
    latency: LatencyFamily,
    in_flight: Family<BackendOpLabels, Gauge>,
}

impl BackendMetrics {
    /// Register the families on `registry` and return handles to them.
    ///
    /// Registered without the `_total` suffix; the encoder appends it.
    pub fn register(registry: &mut Registry) -> Self {
        let ops = Family::<BackendLabels, Counter>::default();
        let latency: LatencyFamily =
            Family::new_with_constructor(fast_histogram as fn() -> Histogram);
        let in_flight = Family::<BackendOpLabels, Gauge>::default();

        let backend = registry.sub_registry_with_prefix("backend");
        backend.register(
            "ops",
            "Storage backend operations by subsystem, method and result",
            ops.clone(),
        );
        backend.register(
            "op_seconds",
            "Storage backend operation latency by subsystem, method and result",
            latency.clone(),
        );

        // See the note in `metrics::grpc`: a plain Gauge is safe here because the
        // server is one process with an in-process registry, so there is no
        // SIGKILLed rank whose gauge could wedge.
        backend.register(
            "ops_in_flight",
            "Storage backend operations currently running, by subsystem and method",
            in_flight.clone(),
        );

        Self {
            ops,
            latency,
            in_flight,
        }
    }

    /// Mark one operation as started.
    fn in_flight_inc(&self, store: Store, op: &'static str) {
        self.in_flight
            .get_or_create(&BackendOpLabels { store, op })
            .inc();
    }

    /// Mark one operation as finished, however it finished.
    fn in_flight_dec(&self, store: Store, op: &'static str) {
        self.in_flight
            .get_or_create(&BackendOpLabels { store, op })
            .dec();
    }

    /// Count an operation whose future was dropped before it completed.
    ///
    /// The counter only, never the histogram: a partial duration would enter the
    /// distribution as a fast sample for an operation that was in fact still
    /// running, which biases the tail in exactly the wrong direction.
    fn record_cancelled(&self, store: Store, op: &'static str) {
        self.ops
            .get_or_create(&BackendLabels {
                store,
                op,
                result: OpResult::Cancelled,
            })
            .inc();
    }

    /// Await `future`, then record its latency and result.
    ///
    /// The recording happens strictly after the await. That ordering is
    /// load-bearing: `Family::get_or_create` returns a `!Send` guard, so holding
    /// one across an await would make the caller's future non-`Send` and break
    /// the `#[async_trait]` bound with an error pointing at lock internals.
    pub async fn time<T, E, F>(&self, store: Store, op: &'static str, future: F) -> Result<T, E>
    where
        F: Future<Output = Result<T, E>>,
    {
        // A hung store is precisely what this histogram exists to reveal, and a
        // hang is precisely the case that produces no sample: whoever is waiting
        // gives up, this future is dropped, and the recording below never runs.
        // The guard turns that into a counter increment, and holds the in-flight
        // gauge so a wedged store is visible while it is still wedged rather than
        // only afterwards.
        let mut guard = InFlightGuard::enter(self.clone(), store, op);
        let started = Instant::now();
        let outcome = future.await;
        guard.pending = false;
        let labels = BackendLabels {
            store,
            op,
            result: if outcome.is_ok() {
                OpResult::Ok
            } else {
                OpResult::Error
            },
        };
        // `as_secs_f64` is float division; `Instant - Instant` would trip
        // `arithmetic_side_effects`.
        let elapsed = started.elapsed().as_secs_f64();
        self.ops.get_or_create(&labels).inc();
        self.latency.get_or_create(&labels).observe(elapsed);
        outcome
    }
}

/// Tracks one in-flight operation, and records a cancellation if the future is
/// dropped before it completed. See the equivalent in [`crate::metrics::grpc`].
struct InFlightGuard {
    metrics: BackendMetrics,
    store: Store,
    op: &'static str,
    /// Cleared once the operation resolves. Still set at drop means cancelled.
    pending: bool,
}

impl InFlightGuard {
    fn enter(metrics: BackendMetrics, store: Store, op: &'static str) -> Self {
        metrics.in_flight_inc(store, op);
        Self {
            metrics,
            store,
            op,
            pending: true,
        }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        // Runs on every exit path, so the gauge cannot drift.
        self.metrics.in_flight_dec(self.store, self.op);
        if self.pending {
            self.metrics.record_cancelled(self.store, self.op);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{encode_text, new_registry};

    #[tokio::test]
    async fn both_results_encode_under_the_documented_names() {
        let mut registry = new_registry();
        let metrics = BackendMetrics::register(&mut registry);

        let ok: Result<(), ()> = metrics
            .time(Store::P2p, "list_workers", async { Ok(()) })
            .await;
        assert!(ok.is_ok());

        let failed: Result<(), ()> = metrics
            .time(Store::Registry, "connect", async { Err(()) })
            .await;
        assert!(failed.is_err());

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));

        assert!(
            encoded
                .contains(r#"mx_backend_ops_total{store="p2p",op="list_workers",result="ok"} 1"#),
            "{encoded}"
        );
        assert!(
            encoded.contains(
                r#"mx_backend_ops_total{store="registry",op="connect",result="error"} 1"#
            ),
            "{encoded}"
        );
        assert!(
            encoded.contains("mx_backend_op_seconds_bucket"),
            "{encoded}"
        );
        assert!(!encoded.contains("_total_total"), "{encoded}");
    }

    /// A dropped operation is counted, and contributes no latency sample.
    #[tokio::test]
    async fn a_dropped_operation_is_recorded_as_cancelled() {
        let mut registry = new_registry();
        let metrics = BackendMetrics::register(&mut registry);

        // Polled, then dropped mid-flight -- the guard is created inside the
        // async body, so it exists only once the future has been polled at least
        // once. That is the real shape of a cancellation: a caller gives up on a
        // store that stopped answering, exactly as the startup timeouts in
        // `run_server` do.
        let timed_out = tokio::time::timeout(
            std::time::Duration::from_millis(10),
            metrics.time(
                Store::P2p,
                "get_metadata",
                std::future::pending::<Result<(), ()>>(),
            ),
        )
        .await;
        assert!(timed_out.is_err(), "the inner future should never resolve");

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(
                r#"mx_backend_ops_total{store="p2p",op="get_metadata",result="cancelled"} 1"#
            ),
            "{encoded}"
        );
        assert!(
            !encoded.contains(r#"mx_backend_op_seconds_count{store="p2p",op="get_metadata"#),
            "a cancellation wrote a latency sample: {encoded}"
        );
        // The guard decrements on every exit path, so a cancelled operation must
        // not leave the gauge stuck above zero -- that is the failure mode a
        // gauge has and a counter pair does not.
        assert!(
            encoded.contains(r#"mx_backend_ops_in_flight{store="p2p",op="get_metadata"} 0"#),
            "in-flight gauge did not return to zero: {encoded}"
        );
    }

    #[tokio::test]
    async fn the_inner_value_is_passed_through_untouched() {
        let mut registry = new_registry();
        let metrics = BackendMetrics::register(&mut registry);

        let value: Result<u32, ()> = metrics.time(Store::Refit, "get", async { Ok(7) }).await;
        assert_eq!(value, Ok(7));
    }
}
