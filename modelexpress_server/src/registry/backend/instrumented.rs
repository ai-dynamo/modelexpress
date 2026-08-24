// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metrics decorator for [`RegistryBackend`].
//!
//! Wraps any backend and records [`crate::metrics::backend`] families around
//! every call, leaving the Redis, Kubernetes and in-memory implementations
//! untouched. A backend added later is instrumented because it is wrapped at
//! construction, not because someone remembered to add timing code to it.
//!
//! Every method is forwarded explicitly. The trait has no default bodies today;
//! if one is added it must still be overridden here, because a default body
//! would run on the decorator, call back through the decorator's other methods,
//! and so bypass any specialised override in the concrete backend while
//! counting the call twice under two op names. That failure compiles and turns
//! no test red.

use async_trait::async_trait;
use std::sync::Arc;

use crate::metrics::backend::{BackendMetrics, Store};
use crate::metrics::registry::{ClaimResult, LeaseResult, RegistryMetrics, StatusLabel};
use crate::registry::backend::{ClaimOutcome, ModelRecord, RegistryBackend, RegistryResult};
use modelexpress_common::models::{ModelProvider, ModelStatus};

/// A [`RegistryBackend`] that records timing and outcome for each operation.
pub struct InstrumentedRegistryBackend {
    inner: Arc<dyn RegistryBackend>,
    metrics: BackendMetrics,
    lifecycle: RegistryMetrics,
}

impl InstrumentedRegistryBackend {
    /// Wrap `inner`, returning it as a trait object so call sites are unchanged.
    #[must_use]
    pub fn wrap(
        inner: Arc<dyn RegistryBackend>,
        metrics: BackendMetrics,
        lifecycle: RegistryMetrics,
    ) -> Arc<dyn RegistryBackend> {
        Arc::new(Self {
            inner,
            metrics,
            lifecycle,
        })
    }
}

#[async_trait]
impl RegistryBackend for InstrumentedRegistryBackend {
    async fn connect(&self) -> RegistryResult<()> {
        self.metrics
            .time(Store::Registry, "connect", self.inner.connect())
            .await
    }

    async fn get_status(&self, model_name: &str) -> RegistryResult<Option<ModelStatus>> {
        self.metrics
            .time(
                Store::Registry,
                "get_status",
                self.inner.get_status(model_name),
            )
            .await
    }

    async fn get_model_record(&self, model_name: &str) -> RegistryResult<Option<ModelRecord>> {
        self.metrics
            .time(
                Store::Registry,
                "get_model_record",
                self.inner.get_model_record(model_name),
            )
            .await
    }

    async fn set_status(
        &self,
        model_name: &str,
        provider: ModelProvider,
        status: ModelStatus,
        message: Option<String>,
    ) -> RegistryResult<()> {
        self.metrics
            .time(
                Store::Registry,
                "set_status",
                self.inner.set_status(model_name, provider, status, message),
            )
            .await
    }

    async fn touch_model(&self, model_name: &str) -> RegistryResult<()> {
        self.metrics
            .time(
                Store::Registry,
                "touch_model",
                self.inner.touch_model(model_name),
            )
            .await
    }

    async fn delete_model(&self, model_name: &str) -> RegistryResult<()> {
        self.metrics
            .time(
                Store::Registry,
                "delete_model",
                self.inner.delete_model(model_name),
            )
            .await
    }

    async fn get_models_by_last_used(
        &self,
        limit: Option<u32>,
    ) -> RegistryResult<Vec<ModelRecord>> {
        self.metrics
            .time(
                Store::Registry,
                "get_models_by_last_used",
                self.inner.get_models_by_last_used(limit),
            )
            .await
    }

    async fn get_status_counts(&self) -> RegistryResult<(u32, u32, u32)> {
        self.metrics
            .time(
                Store::Registry,
                "get_status_counts",
                self.inner.get_status_counts(),
            )
            .await
    }

    async fn try_claim_for_download(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        lease_duration: std::time::Duration,
    ) -> RegistryResult<ClaimOutcome> {
        let outcome = self
            .metrics
            .time(
                Store::Registry,
                "try_claim_for_download",
                self.inner
                    .try_claim_for_download(model_name, provider, claim_id, lease_duration),
            )
            .await;
        self.lifecycle.record_claim(match &outcome {
            Ok(ClaimOutcome::Claimed) => ClaimResult::Claimed,
            Ok(ClaimOutcome::TookOver) => ClaimResult::Takeover,
            Ok(ClaimOutcome::AlreadyExists(_)) => ClaimResult::AlreadyExists,
            Err(_) => ClaimResult::Error,
        });
        outcome
    }

    async fn try_reset_error_for_retry(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        lease_duration: std::time::Duration,
    ) -> RegistryResult<bool> {
        let reset = self
            .metrics
            .time(
                Store::Registry,
                "try_reset_error_for_retry",
                self.inner.try_reset_error_for_retry(
                    model_name,
                    provider,
                    claim_id,
                    lease_duration,
                ),
            )
            .await;
        // Only the winner sees `true`, so this counts retries actually started
        // rather than replicas that observed the error.
        if matches!(reset, Ok(true)) {
            self.lifecycle
                .record_transition(StatusLabel::Error, StatusLabel::Downloading);
        }
        reset
    }

    async fn refresh_download_claim(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        lease_duration: std::time::Duration,
    ) -> RegistryResult<bool> {
        let renewed = self
            .metrics
            .time(
                Store::Registry,
                "refresh_download_claim",
                self.inner
                    .refresh_download_claim(model_name, provider, claim_id, lease_duration),
            )
            .await;
        self.lifecycle.record_lease_refresh(match &renewed {
            Ok(true) => LeaseResult::Renewed,
            Ok(false) => LeaseResult::Lost,
            Err(_) => LeaseResult::Error,
        });
        renewed
    }

    async fn finish_download_claim(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        status: ModelStatus,
        message: Option<String>,
    ) -> RegistryResult<bool> {
        let finished = self
            .metrics
            .time(
                Store::Registry,
                "finish_download_claim",
                self.inner
                    .finish_download_claim(model_name, provider, claim_id, status, message),
            )
            .await;
        // `false` means a stale owner was fenced after its lease was taken over:
        // the entry did not leave `downloading` on this call, and counting it
        // would make the in-flight derivation go negative.
        if matches!(finished, Ok(true)) {
            self.lifecycle
                .record_transition(StatusLabel::Downloading, status.into());
        }
        finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{encode_text, new_registry};
    use crate::registry::backend::MockRegistryBackend;

    #[tokio::test]
    async fn a_successful_op_is_recorded_as_ok() {
        let mut mock = MockRegistryBackend::new();
        mock.expect_get_status()
            .times(1)
            .returning(|_| Ok(Some(ModelStatus::DOWNLOADED)));

        let mut registry = new_registry();
        let metrics = BackendMetrics::register(&mut registry);
        let backend = InstrumentedRegistryBackend::wrap(
            Arc::new(mock),
            metrics,
            RegistryMetrics::register(&mut new_registry()),
        );

        let status = backend.get_status("google-t5/t5-small").await;
        assert_eq!(status.ok().flatten(), Some(ModelStatus::DOWNLOADED));

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(
                r#"mx_backend_ops_total{store="registry",op="get_status",result="ok"} 1"#
            ),
            "{encoded}"
        );
    }

    #[tokio::test]
    async fn a_backend_failure_is_recorded_as_an_error() {
        let mut mock = MockRegistryBackend::new();
        mock.expect_connect()
            .times(1)
            .returning(|| Err("redis is down".into()));

        let mut registry = new_registry();
        let metrics = BackendMetrics::register(&mut registry);
        let backend = InstrumentedRegistryBackend::wrap(
            Arc::new(mock),
            metrics,
            RegistryMetrics::register(&mut new_registry()),
        );

        assert!(backend.connect().await.is_err());

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(
                r#"mx_backend_ops_total{store="registry",op="connect",result="error"} 1"#
            ),
            "{encoded}"
        );
    }

    /// Twelve near-identical forwarding bodies invite a copy-pasted op literal, so
    /// pin that each method reports under its own name and none under another's.
    #[tokio::test]
    async fn every_method_reports_under_its_own_op_name() {
        let lease = std::time::Duration::from_secs(30);
        let mut mock = MockRegistryBackend::new();
        mock.expect_connect().times(1).returning(|| Ok(()));
        mock.expect_get_status().times(1).returning(|_| Ok(None));
        mock.expect_get_model_record()
            .times(1)
            .returning(|_| Ok(None));
        mock.expect_set_status()
            .times(1)
            .returning(|_, _, _, _| Ok(()));
        mock.expect_touch_model().times(1).returning(|_| Ok(()));
        mock.expect_delete_model().times(1).returning(|_| Ok(()));
        mock.expect_get_models_by_last_used()
            .times(1)
            .returning(|_| Ok(Vec::new()));
        mock.expect_get_status_counts()
            .times(1)
            .returning(|| Ok((0, 0, 0)));
        mock.expect_try_claim_for_download()
            .times(1)
            .returning(|_, _, _, _| Ok(ClaimOutcome::Claimed));
        mock.expect_try_reset_error_for_retry()
            .times(1)
            .returning(|_, _, _, _| Ok(false));
        mock.expect_refresh_download_claim()
            .times(1)
            .returning(|_, _, _, _| Ok(true));
        mock.expect_finish_download_claim()
            .times(1)
            .returning(|_, _, _, _, _| Ok(true));

        let mut registry = new_registry();
        let metrics = BackendMetrics::register(&mut registry);
        let backend = InstrumentedRegistryBackend::wrap(
            Arc::new(mock),
            metrics,
            RegistryMetrics::register(&mut new_registry()),
        );

        let model = "google-t5/t5-small";
        let _ = backend.connect().await;
        let _ = backend.get_status(model).await;
        let _ = backend.get_model_record(model).await;
        let _ = backend
            .set_status(
                model,
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADED,
                None,
            )
            .await;
        let _ = backend.touch_model(model).await;
        let _ = backend.delete_model(model).await;
        let _ = backend.get_models_by_last_used(Some(10)).await;
        let _ = backend.get_status_counts().await;
        let _ = backend
            .try_claim_for_download(model, ModelProvider::HuggingFace, "claim-1", lease)
            .await;
        let _ = backend
            .try_reset_error_for_retry(model, ModelProvider::HuggingFace, "claim-1", lease)
            .await;
        let _ = backend
            .refresh_download_claim(model, ModelProvider::HuggingFace, "claim-1", lease)
            .await;
        let _ = backend
            .finish_download_claim(
                model,
                ModelProvider::HuggingFace,
                "claim-1",
                ModelStatus::DOWNLOADED,
                None,
            )
            .await;

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        for op in [
            "connect",
            "get_status",
            "get_model_record",
            "set_status",
            "touch_model",
            "delete_model",
            "get_models_by_last_used",
            "get_status_counts",
            "try_claim_for_download",
            "try_reset_error_for_retry",
            "refresh_download_claim",
            "finish_download_claim",
        ] {
            let expected =
                format!(r#"mx_backend_ops_total{{store="registry",op="{op}",result="ok"}} 1"#);
            assert!(encoded.contains(&expected), "missing {op}: {encoded}");
        }
    }

    // --- download lifecycle ---------------------------------------------------
    //
    // The three tests above hand `RegistryMetrics::register` a throwaway
    // `new_registry()` that is dropped on the spot. That is harmless for them --
    // they only assert on `mx_backend_ops_total` -- but it means the lifecycle
    // families are never encoded, so nothing below can be written that way. All
    // four PR-673 lifecycle counters reach Prometheus only through this
    // decorator, so these register both families on the one registry the
    // assertions actually read.

    const MODEL: &str = "google-t5/t5-small";
    const CLAIM: &str = "claim-1";

    fn lease() -> std::time::Duration {
        std::time::Duration::from_secs(30)
    }

    /// Wrap `mock` with both metric families on a single registry, returned so
    /// the caller can encode it.
    fn instrumented(
        mock: MockRegistryBackend,
    ) -> (
        Arc<dyn RegistryBackend>,
        prometheus_client::registry::Registry,
    ) {
        let mut prom = new_registry();
        let backend = InstrumentedRegistryBackend::wrap(
            Arc::new(mock),
            BackendMetrics::register(&mut prom),
            RegistryMetrics::register(&mut prom),
        );
        (backend, prom)
    }

    fn scrape(prom: &prometheus_client::registry::Registry) -> String {
        encode_text(prom).unwrap_or_else(|_| String::from("<encode failed>"))
    }

    /// Sum the transition series the way the documented flow query does, so the
    /// assertion is the derivation operators read rather than one series.
    ///
    /// Deliberately a local copy of the helper in [`crate::metrics::registry`]:
    /// that one proves the counter's arithmetic given hand-written labels, this
    /// one proves the decorator feeds it the right ones.
    fn in_flight(encoded: &str) -> i64 {
        let mut arrivals: i64 = 0;
        let mut departures: i64 = 0;
        for line in encoded.lines() {
            let Some((labels, value)) = line
                .strip_prefix("mx_registry_status_transitions_total{")
                .and_then(|rest| rest.split_once("} "))
            else {
                continue;
            };
            let count: i64 = value.trim().parse().unwrap_or_default();
            if labels.contains(r#"to="downloading""#) {
                arrivals = arrivals.saturating_add(count);
            }
            if labels.contains(r#"from="downloading""#) {
                departures = departures.saturating_add(count);
            }
        }
        arrivals.saturating_sub(departures)
    }

    /// Every `ClaimOutcome` must land on its own `result` label, and only the two
    /// owning outcomes may imply a transition.
    ///
    /// A takeover reported as a fresh claim is the regression `ClaimOutcome::TookOver`
    /// was introduced to prevent: a re-pull of hundreds of gigabytes reads as a
    /// first fetch. A backend error counted as a claim is worse still -- it books
    /// an arrival that never departs.
    #[tokio::test]
    async fn each_claim_outcome_reaches_its_own_result_label() {
        let mut mock = MockRegistryBackend::new();
        let call = std::sync::atomic::AtomicUsize::new(0);
        mock.expect_try_claim_for_download()
            .times(4)
            .returning(move |_, _, _, _| {
                match call.fetch_add(1, std::sync::atomic::Ordering::SeqCst) {
                    0 => Ok(ClaimOutcome::Claimed),
                    1 => Ok(ClaimOutcome::TookOver),
                    2 => Ok(ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADING)),
                    _ => Err("redis is down".into()),
                }
            });

        let (backend, prom) = instrumented(mock);
        for _ in 0..4 {
            let _ = backend
                .try_claim_for_download(MODEL, ModelProvider::HuggingFace, CLAIM, lease())
                .await;
        }

        let encoded = scrape(&prom);
        for result in ["claimed", "takeover", "already_exists", "error"] {
            let expected = format!(r#"mx_download_claims_total{{result="{result}"}} 1"#);
            assert!(encoded.contains(&expected), "missing {expected}: {encoded}");
        }
        // The two owning outcomes differ only in where they came from: a fresh
        // claim creates the record, a takeover finds it already `DOWNLOADING`.
        assert!(
            encoded.contains(
                r#"mx_registry_status_transitions_total{from="absent",to="downloading"} 1"#
            ),
            "a fresh claim did not book absent -> downloading: {encoded}"
        );
        assert!(
            encoded.contains(
                r#"mx_registry_status_transitions_total{from="downloading",to="downloading"} 1"#
            ),
            "a takeover did not book downloading -> downloading: {encoded}"
        );
        // A waiter and a failed call changed nothing, so the level is one
        // download, not three.
        assert_eq!(
            in_flight(&encoded),
            1,
            "waiters and errors must not move the level: {encoded}"
        );
    }

    /// One download, taken over mid-flight, then finished -- driven through the
    /// backend rather than by handing the counter its own labels.
    ///
    /// This is the module doc's invariant end to end: a takeover is an arrival
    /// and a departure that cancel, so the only thing left at the end is the
    /// finish. Booking the takeover as `absent -> downloading` drifts the level
    /// up by one per takeover and it never comes back down.
    #[tokio::test]
    async fn a_claim_taken_over_then_finished_leaves_nothing_in_flight() {
        let mut mock = MockRegistryBackend::new();
        let call = std::sync::atomic::AtomicUsize::new(0);
        mock.expect_try_claim_for_download()
            .times(2)
            .returning(move |_, _, _, _| {
                if call.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                    Ok(ClaimOutcome::Claimed)
                } else {
                    Ok(ClaimOutcome::TookOver)
                }
            });
        mock.expect_finish_download_claim()
            .times(1)
            .returning(|_, _, _, _, _| Ok(true));

        let (backend, prom) = instrumented(mock);

        let _ = backend
            .try_claim_for_download(MODEL, ModelProvider::HuggingFace, CLAIM, lease())
            .await;
        assert_eq!(in_flight(&scrape(&prom)), 1, "one download in flight");

        // Ownership moves; the download itself is still the same one.
        let _ = backend
            .try_claim_for_download(MODEL, ModelProvider::HuggingFace, "claim-2", lease())
            .await;
        assert_eq!(
            in_flight(&scrape(&prom)),
            1,
            "a takeover is not a second download"
        );

        let _ = backend
            .finish_download_claim(
                MODEL,
                ModelProvider::HuggingFace,
                "claim-2",
                ModelStatus::DOWNLOADED,
                None,
            )
            .await;
        let encoded = scrape(&prom);
        assert_eq!(in_flight(&encoded), 0, "the download finished: {encoded}");
    }

    /// Only the replica that wins the compare-and-set spawns the retry, so only
    /// it may book the arrival.
    #[tokio::test]
    async fn a_won_retry_reset_records_the_arrival_from_error() {
        let mut mock = MockRegistryBackend::new();
        mock.expect_try_reset_error_for_retry()
            .times(1)
            .returning(|_, _, _, _| Ok(true));

        let (backend, prom) = instrumented(mock);
        let _ = backend
            .try_reset_error_for_retry(MODEL, ModelProvider::HuggingFace, CLAIM, lease())
            .await;

        let encoded = scrape(&prom);
        // Direction matters as much as the fact of recording: reversed, a retry
        // being started reads as a download failing.
        assert!(
            encoded.contains(
                r#"mx_registry_status_transitions_total{from="error",to="downloading"} 1"#
            ),
            "a won retry did not book error -> downloading: {encoded}"
        );
        assert_eq!(in_flight(&encoded), 1, "the retry is in flight: {encoded}");
    }

    /// The losers of the compare-and-set must record nothing.
    ///
    /// Under a thundering herd on a failed model every waiting replica calls
    /// this and sees `false`; counting those would inflate the in-flight level by
    /// the replica count, with no matching departures to bring it back.
    #[tokio::test]
    async fn a_lost_retry_reset_records_no_arrival() {
        let mut mock = MockRegistryBackend::new();
        mock.expect_try_reset_error_for_retry()
            .times(1)
            .returning(|_, _, _, _| Ok(false));

        let (backend, prom) = instrumented(mock);
        let _ = backend
            .try_reset_error_for_retry(MODEL, ModelProvider::HuggingFace, CLAIM, lease())
            .await;

        let encoded = scrape(&prom);
        assert!(
            !encoded.contains("mx_registry_status_transitions_total{"),
            "a replica that lost the retry race booked a transition: {encoded}"
        );
    }

    /// `result="lost"` is the leading indicator of a duplicate download, so it
    /// must not be collapsed into `renewed` -- the panel would read flat-green
    /// during exactly the incident it was built for.
    #[tokio::test]
    async fn each_lease_heartbeat_outcome_reaches_its_own_result_label() {
        let mut mock = MockRegistryBackend::new();
        let call = std::sync::atomic::AtomicUsize::new(0);
        mock.expect_refresh_download_claim()
            .times(3)
            .returning(move |_, _, _, _| {
                match call.fetch_add(1, std::sync::atomic::Ordering::SeqCst) {
                    0 => Ok(true),
                    1 => Ok(false),
                    _ => Err("redis is down".into()),
                }
            });

        let (backend, prom) = instrumented(mock);
        for _ in 0..3 {
            let _ = backend
                .refresh_download_claim(MODEL, ModelProvider::HuggingFace, CLAIM, lease())
                .await;
        }

        let encoded = scrape(&prom);
        for result in ["renewed", "lost", "error"] {
            let expected = format!(r#"mx_download_lease_refresh_total{{result="{result}"}} 1"#);
            assert!(encoded.contains(&expected), "missing {expected}: {encoded}");
        }
    }

    /// The departure carries the status the download actually finished with.
    ///
    /// `ERROR` is exercised alongside `DOWNLOADED` because a hardcoded `to` label
    /// looks correct for the success case while reporting every failed download
    /// as a successful one.
    #[tokio::test]
    async fn a_finished_download_departs_to_its_terminal_status() {
        let mut mock = MockRegistryBackend::new();
        mock.expect_finish_download_claim()
            .times(2)
            .returning(|_, _, _, _, _| Ok(true));

        let (backend, prom) = instrumented(mock);
        for status in [ModelStatus::ERROR, ModelStatus::DOWNLOADED] {
            let _ = backend
                .finish_download_claim(MODEL, ModelProvider::HuggingFace, CLAIM, status, None)
                .await;
        }

        let encoded = scrape(&prom);
        for to in ["error", "downloaded"] {
            let expected = format!(
                r#"mx_registry_status_transitions_total{{from="downloading",to="{to}"}} 1"#
            );
            assert!(encoded.contains(&expected), "missing {expected}: {encoded}");
        }
    }

    /// `Ok(false)` means a stale owner was fenced after its lease was taken over.
    ///
    /// The entry did not leave `DOWNLOADING` on this call -- the new owner will
    /// book its own departure later -- so counting this one makes the in-flight
    /// derivation go negative.
    #[tokio::test]
    async fn a_fenced_stale_owner_records_no_departure() {
        let mut mock = MockRegistryBackend::new();
        mock.expect_finish_download_claim()
            .times(1)
            .returning(|_, _, _, _, _| Ok(false));

        let (backend, prom) = instrumented(mock);
        let _ = backend
            .finish_download_claim(
                MODEL,
                ModelProvider::HuggingFace,
                CLAIM,
                ModelStatus::DOWNLOADED,
                None,
            )
            .await;

        let encoded = scrape(&prom);
        assert!(
            !encoded.contains("mx_registry_status_transitions_total{"),
            "a fenced stale owner booked a departure it did not make: {encoded}"
        );
    }
}
