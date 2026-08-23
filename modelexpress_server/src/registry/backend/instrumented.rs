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
}
