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
use crate::registry::backend::{ClaimOutcome, ModelRecord, RegistryBackend, RegistryResult};
use modelexpress_common::models::{ModelProvider, ModelStatus};

/// A [`RegistryBackend`] that records timing and outcome for each operation.
pub struct InstrumentedRegistryBackend {
    inner: Arc<dyn RegistryBackend>,
    metrics: BackendMetrics,
}

impl InstrumentedRegistryBackend {
    /// Wrap `inner`, returning it as a trait object so call sites are unchanged.
    #[must_use]
    pub fn wrap(
        inner: Arc<dyn RegistryBackend>,
        metrics: BackendMetrics,
    ) -> Arc<dyn RegistryBackend> {
        Arc::new(Self { inner, metrics })
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
        self.metrics
            .time(
                Store::Registry,
                "try_claim_for_download",
                self.inner
                    .try_claim_for_download(model_name, provider, claim_id, lease_duration),
            )
            .await
    }

    async fn try_reset_error_for_retry(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        lease_duration: std::time::Duration,
    ) -> RegistryResult<bool> {
        self.metrics
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
            .await
    }

    async fn refresh_download_claim(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        lease_duration: std::time::Duration,
    ) -> RegistryResult<bool> {
        self.metrics
            .time(
                Store::Registry,
                "refresh_download_claim",
                self.inner
                    .refresh_download_claim(model_name, provider, claim_id, lease_duration),
            )
            .await
    }

    async fn finish_download_claim(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        status: ModelStatus,
        message: Option<String>,
    ) -> RegistryResult<bool> {
        self.metrics
            .time(
                Store::Registry,
                "finish_download_claim",
                self.inner
                    .finish_download_claim(model_name, provider, claim_id, status, message),
            )
            .await
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
        let backend = InstrumentedRegistryBackend::wrap(Arc::new(mock), metrics);

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
        let backend = InstrumentedRegistryBackend::wrap(Arc::new(mock), metrics);

        assert!(backend.connect().await.is_err());

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(
                r#"mx_backend_ops_total{store="registry",op="connect",result="error"} 1"#
            ),
            "{encoded}"
        );
    }
}
