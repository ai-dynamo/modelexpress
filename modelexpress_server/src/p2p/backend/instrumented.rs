// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metrics decorator for [`MetadataBackend`].
//!
//! Wraps any backend and records [`crate::metrics::backend`] families around
//! every call, leaving the Redis, Kubernetes and in-memory implementations
//! untouched. A backend added later is instrumented because it is wrapped at
//! construction, not because someone remembered to add timing code to it.

use async_trait::async_trait;
use std::sync::Arc;

use crate::metrics::backend::{BackendMetrics, Store};
use crate::p2p::backend::{
    MetadataBackend, MetadataResult, ModelMetadataRecord, SourceInstanceInfo,
};
use modelexpress_common::grpc::p2p::{SourceIdentity, SourceStatus, WorkerMetadata};

/// A [`MetadataBackend`] that records timing and outcome for each operation.
pub struct InstrumentedMetadataBackend {
    inner: Arc<dyn MetadataBackend>,
    metrics: BackendMetrics,
}

impl InstrumentedMetadataBackend {
    /// Wrap `inner`, returning it as a trait object so call sites are unchanged.
    #[must_use]
    pub fn wrap(
        inner: Arc<dyn MetadataBackend>,
        metrics: BackendMetrics,
    ) -> Arc<dyn MetadataBackend> {
        Arc::new(Self { inner, metrics })
    }
}

#[async_trait]
impl MetadataBackend for InstrumentedMetadataBackend {
    async fn connect(&self) -> MetadataResult<()> {
        self.metrics
            .time(Store::P2p, "connect", self.inner.connect())
            .await
    }

    async fn publish_metadata(
        &self,
        identity: &SourceIdentity,
        worker_id: &str,
        worker: WorkerMetadata,
        pod_name: &str,
        pod_uid: &str,
        pod_namespace: &str,
    ) -> MetadataResult<()> {
        self.metrics
            .time(
                Store::P2p,
                "publish_metadata",
                self.inner.publish_metadata(
                    identity,
                    worker_id,
                    worker,
                    pod_name,
                    pod_uid,
                    pod_namespace,
                ),
            )
            .await
    }

    async fn get_metadata(
        &self,
        source_id: &str,
        worker_id: &str,
    ) -> MetadataResult<Option<ModelMetadataRecord>> {
        self.metrics
            .time(
                Store::P2p,
                "get_metadata",
                self.inner.get_metadata(source_id, worker_id),
            )
            .await
    }

    async fn list_workers(
        &self,
        source_id: Option<String>,
        status_filter: Option<SourceStatus>,
    ) -> MetadataResult<Vec<SourceInstanceInfo>> {
        self.metrics
            .time(
                Store::P2p,
                "list_workers",
                self.inner.list_workers(source_id, status_filter),
            )
            .await
    }

    /// Forwarded explicitly, and it must stay that way.
    ///
    /// This is the one trait method with a default body, and only the Redis
    /// backend overrides it. If this decorator inherited the default, the default
    /// would run *here* and call `self.list_workers` -- the decorator's -- which
    /// delegates to the inner backend. Redis's pipelined implementation would
    /// never run, reintroducing the per-source and per-worker round trips it
    /// exists to avoid, and every call would be counted twice: once as
    /// `list_workers_filtered` and once as `list_workers`. Nothing would fail to
    /// compile and no existing test would go red.
    async fn list_workers_filtered(
        &self,
        source_id: Option<String>,
        status_filter: Option<SourceStatus>,
        model_name_filter: Option<String>,
        worker_rank_filter: Option<u32>,
        min_training_step: Option<u64>,
        min_updated_at: Option<i64>,
        limit: Option<usize>,
    ) -> MetadataResult<Vec<SourceInstanceInfo>> {
        self.metrics
            .time(
                Store::P2p,
                "list_workers_filtered",
                self.inner.list_workers_filtered(
                    source_id,
                    status_filter,
                    model_name_filter,
                    worker_rank_filter,
                    min_training_step,
                    min_updated_at,
                    limit,
                ),
            )
            .await
    }

    async fn remove_metadata(&self, source_id: &str) -> MetadataResult<()> {
        self.metrics
            .time(
                Store::P2p,
                "remove_metadata",
                self.inner.remove_metadata(source_id),
            )
            .await
    }

    async fn remove_worker(&self, source_id: &str, worker_id: &str) -> MetadataResult<()> {
        self.metrics
            .time(
                Store::P2p,
                "remove_worker",
                self.inner.remove_worker(source_id, worker_id),
            )
            .await
    }

    async fn list_sources(&self) -> MetadataResult<Vec<(String, String)>> {
        self.metrics
            .time(Store::P2p, "list_sources", self.inner.list_sources())
            .await
    }

    async fn update_status(
        &self,
        source_id: &str,
        worker_id: &str,
        worker_rank: u32,
        status: SourceStatus,
        updated_at: i64,
    ) -> MetadataResult<()> {
        self.metrics
            .time(
                Store::P2p,
                "update_status",
                self.inner
                    .update_status(source_id, worker_id, worker_rank, status, updated_at),
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{encode_text, new_registry};
    use crate::p2p::backend::MockMetadataBackend;

    /// The forwarding trap: `list_workers_filtered` must reach the inner
    /// backend's implementation, not fall through the default body onto
    /// `list_workers`.
    #[tokio::test]
    async fn list_workers_filtered_does_not_fall_through_to_list_workers() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers_filtered()
            .times(1)
            .returning(|_, _, _, _, _, _, _| Ok(Vec::new()));
        // If the decorator inherited the default body, this would be called.
        mock.expect_list_workers().times(0);

        let mut registry = new_registry();
        let metrics = BackendMetrics::register(&mut registry);
        let backend = InstrumentedMetadataBackend::wrap(Arc::new(mock), metrics);

        let result = backend
            .list_workers_filtered(None, None, None, None, None, None, None)
            .await;
        assert!(result.is_ok());

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(
                r#"mx_backend_ops_total{store="p2p",op="list_workers_filtered",result="ok"} 1"#
            ),
            "{encoded}"
        );
        // Counted once, under one name.
        assert!(
            !encoded.contains(r#"op="list_workers","#),
            "list_workers was also recorded: {encoded}"
        );
    }

    /// Nine near-identical forwarding bodies invite a copy-pasted op literal, so
    /// pin that each method reports under its own name and none under another's.
    #[tokio::test]
    async fn every_method_reports_under_its_own_op_name() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_connect().times(1).returning(|| Ok(()));
        mock.expect_publish_metadata()
            .times(1)
            .returning(|_, _, _, _, _, _| Ok(()));
        mock.expect_get_metadata()
            .times(1)
            .returning(|_, _| Ok(None));
        mock.expect_list_workers()
            .times(1)
            .returning(|_, _| Ok(Vec::new()));
        mock.expect_list_workers_filtered()
            .times(1)
            .returning(|_, _, _, _, _, _, _| Ok(Vec::new()));
        mock.expect_remove_metadata().times(1).returning(|_| Ok(()));
        mock.expect_remove_worker()
            .times(1)
            .returning(|_, _| Ok(()));
        mock.expect_list_sources()
            .times(1)
            .returning(|| Ok(Vec::new()));
        mock.expect_update_status()
            .times(1)
            .returning(|_, _, _, _, _| Ok(()));

        let mut registry = new_registry();
        let metrics = BackendMetrics::register(&mut registry);
        let backend = InstrumentedMetadataBackend::wrap(Arc::new(mock), metrics);

        let _ = backend.connect().await;
        let _ = backend
            .publish_metadata(
                &SourceIdentity::default(),
                "worker-0",
                WorkerMetadata::default(),
                "pod",
                "uid",
                "namespace",
            )
            .await;
        let _ = backend.get_metadata("source-0", "worker-0").await;
        let _ = backend.list_workers(None, None).await;
        let _ = backend
            .list_workers_filtered(None, None, None, None, None, None, None)
            .await;
        let _ = backend.remove_metadata("source-0").await;
        let _ = backend.remove_worker("source-0", "worker-0").await;
        let _ = backend.list_sources().await;
        let _ = backend
            .update_status("source-0", "worker-0", 0, SourceStatus::Ready, 0)
            .await;

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        for op in [
            "connect",
            "publish_metadata",
            "get_metadata",
            "list_workers",
            "list_workers_filtered",
            "remove_metadata",
            "remove_worker",
            "list_sources",
            "update_status",
        ] {
            let expected =
                format!(r#"mx_backend_ops_total{{store="p2p",op="{op}",result="ok"}} 1"#);
            assert!(encoded.contains(&expected), "missing {op}: {encoded}");
        }
    }

    #[tokio::test]
    async fn a_backend_failure_is_recorded_as_an_error() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_connect()
            .times(1)
            .returning(|| Err("redis is down".into()));

        let mut registry = new_registry();
        let metrics = BackendMetrics::register(&mut registry);
        let backend = InstrumentedMetadataBackend::wrap(Arc::new(mock), metrics);

        assert!(backend.connect().await.is_err());

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(r#"mx_backend_ops_total{store="p2p",op="connect",result="error"} 1"#),
            "{encoded}"
        );
    }
}
