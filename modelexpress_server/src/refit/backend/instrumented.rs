// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metrics decorator for [`RefitBackend`].
//!
//! Wraps any backend and records [`crate::metrics::backend`] families around
//! every call, leaving the Redis implementation untouched. A backend added later
//! is instrumented because it is wrapped at construction, not because someone
//! remembered to add timing code to it.
//!
//! Every trait method is forwarded explicitly. The trait has no default bodies
//! today; if one is added later it must be overridden here as well, because an
//! inherited default would run on the decorator, call back into the decorator's
//! other methods, bypass any specialised override in the concrete backend and
//! count the call twice under two op names -- all without failing to compile.

use async_trait::async_trait;
use std::sync::Arc;

use crate::metrics::backend::{BackendMetrics, Store};
use crate::refit::backend::{RefitBackend, RefitResult};
use modelexpress_common::grpc::refit::{
    CreateWeightVersionRequest, DeleteVersionLeaseRequest, DeleteWeightVersionShardRequest,
    RegisterVersionLeaseRequest, UpdateWeightVersionStateRequest, VersionLease, WeightVersion,
    WeightVersionShard, WorkerRegistration,
};

/// A [`RefitBackend`] that records timing and outcome for each operation.
pub struct InstrumentedRefitBackend {
    inner: Arc<dyn RefitBackend>,
    metrics: BackendMetrics,
}

impl InstrumentedRefitBackend {
    /// Wrap `inner`, returning it as a trait object so call sites are unchanged.
    #[must_use]
    pub fn wrap(inner: Arc<dyn RefitBackend>, metrics: BackendMetrics) -> Arc<dyn RefitBackend> {
        Arc::new(Self { inner, metrics })
    }
}

#[async_trait]
impl RefitBackend for InstrumentedRefitBackend {
    async fn register_worker(
        &self,
        worker: WorkerRegistration,
        ttl_seconds: u32,
    ) -> RefitResult<WorkerRegistration> {
        self.metrics
            .time(
                Store::Refit,
                "register_worker",
                self.inner.register_worker(worker, ttl_seconds),
            )
            .await
    }

    async fn create_weight_version(
        &self,
        request: &CreateWeightVersionRequest,
    ) -> RefitResult<WeightVersion> {
        self.metrics
            .time(
                Store::Refit,
                "create_weight_version",
                self.inner.create_weight_version(request),
            )
            .await
    }

    async fn get_weight_version(&self, uid: &str) -> RefitResult<WeightVersion> {
        self.metrics
            .time(
                Store::Refit,
                "get_weight_version",
                self.inner.get_weight_version(uid),
            )
            .await
    }

    async fn delete_weight_version(&self, uid: &str) -> RefitResult<WeightVersion> {
        self.metrics
            .time(
                Store::Refit,
                "delete_weight_version",
                self.inner.delete_weight_version(uid),
            )
            .await
    }

    async fn update_weight_version_state(
        &self,
        request: &UpdateWeightVersionStateRequest,
    ) -> RefitResult<WeightVersion> {
        self.metrics
            .time(
                Store::Refit,
                "update_weight_version_state",
                self.inner.update_weight_version_state(request),
            )
            .await
    }

    async fn create_weight_version_shard(
        &self,
        shard: WeightVersionShard,
    ) -> RefitResult<(WeightVersionShard, WeightVersion)> {
        self.metrics
            .time(
                Store::Refit,
                "create_weight_version_shard",
                self.inner.create_weight_version_shard(shard),
            )
            .await
    }

    async fn list_weight_version_shards(
        &self,
        version_id: &str,
    ) -> RefitResult<Vec<WeightVersionShard>> {
        self.metrics
            .time(
                Store::Refit,
                "list_weight_version_shards",
                self.inner.list_weight_version_shards(version_id),
            )
            .await
    }

    async fn delete_weight_version_shard(
        &self,
        request: &DeleteWeightVersionShardRequest,
    ) -> RefitResult<bool> {
        self.metrics
            .time(
                Store::Refit,
                "delete_weight_version_shard",
                self.inner.delete_weight_version_shard(request),
            )
            .await
    }

    async fn register_version_lease(
        &self,
        request: &RegisterVersionLeaseRequest,
    ) -> RefitResult<VersionLease> {
        self.metrics
            .time(
                Store::Refit,
                "register_version_lease",
                self.inner.register_version_lease(request),
            )
            .await
    }

    async fn delete_version_lease(&self, request: &DeleteVersionLeaseRequest) -> RefitResult<bool> {
        self.metrics
            .time(
                Store::Refit,
                "delete_version_lease",
                self.inner.delete_version_lease(request),
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{encode_text, new_registry};
    use crate::refit::backend::RefitBackendError;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// `RefitBackend` carries no `mockall` mock, and adding `automock` to it
    /// would change shared code, so the tests drive a stub that returns a fixed
    /// outcome and counts how many calls reached it.
    #[derive(Default)]
    struct StubRefitBackend {
        fail: bool,
        calls: AtomicUsize,
    }

    impl StubRefitBackend {
        fn failing() -> Self {
            Self {
                fail: true,
                calls: AtomicUsize::new(0),
            }
        }

        fn outcome<T: Default>(&self) -> RefitResult<T> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.fail {
                Err(RefitBackendError::Unavailable("redis is down".to_owned()))
            } else {
                Ok(T::default())
            }
        }
    }

    #[async_trait]
    impl RefitBackend for StubRefitBackend {
        async fn register_worker(
            &self,
            _worker: WorkerRegistration,
            _ttl_seconds: u32,
        ) -> RefitResult<WorkerRegistration> {
            self.outcome()
        }

        async fn create_weight_version(
            &self,
            _request: &CreateWeightVersionRequest,
        ) -> RefitResult<WeightVersion> {
            self.outcome()
        }

        async fn get_weight_version(&self, _uid: &str) -> RefitResult<WeightVersion> {
            self.outcome()
        }

        async fn delete_weight_version(&self, _uid: &str) -> RefitResult<WeightVersion> {
            self.outcome()
        }

        async fn update_weight_version_state(
            &self,
            _request: &UpdateWeightVersionStateRequest,
        ) -> RefitResult<WeightVersion> {
            self.outcome()
        }

        async fn create_weight_version_shard(
            &self,
            _shard: WeightVersionShard,
        ) -> RefitResult<(WeightVersionShard, WeightVersion)> {
            self.outcome()
        }

        async fn list_weight_version_shards(
            &self,
            _version_id: &str,
        ) -> RefitResult<Vec<WeightVersionShard>> {
            self.outcome()
        }

        async fn delete_weight_version_shard(
            &self,
            _request: &DeleteWeightVersionShardRequest,
        ) -> RefitResult<bool> {
            self.outcome()
        }

        async fn register_version_lease(
            &self,
            _request: &RegisterVersionLeaseRequest,
        ) -> RefitResult<VersionLease> {
            self.outcome()
        }

        async fn delete_version_lease(
            &self,
            _request: &DeleteVersionLeaseRequest,
        ) -> RefitResult<bool> {
            self.outcome()
        }
    }

    #[tokio::test]
    async fn a_successful_op_is_recorded_against_its_method_name() {
        let stub = Arc::new(StubRefitBackend::default());
        let mut registry = new_registry();
        let metrics = BackendMetrics::register(&mut registry);
        let backend =
            InstrumentedRefitBackend::wrap(Arc::clone(&stub) as Arc<dyn RefitBackend>, metrics);

        assert!(backend.get_weight_version("v1").await.is_ok());
        assert_eq!(stub.calls.load(Ordering::Relaxed), 1);

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(
                r#"mx_backend_ops_total{store="refit",op="get_weight_version",result="ok"} 1"#
            ),
            "{encoded}"
        );
    }

    #[tokio::test]
    async fn a_backend_failure_is_recorded_as_an_error() {
        let stub = Arc::new(StubRefitBackend::failing());
        let mut registry = new_registry();
        let metrics = BackendMetrics::register(&mut registry);
        let backend =
            InstrumentedRefitBackend::wrap(Arc::clone(&stub) as Arc<dyn RefitBackend>, metrics);

        assert!(
            backend
                .register_version_lease(&RegisterVersionLeaseRequest::default())
                .await
                .is_err()
        );

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(
                r#"mx_backend_ops_total{store="refit",op="register_version_lease",result="error"} 1"#
            ),
            "{encoded}"
        );
    }

    /// Ten near-identical forwarding bodies invite a copy-pasted op literal, so
    /// pin that each method reports under its own name and none under another's.
    #[tokio::test]
    async fn every_method_reports_under_its_own_op_name() {
        let stub = Arc::new(StubRefitBackend::default());
        let mut registry = new_registry();
        let metrics = BackendMetrics::register(&mut registry);
        let backend =
            InstrumentedRefitBackend::wrap(Arc::clone(&stub) as Arc<dyn RefitBackend>, metrics);

        let _ = backend
            .register_worker(WorkerRegistration::default(), 30)
            .await;
        let _ = backend
            .create_weight_version(&CreateWeightVersionRequest::default())
            .await;
        let _ = backend.get_weight_version("v1").await;
        let _ = backend.delete_weight_version("v1").await;
        let _ = backend
            .update_weight_version_state(&UpdateWeightVersionStateRequest::default())
            .await;
        let _ = backend
            .create_weight_version_shard(WeightVersionShard::default())
            .await;
        let _ = backend.list_weight_version_shards("v1").await;
        let _ = backend
            .delete_weight_version_shard(&DeleteWeightVersionShardRequest::default())
            .await;
        let _ = backend
            .register_version_lease(&RegisterVersionLeaseRequest::default())
            .await;
        let _ = backend
            .delete_version_lease(&DeleteVersionLeaseRequest::default())
            .await;

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        for op in [
            "register_worker",
            "create_weight_version",
            "get_weight_version",
            "delete_weight_version",
            "update_weight_version_state",
            "create_weight_version_shard",
            "list_weight_version_shards",
            "delete_weight_version_shard",
            "register_version_lease",
            "delete_version_lease",
        ] {
            let expected =
                format!(r#"mx_backend_ops_total{{store="refit",op="{op}",result="ok"}} 1"#);
            assert!(encoded.contains(&expected), "missing {op}: {encoded}");
        }
        assert_eq!(stub.calls.load(Ordering::Relaxed), 10);
    }
}
