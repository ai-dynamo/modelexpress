// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;
use modelexpress_common::grpc::revision::{ReceiverStateRecord, RevisionRecord};

use crate::backend_config::BackendConfig;

pub mod redis;
#[cfg(any(test, feature = "integration-tests"))]
pub(crate) mod testing;

pub type CatalogResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Debug, Clone, PartialEq)]
pub enum PublishReadyOutcome {
    Created(RevisionRecord),
    Existing(RevisionRecord),
    Conflict,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CommitOutcome {
    Committed(RevisionRecord),
    AlreadyCommitted(RevisionRecord),
    NotFound,
    InvalidState(RevisionRecord),
}

#[async_trait]
pub trait RevisionCatalogBackend: Send + Sync {
    async fn connect(&self) -> CatalogResult<()>;

    async fn publish_ready(&self, record: RevisionRecord) -> CatalogResult<PublishReadyOutcome>;

    async fn get_revision(
        &self,
        model_id: &str,
        version: &str,
    ) -> CatalogResult<Option<RevisionRecord>>;

    async fn list_revisions(&self, model_id: &str) -> CatalogResult<Vec<RevisionRecord>>;

    async fn commit_revision(
        &self,
        model_id: &str,
        version: &str,
        changed_at_unix_ms: u64,
    ) -> CatalogResult<CommitOutcome>;

    async fn upsert_receiver_state(
        &self,
        record: ReceiverStateRecord,
    ) -> CatalogResult<ReceiverStateRecord>;

    async fn list_receiver_states(
        &self,
        model_id: &str,
        version: &str,
    ) -> CatalogResult<Vec<ReceiverStateRecord>>;
}

pub type DynRevisionCatalogBackend = Arc<dyn RevisionCatalogBackend>;

pub async fn create_revision_catalog_backend(
    config: BackendConfig,
) -> CatalogResult<DynRevisionCatalogBackend> {
    match config {
        BackendConfig::Redis { url } => {
            let backend = redis::RedisRevisionCatalogBackend::new(&url);
            backend.connect().await?;
            Ok(Arc::new(backend))
        }
        BackendConfig::Kubernetes { .. } => {
            Err("revision catalog currently supports only the Redis backend".into())
        }
        #[cfg(feature = "memory-backend")]
        BackendConfig::Memory => {
            #[cfg(feature = "integration-tests")]
            {
                let backend = testing::TestRevisionCatalogBackend::new();
                backend.connect().await?;
                Ok(Arc::new(backend))
            }
            #[cfg(not(feature = "integration-tests"))]
            {
                Err("revision catalog currently supports only the Redis backend".into())
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn revision_catalog_rejects_kubernetes_until_future_support_lands() {
        let error = create_revision_catalog_backend(BackendConfig::Kubernetes {
            namespace: "test".to_string(),
        })
        .await
        .err()
        .expect("Kubernetes revision catalog is not implemented");
        assert_eq!(
            error.to_string(),
            "revision catalog currently supports only the Redis backend"
        );
    }
}
