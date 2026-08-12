// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;
use modelexpress_common::grpc::revision::RevisionRecord;

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
        target_version: &str,
    ) -> CatalogResult<Option<RevisionRecord>>;

    async fn commit_revision(
        &self,
        model_id: &str,
        target_version: &str,
    ) -> CatalogResult<CommitOutcome>;
}

pub type DynRevisionCatalogBackend = Arc<dyn RevisionCatalogBackend>;

pub async fn create_revision_catalog_backend(
    config: BackendConfig,
) -> CatalogResult<Option<DynRevisionCatalogBackend>> {
    match config {
        BackendConfig::Redis { url } => {
            let backend = redis::RedisRevisionCatalogBackend::new(&url);
            backend.connect().await?;
            Ok(Some(Arc::new(backend)))
        }
        BackendConfig::Kubernetes { .. } => Ok(None),
        #[cfg(feature = "memory-backend")]
        BackendConfig::Memory => {
            #[cfg(feature = "integration-tests")]
            {
                let backend = testing::TestRevisionCatalogBackend::new();
                backend.connect().await?;
                Ok(Some(Arc::new(backend)))
            }
            #[cfg(not(feature = "integration-tests"))]
            {
                Err("in-memory revision catalog requires the 'integration-tests' feature".into())
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn revision_catalog_is_disabled_for_kubernetes_metadata() {
        let backend = create_revision_catalog_backend(BackendConfig::Kubernetes {
            namespace: "test".to_string(),
        })
        .await
        .expect("Kubernetes metadata must not prevent server startup");

        assert!(backend.is_none());
    }
}
