// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Layered metadata backend: in-memory cache + optional persistent backend.
//!
//! Architecture:
//!   Request → InMemoryBackend (nanosecond reads, always present)
//!               │
//!               └── Write-through to persistent backend (if configured)
//!                   - Redis: for HA across server restarts
//!                   - Kubernetes CRD: for K8s-native observability
//!
//! On startup, the in-memory cache is hydrated from the persistent backend
//! so the server recovers state without sources needing to re-publish.

use super::{MetadataBackend, MetadataResult, ModelMetadataRecord, memory::InMemoryBackend};
use async_trait::async_trait;
use modelexpress_common::grpc::p2p::WorkerMetadata;
use std::sync::Arc;
use tracing::{info, warn};

/// Layered backend that combines an in-memory cache with an optional
/// persistent backend for write-through durability.
pub struct LayeredBackend {
    /// Primary: always-present in-memory cache.
    cache: InMemoryBackend,
    /// Optional persistent backend (Redis or Kubernetes).
    persistent: Option<Arc<dyn MetadataBackend>>,
}

impl LayeredBackend {
    /// Create a layered backend with only in-memory cache (standalone mode).
    pub fn memory_only() -> Self {
        Self {
            cache: InMemoryBackend::new(),
            persistent: None,
        }
    }

    /// Create a layered backend with in-memory cache + persistent backend.
    pub fn with_persistent(persistent: Arc<dyn MetadataBackend>) -> Self {
        Self {
            cache: InMemoryBackend::new(),
            persistent: Some(persistent),
        }
    }

    /// Hydrate the in-memory cache from the persistent backend.
    /// Called once on startup to recover state after server restart.
    async fn hydrate_cache(&self) -> MetadataResult<()> {
        let persistent = match &self.persistent {
            Some(p) => p,
            None => return Ok(()),
        };

        let models = persistent.list_models().await?;
        if models.is_empty() {
            info!("No existing metadata found in persistent backend");
            return Ok(());
        }

        let mut hydrated: usize = 0;
        for model_name in &models {
            if let Some(record) = persistent.get_metadata(model_name).await? {
                // Re-publish into in-memory cache
                let workers: Vec<WorkerMetadata> = record
                    .workers
                    .into_iter()
                    .map(WorkerMetadata::from)
                    .collect();
                self.cache.publish_metadata(model_name, workers).await?;
                hydrated = hydrated.saturating_add(1);
            }
        }

        info!(
            "Hydrated in-memory cache with {} models from persistent backend",
            hydrated
        );
        Ok(())
    }
}

#[async_trait]
impl MetadataBackend for LayeredBackend {
    async fn connect(&self) -> MetadataResult<()> {
        self.cache.connect().await?;

        if let Some(persistent) = &self.persistent {
            persistent.connect().await?;

            // Hydrate cache from persistent backend on startup
            if let Err(e) = self.hydrate_cache().await {
                warn!(
                    "Failed to hydrate cache from persistent backend: {} - starting with empty cache",
                    e
                );
            }
        }

        let mode = if self.persistent.is_some() {
            "layered (in-memory + persistent)"
        } else {
            "in-memory only"
        };
        info!("Layered metadata backend initialized: {}", mode);
        Ok(())
    }

    async fn publish_metadata(
        &self,
        model_name: &str,
        workers: Vec<WorkerMetadata>,
    ) -> MetadataResult<()> {
        // Write to in-memory cache first (fast path)
        self.cache
            .publish_metadata(model_name, workers.clone())
            .await?;

        // Write-through to persistent backend (best-effort)
        if let Some(persistent) = &self.persistent
            && let Err(e) = persistent.publish_metadata(model_name, workers).await
        {
            warn!(
                "Failed to write-through to persistent backend for '{}': {} - data is in cache only",
                model_name, e
            );
        }

        Ok(())
    }

    async fn get_metadata(&self, model_name: &str) -> MetadataResult<Option<ModelMetadataRecord>> {
        // Always read from in-memory cache (fast path)
        self.cache.get_metadata(model_name).await
    }

    async fn remove_metadata(&self, model_name: &str) -> MetadataResult<()> {
        self.cache.remove_metadata(model_name).await?;

        if let Some(persistent) = &self.persistent
            && let Err(e) = persistent.remove_metadata(model_name).await
        {
            warn!(
                "Failed to remove '{}' from persistent backend: {}",
                model_name, e
            );
        }

        Ok(())
    }

    async fn list_models(&self) -> MetadataResult<Vec<String>> {
        // List from cache (should be complete after hydration)
        self.cache.list_models().await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn test_workers(rank: u32) -> Vec<WorkerMetadata> {
        vec![WorkerMetadata {
            worker_rank: rank,
            nixl_metadata: vec![1, 2, 3],
            tensors: vec![],
        }]
    }

    #[tokio::test]
    async fn test_memory_only() {
        let backend = LayeredBackend::memory_only();
        backend.connect().await.unwrap();

        backend
            .publish_metadata("test", test_workers(0))
            .await
            .unwrap();

        let result = backend.get_metadata("test").await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().workers.len(), 1);
    }

    #[tokio::test]
    async fn test_write_through() {
        let persistent = Arc::new(InMemoryBackend::new());
        persistent.connect().await.unwrap();

        let backend = LayeredBackend::with_persistent(persistent.clone());
        backend.connect().await.unwrap();

        backend
            .publish_metadata("model-a", test_workers(0))
            .await
            .unwrap();

        // Both cache and persistent should have the data
        let from_cache = backend.get_metadata("model-a").await.unwrap();
        assert!(from_cache.is_some(), "cache should have the data");

        let from_persistent = persistent.get_metadata("model-a").await.unwrap();
        assert!(
            from_persistent.is_some(),
            "persistent backend should have the data"
        );
        assert_eq!(from_persistent.unwrap().workers.len(), 1);
    }

    #[tokio::test]
    async fn test_hydration_on_connect() {
        // Pre-populate a persistent backend
        let persistent = Arc::new(InMemoryBackend::new());
        persistent.connect().await.unwrap();
        persistent
            .publish_metadata("existing-model", test_workers(0))
            .await
            .unwrap();

        // Create a NEW layered backend pointing to the same persistent store
        let backend = LayeredBackend::with_persistent(persistent.clone());
        backend.connect().await.unwrap(); // should hydrate

        // Cache should have the data from persistent without re-publishing
        let result = backend.get_metadata("existing-model").await.unwrap();
        assert!(result.is_some(), "cache should be hydrated from persistent");
        assert_eq!(result.unwrap().workers[0].worker_rank, 0);
    }

    #[tokio::test]
    async fn test_hydration_multiple_models() {
        let persistent = Arc::new(InMemoryBackend::new());
        persistent.connect().await.unwrap();
        persistent
            .publish_metadata("model-1", test_workers(0))
            .await
            .unwrap();
        persistent
            .publish_metadata("model-2", test_workers(1))
            .await
            .unwrap();

        let backend = LayeredBackend::with_persistent(persistent);
        backend.connect().await.unwrap();

        let models = backend.list_models().await.unwrap();
        assert_eq!(models.len(), 2);

        let m1 = backend.get_metadata("model-1").await.unwrap().unwrap();
        assert_eq!(m1.workers[0].worker_rank, 0);
        let m2 = backend.get_metadata("model-2").await.unwrap().unwrap();
        assert_eq!(m2.workers[0].worker_rank, 1);
    }

    #[tokio::test]
    async fn test_remove_propagates_to_persistent() {
        let persistent = Arc::new(InMemoryBackend::new());
        persistent.connect().await.unwrap();

        let backend = LayeredBackend::with_persistent(persistent.clone());
        backend.connect().await.unwrap();

        backend
            .publish_metadata("to-delete", test_workers(0))
            .await
            .unwrap();

        backend.remove_metadata("to-delete").await.unwrap();

        assert!(backend.get_metadata("to-delete").await.unwrap().is_none());
        assert!(
            persistent
                .get_metadata("to-delete")
                .await
                .unwrap()
                .is_none()
        );
    }
}
