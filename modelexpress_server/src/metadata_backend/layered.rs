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

use super::{
    memory::InMemoryBackend, MetadataBackend, MetadataResult, ModelMetadataRecord,
};
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

        let mut hydrated = 0;
        for model_name in &models {
            if let Some(record) = persistent.get_metadata(model_name).await? {
                // Re-publish into in-memory cache
                let workers: Vec<WorkerMetadata> = record
                    .workers
                    .into_iter()
                    .map(WorkerMetadata::from)
                    .collect();
                self.cache
                    .publish_metadata(model_name, workers)
                    .await?;
                hydrated += 1;
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
        if let Some(persistent) = &self.persistent {
            if let Err(e) = persistent.publish_metadata(model_name, workers).await {
                warn!(
                    "Failed to write-through to persistent backend for '{}': {} - data is in cache only",
                    model_name, e
                );
            }
        }

        Ok(())
    }

    async fn get_metadata(&self, model_name: &str) -> MetadataResult<Option<ModelMetadataRecord>> {
        // Always read from in-memory cache (fast path)
        self.cache.get_metadata(model_name).await
    }

    async fn remove_metadata(&self, model_name: &str) -> MetadataResult<()> {
        self.cache.remove_metadata(model_name).await?;

        if let Some(persistent) = &self.persistent {
            if let Err(e) = persistent.remove_metadata(model_name).await {
                warn!(
                    "Failed to remove '{}' from persistent backend: {}",
                    model_name, e
                );
            }
        }

        Ok(())
    }

    async fn list_models(&self) -> MetadataResult<Vec<String>> {
        // List from cache (should be complete after hydration)
        self.cache.list_models().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_only() {
        let backend = LayeredBackend::memory_only();
        backend.connect().await.unwrap();

        let workers = vec![WorkerMetadata {
            worker_rank: 0,
            nixl_metadata: vec![1],
            tensors: vec![],
        }];

        backend
            .publish_metadata("test", workers)
            .await
            .unwrap();

        let result = backend.get_metadata("test").await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().workers.len(), 1);
    }
}
