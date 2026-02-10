// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory backend for P2P model metadata storage.
//!
//! Zero-dependency, lowest-latency backend. Ideal as the primary cache layer
//! or as a standalone backend for development and simple deployments.
//!
//! Data is lost on server restart — pair with Redis or Kubernetes backends
//! for persistence and high availability.

use super::{MetadataBackend, MetadataResult, ModelMetadataRecord, WorkerRecord};
use async_trait::async_trait;
use modelexpress_common::grpc::p2p::WorkerMetadata;
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// In-memory backend using `RwLock<HashMap>`.
///
/// Concurrent reads are lock-free; writes acquire an exclusive lock.
/// Worker merge logic mirrors the Redis Lua script (merge by rank, sort).
pub struct InMemoryBackend {
    models: RwLock<HashMap<String, ModelMetadataRecord>>,
}

impl InMemoryBackend {
    /// Create a new in-memory backend.
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MetadataBackend for InMemoryBackend {
    async fn connect(&self) -> MetadataResult<()> {
        info!("In-memory metadata backend initialized");
        Ok(())
    }

    async fn publish_metadata(
        &self,
        model_name: &str,
        workers: Vec<WorkerMetadata>,
    ) -> MetadataResult<()> {
        let new_workers: Vec<WorkerRecord> =
            workers.into_iter().map(WorkerRecord::from).collect();
        let total_tensors: usize = new_workers.iter().map(|w| w.tensors.len()).sum();
        let timestamp = chrono::Utc::now().timestamp();

        let mut models = self.models.write().await;

        let record = models
            .entry(model_name.to_string())
            .or_insert_with(|| ModelMetadataRecord {
                model_name: model_name.to_string(),
                workers: Vec::new(),
                published_at: timestamp,
            });

        // Atomic merge: update existing workers by rank, append new ones
        for new_worker in new_workers {
            if let Some(existing) = record
                .workers
                .iter_mut()
                .find(|w| w.worker_rank == new_worker.worker_rank)
            {
                *existing = new_worker;
            } else {
                record.workers.push(new_worker);
            }
        }

        // Sort by rank (mirrors Redis Lua script)
        record.workers.sort_by_key(|w| w.worker_rank);
        record.published_at = timestamp;

        let worker_count = record.workers.len();

        info!(
            "Published metadata for model '{}': {} workers total ({} new tensors)",
            model_name, worker_count, total_tensors
        );
        Ok(())
    }

    async fn get_metadata(&self, model_name: &str) -> MetadataResult<Option<ModelMetadataRecord>> {
        let models = self.models.read().await;
        let result = models.get(model_name).cloned();
        debug!(
            "get_metadata '{}': {}",
            model_name,
            if result.is_some() { "found" } else { "not found" }
        );
        Ok(result)
    }

    async fn remove_metadata(&self, model_name: &str) -> MetadataResult<()> {
        let mut models = self.models.write().await;
        models.remove(model_name);
        info!("Removed metadata for model '{}'", model_name);
        Ok(())
    }

    async fn list_models(&self) -> MetadataResult<Vec<String>> {
        let models = self.models.read().await;
        Ok(models.keys().cloned().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use modelexpress_common::grpc::p2p::TensorDescriptor;

    #[tokio::test]
    async fn test_publish_and_get() {
        let backend = InMemoryBackend::new();
        backend.connect().await.unwrap();

        let workers = vec![WorkerMetadata {
            worker_rank: 0,
            nixl_metadata: vec![1, 2, 3],
            tensors: vec![TensorDescriptor {
                name: "layer.0.weight".to_string(),
                addr: 0x1000,
                size: 4096,
                device_id: 0,
                dtype: "bfloat16".to_string(),
            }],
        }];

        backend
            .publish_metadata("test-model", workers)
            .await
            .unwrap();

        let result = backend.get_metadata("test-model").await.unwrap();
        assert!(result.is_some());

        let record = result.unwrap();
        assert_eq!(record.model_name, "test-model");
        assert_eq!(record.workers.len(), 1);
        assert_eq!(record.workers[0].worker_rank, 0);
    }

    #[tokio::test]
    async fn test_merge_workers() {
        let backend = InMemoryBackend::new();
        backend.connect().await.unwrap();

        // Publish rank 0
        backend
            .publish_metadata(
                "test-model",
                vec![WorkerMetadata {
                    worker_rank: 0,
                    nixl_metadata: vec![1],
                    tensors: vec![],
                }],
            )
            .await
            .unwrap();

        // Publish rank 1 (should merge, not replace)
        backend
            .publish_metadata(
                "test-model",
                vec![WorkerMetadata {
                    worker_rank: 1,
                    nixl_metadata: vec![2],
                    tensors: vec![],
                }],
            )
            .await
            .unwrap();

        let record = backend
            .get_metadata("test-model")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(record.workers.len(), 2);
        assert_eq!(record.workers[0].worker_rank, 0);
        assert_eq!(record.workers[1].worker_rank, 1);
    }

    #[tokio::test]
    async fn test_remove_and_list() {
        let backend = InMemoryBackend::new();
        backend.connect().await.unwrap();

        backend
            .publish_metadata(
                "model-a",
                vec![WorkerMetadata {
                    worker_rank: 0,
                    nixl_metadata: vec![],
                    tensors: vec![],
                }],
            )
            .await
            .unwrap();

        backend
            .publish_metadata(
                "model-b",
                vec![WorkerMetadata {
                    worker_rank: 0,
                    nixl_metadata: vec![],
                    tensors: vec![],
                }],
            )
            .await
            .unwrap();

        let models = backend.list_models().await.unwrap();
        assert_eq!(models.len(), 2);

        backend.remove_metadata("model-a").await.unwrap();

        let models = backend.list_models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert!(models.contains(&"model-b".to_string()));
    }
}
