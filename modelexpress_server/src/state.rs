// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! State management for P2P model metadata.
//!
//! `P2pStateManager` wraps a metadata backend (Redis or Kubernetes CRD).
//! All state — model metadata and source status — is persisted to the backend,
//! making the server stateless and horizontally scalable.

use crate::metadata_backend::{BackendConfig, MetadataBackend, MetadataResult, create_backend};
use modelexpress_common::grpc::p2p::WorkerMetadata;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

// Re-export types for backwards compatibility
pub use crate::metadata_backend::{ModelMetadataRecord, TensorRecord, WorkerRecord};

/// State manager that handles P2P metadata operations.
///
/// Wraps the metadata backend abstraction and provides a simpler API for
/// common operations. Configure via `MX_METADATA_BACKEND` env var.
#[derive(Clone)]
pub struct P2pStateManager {
    backend: Arc<RwLock<Option<Arc<dyn MetadataBackend>>>>,
    config: Option<BackendConfig>,
}

impl Default for P2pStateManager {
    fn default() -> Self {
        Self::new()
    }
}

impl P2pStateManager {
    /// Create a new state manager, resolving backend config from the environment.
    ///
    /// Configure via `MX_METADATA_BACKEND` (required) and `REDIS_URL` /
    /// `MX_REDIS_HOST` / `MX_REDIS_PORT` (for Redis).
    pub fn new() -> Self {
        Self {
            backend: Arc::new(RwLock::new(None)),
            config: BackendConfig::from_env().ok(),
        }
    }

    /// Create a new state manager with an explicit backend configuration.
    pub fn with_config(config: BackendConfig) -> Self {
        Self {
            backend: Arc::new(RwLock::new(None)),
            config: Some(config),
        }
    }

    /// Inject a pre-built backend directly (test only).
    #[cfg(test)]
    pub fn with_backend(backend: Arc<dyn MetadataBackend>) -> Self {
        Self {
            backend: Arc::new(RwLock::new(Some(backend))),
            config: None,
        }
    }

    /// Initialize the backend connection.
    pub async fn connect(&self) -> MetadataResult<()> {
        let config = self.config.clone().ok_or(
            "MX_METADATA_BACKEND is not set or invalid. Set it to 'redis' or 'kubernetes'.",
        )?;

        let backend = create_backend(config.clone()).await?;
        let mut guard = self.backend.write().await;
        *guard = Some(backend);

        info!("P2pStateManager connected with {:?}", config);
        Ok(())
    }

    /// Get the backend, connecting lazily if not yet connected.
    async fn get_backend(&self) -> MetadataResult<Arc<dyn MetadataBackend>> {
        {
            let guard = self.backend.read().await;
            if let Some(backend) = guard.as_ref() {
                return Ok(backend.clone());
            }
        }

        let mut guard = self.backend.write().await;
        if let Some(backend) = guard.as_ref() {
            return Ok(backend.clone());
        }

        let config = self.config.clone().ok_or(
            "MX_METADATA_BACKEND is not set or invalid. Set it to 'redis' or 'kubernetes'.",
        )?;

        let backend = create_backend(config.clone()).await?;
        info!("P2pStateManager connected with {:?}", config);
        *guard = Some(backend.clone());
        Ok(backend)
    }

    // ========================================================================
    // Model Metadata
    // ========================================================================

    /// Publish metadata for a model (merges workers with existing data).
    pub async fn publish_metadata(
        &self,
        model_name: &str,
        workers: Vec<WorkerMetadata>,
    ) -> MetadataResult<()> {
        self.get_backend()
            .await?
            .publish_metadata(model_name, workers)
            .await
    }

    /// Get metadata for a model.
    pub async fn get_metadata(
        &self,
        model_name: &str,
    ) -> MetadataResult<Option<ModelMetadataRecord>> {
        self.get_backend().await?.get_metadata(model_name).await
    }

    /// Remove metadata for a model.
    pub async fn remove_metadata(&self, model_name: &str) -> MetadataResult<()> {
        self.get_backend().await?.remove_metadata(model_name).await
    }

    /// List all registered model names.
    pub async fn list_models(&self) -> MetadataResult<Vec<String>> {
        self.get_backend().await?.list_models().await
    }

    // ========================================================================
    // Worker Status
    // ========================================================================

    /// Update the status of a worker within its stored metadata record.
    /// `status` is the `SourceStatus` proto enum value (i32).
    pub async fn update_worker_status(
        &self,
        model_name: &str,
        worker_id: u32,
        status: i32,
    ) -> MetadataResult<()> {
        let updated_at = chrono::Utc::now().timestamp_millis();
        self.get_backend()
            .await?
            .update_status(model_name, worker_id, status, updated_at)
            .await?;

        debug!(
            "Updated status for model '{}' worker {} -> {}",
            model_name, worker_id, status
        );
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::metadata_backend::MockMetadataBackend;
    use mockall::predicate::eq;
    use modelexpress_common::grpc::p2p::{SourceStatus, TensorDescriptor};

    #[test]
    fn test_tensor_record_conversion() {
        let desc = TensorDescriptor {
            name: "model.layers.0.weight".to_string(),
            addr: 0x7f0000000000,
            size: 1024 * 1024 * 1024,
            device_id: 0,
            dtype: "bfloat16".to_string(),
        };

        let record = TensorRecord::from(desc.clone());
        assert_eq!(record.name, "model.layers.0.weight");
        assert_eq!(record.size, 1024 * 1024 * 1024);

        let back: TensorDescriptor = record.into();
        assert_eq!(back.name, desc.name);
        assert_eq!(back.addr, desc.addr);
    }

    #[test]
    fn test_worker_record_conversion() {
        let meta = WorkerMetadata {
            worker_rank: 3,
            nixl_metadata: vec![1, 2, 3, 4, 5],
            tensors: vec![TensorDescriptor {
                name: "test.weight".to_string(),
                addr: 0x1000,
                size: 4096,
                device_id: 3,
                dtype: "float16".to_string(),
            }],
            status: SourceStatus::Initializing as i32,
            updated_at: 1234567890000,
        };

        let record = WorkerRecord::from(meta.clone());
        assert_eq!(record.worker_rank, 3);
        assert_eq!(record.nixl_metadata, vec![1, 2, 3, 4, 5]);
        assert_eq!(record.tensors.len(), 1);
        assert_eq!(record.status, SourceStatus::Initializing as i32);
        assert_eq!(record.updated_at, 1234567890000);

        let back: WorkerMetadata = record.into();
        assert_eq!(back.worker_rank, meta.worker_rank);
        assert_eq!(back.nixl_metadata, meta.nixl_metadata);
        assert_eq!(back.status, meta.status);
        assert_eq!(back.updated_at, meta.updated_at);
    }

    #[test]
    fn test_model_record_creation() {
        let record = ModelMetadataRecord {
            model_name: "meta-llama/Llama-3.1-70B".to_string(),
            workers: vec![
                WorkerRecord {
                    worker_rank: 0,
                    nixl_metadata: vec![10, 20, 30],
                    tensors: vec![TensorRecord {
                        name: "layer.0.weight".to_string(),
                        addr: 0x7f00_0000_0000,
                        size: 1_000_000,
                        device_id: 0,
                        dtype: "bfloat16".to_string(),
                    }],
                    status: SourceStatus::Ready as i32,
                    updated_at: 1234567890000,
                },
                WorkerRecord {
                    worker_rank: 1,
                    nixl_metadata: vec![40, 50, 60],
                    tensors: vec![TensorRecord {
                        name: "layer.0.weight".to_string(),
                        addr: 0x7f00_0000_0000,
                        size: 1_000_000,
                        device_id: 1,
                        dtype: "bfloat16".to_string(),
                    }],
                    status: SourceStatus::Ready as i32,
                    updated_at: 1234567890000,
                },
            ],
            published_at: 1234567890,
        };

        assert_eq!(record.model_name, "meta-llama/Llama-3.1-70B");
        assert_eq!(record.workers.len(), 2);
        assert_eq!(record.workers[0].worker_rank, 0);
        assert_eq!(record.workers[1].worker_rank, 1);
    }

    #[test]
    fn test_backend_config_parsing() {
        let default_redis = "redis://localhost:6379";
        let default_ns = "default";

        // Redis
        let config =
            BackendConfig::from_type_str("redis", "redis://myhost:6379", default_ns).expect("ok");
        assert!(matches!(config, BackendConfig::Redis { url } if url == "redis://myhost:6379"));

        // Kubernetes aliases
        for alias in &["kubernetes", "k8s", "crd"] {
            let config = BackendConfig::from_type_str(alias, default_redis, "prod-ns").expect("ok");
            assert!(
                matches!(config, BackendConfig::Kubernetes { namespace } if namespace == "prod-ns")
            );
        }

        // Unknown returns Err
        assert!(BackendConfig::from_type_str("bogus", default_redis, default_ns).is_err());
        assert!(BackendConfig::from_type_str("memory", default_redis, default_ns).is_err());
        assert!(BackendConfig::from_type_str("", default_redis, default_ns).is_err());

        // Case insensitive
        let config = BackendConfig::from_type_str("REDIS", default_redis, default_ns).expect("ok");
        assert!(matches!(config, BackendConfig::Redis { .. }));
    }

    #[tokio::test]
    async fn test_connect_fails_without_config() {
        let manager = P2pStateManager {
            backend: Arc::new(RwLock::new(None)),
            config: None,
        };
        assert!(manager.connect().await.is_err());
    }

    #[tokio::test]
    async fn test_update_worker_status_calls_backend() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_update_status()
            .with(
                eq("my-model"),
                eq(2u32),
                eq(2i32),
                mockall::predicate::always(),
            )
            .once()
            .returning(|_, _, _, _| Ok(()));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        manager
            .update_worker_status("my-model", 2, SourceStatus::Ready as i32)
            .await
            .expect("update_worker_status failed");
    }

    #[tokio::test]
    async fn test_update_worker_status_propagates_backend_error() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_update_status()
            .once()
            .returning(|_, _, _, _| Err("redis unavailable".into()));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        assert!(
            manager
                .update_worker_status("my-model", 0, SourceStatus::Ready as i32)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_update_worker_status_stores_correct_status() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_update_status()
            .withf(|model, worker_id, status, _updated_at| {
                model == "deepseek-ai/DeepSeek-V3"
                    && *worker_id == 7
                    && *status == SourceStatus::Ready as i32
            })
            .once()
            .returning(|_, _, _, _| Ok(()));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        manager
            .update_worker_status("deepseek-ai/DeepSeek-V3", 7, SourceStatus::Ready as i32)
            .await
            .expect("update_worker_status failed");
    }
}
