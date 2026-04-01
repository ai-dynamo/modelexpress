// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! State management for P2P model metadata.
//!
//! `P2pStateManager` wraps a metadata backend (Redis or Kubernetes CRD).
//! All state — model metadata and source status — is persisted to the backend,
//! making the server stateless and horizontally scalable.

use crate::metadata_backend::{BackendConfig, MetadataBackend, MetadataResult, create_backend};
use modelexpress_common::grpc::p2p::{SourceIdentity, WorkerMetadata};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

// Re-export types for backwards compatibility
pub use crate::metadata_backend::{
    BackendMetadataRecord, ModelMetadataRecord, TensorRecord, WorkerRecord,
};

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

    /// Initialize the backend connection. Returns the backend type name on success.
    pub async fn connect(&self) -> MetadataResult<String> {
        let config = self.config.clone().ok_or(
            "MX_METADATA_BACKEND is not set or invalid. Set it to 'redis' or 'kubernetes'.",
        )?;

        let backend_name = config.to_string();
        let backend = create_backend(config).await?;
        let mut guard = self.backend.write().await;
        *guard = Some(backend);

        info!("P2pStateManager connected (backend: {})", backend_name);
        Ok(backend_name)
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

    /// Publish metadata for a source instance.
    pub async fn publish_metadata(
        &self,
        identity: &SourceIdentity,
        worker_id: &str,
        worker: WorkerMetadata,
    ) -> MetadataResult<()> {
        self.get_backend()
            .await?
            .publish_metadata(identity, worker_id, worker)
            .await
    }

    /// Get full tensor metadata for one specific instance.
    pub async fn get_metadata(
        &self,
        source_id: &str,
        worker_id: &str,
    ) -> MetadataResult<Option<ModelMetadataRecord>> {
        self.get_backend()
            .await?
            .get_metadata(source_id, worker_id)
            .await
    }

    /// List available source instances, optionally filtered by status.
    pub async fn list_workers(
        &self,
        source_id: Option<String>,
        status_filter: Option<modelexpress_common::grpc::p2p::SourceStatus>,
    ) -> MetadataResult<Vec<crate::metadata_backend::SourceInstanceInfo>> {
        self.get_backend()
            .await?
            .list_workers(source_id, status_filter)
            .await
    }

    /// Remove metadata by mx_source_id.
    pub async fn remove_metadata(&self, source_id: &str) -> MetadataResult<()> {
        self.get_backend().await?.remove_metadata(source_id).await
    }

    /// Remove a single worker by source_id and worker_id.
    pub async fn remove_worker(&self, source_id: &str, worker_id: &str) -> MetadataResult<()> {
        self.get_backend()
            .await?
            .remove_worker(source_id, worker_id)
            .await
    }

    /// List all registered source IDs and model names.
    pub async fn list_sources(&self) -> MetadataResult<Vec<(String, String)>> {
        self.get_backend().await?.list_sources().await
    }

    // ========================================================================
    // Worker Status
    // ========================================================================

    /// Update the status of a worker within its stored metadata record.
    pub async fn update_worker_status(
        &self,
        source_id: &str,
        worker_id: &str,
        worker_rank: u32,
        status: modelexpress_common::grpc::p2p::SourceStatus,
    ) -> MetadataResult<()> {
        let updated_at = chrono::Utc::now().timestamp_millis();
        self.get_backend()
            .await?
            .update_status(source_id, worker_id, worker_rank, status, updated_at)
            .await?;

        debug!(
            "Updated status for source '{}' worker '{}' rank {} -> {}",
            source_id, worker_id, worker_rank, status as i32
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
    use modelexpress_common::grpc::p2p::{
        MxSourceType, SourceIdentity, SourceStatus, TensorDescriptor,
    };

    fn test_identity() -> SourceIdentity {
        SourceIdentity {
            mx_version: "0.3.0".to_string(),
            mx_source_type: MxSourceType::Weights as i32,
            model_name: "my-model".to_string(),
            backend_framework: 1,
            tensor_parallel_size: 8,
            pipeline_parallel_size: 1,
            expert_parallel_size: 0,
            dtype: "bfloat16".to_string(),
            quantization: String::new(),
            extra_parameters: Default::default(),
        }
    }

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
        use modelexpress_common::grpc::p2p::worker_metadata::BackendMetadata;

        let meta = WorkerMetadata {
            worker_rank: 3,
            backend_metadata: Some(BackendMetadata::NixlMetadata(vec![1, 2, 3, 4, 5])),
            tensors: vec![TensorDescriptor {
                name: "test.weight".to_string(),
                addr: 0x1000,
                size: 4096,
                device_id: 3,
                dtype: "float16".to_string(),
            }],
            status: SourceStatus::Initializing as i32,
            updated_at: 1234567890000,
            ..Default::default()
        };

        let record = WorkerRecord::from(meta.clone());
        assert_eq!(record.worker_rank, 3);
        assert!(matches!(
            &record.backend_metadata,
            BackendMetadataRecord::Nixl(d) if d == &vec![1, 2, 3, 4, 5]
        ));
        assert_eq!(record.tensors.len(), 1);
        assert_eq!(record.status, SourceStatus::Initializing as i32);
        assert_eq!(record.updated_at, 1234567890000);

        let back: WorkerMetadata = record.into();
        assert_eq!(back.worker_rank, meta.worker_rank);
        assert_eq!(back.backend_metadata, meta.backend_metadata);
    }

    #[test]
    fn test_worker_record_transfer_engine_roundtrip() {
        use modelexpress_common::grpc::p2p::worker_metadata::BackendMetadata;

        let meta = WorkerMetadata {
            worker_rank: 1,
            backend_metadata: Some(BackendMetadata::TransferEngineSessionId(
                "192.168.1.10:12345".to_string(),
            )),
            tensors: vec![TensorDescriptor {
                name: "test.weight".to_string(),
                addr: 0x2000,
                size: 8192,
                device_id: 0,
                dtype: "float16".to_string(),
            }],
            status: 0,
            updated_at: 0,
            ..Default::default()
        };

        let record = WorkerRecord::from(meta.clone());
        assert_eq!(record.worker_rank, 1);
        assert!(matches!(
            &record.backend_metadata,
            BackendMetadataRecord::TransferEngine(sid) if sid == "192.168.1.10:12345"
        ));
        assert_eq!(
            record.backend_metadata.backend_type_str(),
            "transfer_engine"
        );

        let back: WorkerMetadata = record.into();
        assert_eq!(back.worker_rank, meta.worker_rank);
        assert_eq!(back.backend_metadata, meta.backend_metadata);
    }

    #[test]
    fn test_backend_metadata_from_flat_with_discriminator() {
        // Explicit backend_type takes precedence
        let te = BackendMetadataRecord::from_flat(
            Vec::new(),
            Some("10.0.0.1:5000".into()),
            Some("transfer_engine"),
        );
        assert!(matches!(te, BackendMetadataRecord::TransferEngine(ref s) if s == "10.0.0.1:5000"));

        let nixl = BackendMetadataRecord::from_flat(vec![1, 2, 3], None, Some("nixl"));
        assert!(matches!(nixl, BackendMetadataRecord::Nixl(ref d) if d == &vec![1, 2, 3]));

        let none = BackendMetadataRecord::from_flat(Vec::new(), None, Some("none"));
        assert!(matches!(none, BackendMetadataRecord::None));

        // Backwards compat: missing backend_type infers from fields
        let inferred_te =
            BackendMetadataRecord::from_flat(Vec::new(), Some("10.0.0.1:5000".into()), None);
        assert!(matches!(
            inferred_te,
            BackendMetadataRecord::TransferEngine(_)
        ));

        let inferred_nixl = BackendMetadataRecord::from_flat(vec![1, 2], None, None);
        assert!(matches!(inferred_nixl, BackendMetadataRecord::Nixl(_)));

        let inferred_none = BackendMetadataRecord::from_flat(Vec::new(), None, None);
        assert!(matches!(inferred_none, BackendMetadataRecord::None));
    }

    #[test]
    fn test_model_record_creation() {
        let record = ModelMetadataRecord {
            source_id: "abc123def456abcd".to_string(),
            worker_id: "test-instance-id".to_string(),
            model_name: "meta-llama/Llama-3.1-70B".to_string(),
            workers: vec![
                WorkerRecord {
                    worker_rank: 0,
                    backend_metadata: BackendMetadataRecord::Nixl(vec![10, 20, 30]),
                    tensors: vec![TensorRecord {
                        name: "layer.0.weight".to_string(),
                        addr: 0x7f00_0000_0000,
                        size: 1_000_000,
                        device_id: 0,
                        dtype: "bfloat16".to_string(),
                    }],
                    status: SourceStatus::Ready as i32,
                    updated_at: 1234567890000,
                    metadata_endpoint: String::new(),
                    agent_name: String::new(),
                    worker_grpc_endpoint: String::new(),
                },
                WorkerRecord {
                    worker_rank: 1,
                    backend_metadata: BackendMetadataRecord::Nixl(vec![40, 50, 60]),
                    tensors: vec![TensorRecord {
                        name: "layer.0.weight".to_string(),
                        addr: 0x7f00_0000_0000,
                        size: 1_000_000,
                        device_id: 1,
                        dtype: "bfloat16".to_string(),
                    }],
                    status: SourceStatus::Ready as i32,
                    updated_at: 1234567890000,
                    metadata_endpoint: String::new(),
                    agent_name: String::new(),
                    worker_grpc_endpoint: String::new(),
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
    async fn test_publish_metadata_calls_backend() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_publish_metadata()
            .withf(|identity, worker_id, worker| {
                identity.model_name == "my-model"
                    && identity.tensor_parallel_size == 8
                    && worker_id == "a1b2c3d4"
                    && worker.worker_rank == 3
            })
            .once()
            .returning(|_, _, _| Ok(()));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        manager
            .publish_metadata(
                &test_identity(),
                "a1b2c3d4",
                WorkerMetadata {
                    worker_rank: 3,
                    backend_metadata: None,
                    tensors: vec![],
                    status: SourceStatus::Initializing as i32,
                    updated_at: 0,
                    ..Default::default()
                },
            )
            .await
            .expect("publish_metadata failed");
    }

    #[tokio::test]
    async fn test_publish_metadata_propagates_backend_error() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_publish_metadata()
            .once()
            .returning(|_, _, _| Err("storage unavailable".into()));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        assert!(
            manager
                .publish_metadata(&test_identity(), "a1b2c3d4", WorkerMetadata::default())
                .await
                .is_err()
        );
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
                eq("abc123def456abcd"),
                eq("test-instance"),
                eq(2u32),
                eq(SourceStatus::Ready),
                mockall::predicate::always(),
            )
            .once()
            .returning(|_, _, _, _, _| Ok(()));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        manager
            .update_worker_status("abc123def456abcd", "test-instance", 2, SourceStatus::Ready)
            .await
            .expect("update_worker_status failed");
    }

    #[tokio::test]
    async fn test_update_worker_status_propagates_backend_error() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_update_status()
            .once()
            .returning(|_, _, _, _, _| Err("redis unavailable".into()));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        assert!(
            manager
                .update_worker_status("abc123def456abcd", "test-instance", 0, SourceStatus::Ready)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_list_workers_calls_backend() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers()
            .withf(|source_id, status_filter| {
                source_id.as_deref() == Some("abc123def456abcd")
                    && *status_filter == Some(SourceStatus::Ready)
            })
            .once()
            .returning(|_, _| {
                Ok(vec![crate::metadata_backend::SourceInstanceInfo {
                    source_id: "abc123def456abcd".to_string(),
                    worker_id: "w1".to_string(),
                    model_name: "my-model".to_string(),
                    worker_rank: 0,
                    status: SourceStatus::Ready as i32,
                    updated_at: 1234567890000,
                }])
            });

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        let result = manager
            .list_workers(
                Some("abc123def456abcd".to_string()),
                Some(SourceStatus::Ready),
            )
            .await
            .expect("list_workers failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].worker_id, "w1");
    }

    #[tokio::test]
    async fn test_list_workers_propagates_backend_error() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers()
            .once()
            .returning(|_, _| Err("backend error".into()));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        assert!(manager.list_workers(None, None).await.is_err());
    }

    #[tokio::test]
    async fn test_remove_metadata_calls_backend() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_remove_metadata()
            .with(eq("abc123def456abcd"))
            .once()
            .returning(|_| Ok(()));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        manager
            .remove_metadata("abc123def456abcd")
            .await
            .expect("remove_metadata failed");
    }

    #[tokio::test]
    async fn test_remove_metadata_propagates_backend_error() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_remove_metadata()
            .once()
            .returning(|_| Err("delete failed".into()));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        assert!(manager.remove_metadata("abc123def456abcd").await.is_err());
    }

    #[tokio::test]
    async fn test_list_sources_calls_backend() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_sources()
            .once()
            .returning(|| Ok(vec![("src1".to_string(), "model-a".to_string())]));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        let result = manager.list_sources().await.expect("list_sources failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "src1");
        assert_eq!(result[0].1, "model-a");
    }

    #[tokio::test]
    async fn test_update_worker_status_stores_correct_status() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_update_status()
            .withf(|source_id, worker_id, worker_rank, status, _updated_at| {
                source_id == "abc123def456abcd"
                    && worker_id == "test-instance"
                    && *worker_rank == 7
                    && *status == SourceStatus::Ready
            })
            .once()
            .returning(|_, _, _, _, _| Ok(()));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        manager
            .update_worker_status("abc123def456abcd", "test-instance", 7, SourceStatus::Ready)
            .await
            .expect("update_worker_status failed");
    }
}
