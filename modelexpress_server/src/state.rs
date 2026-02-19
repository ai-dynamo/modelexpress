// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! State management for P2P model metadata.
//!
//! This module provides the `P2pStateManager` which wraps a metadata backend
//! (Redis or Kubernetes CRD) for storing model metadata.
//!
//! For new code, prefer using `metadata_backend` directly. This module exists
//! for backwards compatibility with existing code.

use crate::metadata_backend::{BackendConfig, MetadataBackend, MetadataResult, create_backend};
use modelexpress_common::grpc::p2p::WorkerMetadata;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

// Re-export types for backwards compatibility
pub use crate::metadata_backend::{ModelMetadataRecord, TensorRecord, WorkerRecord};

/// Ready state for a source worker (stored in-memory, always ephemeral).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadyRecord {
    pub session_id: String,
    pub metadata_hash: String,
    pub nixl_ready: bool,
    pub stability_verified: bool,
    pub timestamp: f64,
}

/// State manager that handles P2P metadata operations.
///
/// This is a wrapper around the metadata backend abstraction that provides
/// a simpler API for common operations. It automatically selects the backend
/// based on environment variables (MX_METADATA_BACKEND).
#[derive(Clone)]
pub struct P2pStateManager {
    backend: Arc<RwLock<Option<Arc<dyn MetadataBackend>>>>,
    config: BackendConfig,
    /// In-memory ready flags (always ephemeral - not persisted to backend).
    /// Key: "{model_name}:worker:{worker_id}"
    ready_flags: Arc<RwLock<HashMap<String, ReadyRecord>>>,
}

impl P2pStateManager {
    /// Create a new state manager with default configuration from environment.
    ///
    /// The `redis_url` is used as a fallback when `MX_METADATA_BACKEND=redis`
    /// and no `REDIS_URL` env var is set. The default backend is in-memory.
    pub fn new(redis_url: &str) -> Self {
        let mut config = BackendConfig::from_env();

        // If the env resolved to a layered-redis or redis-only config but
        // the URL came from the default, override with the explicit redis_url
        // passed by main.rs for backward compatibility.
        match &mut config {
            BackendConfig::LayeredRedis { url } | BackendConfig::Redis { url }
                if std::env::var("REDIS_URL").is_err() =>
            {
                *url = redis_url.to_string();
            }
            _ => {}
        }

        Self {
            backend: Arc::new(RwLock::new(None)),
            config,
            ready_flags: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new state manager with explicit backend configuration
    pub fn with_config(config: BackendConfig) -> Self {
        Self {
            backend: Arc::new(RwLock::new(None)),
            config,
            ready_flags: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize the backend connection
    pub async fn connect(&self) -> MetadataResult<()> {
        let backend = create_backend(self.config.clone()).await?;

        let mut guard = self.backend.write().await;
        *guard = Some(backend);

        info!("P2pStateManager connected with {:?}", self.config);
        Ok(())
    }

    /// Get the backend, connecting if necessary.
    /// Uses double-checked locking to prevent duplicate backend creation
    /// when multiple callers race on first access.
    async fn get_backend(&self) -> MetadataResult<Arc<dyn MetadataBackend>> {
        // Fast path: read lock
        {
            let guard = self.backend.read().await;
            if let Some(backend) = guard.as_ref() {
                return Ok(backend.clone());
            }
        }

        // Slow path: write lock with double-check
        let mut guard = self.backend.write().await;
        if let Some(backend) = guard.as_ref() {
            return Ok(backend.clone());
        }

        let backend = create_backend(self.config.clone()).await?;
        info!("P2pStateManager connected with {:?}", self.config);
        *guard = Some(backend.clone());
        Ok(backend)
    }

    // ========================================================================
    // Model Metadata Management
    // ========================================================================

    /// Publish metadata for a model
    /// NOTE: This MERGES workers with existing data, allowing incremental publishing
    /// from multiple workers in a distributed system.
    pub async fn publish_metadata(
        &self,
        model_name: &str,
        workers: Vec<WorkerMetadata>,
    ) -> MetadataResult<()> {
        let backend = self.get_backend().await?;
        backend.publish_metadata(model_name, workers).await
    }

    /// Get metadata for a model
    pub async fn get_metadata(
        &self,
        model_name: &str,
    ) -> MetadataResult<Option<ModelMetadataRecord>> {
        let backend = self.get_backend().await?;
        backend.get_metadata(model_name).await
    }

    /// Remove metadata for a model (cleanup)
    pub async fn remove_metadata(&self, model_name: &str) -> MetadataResult<()> {
        let backend = self.get_backend().await?;
        backend.remove_metadata(model_name).await
    }

    /// List all registered model names
    pub async fn list_models(&self) -> MetadataResult<Vec<String>> {
        let backend = self.get_backend().await?;
        backend.list_models().await
    }

    // ========================================================================
    // Ready Coordination (always in-memory, ephemeral)
    // ========================================================================

    /// Publish a source ready flag for a worker.
    pub async fn publish_ready(
        &self,
        model_name: &str,
        worker_id: u32,
        session_id: &str,
        metadata_hash: &str,
        nixl_ready: bool,
        stability_verified: bool,
    ) -> MetadataResult<()> {
        let key = format!("{}:worker:{}", model_name, worker_id);
        let record = ReadyRecord {
            session_id: session_id.to_string(),
            metadata_hash: metadata_hash.to_string(),
            nixl_ready,
            stability_verified,
            timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
        };

        let mut flags = self.ready_flags.write().await;
        flags.insert(key, record);

        info!(
            "Published ready flag for model '{}' worker {}: nixl_ready={}, stability_verified={}",
            model_name, worker_id, nixl_ready, stability_verified
        );
        Ok(())
    }

    /// Get the ready flag for a specific worker.
    pub async fn get_ready(
        &self,
        model_name: &str,
        worker_id: u32,
    ) -> MetadataResult<Option<ReadyRecord>> {
        let key = format!("{}:worker:{}", model_name, worker_id);
        let flags = self.ready_flags.read().await;
        let result = flags.get(&key).cloned();

        debug!(
            "get_ready '{}' worker {}: {}",
            model_name,
            worker_id,
            if result.is_some() {
                "found"
            } else {
                "not found"
            }
        );
        Ok(result)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use modelexpress_common::grpc::p2p::TensorDescriptor;

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
        };

        let record = WorkerRecord::from(meta.clone());
        assert_eq!(record.worker_rank, 3);
        assert_eq!(record.nixl_metadata, vec![1, 2, 3, 4, 5]);
        assert_eq!(record.tensors.len(), 1);

        let back: WorkerMetadata = record.into();
        assert_eq!(back.worker_rank, meta.worker_rank);
        assert_eq!(back.nixl_metadata, meta.nixl_metadata);
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
    fn test_backend_config_from_env() {
        // SAFETY: Test runs in isolation, no concurrent access to env vars
        unsafe {
            // Test default (Memory)
            std::env::remove_var("MX_METADATA_BACKEND");
            let config = BackendConfig::from_env();
            assert!(matches!(config, BackendConfig::Memory));

            // Test Redis
            std::env::set_var("MX_METADATA_BACKEND", "redis");
            let config = BackendConfig::from_env();
            assert!(matches!(config, BackendConfig::LayeredRedis { .. }));

            // Test Kubernetes
            std::env::set_var("MX_METADATA_BACKEND", "kubernetes");
            std::env::set_var("MX_METADATA_NAMESPACE", "test-ns");
            let config = BackendConfig::from_env();
            assert!(
                matches!(config, BackendConfig::LayeredKubernetes { namespace } if namespace == "test-ns")
            );

            // Cleanup
            std::env::remove_var("MX_METADATA_BACKEND");
            std::env::remove_var("MX_METADATA_NAMESPACE");
        }
    }
}
