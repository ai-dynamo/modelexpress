// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metadata backend abstraction for P2P model metadata.
//!
//! Supports two persistent backends:
//! - **Redis**: Persistent storage via Redis keys + atomic Lua merge
//! - **Kubernetes**: CRDs and ConfigMaps for native K8s integration
//!
//! Select the backend via `MX_METADATA_BACKEND=redis` or `MX_METADATA_BACKEND=kubernetes`.

use async_trait::async_trait;
use modelexpress_common::grpc::p2p::WorkerMetadata;
use std::sync::Arc;

pub mod kubernetes;
pub mod redis;

/// Result type for metadata operations
pub type MetadataResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Model metadata record returned from backends
#[derive(Debug, Clone)]
pub struct ModelMetadataRecord {
    pub model_name: String,
    pub workers: Vec<WorkerRecord>,
    pub published_at: i64,
}

/// Worker metadata record
#[derive(Debug, Clone)]
pub struct WorkerRecord {
    pub worker_rank: u32,
    pub nixl_metadata: Vec<u8>,
    pub tensors: Vec<TensorRecord>,
    /// Worker lifecycle status (maps to `SourceStatus` proto enum)
    pub status: i32,
    /// Timestamp of last status update (unix millis)
    pub updated_at: i64,
}

/// Tensor descriptor record
#[derive(Debug, Clone)]
pub struct TensorRecord {
    pub name: String,
    pub addr: u64,
    pub size: u64,
    pub device_id: u32,
    pub dtype: String,
}

// Conversions from gRPC types
impl From<WorkerMetadata> for WorkerRecord {
    fn from(meta: WorkerMetadata) -> Self {
        Self {
            worker_rank: meta.worker_rank,
            nixl_metadata: meta.nixl_metadata,
            tensors: meta.tensors.into_iter().map(TensorRecord::from).collect(),
            status: meta.status,
            updated_at: meta.updated_at,
        }
    }
}

impl From<modelexpress_common::grpc::p2p::TensorDescriptor> for TensorRecord {
    fn from(desc: modelexpress_common::grpc::p2p::TensorDescriptor) -> Self {
        Self {
            name: desc.name,
            addr: desc.addr,
            size: desc.size,
            device_id: desc.device_id,
            dtype: desc.dtype,
        }
    }
}

// Conversions back to gRPC types
impl From<WorkerRecord> for WorkerMetadata {
    fn from(record: WorkerRecord) -> Self {
        Self {
            worker_rank: record.worker_rank,
            nixl_metadata: record.nixl_metadata,
            tensors: record
                .tensors
                .into_iter()
                .map(modelexpress_common::grpc::p2p::TensorDescriptor::from)
                .collect(),
            status: record.status,
            updated_at: record.updated_at,
        }
    }
}

impl From<TensorRecord> for modelexpress_common::grpc::p2p::TensorDescriptor {
    fn from(record: TensorRecord) -> Self {
        Self {
            name: record.name,
            addr: record.addr,
            size: record.size,
            device_id: record.device_id,
            dtype: record.dtype,
        }
    }
}

/// Trait for metadata backend implementations
#[cfg_attr(test, mockall::automock)]
#[async_trait]
pub trait MetadataBackend: Send + Sync {
    /// Connect to the backend (initialize connections, etc.)
    async fn connect(&self) -> MetadataResult<()>;

    /// Publish metadata for a model.
    /// NOTE: This should MERGE workers with existing data for incremental publishing.
    async fn publish_metadata(
        &self,
        model_name: &str,
        workers: Vec<WorkerMetadata>,
    ) -> MetadataResult<()>;

    /// Get metadata for a model
    async fn get_metadata(&self, model_name: &str) -> MetadataResult<Option<ModelMetadataRecord>>;

    /// Remove metadata for a model
    async fn remove_metadata(&self, model_name: &str) -> MetadataResult<()>;

    /// List all registered model names
    async fn list_models(&self) -> MetadataResult<Vec<String>>;

    /// Patch the status of a worker within the stored metadata record.
    /// `status` is the `SourceStatus` proto enum value; `updated_at` is unix millis.
    async fn update_status(
        &self,
        model_name: &str,
        worker_id: u32,
        status: i32,
        updated_at: i64,
    ) -> MetadataResult<()>;
}

/// Configuration for metadata backends
#[derive(Debug, Clone)]
pub enum BackendConfig {
    /// Redis backend — persistent, horizontally scalable
    Redis { url: String },
    /// Kubernetes CRD backend — native K8s integration
    Kubernetes { namespace: String },
}

impl BackendConfig {
    /// Create backend config from environment variables.
    ///
    /// `MX_METADATA_BACKEND` is required. Valid values:
    /// - `redis`: Redis
    /// - `kubernetes` | `k8s` | `crd`: Kubernetes CRD
    pub fn from_env() -> Result<Self, String> {
        let backend_type = std::env::var("MX_METADATA_BACKEND").unwrap_or_default();
        let redis_url = Self::redis_url_from_env();
        let k8s_namespace = Self::k8s_namespace_from_env();
        Self::from_type_str(&backend_type, &redis_url, &k8s_namespace)
    }

    /// Parse a backend type string into a config. Testable without env vars.
    pub fn from_type_str(
        backend_type: &str,
        redis_url: &str,
        k8s_namespace: &str,
    ) -> Result<Self, String> {
        match backend_type.to_lowercase().as_str() {
            "redis" => Ok(Self::Redis {
                url: redis_url.to_string(),
            }),
            "kubernetes" | "k8s" | "crd" => Ok(Self::Kubernetes {
                namespace: k8s_namespace.to_string(),
            }),
            other => Err(format!(
                "MX_METADATA_BACKEND='{}' is not valid. Use 'redis' or 'kubernetes'.",
                other
            )),
        }
    }

    pub fn redis_url_from_env() -> String {
        if let Ok(url) = std::env::var("REDIS_URL") {
            return url;
        }
        let host = std::env::var("MX_REDIS_HOST")
            .or_else(|_| std::env::var("REDIS_HOST"))
            .unwrap_or_else(|_| "localhost".to_string());
        let port = std::env::var("MX_REDIS_PORT")
            .or_else(|_| std::env::var("REDIS_PORT"))
            .unwrap_or_else(|_| "6379".to_string());
        format!("redis://{}:{}", host, port)
    }

    fn k8s_namespace_from_env() -> String {
        std::env::var("MX_METADATA_NAMESPACE")
            .or_else(|_| std::env::var("POD_NAMESPACE"))
            .unwrap_or_else(|_| "default".to_string())
    }
}

/// Create a backend from configuration.
pub async fn create_backend(config: BackendConfig) -> MetadataResult<Arc<dyn MetadataBackend>> {
    match config {
        BackendConfig::Redis { url } => {
            let backend = redis::RedisBackend::new(&url);
            backend.connect().await?;
            Ok(Arc::new(backend) as Arc<dyn MetadataBackend>)
        }
        BackendConfig::Kubernetes { namespace } => {
            let backend = kubernetes::KubernetesBackend::new(&namespace).await?;
            backend.connect().await?;
            Ok(Arc::new(backend) as Arc<dyn MetadataBackend>)
        }
    }
}
