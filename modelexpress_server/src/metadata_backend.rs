// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metadata backend abstraction for P2P model metadata.
//!
//! Supports multiple backends:
//! - **Memory**: In-memory cache, zero dependencies, lowest latency (default)
//! - **Redis**: Persistent storage via Redis keys + atomic Lua merge
//! - **Kubernetes**: CRDs and ConfigMaps for native K8s integration
//! - **Layered**: In-memory cache + write-through to Redis or Kubernetes for HA
//!
//! The layered backend is the recommended production configuration:
//! all reads hit the in-memory cache while writes are persisted for redundancy.

use async_trait::async_trait;
use modelexpress_common::grpc::p2p::WorkerMetadata;
use std::sync::Arc;

pub mod kubernetes;
pub mod layered;
pub mod memory;
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
#[async_trait]
pub trait MetadataBackend: Send + Sync {
    /// Connect to the backend (initialize connections, etc.)
    async fn connect(&self) -> MetadataResult<()>;

    /// Publish metadata for a model
    /// NOTE: This should MERGE workers with existing data for incremental publishing
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
}

/// Configuration for metadata backends
#[derive(Debug, Clone)]
pub enum BackendConfig {
    /// In-memory only (default) — zero dependencies, data lost on restart
    Memory,
    /// Redis backend — persistent, supports HA
    Redis { url: String },
    /// Kubernetes CRD backend — native K8s integration
    Kubernetes { namespace: String },
    /// Layered: in-memory cache + Redis persistence for HA
    LayeredRedis { url: String },
    /// Layered: in-memory cache + Kubernetes CRD persistence for HA
    LayeredKubernetes { namespace: String },
}

impl BackendConfig {
    /// Create backend from environment variables.
    ///
    /// `MX_METADATA_BACKEND` selects the backend type:
    /// - `memory` (default): in-memory only, zero dependencies
    /// - `redis`: Redis with in-memory cache (layered)
    /// - `kubernetes` | `k8s` | `crd`: K8s CRD with in-memory cache (layered)
    /// - `redis-only`: Redis without in-memory cache (legacy)
    /// - `kubernetes-only`: K8s CRD without in-memory cache
    pub fn from_env() -> Self {
        let backend_type =
            std::env::var("MX_METADATA_BACKEND").unwrap_or_else(|_| "memory".to_string());

        match backend_type.to_lowercase().as_str() {
            "redis" => {
                let url = Self::redis_url_from_env();
                Self::LayeredRedis { url }
            }
            "redis-only" => {
                let url = Self::redis_url_from_env();
                Self::Redis { url }
            }
            "kubernetes" | "k8s" | "crd" => {
                let namespace = Self::k8s_namespace_from_env();
                Self::LayeredKubernetes { namespace }
            }
            "kubernetes-only" | "k8s-only" | "crd-only" => {
                let namespace = Self::k8s_namespace_from_env();
                Self::Kubernetes { namespace }
            }
            _ => Self::Memory,
        }
    }

    fn redis_url_from_env() -> String {
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
///
/// For layered configurations, this creates an in-memory cache that
/// writes through to the persistent backend and hydrates on startup.
pub async fn create_backend(config: BackendConfig) -> MetadataResult<Arc<dyn MetadataBackend>> {
    match config {
        BackendConfig::Memory => {
            let backend = layered::LayeredBackend::memory_only();
            backend.connect().await?;
            Ok(Arc::new(backend))
        }
        BackendConfig::Redis { url } => {
            let backend = redis::RedisBackend::new(&url);
            backend.connect().await?;
            Ok(Arc::new(backend))
        }
        BackendConfig::Kubernetes { namespace } => {
            let backend = kubernetes::KubernetesBackend::new(&namespace).await?;
            backend.connect().await?;
            Ok(Arc::new(backend))
        }
        BackendConfig::LayeredRedis { url } => {
            let persistent = redis::RedisBackend::new(&url);
            persistent.connect().await?;
            let backend =
                layered::LayeredBackend::with_persistent(Arc::new(persistent));
            backend.connect().await?;
            Ok(Arc::new(backend))
        }
        BackendConfig::LayeredKubernetes { namespace } => {
            let persistent = kubernetes::KubernetesBackend::new(&namespace).await?;
            persistent.connect().await?;
            let backend =
                layered::LayeredBackend::with_persistent(Arc::new(persistent));
            backend.connect().await?;
            Ok(Arc::new(backend))
        }
    }
}
