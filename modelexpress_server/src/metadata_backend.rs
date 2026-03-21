// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metadata backend abstraction for P2P model metadata.
//!
//! Supports two persistent backends:
//! - **Redis**: Persistent storage via Redis keys + atomic Lua merge
//! - **Kubernetes**: CRDs and ConfigMaps for native K8s integration
//!
//! Select the backend via `MX_METADATA_BACKEND=redis` or `MX_METADATA_BACKEND=kubernetes`.
//!
//! NIXL agent blobs are NOT stored in the backend. Workers exchange them
//! peer-to-peer via NIXL's native listen thread. The backend only stores
//! lightweight directory entries: endpoints, tensor layouts, and status.

use async_trait::async_trait;
use modelexpress_common::grpc::p2p::{SourceIdentity, SourceStatus, WorkerMetadata};
use std::sync::Arc;

pub mod kubernetes;
pub mod redis;

/// Result type for metadata operations
pub type MetadataResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Convert an empty string to `None`, non-empty to `Some`.
/// Used by Redis and Kubernetes backends when serializing optional string fields.
pub fn none_if_empty(s: String) -> Option<String> {
    if s.is_empty() { None } else { Some(s) }
}

/// Model metadata record returned from backends
#[derive(Debug, Clone)]
pub struct ModelMetadataRecord {
    /// 16-char hex key derived from SourceIdentity hash
    pub source_id: String,
    /// Unique identifier for this running worker (UUID)
    pub worker_id: String,
    /// Human-readable model name from SourceIdentity
    pub model_name: String,
    pub workers: Vec<WorkerRecord>,
    pub published_at: i64,
}

/// Lightweight reference to a source worker (no tensor metadata).
/// Used by `list_workers` to support the `ListSources` RPC.
#[derive(Debug, Clone)]
pub struct SourceInstanceInfo {
    pub source_id: String,
    pub worker_id: String,
    pub model_name: String,
    /// Global rank of this worker.
    pub worker_rank: u32,
}

/// Backend-specific metadata for a worker
#[derive(Debug, Clone, PartialEq)]
pub enum BackendMetadataRecord {
    /// NIXL transfer backend. The NIXL agent blob is exchanged peer-to-peer
    /// via the worker's `metadata_endpoint`, not stored here.
    Nixl,
    /// Mooncake TransferEngine session ID ("ip:port")
    TransferEngine(String),
    /// No backend metadata provided
    None,
}

impl BackendMetadataRecord {
    /// Reconstruct from flat fields (used by Redis JSON and K8s CRD deserialization).
    ///
    /// When `backend_type` is provided, it is used as the authoritative discriminator.
    /// Falls back to field-inference for backwards compatibility with records written
    /// before `backend_type` was persisted. Note: NIXL can no longer be inferred from
    /// blob presence (blobs are exchanged peer-to-peer now), so the fallback path
    /// only detects TransferEngine. NIXL requires an explicit `backend_type` tag.
    pub fn from_flat(
        transfer_engine_session_id: Option<String>,
        backend_type: Option<&str>,
    ) -> Self {
        match backend_type {
            Some("transfer_engine") => {
                let sid = transfer_engine_session_id.unwrap_or_default();
                Self::TransferEngine(sid)
            }
            Some("nixl") => Self::Nixl,
            Some("none") => Self::None,
            // Unknown or missing backend_type: infer from fields (backwards compat)
            _ => {
                if let Some(sid) = transfer_engine_session_id
                    && !sid.is_empty()
                {
                    return Self::TransferEngine(sid);
                }
                Self::None
            }
        }
    }

    /// Returns the backend type string for persistence.
    pub fn backend_type_str(&self) -> &'static str {
        match self {
            Self::Nixl => "nixl",
            Self::TransferEngine(_) => "transfer_engine",
            Self::None => "none",
        }
    }
}

/// Worker metadata record
#[derive(Debug, Clone)]
pub struct WorkerRecord {
    pub worker_rank: u32,
    pub backend_metadata: BackendMetadataRecord,
    /// Endpoint (host:port) where this worker's NIXL listen thread serves metadata.
    pub metadata_endpoint: String,
    /// NIXL agent name for this worker (needed for check_remote_metadata).
    pub agent_name: String,
    /// Endpoint (host:port) for this worker's gRPC server (WorkerService).
    /// Targets connect here to fetch the tensor manifest directly.
    pub worker_grpc_endpoint: String,
    /// Worker lifecycle status (maps to `SourceStatus` proto enum)
    pub status: i32,
    /// Timestamp of last status update (unix millis)
    pub updated_at: i64,
}

// Conversions from gRPC types
impl From<WorkerMetadata> for WorkerRecord {
    fn from(meta: WorkerMetadata) -> Self {
        // TransferEngine takes priority when both fields are present because
        // it requires an explicit session ID, whereas NIXL is inferred from
        // the metadata_endpoint being set.
        let backend_metadata = if !meta.transfer_engine_session_id.is_empty() {
            BackendMetadataRecord::TransferEngine(meta.transfer_engine_session_id.clone())
        } else if !meta.metadata_endpoint.is_empty() {
            BackendMetadataRecord::Nixl
        } else {
            BackendMetadataRecord::None
        };
        Self {
            worker_rank: meta.worker_rank,
            backend_metadata,
            metadata_endpoint: meta.metadata_endpoint,
            agent_name: meta.agent_name,
            worker_grpc_endpoint: meta.worker_grpc_endpoint,
            status: meta.status,
            updated_at: meta.updated_at,
        }
    }
}

// Conversions back to gRPC types
impl From<WorkerRecord> for WorkerMetadata {
    fn from(record: WorkerRecord) -> Self {
        let transfer_engine_session_id = match &record.backend_metadata {
            BackendMetadataRecord::TransferEngine(sid) => sid.clone(),
            _ => String::new(),
        };
        Self {
            worker_rank: record.worker_rank,
            metadata_endpoint: record.metadata_endpoint,
            agent_name: record.agent_name,
            worker_grpc_endpoint: record.worker_grpc_endpoint,
            transfer_engine_session_id,
            status: record.status,
            updated_at: record.updated_at,
        }
    }
}

/// Trait for metadata backend implementations
#[cfg_attr(test, mockall::automock)]
#[async_trait]
pub trait MetadataBackend: Send + Sync {
    /// Connect to the backend (initialize connections, etc.)
    async fn connect(&self) -> MetadataResult<()>;

    /// Publish metadata for a source worker.
    /// `worker_id` uniquely identifies this running pod/process among all replicas
    /// with the same identity. The backend derives `mx_source_id` from `identity`.
    async fn publish_metadata(
        &self,
        identity: &SourceIdentity,
        worker_id: &str,
        workers: Vec<WorkerMetadata>,
    ) -> MetadataResult<()>;

    /// Get full tensor metadata for one specific worker.
    /// Returns `None` if the worker is not found.
    async fn get_metadata(
        &self,
        source_id: &str,
        worker_id: &str,
    ) -> MetadataResult<Option<ModelMetadataRecord>>;

    /// List available workers, optionally filtered by source_id and status.
    /// `source_id`: if `Some`, return only workers for that source; if `None`, all sources.
    /// `status_filter`: if `Some(s)`, return only workers where all workers have status `s`.
    async fn list_workers(
        &self,
        source_id: Option<String>,
        status_filter: Option<SourceStatus>,
    ) -> MetadataResult<Vec<SourceInstanceInfo>>;

    /// Remove all workers of a source by mx_source_id
    async fn remove_metadata(&self, source_id: &str) -> MetadataResult<()>;

    /// List all registered source IDs and their model names
    async fn list_sources(&self) -> MetadataResult<Vec<(String, String)>>;

    /// Patch the status of a worker for a specific worker.
    async fn update_status(
        &self,
        source_id: &str,
        worker_id: &str,
        worker_rank: u32,
        status: SourceStatus,
        updated_at: i64,
    ) -> MetadataResult<()>;
}

/// Configuration for metadata backends
#[derive(Debug, Clone)]
pub enum BackendConfig {
    /// Redis backend -- persistent, horizontally scalable
    Redis { url: String },
    /// Kubernetes CRD backend -- native K8s integration
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
///
/// # Errors
///
/// Returns an error if the backend cannot connect (e.g. Redis unreachable,
/// K8s API unavailable).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_flat_explicit_nixl() {
        let record = BackendMetadataRecord::from_flat(None, Some("nixl"));
        assert_eq!(record, BackendMetadataRecord::Nixl);
    }

    #[test]
    fn from_flat_explicit_transfer_engine() {
        let record =
            BackendMetadataRecord::from_flat(Some("1.2.3.4:9000".into()), Some("transfer_engine"));
        assert_eq!(
            record,
            BackendMetadataRecord::TransferEngine("1.2.3.4:9000".into())
        );
    }

    #[test]
    fn from_flat_explicit_none() {
        let record = BackendMetadataRecord::from_flat(None, Some("none"));
        assert_eq!(record, BackendMetadataRecord::None);
    }

    #[test]
    fn from_flat_no_discriminator_no_te_returns_none() {
        // With the P2P redesign, NIXL can no longer be inferred without an
        // explicit backend_type tag. When both discriminator and TE session
        // are absent, the result is None (not Nixl).
        let record = BackendMetadataRecord::from_flat(None, None);
        assert_eq!(record, BackendMetadataRecord::None);
    }

    #[test]
    fn from_flat_no_discriminator_with_te_infers_transfer_engine() {
        let record = BackendMetadataRecord::from_flat(Some("1.2.3.4:9000".into()), None);
        assert_eq!(
            record,
            BackendMetadataRecord::TransferEngine("1.2.3.4:9000".into())
        );
    }

    #[test]
    fn from_flat_no_discriminator_empty_te_returns_none() {
        let record = BackendMetadataRecord::from_flat(Some(String::new()), None);
        assert_eq!(record, BackendMetadataRecord::None);
    }

    #[test]
    fn none_if_empty_with_empty_string() {
        assert_eq!(none_if_empty(String::new()), None);
    }

    #[test]
    fn none_if_empty_with_non_empty_string() {
        assert_eq!(none_if_empty("hello".into()), Some("hello".into()));
    }
}
