// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metadata backend abstraction for P2P model metadata.
//!
//! Backends:
//! - **Redis**: Persistent storage via Redis keys + atomic Lua merge
//! - **Kubernetes**: CRDs and ConfigMaps for native K8s integration
//! - **Memory** (`memory-backend` feature): non-persistent, single-process, for tests
//!   and local dev
//!
//! Select the backend via `MX_METADATA_BACKEND=redis`, `=kubernetes`, or `=memory`.

use async_trait::async_trait;
use modelexpress_common::grpc::p2p::{
    ArtifactSourceMetadata, SourceIdentity, SourceStatus, TensorSourceMetadata, WorkerMetadata,
};
use std::collections::HashMap;
use std::sync::Arc;

pub mod kubernetes;
#[cfg(feature = "memory-backend")]
pub mod memory;
pub mod redis;

/// Result type for metadata operations
pub type MetadataResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

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
    /// Full SourceIdentity that produced ``source_id``.
    ///
    /// Older records (written before this field was added) leave it ``None``.
    /// New v2 RL clients (NemoRL `update_weights_via_mx`) read framework-level
    /// state from `identity.extra_parameters` (training_step, role, shape
    /// registry, dirty experts). Backends are responsible for round-tripping
    /// the entire SourceIdentity message; pre-existing records continue to
    /// work because the field is optional.
    pub identity: Option<SourceIdentity>,
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
    /// Worker lifecycle status (maps to `SourceStatus` proto enum).
    pub status: i32,
    /// Timestamp of last status update (unix millis).
    pub updated_at: i64,
    /// Runtime accelerator family for compatibility filtering (e.g. "cuda").
    /// Empty when unknown (records that predate the field).
    pub accelerator: String,
    /// Training step/version from SourceIdentity.extra_parameters, when present.
    pub training_step: Option<u64>,
    /// Stable topology/registry digest, when published by the source.
    pub layout_signature: Option<String>,
}

pub(crate) fn parse_training_step(extra_parameters: &HashMap<String, String>) -> Option<u64> {
    extra_parameters
        .get("training_step")
        .or_else(|| extra_parameters.get("version"))
        .and_then(|value| value.parse::<u64>().ok())
}

pub(crate) fn parse_layout_signature(extra_parameters: &HashMap<String, String>) -> Option<String> {
    extra_parameters
        .get("layout_signature")
        .filter(|value| !value.is_empty())
        .cloned()
}

/// Backend-specific metadata for a worker
#[derive(Debug, Clone, PartialEq)]
pub enum BackendMetadataRecord {
    /// Serialized NIXL agent metadata for RDMA connections
    Nixl(Vec<u8>),
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
    /// before `backend_type` was persisted.
    pub fn from_flat(
        nixl_metadata: Vec<u8>,
        transfer_engine_session_id: Option<String>,
        backend_type: Option<&str>,
    ) -> Self {
        match backend_type {
            Some("transfer_engine") => {
                let sid = transfer_engine_session_id.unwrap_or_default();
                Self::TransferEngine(sid)
            }
            Some("nixl") => Self::Nixl(nixl_metadata),
            Some("none") => Self::None,
            // Unknown or missing backend_type: infer from fields (backwards compat)
            _ => {
                if let Some(sid) = transfer_engine_session_id
                    && !sid.is_empty()
                {
                    return Self::TransferEngine(sid);
                }
                if !nixl_metadata.is_empty() {
                    return Self::Nixl(nixl_metadata);
                }
                Self::None
            }
        }
    }

    /// Returns the backend type string for persistence.
    pub fn backend_type_str(&self) -> &'static str {
        match self {
            Self::Nixl(_) => "nixl",
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
    pub tensors: Vec<TensorRecord>,
    /// Worker lifecycle status (maps to `SourceStatus` proto enum)
    pub status: i32,
    /// Timestamp of last status update (unix millis)
    pub updated_at: i64,
    /// P2P: NIXL listen thread endpoint (host:port)
    pub metadata_endpoint: String,
    /// P2P: NIXL agent name for remote identification
    pub agent_name: String,
    /// P2P: Worker gRPC endpoint for tensor manifest (host:port)
    pub worker_grpc_endpoint: String,
    /// Runtime accelerator family for compatibility filtering.
    pub accelerator: String,
    /// Small discovery summary for file-backed artifact sources.
    pub artifact_source: Option<ArtifactSourceMetadataRecord>,
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

/// Bounded artifact discovery summary stored with worker metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ArtifactSourceMetadataRecord {
    pub artifact_id: String,
    pub total_size: u64,
    pub file_count: u32,
    pub chunk_count: u32,
    pub node_rank: u32,
}

// Conversions from gRPC types
impl From<WorkerMetadata> for WorkerRecord {
    #[allow(deprecated)]
    fn from(meta: WorkerMetadata) -> Self {
        use modelexpress_common::grpc::p2p::worker_metadata::BackendMetadata;
        use modelexpress_common::grpc::p2p::worker_metadata::SourcePayload;
        let backend_metadata = match meta.backend_metadata {
            Some(BackendMetadata::NixlMetadata(data)) => BackendMetadataRecord::Nixl(data),
            Some(BackendMetadata::TransferEngineSessionId(sid)) => {
                BackendMetadataRecord::TransferEngine(sid)
            }
            None => BackendMetadataRecord::None,
        };
        let (tensors, artifact_source) = match meta.source_payload {
            Some(SourcePayload::TensorSource(tensor_source)) => (tensor_source.tensors, None),
            Some(SourcePayload::ArtifactSource(artifact)) => (
                Vec::new(),
                Some(ArtifactSourceMetadataRecord::from(artifact)),
            ),
            None => (meta.tensors, None),
        };
        Self {
            worker_rank: meta.worker_rank,
            backend_metadata,
            tensors: tensors.into_iter().map(TensorRecord::from).collect(),
            status: meta.status,
            updated_at: meta.updated_at,
            metadata_endpoint: meta.metadata_endpoint,
            agent_name: meta.agent_name,
            worker_grpc_endpoint: meta.worker_grpc_endpoint,
            accelerator: meta.accelerator,
            artifact_source,
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
    #[allow(deprecated)]
    fn from(record: WorkerRecord) -> Self {
        use modelexpress_common::grpc::p2p::worker_metadata::BackendMetadata;
        use modelexpress_common::grpc::p2p::worker_metadata::SourcePayload;
        let tensors: Vec<modelexpress_common::grpc::p2p::TensorDescriptor> = record
            .tensors
            .into_iter()
            .map(modelexpress_common::grpc::p2p::TensorDescriptor::from)
            .collect();
        let (legacy_tensors, source_payload) = match record.artifact_source {
            Some(artifact) => (
                Vec::new(),
                SourcePayload::ArtifactSource(ArtifactSourceMetadata::from(artifact)),
            ),
            None => (
                tensors.clone(),
                SourcePayload::TensorSource(TensorSourceMetadata {
                    tensors: tensors.clone(),
                }),
            ),
        };
        let backend_metadata = match record.backend_metadata {
            BackendMetadataRecord::Nixl(data) => Some(BackendMetadata::NixlMetadata(data)),
            BackendMetadataRecord::TransferEngine(sid) => {
                Some(BackendMetadata::TransferEngineSessionId(sid))
            }
            BackendMetadataRecord::None => None,
        };
        Self {
            worker_rank: record.worker_rank,
            backend_metadata,
            status: record.status,
            updated_at: record.updated_at,
            metadata_endpoint: record.metadata_endpoint,
            agent_name: record.agent_name,
            worker_grpc_endpoint: record.worker_grpc_endpoint,
            accelerator: record.accelerator,
            tensors: legacy_tensors,
            source_payload: Some(source_payload),
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

impl From<ArtifactSourceMetadata> for ArtifactSourceMetadataRecord {
    fn from(meta: ArtifactSourceMetadata) -> Self {
        Self {
            artifact_id: meta.artifact_id,
            total_size: meta.total_size,
            file_count: meta.file_count,
            chunk_count: meta.chunk_count,
            node_rank: meta.node_rank,
        }
    }
}

impl From<ArtifactSourceMetadataRecord> for ArtifactSourceMetadata {
    fn from(record: ArtifactSourceMetadataRecord) -> Self {
        Self {
            artifact_id: record.artifact_id,
            total_size: record.total_size,
            file_count: record.file_count,
            chunk_count: record.chunk_count,
            node_rank: record.node_rank,
        }
    }
}

/// Trait for metadata backend implementations
#[cfg_attr(test, mockall::automock)]
#[allow(clippy::too_many_arguments)]
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
        worker: WorkerMetadata,
        pod_name: &str,
        pod_uid: &str,
        pod_namespace: &str,
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
    /// `status_filter`: if `Some(s)`, return only workers where at least one rank has
    /// status `s`.
    async fn list_workers(
        &self,
        source_id: Option<String>,
        status_filter: Option<SourceStatus>,
    ) -> MetadataResult<Vec<SourceInstanceInfo>>;

    /// List workers with lightweight discovery predicates applied before the
    /// caller fetches full tensor metadata. Backends may override this to push
    /// filtering and batching into their storage engine.
    async fn list_workers_filtered(
        &self,
        source_id: Option<String>,
        status_filter: Option<SourceStatus>,
        model_name_filter: Option<String>,
        worker_rank_filter: Option<u32>,
        min_training_step: Option<u64>,
        min_updated_at: Option<i64>,
        limit: Option<usize>,
    ) -> MetadataResult<Vec<SourceInstanceInfo>> {
        let mut workers = self.list_workers(source_id, status_filter).await?;
        workers.retain(|worker| {
            model_name_filter
                .as_ref()
                .is_none_or(|model| worker.model_name == *model)
                && worker_rank_filter.is_none_or(|rank| worker.worker_rank == rank)
                && min_training_step
                    .is_none_or(|step| worker.training_step.is_some_and(|v| v >= step))
                && min_updated_at.is_none_or(|timestamp| worker.updated_at >= timestamp)
        });
        workers.sort_by_key(|worker| std::cmp::Reverse(worker.updated_at));
        if let Some(limit) = limit.filter(|value| *value > 0) {
            workers.truncate(limit);
        }
        Ok(workers)
    }

    /// Remove all workers of a source by mx_source_id
    async fn remove_metadata(&self, source_id: &str) -> MetadataResult<()>;

    /// Remove a single worker by source_id and worker_id.
    /// Used by the reaper to garbage-collect individual stale entries.
    async fn remove_worker(&self, source_id: &str, worker_id: &str) -> MetadataResult<()>;

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

pub use crate::backend_config::BackendConfig;

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
        #[cfg(feature = "memory-backend")]
        BackendConfig::Memory => {
            let backend = memory::InMemoryMetadataBackend::new();
            backend.connect().await?;
            Ok(Arc::new(backend) as Arc<dyn MetadataBackend>)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_layout_signature, parse_training_step};
    use std::collections::HashMap;

    #[test]
    fn parses_training_step_with_version_fallback() {
        let mut values = HashMap::new();
        values.insert("version".to_string(), "41".to_string());
        assert_eq!(parse_training_step(&values), Some(41));

        values.insert("training_step".to_string(), "42".to_string());
        assert_eq!(parse_training_step(&values), Some(42));

        values.insert("training_step".to_string(), "not-a-number".to_string());
        assert_eq!(parse_training_step(&values), None);
    }

    #[test]
    fn parses_non_empty_layout_signature() {
        let mut values = HashMap::new();
        assert_eq!(parse_layout_signature(&values), None);
        values.insert("layout_signature".to_string(), String::new());
        assert_eq!(parse_layout_signature(&values), None);
        values.insert("layout_signature".to_string(), "abc123".to_string());
        assert_eq!(parse_layout_signature(&values).as_deref(), Some("abc123"));
    }
}
