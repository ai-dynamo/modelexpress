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
use modelexpress_common::grpc::p2p::{
    SliceOwnershipDescriptor, SourceIdentity, SourceStatus, TensorAxisRange, WorkerMetadata,
};
use std::collections::BTreeMap;
use std::sync::Arc;

pub mod kubernetes;
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
    pub slice_ownerships: Vec<SliceOwnershipRecord>,
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

/// Half-open logical tensor range for one axis: [start, end).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorAxisRangeRecord {
    pub start: u64,
    pub end: u64,
}

/// Source-published slice ownership for cross-parallelism refit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceOwnershipRecord {
    pub model_name: String,
    pub model_version: String,
    pub tensor_name: String,
    pub global_shape: Vec<u64>,
    pub dtype: String,
    pub source_range: Vec<TensorAxisRangeRecord>,
    pub storage_offset_bytes: u64,
    pub strides: Vec<i64>,
    pub contiguous: bool,
    pub worker_id: String,
    pub worker_rank: u32,
    pub source_id: String,
    pub source_lease: String,
    pub nixl_descriptor_id: String,
    pub layout_tags: BTreeMap<String, String>,
    pub quantization_scope: String,
    pub element_size_bytes: Option<u32>,
    pub tensor_family: String,
}

// Conversions from gRPC types
impl From<WorkerMetadata> for WorkerRecord {
    fn from(meta: WorkerMetadata) -> Self {
        use modelexpress_common::grpc::p2p::worker_metadata::BackendMetadata;
        let backend_metadata = match meta.backend_metadata {
            Some(BackendMetadata::NixlMetadata(data)) => BackendMetadataRecord::Nixl(data),
            Some(BackendMetadata::TransferEngineSessionId(sid)) => {
                BackendMetadataRecord::TransferEngine(sid)
            }
            None => BackendMetadataRecord::None,
        };
        Self {
            worker_rank: meta.worker_rank,
            backend_metadata,
            tensors: meta.tensors.into_iter().map(TensorRecord::from).collect(),
            slice_ownerships: meta
                .slice_ownerships
                .into_iter()
                .map(SliceOwnershipRecord::from)
                .collect(),
            status: meta.status,
            updated_at: meta.updated_at,
            metadata_endpoint: meta.metadata_endpoint,
            agent_name: meta.agent_name,
            worker_grpc_endpoint: meta.worker_grpc_endpoint,
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

impl From<TensorAxisRange> for TensorAxisRangeRecord {
    fn from(range: TensorAxisRange) -> Self {
        Self {
            start: range.start,
            end: range.end,
        }
    }
}

impl From<SliceOwnershipDescriptor> for SliceOwnershipRecord {
    fn from(desc: SliceOwnershipDescriptor) -> Self {
        Self {
            model_name: desc.model_name,
            model_version: desc.model_version,
            tensor_name: desc.tensor_name,
            global_shape: desc.global_shape,
            dtype: desc.dtype,
            source_range: desc
                .source_range
                .into_iter()
                .map(TensorAxisRangeRecord::from)
                .collect(),
            storage_offset_bytes: desc.storage_offset_bytes,
            strides: desc.strides,
            contiguous: desc.contiguous,
            worker_id: desc.worker_id,
            worker_rank: desc.worker_rank,
            source_id: desc.source_id,
            source_lease: desc.source_lease,
            nixl_descriptor_id: desc.nixl_descriptor_id,
            layout_tags: desc.layout_tags.into_iter().collect(),
            quantization_scope: desc.quantization_scope,
            element_size_bytes: desc.element_size_bytes,
            tensor_family: desc.tensor_family,
        }
    }
}

// Conversions back to gRPC types
impl From<WorkerRecord> for WorkerMetadata {
    fn from(record: WorkerRecord) -> Self {
        use modelexpress_common::grpc::p2p::worker_metadata::BackendMetadata;
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
            tensors: record
                .tensors
                .into_iter()
                .map(modelexpress_common::grpc::p2p::TensorDescriptor::from)
                .collect(),
            slice_ownerships: record
                .slice_ownerships
                .into_iter()
                .map(SliceOwnershipDescriptor::from)
                .collect(),
            status: record.status,
            updated_at: record.updated_at,
            metadata_endpoint: record.metadata_endpoint,
            agent_name: record.agent_name,
            worker_grpc_endpoint: record.worker_grpc_endpoint,
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

impl From<TensorAxisRangeRecord> for TensorAxisRange {
    fn from(record: TensorAxisRangeRecord) -> Self {
        Self {
            start: record.start,
            end: record.end,
        }
    }
}

impl From<SliceOwnershipRecord> for SliceOwnershipDescriptor {
    fn from(record: SliceOwnershipRecord) -> Self {
        Self {
            model_name: record.model_name,
            model_version: record.model_version,
            tensor_name: record.tensor_name,
            global_shape: record.global_shape,
            dtype: record.dtype,
            source_range: record
                .source_range
                .into_iter()
                .map(TensorAxisRange::from)
                .collect(),
            storage_offset_bytes: record.storage_offset_bytes,
            strides: record.strides,
            contiguous: record.contiguous,
            worker_id: record.worker_id,
            worker_rank: record.worker_rank,
            source_id: record.source_id,
            source_lease: record.source_lease,
            nixl_descriptor_id: record.nixl_descriptor_id,
            layout_tags: record.layout_tags.into_iter().collect(),
            quantization_scope: record.quantization_scope,
            element_size_bytes: record.element_size_bytes,
            tensor_family: record.tensor_family,
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
        worker: WorkerMetadata,
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
    }
}
