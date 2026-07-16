// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic storage contract for NCCL M2N bootstrap records.

use async_trait::async_trait;
use modelexpress_common::grpc::m2n_bootstrap::{
    M2nBootstrapKey as ProtoBootstrapKey, M2nBootstrapRecord as ProtoBootstrapRecord,
    M2nBootstrapState,
};
use prost::Message;
use sha2::{Digest, Sha256};

use crate::backend_config::BackendConfig;

#[cfg(any(test, feature = "memory-backend"))]
pub mod memory;
pub mod redis;

pub const TOMBSTONE_RETENTION_MS: u64 = 86_400_000;

#[derive(Debug, thiserror::Error)]
pub enum BootstrapError {
    #[error("invalid bootstrap request: {0}")]
    InvalidArgument(String),
    #[error("bootstrap record conflict: {0}")]
    Conflict(String),
    #[error("bootstrap backend is unsupported: {0}")]
    Unsupported(String),
    #[error("invalid stored bootstrap record: {0}")]
    InvalidStoredRecord(String),
    #[error("bootstrap backend error: {0}")]
    Backend(String),
}

pub type BootstrapResult<T> = Result<T, BootstrapError>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BootstrapKey {
    pub job_id: String,
    pub attempt_id: String,
    pub cohort_id: String,
}

impl From<&BootstrapKey> for ProtoBootstrapKey {
    fn from(key: &BootstrapKey) -> Self {
        Self {
            job_id: key.job_id.clone(),
            attempt_id: key.attempt_id.clone(),
            cohort_id: key.cohort_id.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BootstrapRecord {
    pub key: BootstrapKey,
    pub nccl_unique_id: Vec<u8>,
    pub source_world_size: u32,
    pub destination_world_size: u32,
    pub world_size: u32,
    pub roster_digest: Vec<u8>,
    pub config_digest: Vec<u8>,
    pub publisher_participant_id: String,
    pub state: M2nBootstrapState,
    pub expires_at_ms: i64,
    pub reason: String,
    pub revision: u64,
    pub publication_fingerprint: String,
}

impl BootstrapRecord {
    #[must_use]
    pub fn to_proto(&self) -> ProtoBootstrapRecord {
        ProtoBootstrapRecord {
            key: Some((&self.key).into()),
            nccl_unique_id: self.nccl_unique_id.clone(),
            source_world_size: self.source_world_size,
            destination_world_size: self.destination_world_size,
            world_size: self.world_size,
            roster_digest: self.roster_digest.clone(),
            config_digest: self.config_digest.clone(),
            publisher_participant_id: self.publisher_participant_id.clone(),
            state: self.state as i32,
            expires_at_ms: self.expires_at_ms,
            reason: self.reason.clone(),
            revision: self.revision,
        }
    }

    #[must_use]
    pub fn aborted_tombstone(
        key: BootstrapKey,
        requested_by: String,
        reason: String,
        now_ms: i64,
    ) -> Self {
        Self {
            key,
            nccl_unique_id: Vec::new(),
            source_world_size: 0,
            destination_world_size: 0,
            world_size: 0,
            roster_digest: Vec::new(),
            config_digest: Vec::new(),
            publisher_participant_id: requested_by,
            state: M2nBootstrapState::Aborted,
            expires_at_ms: now_ms,
            reason,
            revision: 1,
            publication_fingerprint: String::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublishOutcome {
    pub record: BootstrapRecord,
    pub created: bool,
}

#[must_use]
pub fn publication_fingerprint(record: &BootstrapRecord) -> String {
    let mut proto = record.to_proto();
    proto.state = M2nBootstrapState::Published as i32;
    proto.expires_at_ms = 0;
    proto.reason.clear();
    proto.revision = 0;
    let digest = Sha256::digest(proto.encode_to_vec());
    format!("{digest:x}")
}

#[cfg_attr(test, mockall::automock)]
#[async_trait]
pub trait M2nBootstrapBackend: Send + Sync {
    async fn connect(&self) -> BootstrapResult<()>;

    async fn publish(
        &self,
        record: BootstrapRecord,
        now_ms: i64,
    ) -> BootstrapResult<PublishOutcome>;

    async fn get(
        &self,
        key: &BootstrapKey,
        now_ms: i64,
    ) -> BootstrapResult<Option<BootstrapRecord>>;

    async fn abort(
        &self,
        key: &BootstrapKey,
        requested_by: &str,
        reason: &str,
        now_ms: i64,
    ) -> BootstrapResult<BootstrapRecord>;
}

pub async fn create_backend(
    config: BackendConfig,
) -> BootstrapResult<std::sync::Arc<dyn M2nBootstrapBackend>> {
    match config {
        BackendConfig::Redis { url } => {
            let backend = std::sync::Arc::new(redis::RedisM2nBootstrapBackend::new(&url));
            backend.connect().await?;
            Ok(backend)
        }
        BackendConfig::Kubernetes { .. } => Ok(std::sync::Arc::new(UnsupportedM2nBootstrapBackend)),
        #[cfg(feature = "memory-backend")]
        BackendConfig::Memory => {
            let backend = std::sync::Arc::new(memory::InMemoryM2nBootstrapBackend::new());
            backend.connect().await?;
            Ok(backend)
        }
    }
}

struct UnsupportedM2nBootstrapBackend;

#[async_trait]
impl M2nBootstrapBackend for UnsupportedM2nBootstrapBackend {
    async fn connect(&self) -> BootstrapResult<()> {
        Ok(())
    }

    async fn publish(
        &self,
        _record: BootstrapRecord,
        _now_ms: i64,
    ) -> BootstrapResult<PublishOutcome> {
        Err(BootstrapError::Unsupported(
            "Kubernetes bootstrap storage requires resourceVersion CAS support".to_string(),
        ))
    }

    async fn get(
        &self,
        _key: &BootstrapKey,
        _now_ms: i64,
    ) -> BootstrapResult<Option<BootstrapRecord>> {
        Err(BootstrapError::Unsupported(
            "Kubernetes bootstrap storage requires resourceVersion CAS support".to_string(),
        ))
    }

    async fn abort(
        &self,
        _key: &BootstrapKey,
        _requested_by: &str,
        _reason: &str,
        _now_ms: i64,
    ) -> BootstrapResult<BootstrapRecord> {
        Err(BootstrapError::Unsupported(
            "Kubernetes bootstrap storage requires resourceVersion CAS support".to_string(),
        ))
    }
}
