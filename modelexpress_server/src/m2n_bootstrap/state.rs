// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validated state-manager facade for M2N bootstrap storage.

use std::sync::Arc;

use modelexpress_common::grpc::m2n_bootstrap::{
    AbortM2nBootstrapRequest, M2nBootstrapKey as ProtoBootstrapKey, M2nBootstrapRecord,
    M2nBootstrapState, PublishM2nBootstrapRequest,
};
use tokio::sync::RwLock;
use tracing::{info, warn};
use uuid::{Uuid, Version};

use super::backend::{
    BootstrapError, BootstrapKey, BootstrapRecord, BootstrapResult, M2nBootstrapBackend,
    PublishOutcome, create_backend, publication_fingerprint,
};
use crate::backend_config::BackendConfig;

const NCCL_UNIQUE_ID_BYTES: usize = 128;
const SHA256_BYTES: usize = 32;
const MAX_TTL_MS: u64 = 10 * 60 * 1000;
const MAX_ID_BYTES: usize = 256;
const MAX_REASON_BYTES: usize = 1024;

#[derive(Clone)]
pub struct M2nBootstrapStateManager {
    backend: Arc<RwLock<Option<Arc<dyn M2nBootstrapBackend>>>>,
    config: Option<BackendConfig>,
}

impl M2nBootstrapStateManager {
    #[must_use]
    pub fn with_config(config: BackendConfig) -> Self {
        Self {
            backend: Arc::new(RwLock::new(None)),
            config: Some(config),
        }
    }

    #[cfg(test)]
    #[must_use]
    pub fn with_backend(backend: Arc<dyn M2nBootstrapBackend>) -> Self {
        Self {
            backend: Arc::new(RwLock::new(Some(backend))),
            config: None,
        }
    }

    pub async fn connect(&self) -> BootstrapResult<String> {
        let config = self.config.clone().ok_or_else(|| {
            BootstrapError::Backend("bootstrap backend configuration is missing".to_string())
        })?;
        let backend_name = match &config {
            BackendConfig::Kubernetes { .. } => {
                warn!("M2N bootstrap is unavailable with the Kubernetes metadata backend");
                "unsupported(kubernetes)".to_string()
            }
            _ => config.to_string(),
        };
        let backend = create_backend(config).await?;
        *self.backend.write().await = Some(backend);
        info!("M2N bootstrap state connected (backend: {backend_name})");
        Ok(backend_name)
    }

    async fn backend(&self) -> BootstrapResult<Arc<dyn M2nBootstrapBackend>> {
        if let Some(backend) = self.backend.read().await.as_ref() {
            return Ok(backend.clone());
        }
        let config = self.config.clone().ok_or_else(|| {
            BootstrapError::Backend("bootstrap backend configuration is missing".to_string())
        })?;
        let backend = create_backend(config).await?;
        *self.backend.write().await = Some(backend.clone());
        Ok(backend)
    }

    pub async fn publish(
        &self,
        request: PublishM2nBootstrapRequest,
        now_ms: i64,
    ) -> BootstrapResult<PublishOutcome> {
        let key = validate_key(request.key)?;
        validate_nonempty(
            "publisher_participant_id",
            &request.publisher_participant_id,
        )?;
        if request.nccl_unique_id.len() != NCCL_UNIQUE_ID_BYTES {
            return Err(invalid(format!(
                "nccl_unique_id must be exactly {NCCL_UNIQUE_ID_BYTES} bytes"
            )));
        }
        if request.roster_digest.len() != SHA256_BYTES {
            return Err(invalid("roster_digest must be a 32-byte SHA-256 digest"));
        }
        if request.config_digest.len() != SHA256_BYTES {
            return Err(invalid("config_digest must be a 32-byte SHA-256 digest"));
        }
        if request.source_world_size == 0 || request.destination_world_size == 0 {
            return Err(invalid(
                "source_world_size and destination_world_size must be non-zero",
            ));
        }
        let expected_world = request
            .source_world_size
            .checked_add(request.destination_world_size)
            .ok_or_else(|| invalid("source and destination world sizes overflow"))?;
        if request.world_size != expected_world {
            return Err(invalid(
                "world_size must equal source_world_size + destination_world_size",
            ));
        }
        if request.ttl_ms == 0 || request.ttl_ms > MAX_TTL_MS {
            return Err(invalid(format!(
                "ttl_ms must be in the range 1..={MAX_TTL_MS}"
            )));
        }
        let ttl = i64::try_from(request.ttl_ms)
            .map_err(|_| invalid("ttl_ms cannot be represented by the server clock"))?;
        let expires_at_ms = now_ms
            .checked_add(ttl)
            .ok_or_else(|| invalid("bootstrap expiration overflows the server clock"))?;

        let mut record = BootstrapRecord {
            key,
            nccl_unique_id: request.nccl_unique_id,
            source_world_size: request.source_world_size,
            destination_world_size: request.destination_world_size,
            world_size: request.world_size,
            roster_digest: request.roster_digest,
            config_digest: request.config_digest,
            publisher_participant_id: request.publisher_participant_id,
            state: M2nBootstrapState::Published,
            expires_at_ms,
            reason: String::new(),
            revision: 1,
            publication_fingerprint: String::new(),
        };
        record.publication_fingerprint = publication_fingerprint(&record);
        self.backend().await?.publish(record, now_ms).await
    }

    pub async fn get(
        &self,
        key: Option<ProtoBootstrapKey>,
        now_ms: i64,
    ) -> BootstrapResult<Option<M2nBootstrapRecord>> {
        let key = validate_key(key)?;
        Ok(self
            .backend()
            .await?
            .get(&key, now_ms)
            .await?
            .map(|record| record.to_proto()))
    }

    pub async fn abort(
        &self,
        request: AbortM2nBootstrapRequest,
        now_ms: i64,
    ) -> BootstrapResult<M2nBootstrapRecord> {
        let key = validate_key(request.key)?;
        validate_nonempty("requested_by", &request.requested_by)?;
        if request.reason.is_empty() {
            return Err(invalid("reason is required"));
        }
        if request.reason.len() > MAX_REASON_BYTES {
            return Err(invalid(format!(
                "reason must be at most {MAX_REASON_BYTES} bytes"
            )));
        }
        Ok(self
            .backend()
            .await?
            .abort(&key, &request.requested_by, &request.reason, now_ms)
            .await?
            .to_proto())
    }
}

fn invalid(message: impl Into<String>) -> BootstrapError {
    BootstrapError::InvalidArgument(message.into())
}

fn validate_nonempty(name: &str, value: &str) -> BootstrapResult<()> {
    if value.is_empty() {
        return Err(invalid(format!("{name} is required")));
    }
    if value.len() > MAX_ID_BYTES {
        return Err(invalid(format!(
            "{name} must be at most {MAX_ID_BYTES} bytes"
        )));
    }
    Ok(())
}

fn validate_key(key: Option<ProtoBootstrapKey>) -> BootstrapResult<BootstrapKey> {
    let key = key.ok_or_else(|| invalid("key is required"))?;
    validate_nonempty("job_id", &key.job_id)?;
    validate_nonempty("attempt_id", &key.attempt_id)?;
    validate_nonempty("cohort_id", &key.cohort_id)?;
    let attempt_id = Uuid::parse_str(&key.attempt_id).map_err(|_| {
        invalid("attempt_id must be a canonical random UUIDv4 and must never be reused")
    })?;
    if attempt_id.get_version() != Some(Version::Random)
        || attempt_id.hyphenated().to_string() != key.attempt_id
    {
        return Err(invalid(
            "attempt_id must be a canonical random UUIDv4 and must never be reused",
        ));
    }
    Ok(BootstrapKey {
        job_id: key.job_id,
        attempt_id: key.attempt_id,
        cohort_id: key.cohort_id,
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::m2n_bootstrap::backend::memory::InMemoryM2nBootstrapBackend;

    fn key() -> ProtoBootstrapKey {
        ProtoBootstrapKey {
            job_id: "job".to_string(),
            attempt_id: "123e4567-e89b-42d3-a456-426614174000".to_string(),
            cohort_id: "cohort".to_string(),
        }
    }

    fn publish_request() -> PublishM2nBootstrapRequest {
        PublishM2nBootstrapRequest {
            key: Some(key()),
            nccl_unique_id: vec![7; NCCL_UNIQUE_ID_BYTES],
            source_world_size: 2,
            destination_world_size: 4,
            world_size: 6,
            roster_digest: vec![1; SHA256_BYTES],
            config_digest: vec![2; SHA256_BYTES],
            publisher_participant_id: "source-0".to_string(),
            ttl_ms: 1_000,
        }
    }

    fn manager() -> M2nBootstrapStateManager {
        M2nBootstrapStateManager::with_backend(Arc::new(InMemoryM2nBootstrapBackend::new()))
    }

    #[tokio::test]
    async fn validates_and_publishes_a_record() {
        let outcome = manager()
            .publish(publish_request(), 100)
            .await
            .expect("publish");
        assert!(outcome.created);
        assert_eq!(outcome.record.expires_at_ms, 1_100);
    }

    #[tokio::test]
    async fn rejects_world_size_mismatch() {
        let mut request = publish_request();
        request.world_size = 7;
        let error = manager().publish(request, 100).await.expect_err("invalid");
        assert!(error.to_string().contains("world_size"));
    }

    #[tokio::test]
    async fn rejects_malformed_attempt_id() {
        let mut request = publish_request();
        request.key.as_mut().expect("key").attempt_id = "reused-name".to_string();
        let error = manager().publish(request, 100).await.expect_err("invalid");
        assert!(error.to_string().contains("UUID"));
    }

    #[tokio::test]
    async fn rejects_noncanonical_or_non_v4_attempt_id() {
        for attempt_id in [
            "123E4567-E89B-42D3-A456-426614174000",
            "123e4567-e89b-12d3-a456-426614174000",
        ] {
            let mut request = publish_request();
            request.key.as_mut().expect("key").attempt_id = attempt_id.to_string();
            let error = manager().publish(request, 100).await.expect_err("invalid");
            assert!(error.to_string().contains("UUIDv4"));
        }
    }
}
