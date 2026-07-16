// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-process bootstrap backend for tests and local development.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use async_trait::async_trait;
use modelexpress_common::grpc::m2n_bootstrap::M2nBootstrapState;

use super::{
    BootstrapError, BootstrapKey, BootstrapRecord, BootstrapResult, M2nBootstrapBackend,
    PublishOutcome,
};

#[derive(Default)]
pub struct InMemoryM2nBootstrapBackend {
    records: Mutex<HashMap<String, BootstrapRecord>>,
}

impl InMemoryM2nBootstrapBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, BootstrapRecord>> {
        self.records.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn expire_if_needed(record: &mut BootstrapRecord, now_ms: i64) {
        if record.state == M2nBootstrapState::Published && now_ms >= record.expires_at_ms {
            record.state = M2nBootstrapState::Expired;
            record.nccl_unique_id.clear();
            record.reason = "bootstrap attempt expired".to_string();
            record.revision = record.revision.saturating_add(1);
        }
    }

    fn ensure_key(expected: &BootstrapKey, actual: &BootstrapKey) -> BootstrapResult<()> {
        if expected == actual {
            Ok(())
        } else {
            Err(BootstrapError::Conflict(format!(
                "attempt_id '{}' is already bound to another job or cohort",
                expected.attempt_id
            )))
        }
    }
}

#[async_trait]
impl M2nBootstrapBackend for InMemoryM2nBootstrapBackend {
    async fn connect(&self) -> BootstrapResult<()> {
        Ok(())
    }

    async fn publish(
        &self,
        record: BootstrapRecord,
        now_ms: i64,
    ) -> BootstrapResult<PublishOutcome> {
        let mut records = self.lock();
        if let Some(existing) = records.get_mut(&record.key.attempt_id) {
            Self::expire_if_needed(existing, now_ms);
            Self::ensure_key(&record.key, &existing.key)?;
            if existing.state == M2nBootstrapState::Published
                && existing.publication_fingerprint == record.publication_fingerprint
            {
                return Ok(PublishOutcome {
                    record: existing.clone(),
                    created: false,
                });
            }
            return Err(BootstrapError::Conflict(format!(
                "attempt '{}' is already {}",
                record.key.attempt_id,
                existing.state.as_str_name()
            )));
        }
        records.insert(record.key.attempt_id.clone(), record.clone());
        Ok(PublishOutcome {
            record,
            created: true,
        })
    }

    async fn get(
        &self,
        key: &BootstrapKey,
        now_ms: i64,
    ) -> BootstrapResult<Option<BootstrapRecord>> {
        let mut records = self.lock();
        let Some(record) = records.get_mut(&key.attempt_id) else {
            return Ok(None);
        };
        Self::ensure_key(key, &record.key)?;
        Self::expire_if_needed(record, now_ms);
        Ok(Some(record.clone()))
    }

    async fn abort(
        &self,
        key: &BootstrapKey,
        requested_by: &str,
        reason: &str,
        now_ms: i64,
    ) -> BootstrapResult<BootstrapRecord> {
        let mut records = self.lock();
        if let Some(record) = records.get_mut(&key.attempt_id) {
            Self::ensure_key(key, &record.key)?;
            Self::expire_if_needed(record, now_ms);
            if record.state == M2nBootstrapState::Published {
                record.state = M2nBootstrapState::Aborted;
                record.nccl_unique_id.clear();
                record.reason = reason.to_string();
                record.revision = record.revision.saturating_add(1);
            }
            return Ok(record.clone());
        }

        let record = BootstrapRecord::aborted_tombstone(
            key.clone(),
            requested_by.to_string(),
            reason.to_string(),
            now_ms,
        );
        records.insert(key.attempt_id.clone(), record.clone());
        Ok(record)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::m2n_bootstrap::backend::publication_fingerprint;

    fn key() -> BootstrapKey {
        BootstrapKey {
            job_id: "job".to_string(),
            attempt_id: "123e4567-e89b-42d3-a456-426614174000".to_string(),
            cohort_id: "cohort".to_string(),
        }
    }

    fn record(uid_byte: u8) -> BootstrapRecord {
        let mut record = BootstrapRecord {
            key: key(),
            nccl_unique_id: vec![uid_byte; 128],
            source_world_size: 2,
            destination_world_size: 2,
            world_size: 4,
            roster_digest: vec![1; 32],
            config_digest: vec![2; 32],
            publisher_participant_id: "source-0".to_string(),
            state: M2nBootstrapState::Published,
            expires_at_ms: 1_000,
            reason: String::new(),
            revision: 1,
            publication_fingerprint: String::new(),
        };
        record.publication_fingerprint = publication_fingerprint(&record);
        record
    }

    #[tokio::test]
    async fn identical_publish_is_idempotent_and_conflict_is_rejected() {
        let backend = InMemoryM2nBootstrapBackend::new();
        let first = backend.publish(record(1), 10).await.expect("first");
        assert!(first.created);

        let retry = backend.publish(record(1), 20).await.expect("retry");
        assert!(!retry.created);

        let err = backend.publish(record(2), 30).await.expect_err("conflict");
        assert!(matches!(err, BootstrapError::Conflict(_)));
    }

    #[tokio::test]
    async fn abort_before_publish_creates_a_fencing_tombstone() {
        let backend = InMemoryM2nBootstrapBackend::new();
        let aborted = backend
            .abort(&key(), "coordinator", "cancelled", 10)
            .await
            .expect("abort");
        assert_eq!(aborted.state, M2nBootstrapState::Aborted);
        assert!(
            backend
                .publish(record(1), 20)
                .await
                .expect_err("publish after abort")
                .to_string()
                .contains("already")
        );
    }

    #[tokio::test]
    async fn get_expires_and_redacts_uid() {
        let backend = InMemoryM2nBootstrapBackend::new();
        backend.publish(record(1), 10).await.expect("publish");
        let expired = backend
            .get(&key(), 1_000)
            .await
            .expect("get")
            .expect("record");
        assert_eq!(expired.state, M2nBootstrapState::Expired);
        assert!(expired.nccl_unique_id.is_empty());
    }
}
