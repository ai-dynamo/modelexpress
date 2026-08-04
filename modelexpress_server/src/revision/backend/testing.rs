// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic test double for catalog state and in-process router tests.
//! This module is not a selectable revision-catalog backend.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use async_trait::async_trait;
use modelexpress_common::grpc::revision::{
    ReceiverStateRecord, RevisionLifecycleState, RevisionRecord,
};

use super::{CatalogResult, CommitOutcome, PublishReadyOutcome, RevisionCatalogBackend};

#[derive(Clone)]
struct StoredRevision {
    record: RevisionRecord,
}

#[derive(Default)]
struct TestCatalog {
    revisions: HashMap<(String, String), StoredRevision>,
    receivers: HashMap<(String, String, String), ReceiverStateRecord>,
}

#[derive(Default)]
pub struct TestRevisionCatalogBackend {
    state: Mutex<TestCatalog>,
}

impl TestRevisionCatalogBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, TestCatalog> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[async_trait]
impl RevisionCatalogBackend for TestRevisionCatalogBackend {
    async fn connect(&self) -> CatalogResult<()> {
        Ok(())
    }

    async fn publish_ready(&self, record: RevisionRecord) -> CatalogResult<PublishReadyOutcome> {
        let manifest = record
            .manifest
            .as_ref()
            .ok_or_else(|| "revision record is missing manifest".to_string())?;
        let key = (manifest.model_id.clone(), manifest.version.clone());
        let mut state = self.lock();
        match state.revisions.get(&key) {
            Some(existing) if existing.record.manifest == record.manifest => {
                Ok(PublishReadyOutcome::Existing(existing.record.clone()))
            }
            Some(_) => Ok(PublishReadyOutcome::Conflict),
            None => {
                state.revisions.insert(
                    key,
                    StoredRevision {
                        record: record.clone(),
                    },
                );
                Ok(PublishReadyOutcome::Created(record))
            }
        }
    }

    async fn get_revision(
        &self,
        model_id: &str,
        version: &str,
    ) -> CatalogResult<Option<RevisionRecord>> {
        Ok(self
            .lock()
            .revisions
            .get(&(model_id.to_string(), version.to_string()))
            .map(|stored| stored.record.clone()))
    }

    async fn list_revisions(&self, model_id: &str) -> CatalogResult<Vec<RevisionRecord>> {
        let mut records: Vec<_> = self
            .lock()
            .revisions
            .values()
            .filter(|stored| {
                stored
                    .record
                    .manifest
                    .as_ref()
                    .is_some_and(|manifest| manifest.model_id == model_id)
            })
            .map(|stored| stored.record.clone())
            .collect();
        records.sort_by(|left, right| {
            left.created_at_unix_ms
                .cmp(&right.created_at_unix_ms)
                .then_with(|| {
                    let left_version = left
                        .manifest
                        .as_ref()
                        .map_or("", |manifest| manifest.version.as_str());
                    let right_version = right
                        .manifest
                        .as_ref()
                        .map_or("", |manifest| manifest.version.as_str());
                    left_version.cmp(right_version)
                })
        });
        Ok(records)
    }

    async fn commit_revision(
        &self,
        model_id: &str,
        version: &str,
        changed_at_unix_ms: u64,
    ) -> CatalogResult<CommitOutcome> {
        let key = (model_id.to_string(), version.to_string());
        let mut state = self.lock();
        let Some(stored) = state.revisions.get_mut(&key) else {
            return Ok(CommitOutcome::NotFound);
        };
        let record = &mut stored.record;
        match RevisionLifecycleState::try_from(record.state).ok() {
            Some(RevisionLifecycleState::Ready) => {
                record.state = RevisionLifecycleState::Committed as i32;
                record.state_changed_at_unix_ms = changed_at_unix_ms;
                Ok(CommitOutcome::Committed(record.clone()))
            }
            Some(RevisionLifecycleState::Committed) => {
                Ok(CommitOutcome::AlreadyCommitted(record.clone()))
            }
            _ => Ok(CommitOutcome::InvalidState(record.clone())),
        }
    }

    async fn upsert_receiver_state(
        &self,
        record: ReceiverStateRecord,
    ) -> CatalogResult<ReceiverStateRecord> {
        let key = (
            record.model_id.clone(),
            record.version.clone(),
            record.receiver_id.clone(),
        );
        let mut state = self.lock();
        if let Some(existing) = state.receivers.get(&key)
            && existing.model_id == record.model_id
            && existing.version == record.version
            && existing.receiver_id == record.receiver_id
            && existing.state == record.state
            && existing.installed_version == record.installed_version
            && existing.detail == record.detail
        {
            return Ok(existing.clone());
        }
        state.receivers.insert(key, record.clone());
        Ok(record)
    }

    async fn list_receiver_states(
        &self,
        model_id: &str,
        version: &str,
    ) -> CatalogResult<Vec<ReceiverStateRecord>> {
        let mut receivers: Vec<_> = self
            .lock()
            .receivers
            .values()
            .filter(|record| record.model_id == model_id && record.version == version)
            .cloned()
            .collect();
        receivers.sort_by(|left, right| left.receiver_id.cmp(&right.receiver_id));
        Ok(receivers)
    }
}
