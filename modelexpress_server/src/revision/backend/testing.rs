// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic test double selected for `BackendConfig::Memory` when the
//! `integration-tests` feature is enabled.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use async_trait::async_trait;
use modelexpress_common::grpc::revision::{RevisionRecord, RevisionState};

use super::{CatalogResult, CommitOutcome, PublishReadyOutcome, RevisionCatalogBackend};

#[derive(Default)]
struct TestCatalog {
    revisions: HashMap<(String, String), RevisionRecord>,
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

    #[cfg(test)]
    pub fn insert(&self, record: RevisionRecord) {
        let Some(manifest) = record.manifest.as_ref() else {
            panic!("test record must include a manifest");
        };
        self.lock().revisions.insert(
            (manifest.model_id.clone(), manifest.target_version.clone()),
            record,
        );
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
        let key = (manifest.model_id.clone(), manifest.target_version.clone());
        let mut state = self.lock();
        match state.revisions.get(&key) {
            Some(existing) if existing.manifest == record.manifest => {
                Ok(PublishReadyOutcome::Existing(existing.clone()))
            }
            Some(_) => Ok(PublishReadyOutcome::Conflict),
            None => {
                state.revisions.insert(key, record.clone());
                Ok(PublishReadyOutcome::Created(record))
            }
        }
    }

    async fn get_revision(
        &self,
        model_id: &str,
        target_version: &str,
    ) -> CatalogResult<Option<RevisionRecord>> {
        Ok(self
            .lock()
            .revisions
            .get(&(model_id.to_string(), target_version.to_string()))
            .cloned())
    }

    async fn commit_revision(
        &self,
        model_id: &str,
        target_version: &str,
    ) -> CatalogResult<CommitOutcome> {
        let key = (model_id.to_string(), target_version.to_string());
        let mut state = self.lock();
        let Some(record) = state.revisions.get_mut(&key) else {
            return Ok(CommitOutcome::NotFound);
        };
        match RevisionState::try_from(record.state).ok() {
            Some(RevisionState::Ready) => {
                record.state = RevisionState::Committed as i32;
                Ok(CommitOutcome::Committed(record.clone()))
            }
            Some(RevisionState::Committed) => Ok(CommitOutcome::AlreadyCommitted(record.clone())),
            _ => Ok(CommitOutcome::InvalidState(record.clone())),
        }
    }
}
