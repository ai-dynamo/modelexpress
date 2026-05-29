// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory model-registry backend for tests and local dev. Not persistent, single
//! process. Pairs with the in-memory P2P backend behind `MX_METADATA_BACKEND=memory`.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use async_trait::async_trait;
use chrono::Utc;
use modelexpress_common::models::{ModelProvider, ModelStatus};

use crate::registry::backend::{ClaimOutcome, ModelRecord, RegistryBackend, RegistryResult};

#[derive(Default)]
pub struct InMemoryRegistryBackend {
    models: Mutex<HashMap<String, ModelRecord>>,
}

impl InMemoryRegistryBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, ModelRecord>> {
        self.models.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[async_trait]
impl RegistryBackend for InMemoryRegistryBackend {
    async fn connect(&self) -> RegistryResult<()> {
        Ok(())
    }

    async fn get_status(&self, model_name: &str) -> RegistryResult<Option<ModelStatus>> {
        Ok(self.lock().get(model_name).map(|r| r.status))
    }

    async fn get_model_record(&self, model_name: &str) -> RegistryResult<Option<ModelRecord>> {
        Ok(self.lock().get(model_name).cloned())
    }

    async fn set_status(
        &self,
        model_name: &str,
        provider: ModelProvider,
        status: ModelStatus,
        message: Option<String>,
    ) -> RegistryResult<()> {
        let now = Utc::now();
        let mut models = self.lock();
        models
            .entry(model_name.to_string())
            .and_modify(|record| {
                record.provider = provider;
                record.status = status;
                record.message = message.clone();
                record.last_used_at = now;
            })
            .or_insert_with(|| ModelRecord {
                model_name: model_name.to_string(),
                provider,
                status,
                created_at: now,
                last_used_at: now,
                message,
            });
        Ok(())
    }

    async fn touch_model(&self, model_name: &str) -> RegistryResult<()> {
        if let Some(record) = self.lock().get_mut(model_name) {
            record.last_used_at = Utc::now();
        }
        Ok(())
    }

    async fn delete_model(&self, model_name: &str) -> RegistryResult<()> {
        self.lock().remove(model_name);
        Ok(())
    }

    async fn get_models_by_last_used(
        &self,
        limit: Option<u32>,
    ) -> RegistryResult<Vec<ModelRecord>> {
        let mut records: Vec<ModelRecord> = self.lock().values().cloned().collect();
        records.sort_by_key(|r| r.last_used_at);
        if let Some(limit) = limit {
            records.truncate(limit as usize);
        }
        Ok(records)
    }

    async fn get_status_counts(&self) -> RegistryResult<(u32, u32, u32)> {
        let models = self.lock();
        let mut downloading = 0u32;
        let mut downloaded = 0u32;
        let mut error = 0u32;
        for record in models.values() {
            match record.status {
                ModelStatus::DOWNLOADING => downloading = downloading.saturating_add(1),
                ModelStatus::DOWNLOADED => downloaded = downloaded.saturating_add(1),
                ModelStatus::ERROR => error = error.saturating_add(1),
            }
        }
        Ok((downloading, downloaded, error))
    }

    async fn try_claim_for_download(
        &self,
        model_name: &str,
        provider: ModelProvider,
    ) -> RegistryResult<ClaimOutcome> {
        let now = Utc::now();
        let mut models = self.lock();
        match models.get(model_name) {
            Some(existing) => Ok(ClaimOutcome::AlreadyExists(existing.status)),
            None => {
                models.insert(
                    model_name.to_string(),
                    ModelRecord {
                        model_name: model_name.to_string(),
                        provider,
                        status: ModelStatus::DOWNLOADING,
                        created_at: now,
                        last_used_at: now,
                        message: None,
                    },
                );
                Ok(ClaimOutcome::Claimed)
            }
        }
    }

    async fn try_reset_error_for_retry(
        &self,
        model_name: &str,
        provider: ModelProvider,
    ) -> RegistryResult<bool> {
        let mut models = self.lock();
        match models.get_mut(model_name) {
            Some(record) if record.status == ModelStatus::ERROR => {
                record.provider = provider;
                record.status = ModelStatus::DOWNLOADING;
                record.message = Some("Retrying download...".to_string());
                record.last_used_at = Utc::now();
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    // status round-trips, and a claim is exclusive
    #[tokio::test]
    async fn set_get_and_claim() {
        let backend = InMemoryRegistryBackend::new();
        backend
            .set_status(
                "m",
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADED,
                None,
            )
            .await
            .expect("set");
        assert_eq!(
            backend.get_status("m").await.expect("get"),
            Some(ModelStatus::DOWNLOADED)
        );

        assert_eq!(
            backend
                .try_claim_for_download("fresh", ModelProvider::HuggingFace)
                .await
                .expect("claim"),
            ClaimOutcome::Claimed
        );
        assert_eq!(
            backend
                .try_claim_for_download("fresh", ModelProvider::HuggingFace)
                .await
                .expect("claim again"),
            ClaimOutcome::AlreadyExists(ModelStatus::DOWNLOADING)
        );
    }

    // created_at survives an update; status and message are overwritten
    #[tokio::test]
    async fn set_status_update_preserves_created_at() {
        let backend = InMemoryRegistryBackend::new();
        backend
            .set_status(
                "m",
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADING,
                None,
            )
            .await
            .expect("first set");
        let first = backend
            .get_model_record("m")
            .await
            .expect("get")
            .expect("present");

        backend
            .set_status(
                "m",
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADED,
                Some("done".to_string()),
            )
            .await
            .expect("second set");
        let second = backend
            .get_model_record("m")
            .await
            .expect("get")
            .expect("present");

        assert_eq!(
            second.created_at, first.created_at,
            "created_at must survive an update"
        );
        assert_eq!(second.status, ModelStatus::DOWNLOADED);
        assert_eq!(second.message.as_deref(), Some("done"));
    }

    // oldest-first ordering, touch bumps to newest, and limit truncates
    #[tokio::test]
    async fn get_models_by_last_used_orders_oldest_first_and_limits() {
        let backend = InMemoryRegistryBackend::new();
        for name in ["a", "b", "c"] {
            backend
                .set_status(
                    name,
                    ModelProvider::HuggingFace,
                    ModelStatus::DOWNLOADED,
                    None,
                )
                .await
                .expect("set");
            // distinct last_used_at so the ordering assertion is deterministic
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }

        let all = backend.get_models_by_last_used(None).await.expect("all");
        let names: Vec<_> = all.iter().map(|r| r.model_name.as_str()).collect();
        assert_eq!(names, ["a", "b", "c"], "oldest first");

        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        backend.touch_model("a").await.expect("touch");
        let reordered = backend.get_models_by_last_used(None).await.expect("all");
        let names: Vec<_> = reordered.iter().map(|r| r.model_name.as_str()).collect();
        assert_eq!(names, ["b", "c", "a"], "touched model moves to newest");

        let limited = backend
            .get_models_by_last_used(Some(2))
            .await
            .expect("limited");
        assert_eq!(limited.len(), 2, "limit truncates");
    }

    // counts tally per status
    #[tokio::test]
    async fn get_status_counts_tallies_each_status() {
        let backend = InMemoryRegistryBackend::new();
        for (name, status) in [
            ("dl", ModelStatus::DOWNLOADING),
            ("done1", ModelStatus::DOWNLOADED),
            ("done2", ModelStatus::DOWNLOADED),
            ("err", ModelStatus::ERROR),
        ] {
            backend
                .set_status(name, ModelProvider::HuggingFace, status, None)
                .await
                .expect("set");
        }
        assert_eq!(
            backend.get_status_counts().await.expect("counts"),
            (1, 2, 1)
        );
    }

    // touch and delete on an unknown model are no-ops, not errors
    #[tokio::test]
    async fn touch_and_delete_unknown_are_noops() {
        let backend = InMemoryRegistryBackend::new();
        backend.touch_model("ghost").await.expect("touch unknown");
        backend.delete_model("ghost").await.expect("delete unknown");
        assert!(backend.get_status("ghost").await.expect("get").is_none());
    }

    // the error-retry CAS only fires when the model is currently in ERROR
    #[tokio::test]
    async fn try_reset_error_for_retry_only_from_error() {
        let backend = InMemoryRegistryBackend::new();
        assert!(
            !backend
                .try_reset_error_for_retry("m", ModelProvider::HuggingFace)
                .await
                .expect("reset unknown"),
            "unknown model: nothing to reset"
        );

        backend
            .set_status(
                "m",
                ModelProvider::HuggingFace,
                ModelStatus::DOWNLOADED,
                None,
            )
            .await
            .expect("set downloaded");
        assert!(
            !backend
                .try_reset_error_for_retry("m", ModelProvider::HuggingFace)
                .await
                .expect("reset non-error"),
            "non-error status: no reset"
        );

        backend
            .set_status(
                "m",
                ModelProvider::HuggingFace,
                ModelStatus::ERROR,
                Some("boom".to_string()),
            )
            .await
            .expect("set error");
        assert!(
            backend
                .try_reset_error_for_retry("m", ModelProvider::HuggingFace)
                .await
                .expect("reset from error"),
            "ERROR flips to DOWNLOADING"
        );
        assert_eq!(
            backend.get_status("m").await.expect("get"),
            Some(ModelStatus::DOWNLOADING)
        );
    }
}
