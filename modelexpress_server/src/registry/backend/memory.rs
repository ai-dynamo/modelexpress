// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory model-registry backend for tests and local dev. Not persistent, single
//! process. Pairs with the in-memory P2P backend behind `MX_METADATA_BACKEND=memory`.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use modelexpress_common::models::{ModelProvider, ModelStatus};

use crate::registry::backend::{ClaimOutcome, ModelRecord, RegistryBackend, RegistryResult};

#[derive(Debug)]
struct DownloadLease {
    claim_id: String,
    expires_at: DateTime<Utc>,
}

#[derive(Default)]
struct InMemoryState {
    models: HashMap<String, ModelRecord>,
    leases: HashMap<String, DownloadLease>,
}

#[derive(Default)]
pub struct InMemoryRegistryBackend {
    state: Mutex<InMemoryState>,
}

impl InMemoryRegistryBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, InMemoryState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn lease_deadline(now: DateTime<Utc>, duration: std::time::Duration) -> DateTime<Utc> {
        chrono::TimeDelta::from_std(duration)
            .ok()
            .and_then(|duration| now.checked_add_signed(duration))
            .unwrap_or(now)
    }
}

#[async_trait]
impl RegistryBackend for InMemoryRegistryBackend {
    async fn connect(&self) -> RegistryResult<()> {
        Ok(())
    }

    async fn get_status(&self, model_name: &str) -> RegistryResult<Option<ModelStatus>> {
        Ok(self.lock().models.get(model_name).map(|r| r.status))
    }

    async fn get_model_record(&self, model_name: &str) -> RegistryResult<Option<ModelRecord>> {
        Ok(self.lock().models.get(model_name).cloned())
    }

    async fn set_status(
        &self,
        model_name: &str,
        provider: ModelProvider,
        status: ModelStatus,
        message: Option<String>,
    ) -> RegistryResult<()> {
        let now = Utc::now();
        let mut state = self.lock();
        state
            .models
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
        if status != ModelStatus::DOWNLOADING {
            state.leases.remove(model_name);
        }
        Ok(())
    }

    async fn touch_model(&self, model_name: &str) -> RegistryResult<()> {
        if let Some(record) = self.lock().models.get_mut(model_name) {
            record.last_used_at = Utc::now();
        }
        Ok(())
    }

    async fn delete_model(&self, model_name: &str) -> RegistryResult<()> {
        let mut state = self.lock();
        state.models.remove(model_name);
        state.leases.remove(model_name);
        Ok(())
    }

    async fn get_models_by_last_used(
        &self,
        limit: Option<u32>,
    ) -> RegistryResult<Vec<ModelRecord>> {
        let mut records: Vec<ModelRecord> = self.lock().models.values().cloned().collect();
        records.sort_by_key(|r| r.last_used_at);
        if let Some(limit) = limit {
            records.truncate(limit as usize);
        }
        Ok(records)
    }

    async fn get_status_counts(&self) -> RegistryResult<(u32, u32, u32)> {
        let state = self.lock();
        let mut downloading = 0u32;
        let mut downloaded = 0u32;
        let mut error = 0u32;
        for record in state.models.values() {
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
        claim_id: &str,
        lease_duration: std::time::Duration,
    ) -> RegistryResult<ClaimOutcome> {
        let now = Utc::now();
        let lease_expires_at = Self::lease_deadline(now, lease_duration);
        let mut state = self.lock();
        let expired = state
            .leases
            .get(model_name)
            .is_none_or(|lease| lease.expires_at <= now);
        match state.models.get_mut(model_name) {
            Some(existing) if existing.status == ModelStatus::DOWNLOADING && expired => {
                existing.provider = provider;
                existing.last_used_at = now;
                existing.message = Some("Taking over expired download lease...".to_string());
                state.leases.insert(
                    model_name.to_string(),
                    DownloadLease {
                        claim_id: claim_id.to_string(),
                        expires_at: lease_expires_at,
                    },
                );
                Ok(ClaimOutcome::TookOver)
            }
            Some(existing) => Ok(ClaimOutcome::AlreadyExists(existing.status)),
            None => {
                state.models.insert(
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
                state.leases.insert(
                    model_name.to_string(),
                    DownloadLease {
                        claim_id: claim_id.to_string(),
                        expires_at: lease_expires_at,
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
        claim_id: &str,
        lease_duration: std::time::Duration,
    ) -> RegistryResult<bool> {
        let mut state = self.lock();
        let now = Utc::now();
        let lease_expires_at = Self::lease_deadline(now, lease_duration);
        match state.models.get_mut(model_name) {
            Some(record) if record.status == ModelStatus::ERROR => {
                record.provider = provider;
                record.status = ModelStatus::DOWNLOADING;
                record.message = Some("Retrying download...".to_string());
                record.last_used_at = now;
                state.leases.insert(
                    model_name.to_string(),
                    DownloadLease {
                        claim_id: claim_id.to_string(),
                        expires_at: lease_expires_at,
                    },
                );
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    async fn refresh_download_claim(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        lease_duration: std::time::Duration,
    ) -> RegistryResult<bool> {
        let mut state = self.lock();
        let now = Utc::now();
        let lease_expires_at = Self::lease_deadline(now, lease_duration);
        let owns_claim = state
            .leases
            .get(model_name)
            .is_some_and(|lease| lease.claim_id == claim_id);
        if !owns_claim
            || state
                .models
                .get(model_name)
                .is_none_or(|record| record.status != ModelStatus::DOWNLOADING)
        {
            return Ok(false);
        }
        let InMemoryState { models, leases } = &mut *state;
        match (models.get_mut(model_name), leases.get_mut(model_name)) {
            (Some(record), Some(lease)) => {
                record.provider = provider;
                record.last_used_at = now;
                lease.expires_at = lease_expires_at;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    async fn finish_download_claim(
        &self,
        model_name: &str,
        provider: ModelProvider,
        claim_id: &str,
        status: ModelStatus,
        message: Option<String>,
    ) -> RegistryResult<bool> {
        let mut state = self.lock();
        let owns_claim = state
            .leases
            .get(model_name)
            .is_some_and(|lease| lease.claim_id == claim_id);
        if !owns_claim
            || state
                .models
                .get(model_name)
                .is_none_or(|record| record.status != ModelStatus::DOWNLOADING)
        {
            return Ok(false);
        }
        let Some(record) = state.models.get_mut(model_name) else {
            return Ok(false);
        };
        record.provider = provider;
        record.status = status;
        record.message = message;
        record.last_used_at = Utc::now();
        state.leases.remove(model_name);
        Ok(true)
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
                .try_claim_for_download(
                    "fresh",
                    ModelProvider::HuggingFace,
                    "owner",
                    std::time::Duration::from_secs(30),
                )
                .await
                .expect("claim"),
            ClaimOutcome::Claimed
        );
        assert_eq!(
            backend
                .try_claim_for_download(
                    "fresh",
                    ModelProvider::HuggingFace,
                    "other",
                    std::time::Duration::from_secs(30),
                )
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
                .try_reset_error_for_retry(
                    "m",
                    ModelProvider::HuggingFace,
                    "owner",
                    std::time::Duration::from_secs(30),
                )
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
                .try_reset_error_for_retry(
                    "m",
                    ModelProvider::HuggingFace,
                    "owner",
                    std::time::Duration::from_secs(30),
                )
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
                .try_reset_error_for_retry(
                    "m",
                    ModelProvider::HuggingFace,
                    "owner",
                    std::time::Duration::from_secs(30),
                )
                .await
                .expect("reset from error"),
            "ERROR flips to DOWNLOADING"
        );
        assert_eq!(
            backend.get_status("m").await.expect("get"),
            Some(ModelStatus::DOWNLOADING)
        );
    }

    #[tokio::test]
    async fn expired_lease_can_be_reclaimed_and_stale_owner_is_fenced() {
        let backend = InMemoryRegistryBackend::new();
        let provider = ModelProvider::HuggingFace;
        assert_eq!(
            backend
                .try_claim_for_download("m", provider, "owner-1", std::time::Duration::ZERO,)
                .await
                .expect("initial claim"),
            ClaimOutcome::Claimed
        );
        assert_eq!(
            backend
                .try_claim_for_download(
                    "m",
                    provider,
                    "owner-2",
                    std::time::Duration::from_secs(30),
                )
                .await
                .expect("takeover"),
            ClaimOutcome::TookOver
        );
        assert!(
            !backend
                .finish_download_claim("m", provider, "owner-1", ModelStatus::DOWNLOADED, None,)
                .await
                .expect("stale finish")
        );
        assert!(
            backend
                .refresh_download_claim(
                    "m",
                    provider,
                    "owner-2",
                    std::time::Duration::from_secs(30),
                )
                .await
                .expect("refresh")
        );
        assert!(
            backend
                .finish_download_claim(
                    "m",
                    provider,
                    "owner-2",
                    ModelStatus::DOWNLOADED,
                    Some("done".to_string()),
                )
                .await
                .expect("finish")
        );
        assert_eq!(
            backend.get_status("m").await.expect("get"),
            Some(ModelStatus::DOWNLOADED)
        );
    }
}
