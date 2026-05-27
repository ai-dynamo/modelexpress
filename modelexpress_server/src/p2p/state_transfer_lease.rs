// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer lease lifecycle operations for `P2pStateManager`.

use crate::p2p::backend::MetadataResult;
use crate::p2p::lease::{TransferLeaseRecord, bounded_lease_ttl_millis};
use crate::p2p::state::{BeginTransferLeaseParams, P2pStateManager};
use modelexpress_common::grpc::p2p::TransferLeaseStatus;
use tracing::debug;
use uuid::Uuid;

impl P2pStateManager {
    /// Create a durable ACTIVE transfer lease.
    pub async fn begin_transfer_lease(
        &self,
        params: BeginTransferLeaseParams,
    ) -> MetadataResult<TransferLeaseRecord> {
        let now = chrono::Utc::now().timestamp_millis();
        let ttl = bounded_lease_ttl_millis(params.ttl_millis);
        let record = TransferLeaseRecord {
            lease_id: params
                .lease_id
                .unwrap_or_else(|| Uuid::new_v4().to_string()),
            mx_source_id: params.mx_source_id,
            source_worker_id: params.source_worker_id,
            target_worker_id: params.target_worker_id,
            target_worker_rank: params.target_worker_rank,
            model_version: params.model_version,
            status: TransferLeaseStatus::Active as i32,
            created_at: now,
            updated_at: now,
            expires_at: now.saturating_add(ttl),
            error_message: String::new(),
            metadata: params.metadata,
        };

        self.get_backend()
            .await?
            .create_transfer_lease(record.clone())
            .await?;
        debug!(
            "Began transfer lease '{}' for source '{}' worker '{}'",
            record.lease_id, record.mx_source_id, record.source_worker_id
        );
        Ok(record)
    }

    /// Fetch a transfer lease and persist EXPIRED if its active TTL elapsed.
    pub async fn get_transfer_lease(
        &self,
        lease_id: &str,
    ) -> MetadataResult<Option<TransferLeaseRecord>> {
        let Some(record) = self
            .get_backend()
            .await?
            .get_transfer_lease(lease_id)
            .await?
        else {
            return Ok(None);
        };
        self.observe_transfer_lease(record).await.map(Some)
    }

    /// List transfer leases and persist EXPIRED for active records whose TTL elapsed.
    pub async fn list_transfer_leases(
        &self,
        mx_source_id: Option<String>,
        target_worker_id: Option<String>,
        status_filter: Option<TransferLeaseStatus>,
        model_version_filter: Option<u64>,
    ) -> MetadataResult<Vec<TransferLeaseRecord>> {
        let leases = self
            .get_backend()
            .await?
            .list_transfer_leases(mx_source_id, target_worker_id, None, model_version_filter)
            .await?;
        let mut observed = Vec::with_capacity(leases.len());
        for lease in leases {
            observed.push(self.observe_transfer_lease(lease).await?);
        }
        if let Some(status) = status_filter {
            observed.retain(|lease| lease.status == status as i32);
        }
        Ok(observed)
    }

    /// Observe active transfer leases and persist EXPIRED for abandoned attempts.
    pub async fn expire_transfer_leases(&self) -> MetadataResult<u32> {
        let leases = self
            .get_backend()
            .await?
            .list_transfer_leases(None, None, Some(TransferLeaseStatus::Active), None)
            .await?;
        let mut expired_count = 0u32;
        for lease in leases {
            let observed = self.observe_transfer_lease(lease).await?;
            if observed.status == TransferLeaseStatus::Expired as i32 {
                expired_count = expired_count.saturating_add(1);
            }
        }
        Ok(expired_count)
    }

    /// Remove terminal transfer leases whose last update is older than `max_age_millis`.
    pub async fn cleanup_terminal_transfer_leases(
        &self,
        max_age_millis: u64,
    ) -> MetadataResult<u32> {
        let now = chrono::Utc::now().timestamp_millis();
        let backend = self.get_backend().await?;
        let leases = backend.list_transfer_leases(None, None, None, None).await?;
        let mut removed_count = 0u32;
        for lease in leases {
            let age_millis = now.saturating_sub(lease.updated_at).max(0) as u64;
            if lease.is_terminal() && age_millis > max_age_millis {
                backend.remove_transfer_lease(&lease.lease_id).await?;
                removed_count = removed_count.saturating_add(1);
            }
        }
        Ok(removed_count)
    }

    /// Extend an active transfer lease with a fresh expiry.
    pub async fn renew_transfer_lease(
        &self,
        lease_id: &str,
        ttl_millis: u64,
    ) -> MetadataResult<TransferLeaseRecord> {
        let record = self
            .get_transfer_lease(lease_id)
            .await?
            .ok_or_else(|| format!("transfer lease '{}' not found", lease_id))?;
        if !record.is_active() {
            return Err(format!("transfer lease '{}' is not active", lease_id).into());
        }

        let now = chrono::Utc::now().timestamp_millis();
        let expires_at = now.saturating_add(bounded_lease_ttl_millis(ttl_millis));
        self.get_backend()
            .await?
            .renew_transfer_lease(lease_id, now, expires_at)
            .await?;

        self.get_transfer_lease(lease_id)
            .await?
            .ok_or_else(|| format!("transfer lease '{}' not found after renew", lease_id).into())
    }

    /// Mark an active transfer lease terminal.
    pub async fn complete_transfer_lease(
        &self,
        lease_id: &str,
        status: TransferLeaseStatus,
        error_message: &str,
    ) -> MetadataResult<TransferLeaseRecord> {
        let record = self
            .get_transfer_lease(lease_id)
            .await?
            .ok_or_else(|| format!("transfer lease '{}' not found", lease_id))?;
        if !record.is_active() && status != TransferLeaseStatus::Expired {
            return Err(format!("transfer lease '{}' is not active", lease_id).into());
        }

        let now = chrono::Utc::now().timestamp_millis();
        self.get_backend()
            .await?
            .finish_transfer_lease(lease_id, status, now, error_message)
            .await?;

        self.get_transfer_lease(lease_id).await?.ok_or_else(|| {
            format!("transfer lease '{}' not found after completion", lease_id).into()
        })
    }

    async fn observe_transfer_lease(
        &self,
        record: TransferLeaseRecord,
    ) -> MetadataResult<TransferLeaseRecord> {
        let now = chrono::Utc::now().timestamp_millis();
        if !record.is_expired_at(now) {
            return Ok(record);
        }
        self.get_backend()
            .await?
            .finish_transfer_lease(
                &record.lease_id,
                TransferLeaseStatus::Expired,
                now,
                "transfer lease expired",
            )
            .await?;
        Ok(record.observed_at(now))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::p2p::backend::MockMetadataBackend;
    use mockall::predicate::eq;
    use std::sync::Arc;

    fn active_lease_record(lease_id: &str) -> TransferLeaseRecord {
        TransferLeaseRecord {
            lease_id: lease_id.to_string(),
            mx_source_id: "abc123def456abcd".to_string(),
            source_worker_id: "source-worker".to_string(),
            target_worker_id: "target-worker".to_string(),
            target_worker_rank: 2,
            model_version: 9,
            status: TransferLeaseStatus::Active as i32,
            created_at: 1_000,
            updated_at: 1_000,
            expires_at: chrono::Utc::now().timestamp_millis() + 60_000,
            error_message: String::new(),
            metadata: std::collections::HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_begin_transfer_lease_calls_backend() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_create_transfer_lease()
            .withf(|lease| {
                lease.lease_id == "lease-1"
                    && lease.mx_source_id == "abc123def456abcd"
                    && lease.source_worker_id == "source-worker"
                    && lease.target_worker_id == "target-worker"
                    && lease.target_worker_rank == 3
                    && lease.model_version == 11
                    && lease.status == TransferLeaseStatus::Active as i32
                    && lease.expires_at > lease.created_at
                    && lease.metadata.get("role").map(String::as_str) == Some("trainer")
            })
            .once()
            .returning(|_| Ok(()));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("role".to_string(), "trainer".to_string());

        let lease = manager
            .begin_transfer_lease(BeginTransferLeaseParams {
                lease_id: Some("lease-1".to_string()),
                mx_source_id: "abc123def456abcd".to_string(),
                source_worker_id: "source-worker".to_string(),
                target_worker_id: "target-worker".to_string(),
                target_worker_rank: 3,
                model_version: 11,
                ttl_millis: 5_000,
                metadata,
            })
            .await
            .expect("begin_transfer_lease failed");

        assert_eq!(lease.lease_id, "lease-1");
        assert_eq!(lease.status, TransferLeaseStatus::Active as i32);
    }

    #[tokio::test]
    async fn test_renew_transfer_lease_calls_backend_for_active_lease() {
        let mut mock = MockMetadataBackend::new();
        let record = active_lease_record("lease-1");
        mock.expect_get_transfer_lease()
            .with(eq("lease-1"))
            .times(2)
            .returning(move |_| Ok(Some(record.clone())));
        mock.expect_renew_transfer_lease()
            .withf(|lease_id, updated_at, expires_at| {
                lease_id == "lease-1" && *expires_at > *updated_at
            })
            .once()
            .returning(|_, _, _| Ok(()));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        let lease = manager
            .renew_transfer_lease("lease-1", 5_000)
            .await
            .expect("renew_transfer_lease failed");

        assert_eq!(lease.lease_id, "lease-1");
    }

    #[tokio::test]
    async fn test_complete_transfer_lease_calls_backend() {
        let mut mock = MockMetadataBackend::new();
        let record = active_lease_record("lease-1");
        mock.expect_get_transfer_lease()
            .with(eq("lease-1"))
            .times(2)
            .returning(move |_| Ok(Some(record.clone())));
        mock.expect_finish_transfer_lease()
            .withf(|lease_id, status, _updated_at, error_message| {
                lease_id == "lease-1"
                    && *status == TransferLeaseStatus::Completed
                    && error_message.is_empty()
            })
            .once()
            .returning(|_, _, _, _| Ok(()));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        let lease = manager
            .complete_transfer_lease("lease-1", TransferLeaseStatus::Completed, "")
            .await
            .expect("complete_transfer_lease failed");

        assert_eq!(lease.lease_id, "lease-1");
    }

    #[tokio::test]
    async fn test_list_transfer_leases_filters_after_observing() {
        let mut mock = MockMetadataBackend::new();
        let active = active_lease_record("lease-active");
        let mut completed = active_lease_record("lease-complete");
        completed.status = TransferLeaseStatus::Completed as i32;
        mock.expect_list_transfer_leases()
            .with(
                eq(Some("abc123def456abcd".to_string())),
                eq(Some("target-worker".to_string())),
                eq(None),
                eq(Some(9)),
            )
            .once()
            .returning(move |_, _, _, _| Ok(vec![active.clone(), completed.clone()]));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        let leases = manager
            .list_transfer_leases(
                Some("abc123def456abcd".to_string()),
                Some("target-worker".to_string()),
                Some(TransferLeaseStatus::Completed),
                Some(9),
            )
            .await
            .expect("list_transfer_leases failed");

        assert_eq!(leases.len(), 1);
        assert_eq!(leases[0].lease_id, "lease-complete");
    }

    #[tokio::test]
    async fn test_expire_transfer_leases_marks_only_expired_active_records() {
        let mut mock = MockMetadataBackend::new();
        let now = chrono::Utc::now().timestamp_millis();
        let mut expired = active_lease_record("lease-expired");
        expired.expires_at = now - 1_000;
        let mut active = active_lease_record("lease-active");
        active.expires_at = now + 60_000;

        mock.expect_list_transfer_leases()
            .with(
                eq(None),
                eq(None),
                eq(Some(TransferLeaseStatus::Active)),
                eq(None),
            )
            .once()
            .returning(move |_, _, _, _| Ok(vec![expired.clone(), active.clone()]));
        mock.expect_finish_transfer_lease()
            .withf(|lease_id, status, _updated_at, error_message| {
                lease_id == "lease-expired"
                    && *status == TransferLeaseStatus::Expired
                    && error_message == "transfer lease expired"
            })
            .once()
            .returning(|_, _, _, _| Ok(()));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        let expired_count = manager
            .expire_transfer_leases()
            .await
            .expect("expire_transfer_leases failed");

        assert_eq!(expired_count, 1);
    }

    #[tokio::test]
    async fn test_cleanup_terminal_transfer_leases_removes_old_terminal_records() {
        let mut mock = MockMetadataBackend::new();
        let now = chrono::Utc::now().timestamp_millis();
        let mut completed_old = active_lease_record("lease-completed-old");
        completed_old.status = TransferLeaseStatus::Completed as i32;
        completed_old.updated_at = now - 7_200_000;
        let mut failed_recent = active_lease_record("lease-failed-recent");
        failed_recent.status = TransferLeaseStatus::Failed as i32;
        failed_recent.updated_at = now - 10_000;
        let active_old = active_lease_record("lease-active-old");

        mock.expect_list_transfer_leases()
            .with(eq(None), eq(None), eq(None), eq(None))
            .once()
            .returning(move |_, _, _, _| {
                Ok(vec![
                    completed_old.clone(),
                    failed_recent.clone(),
                    active_old.clone(),
                ])
            });
        mock.expect_remove_transfer_lease()
            .with(eq("lease-completed-old"))
            .once()
            .returning(|_| Ok(()));

        let manager = P2pStateManager::with_backend(Arc::new(mock));
        let removed_count = manager
            .cleanup_terminal_transfer_leases(3_600_000)
            .await
            .expect("cleanup_terminal_transfer_leases failed");

        assert_eq!(removed_count, 1);
    }
}
