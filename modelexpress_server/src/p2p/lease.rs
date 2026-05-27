// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable transfer lease records for P2P/RL weight pulls.

use modelexpress_common::grpc::p2p::{TransferLease, TransferLeaseStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default active lease TTL when the caller does not provide one.
pub const DEFAULT_TRANSFER_LEASE_TTL_MILLIS: i64 = 30_000;

/// Bound caller-provided TTLs so stale ACTIVE records converge quickly.
pub const MAX_TRANSFER_LEASE_TTL_MILLIS: i64 = 10 * 60 * 1000;

/// Persisted transfer lease record shared by metadata backends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferLeaseRecord {
    pub lease_id: String,
    pub mx_source_id: String,
    pub source_worker_id: String,
    pub target_worker_id: String,
    pub target_worker_rank: u32,
    pub model_version: u64,
    pub status: i32,
    pub created_at: i64,
    pub updated_at: i64,
    pub expires_at: i64,
    pub error_message: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl TransferLeaseRecord {
    pub fn active_status() -> i32 {
        TransferLeaseStatus::Active as i32
    }

    pub fn is_active(&self) -> bool {
        self.status == Self::active_status()
    }

    pub fn is_expired_at(&self, now_millis: i64) -> bool {
        self.is_active() && self.expires_at <= now_millis
    }

    pub fn observed_at(mut self, now_millis: i64) -> Self {
        if self.is_expired_at(now_millis) {
            self.status = TransferLeaseStatus::Expired as i32;
            self.updated_at = now_millis;
            self.error_message = "transfer lease expired".to_string();
        }
        self
    }
}

impl From<TransferLeaseRecord> for TransferLease {
    fn from(record: TransferLeaseRecord) -> Self {
        Self {
            lease_id: record.lease_id,
            mx_source_id: record.mx_source_id,
            source_worker_id: record.source_worker_id,
            target_worker_id: record.target_worker_id,
            target_worker_rank: record.target_worker_rank,
            model_version: record.model_version,
            status: record.status,
            created_at: record.created_at,
            updated_at: record.updated_at,
            expires_at: record.expires_at,
            error_message: record.error_message,
            metadata: record.metadata,
        }
    }
}

pub fn bounded_lease_ttl_millis(ttl_millis: u64) -> i64 {
    if ttl_millis == 0 {
        return DEFAULT_TRANSFER_LEASE_TTL_MILLIS;
    }
    let ttl = i64::try_from(ttl_millis).unwrap_or(MAX_TRANSFER_LEASE_TTL_MILLIS);
    ttl.clamp(1_000, MAX_TRANSFER_LEASE_TTL_MILLIS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_lease_ttl_uses_default_for_zero() {
        assert_eq!(
            bounded_lease_ttl_millis(0),
            DEFAULT_TRANSFER_LEASE_TTL_MILLIS
        );
    }

    #[test]
    fn test_bounded_lease_ttl_clamps_extremes() {
        assert_eq!(bounded_lease_ttl_millis(1), 1_000);
        assert_eq!(
            bounded_lease_ttl_millis(u64::MAX),
            MAX_TRANSFER_LEASE_TTL_MILLIS
        );
    }

    #[test]
    fn test_observed_at_marks_active_lease_expired() {
        let record = TransferLeaseRecord {
            lease_id: "lease".to_string(),
            mx_source_id: "source".to_string(),
            source_worker_id: "source-worker".to_string(),
            target_worker_id: "target-worker".to_string(),
            target_worker_rank: 0,
            model_version: 7,
            status: TransferLeaseStatus::Active as i32,
            created_at: 100,
            updated_at: 100,
            expires_at: 200,
            error_message: String::new(),
            metadata: HashMap::new(),
        };

        let observed = record.observed_at(201);

        assert_eq!(observed.status, TransferLeaseStatus::Expired as i32);
        assert_eq!(observed.updated_at, 201);
        assert_eq!(observed.error_message, "transfer lease expired");
    }
}
