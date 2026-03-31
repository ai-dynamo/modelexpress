// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Server-side reaper for stale source detection and garbage collection.
//!
//! Periodically scans all workers in the metadata backend:
//! 1. **Stale detection**: READY workers whose `updated_at` exceeds the heartbeat
//!    timeout are marked STALE.
//! 2. **Garbage collection**: STALE workers older than the GC timeout are deleted.
//!
//! Safe to run on every server replica — all operations are idempotent.

use crate::state::P2pStateManager;
use modelexpress_common::grpc::p2p::SourceStatus;
use std::sync::Arc;
use tokio::sync::oneshot;
use tracing::{debug, error, info, warn};

/// Read an environment variable as `u64`, falling back to `default`.
fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Run the reaper loop until the shutdown signal fires.
pub async fn run_reaper(state: Arc<P2pStateManager>, shutdown: oneshot::Receiver<()>) {
    let scan_interval_secs = env_u64("MX_REAPER_SCAN_INTERVAL_SECS", 30);
    let heartbeat_timeout_secs = env_u64("MX_HEARTBEAT_TIMEOUT_SECS", 90);
    let gc_timeout_secs = env_u64("MX_GC_TIMEOUT_SECS", 3600);
    let heartbeat_timeout_ms = heartbeat_timeout_secs.saturating_mul(1000);
    let gc_timeout_ms = gc_timeout_secs.saturating_mul(1000);

    info!(
        "Reaper started (scan_interval={}s, heartbeat_timeout={}s, gc_timeout={}s)",
        scan_interval_secs, heartbeat_timeout_secs, gc_timeout_secs,
    );

    let mut interval = tokio::time::interval(std::time::Duration::from_secs(scan_interval_secs));
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            _ = interval.tick() => {
                if let Err(e) = reap_once(&state, heartbeat_timeout_ms, gc_timeout_ms).await {
                    warn!("Reaper scan failed: {}", e);
                }
            }
            _ = &mut shutdown => {
                info!("Reaper received shutdown signal");
                break;
            }
        }
    }
}

/// Single reaper pass: mark stale, then garbage-collect.
async fn reap_once(
    state: &P2pStateManager,
    heartbeat_timeout_ms: u64,
    gc_timeout_ms: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let now = chrono::Utc::now().timestamp_millis();
    let workers = state.list_workers(None, None).await?;

    let mut stale_count = 0u32;
    let mut gc_count = 0u32;

    for w in &workers {
        let age_ms = now.saturating_sub(w.updated_at).max(0) as u64;

        let is_active =
            w.status == SourceStatus::Ready as i32 || w.status == SourceStatus::Initializing as i32;

        if is_active && age_ms > heartbeat_timeout_ms {
            if let Err(e) = state
                .update_worker_status(
                    &w.source_id,
                    &w.worker_id,
                    w.worker_rank,
                    SourceStatus::Stale,
                )
                .await
            {
                error!(
                    "Reaper: failed to mark STALE: source={} worker={}: {}",
                    w.source_id, w.worker_id, e
                );
            } else {
                stale_count = stale_count.saturating_add(1);
            }
        } else if w.status == SourceStatus::Stale as i32 && age_ms > gc_timeout_ms {
            if let Err(e) = state.remove_worker(&w.source_id, &w.worker_id).await {
                error!(
                    "Reaper: failed to GC worker: source={} worker={}: {}",
                    w.source_id, w.worker_id, e
                );
            } else {
                gc_count = gc_count.saturating_add(1);
            }
        }
    }

    if stale_count > 0 || gc_count > 0 {
        info!(
            "Reaper: marked {} stale, garbage-collected {}",
            stale_count, gc_count
        );
    } else {
        debug!(
            "Reaper: no action needed ({} workers scanned)",
            workers.len()
        );
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::metadata_backend::{MockMetadataBackend, SourceInstanceInfo};

    #[tokio::test]
    async fn test_reap_marks_stale_when_heartbeat_expired() {
        let now = chrono::Utc::now().timestamp_millis();
        let old_time = now - 120_000; // 120s ago

        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers().once().returning(move |_, _| {
            Ok(vec![SourceInstanceInfo {
                source_id: "src1".into(),
                worker_id: "w1".into(),
                model_name: "model".into(),
                worker_rank: 0,
                status: SourceStatus::Ready as i32,
                updated_at: old_time,
            }])
        });
        mock.expect_update_status()
            .withf(|sid, wid, rank, status, _| {
                sid == "src1" && wid == "w1" && *rank == 0 && *status == SourceStatus::Stale
            })
            .once()
            .returning(|_, _, _, _, _| Ok(()));

        let state = P2pStateManager::with_backend(Arc::new(mock));
        reap_once(&state, 90_000, 3_600_000)
            .await
            .expect("reap_once failed");
    }

    #[tokio::test]
    async fn test_reap_gc_removes_old_stale_workers() {
        let now = chrono::Utc::now().timestamp_millis();
        let very_old = now - 7_200_000; // 2 hours ago

        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers().once().returning(move |_, _| {
            Ok(vec![SourceInstanceInfo {
                source_id: "src1".into(),
                worker_id: "w1".into(),
                model_name: "model".into(),
                worker_rank: 0,
                status: SourceStatus::Stale as i32,
                updated_at: very_old,
            }])
        });
        mock.expect_remove_worker()
            .withf(|sid, wid| sid == "src1" && wid == "w1")
            .once()
            .returning(|_, _| Ok(()));

        let state = P2pStateManager::with_backend(Arc::new(mock));
        reap_once(&state, 90_000, 3_600_000)
            .await
            .expect("reap_once failed");
    }

    #[tokio::test]
    async fn test_reap_skips_healthy_workers() {
        let now = chrono::Utc::now().timestamp_millis();
        let recent = now - 10_000; // 10s ago

        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers().once().returning(move |_, _| {
            Ok(vec![SourceInstanceInfo {
                source_id: "src1".into(),
                worker_id: "w1".into(),
                model_name: "model".into(),
                worker_rank: 0,
                status: SourceStatus::Ready as i32,
                updated_at: recent,
            }])
        });
        // No update_status or remove_worker calls expected

        let state = P2pStateManager::with_backend(Arc::new(mock));
        reap_once(&state, 90_000, 3_600_000)
            .await
            .expect("reap_once failed");
    }
}
