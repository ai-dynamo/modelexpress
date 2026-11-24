// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Server-side reaper for stale source detection and garbage collection.
//!
//! Periodically scans all workers in the metadata backend:
//! 1. **Stale detection**: workers not already STALE whose `updated_at` exceeds the
//!    heartbeat timeout are marked STALE. That includes UNKNOWN (the proto default),
//!    which a worker is left in when it dies between publishing its metadata and its
//!    first status update.
//! 2. **Garbage collection**: STALE workers older than the GC timeout are deleted.
//!
//! Safe to run on every server replica — all operations are idempotent.

use crate::p2p::state::P2pStateManager;
use modelexpress_common::grpc::p2p::SourceStatus;
use std::sync::Arc;
use tokio::sync::oneshot;
use tracing::{debug, error, info, warn};

/// Run the reaper loop until the shutdown signal fires.
pub async fn run_reaper(state: Arc<P2pStateManager>, shutdown: oneshot::Receiver<()>) {
    let scan_interval_secs = modelexpress_common::envs::reaper_scan_interval_secs();
    let heartbeat_timeout_secs = modelexpress_common::envs::heartbeat_timeout_secs();
    let gc_timeout_secs = modelexpress_common::envs::gc_timeout_secs();
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

        // Anything but STALE: a record left at UNKNOWN (or an unrecognized value) gets no
        // more heartbeats, and STALE is the only state the GC branch below removes.
        let is_active = w.status != SourceStatus::Stale as i32;

        if is_active && age_ms > heartbeat_timeout_ms {
            if let Err(e) = state
                .update_worker_status(
                    &w.source_id,
                    &w.worker_id,
                    w.worker_rank,
                    SourceStatus::Stale,
                    None,
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
mod tests {
    use super::*;
    use crate::p2p::backend::{MockMetadataBackend, SourceInstanceInfo};

    /// Fresh publications receive server timestamps and survive an immediate reaper scan.
    #[cfg(feature = "memory-backend")]
    #[tokio::test]
    async fn test_reap_preserves_newly_published_unknown_workers() {
        use crate::p2p::backend::memory::InMemoryMetadataBackend;
        use modelexpress_common::grpc::p2p::{SourceIdentity, WorkerMetadata};

        for updated_at in [0, i64::MAX] {
            let state = P2pStateManager::with_backend(Arc::new(InMemoryMetadataBackend::new()));
            let before_publish = chrono::Utc::now().timestamp_millis();
            state
                .publish_metadata(
                    &SourceIdentity {
                        model_name: "model".into(),
                        ..Default::default()
                    },
                    "w1",
                    WorkerMetadata {
                        updated_at,
                        ..Default::default()
                    },
                    "",
                    "",
                    "",
                )
                .await
                .expect("publish_metadata failed");
            let after_publish = chrono::Utc::now().timestamp_millis();

            reap_once(&state, 90_000, 3_600_000)
                .await
                .expect("reap_once failed");

            let workers = state.list_workers(None, None).await.expect("list failed");
            assert_eq!(workers.len(), 1);
            assert_eq!(workers[0].status, SourceStatus::Unknown as i32);
            assert!((before_publish..=after_publish).contains(&workers[0].updated_at));
        }
    }

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
                accelerator: "cuda".into(),
                source_load: None,
                topology: Default::default(),
                training_step: None,
                layout_signature: None,
            }])
        });
        mock.expect_update_status()
            .withf(|sid, wid, rank, status, _, _| {
                sid == "src1" && wid == "w1" && *rank == 0 && *status == SourceStatus::Stale
            })
            .once()
            .returning(|_, _, _, _, _, _| Ok(()));

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
                accelerator: "cuda".into(),
                source_load: None,
                topology: Default::default(),
                training_step: None,
                layout_signature: None,
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

    fn worker_with(status: i32, updated_at: i64) -> SourceInstanceInfo {
        SourceInstanceInfo {
            source_id: "src1".into(),
            worker_id: "w1".into(),
            model_name: "model".into(),
            worker_rank: 0,
            status,
            updated_at,
            accelerator: "cuda".into(),
            source_load: None,
            topology: Default::default(),
            training_step: None,
            layout_signature: None,
        }
    }

    /// Expired UNKNOWN and unrecognized statuses enter the normal STALE cleanup path.
    #[tokio::test]
    async fn test_reap_marks_expired_unknown_and_unrecognized_status_stale() {
        // A worker that dies between PublishMetadata and its first UpdateStatus stays at
        // UNKNOWN with no heartbeat; before this it was never reaped.
        let old_time = chrono::Utc::now().timestamp_millis() - 120_000;
        for status in [SourceStatus::Unknown as i32, 99] {
            let mut mock = MockMetadataBackend::new();
            mock.expect_list_workers()
                .once()
                .returning(move |_, _| Ok(vec![worker_with(status, old_time)]));
            mock.expect_update_status()
                .withf(|sid, wid, rank, status, _, _| {
                    sid == "src1" && wid == "w1" && *rank == 0 && *status == SourceStatus::Stale
                })
                .once()
                .returning(|_, _, _, _, _, _| Ok(()));

            let state = P2pStateManager::with_backend(Arc::new(mock));
            reap_once(&state, 90_000, 3_600_000)
                .await
                .expect("reap_once failed");
        }
    }

    #[tokio::test]
    async fn test_reap_skips_recent_unknown_workers() {
        let recent = chrono::Utc::now().timestamp_millis() - 10_000;
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers()
            .once()
            .returning(move |_, _| Ok(vec![worker_with(SourceStatus::Unknown as i32, recent)]));
        // No update_status or remove_worker calls expected

        let state = P2pStateManager::with_backend(Arc::new(mock));
        reap_once(&state, 90_000, 3_600_000)
            .await
            .expect("reap_once failed");
    }

    #[tokio::test]
    async fn test_reap_does_not_restale_stale_workers_within_gc_timeout() {
        let old_time = chrono::Utc::now().timestamp_millis() - 120_000;
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers()
            .once()
            .returning(move |_, _| Ok(vec![worker_with(SourceStatus::Stale as i32, old_time)]));
        // STALE past the heartbeat timeout but inside the GC timeout: no update, no removal.

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
                accelerator: "cuda".into(),
                source_load: None,
                topology: Default::default(),
                training_step: None,
                layout_signature: None,
            }])
        });
        // No update_status or remove_worker calls expected

        let state = P2pStateManager::with_backend(Arc::new(mock));
        reap_once(&state, 90_000, 3_600_000)
            .await
            .expect("reap_once failed");
    }
}
