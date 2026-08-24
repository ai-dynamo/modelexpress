// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Background refresh of the registry-statistics gauges.
//!
//! Counting registry entries walks the keyspace, so it cannot be done at scrape
//! time -- see [`crate::metrics::cache`] for the cost. This task recomputes the
//! numbers on its own interval and writes plain gauges; the scrape only encodes.
//!
//! It is deliberately **not** hosted on the cache-eviction service. That service
//! ticks hourly by default and is skipped entirely when eviction is disabled,
//! which would leave these gauges permanently absent -- indistinguishable from a
//! crashed exporter. Running independently also means the statistics survive a
//! deployment that never evicts anything.
//!
//! Safe on every replica: it only reads.

use std::sync::Arc;

use tokio::sync::oneshot;
use tracing::{debug, info, warn};

use crate::metrics::cache::CacheMetrics;
use crate::registry::state::RegistryManager;

/// Task name reported by `mx_task_last_success_timestamp_seconds`.
pub const TASK_NAME: &str = "registry_stats_refresh";

/// Run the refresh loop until the shutdown signal fires.
pub async fn run_stats_refresh(
    registry: Arc<RegistryManager>,
    metrics: CacheMetrics,
    waiters: WaiterCount,
    shutdown: oneshot::Receiver<()>,
) {
    let interval_secs = modelexpress_common::envs::registry_stats_interval_secs();
    info!("Registry stats refresh started (interval={interval_secs}s)");

    let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            _ = interval.tick() => {
                refresh_once(&registry, &metrics, &waiters).await;
            }
            _ = &mut shutdown => {
                info!("Registry stats refresh received shutdown signal");
                break;
            }
        }
    }
}

/// Reads the current size of the in-process waiter map.
///
/// Boxed rather than taking the tracker directly so this task does not depend on
/// the whole download path just to read one length.
pub type WaiterCount = Arc<dyn Fn() -> usize + Send + Sync>;

/// One refresh pass.
///
/// On failure the gauges are left holding their previous values and the task
/// heartbeat is not stamped. Zeroing them instead would be worse than useless:
/// an empty registry and an unreachable one would look identical, and the
/// staleness of the heartbeat is the signal that says which.
async fn refresh_once(registry: &RegistryManager, metrics: &CacheMetrics, waiters: &WaiterCount) {
    // In-process, so it is refreshed even when the backend is unreachable.
    metrics.set_state_entries(
        "download_waiters",
        i64::try_from(waiters()).unwrap_or(i64::MAX),
    );

    match registry.get_status_counts().await {
        Ok((downloading, downloaded, errored)) => {
            metrics.set_registry_entries(
                i64::from(downloading),
                i64::from(downloaded),
                i64::from(errored),
            );
            metrics.stamp_task_success(TASK_NAME, chrono::Utc::now().timestamp());
            debug!(
                "Registry stats refreshed: downloading={downloading} downloaded={downloaded} error={errored}"
            );
        }
        Err(e) => {
            // Not an error log: a backend blip is expected and the staleness of
            // the heartbeat gauge is the alertable signal, not this line.
            warn!("Registry stats refresh failed, keeping previous values: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{encode_text, new_registry};

    /// The waiter gauge is in-process, so it must still be published on a pass
    /// where the backend lookup fails.
    #[tokio::test]
    async fn a_failed_pass_still_publishes_the_in_process_gauge() {
        let mut prom = new_registry();
        let metrics = CacheMetrics::register(&mut prom);
        // A mock that fails, rather than a real unreachable address: dialling a
        // dead port waits out the connect timeout and made this test take
        // minutes.
        let mut backend = crate::registry::backend::MockRegistryBackend::new();
        backend
            .expect_get_status_counts()
            .times(1)
            .returning(|| Err("registry unreachable".into()));
        let registry = RegistryManager::with_backend(Arc::new(backend));
        let waiters: WaiterCount = Arc::new(|| 4);

        refresh_once(&registry, &metrics, &waiters).await;

        let encoded = encode_text(&prom).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(r#"mx_state_entries{map="download_waiters"} 4"#),
            "{encoded}"
        );
        // The heartbeat must NOT be stamped: the pass did not succeed. Named
        // explicitly rather than matching the bare `task="` prefix -- both
        // catch it, but this one cannot be misread as allowing the real task
        // through.
        assert!(
            !encoded.contains(&format!(
                r#"mx_task_last_success_timestamp_seconds{{task="{TASK_NAME}""#
            )),
            "a failed pass stamped the heartbeat: {encoded}"
        );
    }
}
