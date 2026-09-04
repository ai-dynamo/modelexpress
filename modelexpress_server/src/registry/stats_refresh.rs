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

    /// The negative assertion above only carries information if the positive
    /// case is pinned too: "correctly not stamped because the pass failed" and
    /// "never stamped under any circumstances" satisfy it identically.
    ///
    /// The three counts are deliberately distinct so a rotated or mis-ordered
    /// tuple lands a real-looking but wrong number under each status label
    /// rather than a value that happens to match.
    #[tokio::test]
    async fn a_successful_pass_publishes_each_count_under_its_own_status() {
        let mut prom = new_registry();
        let metrics = CacheMetrics::register(&mut prom);
        let mut backend = crate::registry::backend::MockRegistryBackend::new();
        backend
            .expect_get_status_counts()
            .times(1)
            .returning(|| Ok((2, 7, 1)));
        let registry = RegistryManager::with_backend(Arc::new(backend));
        let waiters: WaiterCount = Arc::new(|| 4);

        refresh_once(&registry, &metrics, &waiters).await;

        let encoded = encode_text(&prom).unwrap_or_else(|_| String::from("<encode failed>"));
        for (status, count) in [("downloading", 2), ("downloaded", 7), ("error", 1)] {
            let expected = format!(r#"mx_registry_entries{{status="{status}"}} {count}"#);
            assert!(encoded.contains(&expected), "missing {expected}: {encoded}");
        }
        // Against `TASK_NAME` rather than a copy of the string, so renaming the
        // constant cannot leave every dashboard keyed on the old label while
        // this stays green.
        assert!(
            encoded.contains(&format!(
                r#"mx_task_last_success_timestamp_seconds{{task="{TASK_NAME}"}}"#
            )),
            "a successful pass did not stamp the heartbeat: {encoded}"
        );
    }

    /// Gauges hold their previous values across a failed pass.
    ///
    /// Zeroing them instead makes an unreachable backend and a genuinely empty
    /// registry produce identical output, so on-call reads
    /// `mx_registry_entries{status="downloading"} 0` during an outage and
    /// concludes nothing is downloading.
    #[tokio::test]
    async fn a_failed_pass_leaves_the_registry_gauges_at_their_last_good_values() {
        let mut prom = new_registry();
        let metrics = CacheMetrics::register(&mut prom);
        let mut backend = crate::registry::backend::MockRegistryBackend::new();
        // Succeed once, then fail: the second pass is the one under test.
        let pass = std::sync::atomic::AtomicUsize::new(0);
        backend
            .expect_get_status_counts()
            .times(2)
            .returning(move || {
                if pass.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                    Ok((3, 9, 2))
                } else {
                    Err("registry unreachable".into())
                }
            });
        let registry = RegistryManager::with_backend(Arc::new(backend));
        let waiters: WaiterCount = Arc::new(|| 4);

        refresh_once(&registry, &metrics, &waiters).await;
        refresh_once(&registry, &metrics, &waiters).await;

        let encoded = encode_text(&prom).unwrap_or_else(|_| String::from("<encode failed>"));
        for (status, count) in [("downloading", 3), ("downloaded", 9), ("error", 2)] {
            let expected = format!(r#"mx_registry_entries{{status="{status}"}} {count}"#);
            assert!(
                encoded.contains(&expected),
                "the failed pass clobbered {status}, expected {expected}: {encoded}"
            );
        }
    }

    /// `run_server` awaits this task's join handle during shutdown, so a loop
    /// that stops observing its oneshot does not just leak a task -- it hangs
    /// the whole shutdown sequence until the kubelet SIGKILLs the pod.
    ///
    /// The `timeout` is what turns the regression into a failure instead of a
    /// hung suite; it costs nothing on the passing path, where the task is woken
    /// by the oneshot and joins immediately, and only elapses when the loop has
    /// actually stopped watching for shutdown.
    #[tokio::test]
    async fn the_refresh_loop_stops_when_shutdown_fires() {
        let mut prom = new_registry();
        let metrics = CacheMetrics::register(&mut prom);
        let mut backend = crate::registry::backend::MockRegistryBackend::new();
        // No `times`: the immediate first tick fires one pass, and a loop that
        // ignored shutdown would fire more.
        backend
            .expect_get_status_counts()
            .returning(|| Ok((0, 0, 0)));
        let registry = Arc::new(RegistryManager::with_backend(Arc::new(backend)));
        let waiters: WaiterCount = Arc::new(|| 0);
        let (tx, rx) = oneshot::channel();

        let task = tokio::spawn(run_stats_refresh(registry, metrics, waiters, rx));
        assert!(
            tx.send(()).is_ok(),
            "the task dropped its shutdown receiver"
        );

        let stopped = tokio::time::timeout(std::time::Duration::from_secs(2), task).await;
        assert!(
            stopped.is_ok(),
            "the refresh loop ignored shutdown and would hang run_server"
        );
    }
}
