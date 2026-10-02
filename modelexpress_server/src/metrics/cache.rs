// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cache and registry gauges, plus the eviction counter.
//!
//! # Why these are refreshed rather than collected at scrape time
//!
//! Counting registry entries means walking the keyspace: the Redis backend does a
//! `SCAN` plus a pipelined per-key fetch, and the Kubernetes backend does an
//! unpaginated `list` of every `ModelCacheEntry`. Registering that as a
//! scrape-time collector would put a keyspace walk on the metadata store every
//! fifteen seconds, and on Kubernetes it would put an unbounded list on the API
//! server -- a cost paid forever for a number that changes slowly.
//!
//! So a background task recomputes them on its own interval and writes plain
//! gauges; the scrape only encodes what is already in memory.
//!
//! # Why the task is not hosted on the cache-eviction interval
//!
//! The obvious home is [`crate::cache::CacheEvictionService`], and it is the wrong
//! one twice over. Its interval defaults to an hour, which is far too coarse for a
//! gauge, and the whole task is skipped when eviction is disabled -- which would
//! make every gauge here permanently *absent* rather than merely stale. An absent
//! gauge and a broken exporter look identical.

use prometheus_client::encoding::{EncodeLabelSet, EncodeLabelValue, LabelValueEncoder};
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::registry::Registry;

use crate::cache::EvictionReason;
use crate::metrics::registry::StatusLabel;

impl EvictionReason {
    /// Wire form of the eviction reason.
    const fn as_metric_str(&self) -> &'static str {
        match self {
            Self::TimeThreshold => "time_threshold",
            Self::CountLimit => "count_limit",
            Self::DiskSpace => "disk_space",
            Self::Manual => "manual",
        }
    }
}

impl EncodeLabelValue for EvictionReason {
    fn encode(&self, encoder: &mut LabelValueEncoder<'_>) -> Result<(), std::fmt::Error> {
        EncodeLabelValue::encode(&self.as_metric_str(), encoder)
    }
}

/// Label set for the eviction counter.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EvictionLabels {
    /// Which rule selected the model.
    pub reason: EvictionReason,
}

/// Label set for the registry-entry gauge.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EntryStatusLabels {
    /// Status of the counted entries.
    pub status: StatusLabel,
}

/// Label set for the background-task heartbeat.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct TaskLabels {
    /// Task name.
    pub task: &'static str,
}

/// Label set for the in-process map gauge.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct MapLabels {
    /// Which map is being reported.
    pub map: &'static str,
}

/// Handles for the cache and registry gauges. Cloning shares the storage.
#[derive(Clone)]
pub struct CacheMetrics {
    evictions: Family<EvictionLabels, Counter>,
    registry_entries: Family<EntryStatusLabels, Gauge>,
    state_entries: Family<MapLabels, Gauge>,
    task_last_success: Family<TaskLabels, Gauge>,
}

impl CacheMetrics {
    /// Register the families and return handles.
    pub fn register(registry: &mut Registry) -> Self {
        let evictions = Family::<EvictionLabels, Counter>::default();
        let registry_entries = Family::<EntryStatusLabels, Gauge>::default();
        let state_entries = Family::<MapLabels, Gauge>::default();
        let task_last_success = Family::<TaskLabels, Gauge>::default();

        let cache = registry.sub_registry_with_prefix("cache");
        cache.register(
            "evictions",
            "Models evicted from the cache, by the rule that selected each one",
            evictions.clone(),
        );

        let reg = registry.sub_registry_with_prefix("registry");
        // Named "registry entries", not "models": one logical model can hold
        // several entries at once -- one per revision, and separate ones for
        // metadata-only downloads.
        reg.register(
            "entries",
            "Registry entries by status, refreshed on the stats interval",
            registry_entries.clone(),
        );

        registry.register(
            "state_entries",
            "Size of never-evicted in-process maps; the OOM early warning",
            state_entries.clone(),
        );
        registry.register(
            "task_last_success_timestamp_seconds",
            "Unix time of each background task's last successful run",
            task_last_success.clone(),
        );

        Self {
            evictions,
            registry_entries,
            state_entries,
            task_last_success,
        }
    }

    /// Count one evicted model against the rule that selected it.
    pub fn record_eviction(&self, reason: EvictionReason) {
        self.evictions
            .get_or_create(&EvictionLabels { reason })
            .inc();
    }

    /// Publish the registry entry counts observed by a refresh pass.
    pub fn set_registry_entries(&self, downloading: i64, downloaded: i64, errored: i64) {
        for (status, count) in [
            (StatusLabel::Downloading, downloading),
            (StatusLabel::Downloaded, downloaded),
            (StatusLabel::Error, errored),
        ] {
            self.registry_entries
                .get_or_create(&EntryStatusLabels { status })
                .set(count);
        }
    }

    /// Publish the size of an in-process map.
    pub fn set_state_entries(&self, map: &'static str, count: i64) {
        self.state_entries
            .get_or_create(&MapLabels { map })
            .set(count);
    }

    /// Stamp a task's last successful run.
    ///
    /// Only on success. A task that starts failing leaves its previous timestamp
    /// in place and the value goes stale, which is what makes
    /// `time() - mx_task_last_success_timestamp_seconds` a usable liveness alert;
    /// stamping unconditionally would report a wedged task as healthy.
    pub fn stamp_task_success(&self, task: &'static str, unix_seconds: i64) {
        self.task_last_success
            .get_or_create(&TaskLabels { task })
            .set(unix_seconds);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{encode_text, new_registry};

    #[test]
    fn the_count_limit_path_is_not_reported_as_a_time_threshold() {
        let mut registry = new_registry();
        let metrics = CacheMetrics::register(&mut registry);

        metrics.record_eviction(EvictionReason::TimeThreshold);
        metrics.record_eviction(EvictionReason::CountLimit);
        metrics.record_eviction(EvictionReason::CountLimit);

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(r#"mx_cache_evictions_total{reason="time_threshold"} 1"#),
            "{encoded}"
        );
        assert!(
            encoded.contains(r#"mx_cache_evictions_total{reason="count_limit"} 2"#),
            "{encoded}"
        );
        assert!(!encoded.contains("_total_total"), "{encoded}");
    }

    /// Every `EvictionReason` needs a wire form, not just the two the policy can
    /// produce today: `Manual` is a live variant and `DiskSpace` is declared, and
    /// a typo in either arm ships a label value no dashboard query matches.
    ///
    /// One increment per reason, so a collision between two arms shows up as a
    /// series holding 2 and the expected one missing.
    #[test]
    fn every_eviction_reason_has_its_own_label_value() {
        let mut registry = new_registry();
        let metrics = CacheMetrics::register(&mut registry);

        for (reason, expected) in [
            (EvictionReason::TimeThreshold, "time_threshold"),
            (EvictionReason::CountLimit, "count_limit"),
            (EvictionReason::DiskSpace, "disk_space"),
            (EvictionReason::Manual, "manual"),
        ] {
            metrics.record_eviction(reason);

            let encoded =
                encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
            assert!(
                encoded.contains(&format!(
                    r#"mx_cache_evictions_total{{reason="{expected}"}} 1"#
                )),
                "{reason:?} did not encode as {expected}: {encoded}"
            );
        }
    }

    /// The three status gauges must not be cross-wired: a rotated argument list
    /// publishes the error count under `status="downloading"`, which reads as a
    /// plausible level and is what a wedged-download alert fires on. Distinct
    /// values per status, so no rotation survives.
    ///
    /// Scope: the neighbouring invariant -- that a *failed* refresh leaves these
    /// gauges at their previous values instead of zeroing them, since a zeroed
    /// gauge and an empty registry are indistinguishable -- is a property of
    /// `registry::stats_refresh::refresh_once`, not of this setter, and is not
    /// checked here.
    #[test]
    fn registry_entry_gauges_are_set_per_status() {
        let mut registry = new_registry();
        let metrics = CacheMetrics::register(&mut registry);

        metrics.set_registry_entries(2, 7, 1);
        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(r#"mx_registry_entries{status="downloading"} 2"#),
            "{encoded}"
        );
        assert!(
            encoded.contains(r#"mx_registry_entries{status="downloaded"} 7"#),
            "{encoded}"
        );
        assert!(
            encoded.contains(r#"mx_registry_entries{status="error"} 1"#),
            "{encoded}"
        );
    }

    /// Stamped with the constant the production task actually publishes, not a
    /// local copy of the string, so this exercises the same label value the
    /// refresh task emits.
    ///
    /// That alone would go green on a rename, and a rename silently breaks every
    /// dashboard and alert keyed on the old value -- hence the separate equality
    /// on `TASK_NAME`, which pins the wire form as a contract.
    #[test]
    fn the_task_heartbeat_is_stamped_per_task() {
        use crate::registry::stats_refresh::TASK_NAME;

        let mut registry = new_registry();
        let metrics = CacheMetrics::register(&mut registry);

        assert_eq!(TASK_NAME, "registry_stats_refresh");

        metrics.stamp_task_success(TASK_NAME, 1_760_000_000);
        metrics.set_state_entries("download_waiters", 3);

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(&format!(
                r#"mx_task_last_success_timestamp_seconds{{task="{TASK_NAME}"}} 1760000000"#
            )),
            "{encoded}"
        );
        assert!(
            encoded.contains(r#"mx_state_entries{map="download_waiters"} 3"#),
            "{encoded}"
        );
    }
}
