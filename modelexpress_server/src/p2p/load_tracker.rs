// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory, TTL-decayed estimate of per-source transfer load.
//!
//! The central coordinator observes every source selection: a target that has
//! ranked its candidates and chosen one calls `GetMetadata(mx_source_id,
//! worker_id)` on the server right before pulling weights. This tracker counts
//! those selections per source within a sliding TTL window as a proxy for
//! "transfers currently being served" -- the signal the client `load_aware`
//! selector uses to steer new targets away from busy sources.
//!
//! It is transient in-memory state, not persisted to the metadata backend --
//! intentionally. On a server restart the counts reset to zero, which makes the
//! client `load_aware` policy collapse to `rendezvous_hash` (its tested
//! zero-load path) until the window repopulates from live selections, within
//! one TTL. Persisting would not help: the signal decays within the TTL (60s
//! default), which is comparable to a crash-restart itself, and it would turn
//! an in-memory increment on every `GetMetadata` into a per-selection backend
//! write. Scope is a single central-coordinator server; sharing the estimate
//! across replicated servers (e.g. via Redis) is a follow-up about visibility,
//! not durability. The tensor RDMA path has no source-side completion signal,
//! so a TTL window (rather than an explicit release) is what bounds a
//! selection's contribution to the count.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// TTL-decayed per-source selection counter keyed by `(mx_source_id, worker_id)`.
pub struct SourceLoadTracker {
    ttl: Duration,
    // Recent selection timestamps per source worker, oldest-first. Pruned lazily
    // on every access, so a key's deque only ever holds within-window entries.
    inner: Mutex<HashMap<(String, String), VecDeque<Instant>>>,
}

impl SourceLoadTracker {
    /// Create a tracker whose selections decay after `ttl`.
    pub fn new(ttl: Duration) -> Self {
        Self {
            ttl,
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Record a selection (a successful `GetMetadata`) for this source worker.
    pub fn record_selection(&self, mx_source_id: &str, worker_id: &str) {
        self.record_at(mx_source_id, worker_id, Instant::now());
    }

    /// Current TTL-windowed selection count for this source worker.
    pub fn active_count(&self, mx_source_id: &str, worker_id: &str) -> u32 {
        self.active_count_at(mx_source_id, worker_id, Instant::now())
    }

    fn record_at(&self, mx_source_id: &str, worker_id: &str, now: Instant) {
        let key = (mx_source_id.to_string(), worker_id.to_string());
        let mut map = self.lock();
        let dq = map.entry(key).or_default();
        prune(dq, now, self.ttl);
        dq.push_back(now);
    }

    fn active_count_at(&self, mx_source_id: &str, worker_id: &str, now: Instant) -> u32 {
        let key = (mx_source_id.to_string(), worker_id.to_string());
        let mut map = self.lock();
        let count = match map.get_mut(&key) {
            Some(dq) => {
                prune(dq, now, self.ttl);
                dq.len() as u32
            }
            None => 0,
        };
        // Drop keys that have fully decayed so the map stays bounded by the set
        // of recently active sources rather than every source ever selected.
        if count == 0 {
            map.remove(&key);
        }
        count
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<(String, String), VecDeque<Instant>>> {
        // Best-effort telemetry: recover from a poisoned lock rather than panic
        // (unwrap is forbidden) so a caller panicking mid-update never wedges
        // source selection.
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// Drop timestamps at or beyond the TTL horizon (deque is oldest-first).
fn prune(dq: &mut VecDeque<Instant>, now: Instant, ttl: Duration) {
    while let Some(&front) = dq.front() {
        if now.duration_since(front) >= ttl {
            dq.pop_front();
        } else {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_source_is_zero() {
        let t = SourceLoadTracker::new(Duration::from_secs(60));
        assert_eq!(t.active_count("src", "w0"), 0);
    }

    #[test]
    fn selections_accumulate_within_window() {
        let t = SourceLoadTracker::new(Duration::from_secs(60));
        let now = Instant::now();
        t.record_at("src", "w0", now);
        t.record_at("src", "w0", now);
        t.record_at("src", "w0", now);
        assert_eq!(t.active_count_at("src", "w0", now), 3);
    }

    #[test]
    fn selections_decay_after_ttl() {
        let ttl = Duration::from_secs(60);
        let t = SourceLoadTracker::new(ttl);
        let t0 = Instant::now();
        t.record_at("src", "w0", t0);
        t.record_at("src", "w0", t0 + Duration::from_secs(30));
        // At t0 + 61s the first (t0) selection has aged out, the second has not.
        assert_eq!(
            t.active_count_at("src", "w0", t0 + Duration::from_secs(61)),
            1
        );
        // At t0 + 91s both are gone.
        assert_eq!(
            t.active_count_at("src", "w0", t0 + Duration::from_secs(91)),
            0
        );
    }

    #[test]
    fn keys_are_independent() {
        let t = SourceLoadTracker::new(Duration::from_secs(60));
        let now = Instant::now();
        t.record_at("src", "w0", now);
        t.record_at("src", "w1", now);
        t.record_at("src", "w1", now);
        assert_eq!(t.active_count_at("src", "w0", now), 1);
        assert_eq!(t.active_count_at("src", "w1", now), 2);
        // Same worker_id under a different source is a distinct key.
        assert_eq!(t.active_count_at("other", "w0", now), 0);
    }
}
