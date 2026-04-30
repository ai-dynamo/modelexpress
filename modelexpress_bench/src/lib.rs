// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared building blocks for ModelExpress benchmark binaries.
//!
//! The harness is intentionally separate from production crates so that
//! benchmark-only paths (synthetic byte sources, in-memory sinks, latency
//! capture, JSON-line result emission) cannot leak into release artifacts.

use std::time::Duration;

pub mod bench_service;
pub mod model_name;
pub mod validation;

/// Histogram-style summary of a latency sample set.
///
/// Computed by sorting the input slice once and indexing for percentile
/// positions. Cheap enough for benchmark-scale sample counts (millions).
#[derive(Debug, Clone, serde::Serialize)]
pub struct LatencySummary {
    pub samples: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub mean_ns: u64,
    pub p50_ns: u64,
    pub p90_ns: u64,
    pub p99_ns: u64,
    pub p999_ns: u64,
}

impl LatencySummary {
    /// Build a summary from a mutable slice of durations. The slice is sorted
    /// in place; callers should not rely on the original order afterwards.
    #[allow(clippy::arithmetic_side_effects, clippy::cast_precision_loss)]
    pub fn from_samples(samples: &mut [Duration]) -> Self {
        if samples.is_empty() {
            return Self {
                samples: 0,
                min_ns: 0,
                max_ns: 0,
                mean_ns: 0,
                p50_ns: 0,
                p90_ns: 0,
                p99_ns: 0,
                p999_ns: 0,
            };
        }
        samples.sort_unstable();
        let len = samples.len();
        let last_idx = len.saturating_sub(1);
        let total_ns: u128 = samples.iter().map(Duration::as_nanos).sum();
        let mean_ns = u64::try_from(total_ns / u128::from(len as u64)).unwrap_or(u64::MAX);
        let percentile = |p: f64| -> u64 {
            // Clamp the index into the slice bounds. Float math is intentionally
            // approximate; benchmark percentile placement does not need bit-exact.
            let idx_f = (p * last_idx as f64).round();
            let idx = if idx_f.is_finite() && idx_f >= 0.0 {
                (idx_f as usize).min(last_idx)
            } else {
                0
            };
            u64::try_from(samples[idx].as_nanos()).unwrap_or(u64::MAX)
        };
        let min_ns = u64::try_from(samples[0].as_nanos()).unwrap_or(u64::MAX);
        let max_ns = u64::try_from(samples[last_idx].as_nanos()).unwrap_or(u64::MAX);
        Self {
            samples: len as u64,
            min_ns,
            max_ns,
            mean_ns,
            p50_ns: percentile(0.5),
            p90_ns: percentile(0.9),
            p99_ns: percentile(0.99),
            p999_ns: percentile(0.999),
        }
    }
}

/// Top-level result emitted as a single JSON object after a benchmark run.
///
/// Designed to stack: each invocation prints one JSON object on stdout, so a
/// sweep of chunk sizes can be concatenated into a JSON-lines file and
/// post-processed with `jq` or pandas.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunResult {
    pub label: String,
    pub server_addr: String,
    pub total_bytes: u64,
    pub chunk_size: u32,
    pub strict_validation: bool,
    pub mpsc_cap: u32,
    pub warmup_bytes: u64,
    pub elapsed_ns: u64,
    pub bytes_per_sec: f64,
    pub gibibits_per_sec: f64,
    pub chunks_received: u64,
    pub chunks_per_sec: f64,
    pub ttfb_ns: u64,
    pub chunk_recv_latency: LatencySummary,
}

/// Deterministic byte-fill for the synthetic source buffer. Uses a tiny
/// xorshift64 so producing a 16 MB pattern stays well under a millisecond.
///
/// The exact pattern does not matter for throughput; what matters is that
/// the buffer is not all-zeros so any pathological compression in the
/// transport layer cannot inflate the measured rate.
#[allow(clippy::arithmetic_side_effects)]
pub fn fill_pattern(buf: &mut [u8], seed: u64) {
    let mut state = seed | 1;
    for chunk in buf.chunks_mut(8) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bytes = state.to_le_bytes();
        for (dst, src) in chunk.iter_mut().zip(bytes.iter()) {
            *dst = *src;
        }
    }
}
