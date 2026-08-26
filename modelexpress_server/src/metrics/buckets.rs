// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared histogram bucket boundaries.
//!
//! Latency families draw their boundaries from a shared constant rather than
//! writing literals, so two families measuring the same class of work stay
//! comparable and one recording rule covers both. Boundaries are written out as
//! literals rather than generated: a generator emits values like
//! `0.00193069772888325`, which makes the `le` label churn between releases and
//! breaks every dashboard pinned to a boundary.
//!
//! [`Histogram::new`](prometheus_client::metrics::histogram::Histogram::new)
//! appends the `+Inf` bucket itself, so these arrays must not carry one.
//!
//! Bands land with the families that need them rather than sitting here unused.

/// 1 ms to 10 s: gRPC handlers and metadata-backend operations.
///
/// The upper bound is deliberately well past any healthy value for this class of
/// work. Anything slower is already an outage, and the `+Inf` bucket is enough to
/// see it.
pub const FAST: [f64; 13] = [
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// 5 s to 1 hour: whole-model download and load.
///
/// The top boundary is deliberately an hour. A cold DeepSeek-V3 warm-up runs for
/// roughly forty minutes, so a band topping out at the transfer scale would put
/// every cold load in `+Inf` and make the one case worth measuring unmeasurable.
pub const XSLOW: [f64; 12] = [
    5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1200.0, 1800.0, 2700.0, 3600.0,
];
