// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distributed model registry: model-download lifecycle (`DOWNLOADING` / `DOWNLOADED` /
//! `ERROR`) and LRU cache-eviction timestamps.
//!
//! `backend` owns the `RegistryBackend` trait plus its Redis, Kubernetes CRD, and
//! (behind the `memory-backend` feature) in-memory implementations. `state` wraps the
//! backend in a lazy-connect manager used by `ModelDownloadTracker` and
//! `CacheEvictionService`. `entry_key` defines the identity every backend keys on.

pub mod backend;
pub mod entry_key;
pub mod k8s_types;
pub mod state;
pub mod stats_refresh;
