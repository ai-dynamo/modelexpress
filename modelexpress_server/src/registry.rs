// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distributed model registry: model-download lifecycle (`DOWNLOADING` / `DOWNLOADED` /
//! `ERROR`) and LRU cache-eviction timestamps.
//!
//! `backend` owns the `RegistryBackend` trait plus Redis and Kubernetes CRD
//! implementations. `state` wraps the backend in a lazy-connect manager used by
//! `ModelDownloadTracker` and `CacheEvictionService`.

pub mod backend;
pub mod k8s_types;
pub mod state;
