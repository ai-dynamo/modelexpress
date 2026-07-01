// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P2P model metadata: worker coordination, tensor descriptors, and NIXL RDMA transfer.
//!
//! The backend abstraction (`backend`) persists worker metadata to Redis or Kubernetes CRDs;
//! `state` wraps it with a lazy-connect manager; `service` exposes the gRPC surface; `reaper`
//! garbage-collects stale workers; `source_identity` derives deterministic source IDs;
//! `load_tracker` estimates per-source transfer load for load-aware client selection.

pub mod backend;
pub mod k8s_types;
pub mod load_tracker;
pub mod reaper;
pub mod service;
pub mod source_identity;
pub mod state;
