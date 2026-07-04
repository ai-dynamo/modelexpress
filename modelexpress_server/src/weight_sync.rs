// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WeightSync module: server-side plan building for trainer-inference sync.

pub mod router;
pub mod service;

pub use service::WeightSyncServiceImpl;
