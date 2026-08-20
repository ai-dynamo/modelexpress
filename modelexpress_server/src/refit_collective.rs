// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rendezvous and admission for the NCCL M2N collective refit path.
//!
//! Kept separate from [`crate::refit`] on purpose: that module is the directory
//! for the NIXL pull path, and the two share no types. See
//! `docs/NCCL_M2N_REFIT.md`.

pub mod backend;
pub mod lanes;
pub mod service;
