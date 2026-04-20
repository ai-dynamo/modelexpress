// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Peer discovery substrate probes for ModelExpress.
//!
//! This crate provides substrate-agnostic peer discovery mechanisms used
//! to bootstrap higher-level peer networks. The companion Python package
//! (`mx_peer_discovery_py`) implements the same substrates with
//! wire-format compatibility.
//!
//! Modules:
//!
//! - [`dns`]: hostname resolution via tokio async DNS.
//! - [`mdns`]: multicast DNS service discovery (RFC 6762/6763),
//!   wire-compatible with the Python half. Default service name
//!   `_mx-peer._tcp.local`.
//! - [`slurm`]: SLURM compact hostlist expansion.
//! - [`static_peers`]: explicit `host:port` peer lists from strings or env.

pub mod dns;
pub mod mdns;
pub mod slurm;
pub mod static_peers;
