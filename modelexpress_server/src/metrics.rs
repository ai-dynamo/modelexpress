// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus exposition for the ModelExpress server.
//!
//! Before this module the server had no metrics surface at all, and the Helm
//! chart's `prometheus.io/port` annotation pointed at the tonic gRPC listener.
//! tonic speaks HTTP/2 only, so Prometheus's HTTP/1.1 `GET /metrics` could never
//! succeed and every server pod reported `up == 0` permanently — worse than no
//! annotation, because a permanently-down target is indistinguishable from a
//! crashed pod. This module provides the listener that annotation should have
//! been pointing at; [`crate::server::run_server`] starts it and the chart is
//! repointed in the same change.
//!
//! # Structure
//!
//! [`MetricsRegistry`] owns a `prometheus_client` [`Registry`] rooted at the `mx`
//! prefix. A subsystem is added by adding a module under `metrics/` that takes a
//! sub-registry from [`MetricsRegistry::subsystem`] and registers its own
//! families; because the sub-registry carries the subsystem's prefix, a module
//! physically cannot emit a family name outside its own segment.
//!
//! # Constraints this module is written against
//!
//! - The workspace denies `clippy::unwrap_used` and `clippy::expect_used`, which
//!   rules out the `lazy_static! { register_int_counter!(..).unwrap() }` idiom of
//!   the tikv `prometheus` crate. `prometheus_client`'s registration and
//!   `get_or_create` are infallible, so nothing here needs to panic.
//! - The workspace denies `clippy::mod_module_files`, so this is `metrics.rs`
//!   plus a `metrics/` directory, never `metrics/mod.rs`.
//! - Collection must never be scrape-time-expensive. Anything derived from a
//!   Redis `SCAN` or a similar keyspace walk belongs in a refresh task that
//!   writes a plain gauge; the scrape only encodes what is already in memory.

pub mod build_info;
pub mod exposition;

use prometheus_client::encoding::text::encode;
use prometheus_client::registry::Registry;

pub use build_info::BuildInfo;
pub use exposition::serve;

/// Content type for the OpenMetrics text exposition format written by
/// `prometheus_client`. Prometheus negotiates OpenMetrics when offered it and
/// falls back to its own text parser otherwise; that parser treats the trailing
/// `# EOF` line as a comment, so this response is readable either way.
pub const OPENMETRICS_CONTENT_TYPE: &str =
    "application/openmetrics-text; version=1.0.0; charset=utf-8";

/// Namespace prefix applied to every family the server exports.
const NAMESPACE: &str = "mx";

/// The server's metric registry.
///
/// Built once during startup, then shared read-only with the exposition task —
/// registration happens before the listener starts, so a scrape only ever reads.
#[derive(Debug)]
pub struct MetricsRegistry {
    registry: Registry,
}

impl MetricsRegistry {
    /// Create an empty registry rooted at the `mx` namespace.
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: Registry::with_prefix(NAMESPACE),
        }
    }

    /// Take a sub-registry scoped to `subsystem`, so families registered through
    /// it are named `mx_<subsystem>_<family>`.
    ///
    /// This is the extension point: a new subsystem is a new module under
    /// `metrics/` that registers into its own sub-registry and cannot name a
    /// family outside its segment.
    pub fn subsystem(&mut self, subsystem: &str) -> &mut Registry {
        self.registry.sub_registry_with_prefix(subsystem)
    }

    /// Mutable access to the root registry, for families that are deliberately
    /// namespace-level rather than subsystem-level (`mx_build_info`).
    pub fn root(&mut self) -> &mut Registry {
        &mut self.registry
    }

    /// Encode the current values in the OpenMetrics text format.
    ///
    /// # Errors
    /// Returns [`std::fmt::Error`] if the encoder fails to write into the buffer.
    pub fn encode_text(&self) -> Result<String, std::fmt::Error> {
        let mut buffer = String::new();
        encode(&mut buffer, &self.registry)?;
        Ok(buffer)
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_config::BackendConfig;

    #[test]
    fn registry_is_rooted_at_the_mx_namespace() {
        let mut registry = MetricsRegistry::new();
        BuildInfo::register(
            &mut registry,
            &BackendConfig::Kubernetes {
                namespace: "default".to_string(),
            },
        );
        let encoded = registry
            .encode_text()
            .unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains("mx_build_info"),
            "expected the mx-prefixed family, got: {encoded}"
        );
    }

    #[test]
    fn subsystem_registries_carry_their_segment() {
        let mut registry = MetricsRegistry::new();
        let sub = registry.subsystem("p2p");
        let workers = prometheus_client::metrics::gauge::Gauge::<i64>::default();
        sub.register("workers", "Live P2P source workers", workers.clone());
        workers.set(3);

        let encoded = registry
            .encode_text()
            .unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains("mx_p2p_workers"),
            "expected the subsystem prefix to be applied, got: {encoded}"
        );
    }

    #[test]
    fn encoded_output_is_parseable_openmetrics() {
        let mut registry = MetricsRegistry::new();
        BuildInfo::register(
            &mut registry,
            &BackendConfig::Redis {
                url: "redis://localhost:6379".to_string(),
            },
        );
        let encoded = registry
            .encode_text()
            .unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(encoded.contains("# TYPE mx_build_info gauge"));
        assert!(encoded.trim_end().ends_with("# EOF"));
    }
}
