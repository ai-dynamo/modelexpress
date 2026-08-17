// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `mx_build_info` — the process constants, and the exporter's proof of life.
//!
//! Two jobs:
//!
//! 1. **Proof of life.** It is registered unconditionally at startup, so a
//!    successful scrape proves the exporter came up even on a server that has
//!    served no traffic. This is the exit criterion for the first rollout phase:
//!    `up == 1` on a pod that has done nothing yet.
//! 2. **Join target for process constants.** Version, metadata backend and the
//!    benchmark scheme are process-constant, so they do not belong on every
//!    family as labels; they live here once and are joined in PromQL with
//!    `group_left`.
//!
//! It is a `Gauge` set to `1`, deliberately **not** an `Info`. That choice is
//! forced by the Python client, where under `prometheus_client` multiprocess
//! mode an `Info` writes no file, exposes nothing, and raises nothing — it would
//! pass its own health check while being invisible and silently emptying every
//! `group_left` join. Keeping both sides on the same representation means the
//! same PromQL works against the server and the client.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;

use crate::backend_config::BackendConfig;

use super::MetricsRegistry;

/// Process constants carried by `mx_build_info`.
///
/// Every field is fixed for the lifetime of the process, so this family has
/// exactly one series per pod.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct BuildInfoLabels {
    /// Which half of ModelExpress is reporting: `server` here, `client` from the
    /// Python collector. Component identity is carried by this label rather than
    /// by a family-name suffix, so family names stay globally unique.
    pub component: &'static str,
    /// Crate version, from `CARGO_PKG_VERSION` at compile time.
    pub version: &'static str,
    /// Metadata backend in use: `redis`, `kubernetes`, or `memory`.
    pub backend: String,
    /// Benchmark run label from `MX_METRICS_SCHEME`; empty when unset.
    pub scheme: String,
}

/// Handle to the registered `mx_build_info` family.
#[derive(Clone, Debug)]
pub struct BuildInfo {
    family: Family<BuildInfoLabels, Gauge>,
}

impl BuildInfo {
    /// Register `mx_build_info` on `registry` and set it to 1.
    ///
    /// Called during startup before the listener binds, so the very first scrape
    /// already sees it.
    pub fn register(registry: &mut MetricsRegistry, backend: &BackendConfig) -> Self {
        let family = Family::<BuildInfoLabels, Gauge>::default();
        registry.root().register(
            "build_info",
            "Build and deployment constants for this process; always 1",
            family.clone(),
        );
        let build_info = Self { family };
        build_info.set(backend);
        build_info
    }

    /// Set the single series to 1 for the current process constants.
    fn set(&self, backend: &BackendConfig) {
        self.family
            .get_or_create(&BuildInfoLabels {
                component: "server",
                version: env!("CARGO_PKG_VERSION"),
                backend: backend.to_string(),
                scheme: modelexpress_common::envs::metrics_scheme(),
            })
            .set(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_info_is_one_and_carries_the_backend() {
        let mut registry = MetricsRegistry::new();
        let _build_info = BuildInfo::register(
            &mut registry,
            &BackendConfig::Redis {
                url: "redis://localhost:6379".to_string(),
            },
        );

        let encoded = registry
            .encode_text()
            .unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(r#"backend="redis""#),
            "expected the backend label, got: {encoded}"
        );
        assert!(
            encoded.contains(r#"component="server""#),
            "expected the component label, got: {encoded}"
        );
        // Gauge, set to 1 — never an Info. See the module docs.
        assert!(
            encoded.contains("# TYPE mx_build_info gauge"),
            "mx_build_info must be a gauge, got: {encoded}"
        );
        assert!(
            encoded
                .lines()
                .any(|line| { line.starts_with("mx_build_info{") && line.ends_with(" 1") }),
            "expected the series to be set to 1, got: {encoded}"
        );
    }

    #[test]
    fn build_info_has_exactly_one_series() {
        let mut registry = MetricsRegistry::new();
        let build_info = BuildInfo::register(
            &mut registry,
            &BackendConfig::Kubernetes {
                namespace: "mx".to_string(),
            },
        );
        // Re-setting must not mint a second series: every label is a process
        // constant, so this family is capped at one series per pod.
        build_info.set(&BackendConfig::Kubernetes {
            namespace: "mx".to_string(),
        });

        let encoded = registry
            .encode_text()
            .unwrap_or_else(|_| String::from("<encode failed>"));
        let series = encoded
            .lines()
            .filter(|line| line.starts_with("mx_build_info{"))
            .count();
        assert_eq!(series, 1, "expected one series, got: {encoded}");
    }
}
