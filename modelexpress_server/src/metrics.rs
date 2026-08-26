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
//! A subsystem is added by adding a module under `metrics/` that takes a
//! sub-registry (`registry.sub_registry_with_prefix("p2p")`) and registers its
//! own families, so a module cannot emit a family name outside its segment.
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

pub mod backend;
pub mod buckets;
pub mod cache;
pub mod exposition;
pub mod grpc;
pub mod registry;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::registry::Registry;

use crate::backend_config::BackendConfig;

pub use exposition::serve;

/// Content type for the OpenMetrics text exposition format written by
/// `prometheus_client`. Prometheus negotiates OpenMetrics when offered it and
/// falls back to its own text parser otherwise; that parser treats the trailing
/// `# EOF` line as a comment, so this response is readable either way.
pub const OPENMETRICS_CONTENT_TYPE: &str =
    "application/openmetrics-text; version=1.0.0; charset=utf-8";

/// Namespace prefix applied to every family the server exports.
const NAMESPACE: &str = "mx";

/// Create the server's registry, rooted at the `mx` namespace.
///
/// Populated during startup and then shared read-only with the exposition task,
/// so a scrape only ever reads.
#[must_use]
pub fn new_registry() -> Registry {
    Registry::with_prefix(NAMESPACE)
}

/// Encode the current values in the OpenMetrics text format.
///
/// # Errors
/// Returns [`std::fmt::Error`] if the encoder fails to write into the buffer.
pub fn encode_text(registry: &Registry) -> Result<String, std::fmt::Error> {
    let mut buffer = String::new();
    encode(&mut buffer, registry)?;
    Ok(buffer)
}

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

/// Register `mx_build_info` on `registry` and set it to 1.
///
/// Two jobs. It is the exporter's **proof of life** — registered unconditionally
/// at startup, so a successful scrape shows the endpoint came up even on a
/// server that has served no traffic. And it is the **join target** for the
/// process constants (version, backend, scheme), which are therefore not labels
/// on every other family.
///
/// A `Gauge` set to 1, deliberately not an `Info`: under `prometheus_client`
/// multiprocess mode on the Python side an `Info` writes no file, exposes
/// nothing, and raises nothing, so it would pass its own health check while
/// silently emptying every `group_left` join. Keeping both halves on the same
/// representation means one PromQL expression works against either.
///
/// Nothing is returned — the registry holds its own clone of the `Arc`-backed
/// family, and every label is a process constant, so there is never a later
/// update.
pub fn register_build_info(registry: &mut Registry, backend: &BackendConfig) {
    let family = Family::<BuildInfoLabels, Gauge>::default();
    registry.register(
        "build_info",
        "Build and deployment constants for this process; always 1",
        family.clone(),
    );
    family
        .get_or_create(&BuildInfoLabels {
            component: "server",
            version: env!("CARGO_PKG_VERSION"),
            backend: backend.to_string(),
            scheme: modelexpress_common::envs::metrics_scheme(),
        })
        .set(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_info_is_one_series_in_the_mx_namespace() {
        let mut registry = new_registry();
        register_build_info(
            &mut registry,
            &BackendConfig::Redis {
                url: "redis://localhost:6379".to_string(),
            },
        );

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));

        // Gauge, never an Info -- see register_build_info.
        assert!(
            encoded.contains("# TYPE mx_build_info gauge"),
            "expected an mx-prefixed gauge, got: {encoded}"
        );
        // Exactly one series, set to 1: every label is a process constant, so
        // this family is capped at one series per pod.
        let series: Vec<_> = encoded
            .lines()
            .filter(|line| line.starts_with("mx_build_info{"))
            .collect();
        assert_eq!(series.len(), 1, "expected one series, got: {encoded}");
        assert!(
            series[0].ends_with(" 1"),
            "expected value 1, got: {series:?}"
        );
        assert!(series[0].contains(r#"component="server""#), "{series:?}");
        assert!(series[0].contains(r#"backend="redis""#), "{series:?}");

        assert!(encoded.trim_end().ends_with("# EOF"), "not OpenMetrics");
    }
}
