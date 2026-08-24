// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::expect_used)]

//! Static checks over the alerting rules the Helm chart ships.
//!
//! An alert naming a family the server does not export fails silently in the
//! worst possible way: Prometheus parses the expression, evaluates it to an
//! empty vector, and the alert simply never fires. Nothing logs, nothing turns
//! red, and the page that should have woken someone never arrives. Renaming a
//! metric therefore disarms its own alert, and the only symptom is an incident
//! nobody was told about.
//!
//! These operate on the template source rather than on `helm template` output so
//! they run under plain `cargo test` without a helm binary, matching
//! `helm_namespace_scope.rs`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use modelexpress_server::backend_config::BackendConfig;
use modelexpress_server::metrics;
use modelexpress_server::metrics::backend::Store;
use modelexpress_server::metrics::registry::{ClaimResult, LeaseResult, StatusLabel};
use tower::{Layer, ServiceExt};

/// Families exported by the Python client, which cannot be enumerated from Rust.
///
/// `modelexpress_client/python/tests/test_metrics.py::test_alert_rule_client_families_exist`
/// asserts this same list against the client's real registry. The two halves
/// cross-check: a rename on the Python side fails there, and a name added here
/// that Python does not export fails there too. Keep them in step.
const CLIENT_FAMILIES: &[&str] = &[
    "mx_nixl_data_plane_errors_total",
    "mx_nixl_receive_total",
    "mx_p2p_list_sources_total",
];

fn rules_template() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace-tests has a parent directory")
        .join("helm/templates/prometheusrule.yaml")
}

/// Family name to OpenMetrics type, read from the `# TYPE` lines of a registry
/// with every server family registered *and touched*.
///
/// The touching is not incidental. `prometheus_client` emits nothing at all for
/// a `Family` with no children -- not even a `# TYPE` line -- so a registry that
/// is merely registered encodes to just `mx_build_info` and every assertion
/// below would pass against a one-entry map. Each family therefore gets one
/// sample through the same public API the server uses.
///
/// This is the registration sequence `server::run_server` performs, so these are
/// the names a real scrape returns rather than a second list that could drift.
async fn registered_families() -> BTreeMap<String, String> {
    let mut registry = metrics::new_registry();
    let grpc = metrics::grpc::GrpcMetrics::register(&mut registry);
    let backend = metrics::backend::BackendMetrics::register(&mut registry);
    let reg = metrics::registry::RegistryMetrics::register(&mut registry);
    let download = metrics::registry::DownloadMetrics::register(&mut registry);
    let cache = metrics::cache::CacheMetrics::register(&mut registry);
    metrics::register_build_info(
        &mut registry,
        &BackendConfig::Redis {
            url: String::from("redis://127.0.0.1:6379"),
        },
    );

    // GrpcMetrics::record is private, so the families are reached the way
    // production reaches them: through the tower layer. A non-streaming path is
    // required -- `resolves_at_head` suppresses recording for the streaming
    // three, and using one of those here would leave all three gRPC families
    // empty and silently absent from the map.
    let inner = tower::service_fn(|_req: http::Request<()>| async {
        Ok::<_, std::convert::Infallible>(http::Response::new(()))
    });
    let request = http::Request::builder()
        .uri("/model_express.health.HealthService/GetHealth")
        .body(())
        .expect("request builds");
    let _ = metrics::grpc::GrpcMetricsLayer::new(grpc)
        .layer(inner)
        .oneshot(request)
        .await;

    // BackendMetrics only exposes the timing wrapper, which is what populates
    // all three of its families.
    let _: Result<(), ()> = backend
        .time(Store::Registry, "get_status", async { Ok(()) })
        .await;

    reg.record_claim(ClaimResult::Claimed);
    reg.record_lease_refresh(LeaseResult::Renewed);
    reg.record_transition(StatusLabel::Absent, StatusLabel::Downloading);
    download.observe(StatusLabel::Downloaded, 1.0);
    cache.record_eviction(modelexpress_server::cache::EvictionReason::TimeThreshold);
    cache.set_registry_entries(1, 1, 0);
    cache.set_state_entries("download_waiters", 0);
    cache.stamp_task_success("registry_stats_refresh", 0);

    let encoded = metrics::encode_text(&registry).expect("the registry encodes");
    encoded
        .lines()
        .filter_map(|line| {
            let mut parts = line.strip_prefix("# TYPE ")?.split_whitespace();
            Some((parts.next()?.to_string(), parts.next()?.to_string()))
        })
        .collect()
}

/// Every `mx_*` token in the file, from expressions and prose alike.
///
/// Annotation text is scanned deliberately: a description telling on-call to
/// query a family that no longer exists sends them down a dead end at exactly
/// the moment they can least afford it.
fn referenced_names(contents: &str) -> BTreeSet<String> {
    contents
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|token| token.starts_with("mx_"))
        .map(String::from)
        .collect()
}

/// Resolve a referenced name against the registry, accounting for the suffixes
/// the encoder adds.
///
/// Counters are registered without `_total` and exposed with it, so the `# TYPE`
/// line carries the bare name while every query must use the suffixed one.
/// Histograms expose three derived series from one registration.
fn resolves(name: &str, families: &BTreeMap<String, String>) -> bool {
    if CLIENT_FAMILIES.contains(&name) || families.contains_key(name) {
        return true;
    }
    if let Some(base) = name.strip_suffix("_total")
        && families.get(base).is_some_and(|kind| kind == "counter")
    {
        return true;
    }
    ["_bucket", "_sum", "_count"].iter().any(|suffix| {
        name.strip_suffix(suffix)
            .and_then(|base| families.get(base))
            .is_some_and(|kind| kind == "histogram")
    })
}

/// Pins the family inventory the map is built from.
///
/// Two jobs. It makes the touching in `registered_families` self-checking: a
/// family that stops being sampled vanishes from the encoded output entirely,
/// which would silently shrink the set every other assertion checks against.
/// And it forces a deliberate answer to "does this need an alert?" whenever a
/// family is added, rather than letting new metrics arrive unmonitored.
#[tokio::test]
async fn the_server_family_inventory_is_pinned() {
    let families = registered_families().await;
    let found: Vec<&str> = families.keys().map(String::as_str).collect();

    // Bare registered names: counters gain _total and histograms gain
    // _bucket/_sum/_count only when exported.
    let expected = [
        "mx_backend_op_seconds",
        "mx_backend_ops",
        "mx_backend_ops_in_flight",
        "mx_build_info",
        "mx_cache_evictions",
        "mx_download_claims",
        "mx_download_lease_refresh",
        "mx_download_seconds",
        "mx_grpc_request_seconds",
        "mx_grpc_requests",
        "mx_grpc_requests_in_flight",
        "mx_registry_entries",
        "mx_registry_status_transitions",
        "mx_state_entries",
        "mx_task_last_success_timestamp_seconds",
    ];

    assert_eq!(
        found, expected,
        "server metric families changed.\n\
         If you added one, decide whether it needs an alert in \
         helm/templates/prometheusrule.yaml and add it here.\n\
         If one disappeared, it is more likely that registered_families() stopped \
         touching it than that it was deleted -- an untouched family encodes to \
         nothing at all."
    );
}

#[tokio::test]
async fn every_metric_named_by_an_alert_rule_exists() {
    let path = rules_template();
    let contents =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let families = registered_families().await;
    let referenced = referenced_names(&contents);

    // Guards against the whole check quietly becoming a no-op if the template is
    // renamed or the extraction stops matching: an empty set passes every
    // assertion below without testing anything.
    assert!(
        referenced.len() > 5,
        "found only {} mx_* names in {} -- the extraction is broken, not the rules",
        referenced.len(),
        path.display()
    );
    assert!(
        families.len() > 5,
        "registry produced only {} families; registration or `# TYPE` parsing is broken",
        families.len()
    );

    let unknown: Vec<&String> = referenced
        .iter()
        .filter(|name| !resolves(name, &families))
        .collect();

    assert!(
        unknown.is_empty(),
        "alert rules reference metrics the server does not export: {unknown:#?}\n\
         Known server families (bare names; counters gain _total, histograms gain \
         _bucket/_sum/_count when queried): {:#?}\n\
         If a name belongs to the Python client, add it to CLIENT_FAMILIES here and \
         to the matching list in test_metrics.py.",
        families.keys().collect::<Vec<_>>()
    );
}

/// The check above only has teeth if an unknown name would actually be rejected.
/// Without this, a `resolves` that returned `true` unconditionally -- via a
/// stray suffix arm or an over-broad fallback -- would keep the suite green
/// while checking nothing.
#[tokio::test]
async fn an_unexported_metric_name_is_rejected() {
    let families = registered_families().await;

    assert!(
        !resolves("mx_not_a_real_family", &families),
        "resolves() accepts a name that was never registered"
    );
    // A real base name with the wrong suffix class: `_bucket` is only valid on a
    // histogram, and mx_registry_entries is a gauge.
    assert!(
        !resolves("mx_registry_entries_bucket", &families),
        "resolves() accepts a histogram suffix on a gauge"
    );
    // Sanity in the positive direction, so the two assertions above cannot pass
    // by rejecting everything.
    assert!(
        resolves("mx_download_claims_total", &families),
        "resolves() rejects a counter that is genuinely registered"
    );
    assert!(
        resolves("mx_download_seconds_count", &families),
        "resolves() rejects a histogram-derived series that genuinely exists"
    );
}

/// Each alert must carry a `summary`, because Alertmanager renders it as the
/// notification title. An alert without one pages someone with a bare rule name
/// and no indication of what broke.
#[test]
fn every_alert_carries_a_summary() {
    let path = rules_template();
    let contents =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

    let alerts: Vec<&str> = contents
        .lines()
        .filter_map(|line| line.trim().strip_prefix("- alert: "))
        .collect();
    let summaries = contents
        .lines()
        .filter(|line| line.trim().starts_with("summary:"))
        .count();

    assert!(
        alerts.len() >= 10,
        "expected the full rule set, got {alerts:#?}"
    );
    assert_eq!(
        alerts.len(),
        summaries,
        "{} alerts but {summaries} summary annotations: {alerts:#?}",
        alerts.len()
    );
}
