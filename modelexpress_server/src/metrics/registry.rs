// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metrics for the model download lifecycle.
//!
//! These answer two questions the server cannot currently answer at all: are we
//! re-downloading models because leases are being lost, and are models getting
//! stuck part-way through a download.
//!
//! # Why these hang off the claim lifecycle and not off `set_status`
//!
//! A status-transition counter needs the *prior* status, and `set_status` does
//! not have it: the Redis implementation's Lua returns 1 unconditionally without
//! reading what was there, and the in-memory one overwrites without capturing it.
//! Instrumenting it would mean inventing a `from` value.
//!
//! The claim lifecycle does know. A won claim is `absent -> downloading`, an
//! error-retry reset is `error -> downloading`, and a finished claim is
//! `downloading -> {downloaded, error}`. Those three cover every transition that
//! matters for "is a model wedged", they are exact rather than sampled, and none
//! of them needs a keyspace scan to compute.
//!
//! # Reading the wedged-model signal
//!
//! The authoritative count of downloads in flight is the refreshed gauge
//! `mx_registry_entries{status="downloading"}`, not a derivation over these
//! counters. The gauge is recomputed from the registry itself, so it is correct
//! across a restart and across replicas.
//!
//! Differencing the transition counters gives the same number *within one process
//! lifetime* and is useful for seeing flow rather than level:
//!
//! ```promql
//! sum(rate(mx_registry_status_transitions_total{to="downloading"}[5m]))
//! ```
//!
//! It is not a restart-safe level. `DOWNLOADING` persists in Redis while these
//! counters are process-local, so after a restart a download that began in the
//! previous process contributes a departure with no matching arrival and a raw
//! difference can go negative. That is a property of differencing counters, not
//! something a label change fixes -- hence the gauge.
//!
//! Within a process the difference does balance, and that rests on every exit
//! from `DOWNLOADING` being recorded. There are three, and the third is easy to
//! miss:
//!
//! - a finish, `downloading -> {downloaded, error}`;
//! - a takeover, `downloading -> downloading` -- an arrival and a departure that
//!   cancel, because ownership changed while the entry never left `DOWNLOADING`.
//!   Recording it as `absent -> downloading` would add an arrival that never
//!   leaves;
//! - a **delete while downloading**, `downloading -> absent`. Once the record is
//!   gone `finish_download_claim` finds nothing to fence and returns false, so it
//!   records no departure; the deleting side has to book it.

use modelexpress_common::models::ModelStatus;
use prometheus_client::encoding::{EncodeLabelSet, EncodeLabelValue, LabelValueEncoder};
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

use super::buckets;

/// A model status as it appears in a transition label.
///
/// Carries `Absent` because the interesting transition into `downloading` is
/// from a record that did not exist.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum StatusLabel {
    /// No record existed.
    Absent,
    /// `ModelStatus::DOWNLOADING`.
    Downloading,
    /// `ModelStatus::DOWNLOADED`.
    Downloaded,
    /// `ModelStatus::ERROR`.
    Error,
}

impl StatusLabel {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Downloading => "downloading",
            Self::Downloaded => "downloaded",
            Self::Error => "error",
        }
    }
}

impl From<ModelStatus> for StatusLabel {
    fn from(status: ModelStatus) -> Self {
        match status {
            ModelStatus::DOWNLOADING => Self::Downloading,
            ModelStatus::DOWNLOADED => Self::Downloaded,
            ModelStatus::ERROR => Self::Error,
        }
    }
}

// Hand-written for the same reason as the other label enums: the derive would
// encode the variant identifier, giving `Downloading` rather than `downloading`.
impl EncodeLabelValue for StatusLabel {
    fn encode(&self, encoder: &mut LabelValueEncoder<'_>) -> Result<(), std::fmt::Error> {
        EncodeLabelValue::encode(&self.as_str(), encoder)
    }
}

/// Outcome of a download-claim attempt.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ClaimResult {
    /// Created the record; this is the first download of the entry.
    Claimed,
    /// Took over an expired lease, so the bytes are being pulled again.
    Takeover,
    /// Someone else owns it; the caller waits.
    AlreadyExists,
    /// The backend call failed.
    Error,
}

impl ClaimResult {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Claimed => "claimed",
            Self::Takeover => "takeover",
            Self::AlreadyExists => "already_exists",
            Self::Error => "error",
        }
    }
}

impl EncodeLabelValue for ClaimResult {
    fn encode(&self, encoder: &mut LabelValueEncoder<'_>) -> Result<(), std::fmt::Error> {
        EncodeLabelValue::encode(&self.as_str(), encoder)
    }
}

/// Outcome of a lease heartbeat.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum LeaseResult {
    /// The lease was still ours and was extended.
    Renewed,
    /// The lease was no longer ours. Whatever this replica is still downloading
    /// is now unowned work, and someone else has taken over.
    Lost,
    /// The backend call failed, so ownership is unknown.
    Error,
}

impl LeaseResult {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Renewed => "renewed",
            Self::Lost => "lost",
            Self::Error => "error",
        }
    }
}

impl EncodeLabelValue for LeaseResult {
    fn encode(&self, encoder: &mut LabelValueEncoder<'_>) -> Result<(), std::fmt::Error> {
        EncodeLabelValue::encode(&self.as_str(), encoder)
    }
}

/// Label set for the transition counter.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct TransitionLabels {
    /// Status before the transition.
    pub from: StatusLabel,
    /// Status after it.
    pub to: StatusLabel,
}

/// Label set for the claim counter.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ClaimLabels {
    /// What the claim attempt achieved.
    pub result: ClaimResult,
}

/// Label set for the lease-heartbeat counter.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct LeaseLabels {
    /// What the heartbeat achieved.
    pub result: LeaseResult,
}

/// Handles for the download-lifecycle families. Cloning shares the storage.
#[derive(Clone)]
pub struct RegistryMetrics {
    transitions: Family<TransitionLabels, Counter>,
    claims: Family<ClaimLabels, Counter>,
    lease_refreshes: Family<LeaseLabels, Counter>,
}

impl RegistryMetrics {
    /// Register the families and return handles.
    ///
    /// Registered without the `_total` suffix; the encoder appends it.
    pub fn register(registry: &mut Registry) -> Self {
        let transitions = Family::<TransitionLabels, Counter>::default();
        let claims = Family::<ClaimLabels, Counter>::default();
        let lease_refreshes = Family::<LeaseLabels, Counter>::default();

        let sub = registry.sub_registry_with_prefix("registry");
        sub.register(
            "status_transitions",
            "Model registry status transitions observed on the claim lifecycle",
            transitions.clone(),
        );

        let download = registry.sub_registry_with_prefix("download");
        download.register(
            "claims",
            "Download claim attempts by outcome; takeover means the bytes are being pulled again",
            claims.clone(),
        );
        download.register(
            "lease_refresh",
            "Download lease heartbeats by outcome; lost is the leading indicator of a duplicate download",
            lease_refreshes.clone(),
        );

        Self {
            transitions,
            claims,
            lease_refreshes,
        }
    }

    /// Record a claim attempt, and the transition it implies.
    ///
    /// A won claim moves the entry into `downloading`; the two owning outcomes
    /// differ only in where they came from.
    pub fn record_claim(&self, result: ClaimResult) {
        self.claims.get_or_create(&ClaimLabels { result }).inc();
        let from = match result {
            ClaimResult::Claimed => StatusLabel::Absent,
            ClaimResult::Takeover => StatusLabel::Downloading,
            // Not a transition: nothing changed.
            ClaimResult::AlreadyExists | ClaimResult::Error => return,
        };
        self.record_transition(from, StatusLabel::Downloading);
    }

    /// Record a lease heartbeat.
    pub fn record_lease_refresh(&self, result: LeaseResult) {
        self.lease_refreshes
            .get_or_create(&LeaseLabels { result })
            .inc();
    }

    /// Record one status transition.
    pub fn record_transition(&self, from: StatusLabel, to: StatusLabel) {
        self.transitions
            .get_or_create(&TransitionLabels { from, to })
            .inc();
    }
}

/// Histogram of end-to-end model download durations.
///
/// Separate from [`RegistryMetrics`] because it is observed from the download
/// path rather than from the registry backend, and uses the hour-scale band: a
/// large model's cold download runs for tens of minutes, which the transfer-scale
/// buckets cannot represent.
#[derive(Clone)]
pub struct DownloadMetrics {
    seconds: Family<DownloadLabels, Histogram, fn() -> Histogram>,
}

/// Label set for the download histogram.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct DownloadLabels {
    /// Terminal status of the download.
    pub outcome: StatusLabel,
}

fn xslow_histogram() -> Histogram {
    Histogram::new(buckets::XSLOW)
}

impl DownloadMetrics {
    /// Register the family and return a handle.
    pub fn register(registry: &mut Registry) -> Self {
        let seconds: Family<DownloadLabels, Histogram, fn() -> Histogram> =
            Family::new_with_constructor(xslow_histogram as fn() -> Histogram);
        let download = registry.sub_registry_with_prefix("download");
        download.register(
            "seconds",
            "End-to-end model download duration by terminal status",
            seconds.clone(),
        );
        Self { seconds }
    }

    /// Observe one completed download.
    pub fn observe(&self, outcome: StatusLabel, elapsed_seconds: f64) {
        self.seconds
            .get_or_create(&DownloadLabels { outcome })
            .observe(elapsed_seconds);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{encode_text, new_registry};

    /// Each `ClaimResult` gets its own series, so a takeover is queryable apart
    /// from a fresh claim.
    ///
    /// Scope: this pins the wire form only. The decision that turns a backend
    /// `ClaimOutcome::TookOver` into `ClaimResult::Takeover` is made in
    /// [`crate::registry::backend::instrumented`] and cannot be observed from
    /// here -- feeding `ClaimResult` values in by hand says nothing about which
    /// one a real takeover produces.
    #[test]
    fn each_claim_result_encodes_as_its_own_label_value() {
        let mut registry = new_registry();
        let metrics = RegistryMetrics::register(&mut registry);

        metrics.record_claim(ClaimResult::Claimed);
        metrics.record_claim(ClaimResult::Takeover);
        metrics.record_claim(ClaimResult::AlreadyExists);

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));

        assert!(
            encoded.contains(r#"mx_download_claims_total{result="claimed"} 1"#),
            "{encoded}"
        );
        assert!(
            encoded.contains(r#"mx_download_claims_total{result="takeover"} 1"#),
            "{encoded}"
        );
        assert!(
            encoded.contains(r#"mx_download_claims_total{result="already_exists"} 1"#),
            "{encoded}"
        );
        assert!(!encoded.contains("_total_total"), "{encoded}");
    }

    /// `ModelStatus` reaches Prometheus only through `From<ModelStatus>`, and it
    /// feeds two labels operators read as ground truth: the `to` side of the
    /// finish transition and the `outcome` of `mx_download_seconds`. A swapped
    /// arm reports every failed download as a successful one in *both* families
    /// at once, so this asserts the encoded label rather than the enum variant.
    ///
    /// The counts are deliberately distinct (3 / 1 / 2) so no permutation of the
    /// three arms leaves the assertions satisfied. `absent` is pinned here too:
    /// it is the one `StatusLabel` with no `ModelStatus` behind it, and the
    /// `in_flight` helper below matches on `downloading` alone, so nothing else
    /// holds its wire form.
    #[test]
    fn a_model_status_keeps_its_own_label_through_the_encoder() {
        let mut registry = new_registry();
        let metrics = RegistryMetrics::register(&mut registry);
        // Both families on one registry, as `server::run_server` registers them.
        let downloads = DownloadMetrics::register(&mut registry);

        for _ in 0..3 {
            metrics.record_transition(StatusLabel::Absent, ModelStatus::DOWNLOADING.into());
        }
        metrics.record_transition(StatusLabel::Downloading, ModelStatus::DOWNLOADED.into());
        for _ in 0..2 {
            metrics.record_transition(StatusLabel::Downloading, ModelStatus::ERROR.into());
        }
        downloads.observe(ModelStatus::ERROR.into(), 1.0);

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(
                r#"mx_registry_status_transitions_total{from="absent",to="downloading"} 3"#
            ),
            "{encoded}"
        );
        assert!(
            encoded.contains(
                r#"mx_registry_status_transitions_total{from="downloading",to="downloaded"} 1"#
            ),
            "{encoded}"
        );
        assert!(
            encoded.contains(
                r#"mx_registry_status_transitions_total{from="downloading",to="error"} 2"#
            ),
            "{encoded}"
        );
        assert!(
            encoded.contains(r#"mx_download_seconds_count{outcome="error"} 1"#),
            "{encoded}"
        );
        assert!(
            !encoded.contains(r#"mx_download_seconds_count{outcome="downloaded"}"#),
            "a failed download must not be timed as a successful one: {encoded}"
        );
    }

    /// Sum the `to="downloading"` and `from="downloading"` series the way the
    /// documented query does, so the assertion is the derivation itself rather
    /// than the individual series.
    fn in_flight(encoded: &str) -> i64 {
        let mut arrivals: i64 = 0;
        let mut departures: i64 = 0;
        for line in encoded.lines() {
            let Some((labels, value)) = line
                .strip_prefix("mx_registry_status_transitions_total{")
                .and_then(|rest| rest.split_once("} "))
            else {
                continue;
            };
            let count: i64 = value.trim().parse().unwrap_or_default();
            if labels.contains(r#"to="downloading""#) {
                arrivals = arrivals.saturating_add(count);
            }
            if labels.contains(r#"from="downloading""#) {
                departures = departures.saturating_add(count);
            }
        }
        arrivals.saturating_sub(departures)
    }

    /// A takeover records `downloading -> downloading`: an arrival and a
    /// departure that cancel, because ownership changed while the entry never
    /// left `DOWNLOADING`. Recording it as `absent -> downloading` instead would
    /// add an arrival that never leaves and the level would drift up by one per
    /// takeover. That from-label choice lives in `record_claim`, so this test
    /// does exercise it.
    ///
    /// Scope: the finish step here is a hand-written `record_transition`, not a
    /// real one. Whether a real finish records a departure at all -- and whether
    /// a real takeover arrives as `ClaimResult::Takeover` -- is decided in
    /// [`crate::registry::backend::instrumented`] and is not covered from here.
    #[test]
    fn a_takeover_does_not_change_the_in_flight_level() {
        let mut registry = new_registry();
        let metrics = RegistryMetrics::register(&mut registry);

        metrics.record_claim(ClaimResult::Claimed);
        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert_eq!(in_flight(&encoded), 1, "one download in flight");

        // Ownership moves; the download itself is still the same one.
        metrics.record_claim(ClaimResult::Takeover);
        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert_eq!(
            in_flight(&encoded),
            1,
            "a takeover is not a second download"
        );

        metrics.record_transition(StatusLabel::Downloading, StatusLabel::Downloaded);
        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert_eq!(in_flight(&encoded), 0, "the download finished");
    }

    /// `downloading -> absent` is the third exit from `DOWNLOADING`, and the one
    /// the claim lifecycle cannot observe for itself; it must subtract from the
    /// level exactly as a finish does.
    ///
    /// Scope: arithmetic only. The caller that decides whether a delete books
    /// this transition is `services::ModelDownloadTracker::delete_model_entries`,
    /// which this test never reaches -- deleting that decision leaves this green.
    #[test]
    fn a_downloading_to_absent_transition_is_a_departure() {
        let mut registry = new_registry();
        let metrics = RegistryMetrics::register(&mut registry);

        metrics.record_claim(ClaimResult::Claimed);
        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert_eq!(in_flight(&encoded), 1);

        // What `model clear` on a still-downloading model has to book.
        metrics.record_transition(StatusLabel::Downloading, StatusLabel::Absent);
        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert_eq!(
            in_flight(&encoded),
            0,
            "a deleted download must not stay counted as in flight: {encoded}"
        );
    }

    /// A waiter observing an existing claim is not a transition at all.
    #[test]
    fn an_already_exists_claim_records_no_transition() {
        let mut registry = new_registry();
        let metrics = RegistryMetrics::register(&mut registry);

        metrics.record_claim(ClaimResult::Claimed);
        metrics.record_claim(ClaimResult::AlreadyExists);
        metrics.record_claim(ClaimResult::Error);

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert_eq!(
            in_flight(&encoded),
            1,
            "waiters and errors must not move the level: {encoded}"
        );
    }

    #[test]
    fn a_lost_lease_is_recorded_distinctly() {
        let mut registry = new_registry();
        let metrics = RegistryMetrics::register(&mut registry);

        metrics.record_lease_refresh(LeaseResult::Renewed);
        metrics.record_lease_refresh(LeaseResult::Lost);

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(r#"mx_download_lease_refresh_total{result="renewed"} 1"#),
            "{encoded}"
        );
        assert!(
            encoded.contains(r#"mx_download_lease_refresh_total{result="lost"} 1"#),
            "{encoded}"
        );
    }

    #[test]
    fn the_download_histogram_uses_the_hour_scale_band() {
        let mut registry = new_registry();
        let metrics = DownloadMetrics::register(&mut registry);

        // A cold DeepSeek-scale load runs for tens of minutes; the transfer-scale
        // bands top out well below this.
        metrics.observe(StatusLabel::Downloaded, 2400.0);

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(r#"mx_download_seconds_count{outcome="downloaded"} 1"#),
            "{encoded}"
        );
        assert!(
            encoded.contains(r#"mx_download_seconds_bucket{le="3600.0",outcome="downloaded"} 1"#),
            "a 40-minute download must land below the top boundary: {encoded}"
        );
    }
}
