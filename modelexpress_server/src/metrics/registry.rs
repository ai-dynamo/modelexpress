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
    // No ALL constant here, unlike ClaimResult and LeaseResult. Those families
    // pre-create every variant; this one pre-creates only the reachable from/to
    // pairs, listed in RegistryMetrics::REACHABLE_TRANSITIONS. An ALL over the
    // variants would have no caller and would invite a future 4x4 sweep back in.

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
    /// Every variant, so the claim family can be pre-created at zero.
    pub(crate) const ALL: [Self; 4] = [
        Self::Claimed,
        Self::Takeover,
        Self::AlreadyExists,
        Self::Error,
    ];

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
    /// Every variant, so the lease family can be pre-created at zero.
    pub(crate) const ALL: [Self; 3] = [Self::Renewed, Self::Lost, Self::Error];

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
    /// The transitions the code can actually produce, and where each comes from.
    ///
    /// The label domain is 4x4, but ten of those sixteen pairs are unreachable:
    /// nothing moves an entry from `downloaded` to `error`, and `absent` is only
    /// ever a source for a first claim or a target for a delete. Pre-creating
    /// all sixteen exported ten series that could never leave zero.
    ///
    /// Keep this in step with the call sites. A pair that is produced but not
    /// listed still records correctly -- `get_or_create` makes it on demand --
    /// but its first occurrence in each process is invisible to `increase()`,
    /// which is the whole reason this pre-creation exists.
    const REACHABLE_TRANSITIONS: [(StatusLabel, StatusLabel); 6] = [
        // record_claim(Claimed): a first claim on an entry that did not exist.
        (StatusLabel::Absent, StatusLabel::Downloading),
        // record_claim(Takeover): ownership changed, the entry never left
        // DOWNLOADING, so this is an arrival and a departure that cancel.
        (StatusLabel::Downloading, StatusLabel::Downloading),
        // reset_download_claim: an error-retry restarting a failed download.
        (StatusLabel::Error, StatusLabel::Downloading),
        // finish_download_claim, both terminal outcomes.
        (StatusLabel::Downloading, StatusLabel::Downloaded),
        (StatusLabel::Downloading, StatusLabel::Error),
        // A delete while downloading; the deleting side books it because
        // finish_download_claim finds no record to fence.
        (StatusLabel::Downloading, StatusLabel::Absent),
    ];

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

        // Create every label combination at zero.
        //
        // `Family` is lazy: a child appears on first `get_or_create`, so without
        // this a counter's first-ever exported sample is 1. Prometheus has no
        // earlier point to subtract, `rate()` and `increase()` over that window
        // are 0, and an alert watching for a rare discrete event misses the
        // first occurrence in every process -- silently, because an alert whose
        // expression yields nothing simply never fires.
        //
        // That is the difference between catching the first takeover after a
        // restart, which re-pulls the whole model, and only ever catching the
        // second. Born at zero, the increment is a visible step.
        //
        // Scoped to these three families deliberately. Their alerts key on rare
        // one-shot events, and the domains are small: 4 + 3 + 6 series. The
        // gRPC and backend families would cost 154 and 93 permanently-zero
        // series per pod for ratio alerts that need sustained traffic to fire
        // anyway.
        for result in ClaimResult::ALL {
            let _ = claims.get_or_create(&ClaimLabels { result });
        }
        for result in LeaseResult::ALL {
            let _ = lease_refreshes.get_or_create(&LeaseLabels { result });
        }
        for (from, to) in Self::REACHABLE_TRANSITIONS {
            let _ = transitions.get_or_create(&TransitionLabels { from, to });
        }

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
    /// The two outcomes a download can actually end in.
    ///
    /// `Absent` and `Downloading` are transition states, never terminal, so
    /// pre-creating them would claim a download can finish in `downloading`.
    const TERMINAL: [StatusLabel; 2] = [StatusLabel::Downloaded, StatusLabel::Error];

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
        // Same lazy-child problem as the counters above, and the same fix. Left
        // lazy, the first download in a process makes `_count` appear at 1 with
        // no earlier point to subtract, so `increase()` reports 0: the first
        // download after every restart is invisible. That silences
        // MXDownloadFailureRatio for a first-download failure -- 0/0 is no data,
        // not a ratio -- and made the dashboard's own "Downloads completed" tile
        // read 0 on a cluster where a download had demonstrably just succeeded.
        //
        // 30 always-present series per pod: two outcomes over twelve buckets
        // plus +Inf, _sum and _count.
        for outcome in Self::TERMINAL {
            let _ = seconds.get_or_create(&DownloadLabels { outcome });
        }
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
        // Present because the family is pre-created, and zero because nothing
        // succeeded. Asserting absence would now pass for the wrong reason.
        assert!(
            encoded.contains(r#"mx_download_seconds_count{outcome="downloaded"} 0"#),
            "a failed download must not be timed as a successful one: {encoded}"
        );
    }

    /// Both terminal outcomes are exported at zero before any download runs, so
    /// `increase()` has an earlier point to subtract and the first download in a
    /// process is visible. Without this, `MXDownloadFailureRatio` cannot fire on
    /// a first-download failure: 0/0 is no data, not a ratio.
    #[test]
    fn download_outcomes_are_exported_at_zero_before_any_download() {
        let mut registry = new_registry();
        let _downloads = DownloadMetrics::register(&mut registry);

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        for outcome in ["downloaded", "error"] {
            assert!(
                encoded.contains(&format!(
                    r#"mx_download_seconds_count{{outcome="{outcome}"}} 0"#
                )),
                "{outcome} missing before any download: {encoded}"
            );
        }
        // Transition states are not download outcomes and must stay absent.
        for outcome in ["absent", "downloading"] {
            assert!(
                !encoded.contains(&format!(
                    r#"mx_download_seconds_count{{outcome="{outcome}"}}"#
                )),
                "{outcome} is not a terminal outcome: {encoded}"
            );
        }
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

    /// Counters must be exported at zero before anything happens to them.
    ///
    /// `Family` creates children lazily, so without pre-creation a counter's
    /// first exported sample is 1 and Prometheus has no earlier point to
    /// subtract: `rate()` and `increase()` over that window are 0. Every alert
    /// keyed on a rare one-shot event -- a lost lease, a takeover -- then misses
    /// the first occurrence in each process, and misses it *silently*, because
    /// an expression that yields nothing simply never fires.
    ///
    /// The three series asserted here are the ones the shipped alert rules key
    /// on. `takeover` is the sharpest: missing it means missing a full re-pull
    /// of the model.
    #[test]
    fn alertable_counters_are_exported_at_zero_before_any_event() {
        let mut registry = new_registry();
        let _metrics = RegistryMetrics::register(&mut registry);

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        for series in [
            r#"mx_download_claims_total{result="takeover"} 0"#,
            r#"mx_download_lease_refresh_total{result="lost"} 0"#,
            r#"mx_registry_status_transitions_total{from="absent",to="downloading"} 0"#,
        ] {
            assert!(
                encoded.contains(series),
                "missing {series}; a counter born at 1 leaves rate() blind to the first event: {encoded}"
            );
        }
    }

    /// Pre-creation must cover exactly the declared label values -- no more.
    ///
    /// An invented series would sit at zero forever under a name no code can
    /// increment, implying a condition is monitored when nothing reports it.
    #[test]
    fn pre_creation_covers_exactly_the_declared_label_values() {
        let mut registry = new_registry();
        let _metrics = RegistryMetrics::register(&mut registry);
        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));

        let count = |prefix: &str| encoded.lines().filter(|l| l.starts_with(prefix)).count();

        assert_eq!(
            count("mx_download_claims_total{"),
            ClaimResult::ALL.len(),
            "{encoded}"
        );
        assert_eq!(
            count("mx_download_lease_refresh_total{"),
            LeaseResult::ALL.len(),
            "{encoded}"
        );
        assert_eq!(
            count("mx_registry_status_transitions_total{"),
            RegistryMetrics::REACHABLE_TRANSITIONS.len(),
            "{encoded}"
        );

        // The unreachable pairs must stay absent. Pre-creating the full 4x4
        // exported ten series that could never leave zero.
        for (from, to) in [
            (StatusLabel::Absent, StatusLabel::Absent),
            (StatusLabel::Absent, StatusLabel::Downloaded),
            (StatusLabel::Downloaded, StatusLabel::Error),
            (StatusLabel::Error, StatusLabel::Downloaded),
        ] {
            let needle =
                format!(r#"mx_registry_status_transitions_total{{from="{from:?}",to="{to:?}"#)
                    .to_lowercase();
            assert!(!encoded.to_lowercase().contains(&needle), "{encoded}");
        }
    }

    /// Every pair the code books is pre-created, so no first occurrence is
    /// invisible to `increase()`. Drives the real call sites rather than
    /// restating the list, so adding a transition without listing it fails here.
    #[test]
    fn every_booked_transition_was_pre_created() {
        let mut registry = new_registry();
        let metrics = RegistryMetrics::register(&mut registry);

        let before = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        let series = |text: &str| {
            text.lines()
                .filter(|l| l.starts_with("mx_registry_status_transitions_total{"))
                .count()
        };
        let baseline = series(&before);

        metrics.record_claim(ClaimResult::Claimed);
        metrics.record_claim(ClaimResult::Takeover);
        metrics.record_transition(StatusLabel::Error, StatusLabel::Downloading);
        metrics.record_transition(StatusLabel::Downloading, StatusLabel::Downloaded);
        metrics.record_transition(StatusLabel::Downloading, StatusLabel::Error);
        metrics.record_transition(StatusLabel::Downloading, StatusLabel::Absent);

        let after = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert_eq!(
            series(&after),
            baseline,
            "a booked transition was not pre-created, so its first occurrence is \
             invisible to increase(): {after}"
        );
    }
}
