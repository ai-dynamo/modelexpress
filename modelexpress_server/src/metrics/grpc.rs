// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-RPC metrics for every gRPC service the server exposes.
//!
//! # Why the outcome is not the gRPC status code
//!
//! Several handlers report failure *in band*: they catch a backend error, log
//! it, and return `Ok` with an empty or defaulted response body. `ListSources`
//! is the clearest case — a Redis outage and "no peers have published yet" are
//! the same `Ok(ListSourcesResponse { instances: [] })` on the wire. A metric
//! derived from the status code would therefore read 100% success straight
//! through a total backend outage, which is precisely the incident it exists to
//! surface.
//!
//! So the outcome is resolved in three steps, in order:
//!
//! 1. An [`RpcOutcome`] the handler itself inserted into the response
//!    extensions. This is the authoritative value and the only one that can see
//!    an in-band failure.
//! 2. Otherwise a [`tonic::Status`] in the response extensions. Every
//!    `Err(Status)` path puts itself there — `Status::into_http` does
//!    `response.extensions_mut().insert(self)` — so handlers that fail honestly
//!    need no edit at all, and neither do auth rejections or tonic's own
//!    `unimplemented` fallback for an unrouted path.
//! 3. Otherwise success. A successful unary response carries `grpc-status` in
//!    the HTTP/2 trailers, not the head, so its absence here is the success
//!    signal.
//!
//! Step 2 is why instrumenting 21 routed paths costs a handful of one-line tags
//! rather than 21 rewrites: only handlers that lie about their outcome need
//! touching.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use prometheus_client::encoding::{EncodeLabelSet, EncodeLabelValue, LabelValueEncoder};
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;
use tower::{Layer, Service};

use super::buckets;

/// Outcome of one RPC.
///
/// A closed enum, deliberately: `method` is already a 22-value label, so an
/// open-ended outcome would multiply an unbounded domain into it. Anything that
/// does not map onto a named variant becomes [`RpcOutcome::Error`].
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum RpcOutcome {
    /// The handler completed and reported no failure.
    Ok,
    /// The caller's request was malformed.
    InvalidArgument,
    /// The requested model, source, or version does not exist. Distinct from
    /// [`RpcOutcome::BackendError`] because it is a normal, expected answer.
    NotFound,
    /// A backend the handler depends on failed. This is the variant a
    /// status-code-derived metric cannot see, and the reason this module exists.
    ///
    /// Wider than the metadata store: every `Code::Unavailable` maps here, which
    /// includes the auth layer's `Status::unavailable` when the Kubernetes
    /// TokenReview API is down. Because this layer sits outside `AuthLayer`, that
    /// case appears on every method at once and can read like a total
    /// metadata-store outage. `mx_backend_ops_total` disambiguates: a real store
    /// outage lights that family up too.
    BackendError,
    /// Rejected by the ServiceAccount auth layer.
    Unauthenticated,
    /// The caller went away before the handler finished: client disconnect,
    /// `RST_STREAM`, or a deadline. Recorded from a drop guard, never from a
    /// response, because in this case there is no response.
    Cancelled,
    /// Any other failure.
    Error,
}

impl RpcOutcome {
    /// Wire form. `snake_case`, matching every other label value in the schema.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::InvalidArgument => "invalid_argument",
            Self::NotFound => "not_found",
            Self::BackendError => "backend_error",
            Self::Unauthenticated => "unauthenticated",
            Self::Cancelled => "cancelled",
            Self::Error => "error",
        }
    }

    /// Map a gRPC status code onto the closed domain.
    fn from_code(code: tonic::Code) -> Self {
        match code {
            tonic::Code::Ok => Self::Ok,
            tonic::Code::InvalidArgument => Self::InvalidArgument,
            tonic::Code::NotFound => Self::NotFound,
            tonic::Code::Unauthenticated | tonic::Code::PermissionDenied => Self::Unauthenticated,
            tonic::Code::Unavailable => Self::BackendError,
            _ => Self::Error,
        }
    }
}

// Hand-written rather than derived: the derive encodes `stringify!(Variant)`,
// which would emit `BackendError`. Renaming the variants to match the wire form
// would trip `non_camel_case_types` under `-D warnings`, so the mapping lives in
// `as_str` instead.
impl EncodeLabelValue for RpcOutcome {
    fn encode(&self, encoder: &mut LabelValueEncoder<'_>) -> Result<(), std::fmt::Error> {
        EncodeLabelValue::encode(&self.as_str(), encoder)
    }
}

/// Label set for the per-RPC families.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RpcLabels {
    /// `Service/Method`, drawn from a closed set by [`method_label`].
    pub method: &'static str,
    /// Resolved outcome; see the module docs for the resolution order.
    pub outcome: RpcOutcome,
}

/// Label set for in-flight tracking. No `outcome`: a request that is still
/// running does not have one yet.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct MethodLabel {
    /// `Service/Method`, from [`method_label`].
    pub method: &'static str,
}

fn fast_histogram() -> Histogram {
    Histogram::new(buckets::FAST)
}

type LatencyFamily = Family<RpcLabels, Histogram, fn() -> Histogram>;

/// Handles for the per-RPC families.
///
/// Cloning is cheap and shares the underlying storage: every family in
/// `prometheus_client` is `Arc`-backed, so the clone held by the tower layer and
/// the clone held by the registry are the same counters.
#[derive(Clone)]
pub struct GrpcMetrics {
    requests: Family<RpcLabels, Counter>,
    latency: LatencyFamily,
    in_flight: Family<MethodLabel, Gauge>,
}

impl GrpcMetrics {
    /// Register the families on `registry` and return handles to them.
    ///
    /// Counters are registered **without** the `_total` suffix. The OpenMetrics
    /// encoder appends it, so registering `requests_total` would export
    /// `mx_grpc_requests_total_total`. The unit tests assert the encoded names
    /// rather than the registered ones for exactly this reason.
    pub fn register(registry: &mut Registry) -> Self {
        let requests = Family::<RpcLabels, Counter>::default();
        let latency: LatencyFamily =
            Family::new_with_constructor(fast_histogram as fn() -> Histogram);
        let in_flight = Family::<MethodLabel, Gauge>::default();

        let grpc = registry.sub_registry_with_prefix("grpc");
        grpc.register(
            "requests",
            "gRPC requests by method and resolved outcome",
            requests.clone(),
        );
        grpc.register(
            "request_seconds",
            "gRPC handler latency by method and resolved outcome",
            latency.clone(),
        );

        // A plain Gauge, deliberately, unlike the counter pairs the client side
        // uses. That pattern exists because `prometheus_client` multiprocess mode
        // cannot reclaim a gauge owned by a rank that was SIGKILLed, so an
        // in-flight gauge wedges forever. The server is a single process with an
        // in-process registry: if it dies the registry dies with it, so there is
        // nothing to wedge and the direct reading is worth more than a
        // subtraction.
        grpc.register(
            "requests_in_flight",
            "gRPC requests currently being handled, by method",
            in_flight.clone(),
        );

        Self {
            requests,
            latency,
            in_flight,
        }
    }

    /// Record one completed RPC.
    ///
    /// Each `get_or_create` is a complete statement. The guard it returns is
    /// `!Send`, so holding one across an `.await` would make the enclosing future
    /// non-`Send` and break the tonic handler bound, with an error that points at
    /// lock internals rather than at this line.
    fn record(&self, labels: &RpcLabels, elapsed_seconds: f64) {
        self.requests.get_or_create(labels).inc();
        self.latency.get_or_create(labels).observe(elapsed_seconds);
    }

    /// Mark one request as started.
    fn in_flight_inc(&self, method: &'static str) {
        self.in_flight.get_or_create(&MethodLabel { method }).inc();
    }

    /// Mark one request as finished, however it finished.
    fn in_flight_dec(&self, method: &'static str) {
        self.in_flight.get_or_create(&MethodLabel { method }).dec();
    }

    /// Count a request whose future was dropped before it produced a response.
    ///
    /// The counter only. Writing the elapsed time into the histogram would
    /// manufacture a short-latency sample for a request that was still running,
    /// skewing the distribution worse than the gap it closes.
    fn record_cancelled(&self, method: &'static str) {
        self.requests
            .get_or_create(&RpcLabels {
                method,
                outcome: RpcOutcome::Cancelled,
            })
            .inc();
    }
}

/// Map a request path onto the closed `method` domain.
///
/// The layer wraps the whole router, so it sees every path that reaches the
/// listener, including ones no service claims. Taking the label straight from
/// the URI would let any caller mint a new series per request, so unknown paths
/// collapse to `other`.
#[must_use]
pub fn method_label(path: &str) -> &'static str {
    match path {
        "/model_express.api.ApiService/SendRequest" => "ApiService/SendRequest",
        "/model_express.health.HealthService/GetHealth" => "HealthService/GetHealth",

        "/model_express.model.ModelService/EnsureModelDownloaded" => {
            "ModelService/EnsureModelDownloaded"
        }
        "/model_express.model.ModelService/StreamModelFiles" => "ModelService/StreamModelFiles",
        "/model_express.model.ModelService/ListModelFiles" => "ModelService/ListModelFiles",
        "/model_express.model.ModelService/DeleteModel" => "ModelService/DeleteModel",

        "/model_express.p2p.P2pService/PublishMetadata" => "P2pService/PublishMetadata",
        "/model_express.p2p.P2pService/ListSources" => "P2pService/ListSources",
        "/model_express.p2p.P2pService/GetMetadata" => "P2pService/GetMetadata",
        "/model_express.p2p.P2pService/UpdateStatus" => "P2pService/UpdateStatus",

        "/model_express.refit.RefitService/CreateWeightVersion" => {
            "RefitService/CreateWeightVersion"
        }
        "/model_express.refit.RefitService/GetWeightVersion" => "RefitService/GetWeightVersion",
        "/model_express.refit.RefitService/DeleteWeightVersion" => {
            "RefitService/DeleteWeightVersion"
        }
        "/model_express.refit.RefitService/UpdateWeightVersionState" => {
            "RefitService/UpdateWeightVersionState"
        }
        "/model_express.refit.RefitService/RegisterWorker" => "RefitService/RegisterWorker",
        "/model_express.refit.RefitService/CreateWeightVersionShard" => {
            "RefitService/CreateWeightVersionShard"
        }
        "/model_express.refit.RefitService/ListWeightVersionShards" => {
            "RefitService/ListWeightVersionShards"
        }
        "/model_express.refit.RefitService/DeleteWeightVersionShard" => {
            "RefitService/DeleteWeightVersionShard"
        }
        "/model_express.refit.RefitService/RegisterVersionLease" => {
            "RefitService/RegisterVersionLease"
        }
        "/model_express.refit.RefitService/DeleteVersionLease" => "RefitService/DeleteVersionLease",

        // The standard health service, served for kubelet probes. Counted
        // deliberately: it is the cheapest evidence that the gRPC listener is
        // answering at all, including during the drain window.
        "/grpc.health.v1.Health/Check" => "Health/Check",
        "/grpc.health.v1.Health/Watch" => "Health/Watch",

        _ => "other",
    }
}

/// Methods whose response head is produced before the work happens.
///
/// `EnsureModelDownloaded` and `StreamModelFiles` return
/// `Ok(Response::new(ReceiverStream::new(rx)))` once a task is spawned, and report
/// failure as a stream item or a trailer -- `EnsureModelDownloaded` sends
/// `ModelStatus::ERROR` as an item. `Health/Watch` is long-lived by design. For
/// all three, this layer's future resolves at stream setup, so it would record
/// `outcome="ok"` and a sub-millisecond duration for work that can run for forty
/// minutes and fail.
///
/// They are therefore not recorded at all. A confidently wrong number is worse
/// than an absent one, and this module exists to remove exactly that class of
/// lie. Covering them means timing the response body and reading the trailer,
/// which is a separate change.
fn resolves_at_head(method: &str) -> bool {
    matches!(
        method,
        "ModelService/EnsureModelDownloaded" | "ModelService/StreamModelFiles" | "Health/Watch"
    )
}

/// Resolve the outcome of a response. See the module docs for the ordering.
///
/// Everything readable here comes from the response *head*. A failure raised
/// while hyper drains the body lands in the HTTP/2 trailers instead, after this
/// has already run: `map_response` is not async, so the future this layer awaits
/// resolves once the body is wrapped, not once it is encoded. The reachable case
/// is a unary response exceeding `max_encoding_message_size`, which
/// `finish_encoding` rejects during the drain -- the client sees an error and the
/// server records a success. Streaming methods are excluded separately by
/// [`resolves_at_head`].
fn resolve_outcome<B>(response: &http::Response<B>) -> RpcOutcome {
    if let Some(outcome) = response.extensions().get::<RpcOutcome>() {
        return *outcome;
    }
    if let Some(status) = response.extensions().get::<tonic::Status>() {
        return RpcOutcome::from_code(status.code());
    }
    RpcOutcome::Ok
}

/// Tracks one in-flight request, and records a cancellation if the future is
/// dropped before it produced a response.
///
/// Both signals hang off the same `Drop` because they answer the same question
/// from opposite ends. hyper drops the handler future on client disconnect,
/// `RST_STREAM`, or a deadline, and the recording after the await then never
/// runs -- and that gap is not uniform, because the requests most likely to be
/// cancelled are the slow ones. Left alone, the latency histogram ends up
/// conditioned on completion, with a tail that looks healthy precisely because
/// the slowest samples are the ones missing.
///
/// The gauge is the direct reading of the same situation: a backend that wedges
/// while its peers serve normally shows up immediately as requests accumulating,
/// rather than having to be inferred from an absence.
struct InFlightGuard {
    metrics: GrpcMetrics,
    method: &'static str,
    /// Cleared once the call resolves. Still set at drop means cancelled.
    pending: bool,
}

impl InFlightGuard {
    fn enter(metrics: GrpcMetrics, method: &'static str) -> Self {
        metrics.in_flight_inc(method);
        Self {
            metrics,
            method,
            pending: true,
        }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        // Runs on every exit path, so the gauge cannot drift.
        self.metrics.in_flight_dec(self.method);
        if self.pending {
            self.metrics.record_cancelled(self.method);
        }
    }
}

/// Tower layer recording [`GrpcMetrics`] for every request that reaches the router.
///
/// Applied once, to the whole router, rather than per service: it then also
/// covers the health services and any service added later without a second edit.
/// It sits outside the auth layer, so rejected calls are counted as
/// `outcome="unauthenticated"` rather than vanishing.
#[derive(Clone)]
pub struct GrpcMetricsLayer {
    metrics: GrpcMetrics,
}

impl GrpcMetricsLayer {
    #[must_use]
    pub fn new(metrics: GrpcMetrics) -> Self {
        Self { metrics }
    }
}

impl<S> Layer<S> for GrpcMetricsLayer {
    type Service = GrpcMetricsService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        GrpcMetricsService {
            inner,
            metrics: self.metrics.clone(),
        }
    }
}

/// The service produced by [`GrpcMetricsLayer`].
#[derive(Clone)]
pub struct GrpcMetricsService<S> {
    inner: S,
    metrics: GrpcMetrics,
}

impl<S, ReqBody, ResBody> Service<http::Request<ReqBody>> for GrpcMetricsService<S>
where
    S: Service<http::Request<ReqBody>, Response = http::Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    ReqBody: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: http::Request<ReqBody>) -> Self::Future {
        let method = method_label(request.uri().path());
        // See `resolves_at_head`: for streaming methods this future resolves before
        // the work does, so anything recorded here would describe stream setup
        // while claiming to describe the call.
        let record = !resolves_at_head(method);
        let metrics = self.metrics.clone();

        // Readiness was established on `self.inner` by `poll_ready` and does not
        // transfer to a clone. Swap so the future drives the ready service and
        // the fresh clone is the one polled ready next time.
        let ready = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, ready);

        Box::pin(async move {
            // Streaming methods are excluded from this too: their future resolves
            // at the head, so the gauge would decrement immediately and read zero
            // for a stream that is still running.
            let mut guard = record.then(|| InFlightGuard::enter(metrics.clone(), method));
            let started = Instant::now();
            let result = inner.call(request).await;
            if let Some(guard) = guard.as_mut() {
                guard.pending = false;
            }

            if record && let Ok(response) = &result {
                let labels = RpcLabels {
                    method,
                    outcome: resolve_outcome(response),
                };
                // `Duration::as_secs_f64` is float division, which
                // `arithmetic_side_effects` permits; `Instant - Instant` is not.
                metrics.record(&labels, started.elapsed().as_secs_f64());
            }
            result
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{encode_text, new_registry};

    /// The encoder appends `_total` to counters, so the registered name and the
    /// exported name differ. Assert the exported one.
    #[test]
    fn families_encode_under_the_documented_names() {
        let mut registry = new_registry();
        let metrics = GrpcMetrics::register(&mut registry);

        metrics.record(
            &RpcLabels {
                method: "P2pService/ListSources",
                outcome: RpcOutcome::BackendError,
            },
            0.5,
        );
        metrics.record_cancelled("P2pService/GetMetadata");

        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));

        assert!(
            encoded.contains(
                r#"mx_grpc_requests_total{method="P2pService/ListSources",outcome="backend_error"} 1"#
            ),
            "{encoded}"
        );
        assert!(
            encoded.contains(
                r#"mx_grpc_requests_total{method="P2pService/GetMetadata",outcome="cancelled"} 1"#
            ),
            "{encoded}"
        );
        // A cancellation contributes no latency sample: only the one real
        // observation reaches the histogram.
        assert!(
            encoded.contains(r#"mx_grpc_request_seconds_count{method="P2pService/ListSources"#),
            "{encoded}"
        );
        assert!(
            !encoded.contains(r#"mx_grpc_request_seconds_count{method="P2pService/GetMetadata"#),
            "a cancellation wrote a latency sample: {encoded}"
        );
        assert!(
            encoded.contains("# TYPE mx_grpc_request_seconds histogram"),
            "{encoded}"
        );
        // Not `_total_total`.
        assert!(!encoded.contains("_total_total"), "{encoded}");
    }

    /// The guard decrements on every exit path, so the gauge must come back to
    /// zero whether the request completed or was dropped.
    #[test]
    fn the_in_flight_gauge_returns_to_zero() {
        let mut registry = new_registry();
        let metrics = GrpcMetrics::register(&mut registry);

        let guard = InFlightGuard::enter(metrics.clone(), "P2pService/ListSources");
        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(r#"mx_grpc_requests_in_flight{method="P2pService/ListSources"} 1"#),
            "{encoded}"
        );

        drop(guard);
        let encoded = encode_text(&registry).unwrap_or_else(|_| String::from("<encode failed>"));
        assert!(
            encoded.contains(r#"mx_grpc_requests_in_flight{method="P2pService/ListSources"} 0"#),
            "{encoded}"
        );
        // Dropped while still pending, so it also counts as a cancellation.
        assert!(
            encoded.contains(
                r#"mx_grpc_requests_total{method="P2pService/ListSources",outcome="cancelled"} 1"#
            ),
            "{encoded}"
        );
    }

    #[test]
    fn outcome_labels_are_snake_case() {
        // The derive would have emitted `BackendError` here.
        assert_eq!(RpcOutcome::BackendError.as_str(), "backend_error");
        assert_eq!(RpcOutcome::InvalidArgument.as_str(), "invalid_argument");
        assert_eq!(RpcOutcome::Ok.as_str(), "ok");
    }

    /// The cardinality guard: an unknown path must not mint a series.
    #[test]
    fn unknown_paths_collapse_to_other() {
        assert_eq!(
            method_label("/model_express.p2p.P2pService/ListSources"),
            "P2pService/ListSources"
        );
        assert_eq!(method_label("/grpc.health.v1.Health/Check"), "Health/Check");
        assert_eq!(method_label("/model_express.p2p.P2pService/Bogus"), "other");
        assert_eq!(method_label("/nonsense"), "other");
        assert_eq!(method_label(""), "other");
    }

    /// A handler-set outcome outranks anything derivable from the status.
    ///
    /// This covers the resolution order only. It cannot cover the propagation
    /// step -- `tonic::Response::into_http`, which copies handler extensions onto
    /// the `http::Response`, is `pub(crate)`, so no test outside tonic can drive
    /// it directly. That half is covered end to end by the `in_process_server`
    /// integration test, which drives a real client against a real server.
    #[test]
    fn handler_set_outcome_wins_over_the_status_code() {
        let mut response = http::Response::new(());
        response.extensions_mut().insert(RpcOutcome::BackendError);
        // A conflicting status is present and must lose.
        response.extensions_mut().insert(tonic::Status::ok(""));
        assert_eq!(resolve_outcome(&response), RpcOutcome::BackendError);
    }

    #[test]
    fn status_is_used_when_the_handler_did_not_tag() {
        let response = tonic::Status::not_found("nope").into_http::<()>();
        assert_eq!(resolve_outcome(&response), RpcOutcome::NotFound);

        let response = tonic::Status::unavailable("redis down").into_http::<()>();
        assert_eq!(resolve_outcome(&response), RpcOutcome::BackendError);
    }

    /// Services this server actually routes. `WorkerService` is served by the
    /// Python client and `RefitWorkerService` has no ModelExpress-owned server,
    /// so neither reaches this layer.
    const SERVED_SERVICES: [&str; 5] = [
        "ApiService",
        "HealthService",
        "ModelService",
        "P2pService",
        "RefitService",
    ];

    /// Extract `/package.Service/Method` for every RPC a proto declares.
    fn routed_paths(proto: &str) -> Vec<(String, String, bool)> {
        let mut package = String::new();
        let mut service = String::new();
        let mut paths = Vec::new();
        for line in proto.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("package ") {
                package = rest.trim_end_matches(';').trim().to_string();
            } else if let Some(rest) = line.strip_prefix("service ") {
                service = rest
                    .split_whitespace()
                    .next()
                    .unwrap_or_default()
                    .to_string();
            } else if let Some(rest) = line.strip_prefix("rpc ")
                && let Some(name) = rest.split('(').next()
            {
                paths.push((
                    service.clone(),
                    format!("/{package}.{service}/{}", name.trim()),
                    rest.contains("returns (stream "),
                ));
            }
        }
        paths
    }

    /// Every routed RPC must have a `method_label` arm.
    ///
    /// Without this, adding an RPC to a proto silently lands it in `other` --
    /// and because `resolves_at_head("other")` is false, a *streaming* RPC added
    /// that way would be treated as unary and recorded with a bogus `ok` and a
    /// stream-setup duration. This fails the build instead.
    #[test]
    fn every_routed_proto_rpc_has_a_method_label_arm() {
        const PROTOS: [&str; 5] = [
            include_str!("../../../modelexpress_common/proto/api.proto"),
            include_str!("../../../modelexpress_common/proto/health.proto"),
            include_str!("../../../modelexpress_common/proto/model.proto"),
            include_str!("../../../modelexpress_common/proto/p2p.proto"),
            include_str!("../../../modelexpress_common/proto/refit.proto"),
        ];

        let mut checked = 0;
        for proto in PROTOS {
            for (service, path, is_streaming) in routed_paths(proto) {
                if !SERVED_SERVICES.contains(&service.as_str()) {
                    continue;
                }
                let method = method_label(&path);
                assert_ne!(
                    method, "other",
                    "{path} has no method_label arm, so it would be recorded as `other`"
                );
                // Derived from the proto rather than from a hand-kept list, so
                // adding a streaming RPC and its arm -- but forgetting
                // `resolves_at_head` -- fails here instead of silently recording
                // `ok` with a stream-setup duration.
                assert_eq!(
                    resolves_at_head(method),
                    is_streaming,
                    "{path} is {} in the proto but {} excluded by resolves_at_head",
                    if is_streaming { "streaming" } else { "unary" },
                    if resolves_at_head(method) {
                        "is"
                    } else {
                        "is not"
                    }
                );
                checked += 1;
            }
        }
        // Guards the parser itself: a change that stopped matching `rpc` lines
        // would otherwise make this test pass by checking nothing.
        assert_eq!(
            checked, 20,
            "expected 20 routed server RPCs, parsed {checked}"
        );
    }

    /// `Health/Watch` is long-lived but is not declared in our protos, so
    /// `every_routed_proto_rpc_has_a_method_label_arm` cannot derive it. The two
    /// model.proto streams are pinned there against the proto itself; they are
    /// repeated here only to keep this test readable on its own.
    #[test]
    fn streaming_methods_are_not_recorded() {
        assert!(resolves_at_head("ModelService/EnsureModelDownloaded"));
        assert!(resolves_at_head("ModelService/StreamModelFiles"));
        assert!(resolves_at_head("Health/Watch"));

        // Everything unary is recorded.
        assert!(!resolves_at_head("P2pService/ListSources"));
        assert!(!resolves_at_head("ModelService/ListModelFiles"));
        assert!(!resolves_at_head("Health/Check"));
        assert!(!resolves_at_head("other"));
    }

    #[test]
    fn an_untagged_success_is_ok() {
        let response = http::Response::new(());
        assert_eq!(resolve_outcome(&response), RpcOutcome::Ok);
    }
}
