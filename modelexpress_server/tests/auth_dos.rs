// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TokenReview bounds, driven over real gRPC against a fake apiserver.

#![allow(clippy::expect_used)]

mod common;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use http::header::CONTENT_TYPE;
use http_body_util::BodyExt;
use k8s_openapi::api::authentication::v1::{TokenReview, TokenReviewStatus, UserInfo};
use kube::client::{Body, Client as KubeClient};
use modelexpress_common::grpc::health::HealthRequest;
use modelexpress_common::grpc::health::health_service_client::HealthServiceClient;
use tonic::transport::Channel;
use tonic::{Code, Request};
use tower::BoxError;

use common::{free_port, sa_token, security_config, start_auth_server, stop_and_join};

const MAX_CONCURRENT_REVIEWS: usize = 8;
const MAX_CONCURRENT_REVIEWS_PER_CALLER: usize = 4;
const REVIEW_LATENCY: Duration = Duration::from_millis(20);

#[derive(Default)]
struct CallerGauge {
    in_flight: usize,
    peak: usize,
}

#[derive(Default)]
struct ReviewGauge {
    calls: AtomicUsize,
    in_flight: AtomicUsize,
    peak: AtomicUsize,
    per_caller: Mutex<HashMap<String, CallerGauge>>,
}

impl ReviewGauge {
    fn enter(&self, caller: &str) {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let in_flight = self
            .in_flight
            .fetch_add(1, Ordering::SeqCst)
            .saturating_add(1);
        self.peak.fetch_max(in_flight, Ordering::SeqCst);

        let mut per_caller = self.per_caller.lock().expect("per_caller lock");
        let entry = per_caller.entry(caller.to_string()).or_default();
        entry.in_flight = entry.in_flight.saturating_add(1);
        entry.peak = entry.peak.max(entry.in_flight);
    }

    fn exit(&self, caller: &str) {
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        let mut per_caller = self.per_caller.lock().expect("per_caller lock");
        let entry = per_caller.entry(caller.to_string()).or_default();
        entry.in_flight = entry.in_flight.saturating_sub(1);
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn peak(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }

    fn peak_for(&self, caller: &str) -> usize {
        self.per_caller
            .lock()
            .expect("per_caller lock")
            .get(caller)
            .map(|gauge| gauge.peak)
            .unwrap_or_default()
    }
}

fn sub_of(token: &str) -> String {
    let Some(payload) = token.split('.').nth(1) else {
        return token.to_string();
    };
    let Ok(decoded) = URL_SAFE_NO_PAD.decode(payload) else {
        return token.to_string();
    };
    let Ok(claims) = serde_json::from_slice::<serde_json::Value>(&decoded) else {
        return token.to_string();
    };
    claims["sub"].as_str().unwrap_or(token).to_string()
}

fn fake_apiserver() -> (KubeClient, Arc<ReviewGauge>) {
    let gauge = Arc::new(ReviewGauge::default());
    let service = tower::service_fn({
        let gauge = gauge.clone();
        move |request: http::Request<Body>| {
            let gauge = gauge.clone();
            async move {
                let body = request.into_body().collect().await?.to_bytes();
                let review: TokenReview = serde_json::from_slice(&body)?;
                let token = review.spec.token.unwrap_or_default();
                let username = sub_of(&token);

                gauge.enter(&username);
                tokio::time::sleep(REVIEW_LATENCY).await;
                gauge.exit(&username);

                let response = TokenReview {
                    status: Some(TokenReviewStatus {
                        authenticated: Some(true),
                        audiences: Some(vec!["modelexpress".to_string()]),
                        user: Some(UserInfo {
                            username: Some(username),
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                };
                let body = serde_json::to_vec(&response)?;
                Ok::<_, BoxError>(
                    http::Response::builder()
                        .status(http::StatusCode::OK)
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(body))?,
                )
            }
        }
    });
    (KubeClient::new(service, "default"), gauge)
}

async fn connect(port: u16) -> Channel {
    for _ in 0..100 {
        if let Ok(channel) = Channel::from_shared(format!("http://127.0.0.1:{port}"))
            .expect("channel uri")
            .connect()
            .await
        {
            return channel;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("server on port {port} never became reachable");
}

async fn health_check(channel: &Channel, token: &str) -> Result<(), Code> {
    let mut client = HealthServiceClient::new(channel.clone());
    let mut request = Request::new(HealthRequest {});
    request.metadata_mut().insert(
        "authorization",
        format!("Bearer {token}").parse().expect("bearer metadata"),
    );
    client
        .get_health(request)
        .await
        .map(|_| ())
        .map_err(|status| status.code())
}

#[tokio::test]
async fn unique_junk_tokens_neither_reach_the_apiserver_nor_evict_verified_callers() {
    const FLOOD: usize = 20_000;

    let port = free_port();
    let (kube_client, gauge) = fake_apiserver();
    let (shutdown, handle) =
        start_auth_server(port, kube_client, security_config(&[("vllm", "worker")]));
    let channel = connect(port).await;

    let victim = sa_token("vllm", "worker", "legit");
    health_check(&channel, &victim).await.expect("first review");
    assert_eq!(gauge.calls(), 1);

    for nonce in 0..FLOOD {
        assert_eq!(
            health_check(&channel, &format!("junk-token-{nonce}")).await,
            Err(Code::Unauthenticated)
        );
    }

    assert_eq!(
        gauge.calls(),
        1,
        "junk tokens must not produce TokenReview traffic"
    );
    health_check(&channel, &victim)
        .await
        .expect("cached review");
    assert_eq!(
        gauge.calls(),
        1,
        "verified entry was evicted by rejected traffic"
    );
    stop_and_join(shutdown, handle).await;
}

#[tokio::test]
async fn forged_claim_flood_stays_within_permits_and_spares_other_callers() {
    const FLOOD: usize = 300;

    let port = free_port();
    let (kube_client, gauge) = fake_apiserver();
    let (shutdown, handle) = start_auth_server(
        port,
        kube_client,
        security_config(&[("vllm", "worker"), ("vllm", "control")]),
    );
    let channel = connect(port).await;

    let flood = tokio::spawn({
        let channel = channel.clone();
        async move {
            let mut attempts = Vec::new();
            for nonce in 0..FLOOD {
                let channel = channel.clone();
                attempts.push(tokio::spawn(async move {
                    health_check(&channel, &sa_token("vllm", "worker", &format!("f{nonce}"))).await
                }));
            }
            let mut saturated = 0;
            for attempt in attempts {
                if attempt.await.expect("flood task") == Err(Code::Unavailable) {
                    saturated += 1;
                }
            }
            saturated
        }
    });

    let victim = sa_token("vllm", "control", "legit");
    let mut victim_failures = Vec::new();
    for _ in 0..20 {
        if let Err(code) = health_check(&channel, &victim).await {
            victim_failures.push(code);
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let saturated = flood.await.expect("flood driver");

    assert!(
        victim_failures.is_empty(),
        "second allowlisted caller was denied during the flood: {victim_failures:?}"
    );
    assert!(
        gauge.peak() <= MAX_CONCURRENT_REVIEWS,
        "peak concurrent TokenReviews {} exceeded the global permit pool {MAX_CONCURRENT_REVIEWS}",
        gauge.peak()
    );
    assert!(
        saturated > 0,
        "flood should shed load once the pools saturate rather than queueing without bound"
    );
    assert_eq!(
        gauge.peak_for("system:serviceaccount:vllm:worker"),
        MAX_CONCURRENT_REVIEWS_PER_CALLER,
        "flood never filled the caller's pool, so the cap under test was not exercised"
    );
    stop_and_join(shutdown, handle).await;
}
