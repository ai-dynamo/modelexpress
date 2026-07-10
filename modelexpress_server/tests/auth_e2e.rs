// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full-stack ServiceAccount token authentication over a real in-process tonic server.
//! Each scenario boots its own server on a separate ephemeral port with a fake
//! TokenReview backend configured to return specific responses.

#![allow(clippy::expect_used)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use http::header::CONTENT_TYPE;
use k8s_openapi::api::authentication::v1::{TokenReview, TokenReviewStatus, UserInfo};
use kube::client::{Body, Client as KubeClient};
use modelexpress_client::Client;
use modelexpress_common::client_config::ClientConfig;
use modelexpress_common::config::ConnectionConfig;
use modelexpress_common::grpc::health::health_service_server::HealthServiceServer;
use modelexpress_server::auth::{AuthLayer, AuthState};
use modelexpress_server::config::{AuthMode, SecurityConfig, ServiceAccountRef};
use modelexpress_server::services::HealthServiceImpl;
use tempfile::TempDir;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tonic::transport::Server;
use tower::{BoxError, Layer};

type ServerResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local addr").port()
}

fn fake_kube_client_for_review(review: TokenReview) -> (KubeClient, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let service = tower::service_fn({
        let calls = calls.clone();
        move |_request: http::Request<Body>| {
            calls.fetch_add(1, Ordering::SeqCst);
            let body = serde_json::to_vec(&review).expect("serialize TokenReview");
            let response = http::Response::builder()
                .status(http::StatusCode::OK)
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(body))
                .expect("fake TokenReview response body");
            async move { Ok::<_, BoxError>(response) }
        }
    });
    (KubeClient::new(service, "default"), calls)
}

fn start_auth_server(
    port: u16,
    kube_client: KubeClient,
    security_config: SecurityConfig,
) -> (oneshot::Sender<()>, JoinHandle<ServerResult>) {
    let (tx, rx) = oneshot::channel::<()>();
    let shutdown = async move {
        let _ = rx.await;
    };
    let handle = tokio::spawn(async move {
        let addr = format!("127.0.0.1:{port}").parse()?;
        let state = Arc::new(AuthState::new(kube_client, &security_config));
        let auth_layer = AuthLayer::new(state);
        Server::builder()
            .add_service(auth_layer.layer(HealthServiceServer::new(HealthServiceImpl)))
            .serve_with_shutdown(addr, shutdown)
            .await
            .map_err(Into::into)
    });
    (tx, handle)
}

async fn connect_client(port: u16) -> Client {
    let config = ClientConfig {
        connection: ConnectionConfig::new(format!("http://127.0.0.1:{port}")),
        ..Default::default()
    };
    for _ in 0..100 {
        if let Ok(client) = Client::new(config.clone()).await {
            return client;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("server on port {port} never became reachable");
}

async fn stop_and_join(shutdown: oneshot::Sender<()>, handle: JoinHandle<ServerResult>) {
    let _ = shutdown.send(());
    tokio::time::timeout(Duration::from_secs(10), handle)
        .await
        .expect("server task did not exit in time")
        .expect("server task panicked")
        .expect("run_server returned an error");
}

fn token_review_for(username: &str) -> TokenReview {
    TokenReview {
        status: Some(TokenReviewStatus {
            authenticated: Some(true),
            audiences: Some(vec!["modelexpress".to_string()]),
            user: Some(UserInfo {
                username: Some(username.to_string()),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn unauthenticated_review() -> TokenReview {
    TokenReview {
        status: Some(TokenReviewStatus {
            authenticated: Some(false),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn security_config() -> SecurityConfig {
    SecurityConfig {
        mode: Some(AuthMode::Enforce),
        token_audiences: vec!["modelexpress".to_string()],
        allowed_service_accounts: vec![ServiceAccountRef {
            namespace: "vllm".to_string(),
            service_account: "worker".to_string(),
        }],
        cache_ttl_secs: 60,
    }
}

#[tokio::test]
async fn allowlisted_authenticated_caller_succeeds_and_caches() {
    let temp_dir = TempDir::new().expect("temp dir");
    let token_path = temp_dir.path().join("token");
    std::fs::write(&token_path, "good-token\n").expect("write token");
    unsafe {
        std::env::set_var(
            modelexpress_common::envs::MX_AUTH_TOKEN_PATH,
            token_path.to_str().expect("token path"),
        );
    }

    let port = free_port();
    let (kube_client, calls) =
        fake_kube_client_for_review(token_review_for("system:serviceaccount:vllm:worker"));
    let (shutdown, handle) = start_auth_server(port, kube_client, security_config());

    let mut client = connect_client(port).await;
    client.health_check().await.expect("first health_check");
    client.health_check().await.expect("second health_check");

    assert_eq!(calls.load(Ordering::SeqCst), 1);

    stop_and_join(shutdown, handle).await;
}

#[tokio::test]
async fn non_allowlisted_caller_is_denied() {
    let temp_dir = TempDir::new().expect("temp dir");
    let token_path = temp_dir.path().join("token");
    std::fs::write(&token_path, "intruder-token\n").expect("write token");
    unsafe {
        std::env::set_var(
            modelexpress_common::envs::MX_AUTH_TOKEN_PATH,
            token_path.to_str().expect("token path"),
        );
    }

    let port = free_port();
    let (kube_client, _calls) =
        fake_kube_client_for_review(token_review_for("system:serviceaccount:other:intruder"));
    let (shutdown, handle) = start_auth_server(port, kube_client, security_config());

    let mut client = connect_client(port).await;
    let error = client
        .health_check()
        .await
        .expect_err("non-allowlisted SA should be denied");

    let error_str = error.to_string();
    assert!(
        error_str.contains("gRPC error")
            && (error_str.contains("permission")
                || error_str.contains("PermissionDenied")
                || error_str.contains("other:intruder")),
        "expected PERMISSION_DENIED error, got: {error_str}"
    );

    stop_and_join(shutdown, handle).await;
}

#[tokio::test]
async fn unauthenticated_caller_is_rejected() {
    let temp_dir = TempDir::new().expect("temp dir");
    let token_path = temp_dir.path().join("token");
    std::fs::write(&token_path, "bad-token\n").expect("write token");
    unsafe {
        std::env::set_var(
            modelexpress_common::envs::MX_AUTH_TOKEN_PATH,
            token_path.to_str().expect("token path"),
        );
    }

    let port = free_port();
    let (kube_client, _calls) = fake_kube_client_for_review(unauthenticated_review());
    let (shutdown, handle) = start_auth_server(port, kube_client, security_config());

    let mut client = connect_client(port).await;
    let error = client
        .health_check()
        .await
        .expect_err("unauthenticated caller should be rejected");

    let error_str = error.to_string();
    assert!(
        error_str.contains("gRPC error")
            && (error_str.contains("unauthenticated")
                || error_str.contains("Unauthenticated")
                || error_str.contains("authentication")),
        "expected UNAUTHENTICATED error, got: {error_str}"
    );

    stop_and_join(shutdown, handle).await;
}
