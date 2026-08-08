// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixtures shared by the auth integration tests.

#![allow(clippy::expect_used, dead_code)]

use std::sync::Arc;
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use kube::client::Client as KubeClient;
use modelexpress_common::grpc::health::health_service_server::HealthServiceServer;
use modelexpress_server::auth::{AuthLayer, AuthState};
use modelexpress_server::config::{AuthMode, SecurityConfig, ServiceAccountRef};
use modelexpress_server::services::HealthServiceImpl;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tonic::transport::Server;
use tower::Layer;

pub type ServerResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

pub fn sa_token(namespace: &str, service_account: &str, nonce: &str) -> String {
    let payload = serde_json::json!({
        "sub": format!("system:serviceaccount:{namespace}:{service_account}"),
        "jti": nonce,
    });
    let payload = serde_json::to_vec(&payload).expect("serialize claims");
    format!(
        "{}.{}.{}",
        URL_SAFE_NO_PAD.encode(br#"{"alg":"RS256"}"#),
        URL_SAFE_NO_PAD.encode(payload),
        URL_SAFE_NO_PAD.encode(b"not-a-real-signature"),
    )
}

pub fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local addr").port()
}

pub fn security_config(allowed: &[(&str, &str)]) -> SecurityConfig {
    SecurityConfig {
        mode: Some(AuthMode::Enforce),
        token_audiences: vec!["modelexpress".to_string()],
        allowed_service_accounts: allowed
            .iter()
            .map(|(namespace, service_account)| ServiceAccountRef {
                namespace: (*namespace).to_string(),
                service_account: (*service_account).to_string(),
            })
            .collect(),
        cache_ttl_secs: 60,
    }
}

pub fn start_auth_server(
    port: u16,
    kube_client: KubeClient,
    config: SecurityConfig,
) -> (oneshot::Sender<()>, JoinHandle<ServerResult>) {
    let (tx, rx) = oneshot::channel::<()>();
    let shutdown = async move {
        let _ = rx.await;
    };
    let handle = tokio::spawn(async move {
        let addr = format!("127.0.0.1:{port}").parse()?;
        let state = Arc::new(AuthState::new(kube_client, &config));
        let auth_layer = AuthLayer::new(state);
        Server::builder()
            .add_service(auth_layer.layer(HealthServiceServer::new(HealthServiceImpl)))
            .serve_with_shutdown(addr, shutdown)
            .await
            .map_err(Into::into)
    });
    (tx, handle)
}

pub async fn stop_and_join(shutdown: oneshot::Sender<()>, handle: JoinHandle<ServerResult>) {
    let _ = shutdown.send(());
    tokio::time::timeout(Duration::from_secs(10), handle)
        .await
        .expect("server task did not exit in time")
        .expect("server task panicked")
        .expect("server returned an error");
}
